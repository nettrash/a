//! The aarch64 kernels under the addition of `Number`s: `r = x + y + carry` and
//! `r = x - y - borrow` over the magnitudes, in inline assembly. They load and store limbs at
//! any alignment, and so are used only on the 64-bit little-endian aarch64 targets whose
//! operating system allows unaligned accesses; the strict-alignment ones (hermit, nuttx, bare
//! metal) get the portable kernels.
//!
//! # The contract
//!
//! A limb is 8 bytes and `n` counts limbs; with `n == 0` a kernel returns the carry (borrow)
//! unchanged and touches no memory. No pointer needs any alignment. In the little-endian
//! layout limb `i` is the bytes `[p + 8i, p + 8i + 8)` read as a little-endian `u64`, and the
//! pointers point at limb 0. In the big-endian layout limb `i` is the bytes
//! `[end - 8(i + 1), end - 8i)` read as a big-endian `u64`: the pointers point one past the
//! least significant byte, and the limbs go towards lower addresses. The output may be
//! disjoint from the inputs or be exactly one or both of them, and in the big-endian layout it
//! may also lie `s >= 1` bytes above an input (`r_end = x_end + s`), which is where an
//! in-place result that grows towards the end of its vector is written.
//!
//! # One carry chain
//!
//! `adcs` adds two limbs and the carry flag C and leaves the carry out in C; `sbcs` subtracts,
//! with C meaning "no borrow". A chain of them takes one cycle per limb, provided that nothing
//! between two of them writes the flags. So the loop counts down with `sub` and tests the count
//! with `cbnz`, neither of which writes the flags, and the steps of 1, 2 and 4 limbs for the
//! remainder are chosen with `tbz`. LLVM's code for a portable carry loop (pairs of
//! `overflowing_add`) tests its count with `subs` or `cmp`, which clobber C, so it rebuilds
//! the carry every iteration: 4 cycles per limb, about 1.8 unrolled by four.
//!
//! Limbs are loaded straight from the byte vectors, unaligned, two per `ldp`: the core issues
//! three loads per cycle whatever their width, so pairs move twice the limbs of single `ldr`s.
//! The big-endian layout puts each loaded limb through `rev` and each result back through
//! `rev` before it is stored. The loop moves 8 limbs per iteration, with one update per
//! pointer (a writeback on one pair, immediate offsets for the others). A loop of 4 limbs is as
//! fast in general, but called back to back on the same buffer in place, each call waiting for
//! the carry of the one before, it ran at 1.6 to 1.9 cycles per limb for 8 to 64 limbs
//! little-endian and at 1.8 to 2.4 for 8 to 32 big-endian, as if each iteration's loads waited
//! for the stores before them (the core predicts which loads depend on earlier stores, and in
//! place, from one call to the next, some do); the 8-limb loop ran at 1.1 to 1.5 there.
//!
//! # Several carry chains
//!
//! One chain is bound by the latency of its carry, one cycle per limb, while an Apple M4 core
//! issues 3 loads, about 1.6 paired stores and 4 flag-writing operations per cycle. So from 32
//! limbs (64 big-endian) the kernels split the limbs into `K` contiguous parts and run `K`
//! chains, each part added as if its carry in were 0 (part 0 takes the real one). The program
//! order interleaves the chains one block of limbs at a time: before its block, a chain's
//! carry goes back into C with a `cmp` on its own carry register, and after it `cset` takes C
//! out again. The `cmp` writes all of NZCV, so it depends on nothing but that register; with
//! the flags renamed, every chain is a separate dataflow chain of `B + 2` cycles per block of
//! `B` limbs, and the core runs the chains side by side.
//!
//! Blocks are 16 limbs, which leaves the hand-off 1/8 of a cycle per limb and chain: blocks of
//! 8 were 4 to 6 % slower while the operands fit the first-level cache. Past it, out of place,
//! blocks of 8 are 15 to 20 % faster (0.8 to 0.9 cycles per limb against 1.0 to 1.1),
//! presumably because a block of 16 stores into two output lines per chain at a time, lines
//! the core has to fetch first; in place the output lines are the input's, already there, and
//! blocks of 16 stay up to 10 % faster.
//!
//! The split for blocks of 16: `m` rounds of 16-limb blocks over all chains, one round of
//! 8-limb blocks if what is left fills it, then the last chain goes on through the rest, fewer
//! than `8K` limbs. With all of the remainder, up to `16K - 1` limbs, on the last chain, four
//! chains took up to 23 % longer at the sizes where the round of 8 applies, such as 96 to 127
//! limbs.
//!
//! The parts are then fixed up in order: when part `j - 1` carried out, 1 is added into part
//! `j` from its first limb on. The carry gets through a part only if the part came out all
//! ones (all zeros, for a subtraction), and such a part did not carry out itself: a part of
//! `L` limbs computed with carry 0 that carried out holds at most `2^(64L) - 2`, which adding
//! 1 cannot wrap (a part that borrowed holds at least 1). So the carry out of a part is the OR
//! of its own and of the one that went through. The first limb takes the carry without a
//! branch: for random limbs the carry is a coin toss that a branch would mispredict half the
//! time. Going further is out of line and checks four limbs at a time, as it runs only when
//! the first limb of a part is all ones (zero): for random limbs about once in `2^64` fix-ups,
//! but in every fix-up when, say, all ones are added to a number of the same length whose
//! middle limbs are zero, or such a number is subtracted from a power of two. There it costs
//! about 0.4 cycles per limb it passes: 0.75 cycles per limb in all for the worst case at 1024
//! limbs, against 1.0 for one chain.
//!
//! The carry out of a call depends on the carry in only through those fix-ups, whose branches
//! the core predicts: a call does not wait for the one before it to finish its carry chain, so
//! dependent calls cost what independent ones do.
//!
//! # Speed
//!
//! Cycles per limb on an Apple M4 Max at 4.0 to 4.45 GHz, the median of 11 interleaved runs,
//! random limbs, out of place, for independent calls and, in parentheses, for calls that each
//! wait for the previous carry; one chain and GMP 6.3's `mpn_add_n` for comparison:
//!
//! | limbs | one chain | little-endian | big-endian | GMP |
//! |---|---|---|---|---|
//! | 8 | 0.66 (1.46) | 0.79 (1.47) | 1.05 (1.49) | 0.93 (1.71) |
//! | 32 | 0.53 (1.18) | 0.65 (0.69) | 0.81 (1.17) | 0.59 (1.24) |
//! | 64 | 0.63 (1.11) | 0.61 (0.65) | 0.86 (0.88) | 0.70 (1.14) |
//! | 256 | 0.93 (1.11) | 0.49 (0.52) | 0.76 (0.78) | 0.94 (1.11) |
//! | 1024 | 1.04 (1.09) | 0.45 (0.47) | 0.73 (0.76) | 1.04 (1.10) |
//! | 65536 | 1.08 (1.06) | 0.88 (0.86) | 1.10 (1.06) | 1.08 (1.06) |
//!
//! In place: 0.46 and 0.69 cycles per limb little-endian at 1024 and 65536 limbs, 0.76 and
//! 0.91 big-endian. Little-endian, from 256 limbs four chains run near the core's load/store
//! bound: two `ldp` and one `stp` per two limbs measured 0.49 to 0.53 cycles per limb with no
//! arithmetic. Big-endian, the three `rev` per limb make the loop issue-bound (5.7 instructions
//! per limb, 182 per 32, at about 7.5 per cycle), and two chains are as fast as four. Past the
//! first-level cache the limbs stream from the second-level cache.
//!
//! # Sizes
//!
//! - Little-endian: one chain below 32 limbs; two chains of 16-limb blocks from 32; four from
//!   64; from 7168, two chains of 8-limb blocks (of 16 in place). Below 64 limbs four chains
//!   were slower than two (0.65 to 0.75 cycles per limb against 0.57 to 0.63), and at 32 limbs
//!   two chains cost about 4 cycles more than one when calls are independent and save about 15
//!   when each waits for the previous carry.
//! - Big-endian: one chain below 64 limbs, two chains of 16-limb blocks from there. Below 64
//!   the second chain does not pay for the instructions around it, which this issue-bound loop
//!   cannot hide.
//! - Four chains keep twelve streams in flight, and when their parts lie within about 2 KB of
//!   a multiple of 16 KB apart, all twelve meet in the same sets of the 8-way, 128 KB
//!   first-level cache: four chains took 1.3 to 1.9 cycles per limb at 8192 and 16384 limbs,
//!   and 1.4 at 8191 (parts 16320 bytes apart), where two took 0.8 to 1.0. Below 7168 limbs
//!   the parts lie less than 14 KB apart: no three of them meet, and the six streams of two
//!   parts fit the 8 ways, as those of two chains always do.
//!
//! # Aliasing
//!
//! Each step loads all of its inputs before it stores its outputs, and each chain goes from
//! its first limb upwards. In place, a chain reads and writes only its own part, and the
//! fix-ups read only the output. In the shifted big-endian case, output limb `i` covers input
//! limbs `i` and below: one chain has loaded all of them by then, but with several chains
//! the first output limb of part `j` would cover the top of part `j - 1` before chain
//! `j - 1` loads it. The kernels detect the overlap and run one chain for it.

use core::arch::asm;

/// Little-endian `r = x + y + carry` over `n` limbs; returns the carry out.
///
/// # Safety
///
/// For `n > 0`, `x` and `y` must be valid for reading and `r` for writing the `8 * n` bytes
/// from each pointer, and `r` must be disjoint from `x` and `y` or equal to either or both.
#[inline]
pub(super) unsafe fn add_n_le(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    carry: bool,
) -> bool {
    // SAFETY: forwarded; each of the four functions called takes the sizes it is called with.
    unsafe {
        if n < LE_TWO_CHAINS {
            chain_add_le(r, x, y, n, carry)
        } else if n < LE_FOUR_CHAINS {
            parts2_add_le(r, x, y, n, carry)
        } else if n < LE_STREAMING {
            parts4_add_le(r, x, y, n, carry)
        } else if core::ptr::eq(r, x) || core::ptr::eq(r, y) {
            // In place the output lines are already in the cache, and blocks of 16 limbs are
            // faster; out of place, blocks of 8 (see the module documentation).
            parts2_add_le(r, x, y, n, carry)
        } else {
            parts2_8_add_le(r, x, y, n, carry)
        }
    }
}

/// Little-endian `r = x - y - borrow` over `n` limbs; returns the borrow out.
///
/// # Safety
///
/// As for [`add_n_le`].
#[inline]
pub(super) unsafe fn sub_n_le(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    borrow: bool,
) -> bool {
    // SAFETY: as in `add_n_le`.
    unsafe {
        if n < LE_TWO_CHAINS {
            chain_sub_le(r, x, y, n, borrow)
        } else if n < LE_FOUR_CHAINS {
            parts2_sub_le(r, x, y, n, borrow)
        } else if n < LE_STREAMING {
            parts4_sub_le(r, x, y, n, borrow)
        } else if core::ptr::eq(r, x) || core::ptr::eq(r, y) {
            // In place the output lines are already in the cache, and blocks of 16 limbs are
            // faster; out of place, blocks of 8 (see the module documentation).
            parts2_sub_le(r, x, y, n, borrow)
        } else {
            parts2_8_sub_le(r, x, y, n, borrow)
        }
    }
}

/// Big-endian `r = x + y + carry` over the `n` limbs that end at the pointers; returns the
/// carry out.
///
/// # Safety
///
/// For `n > 0`, `x_end` and `y_end` must be valid for reading and `r_end` for writing the
/// `8 * n` bytes below each pointer, and each input region must be disjoint from the output
/// region, be the same region (`r_end == x_end`), or lie `s >= 1` bytes below it
/// (`r_end == x_end + s`).
#[inline]
pub(super) unsafe fn add_n_be(
    r_end: *mut u8,
    x_end: *const u8,
    y_end: *const u8,
    n: usize,
    carry: bool,
) -> bool {
    // SAFETY: forwarded to either path; the shifted overlap takes the single chain.
    unsafe {
        if n < BE_TWO_CHAINS || shifted(r_end, x_end, n) || shifted(r_end, y_end, n) {
            chain_add_be(r_end, x_end, y_end, n, carry)
        } else {
            parts2_add_be(r_end, x_end, y_end, n, carry)
        }
    }
}

/// Big-endian `r = x - y - borrow` over the `n` limbs that end at the pointers; returns the
/// borrow out.
///
/// # Safety
///
/// As for [`add_n_be`].
#[inline]
pub(super) unsafe fn sub_n_be(
    r_end: *mut u8,
    x_end: *const u8,
    y_end: *const u8,
    n: usize,
    borrow: bool,
) -> bool {
    // SAFETY: as in `add_n_be`.
    unsafe {
        if n < BE_TWO_CHAINS || shifted(r_end, x_end, n) || shifted(r_end, y_end, n) {
            chain_sub_be(r_end, x_end, y_end, n, borrow)
        } else {
            parts2_sub_be(r_end, x_end, y_end, n, borrow)
        }
    }
}

/// Little-endian: from this many limbs, two chains.
const LE_TWO_CHAINS: usize = 32;
/// Little-endian: from this many limbs, four chains.
const LE_FOUR_CHAINS: usize = 64;
/// Little-endian: from this many limbs, two chains again, of 8-limb blocks out of place (of 16
/// in place).
const LE_STREAMING: usize = 7168;
/// Big-endian: from this many limbs, two chains.
const BE_TWO_CHAINS: usize = 64;

/// Whether the regions of `n` limbs that end at `r` and `x` overlap without being the same.
#[inline(always)]
fn shifted(r: *mut u8, x: *const u8, n: usize) -> bool {
    let d = r.addr().wrapping_sub(x.addr());
    d != 0 && (d < 8 * n || d.wrapping_neg() < 8 * n)
}

/// The asm operand named `$name`, as `{name}` in a template: the macros below take their
/// pointer and carry registers by name, so that one macro serves every chain.
macro_rules! reg {
    ($name:ident) => {
        concat!("{", stringify!($name), "}")
    };
}

/// The instruction that adds (subtracts) two limbs and the carry (borrow) in C.
macro_rules! op {
    (add) => {
        "adcs"
    };
    (sub) => {
        "sbcs"
    };
}

/// Puts the carry register `c` (0 or 1) into C: `c - 1` does not borrow exactly when `c` is 1.
/// For a subtraction C is the inverse of the borrow: `0 - c` does not borrow exactly when `c`
/// is 0.
macro_rules! restore {
    (add, $c:ident) => {
        concat!("cmp ", reg!($c), ", #1\n")
    };
    (sub, $c:ident) => {
        concat!("cmp xzr, ", reg!($c), "\n")
    };
}

/// Takes C back into the carry register `c`: the carry is C, the borrow its inverse.
macro_rules! save {
    (add, $c:ident) => {
        concat!("cset ", reg!($c), ", cs\n")
    };
    (sub, $c:ident) => {
        concat!("cset ", reg!($c), ", cc\n")
    };
}

/// One limb at the pointers, which then move on by one limb.
macro_rules! step1 {
    (le, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            concat!("ldr {a0}, [", reg!($x), "], #8\n"),
            concat!("ldr {b0}, [", reg!($y), "], #8\n"),
            concat!(op!($op), " {a0}, {a0}, {b0}\n"),
            concat!("str {a0}, [", reg!($r), "], #8\n"),
        )
    };
    (be, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            concat!("ldr {a0}, [", reg!($x), ", #-8]!\n"),
            concat!("ldr {b0}, [", reg!($y), ", #-8]!\n"),
            "rev {a0}, {a0}\n",
            "rev {b0}, {b0}\n",
            concat!(op!($op), " {a0}, {a0}, {b0}\n"),
            "rev {a0}, {a0}\n",
            concat!("str {a0}, [", reg!($r), ", #-8]!\n"),
        )
    };
}

/// Two limbs at the pointers, which then move on by two limbs. A big-endian pair loaded from
/// below the end pointer holds the more significant limb first: `ldp a1, a0` puts limb 0 in
/// `a0`.
macro_rules! step2 {
    (le, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            concat!("ldp {a0}, {a1}, [", reg!($x), "], #16\n"),
            concat!("ldp {b0}, {b1}, [", reg!($y), "], #16\n"),
            concat!(op!($op), " {a0}, {a0}, {b0}\n"),
            concat!(op!($op), " {a1}, {a1}, {b1}\n"),
            concat!("stp {a0}, {a1}, [", reg!($r), "], #16\n"),
        )
    };
    (be, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            concat!("ldp {a1}, {a0}, [", reg!($x), ", #-16]!\n"),
            concat!("ldp {b1}, {b0}, [", reg!($y), ", #-16]!\n"),
            "rev {a0}, {a0}\n",
            "rev {b0}, {b0}\n",
            "rev {a1}, {a1}\n",
            "rev {b1}, {b1}\n",
            concat!(op!($op), " {a0}, {a0}, {b0}\n"),
            concat!(op!($op), " {a1}, {a1}, {b1}\n"),
            "rev {a0}, {a0}\n",
            "rev {a1}, {a1}\n",
            concat!("stp {a1}, {a0}, [", reg!($r), ", #-16]!\n"),
        )
    };
}

/// The adds of four limbs: `a0..a3 += b0..b3` with the carry, limb 0 first.
macro_rules! add4 {
    ($op:ident) => {
        concat!(
            concat!(op!($op), " {a0}, {a0}, {b0}\n"),
            concat!(op!($op), " {a1}, {a1}, {b1}\n"),
            concat!(op!($op), " {a2}, {a2}, {b2}\n"),
            concat!(op!($op), " {a3}, {a3}, {b3}\n"),
        )
    };
}

/// The big-endian adds of four limbs: the inputs reversed into numbers, added, the sums
/// reversed back into bytes.
macro_rules! add4_be {
    ($op:ident) => {
        concat!(
            "rev {a0}, {a0}\n",
            "rev {b0}, {b0}\n",
            "rev {a1}, {a1}\n",
            "rev {b1}, {b1}\n",
            "rev {a2}, {a2}\n",
            "rev {b2}, {b2}\n",
            "rev {a3}, {a3}\n",
            "rev {b3}, {b3}\n",
            add4!($op),
            "rev {a0}, {a0}\n",
            "rev {a1}, {a1}\n",
            "rev {a2}, {a2}\n",
            "rev {a3}, {a3}\n",
        )
    };
}

/// Four little-endian limbs at byte offset `$o` (an assembler expression) from the pointers,
/// which stay where they are.
macro_rules! le4 {
    ($op:ident, $x:ident, $y:ident, $r:ident, $o:literal) => {
        concat!(
            concat!("ldp {a2}, {a3}, [", reg!($x), ", #(", $o, " + 16)]\n"),
            concat!("ldp {a0}, {a1}, [", reg!($x), ", #(", $o, ")]\n"),
            concat!("ldp {b2}, {b3}, [", reg!($y), ", #(", $o, " + 16)]\n"),
            concat!("ldp {b0}, {b1}, [", reg!($y), ", #(", $o, ")]\n"),
            add4!($op),
            concat!("stp {a2}, {a3}, [", reg!($r), ", #(", $o, " + 16)]\n"),
            concat!("stp {a0}, {a1}, [", reg!($r), ", #(", $o, ")]\n"),
        )
    };
}

/// Four little-endian limbs at the pointers, which then move on by `$w` bytes. The pair at
/// +16 comes first, from the old pointer, so that it does not wait for the writeback.
macro_rules! le4w {
    ($op:ident, $x:ident, $y:ident, $r:ident, $w:literal) => {
        concat!(
            concat!("ldp {a2}, {a3}, [", reg!($x), ", #16]\n"),
            concat!("ldp {a0}, {a1}, [", reg!($x), "], #", $w, "\n"),
            concat!("ldp {b2}, {b3}, [", reg!($y), ", #16]\n"),
            concat!("ldp {b0}, {b1}, [", reg!($y), "], #", $w, "\n"),
            add4!($op),
            concat!("stp {a2}, {a3}, [", reg!($r), ", #16]\n"),
            concat!("stp {a0}, {a1}, [", reg!($r), "], #", $w, "\n"),
        )
    };
}

/// Four big-endian limbs below byte offset `$o` (zero or negative) from the end pointers,
/// which stay where they are.
macro_rules! be4 {
    ($op:ident, $x:ident, $y:ident, $r:ident, $o:literal) => {
        concat!(
            concat!("ldp {a1}, {a0}, [", reg!($x), ", #(", $o, " - 16)]\n"),
            concat!("ldp {a3}, {a2}, [", reg!($x), ", #(", $o, " - 32)]\n"),
            concat!("ldp {b1}, {b0}, [", reg!($y), ", #(", $o, " - 16)]\n"),
            concat!("ldp {b3}, {b2}, [", reg!($y), ", #(", $o, " - 32)]\n"),
            add4_be!($op),
            concat!("stp {a1}, {a0}, [", reg!($r), ", #(", $o, " - 16)]\n"),
            concat!("stp {a3}, {a2}, [", reg!($r), ", #(", $o, " - 32)]\n"),
        )
    };
}

/// As [`be4`], with the deepest pair pre-indexed: the pointers move down to `$o - 32`, which
/// is `-$w`.
macro_rules! be4w {
    ($op:ident, $x:ident, $y:ident, $r:ident, $o:literal, $w:literal) => {
        concat!(
            concat!("ldp {a1}, {a0}, [", reg!($x), ", #(", $o, " - 16)]\n"),
            concat!("ldp {a3}, {a2}, [", reg!($x), ", #-", $w, "]!\n"),
            concat!("ldp {b1}, {b0}, [", reg!($y), ", #(", $o, " - 16)]\n"),
            concat!("ldp {b3}, {b2}, [", reg!($y), ", #-", $w, "]!\n"),
            add4_be!($op),
            concat!("stp {a1}, {a0}, [", reg!($r), ", #(", $o, " - 16)]\n"),
            concat!("stp {a3}, {a2}, [", reg!($r), ", #-", $w, "]!\n"),
        )
    };
}

/// A block of 4, 8 or 16 limbs as steps of four, the pointers moved once per block. The
/// steps share the eight data registers: register renaming lets each step's loads issue as
/// early as with registers of their own.
macro_rules! block {
    (le, 4, $op:ident, $x:ident, $y:ident, $r:ident) => {
        le4w!($op, $x, $y, $r, "32")
    };
    (le, 8, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(le4w!($op, $x, $y, $r, "64"), le4!($op, $x, $y, $r, "-32"))
    };
    (le, 16, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            le4w!($op, $x, $y, $r, "128"),
            le4!($op, $x, $y, $r, "-96"),
            le4!($op, $x, $y, $r, "-64"),
            le4!($op, $x, $y, $r, "-32"),
        )
    };
    (be, 4, $op:ident, $x:ident, $y:ident, $r:ident) => {
        be4w!($op, $x, $y, $r, "0", "32")
    };
    (be, 8, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            be4!($op, $x, $y, $r, "0"),
            be4w!($op, $x, $y, $r, "-32", "64")
        )
    };
    (be, 16, $op:ident, $x:ident, $y:ident, $r:ident) => {
        concat!(
            be4!($op, $x, $y, $r, "0"),
            be4!($op, $x, $y, $r, "-32"),
            be4!($op, $x, $y, $r, "-64"),
            be4w!($op, $x, $y, $r, "-96", "128"),
        )
    };
}

/// The limbs left to the last chain, `{t} mod (8 << $bits)` of them: steps of 1, 2 and 4
/// picked by the low bits, then blocks of 8 for bits 3 and up.
macro_rules! rest {
    ($ord:ident, $op:ident, $x:ident, $y:ident, $r:ident, $bits:literal) => {
        concat!(
            "tbz {t}, #0, 3f\n",
            step1!($ord, $op, $x, $y, $r),
            "3:\n",
            "tbz {t}, #1, 4f\n",
            step2!($ord, $op, $x, $y, $r),
            "4:\n",
            "tbz {t}, #2, 5f\n",
            block!($ord, 4, $op, $x, $y, $r),
            "5:\n",
            concat!("ubfx {t}, {t}, #3, #", $bits, "\n"),
            "cbz {t}, 7f\n",
            "6:\n",
            block!($ord, 8, $op, $x, $y, $r),
            "sub {t}, {t}, #1\n",
            "cbnz {t}, 6b\n",
            "7:\n",
        )
    };
}

/// A kernel on one chain: the steps of 1, 2 and 4 limbs for `n mod 8`, then 8 limbs per
/// iteration.
macro_rules! chain {
    ($(#[$doc:meta])* $name:ident, $op:ident, $ord:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// As for the kernel of the same operation and layout, and `n` may be any size.
        #[inline(always)]
        unsafe fn $name(r: *mut u8, x: *const u8, y: *const u8, n: usize, carry: bool) -> bool {
            let mut c = u64::from(carry);
            // SAFETY: the steps load and store exactly the `n` limbs of the regions the caller
            // vouches for, limb 0 first, each step's inputs before its outputs; with `n == 0`
            // every step is skipped. No other memory and no stack is touched, and the flags are
            // declared clobbered (no `preserves_flags`).
            unsafe {
                asm!(
                    restore!($op, c),
                    "tbz {n}, #0, 2f",
                    step1!($ord, $op, x, y, r),
                    "2:",
                    "tbz {n}, #1, 3f",
                    step2!($ord, $op, x, y, r),
                    "3:",
                    "tbz {n}, #2, 4f",
                    block!($ord, 4, $op, x, y, r),
                    "4:",
                    "lsr {n}, {n}, #3",
                    "cbz {n}, 6f",
                    "5:",
                    block!($ord, 8, $op, x, y, r),
                    "sub {n}, {n}, #1",
                    "cbnz {n}, 5b",
                    "6:",
                    save!($op, c),
                    c = inout(reg) c,
                    n = inout(reg) n => _,
                    r = inout(reg) r => _,
                    x = inout(reg) x => _,
                    y = inout(reg) y => _,
                    a0 = out(reg) _,
                    a1 = out(reg) _,
                    a2 = out(reg) _,
                    a3 = out(reg) _,
                    b0 = out(reg) _,
                    b1 = out(reg) _,
                    b2 = out(reg) _,
                    b3 = out(reg) _,
                    options(nostack),
                );
                // SAFETY: `cset` wrote 0 or 1, so the comparison below is the low bit.
                core::hint::assert_unchecked(c <= 1);
            }
            c != 0
        }
    };
}

chain!(
    /// Little-endian `r = x + y + carry` on one chain.
    chain_add_le,
    add,
    le
);
chain!(
    /// Little-endian `r = x - y - borrow` on one chain.
    chain_sub_le,
    sub,
    le
);
chain!(
    /// Big-endian `r = x + y + carry` on one chain, with end pointers.
    chain_add_be,
    add,
    be
);
chain!(
    /// Big-endian `r = x - y - borrow` on one chain, with end pointers.
    chain_sub_be,
    sub,
    be
);

/// The start of a part, `s` bytes on from that of the part before it (its end, `s` bytes down,
/// big-endian).
macro_rules! next_part {
    (le, $d:ident, $p:ident) => {
        concat!("add ", reg!($d), ", ", reg!($p), ", {s}\n")
    };
    (be, $d:ident, $p:ident) => {
        concat!("sub ", reg!($d), ", ", reg!($p), ", {s}\n")
    };
}

/// Two chains: parts 0 and 1 start `s` bytes apart; `m` rounds of 16-limb blocks, a round of
/// 8-limb blocks if bit 4 of `t` is set, then the low 4 bits of `t` on chain 1. Returns the
/// two carries out.
macro_rules! chains2 {
    ($(#[$doc:meta])* $name:ident, $op:ident, $ord:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// The two parts of `16m + 8(t >> 4 & 1)` limbs each and the `t & 15` limbs above them
        /// must lie inside the regions of the calling kernel.
        #[inline(always)]
        unsafe fn $name(r: *mut u8, x: *const u8, y: *const u8, s: usize, m: usize, t: usize, c: u64) -> [u64; 2] {
            let (c0, c1): (u64, u64);
            // SAFETY: each chain loads and stores the limbs of its own part (chain 1 also the
            // rest above it), from its first limb upwards, each step's inputs before its
            // outputs; nothing else is touched. No stack; the flags are declared clobbered.
            unsafe {
                asm!(
                    next_part!($ord, x1, x0),
                    next_part!($ord, y1, y0),
                    next_part!($ord, r1, r0),
                    "mov {c1}, #0",
                    "cbz {m}, 8f",
                    "2:",
                    restore!($op, c0),
                    block!($ord, 16, $op, x0, y0, r0),
                    save!($op, c0),
                    restore!($op, c1),
                    block!($ord, 16, $op, x1, y1, r1),
                    save!($op, c1),
                    "sub {m}, {m}, #1",
                    "cbnz {m}, 2b",
                    "8:",
                    "tbz {t}, #4, 9f",
                    restore!($op, c0),
                    block!($ord, 8, $op, x0, y0, r0),
                    save!($op, c0),
                    restore!($op, c1),
                    block!($ord, 8, $op, x1, y1, r1),
                    save!($op, c1),
                    "9:",
                    restore!($op, c1),
                    rest!($ord, $op, x1, y1, r1, "1"),
                    save!($op, c1),
                    x0 = inout(reg) x => _,
                    y0 = inout(reg) y => _,
                    r0 = inout(reg) r => _,
                    x1 = out(reg) _,
                    y1 = out(reg) _,
                    r1 = out(reg) _,
                    c0 = inout(reg) c => c0,
                    c1 = out(reg) c1,
                    s = in(reg) s,
                    m = inout(reg) m => _,
                    t = inout(reg) t => _,
                    a0 = out(reg) _,
                    a1 = out(reg) _,
                    a2 = out(reg) _,
                    a3 = out(reg) _,
                    b0 = out(reg) _,
                    b1 = out(reg) _,
                    b2 = out(reg) _,
                    b3 = out(reg) _,
                    options(nostack),
                );
                // SAFETY: both carries were written by `cset`, 0 or 1.
                core::hint::assert_unchecked(c0 <= 1 && c1 <= 1);
            }
            [c0, c1]
        }
    };
}

/// Four chains: parts 0 to 3 start `s` bytes apart; `m` rounds of 16-limb blocks, a round of
/// 8-limb blocks if bit 5 of `t` is set, then the low 5 bits of `t` on chain 3. Returns the
/// four carries out.
///
/// The asm keeps 27 general registers live at once: the 12 pointers, the 4 carries, `s`, `m`,
/// `t` and the 8 data registers. On Apple and Windows, where x18 and x29 are reserved, that
/// leaves 2 to spare.
macro_rules! chains4 {
    ($(#[$doc:meta])* $name:ident, $op:ident, $ord:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// The four parts of `16m + 8(t >> 5 & 1)` limbs each and the `t & 31` limbs above
        /// them must lie inside the regions of the calling kernel.
        #[inline(always)]
        unsafe fn $name(r: *mut u8, x: *const u8, y: *const u8, s: usize, m: usize, t: usize, c: u64) -> [u64; 4] {
            let (c0, c1, c2, c3): (u64, u64, u64, u64);
            // SAFETY: as in the two-chain loop, with chain 3 taking the rest.
            unsafe {
                asm!(
                    next_part!($ord, x1, x0),
                    next_part!($ord, y1, y0),
                    next_part!($ord, r1, r0),
                    next_part!($ord, x2, x1),
                    next_part!($ord, y2, y1),
                    next_part!($ord, r2, r1),
                    next_part!($ord, x3, x2),
                    next_part!($ord, y3, y2),
                    next_part!($ord, r3, r2),
                    "mov {c1}, #0",
                    "mov {c2}, #0",
                    "mov {c3}, #0",
                    "cbz {m}, 8f",
                    "2:",
                    restore!($op, c0),
                    block!($ord, 16, $op, x0, y0, r0),
                    save!($op, c0),
                    restore!($op, c1),
                    block!($ord, 16, $op, x1, y1, r1),
                    save!($op, c1),
                    restore!($op, c2),
                    block!($ord, 16, $op, x2, y2, r2),
                    save!($op, c2),
                    restore!($op, c3),
                    block!($ord, 16, $op, x3, y3, r3),
                    save!($op, c3),
                    "sub {m}, {m}, #1",
                    "cbnz {m}, 2b",
                    "8:",
                    "tbz {t}, #5, 9f",
                    restore!($op, c0),
                    block!($ord, 8, $op, x0, y0, r0),
                    save!($op, c0),
                    restore!($op, c1),
                    block!($ord, 8, $op, x1, y1, r1),
                    save!($op, c1),
                    restore!($op, c2),
                    block!($ord, 8, $op, x2, y2, r2),
                    save!($op, c2),
                    restore!($op, c3),
                    block!($ord, 8, $op, x3, y3, r3),
                    save!($op, c3),
                    "9:",
                    restore!($op, c3),
                    rest!($ord, $op, x3, y3, r3, "2"),
                    save!($op, c3),
                    x0 = inout(reg) x => _,
                    y0 = inout(reg) y => _,
                    r0 = inout(reg) r => _,
                    x1 = out(reg) _,
                    y1 = out(reg) _,
                    r1 = out(reg) _,
                    x2 = out(reg) _,
                    y2 = out(reg) _,
                    r2 = out(reg) _,
                    x3 = out(reg) _,
                    y3 = out(reg) _,
                    r3 = out(reg) _,
                    c0 = inout(reg) c => c0,
                    c1 = out(reg) c1,
                    c2 = out(reg) c2,
                    c3 = out(reg) c3,
                    s = in(reg) s,
                    m = inout(reg) m => _,
                    t = inout(reg) t => _,
                    a0 = out(reg) _,
                    a1 = out(reg) _,
                    a2 = out(reg) _,
                    a3 = out(reg) _,
                    b0 = out(reg) _,
                    b1 = out(reg) _,
                    b2 = out(reg) _,
                    b3 = out(reg) _,
                    options(nostack),
                );
                // SAFETY: all four carries were written by `cset`, 0 or 1.
                core::hint::assert_unchecked(c0 <= 1 && c1 <= 1 && c2 <= 1 && c3 <= 1);
            }
            [c0, c1, c2, c3]
        }
    };
}

/// Two chains of 8-limb blocks, for operands that stream from the second-level cache: parts
/// 0 and 1 start `s` bytes apart; `m` rounds of 8-limb blocks, then the low 4 bits of `t` on
/// chain 1. Returns the two carries out.
macro_rules! chains2_8 {
    ($(#[$doc:meta])* $name:ident, $op:ident, $ord:ident) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// The two parts of `8m` limbs each and the `t & 15` limbs above them must lie inside
        /// the regions of the calling kernel.
        #[inline(always)]
        unsafe fn $name(r: *mut u8, x: *const u8, y: *const u8, s: usize, m: usize, t: usize, c: u64) -> [u64; 2] {
            let (c0, c1): (u64, u64);
            // SAFETY: as in the two-chain loop of 16-limb blocks.
            unsafe {
                asm!(
                    next_part!($ord, x1, x0),
                    next_part!($ord, y1, y0),
                    next_part!($ord, r1, r0),
                    "mov {c1}, #0",
                    "cbz {m}, 8f",
                    "2:",
                    restore!($op, c0),
                    block!($ord, 8, $op, x0, y0, r0),
                    save!($op, c0),
                    restore!($op, c1),
                    block!($ord, 8, $op, x1, y1, r1),
                    save!($op, c1),
                    "sub {m}, {m}, #1",
                    "cbnz {m}, 2b",
                    "8:",
                    restore!($op, c1),
                    rest!($ord, $op, x1, y1, r1, "1"),
                    save!($op, c1),
                    x0 = inout(reg) x => _,
                    y0 = inout(reg) y => _,
                    r0 = inout(reg) r => _,
                    x1 = out(reg) _,
                    y1 = out(reg) _,
                    r1 = out(reg) _,
                    c0 = inout(reg) c => c0,
                    c1 = out(reg) c1,
                    s = in(reg) s,
                    m = inout(reg) m => _,
                    t = inout(reg) t => _,
                    a0 = out(reg) _,
                    a1 = out(reg) _,
                    a2 = out(reg) _,
                    a3 = out(reg) _,
                    b0 = out(reg) _,
                    b1 = out(reg) _,
                    b2 = out(reg) _,
                    b3 = out(reg) _,
                    options(nostack),
                );
                // SAFETY: both carries were written by `cset`, 0 or 1.
                core::hint::assert_unchecked(c0 <= 1 && c1 <= 1);
            }
            [c0, c1]
        }
    };
}

chains2!(
    /// Little-endian additions on two chains.
    chains2_add_le,
    add,
    le
);
chains2!(
    /// Little-endian subtractions on two chains.
    chains2_sub_le,
    sub,
    le
);
chains2!(
    /// Big-endian additions on two chains, with end pointers.
    chains2_add_be,
    add,
    be
);
chains2!(
    /// Big-endian subtractions on two chains, with end pointers.
    chains2_sub_be,
    sub,
    be
);
chains4!(
    /// Little-endian additions on four chains.
    chains4_add_le,
    add,
    le
);
chains4!(
    /// Little-endian subtractions on four chains.
    chains4_sub_le,
    sub,
    le
);
chains2_8!(
    /// Little-endian additions on two chains of 8-limb blocks.
    chains2_8_add_le,
    add,
    le
);
chains2_8!(
    /// Little-endian subtractions on two chains of 8-limb blocks.
    chains2_8_sub_le,
    sub,
    le
);

/// A kernel on `K` chains, for `n >= 8K`: the split, the loop, the fix-ups. Out of line: with
/// two chains or four the loop needs callee-saved registers, and the prologue that saves them
/// should not be paid by the sizes one chain takes.
macro_rules! parts {
    (
        $(#[$doc:meta])* $name:ident, $chains:ident, $k:literal, $split:ident, $sub:literal,
        $fix:ident, $at:ident
    ) => {
        $(#[$doc])*
        ///
        /// # Safety
        ///
        /// As for the kernel of the same operation and layout, with `n >= 8K` and, big-endian,
        /// the output not shifted against an input.
        #[inline(never)]
        unsafe fn $name(r: *mut u8, x: *const u8, y: *const u8, n: usize, carry: bool) -> bool {
            let (m, t, len, rest) = $split(n, $k);
            // SAFETY: the parts and the rest tile the `n` limbs; `m` and `t` describe them as
            // the loop expects.
            let c = unsafe { $chains(r, x, y, 8 * len, m, t, u64::from(carry)) };
            let mut carry = c[0] != 0;
            for (j, &cj) in c.iter().enumerate().skip(1) {
                let limbs = if j == $k - 1 { len + rest } else { len };
                // SAFETY: part `j` is the `limbs` limbs from limb `j * len`, inside the output.
                carry = (cj != 0) | unsafe { $fix::<$sub>($at(r, j * len), limbs, carry) };
            }
            carry
        }
    };
}

/// The split of `n >= 8k` limbs over `k` chains of 16-limb blocks: `m` rounds of 16, one
/// round of 8 when the remainder `t` has room for it (bit `log2(8k)` of `t`), and the rest,
/// below `8k` limbs, on the last chain. Returns `(m, t, len, rest)`, `len` the limbs of each
/// part, at least 8.
#[inline(always)]
const fn split16(n: usize, k: usize) -> (usize, usize, usize, usize) {
    let m = n / (16 * k);
    let t = n % (16 * k);
    let len = 16 * m + if t >= 8 * k { 8 } else { 0 };
    (m, t, len, t % (8 * k))
}

/// The split of `n >= 8k` limbs over `k` chains of 8-limb blocks: `m` rounds, and the rest,
/// below `8k` limbs, on the last chain. Returns `(m, t, len, rest)` as [`split16`] does.
#[inline(always)]
const fn split8(n: usize, k: usize) -> (usize, usize, usize, usize) {
    let m = n / (8 * k);
    let t = n % (8 * k);
    (m, t, 8 * m, t)
}

/// The address of limb `i` of a little-endian output.
#[inline(always)]
fn limb_le(r: *mut u8, i: usize) -> *mut u8 {
    r.wrapping_add(8 * i)
}

/// The end of limb `i` of a big-endian output (the address just above it).
#[inline(always)]
fn limb_be(r: *mut u8, i: usize) -> *mut u8 {
    r.wrapping_sub(8 * i)
}

parts!(
    /// Little-endian `r = x + y + carry` on two chains.
    parts2_add_le,
    chains2_add_le,
    2,
    split16,
    false,
    fix_le,
    limb_le
);
parts!(
    /// Little-endian `r = x - y - borrow` on two chains.
    parts2_sub_le,
    chains2_sub_le,
    2,
    split16,
    true,
    fix_le,
    limb_le
);
parts!(
    /// Little-endian `r = x + y + carry` on four chains.
    parts4_add_le,
    chains4_add_le,
    4,
    split16,
    false,
    fix_le,
    limb_le
);
parts!(
    /// Little-endian `r = x - y - borrow` on four chains.
    parts4_sub_le,
    chains4_sub_le,
    4,
    split16,
    true,
    fix_le,
    limb_le
);
parts!(
    /// Little-endian `r = x + y + carry` on two chains of 8-limb blocks.
    parts2_8_add_le,
    chains2_8_add_le,
    2,
    split8,
    false,
    fix_le,
    limb_le
);
parts!(
    /// Little-endian `r = x - y - borrow` on two chains of 8-limb blocks.
    parts2_8_sub_le,
    chains2_8_sub_le,
    2,
    split8,
    true,
    fix_le,
    limb_le
);
parts!(
    /// Big-endian `r = x + y + carry` on two chains, with end pointers.
    parts2_add_be,
    chains2_add_be,
    2,
    split16,
    false,
    fix_be,
    limb_be
);
parts!(
    /// Big-endian `r = x - y - borrow` on two chains, with end pointers.
    parts2_sub_be,
    chains2_sub_be,
    2,
    split16,
    true,
    fix_be,
    limb_be
);

/// The value a limb takes when the carry (borrow) goes on through it: all ones plus one is
/// zero, zero minus one is all ones.
const fn wrapped<const SUB: bool>() -> u64 {
    if SUB { u64::MAX } else { 0 }
}

/// Reads the limb at `p` as a little-endian `u64`.
///
/// # Safety
///
/// `p` must be valid for reading 8 bytes.
#[inline(always)]
unsafe fn load_le(p: *const u8) -> u64 {
    // SAFETY: the caller guarantees 8 readable bytes; `[u8; 8]` is aligned at any address.
    u64::from_le_bytes(unsafe { p.cast::<[u8; 8]>().read() })
}

/// Writes `v` to `p` as a little-endian `u64`.
///
/// # Safety
///
/// `p` must be valid for writing 8 bytes.
#[inline(always)]
unsafe fn store_le(p: *mut u8, v: u64) {
    // SAFETY: the caller guarantees 8 writable bytes; `[u8; 8]` is aligned at any address.
    unsafe { p.cast::<[u8; 8]>().write(v.to_le_bytes()) }
}

/// Reads the limb at `p` as a big-endian `u64`.
///
/// # Safety
///
/// `p` must be valid for reading 8 bytes.
#[inline(always)]
unsafe fn load_be(p: *const u8) -> u64 {
    // SAFETY: as for `load_le`.
    u64::from_be_bytes(unsafe { p.cast::<[u8; 8]>().read() })
}

/// Writes `v` to `p` as a big-endian `u64`.
///
/// # Safety
///
/// `p` must be valid for writing 8 bytes.
#[inline(always)]
unsafe fn store_be(p: *mut u8, v: u64) {
    // SAFETY: as for `store_le`.
    unsafe { p.cast::<[u8; 8]>().write(v.to_be_bytes()) }
}

/// Adds `carry` into the little-endian part of `len >= 1` limbs at `p` (subtracts it, for
/// `SUB`), and returns whether it went through every limb.
///
/// # Safety
///
/// `p` must be valid for reading and writing `8 * len` bytes.
#[inline(always)]
unsafe fn fix_le<const SUB: bool>(p: *mut u8, len: usize, carry: bool) -> bool {
    // SAFETY: `len >= 1` limbs are valid at `p`.
    let v = unsafe { load_le(p) };
    let v = if SUB {
        v.wrapping_sub(u64::from(carry))
    } else {
        v.wrapping_add(u64::from(carry))
    };
    // SAFETY: as above. With no carry this stores the limb as it was.
    unsafe { store_le(p, v) };
    // SAFETY: as above.
    carry && v == wrapped::<SUB>() && unsafe { ripple_le::<SUB>(p, len) }
}

/// Carries on into limbs `1..len` of the little-endian part at `p`, whose limb 0 wrapped:
/// every limb that is all ones (zero, for `SUB`) wraps in turn, and the first that is not takes
/// the carry. Returns whether the carry went through every limb.
///
/// # Safety
///
/// As for [`fix_le`].
#[cold]
#[inline(never)]
unsafe fn ripple_le<const SUB: bool>(p: *mut u8, len: usize) -> bool {
    let through = !wrapped::<SUB>();
    let mut i = 1;
    while i + 4 <= len {
        // SAFETY: limbs `i..i + 4`, all that the four loads and stores touch, are below `len`.
        unsafe {
            let q = p.add(8 * i);
            let (a, b, c, d) = (
                load_le(q),
                load_le(q.add(8)),
                load_le(q.add(16)),
                load_le(q.add(24)),
            );
            let all = if SUB {
                a | b | c | d == 0
            } else {
                a & b & c & d == u64::MAX
            };
            if !all {
                break;
            }
            for k in 0..4 {
                store_le(q.add(8 * k), wrapped::<SUB>());
            }
        }
        i += 4;
    }
    while i < len {
        // SAFETY: limb `i` is below `len`.
        unsafe {
            let q = p.add(8 * i);
            let v = load_le(q);
            store_le(
                q,
                if SUB {
                    v.wrapping_sub(1)
                } else {
                    v.wrapping_add(1)
                },
            );
            if v != through {
                return false;
            }
        }
        i += 1;
    }
    true
}

/// As [`fix_le`], for the big-endian part of `len >= 1` limbs that ends at `p`.
///
/// # Safety
///
/// The `8 * len` bytes below `p` must be valid for reading and writing.
#[inline(always)]
unsafe fn fix_be<const SUB: bool>(p: *mut u8, len: usize, carry: bool) -> bool {
    // SAFETY: `len >= 1` limbs are valid below `p`.
    let q = unsafe { p.sub(8) };
    // SAFETY: as above.
    let v = unsafe { load_be(q) };
    let v = if SUB {
        v.wrapping_sub(u64::from(carry))
    } else {
        v.wrapping_add(u64::from(carry))
    };
    // SAFETY: as above. With no carry this stores the limb as it was.
    unsafe { store_be(q, v) };
    // SAFETY: as above.
    carry && v == wrapped::<SUB>() && unsafe { ripple_be::<SUB>(p, len) }
}

/// As [`ripple_le`], for the big-endian part that ends at `p`.
///
/// # Safety
///
/// As for [`fix_be`].
#[cold]
#[inline(never)]
unsafe fn ripple_be<const SUB: bool>(p: *mut u8, len: usize) -> bool {
    let through = !wrapped::<SUB>();
    let mut i = 1;
    while i + 4 <= len {
        // SAFETY: limbs `i..i + 4`, the 32 bytes below `p - 8 * i` that the four loads and
        // stores touch, are below `len`.
        unsafe {
            let q = p.sub(8 * (i + 4));
            let (a, b, c, d) = (
                load_be(q),
                load_be(q.add(8)),
                load_be(q.add(16)),
                load_be(q.add(24)),
            );
            let all = if SUB {
                a | b | c | d == 0
            } else {
                a & b & c & d == u64::MAX
            };
            if !all {
                break;
            }
            for k in 0..4 {
                store_be(q.add(8 * k), wrapped::<SUB>());
            }
        }
        i += 4;
    }
    while i < len {
        // SAFETY: limb `i`, the 8 bytes below `p - 8 * i`, is below `len`.
        unsafe {
            let q = p.sub(8 * (i + 1));
            let v = load_be(q);
            store_be(
                q,
                if SUB {
                    v.wrapping_sub(1)
                } else {
                    v.wrapping_add(1)
                },
            );
            if v != through {
                return false;
            }
        }
        i += 1;
    }
    true
}

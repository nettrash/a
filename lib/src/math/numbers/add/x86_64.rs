//! The `x86_64` kernels: the carry (or borrow) between limbs lives in CF, which `adc` and `sbb`
//! consume and produce, so the limbs form one chain of additions through the flag, four limbs
//! per round of the loop.
//!
//! Nothing between the first limb and the final `setc` may write CF. The pointers advance
//! with `lea`, which writes no flags; the counters step with `dec`, which writes every flag but
//! CF; `mov`, `bswap` and the jumps write none. A loop that may run zero times needs a test
//! before its first round, and the flag-setting tests (`test`, `cmp`) clear CF, so each loop is
//! entered at its closing `dec`/`jnz` with its count one higher: a count of zero falls through
//! at once. (`jrcxz`, the one jump on zero that reads no flags, only reaches 127 bytes, less
//! than a big-endian round with alignment padding before it.) Both counts are computed, and
//! CF set from the carry, before either loop starts.
//!
//! `adc` and `sbb` read only CF, which the previous `adc` or `sbb` wrote in full, so the flags
//! `dec` writes need no merging on Sandy Bridge and later, or on AMD's cores. The older
//! P6-derived cores (Core 2, Nehalem) stall on that read instead: GMP measures this shape of
//! loop at about 5 cycles per limb there, against 1.1 to 1.5 from Sandy Bridge on and on Zen.
//! A loop driven by `lea` and `jrcxz` alone avoids the stall, at the cost of a second branch
//! per round on every newer core.
//!
//! The loop of rounds starts on a 32-byte boundary; the padding before it is jumped over. The
//! aim is to keep its closing `dec`/`jnz` inside one 32-byte block, and so the loop in the
//! decoded-uop cache, on the Skylake-derived cores whose microcode for the JCC erratum leaves
//! out jumps that cross or end on a 32-byte boundary. Where the pair falls depends on the
//! registers LLVM picks for the pointers. In the builds checked it is at bytes 0x39 to 0x3d of
//! the little-endian round and 0x6c to 0x70 of the big-endian one (0x71 to 0x75 in a copy with
//! a pointer in r12). The little-endian pair ends 2 bytes before the boundary: a pointer in
//! r12, which needs a SIB byte in each of its five operands, would put it across the boundary,
//! and pointers in both rbp and r13, which need a displacement byte where the offset is 0, on
//! it. The big-endian round, whose operands all have offsets, could take 14 bytes more and
//! keep its pair inside the block.
//!
//! The little-endian kernels add each `y` limb straight from memory, `adc reg, [mem]`. The
//! big-endian ones load both limbs with `mov`, reverse their bytes with `bswap`, add, and
//! reverse the sum back before the store: `movbe` would fold the reversals into the loads and
//! the store, but it is not part of baseline x86-64, and Intel's large cores decode it into
//! the same uops as `mov` and `bswap`.
//!
//! The pointers and the count are passed as `u64`: on x32, where pointers and `usize` are 32
//! bits wide, they would otherwise arrive with an undefined upper half, which the 64-bit
//! addressing below would read.

use core::arch::asm;

/// Defines a little-endian kernel; `$op` is `adc` for an addition, `sbb` for a subtraction;
/// `$carry` names the carry (or borrow) coming in.
macro_rules! le_kernel {
    ($(#[$attr:meta])* $name:ident, $op:literal, $carry:ident) => {
        $(#[$attr])*
        #[inline]
        pub(super) unsafe fn $name(
            r: *mut u8,
            x: *const u8,
            y: *const u8,
            n: usize,
            $carry: bool,
        ) -> bool {
            let carry_out: u8;
            // SAFETY: the loops read limbs `0..n` of `x` and `y` and write limbs `0..n` of
            // `r`, which the caller guarantees are valid; x86 loads and stores of eight
            // bytes need no alignment. Each limb's inputs are loaded before its output is
            // stored, from limb 0 upwards, so an output equal to an input only overwrites
            // bytes already read. The asm uses no stack and names every register it
            // writes; the flags it changes are clobbered by every `asm!` by default.
            unsafe {
                asm!(
                    // `rem` = n mod 4 and `cnt` = n / 4, each plus one for the entry at
                    // the loop's test; then CF = the carry (`neg` sets CF exactly when
                    // its operand is not zero).
                    "mov {rem:e}, {cnt:e}",
                    "and {rem:e}, 3",
                    "inc {rem:e}",
                    "shr {cnt}, 2",
                    "inc {cnt}",
                    "neg {cf}",
                    "jmp 3f",
                    // Rounds of four limbs, the loop top aligned to 32 bytes (see above).
                    ".p2align 5",
                    "2:",
                    "mov {a}, qword ptr [{x}]",
                    concat!($op, " {a}, qword ptr [{y}]"),
                    "mov qword ptr [{r}], {a}",
                    "mov {b}, qword ptr [{x} + 8]",
                    concat!($op, " {b}, qword ptr [{y} + 8]"),
                    "mov qword ptr [{r} + 8], {b}",
                    "mov {a}, qword ptr [{x} + 16]",
                    concat!($op, " {a}, qword ptr [{y} + 16]"),
                    "mov qword ptr [{r} + 16], {a}",
                    "mov {b}, qword ptr [{x} + 24]",
                    concat!($op, " {b}, qword ptr [{y} + 24]"),
                    "mov qword ptr [{r} + 24], {b}",
                    "lea {x}, [{x} + 32]",
                    "lea {y}, [{y} + 32]",
                    "lea {r}, [{r} + 32]",
                    "3:",
                    "dec {cnt}",
                    "jnz 2b",
                    "jmp 5f",
                    // The last `n mod 4` limbs, one at a time.
                    "4:",
                    "mov {a}, qword ptr [{x}]",
                    concat!($op, " {a}, qword ptr [{y}]"),
                    "mov qword ptr [{r}], {a}",
                    "lea {x}, [{x} + 8]",
                    "lea {y}, [{y} + 8]",
                    "lea {r}, [{r} + 8]",
                    "5:",
                    "dec {rem:e}",
                    "jnz 4b",
                    "setc {cf}",
                    r = inout(reg) r as u64 => _,
                    x = inout(reg) x as u64 => _,
                    y = inout(reg) y as u64 => _,
                    cnt = inout(reg) n as u64 => _,
                    rem = out(reg) _,
                    cf = inout(reg_byte) u8::from($carry) => carry_out,
                    a = out(reg) _,
                    b = out(reg) _,
                    options(nostack),
                );
            }
            carry_out != 0
        }
    };
}

/// Defines a big-endian kernel; `$op` is `adc` for an addition, `sbb` for a subtraction;
/// `$carry` names the carry (or borrow) coming in.
macro_rules! be_kernel {
    ($(#[$attr:meta])* $name:ident, $op:literal, $carry:ident) => {
        $(#[$attr])*
        #[inline]
        pub(super) unsafe fn $name(
            r_end: *mut u8,
            x_end: *const u8,
            y_end: *const u8,
            n: usize,
            $carry: bool,
        ) -> bool {
            let carry_out: u8;
            // SAFETY: the loops read limbs `0..n` below `x_end` and `y_end` and write limbs
            // `0..n` below `r_end`, which the caller guarantees are valid; x86 loads and
            // stores of eight bytes need no alignment. Each limb's inputs are loaded before
            // its output is stored, from limb 0 upwards, so an output equal to an input, or
            // above it, only overwrites bytes already read. The asm uses no stack and names
            // every register it writes; the flags it changes are clobbered by every `asm!`
            // by default.
            unsafe {
                asm!(
                    // As in the little-endian kernels.
                    "mov {rem:e}, {cnt:e}",
                    "and {rem:e}, 3",
                    "inc {rem:e}",
                    "shr {cnt}, 2",
                    "inc {cnt}",
                    "neg {cf}",
                    "jmp 3f",
                    // Rounds of four limbs, the 32 bytes below each pointer; the loop top
                    // aligned to 32 bytes (see above).
                    ".p2align 5",
                    "2:",
                    "mov {a}, qword ptr [{x} - 8]",
                    "mov {b}, qword ptr [{y} - 8]",
                    "bswap {a}",
                    "bswap {b}",
                    concat!($op, " {a}, {b}"),
                    "bswap {a}",
                    "mov qword ptr [{r} - 8], {a}",
                    "mov {a}, qword ptr [{x} - 16]",
                    "mov {b}, qword ptr [{y} - 16]",
                    "bswap {a}",
                    "bswap {b}",
                    concat!($op, " {a}, {b}"),
                    "bswap {a}",
                    "mov qword ptr [{r} - 16], {a}",
                    "mov {a}, qword ptr [{x} - 24]",
                    "mov {b}, qword ptr [{y} - 24]",
                    "bswap {a}",
                    "bswap {b}",
                    concat!($op, " {a}, {b}"),
                    "bswap {a}",
                    "mov qword ptr [{r} - 24], {a}",
                    "mov {a}, qword ptr [{x} - 32]",
                    "mov {b}, qword ptr [{y} - 32]",
                    "bswap {a}",
                    "bswap {b}",
                    concat!($op, " {a}, {b}"),
                    "bswap {a}",
                    "mov qword ptr [{r} - 32], {a}",
                    "lea {x}, [{x} - 32]",
                    "lea {y}, [{y} - 32]",
                    "lea {r}, [{r} - 32]",
                    "3:",
                    "dec {cnt}",
                    "jnz 2b",
                    "jmp 5f",
                    // The last `n mod 4` limbs, one at a time.
                    "4:",
                    "mov {a}, qword ptr [{x} - 8]",
                    "mov {b}, qword ptr [{y} - 8]",
                    "bswap {a}",
                    "bswap {b}",
                    concat!($op, " {a}, {b}"),
                    "bswap {a}",
                    "mov qword ptr [{r} - 8], {a}",
                    "lea {x}, [{x} - 8]",
                    "lea {y}, [{y} - 8]",
                    "lea {r}, [{r} - 8]",
                    "5:",
                    "dec {rem:e}",
                    "jnz 4b",
                    "setc {cf}",
                    r = inout(reg) r_end as u64 => _,
                    x = inout(reg) x_end as u64 => _,
                    y = inout(reg) y_end as u64 => _,
                    cnt = inout(reg) n as u64 => _,
                    rem = out(reg) _,
                    cf = inout(reg_byte) u8::from($carry) => carry_out,
                    a = out(reg) _,
                    b = out(reg) _,
                    options(nostack),
                );
            }
            carry_out != 0
        }
    };
}

le_kernel!(
    /// Sets the `n` little-endian limbs at `r` to `x + y + carry` and returns the carry out
    /// of the top limb.
    ///
    /// # Safety
    ///
    /// The `8 * n` bytes at `x` and at `y` must be valid for reads, and those at `r` for
    /// writes; no alignment is needed. `r` may equal `x`, `y` or both; otherwise the output
    /// must not overlap an input.
    add_n_le,
    "adc",
    carry
);

le_kernel!(
    /// Sets the `n` little-endian limbs at `r` to `x - y - borrow` and returns the borrow
    /// out of the top limb.
    ///
    /// # Safety
    ///
    /// As for [`add_n_le`].
    sub_n_le,
    "sbb",
    borrow
);

be_kernel!(
    /// Sets the `n` big-endian limbs below `r_end` to `x + y + carry` and returns the carry
    /// out of the top limb.
    ///
    /// # Safety
    ///
    /// The `8 * n` bytes below `x_end` and below `y_end` must be valid for reads, and those
    /// below `r_end` for writes; no alignment is needed. `r_end` may equal `x_end`, `y_end`
    /// or both, or lie above either (`r_end = x_end + s` with `s >= 1`); otherwise the
    /// output must not overlap an input.
    add_n_be,
    "adc",
    carry
);

be_kernel!(
    /// Sets the `n` big-endian limbs below `r_end` to `x - y - borrow` and returns the
    /// borrow out of the top limb.
    ///
    /// # Safety
    ///
    /// As for [`add_n_be`].
    sub_n_be,
    "sbb",
    borrow
);

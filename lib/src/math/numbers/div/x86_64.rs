//! The `x86_64` kernel of the division: a row of limb products subtracted from a partial
//! remainder, in inline assembly with `mulx`.
//!
//! The kernel is that of the multiplication's `addmul_1` with each addition turned into a
//! subtraction: two chains a step of four positions, chain 1 subtracting the low limbs of the
//! products from the row (`sbb mem, reg`), its borrow out of the step folded into the step's top
//! high limb (`adc reg, 0`, as the carry flag is set for a borrow), and chain 2 subtracting the high
//! limbs one position up, its borrow kept in the carry flag for chain 1 of the next step.
//!
//! Without BMI2, which is looked up once and kept by the standard library, the portable kernel
//! runs. This kernel has been tested for correctness under emulation only; its speed has not been
//! measured.

use core::arch::asm;

#[path = "portable.rs"]
mod portable;

/// Subtracts the `n` limbs at `x` times `y` from the `n` limbs at `r`, and returns the limb that
/// borrows out of them: the assembly with BMI2, the portable code without.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for reads and writes,
/// and apart from them.
#[inline]
pub(super) unsafe fn submul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    // SAFETY: forwarded; BMI2 is checked before its kernel runs.
    unsafe {
        if std::is_x86_feature_detected!("bmi2") {
            submul_1_bmi2(r, x, n, y)
        } else {
            portable::submul_1(r, x, n, y)
        }
    }
}

/// [`submul_1`] with `mulx`: two chains a step, as the module documentation describes.
///
/// # Safety
///
/// As for [`submul_1`], and the CPU must have BMI2.
#[target_feature(enable = "bmi2")]
unsafe fn submul_1_bmi2(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let borrow: u64;
    // SAFETY: the loops read limbs `0..n` of `x` and `r` and write limbs `0..n` of `r`; the asm uses
    // no stack and names every register it writes. `dec`, `lea`, `mov` and `mulx` leave the carry
    // flag as it was.
    unsafe {
        asm!(
            // The count of single steps, `n mod 4`, and of steps of four, `n / 4`, each one more,
            // as each loop is entered at its closing `dec`; then the borrow and `hp` cleared.
            "mov {h3}, {n}",
            "and {h3}, 3",
            "inc {h3}",
            "shr {n}, 2",
            "inc {n}",
            "xor {hp:e}, {hp:e}",
            "jmp 3f",
            "2:",
            "mulx {h0}, {l0}, qword ptr [{xp}]",
            "sbb qword ptr [{rp}], {l0}",
            "adc {h0}, 0",
            "sub qword ptr [{rp}], {hp}",
            "mov {hp}, {h0}",
            "lea {xp}, [{xp} + 8]",
            "lea {rp}, [{rp} + 8]",
            "3:",
            "dec {h3}",
            "jnz 2b",
            "jmp 5f",
            "4:",
            "mulx {h0}, {l0}, qword ptr [{xp}]",
            "mulx {h1}, {l1}, qword ptr [{xp} + 8]",
            "mulx {h2}, {l2}, qword ptr [{xp} + 16]",
            "mulx {h3}, {l3}, qword ptr [{xp} + 24]",
            "sbb qword ptr [{rp}], {l0}",
            "sbb qword ptr [{rp} + 8], {l1}",
            "sbb qword ptr [{rp} + 16], {l2}",
            "sbb qword ptr [{rp} + 24], {l3}",
            "adc {h3}, 0",
            "sub qword ptr [{rp}], {hp}",
            "sbb qword ptr [{rp} + 8], {h0}",
            "sbb qword ptr [{rp} + 16], {h1}",
            "sbb qword ptr [{rp} + 24], {h2}",
            "mov {hp}, {h3}",
            "lea {xp}, [{xp} + 32]",
            "lea {rp}, [{rp} + 32]",
            "5:",
            "dec {n}",
            "jnz 4b",
            "adc {hp}, 0",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _,
            in("rdx") y,
            hp = out(reg) borrow,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            options(nostack),
        );
    }
    borrow
}

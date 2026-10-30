//! The `x86_64` kernels of the multiplication: rows of limb products added into a product, in
//! inline assembly with `mulx`.
//!
//! `mulx` (BMI2: Intel from Haswell, AMD from Excavator and Zen) multiplies `rdx` by a limb into
//! two registers and leaves the flags alone, so the multiplications of a row can sit between the
//! `adc` of its carry chains. A row `r += x * y` adds two terms into each position, the low limb of
//! `x_k y` and the high limb of `x_{k-1} y`, in two chains a step of four positions, as the aarch64
//! kernels do: chain 1 adds the row's limbs from memory to the low limbs (`adc reg, [mem]`), and its
//! carry out of the step is folded into the step's top high limb, which can take it; chain 2 adds
//! the high limbs one position up, and its carry stays in the carry flag for chain 1 of the next
//! step. The loops count with `dec`, which leaves the carry flag alone, and move the pointers with
//! `lea`. A step of four takes 13 registers: `rdx`, the eight halves of the products, the high
//! limb carried in, the two pointers and the count; the steps for the limbs left over reuse one of
//! the eight as their count.
//!
//! Without BMI2, which is looked up once and kept by the standard library, the portable kernels
//! run. The products of the doubled triangle of a square, and its squares, are added by the
//! portable code too: that pass is linear in the length.
//!
//! These kernels have been tested for correctness under emulation only; their speed has not been
//! measured.

use core::arch::asm;

#[path = "portable.rs"]
mod portable;

/// Sets the `n` limbs at `r` to the `n` limbs at `x` times `y`, and returns the limb above them:
/// one chain, the low limb of each product plus the high limb of the one below it.
///
/// # Safety
///
/// The CPU must have BMI2; `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r`
/// for writes, and `r` must be `x` or apart from it.
#[target_feature(enable = "bmi2")]
unsafe fn mul_1_bmi2(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let carry: u64;
    // SAFETY: the loops read limbs `0..n` of `x` and write limbs `0..n` of `r`, each limb of `x`
    // before the limb of `r` at its position; the asm uses no stack and names every register it
    // writes. `dec`, `lea`, `mov` and `mulx` leave the carry flag as it was.
    unsafe {
        asm!(
            // The count of single steps, `n mod 4`, and of steps of four, `n / 4`, each one more,
            // as each loop is entered at its closing `dec`; then the carry and `hp` cleared.
            "mov {h3}, {n}",
            "and {h3}, 3",
            "inc {h3}",
            "shr {n}, 2",
            "inc {n}",
            "xor {hp:e}, {hp:e}",
            "jmp 3f",
            "2:",
            "mulx {h0}, {l0}, qword ptr [{xp}]",
            "adc {l0}, {hp}",
            "mov qword ptr [{rp}], {l0}",
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
            "adc {l0}, {hp}",
            "adc {l1}, {h0}",
            "mulx {h2}, {l2}, qword ptr [{xp} + 16]",
            "mulx {hp}, {l3}, qword ptr [{xp} + 24]",
            "adc {l2}, {h1}",
            "adc {l3}, {h2}",
            "mov qword ptr [{rp}], {l0}",
            "mov qword ptr [{rp} + 8], {l1}",
            "mov qword ptr [{rp} + 16], {l2}",
            "mov qword ptr [{rp} + 24], {l3}",
            "lea {xp}, [{xp} + 32]",
            "lea {rp}, [{rp} + 32]",
            "5:",
            "dec {n}",
            "jnz 4b",
            "adc {hp}, 0",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _,
            in("rdx") y,
            hp = out(reg) carry,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            options(nostack),
        );
    }
    carry
}

/// Adds the `n` limbs at `x` times `y` to the `n` limbs at `r`, and returns the limb that carries
/// out of them: two chains a step, as the module documentation describes.
///
/// # Safety
///
/// The CPU must have BMI2; `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r`
/// for reads and writes, and apart from them.
#[target_feature(enable = "bmi2")]
unsafe fn addmul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    let carry: u64;
    // SAFETY: as in `mul_1`, with limbs `0..n` of `r` read before they are written.
    unsafe {
        asm!(
            "mov {h3}, {n}",
            "and {h3}, 3",
            "inc {h3}",
            "shr {n}, 2",
            "inc {n}",
            "xor {hp:e}, {hp:e}",
            "jmp 3f",
            "2:",
            "mulx {h0}, {l0}, qword ptr [{xp}]",
            "adc {l0}, qword ptr [{rp}]",
            "adc {h0}, 0",
            "add {l0}, {hp}",
            "mov qword ptr [{rp}], {l0}",
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
            "adc {l0}, qword ptr [{rp}]",
            "adc {l1}, qword ptr [{rp} + 8]",
            "adc {l2}, qword ptr [{rp} + 16]",
            "adc {l3}, qword ptr [{rp} + 24]",
            "adc {h3}, 0",
            "add {l0}, {hp}",
            "adc {l1}, {h0}",
            "adc {l2}, {h1}",
            "adc {l3}, {h2}",
            "mov qword ptr [{rp}], {l0}",
            "mov qword ptr [{rp} + 8], {l1}",
            "mov qword ptr [{rp} + 16], {l2}",
            "mov qword ptr [{rp} + 24], {l3}",
            "mov {hp}, {h3}",
            "lea {xp}, [{xp} + 32]",
            "lea {rp}, [{rp} + 32]",
            "5:",
            "dec {n}",
            "jnz 4b",
            "adc {hp}, 0",
            rp = inout(reg) r => _, xp = inout(reg) x => _, n = inout(reg) n => _,
            in("rdx") y,
            hp = out(reg) carry,
            l0 = out(reg) _, l1 = out(reg) _, l2 = out(reg) _, l3 = out(reg) _,
            h0 = out(reg) _, h1 = out(reg) _, h2 = out(reg) _, h3 = out(reg) _,
            options(nostack),
        );
    }
    carry
}

/// Sets the `n` limbs at `r` to the `n` limbs at `x` times `y`, and returns the limb above them:
/// the kernel with BMI2, the portable code without, for the linear steps of Toom's methods.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `x` must be valid for reads and those at `r` for writes, and `r`
/// must be `x` or apart from it.
#[inline]
pub(super) unsafe fn mul_1(r: *mut u64, x: *const u64, n: usize, y: u64) -> u64 {
    // SAFETY: forwarded; BMI2 is checked before its kernel runs.
    unsafe {
        if bmi2() {
            mul_1_bmi2(r, x, n, y)
        } else {
            portable::mul_1(r, x, n, y)
        }
    }
}

/// Sets the `n` little-endian limbs at `r` to those at `x` times `y`, at any addresses, and returns
/// the limb above them: the kernel with BMI2, whose `mov`s take any address; the portable code,
/// which reads bytes, without.
///
/// # Safety
///
/// As for [`mul_1`], for the `8 n` bytes at each pointer.
#[inline]
pub(crate) unsafe fn mul_1_bytes(r: *mut u8, x: *const u8, n: usize, y: u64) -> u64 {
    // SAFETY: forwarded; BMI2 is checked before its kernel runs, and its pointers are not
    // dereferenced in Rust.
    unsafe {
        if bmi2() {
            mul_1_bmi2(r.cast(), x.cast(), n, y)
        } else {
            portable::mul_1_bytes(r, x, n, y)
        }
    }
}

/// Completes an exact division in place, as the portable code does.
///
/// # Safety
///
/// `n >= 1`; the `n` limbs at `r` must be valid for reads and writes.
#[inline]
pub(super) unsafe fn divexact_finish(r: *mut u64, n: usize) {
    // SAFETY: forwarded.
    unsafe { portable::divexact_finish(r, n) }
}

/// Whether the CPU has BMI2, which the kernels need: looked up once, by the standard library.
#[inline(always)]
fn bmi2() -> bool {
    std::is_x86_feature_detected!("bmi2")
}

/// Sets the `nx + ny` limbs at `r` to the product of the `nx` limbs at `x` and the `ny` at `y`:
/// the portable code without BMI2 and for the smallest shapes, otherwise one row a pass.
///
/// # Safety
///
/// `nx >= ny >= 1`; the limbs of `x` and `y` must be valid for reads and those of `r` for writes,
/// and `r` apart from both.
#[inline]
pub(super) unsafe fn mul_basecase(r: *mut u64, x: *const u64, nx: usize, y: *const u64, ny: usize) {
    // SAFETY: forwarded; BMI2 is checked before its kernels run, and each row adds into limbs of
    // `r` that the rows before it wrote, and writes the one above them.
    unsafe {
        if !bmi2() || nx <= 3 || (ny == 1 && nx <= 8) {
            portable::mul_basecase(r, x, nx, y, ny);
            return;
        }
        *r.add(nx) = mul_1_bmi2(r, x, nx, *y);
        for j in 1..ny {
            *r.add(nx + j) = addmul_1(r.add(j), x, nx, *y.add(j));
        }
    }
}

/// Sets the `2n` limbs at `r` to the square of the `n` limbs at `x`: the products of two
/// different limbs once, by rows, then doubled with the squares added.
///
/// # Safety
///
/// `n >= 1`; the limbs of `x` must be valid for reads and those of `r` for writes, and apart from
/// them.
#[inline]
pub(super) unsafe fn sqr_basecase(r: *mut u64, x: *const u64, n: usize) {
    // SAFETY: as in `portable::sqr_basecase`, whose last passes this shares.
    unsafe {
        if !bmi2() || n <= 3 {
            portable::sqr_basecase(r, x, n);
            return;
        }
        *r.add(n) = mul_1_bmi2(r.add(1), x.add(1), n - 1, *x);
        for i in 1..n - 1 {
            *r.add(n + i) = addmul_1(r.add(2 * i + 1), x.add(i + 1), n - i - 1, *x.add(i));
        }
        *r = 0;
        *r.add(2 * n - 1) = 0;
        portable::double_and_add_squares(r, x, n);
    }
}

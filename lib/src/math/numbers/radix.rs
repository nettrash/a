//! Numbers as text in a radix from 2 to 36: [`Number::to_str_radix`], [`Number::from_str_radix`],
//! and the formatting traits `Display` (decimal), `LowerHex`, `UpperHex`, `Octal` and `Binary`.
//!
//! The text is the sign and the digits of the magnitude, the most significant first, with `-` for
//! a negative number and nothing for the others: `-255` is `-ff` in hexadecimal, not the two's
//! complement that the formatting of a primitive integer gives. Digits above 9 are the letters,
//! lower-case when written, either case when read.
//!
//! In a radix that is a power of two, each digit is a group of bits of the magnitude, and the
//! text is written or read in one pass over the bytes, where they are: hexadecimal two digits a
//! byte from a table, binary eight a byte spread by a multiplication, the others in groups of
//! bytes. In another radix, the magnitude goes through the largest power `b` of the radix that
//! fits in a limb, `10^19` for decimal, whose remainders are blocks of digits.
//!
//! Written, a short magnitude is divided by `b` again and again, limb by limb, with the
//! reciprocal of `b` (see [`Divisor`]). Each limb of a division waits for the remainder of the
//! limb above it, about ten cycles, and a division waits for the quotient of the one before it;
//! but the next division can start on the top limb of that quotient as soon as it is known, so
//! the divisions run in groups of up to [`PASSES`], each a limb behind the one before it, and the
//! core overlaps their chains. The blocks of decimal digits are made eight digits at a time in the
//! lanes of a `u64`, by multiplications, with no division. A long magnitude is split by divide
//! and conquer, divided by `b^(2^i)` for the largest `i` that halves it, and the quotient and the
//! remainder written the same way, the remainder with as many digits as `b^(2^i)` has places,
//! zeros at the front. Up to two limbs of decimal take a path of their own, in registers.
//!
//! Read, decimal digits are checked and turned into values eight at a time in the lanes of a
//! `u64`, other radices a digit at a time; up to 38 decimal digits are read into a `u128`, and the
//! blocks of longer text multiplied in, `x b + v`, a pass over the limbs each, in place. The
//! passes run in waves as the divisions do, [`READ_PASSES`] at a time. Long text is split in
//! halves, the upper one multiplied by the power of `b` of the lower one's places and the lower
//! one added. The powers of divide and conquer, both ways, drop their zero bytes at the bottom
//! (`10^19` is `5^19 2^19`, so a third of the bytes of its powers are zeros), and the part of a
//! number below them is set aside instead of divided or multiplied.

use super::add::{read, read_small, write};
use super::div::{divide_2by1, reciprocal};
use super::mul::{Operand, Product, to_limbs};
use super::{ByteOrder, Divisor, Number};
use core::fmt;
use core::mem::MaybeUninit;

/// The digits of the radices up to 36.
const DIGITS: &[u8; 36] = b"0123456789abcdefghijklmnopqrstuvwxyz";

/// From this many limbs, a magnitude is written by divide and conquer. Measured on an Apple M4
/// Max in decimal, 80 took at most 5.5% longer than the best of 40, 64, 80, 100 and 150 from 384
/// bytes to 4 KiB, where 64 took up to 20% longer (at 512 bytes), 100 up to 10% (1.5 KiB), 40 36%
/// (384 bytes) and 150 16% (2 KiB). It weighs the waves of divisions against the division of a
/// long magnitude by a power: when the division gets faster, it falls.
const WRITE_SPLIT: usize = 80;

/// From this many blocks of text, it is read by divide and conquer. Measured on an Apple M4 Max in
/// decimal, 220 took at most 7% longer than the best of 160, 220 and 300 from 1 to 64 KiB, the
/// others up to 14% longer at some length; 100 took 33% longer for 1 KiB, and 600 15% to 35%
/// longer from 3 to 12 KiB. It weighs the waves of passes against the products of halves, by
/// Karatsuba's method from 24 limbs: when the multiplication gets faster, it falls.
const READ_SPLIT: usize = 220;

/// The most divisions by `b` that run at once when a magnitude is written: see [`divide_blocks`].
/// Measured on an Apple M4 Max in decimal with one size of group for every length, 6 and 8 were
/// within 4% of each other from 128 to 256 bytes, and up to 12% faster than 4, which was the best
/// for 3 to 8 limbs, as smaller groups waste fewer divisions above the top; 2 was slower from 24
/// bytes. So the groups shrink with the limbs left.
const PASSES: usize = 8;

/// The largest power of ten in a limb, `10^19`, whose remainders are the blocks of decimal digits.
/// Its top bit is set, so it needs no shift to divide by.
const TEN_19: u64 = 10_000_000_000_000_000_000;

/// The reciprocal of [`TEN_19`], `floor((B^2 - 1) / 10^19) - B`, as [`reciprocal`] computes it.
const TEN_19_INVERSE: u64 = (u128::MAX / TEN_19 as u128 - (1 << 64)) as u64;

/// Text of up to this many digits is written in a buffer on the stack by the formatting traits.
const STACK_TEXT: usize = 160;

/// Text of up to this many blocks is read into limbs on the stack.
const STACK_BLOCKS: usize = 64;

/// The blocks multiplied in at once when text is read: see [`mul_add_group`]. Measured on an Apple
/// M4 Max for 1 to 4 KiB of decimal text, 6 and 8 took 0.83 cycles a limb a block, 4 0.88 and 2
/// 1.2, against 1.24 by a pass of the multiplication's kernel and an addition of the block.
const READ_PASSES: usize = 6;

/// From this many limbs, blocks are multiplied in [`READ_PASSES`] at a time: below, the waves
/// that start and end a group outweigh the passes.
const READ_WAVES: usize = 2;

/// The largest power of `radix` that fits in a limb, `b = radix^k`, and its exponent `k`: the
/// places of the blocks of digits.
fn block(radix: u32) -> (u64, usize) {
    let (mut b, mut k) = (u64::from(radix), 1);
    while let Some(next) = b.checked_mul(u64::from(radix)) {
        b = next;
        k += 1;
    }
    (b, k)
}

/// Panics for a radix outside `2..=36`, as the primitive integers' `from_str_radix` does.
#[track_caller]
fn check_radix(radix: u32) {
    assert!(
        (2..=36).contains(&radix),
        "from_str_radix_int: must lie in the range `[2, 36]` - found {radix}"
    );
}

impl Number {
    /// Returns the number as text in `radix`: `-` for a negative number, then the digits of the
    /// magnitude, the most significant first, without zeros in front, letters lower-case; `0` for
    /// zero.
    ///
    /// # Panics
    ///
    /// When `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_i32(-255, ByteOrder::BigEndian);
    /// assert_eq!(n.to_str_radix(16), "-ff");
    /// assert_eq!(n.to_str_radix(10), "-255");
    /// assert_eq!(n.to_str_radix(2), "-11111111");
    /// ```
    #[track_caller]
    pub fn to_str_radix(&self, radix: u32) -> String {
        check_radix(radix);
        let x = Operand::of(self);
        let sign = usize::from(self.negative && x.significant != 0);
        let mut out =
            Vec::<u8>::with_capacity(digits_bound(x.significant, radix).saturating_add(sign));
        // SAFETY: the vector has room for the sign and the digits, which are all written before
        // the length is set; they are ASCII, so the string is UTF-8. `self` is borrowed while it
        // is read.
        unsafe {
            let p = out.as_mut_ptr();
            *p = b'-';
            let n = write_magnitude(x, radix, p.add(sign));
            out.set_len(sign + n);
            String::from_utf8_unchecked(out)
        }
    }

    /// Reads a number from text in `radix`: an optional sign, `+` or `-`, then at least one digit,
    /// from `0` to the radix's last, letters in either case, with no other characters, as the
    /// primitive integers' `from_str_radix` reads; zeros in front count for nothing. The number is
    /// built in `byte_order`, in the shortest form; `-0` is zero, not negative.
    ///
    /// # Errors
    ///
    /// [`ParseNumberError`] of kind [`NumberErrorKind::Empty`] for empty text, and
    /// [`NumberErrorKind::InvalidDigit`] for a character that is not a digit of the radix, or a
    /// sign alone.
    ///
    /// # Panics
    ///
    /// When `radix` is not from 2 to 36.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number, NumberErrorKind};
    ///
    /// let n = Number::from_str_radix("-FF", 16, ByteOrder::BigEndian).unwrap();
    /// assert_eq!(n.value, [0xFF]);
    /// assert!(n.is_negative());
    ///
    /// // 10^21 = 0x36_35C9_ADC5_DEA0_0000.
    /// let n = Number::from_str_radix("1000000000000000000000", 10, ByteOrder::LittleEndian);
    /// assert_eq!(n.unwrap().value, [0, 0, 0xA0, 0xDE, 0xC5, 0xAD, 0xC9, 0x35, 0x36]);
    ///
    /// let error = Number::from_str_radix("12a", 10, ByteOrder::BigEndian).err().unwrap();
    /// assert_eq!(*error.kind(), NumberErrorKind::InvalidDigit);
    /// ```
    #[track_caller]
    pub fn from_str_radix(
        src: &str,
        radix: u32,
        byte_order: ByteOrder,
    ) -> Result<Number, ParseNumberError> {
        check_radix(radix);
        let bytes = src.as_bytes();
        let (negative, digits) = match bytes.first() {
            None => return Err(error(NumberErrorKind::Empty)),
            Some(b'-') => (true, &bytes[1..]),
            Some(b'+') => (false, &bytes[1..]),
            Some(_) => (false, bytes),
        };
        if digits.is_empty() {
            return Err(error(NumberErrorKind::InvalidDigit));
        }
        if radix == 10 {
            read_decimal(digits, negative, byte_order)
        } else if radix.is_power_of_two() {
            read_bits(digits, radix.trailing_zeros(), negative, byte_order)
        } else {
            read_long(digits, radix, negative, byte_order)
        }
        .ok_or(error(NumberErrorKind::InvalidDigit))
    }
}

/// The error of the kind `kind`.
#[cold]
fn error(kind: NumberErrorKind) -> ParseNumberError {
    ParseNumberError { kind }
}

// Writing

/// The most digits that a magnitude of `bytes` significant bytes has in `radix`, at least one:
/// `log_radix(2)` digits a bit, rounded up, and one more. Computed in 64 bits, so that it does not
/// wrap on 32-bit targets, and `usize::MAX` when it does not fit, which no vector can hold.
fn digits_bound(bytes: usize, radix: u32) -> usize {
    let bits = 8 * bytes as u64;
    let digits = if radix == 10 {
        // log10(2) < 1234 / 4096.
        ((bits * 1234) >> 12) + 1
    } else {
        // At least `floor(log2(radix))` bits a digit.
        bits / u64::from(31 - radix.leading_zeros()) + 1
    };
    usize::try_from(digits).unwrap_or(usize::MAX)
}

/// Writes the digits of the magnitude of the operand `x` in `radix` at `dst`, without zeros in
/// front, `0` for zero, and returns how many.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `dst` for writing
/// [`digits_bound`]`(x.significant, radix)` bytes.
unsafe fn write_magnitude(x: Operand, radix: u32, dst: *mut u8) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        if x.significant == 0 {
            *dst = b'0';
            return 1;
        }
        if radix.is_power_of_two() {
            return write_bits(x, radix.trailing_zeros(), dst);
        }
        let n = x.significant.div_ceil(8);
        if radix == 10 && n <= 2 {
            let order = if x.big_endian {
                ByteOrder::BigEndian
            } else {
                ByteOrder::LittleEndian
            };
            return write_u128(read_small(significant(x), order), dst);
        }
        if n < WRITE_SPLIT {
            // The limbs, and the room above them that the divisions write quotients of zero into.
            let mut u = [MaybeUninit::<u64>::uninit(); WRITE_SPLIT + PASSES];
            let u = u.as_mut_ptr().cast::<u64>();
            to_limbs(x, u);
            return write_limbs(u, n, radix, None, dst);
        }
        let mut powers = Vec::new();
        write_split(le_number(x), radix, None, dst, &mut powers)
    }
}

/// The significant bytes of the operand `x`, in its byte order.
///
/// # Safety
///
/// The operand's vector must be valid for reads while the slice is alive.
unsafe fn significant<'a>(x: Operand) -> &'a [u8] {
    let start = if x.big_endian {
        x.len - x.significant
    } else {
        0
    };
    // SAFETY: the significant bytes lie in the operand's vector.
    unsafe { core::slice::from_raw_parts(x.ptr.add(start), x.significant) }
}

/// The magnitude of the operand `x` as a little-endian number in the shortest form.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
unsafe fn le_number(x: Operand) -> Number {
    // SAFETY: as the caller guarantees.
    let bytes = unsafe { significant(x) };
    let le = if x.big_endian {
        bytes.iter().rev().copied().collect()
    } else {
        bytes.to_vec()
    };
    Number {
        value: le,
        negative: false,
        byte_order: ByteOrder::LittleEndian,
    }
}

/// Writes the decimal digits of `x`, not zero, at `dst`, without zeros in front, and returns how
/// many: from 1 to 20.
///
/// # Safety
///
/// `dst` must be valid for writing as many bytes as `x` has digits.
#[inline]
unsafe fn write_u64(x: u64, dst: *mut u8) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        if x >= TEN_19 {
            // 20 digits: `x < 2 10^19`.
            *dst = b'1';
            write_block(x - TEN_19, dst.add(1));
            20
        } else {
            write_top(x, dst)
        }
    }
}

/// Writes the decimal digits of `x`, not zero, at `dst`, without zeros in front, and returns how
/// many: from 1 to 39. Above a limb, one division by `10^19` gives the last block, and the quotient,
/// below `2^65`, is written as a limb, or for 39 digits as its top digit and a block.
///
/// # Safety
///
/// `dst` must be valid for writing as many bytes as `x` has digits.
#[inline]
unsafe fn write_u128(x: u128, dst: *mut u8) -> usize {
    let (hi, lo) = ((x >> 64) as u64, x as u64);
    // SAFETY: as the caller guarantees.
    unsafe {
        if hi == 0 {
            return write_u64(lo, dst);
        }
        // The quotient `top B + q` and the last block `r`.
        let (top, hi) = if hi >= TEN_19 {
            (1, hi - TEN_19)
        } else {
            (0, hi)
        };
        let (q, r) = divide_2by1(hi, lo, TEN_19, TEN_19_INVERSE);
        let n = if top == 0 {
            write_u64(q, dst)
        } else {
            // `B + q`, from `1.8 10^19` to `3.7 10^19`: its first digit is 1, 2 or 3.
            let y = (1 << 64) + u128::from(q);
            let t = 1 + u64::from(y >= 2 * TEN_19 as u128) + u64::from(y >= 3 * TEN_19 as u128);
            *dst = b'0' + t as u8;
            write_block((y - u128::from(t) * u128::from(TEN_19)) as u64, dst.add(1));
            20
        };
        write_block(r, dst.add(n));
        n + 19
    }
}

/// Writes the digits of the magnitude in the `n` limbs at `u`, least significant first, which it
/// clobbers, in `radix`, not a power of two, at `dst`: exactly `width` digits, zeros in front,
/// when given (enough for the magnitude), and otherwise without zeros in front, `0` for zero.
/// Returns how many it wrote.
///
/// # Safety
///
/// `n < WRITE_SPLIT`; the `n + PASSES` limbs at `u` must be valid for reads and writes, and `dst`
/// for writing the digits.
unsafe fn write_limbs(
    u: *mut u64,
    n: usize,
    radix: u32,
    width: Option<usize>,
    dst: *mut u8,
) -> usize {
    // A block takes more than 58 bits, as `b > 2^64 / 36`: at most two blocks a limb, and the
    // groups of divisions may give up to `PASSES` blocks of zero above the top.
    let mut blocks = [MaybeUninit::<u64>::uninit(); 2 * WRITE_SPLIT + PASSES];
    let blocks = blocks.as_mut_ptr().cast::<u64>();
    // SAFETY: as the caller guarantees.
    unsafe {
        let count = if radix == 10 {
            decimal_blocks(u, n, blocks)
        } else {
            let (b, _) = block(radix);
            let s = b.leading_zeros();
            let base = Base {
                s,
                d: b << s,
                v: reciprocal(b << s),
            };
            if SELECTS.contains(&radix) {
                select_blocks(u, n, base, blocks)
            } else {
                other_blocks(u, n, base, blocks)
            }
        };
        // The blocks below `count` are written.
        put_blocks(
            core::slice::from_raw_parts(blocks, count),
            radix,
            width,
            dst,
        )
    }
}

/// The block `b` of a radix as its divisions take it: `b = d / 2^s` for the normalized `d`, whose
/// reciprocal is `v`.
#[derive(Clone, Copy)]
struct Base {
    s: u32,
    d: u64,
    v: u64,
}

/// [`divide_blocks`] by `10^19`, with the constants in the code.
///
/// # Safety
///
/// As for [`divide_blocks`].
unsafe fn decimal_blocks(u: *mut u64, n: usize, blocks: *mut u64) -> usize {
    let base = Base {
        s: 0,
        d: TEN_19,
        v: TEN_19_INVERSE,
    };
    // SAFETY: as the caller guarantees.
    unsafe { divide_blocks::<false>(u, n, base, blocks) }
}

/// [`divide_blocks`] by the block of another radix, with the second correction of a quotient a
/// branch.
///
/// # Safety
///
/// As for [`divide_blocks`].
unsafe fn other_blocks(u: *mut u64, n: usize, base: Base, blocks: *mut u64) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe { divide_blocks::<false>(u, n, base, blocks) }
}

/// [`divide_blocks`] by the block of a radix in [`SELECTS`], with both corrections of a quotient
/// selects.
///
/// # Safety
///
/// As for [`divide_blocks`].
unsafe fn select_blocks(u: *mut u64, n: usize, base: Base, blocks: *mut u64) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe { divide_blocks::<true>(u, n, base, blocks) }
}

/// The radices whose blocks, normalized, often need the second correction of the quotient
/// estimated by the reciprocal: in a simulation of random steps, 6.6% of them for 6 and 36, 3.6%
/// for 29, 2.8% for 34 and 1.5% for 26; 0.44% for 3 and 9, 0.007% for 10, and fewer for the
/// others. Measured on an Apple M4 Max, selects made 512 bytes in radix 36 12% faster, and radices
/// 3 and 7 10% slower, as they lengthen the chain of a division.
const SELECTS: [u32; 5] = [6, 26, 29, 34, 36];

/// Divides the magnitude in the `n` limbs at `u`, least significant first, which it clobbers, by
/// the block `b` again and again until nothing is left, and writes the remainders at `blocks`,
/// the least significant first; returns how many, without those of zero above the top, at least
/// one.
///
/// The limbs of one division wait for each other: the remainder of a limb is the high limb of
/// the next one's dividend, and a step, [`div_2by1`](divide_2by1), takes about ten cycles. But
/// the next division needs only the quotient's limbs, from the top: so the divisions run in
/// groups, in waves, each a limb behind the one before it, which hands it its quotient limbs in
/// registers (see [`group`]). Their chains are independent, and the core runs them side by side.
/// A group of `K` divisions takes `n + K - 1` waves, and leaves a number about `K` limbs shorter:
/// the groups take as many divisions as the limbs left but one, rounded down to 1, 2, 4 or
/// [`PASSES`], so that few blocks of zero come out above the top, and the last limb is divided
/// alone.
///
/// # Safety
///
/// The `n + PASSES` limbs at `u` must be valid for reads and writes, and `blocks` for writing
/// `2 n + PASSES` limbs.
#[inline(always)]
unsafe fn divide_blocks<const SELECT: bool>(
    u: *mut u64,
    mut n: usize,
    base: Base,
    blocks: *mut u64,
) -> usize {
    let mut count = 0;
    // SAFETY: as the caller guarantees: a group of `K` takes `K` blocks of a number of more than
    // `K` limbs, each more than 58 bits, so there are fewer than `2 n` blocks before the last
    // group, which may write `PASSES - 1` zeros above them.
    unsafe {
        while n > 1 {
            let out = blocks.add(count);
            let (taken, left) = match n - 1 {
                1 => (1, group::<1, SELECT>(u, n, base, out)),
                2 | 3 => (2, group::<2, SELECT>(u, n, base, out)),
                4..=7 => (4, group::<4, SELECT>(u, n, base, out)),
                _ => (PASSES, group::<PASSES, SELECT>(u, n, base, out)),
            };
            count += taken;
            n = left;
        }
        if n == 1 {
            let (x, b) = (*u, base.d >> base.s);
            *blocks.add(count) = x % b;
            *blocks.add(count + 1) = x / b;
            count += 2;
        }
        if count == 0 {
            *blocks = 0;
            count = 1;
        }
        while count > 1 && *blocks.add(count - 1) == 0 {
            count -= 1;
        }
    }
    count
}

/// One step of a division by the block `b`: the dividend `r B + x` for the remainder `r < b`
/// carried as `r 2^s`, shifted by `s` bits to be divided by the normalized `d`; returns the
/// quotient and the remainder, again as `r 2^s`.
///
/// With `SELECT`, both corrections of the quotient are selects; otherwise the second is a branch,
/// as in [`div_2by1`](divide_2by1), which costs nothing while it is rare, and is a mispredicted
/// branch and a call when it is not (see [`SELECTS`]).
#[inline(always)]
fn step<const SELECT: bool>(r: u64, x: u64, base: Base) -> (u64, u64) {
    // `x >> (64 - s)` in two steps, so that `s = 0` shifts everything out.
    let (u1, u0) = (r | ((x >> 1) >> (63 - base.s)), x << base.s);
    if !SELECT {
        return divide_2by1(u1, u0, base.d, base.v);
    }
    // Algorithm 4 of Möller and Granlund, as `div_2by1` computes it, with the second correction
    // a select.
    let (d, v) = (base.d, base.v);
    let p = (u128::from(v) * u128::from(u1))
        .wrapping_add((u128::from(u1.wrapping_add(1)) << 64) | u128::from(u0));
    let (q, q0) = ((p >> 64) as u64, p as u64);
    let r = u0.wrapping_sub(q.wrapping_mul(d));
    let mask = u64::from(r > q0).wrapping_neg();
    let (q, r) = (q.wrapping_add(mask), r.wrapping_add(mask & d));
    let more = r >= d;
    (
        q + u64::from(more),
        r - (d & u64::from(more).wrapping_neg()),
    )
}

/// `K` divisions of the `n >= 2` limbs at `u` by the block `b`, one after another, in waves (see
/// [`divide_blocks`]): writes the `K` remainders at `out`, the least significant first, leaves the
/// quotient by `b^K` in the limbs at `u`, and returns its length.
///
/// In wave `t`, division `p` takes the limb at `n - 1 - (t - p)`: the first one reads it from `u`,
/// the others take it from the one before, in the last wave, and the last one writes its quotient
/// limb back. Above the top, the limbs are zero; below the bottom, a division is done, and its
/// remainder is kept when it is.
///
/// # Safety
///
/// The `n + K` limbs at `u` must be valid for reads and writes, and the `K` at `out` for writes.
#[inline(always)]
unsafe fn group<const K: usize, const SELECT: bool>(
    u: *mut u64,
    n: usize,
    base: Base,
    out: *mut u64,
) -> usize {
    // The remainders, as `r 2^s`, and the quotient limb that each division hands the next one.
    let mut r = [0u64; K];
    let mut c = [0u64; K];
    // SAFETY: the first division reads limbs `n - 1` down to 0; the last one writes limb
    // `n - 1 + K - 1 - t` in wave `t`, from `n + K - 2` down to 0, after the first one read it,
    // as `t` grows.
    unsafe {
        for t in 0..n {
            let mut p = K;
            while p > 1 {
                p -= 1;
                let (q, rest) = step::<SELECT>(r[p], c[p - 1], base);
                r[p] = rest;
                if p == K - 1 {
                    *u.add(n + K - 2 - t) = q;
                } else {
                    c[p] = q;
                }
            }
            let (q, rest) = step::<SELECT>(r[0], *u.add(n - 1 - t), base);
            r[0] = rest;
            if K == 1 {
                *u.add(n - 1 - t) = q;
            } else {
                c[0] = q;
            }
        }
        *out = r[0] >> base.s;
        // The waves after the first division is done: in wave `n + i`, divisions `i + 1` up.
        for i in 0..K - 1 {
            let mut p = K;
            while p > i + 1 {
                p -= 1;
                let (q, rest) = step::<SELECT>(r[p], c[p - 1], base);
                r[p] = rest;
                if p == K - 1 {
                    *u.add(K - 2 - i) = q;
                } else {
                    c[p] = q;
                }
            }
            *out.add(i + 1) = r[i + 1] >> base.s;
        }
    }
    let mut len = n;
    // SAFETY: the limbs below `len` are the quotient's.
    while len > 0 && unsafe { *u.add(len - 1) } == 0 {
        len -= 1;
    }
    len
}

/// Writes the blocks, the least significant first, of `k` digits each in `radix`, as digits at
/// `dst`: exactly `width` digits when given, zeros in front, and otherwise without zeros in front,
/// `0` for zero. Returns how many.
///
/// # Safety
///
/// `blocks` must not be empty, and `dst` valid for writing the digits; `width`, when given, must
/// have room for the digits of the blocks but their zeros in front.
unsafe fn put_blocks(blocks: &[u64], radix: u32, width: Option<usize>, dst: *mut u8) -> usize {
    let (_, k) = block(radix);
    let count = blocks.len();
    // SAFETY: as the caller guarantees.
    unsafe {
        // The digits in all, and the blocks written whole, from the least significant; the
        // digits above them first.
        let (total, whole) = match width {
            None => {
                let top = blocks[count - 1];
                let head = if radix == 10 {
                    write_top(top, dst)
                } else {
                    write_short_top(top, Radix::new(radix), dst)
                };
                (head + k * (count - 1), count - 1)
            }
            Some(w) => {
                // The blocks that fit whole; above them the low digits of the next one, or zeros.
                let whole = (w / k).min(count);
                let head = w - k * whole;
                if head > 0 {
                    let next = if whole < count { blocks[whole] } else { 0 };
                    write_partial(next, radix, k, head, dst);
                }
                (w, whole)
            }
        };
        let mut at = dst.add(total);
        if radix == 10 {
            for &b in &blocks[..whole] {
                at = at.sub(k);
                write_block(b, at);
            }
        } else {
            // Four blocks at once: the digits of a block wait for each other, a division each,
            // and the four chains run side by side.
            let radix = Radix::new(radix);
            let (fours, rest) = blocks[..whole].as_chunks::<4>();
            for four in fours {
                at = at.sub(4 * k);
                write_four(*four, radix, at, k);
            }
            for &b in rest {
                at = at.sub(k);
                write_digits(b, radix, at, k);
            }
        }
        total
    }
}

/// Writes the `n` low digits of `x` in `radix`, zeros in front, at `dst`: a block cut short, or
/// zeros alone for `x` of zero.
///
/// # Safety
///
/// `dst` must be valid for writing `n` bytes; `x` must have at most `n` digits.
unsafe fn write_partial(x: u64, radix: u32, k: usize, n: usize, dst: *mut u8) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if n <= k {
            let mut buffer = [0u8; 64];
            if radix == 10 {
                write_block(x, buffer.as_mut_ptr());
            } else {
                write_digits(x, Radix::new(radix), buffer.as_mut_ptr(), k);
            }
            core::ptr::copy_nonoverlapping(buffer.as_ptr().add(k - n), dst, n);
        } else {
            core::ptr::write_bytes(dst, b'0', n - k);
            write_partial(x, radix, k, k, dst.add(n - k));
        }
    }
}

/// Writes `x` in `radix` into the `n` bytes at `dst`, the most significant digit first, with zeros
/// in front: `x` must have at most `n` digits.
///
/// # Safety
///
/// `dst` must be valid for writing `n` bytes.
unsafe fn write_digits(mut x: u64, radix: Radix, dst: *mut u8, n: usize) {
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in (0..n).rev() {
            let (q, digit) = radix.div_rem(x);
            *dst.add(i) = DIGITS[digit as usize];
            x = q;
        }
    }
}

/// Writes the blocks `x`, the least significant first, of `n` digits each in `radix`, into the
/// `4 n` bytes at `dst`, the most significant digit first: the four chains of divisions in one
/// loop.
///
/// # Safety
///
/// `dst` must be valid for writing `4 n` bytes; each block must have at most `n` digits.
unsafe fn write_four(mut x: [u64; 4], radix: Radix, dst: *mut u8, n: usize) {
    // SAFETY: block `j` goes to the `n` bytes at `dst + (3 - j) n`.
    unsafe {
        for i in (0..n).rev() {
            for (j, x) in x.iter_mut().enumerate() {
                let (q, digit) = radix.div_rem(*x);
                *dst.add((3 - j) * n + i) = DIGITS[digit as usize];
                *x = q;
            }
        }
    }
}

/// Writes the digits of the top block `x` in `radix`, not 10, at `dst`, without zeros in front, `0`
/// for zero; returns how many.
///
/// # Safety
///
/// `dst` must be valid for writing the digits.
unsafe fn write_short_top(mut x: u64, radix: Radix, dst: *mut u8) -> usize {
    let mut buffer = [0u8; 64];
    let mut i = 64;
    loop {
        i -= 1;
        let (q, digit) = radix.div_rem(x);
        buffer[i] = DIGITS[digit as usize];
        x = q;
        if x == 0 {
            break;
        }
    }
    // SAFETY: as the caller guarantees.
    unsafe { core::ptr::copy_nonoverlapping(buffer.as_ptr().add(i), dst, 64 - i) };
    64 - i
}

/// A radix from 3 to 36 to divide by with a multiplication, as Granlund and Montgomery divide by
/// an invariant integer: exact for every `u64`. The hardware's division takes about 10 cycles and
/// does not overlap with the next one, which made the digits of radices other than 10 the larger
/// part of writing them: on an Apple M4 Max, 512 bytes in radix 36 took 16,700 cycles with it.
#[derive(Clone, Copy)]
struct Radix {
    /// The radix.
    d: u64,
    /// `floor(2^64 (2^l - d) / d) + 1`, below `2^64` as `2^l - d < d`.
    m: u64,
    /// `ceil(log2 d)`, at least 2.
    l: u32,
}

impl Radix {
    /// The radix `d`, from 3 to 36, prepared.
    fn new(d: u32) -> Radix {
        let d = u64::from(d);
        let l = 64 - (d - 1).leading_zeros();
        let m = (((1u128 << l) - u128::from(d)) << 64) / u128::from(d) + 1;
        Radix { d, m: m as u64, l }
    }

    /// The quotient and the remainder of `n` by the radix.
    #[inline(always)]
    fn div_rem(self, n: u64) -> (u64, u64) {
        // `t <= n`, as `m < 2^64`; the halved difference does not overflow.
        let t = ((u128::from(self.m) * u128::from(n)) >> 64) as u64;
        let q = (t + ((n - t) >> 1)) >> (self.l - 1);
        (q, n - q * self.d)
    }
}

// Decimal digits in lanes

/// The eight decimal digits of `x < 10^8` as values from 0 to 9, the most significant in the
/// lowest byte: split in halves of four digits in the 32-bit lanes, each in pairs in the 16-bit
/// lanes, each in digits in the bytes, by multiplications that divide by 10^4, 100 and 10 in all
/// lanes at once (`x · m >> k` with `m` the reciprocal rounded up: exact for these ranges).
#[inline(always)]
fn eight_digits(x: u64) -> u64 {
    // x / 10^4 for x < 4.9 10^8.
    let hi = (x * 109_951_163) >> 40;
    let x = hi | ((x - hi * 10_000) << 32);
    // Lanes / 100, for lanes below 43,690.
    let q = ((x * 10_486) >> 20) & 0x0000_007F_0000_007F;
    let x = q | ((x - q * 100) << 16);
    // Lanes / 10, for lanes below 170.
    let t = ((x * 103) >> 10) & 0x000F_000F_000F_000F;
    t | ((x - t * 10) << 8)
}

/// Writes the 19 decimal digits of the block `x < 10^19` at `dst`, zeros in front.
///
/// # Safety
///
/// `dst` must be valid for writing 19 bytes.
#[inline(always)]
unsafe fn write_block(x: u64, dst: *mut u8) {
    let (a, b, c) = split_block(x);
    let zeros = 0x3030_3030_3030_3030;
    // SAFETY: the writes lie in the 19 bytes: `[0, 4)`, then `[3, 11)` over its last byte, and
    // `[11, 19)`.
    unsafe {
        write(dst, (three_digits(a) as u32 | 0x0030_3030).to_le_bytes());
        write(dst.add(3), (eight_digits(b) | zeros).to_le_bytes());
        write(dst.add(11), (eight_digits(c) | zeros).to_le_bytes());
    }
}

/// The block `x < 10^19` as its top three digits, `x / 10^16`, and two groups of eight.
#[inline(always)]
fn split_block(x: u64) -> (u64, u64, u64) {
    let a = x / 10_000_000_000_000_000;
    let rest = x - a * 10_000_000_000_000_000;
    let b = rest / 100_000_000;
    (a, b, rest - b * 100_000_000)
}

/// The three decimal digits of `x < 1000` as values, the most significant in the lowest byte.
#[inline(always)]
fn three_digits(x: u64) -> u64 {
    // x / 100 for x < 1024, then the two below.
    let h = (x * 41) >> 12;
    let rest = x - h * 100;
    let t = (rest * 103) >> 10;
    h | (t << 8) | ((rest - t * 10) << 16)
}

/// Writes the decimal digits of the block `x < 10^19` at `dst`, without zeros in front, `0` for
/// zero; returns how many, from 1 to 19. The digits are made in a buffer, and only they are
/// copied.
///
/// # Safety
///
/// `dst` must be valid for writing as many bytes as `x` has digits.
#[inline(always)]
unsafe fn write_top(x: u64, dst: *mut u8) -> usize {
    let (a, b, c) = split_block(x);
    let (a, b, c) = (three_digits(a), eight_digits(b), eight_digits(c));
    // The zeros in front: the zero bytes at the bottom of the first group with a digit that is
    // not zero, the most significant in the lowest byte.
    let zeros = if a != 0 {
        a.trailing_zeros() / 8
    } else if b != 0 {
        3 + b.trailing_zeros() / 8
    } else {
        11 + (c | 1 << 56).trailing_zeros() / 8
    } as usize;
    let mut buffer = [0u8; 24];
    let ascii = 0x3030_3030_3030_3030;
    let p = buffer.as_mut_ptr();
    // SAFETY: the buffer holds the digits at `[5, 24)`, and `dst` has room for the `19 - zeros`
    // copied.
    unsafe {
        write(p.add(4), ((a << 8) as u32 | 0x3030_3030).to_le_bytes());
        write(p.add(8), (b | ascii).to_le_bytes());
        write(p.add(16), (c | ascii).to_le_bytes());
        copy_short(dst, p.add(5 + zeros), 19 - zeros);
    }
    19 - zeros
}

/// Copies the `n` bytes at `src`, from 1 to 24, to `dst`, apart from them: in up to three copies
/// of eight, four or one, which overlap where `n` is not a multiple, and agree there.
///
/// # Safety
///
/// The bytes must be valid for reads at `src` and for writes at `dst`.
#[inline(always)]
unsafe fn copy_short(dst: *mut u8, src: *const u8, n: usize) {
    // SAFETY: every access lies in the `n` bytes.
    unsafe {
        if n >= 8 {
            write::<8>(dst, read(src));
            if n > 16 {
                write::<8>(dst.add(8), read(src.add(8)));
            }
            write::<8>(dst.add(n - 8), read(src.add(n - 8)));
        } else if n >= 4 {
            write::<4>(dst, read(src));
            write::<4>(dst.add(n - 4), read(src.add(n - 4)));
        } else {
            *dst = *src;
            *dst.add(n / 2) = *src.add(n / 2);
            *dst.add(n - 1) = *src.add(n - 1);
        }
    }
}

// Divide and conquer

/// A power `b^(2^i)` of the block for divide and conquer, without its zero bytes at the bottom.
struct Power {
    /// The power divided by `256^zeros`, little-endian, in the shortest form.
    odd: Number,
    /// The zero bytes at the bottom of the power.
    zeros: usize,
    /// Its places: `k 2^i` digits.
    places: usize,
    /// [`Power::odd`] prepared for dividing by it, when it first is.
    divisor: Option<Divisor>,
}

impl Power {
    /// The length of the power in bytes.
    fn bytes(&self) -> usize {
        self.odd.value.len() + self.zeros
    }

    /// The first power, `b`, of `k` places.
    fn first(b: u64, k: usize) -> Power {
        let zeros = (b.trailing_zeros() / 8) as usize;
        Power {
            odd: Number::new(
                (b >> (8 * zeros)).to_le_bytes().to_vec(),
                ByteOrder::LittleEndian,
            ),
            zeros,
            places: k,
            divisor: None,
        }
    }

    /// The square of this power.
    fn square(&self) -> Power {
        let mut odd = &self.odd * &self.odd;
        let more = odd.value.iter().take_while(|&&byte| byte == 0).count();
        odd.value.drain(..more);
        Power {
            odd,
            zeros: 2 * self.zeros + more,
            places: 2 * self.places,
            divisor: None,
        }
    }
}

/// The index of the largest power in `powers`, computed as they are first needed from `b` of `k`
/// places, for which `fits(bytes, places)` holds, or 0. A square has at least twice the bytes of
/// its root but one, and is computed only when that many may fit.
fn largest(
    powers: &mut Vec<Power>,
    b: u64,
    k: usize,
    fits: impl Fn(usize, usize) -> bool,
) -> usize {
    if powers.is_empty() {
        powers.push(Power::first(b, k));
    }
    let mut i = 0;
    loop {
        if i + 1 == powers.len() {
            let p = &powers[i];
            if !fits(2 * p.bytes() - 1, 2 * p.places) {
                return i;
            }
            let next = p.square();
            powers.push(next);
        }
        let p = &powers[i + 1];
        if !fits(p.bytes(), p.places) {
            return i;
        }
        i += 1;
    }
}

/// Writes the digits of the magnitude `x`, little-endian in the shortest form, in `radix` at
/// `dst`, as [`write_limbs`] writes them, and returns how many: by divide and conquer, divided by
/// the largest power `b^(2^i)` that has at most half its limbs and one, the quotient written
/// first, then the remainder in `k 2^i` places.
///
/// The power is `odd 256^zeros`; only the bytes of `x` from `zeros` up are divided by `odd`,
/// and the remainder's are put above the bytes below them.
///
/// # Safety
///
/// `dst` must be valid for writing the digits.
unsafe fn write_split(
    x: Number,
    radix: u32,
    width: Option<usize>,
    dst: *mut u8,
    powers: &mut Vec<Power>,
) -> usize {
    let limbs = x.value.len().div_ceil(8);
    // SAFETY: as the caller guarantees.
    unsafe {
        if limbs < WRITE_SPLIT {
            if limbs == 0 || x.value == [0] {
                return write_limbs([0; PASSES + 1].as_mut_ptr(), 0, radix, width, dst);
            }
            let mut u = [MaybeUninit::<u64>::uninit(); WRITE_SPLIT + PASSES];
            let u = u.as_mut_ptr().cast::<u64>();
            to_limbs(Operand::of(&x), u);
            return write_limbs(u, limbs, radix, width, dst);
        }
        let (b, k) = block(radix);
        let half = 8 * (limbs.div_ceil(2) + 1);
        let i = largest(powers, b, k, |bytes, _| bytes <= half);
        let power = &mut powers[i];
        let (zeros, places) = (power.zeros, power.places);
        let divisor = power
            .divisor
            .get_or_insert_with(|| Divisor::new(&power.odd));
        let (q, r) = if x.value.len() > zeros {
            let high = Number {
                value: x.value[zeros..].to_vec(),
                negative: false,
                byte_order: ByteOrder::LittleEndian,
            };
            let (q, r) = divisor.div_rem(&high);
            let mut low = x.value;
            low.truncate(zeros);
            if r.value != [0] {
                low.extend_from_slice(&r.value);
            }
            (q, trimmed(low))
        } else {
            (Number::new(vec![0], ByteOrder::LittleEndian), x)
        };
        let n = match width {
            // Without a quotient, the remainder is the number, and has no zeros in front.
            None if q.value == [0] => return write_split(r, radix, None, dst, powers),
            None => write_split(q, radix, None, dst, powers),
            Some(w) => write_split(q, radix, Some(w - places), dst, powers),
        };
        n + write_split(r, radix, Some(places), dst.add(n), powers)
    }
}

/// `value` as a little-endian number in the shortest form.
fn trimmed(mut value: Vec<u8>) -> Number {
    while value.len() > 1 && value.last() == Some(&0) {
        value.pop();
    }
    if value.is_empty() {
        value.push(0);
    }
    Number {
        value,
        negative: false,
        byte_order: ByteOrder::LittleEndian,
    }
}

// Radices that are powers of two

/// Writes the digits of the magnitude of the operand `x`, not zero, in the radix `2^bits` at
/// `dst`, without zeros in front, and returns how many: read where they are, from the most
/// significant byte. Hexadecimal is two digits a byte, from a table, binary eight, spread over the
/// bytes of a `u64` by one multiplication; other radices take groups of `bits` bits from the low
/// end, written from the back.
///
/// # Safety
///
/// The operand's vector must be valid for reads, and `dst` for writing the digits.
unsafe fn write_bits(x: Operand, bits: u32, dst: *mut u8) -> usize {
    // SAFETY: as the caller guarantees.
    let bytes = unsafe { significant(x) };
    let l = bytes.len();
    // Byte `i` of the magnitude from the most significant.
    // SAFETY: `i < l`.
    let from_top =
        |i: usize| unsafe { *bytes.get_unchecked(if x.big_endian { i } else { l - 1 - i }) };
    let top = from_top(0);
    // The digits, counted in 64 bits: their number fits, as the caller has room for them.
    let total = 8 * (l as u64 - 1) + u64::from(8 - top.leading_zeros());
    let n = total.div_ceil(u64::from(bits)) as usize;
    // SAFETY: as the caller guarantees, `dst` has room for the `n` digits; the top byte's digits
    // come first, then those of each byte below it.
    unsafe {
        match bits {
            4 => {
                let mut at = dst;
                if top < 16 {
                    *at = DIGITS[usize::from(top)];
                    at = at.add(1);
                } else {
                    write(at, HEX[usize::from(top)]);
                    at = at.add(2);
                }
                for i in 1..l {
                    write(at, HEX[usize::from(from_top(i))]);
                    at = at.add(2);
                }
            }
            1 => {
                let k = 8 - top.leading_zeros() as usize;
                for j in 0..k {
                    *dst.add(j) = b'0' + ((top >> (k - 1 - j)) & 1);
                }
                for i in 1..l {
                    write(dst.add(k + 8 * (i - 1)), binary_digits(from_top(i)));
                }
            }
            _ => {
                // Whole groups of bytes from the bottom, then the digits above them one by one.
                let (mut next, mut j) = match bits {
                    2 => groups::<2, 1, 4>(from_top, l, n, dst),
                    3 => groups::<3, 3, 8>(from_top, l, n, dst),
                    _ => groups::<5, 5, 8>(from_top, l, n, dst),
                };
                let mask = (1u32 << bits) - 1;
                let (mut acc, mut have) = (0u32, 0u32);
                while j > 0 {
                    j -= 1;
                    if have < bits && next > 0 {
                        next -= 1;
                        acc |= u32::from(from_top(next)) << have;
                        have += 8;
                    }
                    *dst.add(j) = DIGITS[(acc & mask) as usize];
                    acc >>= bits;
                    have = have.saturating_sub(bits);
                }
            }
        }
    }
    n
}

/// Writes the digits of the magnitude of `l` bytes, `from_top(i)` its byte `i` from the most
/// significant, of `n` digits in the radix `2^B`, from the bottom in groups of `G` bytes, `D`
/// digits each, the last of the `n` digits at `dst` first; returns the bytes and the digits above
/// the groups, left to write.
///
/// # Safety
///
/// `dst` must be valid for writing `n` bytes.
#[inline(always)]
unsafe fn groups<const B: u32, const G: usize, const D: usize>(
    from_top: impl Fn(usize) -> u8,
    l: usize,
    n: usize,
    dst: *mut u8,
) -> (usize, usize) {
    let (mut bytes, mut digits) = (l, n);
    while bytes >= G && digits >= D {
        let mut v = 0u64;
        for t in 0..G {
            v |= u64::from(from_top(bytes - 1 - t)) << (8 * t);
        }
        for t in 0..D {
            // SAFETY: `digits - 1 - t < n`.
            unsafe {
                *dst.add(digits - 1 - t) =
                    DIGITS[((v >> (B as usize * t)) & ((1 << B) - 1)) as usize]
            };
        }
        bytes -= G;
        digits -= D;
    }
    (bytes, digits)
}

/// The eight binary digits of `byte` as ASCII, the most significant first: bit `7 - i` moved to
/// byte `i` by a multiplication that adds copies of `byte` nine bits apart, which cannot overlap.
#[inline(always)]
fn binary_digits(byte: u8) -> [u8; 8] {
    let spread = (u64::from(byte).wrapping_mul(0x8040_2010_0804_0201) >> 7) & 0x0101_0101_0101_0101;
    (spread | 0x3030_3030_3030_3030).to_le_bytes()
}

/// The two hexadecimal digits of each byte.
const HEX: [[u8; 2]; 256] = {
    let mut t = [[0; 2]; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = [DIGITS[i >> 4], DIGITS[i & 15]];
        i += 1;
    }
    t
};

/// The value of each character as a digit, 36 or more when it is none.
const VALUES: [u8; 256] = {
    let mut t = [255; 256];
    let mut i = 0;
    while i < 36 {
        t[DIGITS[i] as usize] = i as u8;
        t[DIGITS[i].to_ascii_uppercase() as usize] = i as u8;
        i += 1;
    }
    t
};

/// The number of the digits `text` in the radix `2^bits`, the most significant first, in
/// `order`, `negative` if not zero; `None` when a character is not a digit. One pass from the
/// last digit, each byte written in its place in `order`: two digits a byte for hexadecimal, and
/// otherwise the digits' bits gathered in a register; the digits' values are or-ed together, and
/// the radix checked once.
fn read_bits(text: &[u8], bits: u32, negative: bool, order: ByteOrder) -> Option<Number> {
    // Zeros in front count for nothing.
    let start = text.iter().position(|&c| c != b'0').unwrap_or(text.len());
    let digits = &text[start..];
    let Some(&first) = digits.first() else {
        return Some(zero(order));
    };
    let top = VALUES[usize::from(first)];
    if top >> bits != 0 {
        return None;
    }
    let n = digits.len();
    // Counted in 64 bits; the bytes are fewer than the digits, so their number fits.
    let len = (u64::from(bits) * (n as u64 - 1) + u64::from(8 - top.leading_zeros())).div_ceil(8)
        as usize;
    let mut value = Vec::<u8>::with_capacity(len);
    let be = order == ByteOrder::BigEndian;
    let out = value.as_mut_ptr();
    let mut seen = 0u8;
    // SAFETY: the bytes written are `len`, each at its own place below `len`: the significant
    // bits are `len` bytes, and no digit's bits go above them, as the top digit's are counted.
    unsafe {
        let put = |j: usize, byte: u8| *out.add(if be { len - 1 - j } else { j }) = byte;
        let digit = |i: usize| VALUES[usize::from(*digits.get_unchecked(i))];
        if bits == 4 {
            // Hexadecimal: a byte from two digits, `ceil(n / 2)` of them, as the top digit is not
            // zero.
            let mut j = 0;
            let mut i = n;
            while i >= 2 {
                let (high, low) = (digit(i - 2), digit(i - 1));
                seen |= high | low;
                put(j, high << 4 | low);
                j += 1;
                i -= 2;
            }
            if i == 1 {
                put(j, top);
            }
        } else {
            let (mut acc, mut have, mut j) = (0u32, 0u32, 0);
            for i in (0..n).rev() {
                let v = digit(i);
                seen |= v;
                acc |= u32::from(v) << have;
                have += bits;
                if have >= 8 {
                    put(j, acc as u8);
                    j += 1;
                    acc >>= 8;
                    have -= 8;
                }
            }
            if j < len {
                put(j, acc as u8);
            }
        }
        if seen >> bits != 0 {
            return None;
        }
        value.set_len(len);
    }
    Some(Number {
        value,
        negative,
        byte_order: order,
    })
}

/// Zero, `[0]`, in `order`.
fn zero(order: ByteOrder) -> Number {
    Number {
        value: vec![0],
        negative: false,
        byte_order: order,
    }
}

// Reading decimal text

/// Eight ASCII zeros.
const ASCII_ZEROS: u64 = 0x3030_3030_3030_3030;

/// Not zero when a byte of `w` is not an ASCII digit: the high half of each byte must be 3, and
/// stay 3 when 6 is added. A carry from adding 6 to one byte into the next comes only from a byte
/// of 0xFA or more, which fails anyway.
#[inline(always)]
fn not_digits(w: u64) -> u64 {
    let high = 0xF0F0_F0F0_F0F0_F0F0;
    ((w & high) | ((w.wrapping_add(0x0606_0606_0606_0606) & high) >> 4)) ^ 0x3333_3333_3333_3333
}

/// The value of the eight ASCII digits of `w`, the most significant in the lowest byte: pairs in
/// the bytes, `10 x[i] + x[i + 1]`, then the pairs weighted by multiplications whose high halves
/// sum them. Wrapping on bytes that are not digits, whose value does not matter.
#[inline(always)]
fn eight_value(w: u64) -> u64 {
    let x = w.wrapping_sub(ASCII_ZEROS);
    let x = x.wrapping_mul(10).wrapping_add(x >> 8);
    let pairs = 0x0000_00FF_0000_00FF;
    (x & pairs)
        .wrapping_mul(100 + (1_000_000 << 32))
        .wrapping_add(((x >> 16) & pairs).wrapping_mul(1 + (10_000 << 32)))
        >> 32
}

/// The value of the 19 ASCII digits at `p` and whether one is not a digit (not zero then).
///
/// # Safety
///
/// The 19 bytes must be valid for reads.
#[inline(always)]
unsafe fn block_value(p: *const u8) -> (u64, u64) {
    // SAFETY: as the caller guarantees: `[0, 8)`, `[8, 16)` and `[11, 19)`.
    let (a, b, c) = unsafe {
        (
            u64::from_le_bytes(read(p)),
            u64::from_le_bytes(read(p.add(8))),
            u64::from_le_bytes(read(p.add(11))),
        )
    };
    // The last three digits: those of `[11, 19)` with the five that `b` has made zeros.
    let low = (c & 0xFFFF_FF00_0000_0000) | 0x0000_0030_3030_3030;
    let value = eight_value(a)
        .wrapping_mul(100_000_000_000)
        .wrapping_add(eight_value(b).wrapping_mul(1000))
        .wrapping_add(eight_value(low));
    (value, not_digits(a) | not_digits(b) | not_digits(c))
}

/// The value of the `n` ASCII digits at `p`, from 1 to 19, and whether one is not a digit: copied
/// behind zeros to make 19.
///
/// # Safety
///
/// The `n` bytes must be valid for reads.
#[inline(always)]
unsafe fn head_value(p: *const u8, n: usize) -> (u64, u64) {
    let mut buffer = [b'0'; 24];
    // SAFETY: the buffer holds the 19 digits from 5 up, the last `n` copied.
    unsafe {
        copy_short(buffer.as_mut_ptr().add(24 - n), p, n);
        block_value(buffer.as_ptr().add(5))
    }
}

/// The number of the decimal digits `text`, `negative` if not zero, in `order`; `None` when a
/// character is not a digit. Up to 38 digits in registers, up to [`READ_SPLIT`] blocks in limbs.
fn read_decimal(text: &[u8], negative: bool, order: ByteOrder) -> Option<Number> {
    let n = text.len();
    if n <= 38 {
        let p = text.as_ptr();
        let head = n - 19 * ((n - 1) / 19);
        // SAFETY: the reads lie in the text: the head's `head` bytes, then 19 for 20 digits up.
        let (x, bad) = unsafe {
            let (first, bad) = head_value(p, head);
            if n <= 19 {
                (u128::from(first), bad)
            } else {
                let (low, more) = block_value(p.add(head));
                (
                    u128::from(first) * u128::from(TEN_19) + u128::from(low),
                    bad | more,
                )
            }
        };
        return (bad == 0).then(|| small_number(x, negative, order));
    }
    read_long(text, 10, negative, order)
}

/// The number of the digits `text` in `radix`, not a power of two, `negative` if not zero, in
/// `order`; `None` when a character is not a digit. Up to [`READ_SPLIT`] blocks in limbs, on the
/// stack for up to [`STACK_BLOCKS`], and by divide and conquer beyond.
fn read_long(text: &[u8], radix: u32, negative: bool, order: ByteOrder) -> Option<Number> {
    let (b, k) = block(radix);
    let blocks = text.len().div_ceil(k);
    if blocks <= READ_SPLIT {
        let mut stack = [MaybeUninit::<u64>::uninit(); STACK_BLOCKS + 1];
        let mut heap = Vec::new();
        let x = if blocks <= STACK_BLOCKS {
            stack.as_mut_ptr().cast::<u64>()
        } else {
            heap.reserve_exact(blocks + 1);
            heap.as_mut_ptr()
        };
        // SAFETY: the limbs have room for one a block and one more.
        let len = unsafe { read_blocks(text, radix, b, k, x)? };
        return Some(limbs_number(x, len, negative, order));
    }
    let mut powers = Vec::new();
    let x = read_split(text, radix, &mut powers)?;
    Some(in_order(x, negative, order))
}

/// Sets the `len` limbs at `x` to `x b + v`, and returns the limb that carries out of them.
///
/// # Safety
///
/// The limbs must be valid for reads and writes.
#[inline(always)]
unsafe fn mul_add(x: *mut u64, len: usize, b: u64, v: u64) -> u64 {
    let mut carry = v;
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in 0..len {
            let t = u128::from(*x.add(i)) * u128::from(b) + u128::from(carry);
            *x.add(i) = t as u64;
            carry = (t >> 64) as u64;
        }
    }
    carry
}

/// The magnitude `x` as a number with the sign `negative`, in `order`, in one new vector of
/// exactly its length.
fn small_number(x: u128, negative: bool, order: ByteOrder) -> Number {
    let limbs = [x as u64, (x >> 64) as u64];
    let len = usize::from(x != 0) + usize::from(x >> 64 != 0);
    // SAFETY: the limbs hold the `len` limbs.
    limbs_number(limbs.as_ptr().cast_mut(), len, negative, order)
}

/// The `len` limbs at `x`, least significant first, as a number with the sign `negative`, in
/// `order`, in the shortest form, in one new vector of exactly its length.
fn limbs_number(x: *mut u64, len: usize, negative: bool, order: ByteOrder) -> Number {
    // SAFETY: the caller hands over `len` limbs; the top one is not zero.
    let bytes = if len == 0 {
        1
    } else {
        8 * len - unsafe { *x.add(len - 1) }.leading_zeros() as usize / 8
    };
    let p = Product {
        r: if len == 0 { core::ptr::null() } else { x },
        len: bytes,
        negative: negative && len != 0,
    };
    let mut value = Vec::with_capacity(bytes);
    // SAFETY: the vector has room for the bytes, all of which are written before the length is
    // set.
    unsafe {
        p.write(value.as_mut_ptr(), order);
        value.set_len(bytes);
    }
    Number {
        value,
        negative: p.negative,
        byte_order: order,
    }
}

/// `x`, little-endian in the shortest form, as a number with the sign `negative` in `order`.
fn in_order(mut x: Number, negative: bool, order: ByteOrder) -> Number {
    if order == ByteOrder::BigEndian {
        x.value.reverse();
    }
    x.negative = negative && x.value != [0];
    x.byte_order = order;
    x
}

// Blocks of digits

/// Reads the digits `text` in `radix`, not a power of two, in blocks of `k` of `b = radix^k`,
/// the first one shorter when the digits do not fill it, into the limbs `x`, least significant
/// first; returns how many it holds, or `None` when a character is not a digit. Decimal blocks are
/// read eight digits at a time, other ones a digit at a time; they are multiplied in
/// [`READ_PASSES`] at a time, once the number has [`READ_WAVES`] limbs.
///
/// # Safety
///
/// `x` must be valid for writing one limb a block and one more; only the limbs written are read.
unsafe fn read_blocks(text: &[u8], radix: u32, b: u64, k: usize, x: *mut u64) -> Option<usize> {
    let n = text.len();
    let head = n - k * ((n - 1) / k);
    let count = (n - head) / k;
    let p = text.as_ptr();
    let mut bad = 0u64;
    // The value of the `k` digits at `at`, which turns `bad` not zero when one is not a digit.
    let mut value = |at: usize, len: usize| -> u64 {
        if radix == 10 {
            // SAFETY: the `len` bytes at `at` lie in the text, 19 of them for a whole block.
            let (v, more) = unsafe {
                if len == 19 {
                    block_value(p.add(at))
                } else {
                    head_value(p.add(at), len)
                }
            };
            bad |= more;
            v
        } else {
            text[at..at + len].iter().fold(0u64, |acc, &c| {
                let v = VALUES[usize::from(c)];
                bad |= u64::from(u32::from(v) >= radix);
                acc.wrapping_mul(u64::from(radix))
                    .wrapping_add(u64::from(v))
            })
        }
    };
    let first = value(0, head);
    let mut len = usize::from(first != 0);
    let mut j = 0;
    let at = |j: usize| head + k * j;
    // SAFETY: as the caller guarantees: the number has at most as many limbs as the blocks taken,
    // and a group writes at most `READ_PASSES` more limbs than it has, as many as the blocks it
    // takes.
    unsafe {
        *x = first;
        while j < count {
            if len >= READ_WAVES && count - j >= READ_PASSES {
                let v: [u64; READ_PASSES] = core::array::from_fn(|i| value(at(j + i), k));
                len = mul_add_group(x, len, b, v);
                j += READ_PASSES;
            } else {
                let carry = mul_add(x, len, b, value(at(j), k));
                *x.add(len) = carry;
                len += usize::from(carry != 0);
                j += 1;
            }
        }
    }
    (bad == 0).then_some(len)
}

/// Sets the `len` limbs at `x` to `(.. ((x b + v[0]) b + v[1]) ..) b + v[K - 1]`, the steps of
/// [`mul_add`] `K` times, and returns the new length. The steps of one pass wait for each other,
/// two additions on the carry; so the `K` passes run in waves, each a limb behind the one before
/// it, from which it takes its limbs in registers, and the core runs their chains side by side.
/// Every pass takes `len + K - 1` limbs, zeros above the number, so that their carries come out
/// as limbs; the last pass writes its limbs back, after the first one read them.
///
/// # Safety
///
/// The `len + K` limbs at `x` must be valid for writes, and the `len` first ones for reads.
#[inline(always)]
unsafe fn mul_add_group<const K: usize>(x: *mut u64, len: usize, b: u64, v: [u64; K]) -> usize {
    let n = len + K - 1;
    let mut carry = v;
    // The limb that each pass handed the next one in the last wave.
    let mut c = [0u64; K];
    // SAFETY: as the caller guarantees: in wave `t`, the first pass reads limb `t` below `len`,
    // the last one writes limb `t - K + 1`, below `n`, and the last carry goes to limb `n`.
    unsafe {
        // Pass `p` at limb `t - p` in wave `t`.
        let mut wave = |t: usize, from: usize, to: usize| {
            let mut p = to;
            while p > from {
                p -= 1;
                let input = if p > 0 {
                    c[p - 1]
                } else if t < len {
                    *x.add(t)
                } else {
                    0
                };
                let s = u128::from(input) * u128::from(b) + u128::from(carry[p]);
                carry[p] = (s >> 64) as u64;
                if p == K - 1 {
                    *x.add(t - p) = s as u64;
                } else {
                    c[p] = s as u64;
                }
            }
        };
        for t in 0..K - 1 {
            wave(t, 0, t + 1);
        }
        for t in K - 1..n {
            wave(t, 0, K);
        }
        for i in 0..K - 1 {
            wave(n + i, i + 1, K);
        }
        *x.add(n) = carry[K - 1];
        let mut top = n + 1;
        while top > 0 && *x.add(top - 1) == 0 {
            top -= 1;
        }
        top
    }
}

/// The value of the digits `text` in `radix`, not a power of two, as a little-endian number in the
/// shortest form, or `None` when a character is not a digit: halves by divide and conquer from
/// [`READ_SPLIT`] blocks, the upper one times `b^(2^i)` plus the lower one of `k 2^i` digits; below,
/// block by block. `powers` holds the powers `b^(2^i)`, computed as they are first needed.
///
/// With the power `odd 256^zeros`, the upper half is multiplied by `odd` alone, and the lower one
/// added in `zeros` bytes up, but for its bytes below them, which are put below.
fn read_split(text: &[u8], radix: u32, powers: &mut Vec<Power>) -> Option<Number> {
    let (b, k) = block(radix);
    let blocks = text.len().div_ceil(k);
    if blocks <= READ_SPLIT {
        let mut x = Vec::<u64>::with_capacity(blocks + 1);
        // SAFETY: the vector has room for one limb a block and one more.
        let len = unsafe { read_blocks(text, radix, b, k, x.as_mut_ptr())? };
        return Some(limbs_number(
            x.as_mut_ptr(),
            len,
            false,
            ByteOrder::LittleEndian,
        ));
    }
    // The largest `i` whose `k 2^i` places are at most half the digits.
    let half = text.len() / 2;
    let i = largest(powers, b, k, |_, places| places <= half);
    let (places, zeros) = (powers[i].places, powers[i].zeros);
    let (high, low) = text.split_at(text.len() - places);
    let high = read_split(high, radix, powers)?;
    let low = read_split(low, radix, powers)?;
    let product = &high * &powers[i].odd;
    let mut value = low.value;
    let sum = if value.len() > zeros {
        let above = Number {
            value: value[zeros..].to_vec(),
            negative: false,
            byte_order: ByteOrder::LittleEndian,
        };
        &product + &above
    } else {
        product
    };
    value.resize(zeros, 0);
    value.extend_from_slice(&sum.value);
    Some(trimmed(value))
}

impl core::str::FromStr for Number {
    type Err = ParseNumberError;

    /// Reads a number from decimal text, as [`Number::from_str_radix`] does with radix 10, and
    /// builds it little-endian, the byte order of [`Number::default`].
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::Number;
    ///
    /// let n: Number = "-123456789012345678901234567890".parse().unwrap();
    /// assert!(n.is_negative());
    /// assert_eq!(n.to_string(), "-123456789012345678901234567890");
    /// assert!("12a".parse::<Number>().is_err());
    /// ```
    #[inline]
    fn from_str(src: &str) -> Result<Number, ParseNumberError> {
        Number::from_str_radix(src, 10, ByteOrder::LittleEndian)
    }
}

/// An error from [`Number::from_str_radix`]: what was wrong with the text, from
/// [`kind`](Self::kind).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseNumberError {
    kind: NumberErrorKind,
}

/// What [`ParseNumberError`] found wrong with the text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NumberErrorKind {
    /// The text is empty.
    Empty,
    /// A character is not a digit of the radix, or the sign has no digits after it.
    InvalidDigit,
}

impl ParseNumberError {
    /// What was wrong with the text.
    pub fn kind(&self) -> &NumberErrorKind {
        &self.kind
    }
}

impl fmt::Display for ParseNumberError {
    /// Writes what was wrong, in the words of the primitive integers' `ParseIntError`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            NumberErrorKind::Empty => "cannot parse integer from empty string",
            NumberErrorKind::InvalidDigit => "invalid digit found in string",
        }
        .fmt(f)
    }
}

impl std::error::Error for ParseNumberError {}

/// Writes the magnitude of `n` in `radix` through `f.pad_integral`, which adds the sign, the
/// `prefix` with the `#` flag, and the width, fill and alignment asked for. Up to
/// [`STACK_TEXT`] digits are written in a buffer on the stack, longer text in a vector.
fn format(
    n: &Number,
    radix: u32,
    prefix: &str,
    upper: bool,
    f: &mut fmt::Formatter<'_>,
) -> fmt::Result {
    let x = Operand::of(n);
    let bound = digits_bound(x.significant, radix);
    let non_negative = !n.negative || x.significant == 0;
    let mut stack = [MaybeUninit::<u8>::uninit(); STACK_TEXT];
    let mut heap = Vec::new();
    let dst = if bound <= STACK_TEXT {
        stack.as_mut_ptr().cast::<u8>()
    } else {
        heap.reserve_exact(bound);
        heap.as_mut_ptr()
    };
    // SAFETY: the buffer has room for the digits, which are written before they are read; they
    // are ASCII. `n` is borrowed while it is read.
    let digits = unsafe {
        let len = write_magnitude(x, radix, dst);
        let digits = core::slice::from_raw_parts_mut(dst, len);
        if upper {
            digits.make_ascii_uppercase();
        }
        core::str::from_utf8_unchecked(digits)
    };
    f.pad_integral(non_negative, prefix, digits)
}

impl fmt::Display for Number {
    /// Writes the number in decimal: `-` for a negative number, then the digits; the formatter's
    /// flags, width and fill apply as they do to the primitive integers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_i64(-1_234_567_890_123, ByteOrder::LittleEndian);
    /// assert_eq!(n.to_string(), "-1234567890123");
    /// assert_eq!(format!("{:>16}", n), "  -1234567890123");
    /// assert_eq!(format!("{:+}", Number::new_from_u8(7, ByteOrder::BigEndian)), "+7");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format(self, 10, "", false, f)
    }
}

impl fmt::LowerHex for Number {
    /// Writes the number in hexadecimal, lower-case: the sign and the digits of the magnitude, not
    /// a two's complement; `0x` in front with the `#` flag.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_i32(-0xBEEF, ByteOrder::BigEndian);
    /// assert_eq!(format!("{n:x}"), "-beef");
    /// assert_eq!(format!("{n:#x}"), "-0xbeef");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format(self, 16, "0x", false, f)
    }
}

impl fmt::UpperHex for Number {
    /// Writes the number in hexadecimal, upper-case, as [`LowerHex`](fmt::LowerHex) does.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format(self, 16, "0x", true, f)
    }
}

impl fmt::Octal for Number {
    /// Writes the number in octal: the sign and the digits; `0o` in front with the `#` flag.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format(self, 8, "0o", false, f)
    }
}

impl fmt::Binary for Number {
    /// Writes the number in binary: the sign and the digits; `0b` in front with the `#` flag.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format(self, 2, "0b", false, f)
    }
}

//! The arithmetic operators with a primitive integer on either side: `a + 1`, `&a * 10`,
//! `a % 7u64`, `1i32 - &a`, `a -= 5i128`, and so on for `+`, `-`, `*`, `/` and `%` and their
//! assignments, with every primitive integer type.
//!
//! Each computes what the operator computes with the integer made a [`Number`]: the same results,
//! byte orders and panics. With the number on the left, the result is in its byte order, and built
//! in its vector when it is owned and has room, as with two numbers. With the integer on the left,
//! it is in the byte order of the number on the right, the only one there is: `1 - &a` is `-(&a -
//! 1)`, and `10 / &a` is the integer as a number in `a`'s byte order divided by `a`.
//!
//! `+` and `-`, and `*` by an integer of up to 64 bits, have paths of their own, which change the
//! number's bytes in place, in one pass, without reading it into limbs: a sum carries out of the
//! integer's bytes only as far as the number's bytes are all ones, and a product of a number of up
//! to 16 bytes is computed in registers, and of a longer little-endian one by the kernel of the
//! multiplication for one limb (a longer big-endian one multiplies as two numbers do). Measured on an Apple M4 Max with `a` of 1 KiB, `a += 1` took 9 cycles
//! against 29 for `a += &one` with `one` a number, and `a *= 10` 135 against 260 for the product of
//! two numbers before it took the same path (GMP's `mpz_add_ui` and `mpz_mul_ui`: 13 and 122). The
//! others make the integer a number in a vector of 16 bytes that each thread keeps for it, so that
//! they allocate nothing for it; a division costs the same with it as with a number.

use super::add::{read, read_small, write, write_small};
use super::cmp::parts;
use super::mul::mul_1_bytes;
use super::{ByteOrder, Number};
use core::cell::Cell;
use core::mem;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

thread_local! {
    /// The vector that the integer operand of an operator is made a number in, kept between
    /// operators so that it is allocated once in each thread.
    static SCRATCH: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

/// Runs `f` on the integer of sign `negative` and magnitude `m` as a number in `order`, in the
/// shortest form, built in the thread's vector, which it returns to the thread afterwards.
///
/// An operator that panics, dividing by zero, drops the vector; the next takes a new one.
#[inline(always)]
fn with_number<R>(negative: bool, m: u128, order: ByteOrder, f: impl FnOnce(&Number) -> R) -> R {
    let mut value = SCRATCH.take();
    value.clear();
    let len = 16 - (m | 1).leading_zeros() as usize / 8;
    match order {
        ByteOrder::BigEndian => value.extend_from_slice(&m.to_be_bytes()[16 - len..]),
        ByteOrder::LittleEndian => value.extend_from_slice(&m.to_le_bytes()[..len]),
    }
    let n = Number {
        value,
        negative: negative && m != 0,
        byte_order: order,
    };
    let result = f(&n);
    SCRATCH.set(n.value);
    result
}

/// A copy of the significant bytes of `a`, its sign and byte order, in a new vector with room for
/// `spare` bytes more: what the operators with a borrowed number change in place.
#[inline(always)]
fn significant_copy(a: &Number, spare: usize) -> Number {
    let s = a.significant();
    let bytes = if a.byte_order == ByteOrder::BigEndian {
        &a.value[a.value.len() - s..]
    } else {
        &a.value[..s]
    };
    let mut value = Vec::with_capacity(s + spare);
    value.extend_from_slice(bytes);
    Number {
        value,
        negative: a.negative && s != 0,
        byte_order: a.byte_order,
    }
}

/// Makes `a` the sum of `a` and the integer of sign `negative` and magnitude `m`, in the shortest
/// form, in its vector when it has room.
///
/// A number of up to 16 bytes is added in registers, as the addition of two numbers does. A longer
/// one is larger than the integer: the integer's 16 bytes are added to its 16 lowest, or subtracted
/// from them, and the carry or borrow taken up through its bytes while they are all ones or all
/// zeros; a carry out of the top byte makes it one byte longer, and a borrow can make its top bytes
/// zero.
#[inline(never)]
fn add_in_place(a: &mut Number, negative: bool, m: u128) {
    let s = a.significant();
    let be = a.byte_order == ByteOrder::BigEndian;
    if s <= 16 {
        let start = if be { a.value.len() - s } else { 0 };
        // SAFETY: the significant bytes are at most 16.
        let x = unsafe { read_small(&a.value[start..start + s], a.byte_order) };
        let negative_a = a.negative && s != 0;
        let (magnitude, carry, sign) = if negative_a == negative {
            let (sum, carry) = x.overflowing_add(m);
            (sum, carry, negative_a)
        } else if x >= m {
            (x - m, false, negative_a)
        } else {
            (m - x, false, negative)
        };
        let len = if carry {
            17
        } else {
            16 - (magnitude | 1).leading_zeros() as usize / 8
        };
        a.value.clear();
        a.value.reserve(len);
        // SAFETY: the vector has room for the `len` bytes, from 1 to 17, all of which are written
        // before the length is set.
        unsafe {
            write_small(
                a.value.as_mut_ptr(),
                a.byte_order,
                magnitude as u64,
                (magnitude >> 64) as u64,
                len,
            );
            a.value.set_len(len);
        }
        a.negative = sign && (magnitude != 0 || carry);
        return;
    }
    if s != a.value.len() {
        super::sign::trim(a);
    }
    let subtract = a.negative != negative;
    let v = &mut a.value;
    let p = v.as_mut_ptr();
    // SAFETY: the magnitude has `s > 16` bytes: its lowest 16 are the last big-endian and the first
    // little-endian, and byte `i < s` from the least significant one is at `s - 1 - i` big-endian
    // and at `i` little-endian.
    let (top_changed, carried) = unsafe {
        let low = if be { p.add(s - 16) } else { p };
        let x = if be {
            u128::from_be_bytes(read(low))
        } else {
            u128::from_le_bytes(read(low))
        };
        let (y, mut c) = if subtract {
            x.overflowing_sub(m)
        } else {
            x.overflowing_add(m)
        };
        write(low, if be { y.to_be_bytes() } else { y.to_le_bytes() });
        let mut i = 16;
        while c && i < s {
            let b = if be { p.add(s - 1 - i) } else { p.add(i) };
            (*b, c) = if subtract {
                (*b).overflowing_sub(1)
            } else {
                (*b).overflowing_add(1)
            };
            i += 1;
        }
        (i >= s - 1, c)
    };
    if carried {
        // All ones up to the top: one byte more, 1.
        if be {
            v.insert(0, 1);
        } else {
            v.push(1);
        }
    } else if subtract && top_changed {
        super::sign::trim(a);
    }
}

/// The product of the `s` significant bytes of a magnitude at `x` by `m` in registers, for `s` up
/// to 16, in `order`: its three limbs and its length.
///
/// # Safety
///
/// The `s` bytes at `x` must be valid for reads.
#[inline(always)]
unsafe fn small_product(x: &[u8], order: ByteOrder, m: u64) -> ([u64; 3], usize) {
    // SAFETY: the bytes are at most 16, as the caller guarantees.
    let x = unsafe { read_small(x, order) };
    let low = u128::from(x as u64) * u128::from(m);
    let high = u128::from((x >> 64) as u64) * u128::from(m) + (low >> 64);
    let limbs = [low as u64, high as u64, (high >> 64) as u64];
    let len = if limbs[2] != 0 {
        24 - limbs[2].leading_zeros() as usize / 8
    } else {
        16 - ((u128::from(limbs[1]) << 64) | u128::from(limbs[0]) | 1).leading_zeros() as usize / 8
    };
    (limbs, len)
}

/// Writes the product of [`small_product`], of `len` bytes, at `p` in `order`.
///
/// # Safety
///
/// `p` must be valid for writing `len` bytes, from 1 to 24.
#[inline(always)]
unsafe fn write_small_product(p: *mut u8, order: ByteOrder, [l0, l1, l2]: [u64; 3], len: usize) {
    // SAFETY: as the caller guarantees: up to 16 bytes as the addition writes them; above, the two
    // low limbs and the `len - 16` bytes of the top one, from 1 to 8.
    unsafe {
        if len <= 16 {
            write_small(p, order, l0, l1, len);
        } else if order == ByteOrder::BigEndian {
            write(p.add(len - 8), l0.to_be_bytes());
            write(p.add(len - 16), l1.to_be_bytes());
            write_small(p, order, l2, 0, len - 16);
        } else {
            write(p, l0.to_le_bytes());
            write(p.add(8), l1.to_le_bytes());
            write_small(p.add(16), order, l2, 0, len - 16);
        }
    }
}

/// The significant bytes of `a`.
#[inline(always)]
fn significant_bytes(a: &Number, s: usize) -> &[u8] {
    if a.byte_order == ByteOrder::BigEndian {
        &a.value[a.value.len() - s..]
    } else {
        &a.value[..s]
    }
}

/// The whole limbs of the little-endian magnitude `x` times `m`, written at `dst`, which may be
/// `x`, by the kernel of the multiplication for one limb; returns the carry out of them.
///
/// # Safety
///
/// `x` must hold at least 8 bytes; `dst` must be `x` or apart from it, and valid for writing the
/// `x.len() / 8 * 8` bytes of the limbs.
#[inline(always)]
unsafe fn limbs_le(dst: *mut u8, x: *const u8, s: usize, m: u64) -> u64 {
    // SAFETY: as the caller guarantees; the kernel reads limb `i` before it writes it.
    unsafe { mul_1_bytes(dst, x, s / 8, m) }
}

/// The bytes of the little-endian magnitude `x` above its whole limbs times `m`, with the `carry`
/// out of the limbs: the product's top bytes and their count, at least `x.len() % 8` and at most
/// `x.len() % 8 + 8`; none when the magnitude is whole limbs and the carry zero, and the top limb
/// below them is then not zero, as the product is at least `x`.
#[inline(always)]
fn top_le(x: &[u8], m: u64, carry: u64) -> (u128, usize) {
    let top = x[x.len() / 8 * 8..]
        .iter()
        .rev()
        .fold(0u64, |t, &b| (t << 8) | u64::from(b));
    let top = u128::from(top) * u128::from(m) + u128::from(carry);
    (top, (128 - top.leading_zeros() as usize).div_ceil(8))
}

/// Makes `a` the product of `a` and the integer of sign `negative` and magnitude `m`, in the
/// shortest form, as the multiplication of two numbers places its product: in the vector of `a`
/// when it has room for it, else in `donor` when that has room, which then gets the vector of `a`,
/// and otherwise in one new vector. Returns `false`, `a` unchanged, for a big-endian number of
/// more than 16 bytes, which the general path of the multiplication multiplies faster.
///
/// A number of up to 16 bytes is multiplied in registers. A longer little-endian one is multiplied
/// in place by the kernel of the multiplication for one limb, its bytes read as limbs as they are,
/// in one pass. Big-endian, its limbs would move up by the bytes the product gains, a second pass:
/// measured on an Apple M4 Max as `a *= &b` with `b` of one byte, 1 KiB took 370 cycles so against
/// 285 by the general path.
#[inline(never)]
fn mul_in_place(a: &mut Number, negative: bool, m: u64, donor: Option<&mut Vec<u8>>) -> bool {
    let s = a.significant();
    let order = a.byte_order;
    let sign = a.negative != negative;
    if s == 0 || m == 0 {
        place(a, donor, 1);
        a.value.clear();
        a.value.push(0);
        a.negative = false;
        return true;
    }
    if s <= 16 {
        // SAFETY: the significant bytes are at most 16.
        let (limbs, len) = unsafe { small_product(significant_bytes(a, s), order, m) };
        place(a, donor, len);
        a.value.clear();
        // SAFETY: the vector has room for the `len` bytes, all of which are written before the
        // length is set.
        unsafe {
            write_small_product(a.value.as_mut_ptr(), order, limbs, len);
            a.value.set_len(len);
        }
        a.negative = sign;
        return true;
    }
    if order == ByteOrder::BigEndian {
        return false;
    }
    a.value.truncate(s);
    let p = a.value.as_mut_ptr();
    // SAFETY: the limbs are written over themselves, inside the `s` bytes of the magnitude, the
    // bytes above them left as they are.
    let carry = unsafe { limbs_le(p, p, s, m) };
    let (top, t) = top_le(&a.value, m, carry);
    let full = s / 8 * 8;
    place(a, donor, full + t);
    a.value.truncate(full);
    a.value.extend_from_slice(&top.to_le_bytes()[..t]);
    a.negative = sign;
    true
}

/// The product of `a` and the integer of sign `negative` and magnitude `m` in a new vector, or in
/// `donor` when that has room for it; `None` for a big-endian number of more than 16 bytes, as
/// [`mul_in_place`] says.
#[inline(never)]
fn mul_into(a: &Number, negative: bool, m: u64, donor: Option<&mut Vec<u8>>) -> Option<Number> {
    let s = a.significant();
    let order = a.byte_order;
    if s == 0 || m == 0 || s <= 16 {
        // Zero, one byte, or a product in registers, in `donor`'s vector when it has room.
        let (limbs, len) = if s == 0 || m == 0 {
            ([0; 3], 1)
        } else {
            // SAFETY: the significant bytes are at most 16.
            unsafe { small_product(significant_bytes(a, s), order, m) }
        };
        let mut value = match donor {
            Some(d) if d.capacity() >= len => mem::take(d),
            _ => Vec::with_capacity(len),
        };
        value.clear();
        // SAFETY: the vector has room for the `len` bytes, all of which are written before the
        // length is set.
        unsafe {
            write_small_product(value.as_mut_ptr(), order, limbs, len);
            value.set_len(len);
        }
        return Some(Number {
            value,
            negative: (a.negative && s != 0) != negative && len > 0 && limbs != [0; 3],
            byte_order: order,
        });
    }
    if order == ByteOrder::BigEndian {
        return None;
    }
    let x = significant_bytes(a, s);
    // The product's whole limbs go to `donor`'s vector when it has room for them, and stay there
    // when it has room for the product too; otherwise to a new vector with room for the longest
    // product, `s + 8 - (zero bytes at the top of `m`)` bytes.
    let most = s + 8 - m.leading_zeros() as usize / 8;
    let mut value = match donor {
        Some(d) if d.capacity() >= s => mem::take(d),
        _ => Vec::with_capacity(most),
    };
    value.clear();
    let full = s / 8 * 8;
    // SAFETY: the vector has room for the `full` bytes of the limbs, apart from `a`'s; they are
    // written before the length is set.
    unsafe {
        let carry = limbs_le(value.as_mut_ptr(), x.as_ptr(), s, m);
        value.set_len(full);
        let (top, t) = top_le(x, m, carry);
        if value.capacity() < full + t {
            // The donor's vector had room for the limbs only.
            let mut v = Vec::with_capacity(most);
            v.extend_from_slice(&value);
            value = v;
        }
        value.extend_from_slice(&top.to_le_bytes()[..t]);
    }
    Some(Number {
        value,
        negative: (a.negative && s != 0) != negative,
        byte_order: order,
    })
}

/// Gives `a` a vector with room for `len` bytes, its own bytes kept, as the multiplication of two
/// numbers places its product: its own vector when it has room; else `donor`'s when that has room,
/// which then gets the vector of `a`; else its own, grown into one new vector.
#[inline(always)]
fn place(a: &mut Number, donor: Option<&mut Vec<u8>>, len: usize) {
    if a.value.capacity() >= len {
        return;
    }
    match donor {
        Some(d) if d.capacity() >= len => {
            d.clear();
            d.extend_from_slice(&a.value);
            mem::swap(&mut a.value, d);
        }
        _ => a.value.reserve_exact(len - a.value.len()),
    }
}

/// `a *= m` for the multiplication of two numbers, one of which has one limb, `m` with the sign
/// `negative`, as [`mul_in_place`] does it; `false`, `a` unchanged, when the general path is to.
#[inline(always)]
pub(super) fn times_limb(
    a: &mut Number,
    negative: bool,
    m: u64,
    donor: Option<&mut Vec<u8>>,
) -> bool {
    mul_in_place(a, negative, m, donor)
}

/// `&a * m` for the multiplication of two numbers, one of which has one limb, `m` with the sign
/// `negative`, as [`mul_into`] does it; `None` when the general path is to.
#[inline(always)]
pub(super) fn copy_times_limb(
    a: &Number,
    negative: bool,
    m: u64,
    donor: Option<&mut Vec<u8>>,
) -> Option<Number> {
    mul_into(a, negative, m, donor)
}

// Each operator with the integer of sign `negative` and magnitude `m` on the right: in place on
// `a`, and on a borrowed `a`, into a new number.

/// `a += m`.
#[inline(always)]
fn add_assign(a: &mut Number, negative: bool, m: u128) {
    add_in_place(a, negative, m);
}

/// `&a + m`, in a copy of `a` with room for the sum.
#[inline(always)]
fn add_ref(a: &Number, negative: bool, m: u128) -> Number {
    let mut sum = significant_copy(a, 17);
    add_in_place(&mut sum, negative, m);
    sum
}

/// `a -= m`, as `a += -m`.
#[inline(always)]
fn sub_assign(a: &mut Number, negative: bool, m: u128) {
    add_in_place(a, !negative, m);
}

/// `&a - m`, as `&a + -m`.
#[inline(always)]
fn sub_ref(a: &Number, negative: bool, m: u128) -> Number {
    add_ref(a, !negative, m)
}

/// `a *= m`: in place for `m` of up to 64 bits when [`mul_in_place`] can, by the product of two
/// numbers otherwise.
#[inline(always)]
fn mul_assign(a: &mut Number, negative: bool, m: u128) {
    if m >> 64 != 0 || !mul_in_place(a, negative, m as u64, None) {
        with_number(negative, m, ByteOrder::LittleEndian, |b| *a *= b);
    }
}

/// `&a * m`: into a new vector for `m` of up to 64 bits when [`mul_into`] can, by the product of
/// two numbers otherwise.
#[inline(always)]
fn mul_ref(a: &Number, negative: bool, m: u128) -> Number {
    if m >> 64 == 0
        && let Some(product) = mul_into(a, negative, m as u64, None)
    {
        return product;
    }
    with_number(negative, m, ByteOrder::LittleEndian, |b| a * b)
}

/// `a /= m`, by the division of two numbers.
#[inline(always)]
#[track_caller]
fn div_assign(a: &mut Number, negative: bool, m: u128) {
    with_number(negative, m, ByteOrder::LittleEndian, |b| *a /= b);
}

/// `&a / m`, by the division of two numbers.
#[inline(always)]
#[track_caller]
fn div_ref(a: &Number, negative: bool, m: u128) -> Number {
    with_number(negative, m, ByteOrder::LittleEndian, |b| a / b)
}

/// `a %= m`, by the division of two numbers, whose remainder alone by one limb takes eight limbs
/// a step.
#[inline(always)]
#[track_caller]
fn rem_assign(a: &mut Number, negative: bool, m: u128) {
    with_number(negative, m, ByteOrder::LittleEndian, |b| *a %= b);
}

/// `&a % m`, by the division of two numbers.
#[inline(always)]
#[track_caller]
fn rem_ref(a: &Number, negative: bool, m: u128) -> Number {
    with_number(negative, m, ByteOrder::LittleEndian, |b| a % b)
}

/// The operators of the traits `$op`/`$assign` (methods `$f`/`$f_assign`) between numbers and
/// the primitive integer types, with the integer on the right, by the functions `$assign_f`
/// and `$ref_f`.
macro_rules! right {
    ($op:ident $f:ident $assign:ident $f_assign:ident $sym:literal $assign_f:ident $ref_f:ident: $($t:ident),*) => {$(
        impl $op<$t> for &Number {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, as with `rhs` a number.")]
            #[inline]
            #[track_caller]
            fn $f(self, rhs: $t) -> Number {
                let (negative, m) = parts!(rhs, $t);
                $ref_f(self, negative, m)
            }
        }

        impl $op<$t> for Number {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, as with `rhs` a number, in the vector of `self` when it has room.")]
            #[inline]
            #[track_caller]
            fn $f(mut self, rhs: $t) -> Number {
                let (negative, m) = parts!(rhs, $t);
                $assign_f(&mut self, negative, m);
                self
            }
        }

        impl $assign<$t> for Number {
            #[doc = concat!("Makes `self` `self ", $sym, " rhs`, as with `rhs` a number, in the vector of `self` when it has room.")]
            #[inline]
            #[track_caller]
            fn $f_assign(&mut self, rhs: $t) {
                let (negative, m) = parts!(rhs, $t);
                $assign_f(self, negative, m);
            }
        }
    )*};
}

/// The commutative operators `+` and `*` with the integer on the left: the same with the operands
/// the other way round.
macro_rules! left_commutative {
    ($op:ident $f:ident $sym:literal: $($t:ident),*) => {$(
        impl $op<Number> for $t {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, as `rhs ", $sym, " self`, in the vector of `rhs` when it has room.")]
            #[inline]
            fn $f(self, rhs: Number) -> Number {
                rhs.$f(self)
            }
        }

        impl $op<&Number> for $t {
            type Output = Number;

            #[doc = concat!("Returns `self ", $sym, " rhs`, as `rhs ", $sym, " self`.")]
            #[inline]
            fn $f(self, rhs: &Number) -> Number {
                rhs.$f(self)
            }
        }
    )*};
}

/// `-`, `/` and `%` with the integer on the left: `self - rhs` is `-(rhs - self)`, and the others
/// the integer as a number in the byte order of `rhs` divided by `rhs`.
macro_rules! left {
    ($($t:ident),*) => {$(
        impl Sub<Number> for $t {
            type Output = Number;

            /// Returns `self - rhs`, as `-(rhs - self)`, in the vector of `rhs` when it has room.
            #[inline]
            fn sub(self, rhs: Number) -> Number {
                -(rhs - self)
            }
        }

        impl Sub<&Number> for $t {
            type Output = Number;

            /// Returns `self - rhs`, as `-(rhs - self)`.
            #[inline]
            fn sub(self, rhs: &Number) -> Number {
                -(rhs - self)
            }
        }

        impl Div<Number> for $t {
            type Output = Number;

            /// Returns `self / rhs`, as with `self` a number in the byte order of `rhs`, in the
            /// vector of `rhs` when it has room.
            #[inline]
            #[track_caller]
            fn div(self, rhs: Number) -> Number {
                let (negative, m) = parts!(self, $t);
                let order = rhs.byte_order;
                with_number(negative, m, order, |a| a / rhs)
            }
        }

        impl Div<&Number> for $t {
            type Output = Number;

            /// Returns `self / rhs`, as with `self` a number in the byte order of `rhs`.
            #[inline]
            #[track_caller]
            fn div(self, rhs: &Number) -> Number {
                let (negative, m) = parts!(self, $t);
                with_number(negative, m, rhs.byte_order, |a| a / rhs)
            }
        }

        impl Rem<Number> for $t {
            type Output = Number;

            /// Returns `self % rhs`, as with `self` a number in the byte order of `rhs`, in the
            /// vector of `rhs` when it has room.
            #[inline]
            #[track_caller]
            fn rem(self, rhs: Number) -> Number {
                let (negative, m) = parts!(self, $t);
                let order = rhs.byte_order;
                with_number(negative, m, order, |a| a % rhs)
            }
        }

        impl Rem<&Number> for $t {
            type Output = Number;

            /// Returns `self % rhs`, as with `self` a number in the byte order of `rhs`.
            #[inline]
            #[track_caller]
            fn rem(self, rhs: &Number) -> Number {
                let (negative, m) = parts!(self, $t);
                with_number(negative, m, rhs.byte_order, |a| a % rhs)
            }
        }
    )*};
}

/// Every operator with every primitive integer type.
macro_rules! all {
    ($($t:ident),*) => {
        right!(Add add AddAssign add_assign "+" add_assign add_ref: $($t),*);
        right!(Sub sub SubAssign sub_assign "-" sub_assign sub_ref: $($t),*);
        right!(Mul mul MulAssign mul_assign "*" mul_assign mul_ref: $($t),*);
        right!(Div div DivAssign div_assign "/" div_assign div_ref: $($t),*);
        right!(Rem rem RemAssign rem_assign "%" rem_assign rem_ref: $($t),*);
        left_commutative!(Add add "+": $($t),*);
        left_commutative!(Mul mul "*": $($t),*);
        left!($($t),*);
    };
}

all!(
    u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
);

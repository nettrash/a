//! Addition of [`Number`]s: `+` and `+=`, with either operand owned or borrowed.
//!
//! A number is a sign and a magnitude, so a sum is one of two operations on the magnitudes.
//! When the signs agree, or an operand is zero, the magnitudes add up and the sum has the sign
//! of the nonzero operands. When they differ, the smaller magnitude is subtracted from the
//! larger and the sum has the sign of the larger. A zero magnitude is zero whatever `negative`
//! says, and zero bytes above the most significant one count for nothing. Subtraction (in
//! `sub`) is the sum with the sign of its right operand turned over, and takes the same paths.
//!
//! Operands whose vectors both hold at most 16 bytes take a short path in registers. Their
//! magnitudes are read into `u128`s, with loads that stay inside the bytes of each vector, and
//! both their sum and their difference are computed: the one the signs ask for, the sign, and
//! the length in the shortest form (17 bytes when the sum carries out of the sixteenth) are
//! then chosen without a branch. The sum is stored in the byte order of the left operand, in
//! the vector chosen as below, with no reversal and no other pass over its bytes. Longer
//! operands, and a new vector for a short sum, are handled out of line, so that the short path
//! saves no registers.
//!
//! For longer operands, everything about the sum but its bytes is planned first, from the
//! operands as they are: the operation, which magnitude is `x`, the one the other (`y`) is
//! added to or subtracted from, the sign, and the exact length of the sum in its shortest form.
//! A sum of magnitudes of `lx >= ly` significant bytes is `lx` bytes long, or one more when it
//! carries out of the top, and whether it does is found from the top down: the first byte
//! position at which the bytes add up to anything but `0xFF` decides. A difference reaches up
//! to the highest position at which the magnitudes differ, unless that position differs by
//! exactly 1 and a borrow from below cancels it. It then ends at the bottom of the run of
//! positions under it at which `x` has `0x00` and `y` has `0xFF`, which pass the borrow on, or
//! a byte lower when a borrow comes into that run too, as in `0x01_00_00 - 0xFF_FF = 1`. The
//! planning of a difference is out of line, so that additions keep their registers. The scans
//! take eight positions at a time, and for random operands they stop in the first eight; only
//! long runs of carries or borrows, or of zero bytes above the most significant one, make them
//! read further, each byte once.
//!
//! Only then is a vector chosen for the sum, as the operators promise: the vector of an
//! operand taken by value, or of the target of `+=`, when it has room for the sum (the left
//! operand's first), and otherwise one new vector of the sum's length. In it the sum is
//! computed in one pass from the least significant byte up, and only its bytes: there are no
//! zero bytes to trim afterwards and no temporary vectors, and reusing a vector allocates
//! nothing. The sum is in the byte order of the left operand. A right operand in the other
//! order is brought into it without allocating: an owned one is reversed in place; a borrowed
//! one is copied into the new vector of the sum, or, when the sum is built in the left
//! operand's vector, the left operand is reversed into the right operand's order and the sum
//! reversed back.
//!
//! The magnitudes are added and subtracted in limbs of eight bytes by kernels chosen at
//! compile time: inline assembly on `x86_64` (one carry chain through the carry flag) and on
//! 64-bit little-endian `aarch64` (one chain below 32 limbs little-endian and 64 big-endian,
//! several interleaved chains from there), and portable Rust elsewhere and under Miri, which
//! cannot run assembly. A big-endian magnitude is handed to them by its least significant end,
//! the address one past its last byte, and its limbs go towards lower addresses, so the limbs
//! of either byte order are read where they are. A big-endian sum that grows in the vector of
//! an operand ends further on in it than the operand did, as the vector grows at the end where
//! the least significant byte belongs: the kernels allow an output that lies above an input
//! like this, as they read each limb before they write over it. The bytes of the shorter
//! operand above its last whole limb follow as one more limb, read from below their start and
//! shifted, when there are at least three of them and a whole limb under them, and one at a
//! time otherwise. Above the shorter operand, a carry or a borrow goes on through the longer
//! one eight bytes at a time while it lasts, and the rest of it is copied (inline below 16
//! bytes), or left where it is when the sum is written over it at the same place. The carries
//! that are coin tosses for random operands, out of the shorter operand and out of the top of
//! the sum, are taken without a branch, which would mispredict half of them.
//!
//! Which magnitude of a difference is the larger is a coin toss for random operands of the same
//! length too, and `x` and `y` are chosen without a branch as well. But then the kernel waits for
//! the plan to read and compare the top limbs of both operands before it can load anything,
//! which costs a long difference 12 to 23 cycles once the operands are out of the first-level
//! cache. So a difference built in an operand's vector from 256 bytes little-endian and 512
//! big-endian, the lengths from which the kernels run several chains, takes them by branches
//! instead: a predicted one lets the kernel start at once, and a mispredicted one costs about
//! what the wait did.

use super::{ByteOrder, Number};
use core::hint::select_unpredictable;
use core::ops::{Add, AddAssign};
use core::{mem, ptr};

// The kernels behind the computation, one set per target under the name `kernels`.
//
// The `aarch64` assembly is for the 64-bit little-endian targets only: the ILP32 ones
// (`arm64_32-apple-watchos`, `aarch64-unknown-linux-gnu_ilp32`) would pass it 32-bit pointers
// and sizes in registers whose upper halves are undefined. It is also for the operating
// systems that let user code load and store at any address only: the assembly loads and
// stores limbs wherever the bytes of a magnitude lie, and unlike the accesses of Rust code,
// which LLVM splits on a target that requires alignment, those of assembly reach the memory
// as they are written. The `aarch64` targets that require it (`+strict-align` in their target
// specifications, for which rustc has no `cfg`) are bare-metal and embedded ones:
// `aarch64-unknown-none` and `-none-softfloat`, `aarch64v8r-unknown-none`, `-hermit`, `-nuttx`,
// `-hyperlight`, `-qtee` and `-teeos`. So the gate names the systems known to allow unaligned
// accesses rather than those that do not: a new target gets the portable kernels until it is
// added here.
#[cfg(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    any(
        target_vendor = "apple",
        target_os = "linux",
        target_os = "android",
        target_os = "windows",
        target_os = "freebsd",
        target_os = "netbsd",
        target_os = "openbsd",
        target_os = "fuchsia"
    ),
    not(miri)
))]
#[path = "add/aarch64.rs"]
mod kernels;
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[path = "add/x86_64.rs"]
mod kernels;
#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_endian = "little",
        target_pointer_width = "64",
        any(
            target_vendor = "apple",
            target_os = "linux",
            target_os = "android",
            target_os = "windows",
            target_os = "freebsd",
            target_os = "netbsd",
            target_os = "openbsd",
            target_os = "fuchsia"
        ),
        not(miri)
    ),
    all(target_arch = "x86_64", not(miri))
)))]
#[path = "add/portable.rs"]
mod kernels;

use kernels::{add_n_be, add_n_le, sub_n_be, sub_n_le};

impl Add<&Number> for &Number {
    type Output = Number;

    /// Returns `self + rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The sum is built in one new vector of exactly its length (one byte for zero), the only
    /// allocation; both operands are left as they were. The operands may have any signs, byte
    /// orders and lengths, and need not be in the shortest form.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x01_00 + -0x01_01 = -1, in the byte order of the left operand.
    /// let a = Number::new_from_i32(0x0100, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(-0x0101, ByteOrder::LittleEndian);
    /// let sum = &a + &b;
    /// assert_eq!(sum.value, [1]);
    /// assert!(sum.is_negative());
    /// assert!(sum.byte_order() == ByteOrder::BigEndian);
    ///
    /// // A number plus its negation is zero, one zero byte and not negative.
    /// let minus_a = Number::new_from_i32(-0x0100, ByteOrder::LittleEndian);
    /// let zero = &a + &minus_a;
    /// assert_eq!(zero.value, [0]);
    /// assert!(!zero.is_negative());
    /// ```
    fn add(self, rhs: &Number) -> Number {
        if small(self, rhs) {
            return small_number_of(self, rhs, rhs.negative);
        }
        ref_plus_ref(self, rhs)
    }
}

impl Add<&Number> for Number {
    type Output = Number;

    /// Returns `self + rhs`, in the byte order of `self`, in the shortest form, as `self += rhs`
    /// does.
    ///
    /// The sum is built in the vector of `self` when its capacity has room for the sum, which
    /// allocates nothing, and otherwise in one new vector of exactly its length. `rhs` is left
    /// as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut value = Vec::with_capacity(8);
    /// value.extend([0xFF, 0xFF]);
    /// let a = Number::new(value, ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    /// let one = Number::new_from_u8(1, ByteOrder::BigEndian);
    ///
    /// // 0xFF_FF + 1 = 0x01_00_00, in the vector of `a`.
    /// let sum = a + &one;
    /// assert_eq!(sum.value, [1, 0, 0]);
    /// assert_eq!(sum.value.as_ptr(), pointer);
    /// ```
    fn add(mut self, rhs: &Number) -> Number {
        self += rhs;
        self
    }
}

impl Add<Number> for &Number {
    type Output = Number;

    /// Returns `self + rhs`, in the byte order of `self`, in the shortest form.
    ///
    /// The sum is built in the vector of `rhs` when its capacity has room for the sum, which
    /// allocates nothing, also when `rhs` is in the other byte order; otherwise it gets one new
    /// vector of exactly its length. `self` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_u32(0x01_02_03, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(3);
    /// value.extend([0x10, 0x20]);
    /// let b = Number::new(value, ByteOrder::LittleEndian);
    /// let pointer = b.value.as_ptr();
    ///
    /// // 0x01_02_03 + 0x20_10 = 0x01_22_13, big-endian, in the vector of `b`.
    /// let sum = &a + b;
    /// assert_eq!(sum.value, [0x01, 0x22, 0x13]);
    /// assert_eq!(sum.value.as_ptr(), pointer);
    /// ```
    fn add(self, mut rhs: Number) -> Number {
        if small(self, &rhs) {
            let sum = small_sum(self, &rhs, rhs.negative);
            let order = self.byte_order;
            // SAFETY: a small sum is from 1 to 17 bytes long, and the vector it is written in
            // has room for it.
            unsafe {
                if rhs.value.capacity() >= sum.len {
                    write_small_in(&mut rhs.value, order, sum.lo, sum.hi, sum.len);
                } else {
                    replace_small(&mut rhs.value, order, sum.lo, sum.hi, sum.len);
                }
            }
            return Number {
                value: rhs.value,
                negative: sum.negative,
                byte_order: order,
            };
        }
        // `rhs` is dropped on return, with the vector it holds then, if any.
        ref_plus_value(self, &mut rhs)
    }
}

impl Add<Number> for Number {
    type Output = Number;

    /// Returns `self + rhs`, in the byte order of `self`, in the shortest form, as `self += rhs`
    /// does.
    ///
    /// The sum is built in the vector of `self` when its capacity has room for the sum, or else
    /// in the vector of `rhs` when that has room, also when `rhs` is in the other byte order,
    /// either of which allocates nothing; otherwise it gets one new vector of exactly its
    /// length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 0x01_00 + -0x02_FF = -0x01_FF.
    /// let a = Number::new_from_i32(0x0100, ByteOrder::LittleEndian);
    /// let b = Number::new_from_i32(-0x02FF, ByteOrder::BigEndian);
    /// let sum = a + b;
    /// assert_eq!(sum.value, [0xFF, 0x01]);
    /// assert!(sum.is_negative());
    /// ```
    fn add(mut self, rhs: Number) -> Number {
        self += rhs;
        self
    }
}

impl AddAssign<&Number> for Number {
    /// Adds `rhs` to `self`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The sum is built in the vector of `self` when its capacity has room for the sum, which
    /// allocates nothing, and otherwise in one new vector of exactly its length, which
    /// replaces it. `rhs` is left as it was.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let mut n = Number::new_from_u8(0xFF, ByteOrder::LittleEndian);
    /// n += &Number::new_from_u16(0x0101, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [0x00, 0x02]);
    /// n += &Number::new_from_i16(-0x0201, ByteOrder::BigEndian);
    /// assert_eq!(n.value, [1]);
    /// assert!(n.is_negative());
    /// ```
    #[inline]
    fn add_assign(&mut self, rhs: &Number) {
        if small(self, rhs) {
            small_add_to(self, rhs, rhs.negative);
            return;
        }
        add_assign_ref(self, rhs);
    }
}

impl AddAssign<Number> for Number {
    /// Adds `rhs` to `self`, keeping the byte order of `self`, in the shortest form.
    ///
    /// The sum is built in the vector of `self` when its capacity has room for the sum, or else
    /// in the vector of `rhs` when that has room, also when `rhs` is in the other byte order,
    /// either of which allocates nothing; otherwise it gets one new vector of exactly its
    /// length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // `n` has no room for 0x01_00, but the vector of the right operand has.
    /// let mut n = Number::new_from_u8(0xFF, ByteOrder::BigEndian);
    /// let mut value = Vec::with_capacity(2);
    /// value.push(1);
    /// let one = Number::new(value, ByteOrder::BigEndian);
    /// let pointer = one.value.as_ptr();
    /// n += one;
    /// assert_eq!(n.value, [1, 0]);
    /// assert_eq!(n.value.as_ptr(), pointer);
    /// ```
    #[inline]
    fn add_assign(&mut self, mut rhs: Number) {
        if small(self, &rhs) {
            let sum = small_sum(self, &rhs, rhs.negative);
            self.negative = sum.negative;
            let order = self.byte_order;
            // SAFETY: a small sum is from 1 to 17 bytes long, and the vector it is written in
            // has room for it.
            unsafe {
                if self.value.capacity() >= sum.len {
                    write_small_in(&mut self.value, order, sum.lo, sum.hi, sum.len);
                } else if rhs.value.capacity() >= sum.len {
                    write_small_in(&mut rhs.value, order, sum.lo, sum.hi, sum.len);
                    // The vector of `self` goes to `rhs`, and is dropped with it.
                    mem::swap(&mut self.value, &mut rhs.value);
                } else {
                    replace_small(&mut self.value, order, sum.lo, sum.hi, sum.len);
                }
            }
            return;
        }
        // `rhs` is dropped on return, with the vector it holds then.
        add_assign_value(self, &mut rhs);
    }
}

// The general path: operands of more than 16 bytes, out of line, so that the small path saves no
// registers for it.

/// `a + b` for borrowed operands, of which at least one has more than 16 bytes.
#[inline(never)]
pub(super) fn ref_plus_ref(a: &Number, b: &Number) -> Number {
    let plan = plan(a, b);
    // SAFETY: the plan is that of `a` and `b`, in their own byte orders.
    let value = unsafe { sum_new(a, b, &plan) };
    Number {
        value,
        negative: plan.negative,
        byte_order: a.byte_order,
    }
}

/// `a + b` for a borrowed `a` and an owned `b`, of which at least one has more than 16 bytes.
/// The sum takes the vector of `b` when it is built there, leaving `b` an empty one; the
/// caller drops `b` either way.
#[inline(never)]
fn ref_plus_value(a: &Number, b: &mut Number) -> Number {
    let plan = plan(a, b);
    let order = a.byte_order;
    let reverse = b.byte_order != order && plan.len > 0;
    if reverse {
        b.value.reverse();
        b.byte_order = order;
    }
    let value = if plan.fits(b.value.capacity()) {
        // SAFETY: the plan is that of `a` and `b`, which are both in `order` now unless the
        // sum is zero, and the vector of `b` has room for the sum.
        unsafe { sum_in(&mut b.value, order, true, &a.value, &plan, reverse) };
        mem::take(&mut b.value)
    } else {
        // SAFETY: the plan is that of `a` and `b`, in their own byte orders.
        unsafe { sum_new(a, b, &plan) }
    };
    Number {
        value,
        negative: plan.negative,
        byte_order: order,
    }
}

/// `a += b` for a borrowed `b`, where at least one of them has more than 16 bytes.
#[inline(never)]
pub(super) fn add_assign_ref(a: &mut Number, b: &Number) {
    let plan = plan(a, b);
    if !plan.fits(a.value.capacity()) {
        replace_with_sum(a, b, &plan);
        return;
    }
    // A borrowed `b` cannot be reversed: when the byte orders differ, `a` is reversed into the
    // byte order of `b` instead, and the sum reversed back. One call, so that the computation
    // inlined into it is there once.
    let reverse = b.byte_order != a.byte_order && plan.len > 0;
    if reverse {
        a.value.reverse();
    }
    // SAFETY: the plan is that of `a` and `b`, which are in the byte order of `b` now unless the
    // sum is zero, and the vector of `a` has room for the sum.
    unsafe { sum_in(&mut a.value, b.byte_order, false, &b.value, &plan, reverse) };
    if reverse {
        a.value.reverse();
    }
    a.negative = plan.negative;
}

/// `a += b` for an owned `b`, where at least one of them has more than 16 bytes. When the sum
/// is built in the vector of `b`, the vector of `a` goes to `b`; the caller drops `b` either
/// way.
#[inline(never)]
fn add_assign_value(a: &mut Number, b: &mut Number) {
    let plan = plan(a, b);
    let order = a.byte_order;
    let reverse = b.byte_order != order && plan.len > 0;
    if reverse {
        b.value.reverse();
        b.byte_order = order;
    }
    let in_a = plan.fits(a.value.capacity());
    if !in_a && !plan.fits(b.value.capacity()) {
        replace_with_sum(a, b, &plan);
        return;
    }
    // One call, so that the computation inlined into it is there once.
    let (v, other) = if in_a {
        (&mut a.value, &b.value)
    } else {
        (&mut b.value, &a.value)
    };
    // SAFETY: the plan is that of `a` and `b`, which are both in `order` now unless the sum is
    // zero, and the vector of `a`, or else that of `b`, has room for the sum.
    unsafe { sum_in(v, order, !in_a, other, &plan, reverse) };
    if !in_a {
        mem::swap(&mut a.value, &mut b.value);
    }
    a.negative = plan.negative;
}

/// `a += b` for operands of which neither vector has room for the sum: its new vector replaces
/// that of `a`. Out of line and cold, so that the sums built in place keep no registers for
/// its allocation.
#[cold]
#[inline(never)]
fn replace_with_sum(a: &mut Number, b: &Number, plan: &Plan) {
    // SAFETY: the plan is that of `a` and `b`, in their own byte orders.
    a.value = unsafe { sum_new(a, b, plan) };
    a.negative = plan.negative;
}

// Small numbers

/// The longest vector, in bytes, of the operands that take the small path: with both this
/// short, their magnitudes are added or subtracted as `u128`s.
const SMALL: usize = 16;

/// Whether `a + b` takes the small path: both vectors hold at most [`SMALL`] bytes.
#[inline(always)]
pub(super) fn small(a: &Number, b: &Number) -> bool {
    a.value.len() <= SMALL && b.value.len() <= SMALL
}

// The small paths of the forms with a borrowed right operand, which subtraction shares: `b` is
// taken with the sign `negative_b`, its own for a sum and the opposite one for a difference.
// They are inlined, so that the sign is a value in a register either way.

/// `a + b` for small borrowed operands, `b` taken with the sign `negative_b`, in one new
/// vector.
#[inline(always)]
pub(super) fn small_number_of(a: &Number, b: &Number, negative_b: bool) -> Number {
    let sum = small_sum(a, b, negative_b);
    // SAFETY: a small sum is from 1 to 17 bytes long.
    unsafe { small_number(a.byte_order, sum.lo, sum.hi, sum.len, sum.negative) }
}

/// `a += b` for small operands and a borrowed `b`, taken with the sign `negative_b`.
#[inline(always)]
pub(super) fn small_add_to(a: &mut Number, b: &Number, negative_b: bool) {
    let sum = small_sum(a, b, negative_b);
    a.negative = sum.negative;
    let order = a.byte_order;
    // SAFETY: a small sum is from 1 to 17 bytes long, and the vector it is written in has room
    // for it.
    unsafe {
        if a.value.capacity() >= sum.len {
            write_small_in(&mut a.value, order, sum.lo, sum.hi, sum.len);
        } else {
            replace_small(&mut a.value, order, sum.lo, sum.hi, sum.len);
        }
    }
}

/// The sum of two small operands, as the small path computes it in registers.
struct Small {
    /// The low eight bytes of the magnitude.
    lo: u64,
    /// The next eight bytes of the magnitude.
    hi: u64,
    /// The length of the sum in the shortest form, from 1 to 17: one byte for zero, and 17
    /// when it carries out of the sixteenth byte, whose seventeenth byte is then 1.
    len: usize,
    /// Whether the sum is negative; never for zero.
    negative: bool,
}

/// The sum of the small operands `a` and `b`, `b` taken with the sign `negative_b`: both the
/// sum and the difference of their magnitudes are computed, and the one the signs ask for
/// chosen without a branch, as is the difference of the larger magnitude and the smaller.
#[inline(always)]
fn small_sum(a: &Number, b: &Number, negative_b: bool) -> Small {
    // SAFETY: both vectors hold at most 16 bytes, as `small` checked.
    let (x, y) = unsafe {
        (
            read_small(&a.value, a.byte_order),
            read_small(&b.value, b.byte_order),
        )
    };
    let negative_x = a.negative && x != 0;
    let negative_y = negative_b && y != 0;
    let subtract = negative_x != negative_y;
    let (sum, carry) = x.overflowing_add(y);
    let (difference, borrow) = x.overflowing_sub(y);
    let difference = select_unpredictable(borrow, difference.wrapping_neg(), difference);
    let magnitude = select_unpredictable(subtract, difference, sum);
    let carry = carry & !subtract;
    // One byte for zero, and 17 with a carry, chosen without a branch: the carry of two
    // magnitudes of 16 bytes is often a coin toss, also when the signs differ and it is not
    // used.
    let len = select_unpredictable(
        carry,
        SMALL + 1,
        SMALL - (magnitude | 1).leading_zeros() as usize / 8,
    );
    Small {
        lo: magnitude as u64,
        hi: (magnitude >> 64) as u64,
        len,
        // The sign of the larger magnitude, which is that of `y` when it is subtracted from a
        // smaller `x`; never for zero.
        negative: negative_x != (subtract && borrow) && (magnitude != 0 || carry),
    }
}

/// The magnitude in the at most 16 bytes of `v`, in the layout of `order`.
///
/// Only the bytes of `v` are read, never its spare capacity: from 8 bytes up, the eight
/// least significant ones and the eight most significant ones, which overlap below 16 and
/// agree where they do; from 4 up, the same with four, and from 2 with two.
///
/// # Safety
///
/// `v` must hold at most 16 bytes.
#[inline(always)]
unsafe fn read_small(v: &[u8], order: ByteOrder) -> u128 {
    let (p, n) = (v.as_ptr(), v.len());
    let be = order == ByteOrder::BigEndian;
    // SAFETY: every read lies inside the `n` bytes of `v`: `[0, k)` and `[n - k, n)` with
    // `n >= k`, for `k` of 8, 4 and 2, and the byte `0` with `n = 1`.
    unsafe {
        if n >= 4 {
            if n >= 8 {
                let lo = if be {
                    u64::from_be_bytes(read(p.add(n - 8)))
                } else {
                    u64::from_le_bytes(read(p))
                };
                let hi = if n > 8 {
                    let top = if be {
                        u64::from_be_bytes(read(p))
                    } else {
                        u64::from_le_bytes(read(p.add(n - 8)))
                    };
                    // `top` holds positions `n - 8..n`; those from 8 up are the high half, a
                    // shift by `8 * (16 - n)` down, which is below 64 and so `-8n` modulo 64.
                    top.wrapping_shr((n as u32).wrapping_neg().wrapping_mul(8))
                } else {
                    0
                };
                u128::from(lo) | (u128::from(hi) << 64)
            } else {
                let (lo, top) = if be {
                    (
                        u32::from_be_bytes(read(p.add(n - 4))),
                        u32::from_be_bytes(read(p)),
                    )
                } else {
                    (
                        u32::from_le_bytes(read(p)),
                        u32::from_le_bytes(read(p.add(n - 4))),
                    )
                };
                // `top` holds positions `n - 4..n`, which agree with `lo` where they overlap.
                u128::from(u64::from(lo) | (u64::from(top) << (8 * (n - 4))))
            }
        } else if n >= 2 {
            let (lo, top) = if be {
                (
                    u16::from_be_bytes(read(p.add(n - 2))),
                    u16::from_be_bytes(read(p)),
                )
            } else {
                (
                    u16::from_le_bytes(read(p)),
                    u16::from_le_bytes(read(p.add(n - 2))),
                )
            };
            u128::from(u32::from(lo) | (u32::from(top) << (8 * (n - 2))))
        } else if n == 1 {
            u128::from(*p)
        } else {
            0
        }
    }
}

/// Writes the small sum of magnitude `hi:lo` and length `len` over the first `len` bytes at
/// `p`, in the layout of `order`.
///
/// From 8 bytes up, two stores of eight, the most significant ones first, then the eight least
/// significant ones over the bytes they overlap; 17 bytes are 16 and the top one. From 4 up, the
/// same with four. Below 4, the least significant, the middle and the most significant byte. A
/// sum of two operands of `k` bytes is `k` or `k + 1` bytes long, and for `k` of 1, 2, 4 and 8
/// both lengths take the same stores.
///
/// The bytes are the sum's when `len` is its length as [`Small::len`] gives it.
///
/// # Safety
///
/// `len` must be from 1 to 17, and `p` valid for writing `len` bytes.
#[inline(always)]
unsafe fn write_small(p: *mut u8, order: ByteOrder, lo: u64, hi: u64, len: usize) {
    let be = order == ByteOrder::BigEndian;
    // The shift that puts the high bytes of the class from 4 or 8 bytes at the top of their
    // store, `8 * (2k - len)` for stores of `k` bytes, modulo `8k`.
    let shift = (len as u32).wrapping_neg().wrapping_mul(8);
    // SAFETY: every write lies inside the `len` bytes at `p`: `[0, 17)` for 17, `[0, k)` and
    // `[len - k, len)` for `k` of 8 and 4 with `len >= k`, and the bytes `0`, `len / 2` and
    // `len - 1` below 4. Of two writes that overlap, the later one puts the sum's bytes there.
    unsafe {
        if len >= 8 {
            if len > SMALL {
                if be {
                    *p = 1;
                    write(p.add(1), hi.to_be_bytes());
                    write(p.add(9), lo.to_be_bytes());
                } else {
                    write(p, lo.to_le_bytes());
                    write(p.add(8), hi.to_le_bytes());
                    *p.add(16) = 1;
                }
            } else {
                // The high bytes at the top of their eight; the bytes below them do not
                // matter, as the low eight are written over them. For 8 bytes, `hi` and the
                // shift are 0.
                let top = hi.wrapping_shl(shift);
                if be {
                    write(p, top.to_be_bytes());
                    write(p.add(len - 8), lo.to_be_bytes());
                } else {
                    write(p.add(len - 8), top.to_le_bytes());
                    write(p, lo.to_le_bytes());
                }
            }
        } else if len >= 4 {
            let top = ((lo >> 32) as u32).wrapping_shl(shift);
            if be {
                write(p, top.to_be_bytes());
                write(p.add(len - 4), (lo as u32).to_be_bytes());
            } else {
                write(p.add(len - 4), top.to_le_bytes());
                write(p, (lo as u32).to_le_bytes());
            }
        } else {
            let (mid, last) = (len / 2, len - 1);
            let (b0, b1, b2) = (lo as u8, (lo >> (8 * mid)) as u8, (lo >> (8 * last)) as u8);
            if be {
                *p.add(last) = b0;
                *p.add(last - mid) = b1;
                *p = b2;
            } else {
                *p = b0;
                *p.add(mid) = b1;
                *p.add(last) = b2;
            }
        }
    }
}

/// Writes the small sum of magnitude `hi:lo` and length `len` over `v`, in the layout of
/// `order`, as its whole value.
///
/// # Safety
///
/// `len` must be from 1 to 17, and the capacity of `v` at least `len`.
#[inline(always)]
unsafe fn write_small_in(v: &mut Vec<u8>, order: ByteOrder, lo: u64, hi: u64, len: usize) {
    // SAFETY: as the caller guarantees; `write_small` writes all of the first `len` bytes, so
    // they are initialised when the length is set.
    unsafe {
        write_small(v.as_mut_ptr(), order, lo, hi, len);
        v.set_len(len);
    }
}

/// The small sum of magnitude `hi:lo` and length `len` in one new vector of exactly its
/// length, in the layout of `order`.
///
/// # Safety
///
/// `len` must be from 1 to 17.
#[inline(always)]
unsafe fn new_small(order: ByteOrder, lo: u64, hi: u64, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    // SAFETY: as the caller guarantees, and the new vector has room for the `len` bytes.
    unsafe { write_small_in(&mut v, order, lo, hi, len) };
    v
}

/// The small sum of magnitude `hi:lo`, length `len` and sign `negative` as a number in one new
/// vector of exactly its length, in the byte order `order`. Out of line, and given the sum in
/// registers, so that the small path of `&a + &b` saves no registers for the allocation.
///
/// # Safety
///
/// `len` must be from 1 to 17.
#[inline(never)]
unsafe fn small_number(order: ByteOrder, lo: u64, hi: u64, len: usize, negative: bool) -> Number {
    // SAFETY: as the caller guarantees: told so, the compiler checks neither the size of the
    // allocation nor a zero length.
    unsafe {
        core::hint::assert_unchecked(len != 0);
        core::hint::assert_unchecked(len <= SMALL + 1);
        Number {
            value: new_small(order, lo, hi, len),
            negative,
            byte_order: order,
        }
    }
}

/// Replaces `v` with one new vector of exactly the small sum's length, holding the sum of
/// magnitude `hi:lo` and length `len` in the layout of `order`. Out of line and cold, and
/// given the sum in registers, so that the small path saves no registers for it.
///
/// # Safety
///
/// `len` must be from 1 to 17.
#[cold]
#[inline(never)]
unsafe fn replace_small(v: &mut Vec<u8>, order: ByteOrder, lo: u64, hi: u64, len: usize) {
    // SAFETY: as the caller guarantees.
    *v = unsafe { new_small(order, lo, hi, len) };
}

// Planning

/// The sum `a + b` as planned from the untouched operands, before a vector is chosen for it.
///
/// The magnitude the other one is added to or subtracted from is `x`, the other one `y`: for
/// an addition `x` has at least as many significant bytes as `y`, for a subtraction it is the
/// larger magnitude. The significant length of a magnitude counts its bytes up to the most
/// significant nonzero one, and is 0 for zero.
struct Plan {
    /// Whether `y` is subtracted from `x` (the signs differ) rather than added to it.
    sub: bool,
    /// Whether `x` is `b` rather than `a`.
    x_is_b: bool,
    /// The significant length of `a`.
    la: usize,
    /// The significant length of `b`.
    lb: usize,
    /// The significant length of the sum: 0 for zero.
    len: usize,
    /// Whether the sum is negative; never for zero.
    negative: bool,
}

impl Plan {
    /// The significant length of `x`.
    fn lx(&self) -> usize {
        if self.x_is_b { self.lb } else { self.la }
    }

    /// The significant length of `y`.
    fn ly(&self) -> usize {
        if self.x_is_b { self.la } else { self.lb }
    }

    /// Whether a vector of capacity `capacity` has room for the sum in the shortest form, one
    /// byte for zero.
    fn fits(&self, capacity: usize) -> bool {
        capacity >= self.len.max(1)
    }

    /// Whether the operand whose vector holds the sum, `b` if `alias_is_b` and `a` otherwise,
    /// is `x`.
    fn alias_is_x(&self, alias_is_b: bool) -> bool {
        alias_is_b == self.x_is_b
    }
}

/// Plans `a + b`.
fn plan(a: &Number, b: &Number) -> Plan {
    match (a.byte_order, b.byte_order) {
        (ByteOrder::BigEndian, ByteOrder::BigEndian) => plan_in::<true, true>(a, b),
        (ByteOrder::BigEndian, ByteOrder::LittleEndian) => plan_in::<true, false>(a, b),
        (ByteOrder::LittleEndian, ByteOrder::BigEndian) => plan_in::<false, true>(a, b),
        (ByteOrder::LittleEndian, ByteOrder::LittleEndian) => plan_in::<false, false>(a, b),
    }
}

/// Plans `a + b` for `a` in the layout `A` and `b` in the layout `B` (`true` for big-endian).
///
/// An addition is planned here, a subtraction by [`plan_sub`], out of line: it needs more
/// registers, which an addition then does not have to save.
fn plan_in<const A: bool, const B: bool>(a: &Number, b: &Number) -> Plan {
    let la = significant_len::<A>(&a.value);
    let lb = significant_len::<B>(&b.value);
    let (pa, pb) = (end::<A>(&a.value), end::<B>(&b.value));
    let negative_a = a.negative && la > 0;
    let negative_b = b.negative && lb > 0;
    if negative_a == negative_b || la == 0 || lb == 0 {
        let x_is_b = lb > la;
        // SAFETY: the significant bytes of `a` and `b` are inside their vectors, and `x` is
        // the one with more of them.
        let carry = unsafe {
            if x_is_b {
                carries::<B, A>(pb, pa, lb, la)
            } else {
                carries::<A, B>(pa, pb, la, lb)
            }
        };
        return Plan {
            sub: false,
            x_is_b,
            la,
            lb,
            len: la.max(lb) + usize::from(carry),
            // The sign of the operands that are not zero.
            negative: negative_a || negative_b,
        };
    }
    // SAFETY: `pa` and `pb` are the least significant ends of the magnitudes of `a` and `b`,
    // of `la` and `lb` significant bytes, neither zero.
    unsafe { plan_sub::<A, B>(pa, pb, la, lb, negative_a) }
}

/// Plans the difference of the magnitudes at the least significant ends `pa` and `pb` in the
/// layouts `A` and `B`, of `la` and `lb` significant bytes, for `a + b` with the signs of `a`
/// and `b` different: the sum has the sign of the larger magnitude, which `negative_a` gives
/// for `a`.
///
/// # Safety
///
/// The `la` significant bytes at `pa` and the `lb` at `pb` must be valid for reading, and
/// neither `la` nor `lb` may be 0.
#[inline(never)]
unsafe fn plan_sub<const A: bool, const B: bool>(
    pa: *const u8,
    pb: *const u8,
    la: usize,
    lb: usize,
    negative_a: bool,
) -> Plan {
    // The longer magnitude is the larger, and of two as long, the one with the larger byte
    // where they first differ from the top.
    let (x_is_b, top) = if la != lb {
        (lb > la, la.max(lb) - 1)
    } else {
        // SAFETY: both have `la` significant bytes.
        match unsafe { highest_difference::<A, B>(pa, pb, lb, la) } {
            // SAFETY: position `k` is below the significant length of both.
            Some(k) => (unsafe { get::<B>(pb, k) > get::<A>(pa, k) }, k),
            None => {
                return Plan {
                    sub: true,
                    x_is_b: false,
                    la,
                    lb,
                    len: 0,
                    negative: false,
                };
            }
        }
    };
    // SAFETY: `x` is the larger magnitude, and `top` the highest position at which `x` and
    // `y` differ, below the significant length of `x`.
    let len = unsafe {
        if A == B {
            // Which magnitude is the larger depends on their bytes: one copy of the scan, on
            // operands chosen without a branch.
            let (x, y, ly) = select_unpredictable(x_is_b, (pb, pa, la), (pa, pb, lb));
            difference_len::<A, B>(x, y, ly, top)
        } else if x_is_b {
            difference_len::<B, A>(pb, pa, la, top)
        } else {
            difference_len::<A, B>(pa, pb, lb, top)
        }
    };
    Plan {
        sub: true,
        x_is_b,
        la,
        lb,
        len,
        // The signs differ: `b` is negative exactly when `a` is not.
        negative: negative_a != x_is_b,
    }
}

/// The significant length of the magnitude `v` in the layout `BE`: its length without the
/// zero bytes above its most significant nonzero byte, the leading ones big-endian and the
/// trailing ones little-endian. The bytes are read eight at a time while they are zero, so
/// a number in the shortest form costs one read, and a long run of zeros little more.
#[inline]
fn significant_len<const BE: bool>(v: &[u8]) -> usize {
    let (p, len) = (v.as_ptr(), v.len());
    // SAFETY: every read lies inside `v`: big-endian `[i, i + 8)` with `i + 8 <= len` and `i`
    // with `i < len`; little-endian `[end - 8, end)` with `end >= 8` and `end - 1` with
    // `end > 0`, where `end <= len`.
    unsafe {
        if BE {
            let mut i = 0;
            while len - i >= 8 && p.add(i).cast::<[u8; 8]>().read() == [0; 8] {
                i += 8;
            }
            while i < len && *p.add(i) == 0 {
                i += 1;
            }
            len - i
        } else {
            let mut end = len;
            while end >= 8 && p.add(end - 8).cast::<[u8; 8]>().read() == [0; 8] {
                end -= 8;
            }
            while end > 0 && *p.add(end - 1) == 0 {
                end -= 1;
            }
            end
        }
    }
}

/// Whether `x + y` carries out of the top position of `x`, for magnitudes of `lx >= ly`
/// significant bytes at the least significant ends `x` and `y`, in the layouts `X` and `Y`.
///
/// It is found from the top down, without computing the sum: the highest position whose
/// bytes do not add up to `0xFF` decides, with a carry if they add up to more. Above `ly`
/// only `x` has bytes, which can pass a carry on but not start one. The positions go eight at
/// a time, as `u64`s, while that many are left: such a group starts a carry if its two
/// values overflow, passes one on if they add up to `u64::MAX`, and stops one otherwise.
///
/// # Safety
///
/// The `lx` significant bytes of `x` and the `ly` of `y` must be valid for reading.
#[inline]
unsafe fn carries<const X: bool, const Y: bool>(
    x: *const u8,
    y: *const u8,
    lx: usize,
    ly: usize,
) -> bool {
    let mut p = lx;
    // SAFETY: every position read is below `p`, which is at most `lx`, and those of `y` are
    // below `ly` as well.
    unsafe {
        while p >= ly + 8 {
            if load::<X>(x, p - 8) != u64::MAX {
                return false;
            }
            p -= 8;
        }
        while p > ly {
            if get::<X>(x, p - 1) != 0xFF {
                return false;
            }
            p -= 1;
        }
        while p >= 8 {
            let (sum, overflow) = load::<X>(x, p - 8).overflowing_add(load::<Y>(y, p - 8));
            if overflow || sum != u64::MAX {
                return overflow;
            }
            p -= 8;
        }
        while p > 0 {
            let sum = u16::from(get::<X>(x, p - 1)) + u16::from(get::<Y>(y, p - 1));
            if sum != 0xFF {
                return sum > 0xFF;
            }
            p -= 1;
        }
    }
    false
}

/// The highest position below `p` at which the magnitudes at the least significant ends `x`
/// and `y`, in the layouts `X` and `Y`, differ, the positions of `y` from `ly` up counting as
/// zeros; `None` if they agree on every position below `p`.
///
/// The positions go eight at a time while that many are left, as `u64`s whose most
/// significant byte is the highest position: the leading zeros of their difference (their
/// XOR) tell how far down from it the first difference lies.
///
/// # Safety
///
/// The positions of `x` below `p`, and those of `y` below `p` and `ly`, must be valid for
/// reading.
#[inline]
unsafe fn highest_difference<const X: bool, const Y: bool>(
    x: *const u8,
    y: *const u8,
    ly: usize,
    mut p: usize,
) -> Option<usize> {
    // SAFETY: every position read is below `p`, and those of `y` below `ly` as well.
    unsafe {
        // Above `ly`, the nonzero bytes of `x`.
        while p >= ly + 8 {
            let v = load::<X>(x, p - 8);
            if v != 0 {
                return Some(p - 1 - (v.leading_zeros() / 8) as usize);
            }
            p -= 8;
        }
        while p > ly {
            if get::<X>(x, p - 1) != 0 {
                return Some(p - 1);
            }
            p -= 1;
        }
        while p >= 8 {
            let v = load::<X>(x, p - 8) ^ load::<Y>(y, p - 8);
            if v != 0 {
                return Some(p - 1 - (v.leading_zeros() / 8) as usize);
            }
            p -= 8;
        }
        while p > 0 {
            if get::<X>(x, p - 1) != get::<Y>(y, p - 1) {
                return Some(p - 1);
            }
            p -= 1;
        }
    }
    None
}

/// The significant length of `x - y` for magnitudes at the least significant ends `x` and
/// `y`, in the layouts `X` and `Y`, `x` the larger, `y` of `ly` significant bytes, and `k`
/// the highest position at which they differ.
///
/// Position `p` of the difference holds `x_p - y_p - b_p` modulo 256, where the borrow `b_p`
/// into it is 1 exactly when `x` has the smaller byte at the highest position below `p` at
/// which the two differ. Above `k` the difference is zero, and at `k`, where `x` has the
/// larger byte, it is not unless `x_k - y_k = 1`: the length is then `k + 1`.
///
/// Otherwise take the run of positions below `k` at which `x` has `0x00` and `y` has `0xFF`,
/// down to `p`, its lowest (`p = k` for an empty run). Such a position borrows whatever the
/// borrow into it is, and holds `1 - b` for a borrow `b` into it. So the positions of the run
/// above `p`, and `k` itself, hold 0 when the run is not empty, and position `p` holds
/// `1 - b_p`, as `k` does when it is empty. With `b_p = 0` the length is `p + 1`. With
/// `b_p = 1` it is `p`: `x` has the smaller byte at the highest position `q` below `p` at
/// which the two differ, and position `p - 1` is not zero, either because `q = p - 1`, where
/// `x_q - y_q` is between -254 and -1 (the pair there is not `(0x00, 0xFF)`), so that the
/// byte is between 1 and 255 whatever the borrow into it, or because the bytes agree from
/// `q + 1` to `p - 1` and the borrow out of `q` makes them `0xFF`. In both cases the length is
/// `p + 1 - b_p`.
///
/// The run is found eight positions at a time, and `b_p` by [`highest_difference`] from `p`
/// down: a run is read once however long it is (`256^n - (256^n - 1)` has one of `n`
/// positions), and only the eight positions at which it ends may be read twice.
///
/// # Safety
///
/// The significant bytes of `x` and the `ly` of `y` must be valid for reading, and `k` must be
/// below the significant length of `x`.
#[inline]
unsafe fn difference_len<const X: bool, const Y: bool>(
    x: *const u8,
    y: *const u8,
    ly: usize,
    k: usize,
) -> usize {
    // SAFETY: every position read is at most `k`, below the significant length of `x`; those
    // of `y` are read below `ly` only, as the run is looked for only when `k <= ly`.
    unsafe {
        if get::<X>(x, k).wrapping_sub(get_or_zero::<Y>(y, ly, k)) != 1 {
            return k + 1;
        }
        // Above `ly` the bytes of `y` are zeros, and no run starts.
        let mut p = k;
        if p <= ly {
            while p >= 8 {
                // Zero exactly where `x` has 0x00 and `y` has 0xFF.
                let v = load::<X>(x, p - 8) | !load::<Y>(y, p - 8);
                if v != 0 {
                    p -= (v.leading_zeros() / 8) as usize;
                    break;
                }
                p -= 8;
            }
            while p > 0 && get::<X>(x, p - 1) == 0 && get::<Y>(y, p - 1) == 0xFF {
                p -= 1;
            }
        }
        let borrow = match highest_difference::<X, Y>(x, y, ly, p) {
            Some(q) => get::<X>(x, q) < get_or_zero::<Y>(y, ly, q),
            None => false,
        };
        p + 1 - usize::from(borrow)
    }
}

// Building the sum

/// Builds the planned sum of `a` and `b` in one new vector, of exactly its length, in the
/// byte order of `a`.
///
/// When `b` is in the other byte order, its low bytes, those the sum needs, are first copied
/// into the new vector, in the byte order of the sum and where they belong in it, and the sum
/// is then computed over that copy in place.
///
/// # Safety
///
/// `plan` must be the plan of `a + b`.
unsafe fn sum_new(a: &Number, b: &Number, plan: &Plan) -> Vec<u8> {
    if plan.len == 0 {
        return vec![0];
    }
    // SAFETY: as the caller guarantees; the operands are each read in their own layout.
    unsafe {
        match (a.byte_order, b.byte_order) {
            (ByteOrder::BigEndian, ByteOrder::BigEndian) => {
                new_from::<true>(end::<true>(&a.value), end::<true>(&b.value), plan)
            }
            (ByteOrder::LittleEndian, ByteOrder::LittleEndian) => {
                new_from::<false>(end::<false>(&a.value), end::<false>(&b.value), plan)
            }
            (ByteOrder::BigEndian, ByteOrder::LittleEndian) => {
                new_with_copy::<true>(end::<true>(&a.value), &b.value, plan)
            }
            (ByteOrder::LittleEndian, ByteOrder::BigEndian) => {
                new_with_copy::<false>(end::<false>(&a.value), &b.value, plan)
            }
        }
    }
}

/// Builds the planned sum in a new vector in the layout `BE`, from the operands at the least
/// significant ends `a` and `b` in that layout.
///
/// # Safety
///
/// `plan` must be the plan of the sum of `a` and `b`, and not zero.
unsafe fn new_from<const BE: bool>(a: *const u8, b: *const u8, plan: &Plan) -> Vec<u8> {
    let len = plan.len;
    let mut v = Vec::<u8>::with_capacity(len);
    let base = v.as_mut_ptr();
    // SAFETY: the vector has room for the `len` bytes of the sum, and as it is apart from the
    // operands, `compute` writes every one of them. The end of the sum, `base + len`
    // big-endian, is inside or one past the allocation.
    unsafe {
        let r = if BE { base.add(len) } else { base };
        let (x, y) = select_unpredictable(plan.x_is_b, (b, a), (a, b));
        compute_new::<BE>(r, x, y, plan);
        v.set_len(len);
    }
    v
}

/// Builds the planned sum in a new vector in the layout `BE`, from `a` at its least
/// significant end in that layout and `b` in the other byte order: the low positions of `b`
/// that the sum needs are copied into the vector, where they belong in the sum, and the sum is
/// computed over the copy in place.
///
/// # Safety
///
/// `plan` must be the plan of the sum of `a` and `b`, and not zero.
unsafe fn new_with_copy<const BE: bool>(a: *const u8, b: &[u8], plan: &Plan) -> Vec<u8> {
    let len = plan.len;
    // No position of `b` at or above the length of the sum changes the sum.
    let m = plan.lb.min(len);
    let mut v = Vec::<u8>::with_capacity(len);
    let base = v.as_mut_ptr();
    let copy_is_x = plan.alias_is_x(true);
    // The copy is `x` only when `b` has at least as many significant bytes as `a`, so that it
    // holds all of `x` that is kept: `lx` bytes for a sum, `len` for a difference.
    debug_assert!(!copy_is_x || m == if plan.sub { len } else { plan.lx() });
    // SAFETY: the vector has room for the `len` bytes of the sum. `b` holds its `m <= lb`
    // lowest positions at its start if it is little-endian and at its end otherwise, and they
    // go to positions `0..m` of the sum: `[len - m, len)` big-endian, `[0, m)` little-endian.
    // `compute` then reads the copy as an operand that is the output exactly, and writes the
    // positions of the sum that differ from it: all of them when the copy is `y`; when it is
    // `x`, those below the position at which the carry (borrow) out of `y` stops, the copy
    // keeping the rest, which it holds as `m` is all of `x` that the sum needs. So every byte
    // below `len` is written before the length is set.
    unsafe {
        if BE {
            copy_reversed(base.add(len - m), b.as_ptr(), m);
        } else {
            copy_reversed(base, b.as_ptr().add(b.len() - m), m);
        }
        let r = if BE { base.add(len) } else { base };
        let copy = r.cast_const();
        let (x, y) = select_unpredictable(copy_is_x, (copy, a), (a, copy));
        compute_new::<BE>(r, x, y, plan);
        v.set_len(len);
    }
    v
}

/// Builds the planned sum in `v`, the vector of one of the operands (`b` if `alias_is_b`, `a`
/// otherwise), in the layout of `order`; `other` is the other operand. `reversed` says whether
/// one of them was reversed into `order` after the plan was made.
///
/// It is inlined into the operators' general paths, with everything under it but the kernels
/// for many limbs: below a few hundred bytes the calls cost such a sum a sixth of its time. A
/// long difference goes out of line instead, to [`long_in_vector`], unless an operand was
/// reversed: the reversal takes longer than the plan, which is then done before the kernel
/// could start, and the branches of a long difference would only add their mispredictions (3 to
/// 4 % from 512 bytes, measured).
///
/// # Safety
///
/// `plan` must be the plan of the sum, both operands must be in `order` (unless the sum is
/// zero), and the capacity of `v` must have room for the sum.
#[inline(always)]
unsafe fn sum_in(
    v: &mut Vec<u8>,
    order: ByteOrder,
    alias_is_b: bool,
    other: &[u8],
    plan: &Plan,
    reversed: bool,
) {
    // SAFETY: as the caller guarantees: for a zero sum, the capacity of `v` holds one byte.
    unsafe {
        if plan.len == 0 {
            *v.as_mut_ptr() = 0;
            v.set_len(1);
            return;
        }
        match order {
            ByteOrder::BigEndian if !reversed && long_difference(plan, LONG_DIFFERENCE_BE) => {
                long_in_vector::<true>(v, alias_is_b, end::<true>(other), plan)
            }
            ByteOrder::LittleEndian if !reversed && long_difference(plan, LONG_DIFFERENCE_LE) => {
                long_in_vector::<false>(v, alias_is_b, end::<false>(other), plan)
            }
            ByteOrder::BigEndian => {
                in_vector::<true, false>(v, alias_is_b, end::<true>(other), plan)
            }
            ByteOrder::LittleEndian => {
                in_vector::<false, false>(v, alias_is_b, end::<false>(other), plan)
            }
        }
    }
}

/// [`sum_in`] in the layout `BE`, with `other` the least significant end of the other operand,
/// for a sum that is not zero; with `LONG`, for a long difference, whose operands are taken by
/// [`difference_in`].
///
/// Little-endian the positions of the sum are those of the operand in `v`, from its start;
/// the bytes from its length up are its spare capacity, written and never read. Big-endian
/// the sum ends at its own length: a longer sum lies above the operand, which the kernels
/// allow, and a shorter one (a difference, or an operand with zero bytes above its most
/// significant one) is first given the low bytes it needs from the operand, moved down to end
/// where the sum ends. A byte of a sum or a difference depends only on the bytes at or below
/// its position, so the bytes of the operand from the length of the sum up are never needed.
///
/// # Safety
///
/// As for [`sum_in`], with `other` valid for reading the significant bytes of the other
/// operand, and the sum not zero.
#[inline(always)]
unsafe fn in_vector<const BE: bool, const LONG: bool>(
    v: &mut Vec<u8>,
    alias_is_b: bool,
    other: *const u8,
    plan: &Plan,
) {
    let (old_len, len) = (v.len(), plan.len);
    // Every pointer into the vector comes from this one call.
    let base = v.as_mut_ptr();
    // SAFETY: the capacity of `v` is at least `len`, and its first `old_len` bytes hold the
    // operand, so `base + len` and `base + old_len` are inside the allocation or one past it.
    // A shorter big-endian sum moves the `m` low bytes of the operand from the end of its
    // bytes to the end of the sum, within them. `compute` reads the operand where it now is,
    // as the output exactly, or below it big-endian, and writes the positions of the sum that
    // differ from it, keeping the operand's own bytes above them when it is `x` and the output
    // exactly: those below `old_len`, or among the `m` moved, which are all of `x` that the sum
    // needs (`lx` bytes for a sum, `len` for a difference). So every byte below `len` holds the
    // sum before the length is set.
    unsafe {
        let (r, alias) = if !BE {
            (base, base)
        } else if len >= old_len {
            (base.add(len), base.add(old_len))
        } else {
            let m = if alias_is_b { plan.lb } else { plan.la }.min(len);
            move_down(base.add(old_len), base.add(len), m);
            (base.add(len), base.add(len))
        };
        let alias = alias.cast_const();
        let alias_is_x = plan.alias_is_x(alias_is_b);
        if LONG {
            // The significant lengths of the operand in `v` and of the other one.
            let (l_alias, l_other) = if alias_is_b {
                (plan.lb, plan.la)
            } else {
                (plan.la, plan.lb)
            };
            if alias_is_x {
                difference_in::<BE, false>(r, alias, other, l_other, len);
            } else {
                difference_in::<BE, true>(r, alias, other, l_alias, len);
            }
        } else {
            let (x, y) = select_unpredictable(alias_is_x, (alias, other), (other, alias));
            compute::<BE>(r, x, y, plan);
        }
        v.set_len(len);
    }
}

/// Whether the planned sum is a long difference: a subtraction of magnitudes of at least `long`
/// significant bytes each.
#[inline(always)]
fn long_difference(plan: &Plan, long: usize) -> bool {
    plan.sub && plan.la.min(plan.lb) >= long
}

/// [`in_vector`] for a long difference, out of line, so that the code of the other sums is as it
/// would be without it.
///
/// # Safety
///
/// As for [`in_vector`], with the plan that of a long difference.
#[inline(never)]
unsafe fn long_in_vector<const BE: bool>(
    v: &mut Vec<u8>,
    alias_is_b: bool,
    other: *const u8,
    plan: &Plan,
) {
    // SAFETY: as the caller guarantees.
    unsafe { in_vector::<BE, true>(v, alias_is_b, other, plan) }
}

/// Moves the `m` bytes that end at `from` down to end at `to`, for the big-endian sum that is
/// shorter than its operand's vector. Out of line and cold, so that the call does not cost
/// the other sums the registers it needs.
///
/// # Safety
///
/// Both ranges of `m` bytes must lie in one allocation, `to <= from`.
#[cold]
#[inline(never)]
unsafe fn move_down(from: *mut u8, to: *mut u8, m: usize) {
    // SAFETY: as the caller guarantees; `ptr::copy` allows the ranges to overlap.
    unsafe { ptr::copy(from.sub(m), to.sub(m), m) }
}

// Computing the sum

/// Writes the planned sum of `x` and `y`, magnitudes at their least significant ends in the
/// layout `BE`, to the output whose least significant end is `r`: positions `0..plan.len`, of
/// which, when the output is `x` itself (`r == x`), it keeps those above the position at which
/// the carry (borrow) out of `y` stops as they are, the bytes of `x`.
///
/// # Safety
///
/// `x` and `y` must be valid for reading their significant bytes, and `r` for writing the
/// `plan.len` bytes of the sum. The output must be apart from both, equal to one or both
/// (`r == x` or `r == y`), or, big-endian, end `s >= 1` bytes above one of them (`r = x + s` or
/// `r = y + s`), as a sum that grows in the vector of its operand does. When `r == x`, the
/// output must hold `x` initialised up to its significant length for a sum, and up to
/// `plan.len` for a difference.
#[inline(always)]
unsafe fn compute<const BE: bool>(r: *mut u8, x: *const u8, y: *const u8, plan: &Plan) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if plan.sub {
            sub_into::<BE>(r, x, y, plan.ly(), plan.len);
        } else {
            add_into::<BE>(r, x, y, plan.lx(), plan.ly(), plan.len);
        }
    }
}

/// [`compute`] for a sum in a new vector, out of line: there the allocation costs more than the
/// call, and one copy of the computation serves both ways to build a new vector.
///
/// # Safety
///
/// As for [`compute`].
#[inline(never)]
unsafe fn compute_new<const BE: bool>(r: *mut u8, x: *const u8, y: *const u8, plan: &Plan) {
    // SAFETY: as the caller guarantees.
    unsafe { compute::<BE>(r, x, y, plan) }
}

/// Writes the `len` bytes of `x + y` to `r`, all least significant ends in the layout `BE`,
/// for magnitudes of `lx >= ly` significant bytes whose sum is `len` bytes long, `lx` or
/// `lx + 1`.
///
/// The whole limbs of `y` go to the kernel, and the bytes of `y` above them follow, as one
/// partial limb when there are at least three and a whole limb below them, and one at a time
/// otherwise. Above `y`, the carry goes on through `x` a limb at a time while it lasts, and
/// the rest of `x` is copied, unless the output is `x` itself. Each position is read before
/// it is written, from the least significant one up. The top byte, 1 when the sum carries out
/// of `x`, is written without a branch.
///
/// # Safety
///
/// As for [`compute`].
#[inline(always)]
unsafe fn add_into<const BE: bool>(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    lx: usize,
    ly: usize,
    len: usize,
) {
    let n = ly / 8;
    // SAFETY: the positions of `x` read are below `lx`, those of `y` below `ly`, and those of
    // `r` written below `len`; each position of the inputs is read before the same position of
    // the output is written, so an output equal to an input, or above it, overwrites only
    // what has been read.
    unsafe {
        let carry = add_n::<BE>(r, x, y, n, false);
        let mut carry = rest_of_y::<BE, false>(r, x, y, n, ly, carry);
        // With as many bytes as `y`, `x` has no more positions, and only the carry out of the
        // top is looked at.
        if ly < lx {
            let p;
            (p, carry) = carry_on::<BE, false>(r, x, ly, lx, carry);
            if !carry {
                copy_rest::<BE>(r, x, p, lx);
            }
        }
        debug_assert_eq!(len, lx + usize::from(carry), "the planned length is wrong");
        // A carry out of the top is a coin toss for random operands of the same length. The
        // top byte is chosen by the planned length rather than by `carry`: LLVM turns a select
        // on `carry` back into a branch, merged with the branches of `carry_on` on it. Without
        // a carry the byte under it is written again.
        let top = get::<BE>(r.cast_const(), lx - 1);
        set::<BE>(r, len - 1, select_unpredictable(len > lx, 1, top));
    }
}

/// Writes the `len` bytes of `x - y` to `r`, all least significant ends in the layout `BE`,
/// for a larger magnitude `x`, `y` of `ly` significant bytes, and a difference of `len`
/// significant bytes.
///
/// As [`add_into`] does, over the positions below `len` only: those of `y` above it do not
/// change the difference, and the borrow out of the top position, if any, is cancelled by the
/// positions of `x` above it.
///
/// # Safety
///
/// As for [`compute`].
#[inline(always)]
unsafe fn sub_into<const BE: bool>(r: *mut u8, x: *const u8, y: *const u8, ly: usize, len: usize) {
    // SAFETY: as the caller guarantees.
    unsafe { sub_below::<BE>(r, x, y, ly.min(len), len) }
}

/// The significant length from which a little-endian difference built in the vector of one of
/// its operands, both at least this long, is long (see [`difference_in`]): the length from which
/// the kernels run several carry chains.
const LONG_DIFFERENCE_LE: usize = 256;

/// [`LONG_DIFFERENCE_LE`] for big-endian operands.
const LONG_DIFFERENCE_BE: usize = 512;

/// [`sub_into`] for a long difference built in the vector of an operand, at the least
/// significant end `alias` in the layout `BE`, from the other operand at `other`: `x` is `alias`
/// and `y` is `other`, or the other way round with `SWAPPED`. `ly` is the significant length of
/// `y`, and `len` that of the difference.
///
/// The other sums take `x` and `y` without a branch, as which magnitude is the larger is a coin
/// toss for random operands of the same length. The kernel then cannot load a limb before the
/// plan has read and compared the top limbs of both operands: its addresses come from that
/// choice, and the number of limbs it takes from the length of the difference (`ly` is cut to
/// `len`), which the plan finds from those limbs too. Here both come from branches instead: `x`
/// from which of the two copies of this function is called, which LLVM cannot merge into one
/// call with chosen arguments as it can calls of one function, and the number of limbs from the
/// branch on `ly <= len`, a difference shorter than `y` taking the cold path. A predicted branch
/// lets the kernel start while the plan is still reading; a mispredicted one costs about what
/// the wait would have.
///
/// Measured in place on an Apple M4 Max as `a += &b` with opposite signs, over 512 pairs of
/// operands in the second-level cache: with a predictable larger operand, 12 cycles fewer at 256
/// bytes little-endian and 19 to 23 fewer from 320 up, and 7 to 23 fewer big-endian from 512;
/// with a random one, which the branch mispredicts half the time, 10 more at 256 bytes and 3 more
/// at 320 little-endian, but 11 to 13 fewer from 512, and 3 to 19 fewer big-endian from 512.
/// With the operands in the first-level cache, 1 to 24 cycles fewer from 256 bytes. Shorter, a
/// random larger operand cost 17 to 23 cycles more (64 to 192 bytes), and big-endian, whose
/// loop is bound by how many instructions the core issues, saved nothing at 256 and 320 bytes.
///
/// # Safety
///
/// As for [`compute`], with the operands of `x` and `y` as said.
#[inline(never)]
unsafe fn difference_in<const BE: bool, const SWAPPED: bool>(
    r: *mut u8,
    alias: *const u8,
    other: *const u8,
    ly: usize,
    len: usize,
) {
    let (x, y) = if SWAPPED {
        (other, alias)
    } else {
        (alias, other)
    };
    // SAFETY: as the caller guarantees; `sub_below` takes the positions of `y` below `len`.
    unsafe {
        if ly <= len {
            sub_below::<BE>(r, x, y, ly, len);
        } else {
            short_difference::<BE>(r, x, y, len);
        }
    }
}

/// [`sub_below`] for a long difference shorter than `y`, out of line and cold, so that the
/// common case takes the length of `y`, not one that depends on the bytes of the operands.
///
/// # Safety
///
/// As for [`sub_into`].
#[cold]
#[inline(never)]
unsafe fn short_difference<const BE: bool>(r: *mut u8, x: *const u8, y: *const u8, len: usize) {
    // SAFETY: as the caller guarantees.
    unsafe { sub_below::<BE>(r, x, y, len, len) }
}

/// [`sub_into`] for `ly <= len`: every position of `y` it takes is below the length of the
/// difference.
///
/// # Safety
///
/// As for [`sub_into`], with `ly <= len`.
#[inline(always)]
unsafe fn sub_below<const BE: bool>(r: *mut u8, x: *const u8, y: *const u8, ly: usize, len: usize) {
    let n = ly / 8;
    // SAFETY: as in `add_into`, with every position below `len`, which is at most the
    // significant length of `x`.
    unsafe {
        let borrow = sub_n::<BE>(r, x, y, n, false);
        let borrow = rest_of_y::<BE, true>(r, x, y, n, ly, borrow);
        if ly < len {
            let (p, _) = carry_on::<BE, true>(r, x, ly, len, borrow);
            copy_rest::<BE>(r, x, p, len);
        }
        debug_assert!(
            get::<BE>(r.cast_const(), len - 1) != 0,
            "the planned length is wrong"
        );
    }
}

/// Adds (subtracts, with `SUB`) the positions of `y` from `8 * n` up to `ly`, above the `n`
/// whole limbs the kernel took, with the carry (borrow) `carry` out of them; returns the
/// carry (borrow) out of position `ly - 1`.
///
/// From three of them, and with a whole limb below, they go as one partial limb: the eight
/// positions that end at `ly` are read from `x` and `y` and shifted down to the new ones,
/// and written back over the output's last whole limb, whose top bytes the kernel wrote and
/// which are read back for that. Below three, one at a time.
///
/// # Safety
///
/// As for [`add_into`] (for [`sub_into`] with `SUB`), with `n = ly / 8` and the kernel's
/// limbs written.
#[inline(always)]
unsafe fn rest_of_y<const BE: bool, const SUB: bool>(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    ly: usize,
    mut carry: bool,
) -> bool {
    let mut p = 8 * n;
    let rem = ly - p;
    if rem == 0 {
        return carry;
    }
    // SAFETY: positions `ly - 8..ly` are below `ly` and, with `n > 0`, at or above 0; those of
    // the output below `p` are the kernel's, written and so initialised. Every other position
    // is below `ly`.
    unsafe {
        if rem >= 3 && n > 0 {
            let shift = 8 * (8 - rem) as u32;
            let (xs, ys) = (
                load::<BE>(x, ly - 8) >> shift,
                load::<BE>(y, ly - 8) >> shift,
            );
            let c = u64::from(carry);
            let s = if SUB {
                xs.wrapping_sub(ys).wrapping_sub(c)
            } else {
                xs + ys + c
            };
            carry = (s >> (8 * rem)) & 1 != 0;
            let below = load::<BE>(r.cast_const(), p - 8) >> (8 * rem);
            store::<BE>(r, ly - 8, below | (s << shift));
        } else {
            while p < ly {
                let (xb, yb, c) = (get::<BE>(x, p), get::<BE>(y, p), u16::from(carry));
                let s = if SUB {
                    u16::from(xb).wrapping_sub(u16::from(yb)).wrapping_sub(c)
                } else {
                    u16::from(xb) + u16::from(yb) + c
                };
                set::<BE>(r, p, s as u8);
                carry = s >> 8 != 0;
                p += 1;
            }
        }
    }
    carry
}

/// Takes the carry (borrow, with `SUB`) `carry` on from position `p` through the positions of
/// `x` below `top` that pass it on, `0xFF` (`0x00`), writing them to `r`: a limb at a time,
/// then a byte at a time. Returns the position at which it stopped, `top` if it did not, and
/// whether a carry is left.
///
/// The carry out of `y` is a coin toss for random operands, so the first step takes it without
/// a branch: the limb at `p`, plus or minus the carry, or, with fewer than eight positions left
/// and a whole limb under `p`, those positions as one partial limb, read from below their start
/// and shifted, and written back over the output's limb under `p`, which is read back for that,
/// as in [`rest_of_y`]. Only a carry that gets through that step, whose limb passed it on,
/// goes further.
///
/// # Safety
///
/// As for [`add_into`] (for [`sub_into`] with `SUB`), with `p <= top` and the positions of the
/// output below `p` written.
#[inline(always)]
unsafe fn carry_on<const BE: bool, const SUB: bool>(
    r: *mut u8,
    x: *const u8,
    mut p: usize,
    top: usize,
    mut carry: bool,
) -> (usize, bool) {
    let passes = if SUB { 0 } else { u64::MAX };
    // SAFETY: every position read and written is below `top`; the partial limb reads positions
    // `top - 8..top` of `x`, at or above 0 as `top > p >= 8`, and `p - 8..p` of the output,
    // which are written.
    unsafe {
        let c = u64::from(carry);
        if top - p >= 8 {
            let v = load::<BE>(x, p);
            store::<BE>(
                r,
                p,
                if SUB {
                    v.wrapping_sub(c)
                } else {
                    v.wrapping_add(c)
                },
            );
            carry &= v == passes;
            p += 8;
        } else if p < top && p >= 8 {
            let m = top - p;
            let shift = 8 * (8 - m) as u32;
            let v = load::<BE>(x, top - 8) >> shift;
            let s = if SUB { v.wrapping_sub(c) } else { v + c };
            let below = load::<BE>(r.cast_const(), p - 8) >> (8 * m);
            store::<BE>(r, top - 8, below | (s << shift));
            return (top, (s >> (8 * m)) & 1 != 0);
        }
        while carry && top - p >= 8 {
            let v = load::<BE>(x, p);
            store::<BE>(
                r,
                p,
                if SUB {
                    v.wrapping_sub(1)
                } else {
                    v.wrapping_add(1)
                },
            );
            carry = v == passes;
            p += 8;
        }
        while carry && p < top {
            let v = get::<BE>(x, p);
            set::<BE>(
                r,
                p,
                if SUB {
                    v.wrapping_sub(1)
                } else {
                    v.wrapping_add(1)
                },
            );
            carry = v == passes as u8;
            p += 1;
        }
    }
    (p, carry)
}

/// Copies positions `p..top` of `x` to the output `r`, unless the output is `x` itself: from
/// 16 bytes with `ptr::copy`, and below that inline, with two loads and two stores of 8, 4 or
/// 2 bytes that overlap, or one of a byte, the loads first.
///
/// # Safety
///
/// The positions of `x` below `top` must be valid for reading, and those of `r` for writing;
/// the two may overlap.
#[inline(always)]
unsafe fn copy_rest<const BE: bool>(r: *mut u8, x: *const u8, p: usize, top: usize) {
    if p < top && !ptr::eq(r.cast_const(), x) {
        let m = top - p;
        // SAFETY: the `m` bytes from `src` and `dst` are positions `p..top` of `x` and `r`;
        // each load and store below lies within them, and the loads come first.
        unsafe {
            let (src, dst) = if BE {
                (x.sub(top), r.sub(top))
            } else {
                (x.add(p), r.add(p))
            };
            if m >= 16 {
                ptr::copy(src, dst, m);
            } else if m >= 8 {
                let (u, v) = (read::<8>(src), read::<8>(src.add(m - 8)));
                write::<8>(dst, u);
                write::<8>(dst.add(m - 8), v);
            } else if m >= 4 {
                let (u, v) = (read::<4>(src), read::<4>(src.add(m - 4)));
                write::<4>(dst, u);
                write::<4>(dst.add(m - 4), v);
            } else if m >= 2 {
                let (u, v) = (read::<2>(src), read::<2>(src.add(m - 2)));
                write::<2>(dst, u);
                write::<2>(dst.add(m - 2), v);
            } else {
                *dst = *src;
            }
        }
    }
}

/// The kernel `r = x + y + carry` over `n` limbs in the layout `BE`; returns the carry out.
///
/// # Safety
///
/// As for the kernels: `x` and `y` valid for reading and `r` for writing `n` limbs from their
/// least significant ends, the output apart from the inputs, equal to them, or, big-endian,
/// above them.
#[inline(always)]
unsafe fn add_n<const BE: bool>(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    carry: bool,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        if BE {
            add_n_be(r, x, y, n, carry)
        } else {
            add_n_le(r, x, y, n, carry)
        }
    }
}

/// The kernel `r = x - y - borrow` over `n` limbs in the layout `BE`; returns the borrow out.
///
/// # Safety
///
/// As for [`add_n`].
#[inline(always)]
unsafe fn sub_n<const BE: bool>(
    r: *mut u8,
    x: *const u8,
    y: *const u8,
    n: usize,
    borrow: bool,
) -> bool {
    // SAFETY: as the caller guarantees.
    unsafe {
        if BE {
            sub_n_be(r, x, y, n, borrow)
        } else {
            sub_n_le(r, x, y, n, borrow)
        }
    }
}

// Positions

/// The least significant end of the magnitude in `v` in the layout `BE`: the address of its
/// first byte little-endian, and the address one past its last byte big-endian.
#[inline(always)]
fn end<const BE: bool>(v: &[u8]) -> *const u8 {
    if BE { v.as_ptr_range().end } else { v.as_ptr() }
}

/// Reads position `p` of the magnitude at the least significant end `end` in the layout `BE`:
/// the byte at `end + p` little-endian, at `end - 1 - p` big-endian.
///
/// # Safety
///
/// The byte must be valid for reading.
#[inline(always)]
unsafe fn get<const BE: bool>(end: *const u8, p: usize) -> u8 {
    // SAFETY: as the caller guarantees.
    unsafe { if BE { *end.sub(p + 1) } else { *end.add(p) } }
}

/// Reads position `p` of the magnitude of `len` significant bytes at the least significant end
/// `end` in the layout `BE`, as [`get`] does, or 0 from `len` up.
///
/// # Safety
///
/// The significant bytes must be valid for reading.
#[inline(always)]
unsafe fn get_or_zero<const BE: bool>(end: *const u8, len: usize, p: usize) -> u8 {
    if p < len {
        // SAFETY: as the caller guarantees, with `p < len`.
        unsafe { get::<BE>(end, p) }
    } else {
        0
    }
}

/// Writes position `p` of the magnitude at the least significant end `end` in the layout
/// `BE`, as [`get`] reads it.
///
/// # Safety
///
/// The byte must be valid for writing.
#[inline(always)]
unsafe fn set<const BE: bool>(end: *mut u8, p: usize, v: u8) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if BE {
            *end.sub(p + 1) = v;
        } else {
            *end.add(p) = v;
        }
    }
}

/// Reads positions `p..p + 8` of the magnitude at the least significant end `end` in the
/// layout `BE` as a `u64` whose least significant byte is position `p`: the eight bytes from
/// `end + p` as a little-endian `u64`, or the eight bytes up to `end - p` as a big-endian one.
///
/// # Safety
///
/// The eight bytes must be valid for reading; they need no alignment.
#[inline(always)]
unsafe fn load<const BE: bool>(end: *const u8, p: usize) -> u64 {
    // SAFETY: as the caller guarantees; `[u8; 8]` is aligned at any address.
    unsafe {
        if BE {
            u64::from_be_bytes(end.sub(p + 8).cast::<[u8; 8]>().read())
        } else {
            u64::from_le_bytes(end.add(p).cast::<[u8; 8]>().read())
        }
    }
}

/// Writes positions `p..p + 8` of the magnitude at the least significant end `end` in the
/// layout `BE`, as [`load`] reads them.
///
/// # Safety
///
/// The eight bytes must be valid for writing; they need no alignment.
#[inline(always)]
unsafe fn store<const BE: bool>(end: *mut u8, p: usize, v: u64) {
    // SAFETY: as the caller guarantees; `[u8; 8]` is aligned at any address.
    unsafe {
        if BE {
            end.sub(p + 8).cast::<[u8; 8]>().write(v.to_be_bytes());
        } else {
            end.add(p).cast::<[u8; 8]>().write(v.to_le_bytes());
        }
    }
}

/// Copies the `m` bytes at `src` to `dst` in the reverse order, `dst[i] = src[m - 1 - i]`,
/// eight at a time while that many are left.
///
/// # Safety
///
/// `src` must be valid for reading and `dst` for writing `m` bytes, and they must not overlap.
#[inline(always)]
unsafe fn copy_reversed(dst: *mut u8, src: *const u8, m: usize) {
    let mut i = 0;
    // SAFETY: every byte read is below `src + m` and every byte written below `dst + m`.
    unsafe {
        while m - i >= 8 {
            let bytes = src.add(m - i - 8).cast::<[u8; 8]>().read();
            dst.add(i)
                .cast::<[u8; 8]>()
                .write(u64::from_be_bytes(bytes).to_le_bytes());
            i += 8;
        }
        while i < m {
            *dst.add(i) = *src.add(m - 1 - i);
            i += 1;
        }
    }
}

/// Reads the `N` bytes at `p`.
///
/// # Safety
///
/// The bytes must be valid for reading; they need no alignment.
#[inline(always)]
unsafe fn read<const N: usize>(p: *const u8) -> [u8; N] {
    // SAFETY: as the caller guarantees; `[u8; N]` is aligned at any address.
    unsafe { p.cast::<[u8; N]>().read() }
}

/// Writes the `N` bytes `v` at `p`.
///
/// # Safety
///
/// The bytes must be valid for writing; they need no alignment.
#[inline(always)]
unsafe fn write<const N: usize>(p: *mut u8, v: [u8; N]) {
    // SAFETY: as the caller guarantees; `[u8; N]` is aligned at any address.
    unsafe { p.cast::<[u8; N]>().write(v) }
}

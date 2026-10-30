//! A divisor prepared for dividing many numbers by it: [`Divisor`].
//!
//! Before it divides, a division reads its divisor into limbs, shifts them so that the top bit is
//! set, and computes the reciprocal of the top limbs, by which each limb of the quotient is
//! estimated. A [`Divisor`] does that once, and keeps the results: dividing by it skips those
//! steps, and the scan of the divisor's bytes for its length. That is most of the cost of
//! dividing a short number, and little of dividing a long one by a long divisor.

use super::add::{SMALL, read_small};
use super::div::{
    EUCLID, Medium, Normal, Short, TRUNCATED, WIDER, Want, by_medium, by_zero, compute_by,
    normalize, wide_division, wide_number, wider_division, working_memory,
};
use super::mul::{Operand, Product, STACK_LIMBS, WIDE, Wide, on_heap, on_stack, put_wide};
use super::{ByteOrder, Number};
use core::mem::MaybeUninit;
use core::ops::{Div, DivAssign, Rem, RemAssign};

/// A number prepared to divide others by: dividing by it gives what dividing by the number does,
/// with the same operators, but skips reading the divisor and computing its reciprocal each time.
///
/// It keeps the magnitude of the number in limbs of eight bytes, shifted so that the top bit is
/// set, the reciprocal of the top limbs and the sign. A number divided by it with `/`, `/=`, `%`
/// or `%=`, owned or borrowed, or by its methods [`div_rem`](Self::div_rem),
/// [`div_euclid`](Self::div_euclid) and [`rem_euclid`](Self::rem_euclid), gives the results that
/// dividing by the number gives, in the byte order of the dividend, in the shortest form, built in
/// the dividend's vector when the dividend is owned (which always has room, but for zero in an
/// empty vector) and otherwise in one new vector of exactly their length.
///
/// # Examples
///
/// ```
/// use a::math::numbers::{ByteOrder, Divisor, Number};
///
/// // Dividing many numbers by 10^19.
/// let d = Divisor::new(&Number::new_from_u64(10_000_000_000_000_000_000, ByteOrder::BigEndian));
/// let a = Number::new_from_u128(123 * 10_000_000_000_000_000_000 + 45, ByteOrder::LittleEndian);
/// assert_eq!((&a / &d).value, [123]);
/// assert_eq!((&a % &d).value, [45]);
///
/// let (q, r) = d.div_rem(&Number::new_from_i32(-7, ByteOrder::BigEndian));
/// assert!(q.value == [0] && r.value == [7] && r.is_negative());
/// ```
#[derive(Clone)]
pub struct Divisor {
    /// The limbs of the magnitude, shifted, the least significant first, in pairs, so that they
    /// start on 16 bytes, where the kernel of the division reads them best.
    limbs: Vec<Pair>,
    /// The number of limbs.
    n: usize,
    /// The shift that sets the top bit of the top limb.
    shift: u32,
    /// The reciprocal of the top limb for one limb, and of the top two otherwise.
    v: u64,
    /// Whether the divisor is negative.
    negative: bool,
    /// The magnitude and its preparation for the divisions in registers, when it fits in 16
    /// bytes.
    short: Option<Short>,
    /// The limbs as the division in registers by three or four of them takes them, when the
    /// magnitude has 17 to 32 bytes.
    medium: Option<Medium>,
}

/// Two limbs on 16 bytes: the elements of [`Divisor::limbs`].
#[derive(Clone, Copy)]
#[repr(align(16))]
struct Pair(#[expect(dead_code, reason = "read through pointers to its limbs")] [u64; 2]);

impl Divisor {
    /// Prepares `d` for dividing by it: reads its magnitude into limbs, shifted, and computes the
    /// reciprocal of the top ones. `d` may have any sign and byte order, and need not be in the
    /// shortest form.
    ///
    /// Takes O(n) time for `n` bytes, and allocates one vector of their limbs.
    ///
    /// # Panics
    ///
    /// When `d` is zero, as dividing by it would.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_i32(-3, ByteOrder::BigEndian));
    /// assert!(d.is_negative());
    /// let q = &Number::new_from_i32(100, ByteOrder::LittleEndian) / &d;
    /// assert_eq!(q.value, [33]);
    /// assert!(q.is_negative());
    /// ```
    #[track_caller]
    pub fn new(d: &Number) -> Divisor {
        let y = Operand::of(d);
        if y.significant == 0 {
            by_zero::<false>();
        }
        let n = y.significant.div_ceil(8);
        let mut limbs = vec![Pair([0; 2]); n.div_ceil(2)];
        // SAFETY: the pairs hold at least the `n` limbs, from an address on 16 bytes; `d` is
        // borrowed while it is read.
        let normal = unsafe { normalize(y, limbs.as_mut_ptr().cast()) };
        let short = (y.significant <= SMALL).then(|| {
            // SAFETY: the significant bytes of the magnitude are at most 16.
            let magnitude = unsafe {
                let start = if y.big_endian {
                    y.len - y.significant
                } else {
                    0
                };
                read_small(&d.value[start..start + y.significant], d.byte_order)
            };
            Short::of(magnitude, d.negative)
        });
        let medium = (y.significant > SMALL && y.significant <= WIDE).then(|| {
            // SAFETY: the pairs hold the three or four limbs.
            unsafe { Medium::normalized(normal.d, n, normal.shift, normal.v, d.negative) }
        });
        Divisor {
            limbs,
            n,
            shift: normal.shift,
            v: normal.v,
            negative: d.negative,
            short,
            medium,
        }
    }

    /// Returns whether the divisor is below zero.
    pub fn is_negative(&self) -> bool {
        self.negative
    }

    /// Returns the quotient and the remainder of `a` divided by this divisor, `(a / d, a % d)`,
    /// computed together, as [`Number::div_rem`] gives them: the quotient truncated toward zero,
    /// the remainder with the sign of `a`, both in the byte order of `a` and in the shortest form,
    /// each in one new vector of exactly its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_u16(1000, ByteOrder::BigEndian));
    /// let (q, r) = d.div_rem(&Number::new_from_u32(1_234_567, ByteOrder::BigEndian));
    /// assert_eq!(q.value, [0x04, 0xD2]);
    /// assert_eq!(r.value, [0x02, 0x37]);
    /// ```
    pub fn div_rem(&self, a: &Number) -> (Number, Number) {
        let order = a.byte_order;
        if let Some((q, r)) = self.in_registers::<TRUNCATED>(a) {
            return (wide_number(order, &q), wide_number(order, &r));
        }
        // SAFETY: `a` is borrowed until the results are computed.
        unsafe {
            self.divide::<TRUNCATED, _>(Operand::of(a), a.negative, Want::Both, |q, r| {
                (new_number(q, order), new_number(r, order))
            })
        }
    }

    /// Returns the quotient of Euclid's division of `a` by this divisor, as
    /// [`Number::div_euclid`] gives it, in one new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_i32(2, ByteOrder::BigEndian));
    /// let q = d.div_euclid(&Number::new_from_i32(-7, ByteOrder::BigEndian));
    /// assert_eq!(q.value, [4]);
    /// assert!(q.is_negative());
    /// ```
    pub fn div_euclid(&self, a: &Number) -> Number {
        self.one_new::<EUCLID, false>(a)
    }

    /// Returns the remainder of Euclid's division of `a` by this divisor, `a` modulo `|d|`, as
    /// [`Number::rem_euclid`] gives it, never negative, in one new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_i32(-2, ByteOrder::BigEndian));
    /// let r = d.rem_euclid(&Number::new_from_i32(-7, ByteOrder::BigEndian));
    /// assert_eq!(r.value, [1]);
    /// assert!(!r.is_negative());
    /// ```
    pub fn rem_euclid(&self, a: &Number) -> Number {
        self.one_new::<EUCLID, true>(a)
    }

    /// The divisor as the general path of the division reads it.
    #[inline(always)]
    fn normal(&self) -> Normal {
        Normal {
            d: self.limbs.as_ptr().cast(),
            n: self.n,
            shift: self.shift,
            v: self.v,
        }
    }

    /// The quotient and the remainder of `a`, truncated, or Euclid's with `EUCLIDEAN`, computed
    /// in registers when the divisor fits in 32 bytes and `a`'s vector in 64.
    #[inline(always)]
    fn in_registers<const EUCLIDEAN: bool>(&self, a: &Number) -> Option<(Wide, Wide)> {
        let len = a.value.len();
        // SAFETY: the dividend's vector holds at most the bytes each path reads.
        unsafe {
            if let Some(y) = self.short {
                if len <= SMALL {
                    Some(wide_division::<EUCLIDEAN, 2>(a, y))
                } else if len <= WIDE {
                    Some(wide_division::<EUCLIDEAN, 4>(a, y))
                } else if len <= WIDER {
                    Some(wider_division::<EUCLIDEAN>(a, y))
                } else {
                    None
                }
            } else if let Some(y) = self.medium
                && len <= WIDER
            {
                Some(by_medium::<EUCLIDEAN>(a, y))
            } else {
                None
            }
        }
    }

    /// `a / d` (`a % d` when `REM`; Euclid's with `EUCLIDEAN`) in one new vector.
    #[inline(never)]
    fn one_new<const EUCLIDEAN: bool, const REM: bool>(&self, a: &Number) -> Number {
        let order = a.byte_order;
        if let Some((q, r)) = self.in_registers::<EUCLIDEAN>(a) {
            return wide_number(order, if REM { &r } else { &q });
        }
        // SAFETY: `a` is borrowed until the result is computed.
        unsafe {
            self.divide::<EUCLIDEAN, _>(
                Operand::of(a),
                a.negative,
                Want::of(EUCLIDEAN, REM),
                |q, r| new_number(if REM { r } else { q }, order),
            )
        }
    }

    /// `a /= d` (`a %= d` when `REM`), in the vector of `a` when it has room.
    #[inline(never)]
    fn assign<const REM: bool>(&self, a: &mut Number) {
        let order = a.byte_order;
        if let Some((q, r)) = self.in_registers::<TRUNCATED>(a) {
            let w = if REM { &r } else { &q };
            put_wide(&mut a.value, order, w);
            a.negative = w.negative;
            return;
        }
        let (x, negative_x) = (Operand::of(a), a.negative);
        // SAFETY: `a`'s vector is written only once it is read into limbs.
        unsafe {
            self.divide::<TRUNCATED, _>(x, negative_x, Want::of(TRUNCATED, REM), |q, r| {
                let p = if REM { r } else { q };
                p.put_in(&mut a.value, order);
                a.negative = p.negative;
            });
        }
    }

    /// Divides the operand `x`, of sign `negative_x`, by this divisor on the general path and
    /// hands the results to `place`, as the division's `divide` does with a divisor it reads.
    ///
    /// # Safety
    ///
    /// `x` must describe a vector that stays valid for reads until `place` is called.
    #[inline(always)]
    unsafe fn divide<const EUCLIDEAN: bool, R>(
        &self,
        x: Operand,
        negative_x: bool,
        want: Want,
        place: impl FnOnce(&Product, &Product) -> R,
    ) -> R {
        let nx = x.significant.div_ceil(8).max(self.n);
        let need = working_memory(nx, self.n, want);
        let (y, negative_y) = (self.normal(), self.negative);
        let compute = |s: *mut MaybeUninit<u64>| {
            // SAFETY: the working memory holds `need` limbs; the rest as the caller guarantees.
            unsafe {
                compute_by::<EUCLIDEAN, R>(s.cast(), x, y, negative_x, negative_y, want, place)
            }
        };
        if need <= STACK_LIMBS {
            on_stack(compute)
        } else {
            on_heap(need, compute)
        }
    }
}

impl Div<&Divisor> for &Number {
    type Output = Number;

    /// Returns `self / d`, truncated toward zero, as dividing by the number `d` was made from
    /// does, in one new vector of exactly its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_u8(7, ByteOrder::BigEndian));
    /// let q = &Number::new_from_i32(-100, ByteOrder::LittleEndian) / &d;
    /// assert_eq!(q.value, [14]);
    /// assert!(q.is_negative());
    /// ```
    fn div(self, d: &Divisor) -> Number {
        d.one_new::<TRUNCATED, false>(self)
    }
}

impl Div<&Divisor> for Number {
    type Output = Number;

    /// Returns `self / d`, as `self /= d` does, in the vector of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_u8(16, ByteOrder::BigEndian));
    /// let a = Number::new_from_u32(0x1234_5678, ByteOrder::BigEndian);
    /// let pointer = a.value.as_ptr();
    /// let q = a / &d;
    /// assert_eq!(q.value, [0x01, 0x23, 0x45, 0x67]);
    /// assert_eq!(q.value.as_ptr(), pointer);
    /// ```
    fn div(mut self, d: &Divisor) -> Number {
        self /= d;
        self
    }
}

impl DivAssign<&Divisor> for Number {
    /// Divides `self` by `d`, truncating toward zero, keeping the byte order of `self`, in its
    /// vector, which has room for the quotient unless it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let ten = Divisor::new(&Number::new_from_u8(10, ByteOrder::BigEndian));
    /// let mut n = Number::new_from_u32(12_345, ByteOrder::LittleEndian);
    /// n /= &ten;
    /// n /= &ten;
    /// assert_eq!(n.value, [123]);
    /// ```
    fn div_assign(&mut self, d: &Divisor) {
        d.assign::<false>(self);
    }
}

impl Rem<&Divisor> for &Number {
    type Output = Number;

    /// Returns `self % d`, with the sign of `self`, as the remainder by the number `d` was made
    /// from, in one new vector of exactly its length.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_i8(-7, ByteOrder::BigEndian));
    /// let r = &Number::new_from_i32(100, ByteOrder::LittleEndian) % &d;
    /// assert_eq!(r.value, [2]);
    /// assert!(!r.is_negative());
    /// ```
    fn rem(self, d: &Divisor) -> Number {
        d.one_new::<TRUNCATED, true>(self)
    }
}

impl Rem<&Divisor> for Number {
    type Output = Number;

    /// Returns `self % d`, as `self %= d` does, in the vector of `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_u16(1000, ByteOrder::BigEndian));
    /// let r = Number::new_from_u32(1_234_567, ByteOrder::BigEndian) % &d;
    /// assert_eq!(r.value, [0x02, 0x37]);
    /// ```
    fn rem(mut self, d: &Divisor) -> Number {
        self %= d;
        self
    }
}

impl RemAssign<&Divisor> for Number {
    /// Sets `self` to `self % d`, keeping the byte order of `self`, in its vector, which has room
    /// for the remainder unless it is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Divisor, Number};
    ///
    /// let d = Divisor::new(&Number::new_from_u8(9, ByteOrder::BigEndian));
    /// let mut n = Number::new_from_i32(-100, ByteOrder::LittleEndian);
    /// n %= &d;
    /// assert_eq!(n.value, [1]);
    /// assert!(n.is_negative());
    /// ```
    fn rem_assign(&mut self, d: &Divisor) {
        d.assign::<true>(self);
    }
}

/// The result `p` as a number in one new vector of exactly its length, in the byte order `order`.
///
/// # Safety
///
/// As for [`Product::write`].
#[inline(always)]
unsafe fn new_number(p: &Product, order: ByteOrder) -> Number {
    let mut v = Vec::with_capacity(p.len);
    // SAFETY: the new vector has room for the `len` bytes, all of which are written before the
    // length is set.
    unsafe {
        p.write(v.as_mut_ptr(), order);
        v.set_len(p.len);
    }
    Number {
        value: v,
        negative: p.negative,
        byte_order: order,
    }
}

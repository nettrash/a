//! Powers of [`Number`]s: [`Number::pow`], a power by a `u32` exponent, and [`Number::pow_mod`], a
//! power modulo a number.
//!
//! A power is computed by binary exponentiation from the top bit of the exponent, a square for
//! each bit and a product by the base for each one, on limbs of 8 bytes with the multiplication's
//! squares and products. The zero bits at the low end of the base are taken out first and shifted
//! back in at the end, and the result's length is bounded before it starts, so that its working
//! memory is allocated once; a power of up to 64 bytes is computed in registers.
//!
//! A power modulo an odd number is computed in Montgomery's form ([`montgomery`]) by a sliding
//! window over the exponent's bits ([`window`]); modulo a power of two by the low halves of products
//! ([`low`]); and modulo any other even number `m = 2^k m'` as both, modulo `m'` and modulo `2^k`,
//! put together by the Chinese remainder theorem, as GMP's `mpz_powm` does.

// The kernel of Montgomery's reduction under the name `kernels`, chosen as those of the
// multiplication are: assembly on 64-bit little-endian aarch64 where the system allows unaligned
// accesses, and portable Rust elsewhere and under Miri.
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
#[path = "pow/aarch64.rs"]
mod kernels;
#[cfg(not(all(
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
)))]
#[path = "pow/portable.rs"]
mod kernels;
pub(super) mod low;
pub(super) mod montgomery;
pub(super) mod window;

use super::add::{SMALL, read_small, write_small};
use super::div::by_zero;
use super::div::limbs::{bdiv_q, binvert_limb, rshift};
use super::mul::limbs::{add_1, add_n, mul, mul_itch, sqr, sqr_itch, sub_1, sub_n};
use super::mul::{Operand, Product, STACK_LIMBS, on_heap, on_stack, to_limbs};
use super::{ByteOrder, Number};
use core::mem::MaybeUninit;
use core::ptr;
use low::{Low, Low2, LowN, mullo, mullo_itch};
use montgomery::{Mont1, Mont2, MontN, Montgomery, even, lshift, rem, rem_itch};
use window::{power, power_in_registers, window};

impl Number {
    /// Returns `self` raised to the power `exp`, in the byte order of `self`, in the shortest form,
    /// in one new vector of exactly its length.
    ///
    /// As for `i32::pow`, `x^0` is 1 for every `x`, zero included. The power is negative when
    /// `self` is negative and `exp` odd. The result is exact, however long: a power of `n` bytes
    /// by `exp` takes up to `n exp` bytes.
    ///
    /// # Panics
    ///
    /// When the power is too long to allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // (-3)^5 = -243 = -0xF3.
    /// let p = Number::new_from_i32(-3, ByteOrder::BigEndian).pow(5);
    /// assert_eq!(p.value, [0xF3]);
    /// assert!(p.is_negative());
    ///
    /// // 0x100^3 = 0x01_00_00_00, and anything to the power 0 is 1.
    /// let p = Number::new_from_u32(0x100, ByteOrder::LittleEndian).pow(3);
    /// assert_eq!(p.value, [0, 0, 0, 1]);
    /// assert_eq!(Number::new_from_u8(0, ByteOrder::BigEndian).pow(0).value, [1]);
    /// ```
    pub fn pow(&self, exp: u32) -> Number {
        let order = self.byte_order;
        if exp == 0 {
            return small_number(order, 1, false);
        }
        let negative = self.negative && exp & 1 == 1;
        if self.value.len() <= SMALL {
            // SAFETY: the vector holds at most 16 bytes.
            let x = unsafe { read_small(&self.value, order) };
            if x == 0 {
                return small_number(order, 0, false);
            }
            if let Some(p) = small_power(x, exp) {
                return small_number(order, p, negative);
            }
            let z = x.trailing_zeros();
            let o = x >> z;
            if o == 1 || u64::from(128 - o.leading_zeros()) * u64::from(exp) <= 512 {
                return register_power(
                    [o as u64, (o >> 64) as u64],
                    z as usize,
                    exp,
                    negative,
                    order,
                );
            }
        }
        let x = Operand::of(self);
        if x.significant == 0 {
            return small_number(order, 0, false);
        }
        power_of(x, exp, negative, order)
    }

    /// Returns `self` raised to the power `exp` modulo `modulus`: the `r` with `0 <= r < |modulus|`
    /// and `r = self^exp (mod |modulus|)`, Euclid's remainder, as `rem_euclid` gives it, in the
    /// byte order of `self`, in the shortest form, in one new vector of exactly its length.
    ///
    /// `self` may be negative, or longer than the modulus, and the modulus negative: only its
    /// magnitude counts. `x^0` is 1 for every `x`, zero included, so modulo 1, where every
    /// remainder is 0, `x^0 mod 1 = 0`.
    ///
    /// The time it takes depends on the exponent's bits, not only on its length: it is not a
    /// constant-time exponentiation, and is not meant for secret exponents (keys) where whoever
    /// could time it could learn them.
    ///
    /// # Panics
    ///
    /// When `modulus` is zero, with the message of `%`, and when `exp` is negative, with the message
    /// `negative exponent`: a negative power is a power of the inverse of `self` modulo `modulus`,
    /// which has to be computed first, with `mod_inverse`, and raised to `-exp`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 4^13 mod 497 = 445 = 0x01_BD.
    /// let b = Number::new_from_u32(4, ByteOrder::BigEndian);
    /// let e = Number::new_from_u32(13, ByteOrder::LittleEndian);
    /// let m = Number::new_from_u32(497, ByteOrder::BigEndian);
    /// assert_eq!(b.pow_mod(&e, &m).value, [0x01, 0xBD]);
    ///
    /// // (-4)^13 mod 497 = 497 - 445 = 52: the remainder is never negative.
    /// let b = Number::new_from_i32(-4, ByteOrder::BigEndian);
    /// let r = b.pow_mod(&e, &m);
    /// assert_eq!(r.value, [52]);
    /// assert!(!r.is_negative());
    /// ```
    #[track_caller]
    pub fn pow_mod(&self, exp: &Number, modulus: &Number) -> Number {
        let m = Operand::of(modulus);
        if m.significant == 0 {
            by_zero::<true>();
        }
        let e = Operand::of(exp);
        if exp.negative && e.significant != 0 {
            negative_exponent();
        }
        let order = self.byte_order;
        if self.value.len() <= SMALL && exp.value.len() <= SMALL && modulus.value.len() <= SMALL {
            // SAFETY: the vectors hold at most 16 bytes.
            let (b, e, m) = unsafe {
                (
                    read_small(&self.value, order),
                    read_small(&exp.value, exp.byte_order),
                    read_small(&modulus.value, modulus.byte_order),
                )
            };
            let mut r = small_pow_mod(b, e, m);
            if self.negative && e & 1 == 1 && r != 0 {
                r = m - r;
            }
            return small_number(order, r, false);
        }
        let b = Operand::of(self);
        pow_mod_of(b, self.negative, e, m, order)
    }
}

/// `b^e mod m` for `m` not zero, all in registers: modulo the odd part of `m` in Montgomery's
/// form, modulo its power of two by wrapping products, put together as [`pow_mod_in`] puts them.
#[inline(never)]
fn small_pow_mod(b: u128, e: u128, m: u128) -> u128 {
    if m == 1 {
        return 0;
    }
    if e == 0 {
        return 1;
    }
    if b == 0 {
        return 0;
    }
    let el = [e as u64, (e >> 64) as u64];
    let ebits = 128 - e.leading_zeros() as usize;
    let z = m.trailing_zeros();
    let odd = m >> z;
    let bl = [b as u64, (b >> 64) as u64];
    let bn = if bl[1] != 0 { 2 } else { 1 };
    // The power modulo the odd part, by the residue of the base.
    let r1 = if odd == 1 {
        0
    } else {
        let ol = [odd as u64, (odd >> 64) as u64];
        let mut s = [0u64; 16];
        // SAFETY: the dividend of at most 4 limbs by a divisor of 1 or 2 takes at most
        // `rem_itch(4, 2)` = 14 limbs of scratch, which `s` holds; the divisor's alignment is a
        // matter of speed for the long divisions only.
        unsafe {
            if ol[1] == 0 {
                let ring = Mont1::new(ol[0]);
                let mut x = 0;
                rem(&mut x, bl.as_ptr(), bn, 1, ol.as_ptr(), 1, s.as_mut_ptr());
                let y = power_in_registers(&ring, x, el.as_ptr(), ebits);
                u128::from(ring.plain(y))
            } else {
                let ring = Mont2::new(ol);
                let mut x = [0; 2];
                rem(
                    x.as_mut_ptr(),
                    bl.as_ptr(),
                    bn,
                    2,
                    ol.as_ptr(),
                    2,
                    s.as_mut_ptr(),
                );
                // SAFETY: the exponent's limbs hold its `ebits` bits.
                let y = power_in_registers(&ring, x, el.as_ptr(), ebits);
                let y = ring.plain(y);
                u128::from(y[0]) | (u128::from(y[1]) << 64)
            }
        }
    };
    if z == 0 {
        return r1;
    }
    // The power modulo `2^z`, `z < 128`: zero when the base's zero bits fill it.
    let mask = (1u128 << z) - 1;
    let v = b.trailing_zeros();
    let r2 = if v > 0
        && u128::from(v)
            .checked_mul(e)
            .is_none_or(|p| p >= u128::from(z))
    {
        0
    } else {
        // SAFETY: the exponent's limbs hold its `ebits` bits.
        unsafe { power_in_registers(&Low2, b, el.as_ptr(), ebits) & mask }
    };
    if odd == 1 {
        return r2;
    }
    // `r1 + m' ((r2 - r1) / m' mod 2^z)`, with `1 / m' mod 2^128` by Newton's steps.
    let mut inv = u128::from(binvert_limb(odd as u64));
    inv = inv.wrapping_mul(2u128.wrapping_sub(odd.wrapping_mul(inv)));
    let x = r2.wrapping_sub(r1).wrapping_mul(inv) & mask;
    r1 + odd * x
}

/// Panics for a negative exponent of [`Number::pow_mod`].
#[cold]
#[inline(never)]
#[track_caller]
fn negative_exponent() -> ! {
    panic!("negative exponent")
}

// Small powers

/// `x^exp` when it fits in 128 bits, which the bits of `x` show: its odd part `o = x / 2^z` to the
/// power `exp` is below `2^(exp bits(o))`. Binary exponentiation from the top bit, in registers.
#[inline]
fn small_power(x: u128, exp: u32) -> Option<u128> {
    let z = x.trailing_zeros();
    let o = x >> z;
    let bits = 128 - o.leading_zeros();
    if u64::from(bits + z) * u64::from(exp) > 128 {
        return None;
    }
    let mut p = o;
    for i in (0..31 - exp.leading_zeros()).rev() {
        p *= p;
        if exp >> i & 1 == 1 {
            p *= o;
        }
    }
    Some(p << (z * exp))
}

/// The number of magnitude `p` and sign `negative`, which zero does not keep, in one new vector of
/// exactly its length, in the byte order `order`.
fn small_number(order: ByteOrder, p: u128, negative: bool) -> Number {
    let len = 16 - (p | 1).leading_zeros() as usize / 8;
    let mut v = Vec::with_capacity(len);
    // SAFETY: the new vector has room for the `len` bytes, from 1 to 16, all of which are written
    // before the length is set.
    unsafe {
        write_small(v.as_mut_ptr(), order, p as u64, (p >> 64) as u64, len);
        v.set_len(len);
    }
    Number {
        value: v,
        negative: negative && p != 0,
        byte_order: order,
    }
}

// Long powers

/// The number of zero bits at the low end of the magnitude of the operand `x`, not zero, and its
/// number of bits.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
unsafe fn zeros_and_bits(x: Operand) -> (usize, usize) {
    let l = x.significant;
    // SAFETY: every byte read is one of the `l` significant ones: big-endian the last `l` of the
    // vector, little-endian the first `l`.
    unsafe {
        let byte = |i: usize| {
            if x.big_endian {
                *x.ptr.add(x.len - 1 - i)
            } else {
                *x.ptr.add(i)
            }
        };
        let mut i = 0;
        while byte(i) == 0 {
            i += 1;
        }
        let zeros = 8 * i + byte(i).trailing_zeros() as usize;
        (zeros, 8 * l - byte(l - 1).leading_zeros() as usize)
    }
}

/// The limbs of an odd power of `bits` bits by `exp`, a limb more for the shift that puts back the
/// base's low zero bits.
///
/// # Panics
///
/// When it does not fit in memory.
fn power_limbs(bits: usize, exp: u32) -> usize {
    let limbs = (bits as u128 * u128::from(exp)).div_ceil(64) + 1;
    match usize::try_from(limbs) {
        Ok(limbs) if limbs <= isize::MAX as usize / 32 => limbs,
        _ => too_long(),
    }
}

/// Panics for a power too long to allocate.
#[cold]
#[inline(never)]
fn too_long() -> ! {
    panic!("the power is too long to allocate")
}

/// `x^exp` for the operand `x`, not zero, with the sign `negative`, in the byte order `order`.
#[inline(never)]
fn power_of(x: Operand, exp: u32, negative: bool, order: ByteOrder) -> Number {
    let xn = x.significant.div_ceil(8);
    // SAFETY: the operand's vector is valid for reads, and its magnitude not zero.
    let (z, bits) = unsafe { zeros_and_bits(x) };
    let odd_bits = bits - z;
    // An odd part of 1 makes the power a shift, whatever the exponent: in registers too, rather
    // than in working memory sized for `exp` bits.
    if odd_bits == 1 || odd_bits <= 128 && odd_bits as u64 * u64::from(exp) <= 512 {
        // SAFETY: the operand's vector is valid for reads, and its odd part has at most 128 bits.
        return unsafe { short_power(x, z, exp, negative, order) };
    }
    let on = odd_bits.div_ceil(64);
    let rn = power_limbs(odd_bits, exp);
    if rn > STACK_LIMBS {
        // A power too long for the memory fails here, before the scratch of its squares is counted,
        // which takes long for lengths far beyond the memory.
        let mut v = Vec::<Pair>::new();
        if v.try_reserve_exact(rn + 1).is_err() {
            too_long();
        }
    }
    // The scratch of the squares and products, at the lengths the power reaches: none while they
    // are short enough for the schoolbook method (squares below 56 limbs, products by a base below
    // 24). The lengths are bounds: a square's top limb may be zero, and then the power is shorter,
    // and the scratch of a shorter length may be more (`sqr_itch` and `mul_itch` are not
    // monotonic: a square of 202 limbs takes 486 limbs, of 203 and 204 less), which `power_in`
    // allocates when it needs it.
    let mut scratch = 0;
    if rn >= 112 || on >= 24 {
        let mut len = on;
        for i in (0..31 - exp.leading_zeros()).rev() {
            scratch = scratch.max(sqr_itch(len));
            len = (2 * len).min(rn);
            if exp >> i & 1 == 1 {
                scratch = scratch.max(mul_itch(len.max(on), len.min(on)));
                len = (len + on).min(rn);
            }
        }
    }
    // The operand, its odd part, two buffers for the power, and the scratch.
    let need = even(xn) + even(on) + 2 * even(rn + 1) + scratch;
    let compute = |s: *mut MaybeUninit<u64>| {
        // SAFETY: the working memory holds `need` limbs, as the layout below takes them.
        unsafe { power_in(s.cast(), x, xn, z, on, exp, rn, scratch, negative, order) }
    };
    if need <= SHORT_LIMBS {
        let mut s = [PAIR; SHORT_LIMBS / 2];
        compute(s.as_mut_ptr().cast())
    } else if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap_or_panic(need, compute)
    }
}

/// The limbs of working memory up to which a power is computed in a buffer in the frame of its own
/// function, rather than in the multiplication's 8 KiB on the stack, whose frame costs a probe of
/// each of its pages.
const SHORT_LIMBS: usize = 96;

/// Two limbs of working memory, aligned to 16 bytes, as the multiplication's kernels want them.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Pair([MaybeUninit<u64>; 2]);

/// An uninitialised pair of limbs.
const PAIR: Pair = Pair([MaybeUninit::uninit(); 2]);

/// Runs `f` on `need` limbs of working memory on the heap, aligned to 16 bytes, freed on return;
/// panics when they cannot be allocated, as a power too long for the memory is a mistake of the
/// caller's.
#[cold]
#[inline(never)]
fn on_heap_or_panic<R>(need: usize, f: impl FnOnce(*mut MaybeUninit<u64>) -> R) -> R {
    let mut s = Vec::<Pair>::new();
    if s.try_reserve_exact(need.div_ceil(2)).is_err() {
        too_long();
    }
    f(s.as_mut_ptr().cast())
}

/// `x^exp` for the operand `x`, not zero, of `z` zero bits at its low end and an odd part of at most
/// 128 bits whose power has at most 512, or of 1, whose power is 1: in registers, four limbs for a
/// power of up to 256 bits and eight beyond.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
#[inline(never)]
unsafe fn short_power(x: Operand, z: usize, exp: u32, negative: bool, order: ByteOrder) -> Number {
    // The odd part, shifted down from its limbs, which fits in two.
    let (zl, zb) = (z / 64, (z % 64) as u32);
    // SAFETY: as the caller guarantees.
    let l = |i: usize| unsafe { limb_at(x, zl + i) };
    let (l0, l1, l2) = (l(0), l(1), l(2));
    let o = if zb == 0 {
        [l0, l1]
    } else {
        [
            (l0 >> zb) | (l1 << (64 - zb)),
            (l1 >> zb) | (l2 << (64 - zb)),
        ]
    };
    register_power(o, z, exp, negative, order)
}

/// `(o 2^z)^exp` for the odd `o` of two limbs, whose power has at most 512 bits: in registers, four
/// limbs for a power of up to 256 bits and eight beyond, shifted by `z exp` bits as it is written.
#[inline(never)]
fn register_power(o: [u64; 2], z: usize, exp: u32, negative: bool, order: ByteOrder) -> Number {
    let bits = 128
        - if o[1] != 0 {
            o[1].leading_zeros()
        } else {
            64 + o[0].leading_zeros()
        };
    let shift = z as u128 * u128::from(exp);
    // SAFETY: the limbs hold the odd power, with a limb of room above it.
    unsafe {
        if bits == 1 || bits as u64 * u64::from(exp) <= 256 {
            let mut r = odd_power::<4, 5>(o, exp);
            shifted_number(r.as_mut_ptr(), len_of(&r[..4]), shift, negative, order)
        } else {
            let mut r = odd_power::<8, 9>(o, exp);
            shifted_number(r.as_mut_ptr(), len_of(&r[..8]), shift, negative, order)
        }
    }
}

/// The limbs of `r` without the zero limbs at its top.
#[inline(always)]
fn len_of(r: &[u64]) -> usize {
    r.len() - r.iter().rev().take_while(|&&l| l == 0).count()
}

/// `o^exp` in `N` limbs, which hold it, and one more of room, `M = N + 1`: binary exponentiation
/// from the top bit on the limbs the power has so far, a square and, for a one bit, a product by
/// the base's one or two limbs each step; no limb of a square or a product reaches past the power's
/// own `N`.
#[inline(always)]
fn odd_power<const N: usize, const M: usize>(o: [u64; 2], exp: u32) -> [u64; M] {
    let ol = if o[1] != 0 { 2 } else { 1 };
    let mut p = [0u64; N];
    p[0] = o[0];
    p[1] = o[1];
    let mut len = ol;
    for i in (0..31 - exp.leading_zeros()).rev() {
        // The square, `2 len` limbs or one fewer, below `N` limbs as the power is.
        let mut r = [0u64; N];
        for a in 0..len {
            let mut carry = 0u64;
            for b in 0..len {
                let t =
                    u128::from(p[a]) * u128::from(p[b]) + u128::from(r[a + b]) + u128::from(carry);
                r[a + b] = t as u64;
                carry = (t >> 64) as u64;
            }
            if a + len < N {
                r[a + len] = carry;
            }
        }
        len = (2 * len).min(N);
        if r[len - 1] == 0 {
            len -= 1;
        }
        p = r;
        if exp >> i & 1 == 1 {
            let mut r = [0u64; N];
            for (a, &oa) in o.iter().enumerate().take(ol) {
                let mut carry = 0u64;
                for b in 0..len.min(N - a) {
                    let t = u128::from(oa) * u128::from(p[b])
                        + u128::from(r[a + b])
                        + u128::from(carry);
                    r[a + b] = t as u64;
                    carry = (t >> 64) as u64;
                }
                if a + len < N {
                    r[a + len] = carry;
                }
            }
            len = (len + ol).min(N);
            if r[len - 1] == 0 {
                len -= 1;
            }
            p = r;
        }
    }
    let mut r = [0u64; M];
    r[..N].copy_from_slice(&p);
    r
}

/// Limb `i` of the magnitude of the operand `x`, zero above its significant bytes.
///
/// # Safety
///
/// The operand's vector must be valid for reads.
#[inline(always)]
unsafe fn limb_at(x: Operand, i: usize) -> u64 {
    let mut l = 0;
    for k in (0..8).rev() {
        let p = 8 * i + k;
        let byte = if p >= x.significant {
            0
        } else if x.big_endian {
            // SAFETY: byte `p` of the magnitude lies inside the significant bytes.
            unsafe { *x.ptr.add(x.len - 1 - p) }
        } else {
            // SAFETY: as above.
            unsafe { *x.ptr.add(p) }
        };
        l = (l << 8) | u64::from(byte);
    }
    l
}

/// The number of the odd power in the `len` limbs at `r`, the top one not zero, shifted left by
/// `shift` bits, with the sign `negative`, in one new vector of exactly its length.
///
/// # Safety
///
/// The limbs must be valid for reads and writes, with one more above them.
unsafe fn shifted_number(
    r: *mut u64,
    mut len: usize,
    shift: u128,
    negative: bool,
    order: ByteOrder,
) -> Number {
    // SAFETY: as the caller guarantees.
    unsafe {
        // The zero bits of the power: whole bytes, and the bits below a byte, which shift the limbs
        // of the odd power.
        let bytes = usize::try_from(shift / 8).unwrap_or_else(|_| too_long());
        *r.add(len) = lshift(r, r, len, (shift % 8) as u32);
        if *r.add(len) != 0 {
            len += 1;
        }
        let odd_len = 8 * len - (*r.add(len - 1)).leading_zeros() as usize / 8;
        let total = odd_len
            .checked_add(bytes)
            .filter(|&t| t <= isize::MAX as usize)
            .unwrap_or_else(|| too_long());
        let mut v: Vec<u8> = Vec::new();
        if v.try_reserve_exact(total).is_err() {
            too_long();
        }
        let dst = v.as_mut_ptr();
        let product = Product {
            r,
            len: odd_len,
            negative,
        };
        if order == ByteOrder::BigEndian {
            product.write(dst, order);
            ptr::write_bytes(dst.add(odd_len), 0, bytes);
        } else {
            ptr::write_bytes(dst, 0, bytes);
            product.write(dst.add(bytes), order);
        }
        v.set_len(total);
        Number {
            value: v,
            negative,
            byte_order: order,
        }
    }
}

/// [`power_of`] in the working memory at `s`, for the operand `x` of `xn` limbs and `z` zero bits
/// at its low end, whose odd part has `on` limbs; `room` limbs of it are for the scratch of the
/// squares and products, and one that needs more takes a buffer of its own.
///
/// # Safety
///
/// The working memory holds the limbs that `power_of` counts, and starts on 16 bytes; the
/// operand's vector is valid for reads, and its magnitude not zero.
#[allow(clippy::too_many_arguments)]
unsafe fn power_in(
    s: *mut u64,
    x: Operand,
    xn: usize,
    z: usize,
    on: usize,
    exp: u32,
    rn: usize,
    room: usize,
    negative: bool,
    order: ByteOrder,
) -> Number {
    // SAFETY: as the caller guarantees: every buffer lies in the working memory, apart, and every
    // length stays within its buffer, as `power_of` bounds them.
    unsafe {
        let a = s;
        let o = a.add(even(xn));
        let p = o.add(even(on));
        let q = p.add(even(rn + 1));
        let t = q.add(even(rn + 1));
        to_limbs(x, a);
        // The odd part.
        let (zl, zb) = (z / 64, (z % 64) as u32);
        if zb != 0 {
            rshift(a.add(zl), xn - zl, zb);
        }
        ptr::copy_nonoverlapping(a.add(zl), o, on);
        // Binary exponentiation from the top bit, the power going back and forth between `p` and
        // `q`.
        let (mut r, mut w) = (p, q);
        ptr::copy_nonoverlapping(o, r, on);
        let mut len = on;
        for i in (0..31 - exp.leading_zeros()).rev() {
            with_room(t, room, sqr_itch(len), |t| sqr(w, r, len, t));
            len *= 2;
            if *w.add(len - 1) == 0 {
                len -= 1;
            }
            (r, w) = (w, r);
            if exp >> i & 1 == 1 {
                let itch = mul_itch(len.max(on), len.min(on));
                with_room(t, room, itch, |t| {
                    if len >= on {
                        mul(w, r, len, o, on, t);
                    } else {
                        mul(w, o, on, r, len, t);
                    }
                });
                len += on;
                if *w.add(len - 1) == 0 {
                    len -= 1;
                }
                (r, w) = (w, r);
            }
        }
        shifted_number(r, len, z as u128 * u128::from(exp), negative, order)
    }
}

/// Runs `f` on scratch of `need` limbs: the `room` limbs at `t` when they are enough, and otherwise a
/// buffer of its own, on 16 bytes, freed on return.
///
/// # Safety
///
/// The `room` limbs at `t` must be valid for reads and writes.
unsafe fn with_room<R>(t: *mut u64, room: usize, need: usize, f: impl FnOnce(*mut u64) -> R) -> R {
    if need <= room {
        f(t)
    } else {
        more_room(need, f)
    }
}

/// The buffer of [`with_room`] when the planned scratch is not enough. Out of line and cold.
#[cold]
#[inline(never)]
fn more_room<R>(need: usize, f: impl FnOnce(*mut u64) -> R) -> R {
    let mut s = Vec::<Pair>::with_capacity(need.div_ceil(2));
    f(s.as_mut_ptr().cast())
}

// Powers modulo a number

/// The shape of a modulus `m = 2^z m'`, `m'` odd, as [`pow_mod_in`] takes it apart.
#[derive(Clone, Copy)]
struct Shape {
    /// The limbs of `m`.
    mn: usize,
    /// The zero bits at its low end.
    z: usize,
    /// The limbs of `m'`.
    nodd: usize,
    /// The limbs of `2^z - 1`, the residues modulo `2^z`.
    kn: usize,
}

impl Shape {
    /// The shape of the modulus `m`, not zero.
    ///
    /// # Safety
    ///
    /// The operand's vector must be valid for reads.
    unsafe fn of(m: Operand) -> Self {
        // SAFETY: as the caller guarantees.
        let (z, bits) = unsafe { zeros_and_bits(m) };
        Shape {
            mn: m.significant.div_ceil(8),
            z,
            nodd: (bits - z).div_ceil(64),
            kn: z.div_ceil(64),
        }
    }

    /// Whether `m'` is 1, so that `m` is a power of two.
    fn power_of_two(&self, bits: usize) -> bool {
        bits == self.z + 1
    }
}

/// `b^e mod |m|` for the operands `b`, of sign `negative_b`, `e` and `m`, not zero, in the byte
/// order `order`.
#[inline(never)]
fn pow_mod_of(b: Operand, negative_b: bool, e: Operand, m: Operand, order: ByteOrder) -> Number {
    let (bn, en) = (b.significant.div_ceil(8), e.significant.div_ceil(8));
    // SAFETY: the modulus's vector is valid for reads, and its magnitude not zero.
    let shape = unsafe { Shape::of(m) };
    // The exponent's bits, for the window: an upper bound from its bytes, as the table is sized
    // for it.
    let ebits = 8 * e.significant;
    let need = pow_mod_itch(bn, en, ebits, shape);
    let compute = |s: *mut MaybeUninit<u64>| {
        // SAFETY: the working memory holds `need` limbs.
        unsafe { pow_mod_in(s.cast(), b, negative_b, e, m, shape, order) }
    };
    if need <= STACK_LIMBS {
        on_stack(compute)
    } else {
        on_heap(need, compute)
    }
}

/// The limbs of working memory of [`pow_mod_in`] for a base of `bn` limbs, an exponent of `en`
/// limbs and at most `ebits` bits, and a modulus of the shape `shape`.
fn pow_mod_itch(bn: usize, en: usize, ebits: usize, shape: Shape) -> usize {
    let Shape { mn, nodd, kn, .. } = shape;
    let inputs = even(bn) + even(en) + 3 * even(mn + 1);
    let mut work = rem_itch(bn, mn);
    if kn == 0 {
        work = work.max(powm_odd_itch(bn, ebits, nodd));
    } else {
        // The power modulo `2^z`, the inverse of `m'` and Hensel's numerator for it, the
        // difference and the product of `m'` by the correction.
        let crt = 4 * even(kn) + even(nodd + kn + 1);
        let products = mullo_itch(kn).max(mul_itch(nodd.max(kn), nodd.min(kn)));
        let rest = powm_odd_itch(bn, ebits, nodd)
            .max(powlo_itch(ebits, kn))
            .max(products);
        work = work.max(crt + rest);
    }
    inputs + work
}

/// [`pow_mod_of`] in the working memory at `s`.
///
/// # Safety
///
/// The working memory holds [`pow_mod_itch`] limbs and starts on 16 bytes; the operands' vectors
/// are valid for reads, the modulus is not zero, and `shape` is its shape.
unsafe fn pow_mod_in(
    s: *mut u64,
    b: Operand,
    negative_b: bool,
    e: Operand,
    m: Operand,
    shape: Shape,
    order: ByteOrder,
) -> Number {
    let (bn, en) = (b.significant.div_ceil(8), e.significant.div_ceil(8));
    let Shape { mn, z, nodd, kn } = shape;
    // SAFETY: every buffer lies in the working memory that `pow_mod_itch` counts, apart from the
    // others, and starts on 16 bytes.
    unsafe {
        let bp = s;
        let ep = bp.add(even(bn));
        let mp = ep.add(even(en));
        let odd = mp.add(even(mn + 1));
        let r = odd.add(even(mn + 1));
        let t = r.add(even(mn + 1));
        to_limbs(m, mp);
        let one = mn == 1 && *mp == 1;
        if en == 0 {
            // `x^0 = 1`, which is 0 modulo 1.
            return small_number(order, u128::from(!one), false);
        }
        if bn == 0 || one {
            return small_number(order, 0, false);
        }
        to_limbs(e, ep);
        to_limbs(b, bp);
        let ebits = 64 * en - (*ep.add(en - 1)).leading_zeros() as usize;
        let mbits = 64 * mn - (*mp.add(mn - 1)).leading_zeros() as usize;
        if ebits == 1 {
            // `b^1`: the remainder alone.
            rem(r, bp, bn, 0, mp, mn, t);
        } else if z == 0 {
            powm_odd(r, bp, bn, ep, ebits, mp, mn, t);
        } else if shape.power_of_two(mbits) {
            powlo(r, bp, bn, ep, en, ebits, z, t);
            ptr::write_bytes(r.add(kn), 0, mn - kn);
        } else {
            // `m'`, then both powers, put together: `r = r1 + m' ((r2 - r1) / m' mod 2^z)`, below
            // `m`.
            let (zl, zb) = (z / 64, (z % 64) as u32);
            ptr::copy_nonoverlapping(mp.add(zl), odd, mn - zl);
            if zb != 0 {
                rshift(odd, mn - zl, zb);
            }
            let r2 = t;
            let inv = r2.add(even(kn));
            let num = inv.add(even(kn));
            let d = num.add(even(kn));
            let y = d.add(even(kn));
            let rest = y.add(even(nodd + kn + 1));
            powm_odd(r, bp, bn, ep, ebits, odd, nodd, rest);
            powlo(r2, bp, bn, ep, en, ebits, z, rest);
            ptr::write_bytes(num, 0, kn);
            *num = 1;
            bdiv_q(inv, num, kn, odd, nodd, binvert_limb(*odd));
            // `r2 - r1` modulo `2^(64 kn)`, `r1` cut or extended to `kn` limbs.
            if nodd >= kn {
                sub_n(d, r2, r, kn);
            } else {
                let borrow = sub_n(d, r2, r, nodd);
                ptr::copy_nonoverlapping(r2.add(nodd), d.add(nodd), kn - nodd);
                sub_1(d.add(nodd), kn - nodd, u64::from(borrow));
            }
            let x = num;
            mullo(x, d, inv, kn, rest);
            if zb != 0 {
                *x.add(kn - 1) &= (1 << zb) - 1;
            }
            if nodd >= kn {
                mul(y, odd, nodd, x, kn, rest);
            } else {
                mul(y, x, kn, odd, nodd, rest);
            }
            // `r1` and the correction, below `m`, so its limbs from `mn` up are zero.
            let carry = add_n(y, y, r, nodd);
            add_1(y.add(nodd), kn, u64::from(carry));
            ptr::copy_nonoverlapping(y, r, mn);
        }
        // A negative base to an odd power: `|m| - r` unless `r` is zero.
        let mut top = mn;
        while top > 0 && *r.add(top - 1) == 0 {
            top -= 1;
        }
        if negative_b && *ep & 1 == 1 && top != 0 {
            sub_n(r, mp, r, mn);
            top = mn;
            while top > 0 && *r.add(top - 1) == 0 {
                top -= 1;
            }
        }
        number_of(r, top, order)
    }
}

/// The non-negative number of the `n` limbs at `r`, the top one not zero (none for zero), in one
/// new vector of exactly its length, in the byte order `order`.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn number_of(r: *const u64, n: usize, order: ByteOrder) -> Number {
    // SAFETY: as the caller guarantees.
    unsafe {
        if n == 0 {
            return small_number(order, 0, false);
        }
        let len = 8 * n - (*r.add(n - 1)).leading_zeros() as usize / 8;
        let mut v = Vec::with_capacity(len);
        Product {
            r,
            len,
            negative: false,
        }
        .write(v.as_mut_ptr(), order);
        v.set_len(len);
        Number {
            value: v,
            negative: false,
            byte_order: order,
        }
    }
}

/// The limbs of scratch space of [`powm_odd`] for a base of `bn` limbs, an exponent of at most
/// `ebits` bits and a modulus of `n` limbs.
fn powm_odd_itch(bn: usize, ebits: usize, n: usize) -> usize {
    if n <= 2 {
        rem_itch(bn + n, n)
    } else {
        let table = (1 << (window(ebits) - 1)) * even(n);
        Montgomery::new_itch(n) + table + Montgomery::itch(n).max(rem_itch(bn + n, n))
    }
}

/// Sets the `n` limbs at `r` to `b^e mod m` for the `bn` limbs at `b`, not zero, the exponent of
/// `ebits` bits at `e`, at least 2, and the odd `m` of `n` limbs at `m`, above 1, its top limb not
/// zero.
///
/// # Safety
///
/// The limbs must be valid, apart; the scratch at `s` holds [`powm_odd_itch`] limbs and starts on
/// 16 bytes.
#[allow(clippy::too_many_arguments)]
unsafe fn powm_odd(
    r: *mut u64,
    b: *const u64,
    bn: usize,
    e: *const u64,
    ebits: usize,
    m: *const u64,
    n: usize,
    s: *mut u64,
) {
    // SAFETY: as the caller guarantees; the residue of the base goes to the first entry of the
    // table, and the working memory of the Montgomery arithmetic before it.
    unsafe {
        match n {
            1 => {
                let ring = Mont1::new(*m);
                let mut x = 0;
                rem(&mut x, b, bn, 1, m, 1, s);
                let y = power_in_registers(&ring, x, e, ebits);
                *r = ring.plain(y);
            }
            2 => {
                let ring = Mont2::new([*m, *m.add(1)]);
                let mut x = [0; 2];
                rem(x.as_mut_ptr(), b, bn, 2, m, 2, s);
                let y = power_in_registers(&ring, x, e, ebits);
                let z = ring.plain(y);
                (*r, *r.add(1)) = (z[0], z[1]);
            }
            // Three to seven limbs in registers: measured on an Apple M4 Max with exponents as
            // long as the modulus, 6 limbs took 54,300 cycles so against 62,700 in limbs, 7 limbs
            // 83,800 against 92,400, and 8 limbs 123,000 against 117,000.
            3 => powm_registers::<3>(r, b, bn, e, ebits, m, s),
            4 => powm_registers::<4>(r, b, bn, e, ebits, m, s),
            5 => powm_registers::<5>(r, b, bn, e, ebits, m, s),
            6 => powm_registers::<6>(r, b, bn, e, ebits, m, s),
            7 => powm_registers::<7>(r, b, bn, e, ebits, m, s),
            _ => {
                let ring = Montgomery::new(m, n, s);
                let table = s.add(Montgomery::new_itch(n));
                let stride = even(n);
                let t = table.add((1 << (window(ebits) - 1)) * stride);
                rem(table, b, bn, n, m, n, t);
                power(&ring, r, e, ebits, table, stride, t);
                ring.plain(r, r, t);
            }
        }
    }
}

/// [`powm_odd`] for a modulus of `N` limbs, in registers.
///
/// # Safety
///
/// As for [`powm_odd`].
#[inline(never)]
unsafe fn powm_registers<const N: usize>(
    r: *mut u64,
    b: *const u64,
    bn: usize,
    e: *const u64,
    ebits: usize,
    m: *const u64,
    s: *mut u64,
) {
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut mm = [0; N];
        ptr::copy_nonoverlapping(m, mm.as_mut_ptr(), N);
        let ring = MontN::new(mm);
        let mut x = [0; N];
        rem(x.as_mut_ptr(), b, bn, N, m, N, s);
        let y = power_in_registers(&ring, x, e, ebits);
        let z = ring.plain(y);
        ptr::copy_nonoverlapping(z.as_ptr(), r, N);
    }
}

/// The limbs of scratch space of [`powlo`] for an exponent of at most `ebits` bits and a modulus of
/// `n` limbs.
fn powlo_itch(ebits: usize, n: usize) -> usize {
    (1 << (window(ebits) - 1)) * even(n) + Low::itch(n)
}

/// Sets the `z.div_ceil(64)` limbs at `r` to `b^e mod 2^z` for the `bn` limbs at `b`, not zero, the
/// exponent of `en` limbs and `ebits` bits at `e`, at least 2, and `z >= 1`.
///
/// An even base `b = 2^v b'` gives 0 when `v e >= z`; otherwise the power is taken by the low halves
/// of products, in registers up to two limbs.
///
/// # Safety
///
/// The limbs must be valid, apart; the scratch at `s` holds [`powlo_itch`] limbs and starts on 16
/// bytes.
#[allow(clippy::too_many_arguments)]
unsafe fn powlo(
    r: *mut u64,
    b: *const u64,
    bn: usize,
    e: *const u64,
    en: usize,
    ebits: usize,
    z: usize,
    s: *mut u64,
) {
    let kn = z.div_ceil(64);
    // SAFETY: as the caller guarantees.
    unsafe {
        let mut v = 0;
        while *b.add(v / 64) == 0 {
            v += 64;
        }
        v += (*b.add(v / 64)).trailing_zeros() as usize;
        if v > 0 && (en > 1 || u128::from(*e) * v as u128 >= z as u128) {
            ptr::write_bytes(r, 0, kn);
            return;
        }
        if kn <= 2 {
            let x = u128::from(*b)
                | if bn > 1 {
                    u128::from(*b.add(1)) << 64
                } else {
                    0
                };
            let y = power_in_registers(&Low2, x, e, ebits);
            let y = if z < 128 { y & ((1 << z) - 1) } else { y };
            *r = y as u64;
            if kn == 2 {
                *r.add(1) = (y >> 64) as u64;
            }
            return;
        }
        match kn {
            3 => return low_registers::<3>(r, b, bn, e, ebits, z),
            4 => return low_registers::<4>(r, b, bn, e, ebits, z),
            5 => return low_registers::<5>(r, b, bn, e, ebits, z),
            6 => return low_registers::<6>(r, b, bn, e, ebits, z),
            7 => return low_registers::<7>(r, b, bn, e, ebits, z),
            8 => return low_registers::<8>(r, b, bn, e, ebits, z),
            _ => {}
        }
        let table = s;
        let stride = even(kn);
        let t = table.add((1 << (window(ebits) - 1)) * stride);
        let have = bn.min(kn);
        ptr::copy_nonoverlapping(b, table, have);
        ptr::write_bytes(table.add(have), 0, kn - have);
        power(&Low { n: kn }, r, e, ebits, table, stride, t);
        if !z.is_multiple_of(64) {
            *r.add(kn - 1) &= (1 << (z % 64)) - 1;
        }
    }
}

/// [`powlo`] for `N = z.div_ceil(64)` limbs from 3 to 8, in registers.
///
/// # Safety
///
/// As for [`powlo`].
#[inline(never)]
unsafe fn low_registers<const N: usize>(
    r: *mut u64,
    b: *const u64,
    bn: usize,
    e: *const u64,
    ebits: usize,
    z: usize,
) {
    let mut x = [0u64; N];
    // SAFETY: as the caller guarantees: the base's low limbs, the exponent's bits and the `N`
    // limbs of the result.
    unsafe {
        ptr::copy_nonoverlapping(b, x.as_mut_ptr(), bn.min(N));
        let mut y = power_in_registers(&LowN::<N>, x, e, ebits);
        if !z.is_multiple_of(64) {
            y[N - 1] &= (1 << (z % 64)) - 1;
        }
        ptr::copy_nonoverlapping(y.as_ptr(), r, N);
    }
}

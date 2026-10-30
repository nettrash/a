//! Factorials and binomial coefficients: [`Number::factorial`] and [`Number::binomial`].
//!
//! Factorials of up to 34, and binomial coefficients below `2^128`, are computed in registers: the
//! former from a table, the latter as the product of the odd parts of their factors modulo
//! `2^128`, times a tabulated inverse (see [`binomial`]). Longer ones are computed on limbs of
//! eight bytes, native `u64`s, the least significant first: many small factors are multiplied
//! together into each limb first, and the limbs then by balanced product trees on the
//! multiplication, so that the longest products take Karatsuba's and Toom-Cook's methods (see
//! [`product`]).
//!
//! `n!` is `2^(n - popcount(n))` times its odd part, which Luschny's method computes from the
//! prime factors of the swinging factorials (see [`odd`]), as GMP's `mpz_fac_ui` does. A binomial
//! coefficient `C(n, k)` with a small `k` is a product of `k` consecutive numbers divided exactly
//! by the odd part of `k!`; with a large one, it is the product of its prime factors, whose
//! exponents Kummer's theorem gives, as GMP's `mpz_bin_uiui` does (see [`binomial`]). Either way
//! the power of two is counted apart, and the result is written shifted, in the byte order asked
//! for, in the shortest form.

mod binomial;
mod odd;
mod primes;
mod product;

use super::mul::Product;
use super::root::{lshift, small_number};
use super::{ByteOrder, Number};

/// The limbs of a magnitude, the least significant first, without zero limbs at the top.
type Limbs = Vec<u64>;

/// `n!` for `n` up to 34, the last below `2^128`.
const FACTORIALS: [u128; 35] = {
    let mut t = [1u128; 35];
    let mut i = 1;
    while i < 35 {
        t[i] = t[i - 1] * i as u128;
        i += 1;
    }
    t
};

impl Number {
    /// Returns `n!`, the product of the integers from 1 to `n`, in the byte order `byte_order`, in
    /// the shortest form, in one new vector; `0!` is 1.
    ///
    /// Up to 34, from a table; beyond, `2^(n - popcount(n))` times the odd part of `n!`, computed
    /// by Luschny's method from the prime factors of the swinging factorials, with balanced
    /// product trees on the multiplication, as GMP's `mpz_fac_ui` does.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector: with
    /// `capacity overflow` beyond `isize::MAX` bytes, and otherwise as the allocator fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::factorial(5, ByteOrder::BigEndian).value, [120]);
    /// // 25! = 15,511,210,043,330,985,984,000,000 = 0x0C_D4A0_619F_B090_7BC0_0000.
    /// assert_eq!(
    ///     Number::factorial(25, ByteOrder::BigEndian).value,
    ///     [0x0C, 0xD4, 0xA0, 0x61, 0x9F, 0xB0, 0x90, 0x7B, 0xC0, 0x00, 0x00]
    /// );
    /// ```
    pub fn factorial(n: u64, byte_order: ByteOrder) -> Number {
        if n < FACTORIALS.len() as u64 {
            return small_number(byte_order, FACTORIALS[n as usize], false);
        }
        long_factorial(n, byte_order)
    }

    /// Returns the binomial coefficient `C(n, k) = n! / (k! (n - k)!)`, the number of ways to
    /// choose `k` things out of `n`, in the byte order `byte_order`, in the shortest form, in one
    /// new vector; 0 for `k > n`.
    ///
    /// Coefficients below `2^128` are computed in registers. Beyond, for a `k` much smaller than
    /// `n`, as the product of the `k` numbers from `n - k + 1` to `n` divided exactly by the odd
    /// part of `k!`, and otherwise as the product of the coefficient's prime factors, whose
    /// exponents are counted by Kummer's theorem, as GMP's `mpz_bin_uiui` does.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector: with
    /// `capacity overflow` beyond `isize::MAX` bytes, and otherwise as the allocator fails.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// assert_eq!(Number::binomial(5, 2, ByteOrder::BigEndian).value, [10]);
    /// assert_eq!(Number::binomial(2, 5, ByteOrder::BigEndian).value, [0]);
    /// // C(100, 50) = 100,891,344,545,564,193,334,812,497,256
    /// //           = 0x01_45FF_5D3B_1070_380D_C808_5568.
    /// assert_eq!(
    ///     Number::binomial(100, 50, ByteOrder::LittleEndian).value,
    ///     [0x68, 0x55, 0x08, 0xC8, 0x0D, 0x38, 0x70, 0x10, 0x3B, 0x5D, 0xFF, 0x45, 0x01]
    /// );
    /// ```
    pub fn binomial(n: u64, k: u64, byte_order: ByteOrder) -> Number {
        if k > n {
            return small_number(byte_order, 0, false);
        }
        let k = k.min(n - k);
        if let Some(c) = binomial::small(n, k) {
            return small_number(byte_order, c, false);
        }
        long_binomial(n, k, byte_order)
    }
}

/// `n!` for `n` above 34: the odd part shifted left by `n - popcount(n)` bits; on the stack below
/// [`odd::SWING`].
#[inline(never)]
fn long_factorial(n: u64, order: ByteOrder) -> Number {
    let shift = n - u64::from(n.count_ones());
    if n < odd::SWING {
        let mut x = [0u64; odd::SMALL_LIMBS + 1];
        let mut small = [0u64; odd::SMALL_LIMBS];
        let len = odd::small_odd_factorial(n, &mut small);
        x[..len].copy_from_slice(&small[..len]);
        return shifted_number(&mut x, len, shift, order);
    }
    let mut x = odd::odd_factorial(n);
    let len = x.len();
    x.push(0);
    shifted_number(&mut x, len, shift, order)
}

/// `C(n, k)` for `k <= n - k`, above `2^128`.
#[inline(never)]
fn long_binomial(n: u64, k: u64, order: ByteOrder) -> Number {
    binomial::long(n, k, |x, len, shift| shifted_number(x, len, shift, order))
}

/// The magnitude of the `len` limbs `x`, not zero, shifted left by `shift` bits, as a number in
/// one new vector of exactly its length, in the byte order `order`: `x` shifted by the bits below a
/// whole byte in place, into the limb after its own, which `x` must have, then written over the
/// bytes above the `shift / 8` zero bytes at the low end.
fn shifted_number(x: &mut [u64], mut len: usize, shift: u64, order: ByteOrder) -> Number {
    let zeros = usize::try_from(shift / 8).expect("capacity overflow");
    let bits = (shift % 8) as u32;
    if bits != 0 {
        // SAFETY: the slice holds the `len` limbs, shifted in place.
        let out = unsafe { lshift(x.as_mut_ptr(), x.as_ptr(), len, bits) };
        if out != 0 {
            x[len] = out;
            len += 1;
        }
    }
    let top = x[len - 1];
    let bytes = 8 * len - top.leading_zeros() as usize / 8;
    let total = bytes.checked_add(zeros).expect("capacity overflow");
    let p = Product {
        r: x.as_ptr(),
        len: bytes,
        negative: false,
    };
    let mut v = Vec::with_capacity(total);
    // SAFETY: the new vector has room for the `total` bytes, all of which are written before the
    // length is set: the magnitude's and the zeros below it.
    unsafe {
        let dst = v.as_mut_ptr();
        if order == ByteOrder::BigEndian {
            p.write(dst, order);
            core::ptr::write_bytes(dst.add(bytes), 0, zeros);
        } else {
            core::ptr::write_bytes(dst, 0, zeros);
            p.write(dst.add(zeros), order);
        }
        v.set_len(total);
    }
    Number {
        value: v,
        negative: false,
        byte_order: order,
    }
}

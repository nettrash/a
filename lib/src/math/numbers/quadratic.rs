//! The Jacobi symbol and square roots modulo an odd prime.

use super::prime::jacobi as jacobi_1;
use super::{ByteOrder, Number};
use core::cmp::Ordering;

impl Number {
    /// The Jacobi symbol `(self / n)` for an odd positive `n`: 0 when the two have a common
    /// factor, otherwise 1 or -1. For a prime `n` it is the Legendre symbol, 1 for a non-zero
    /// square modulo `n` and -1 for a non-square; for a composite `n`, 1 does not make `self` a
    /// square, though -1 still rules it out.
    ///
    /// # Panics
    ///
    /// Panics if `n` is even, zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let n = Number::new_from_u8(7, ByteOrder::BigEndian);
    /// // 2 = 3^2 modulo 7; 3 is not a square modulo 7; 14 is a multiple of 7.
    /// assert_eq!(Number::new_from_u8(2, ByteOrder::BigEndian).jacobi(&n), 1);
    /// assert_eq!(Number::new_from_u8(3, ByteOrder::BigEndian).jacobi(&n), -1);
    /// assert_eq!(Number::new_from_u8(14, ByteOrder::BigEndian).jacobi(&n), 0);
    /// ```
    #[track_caller]
    pub fn jacobi(&self, n: &Number) -> i32 {
        if n.negative || !n.bit(0) {
            odd_positive();
        }
        let a = limbs(&self.rem_euclid(n));
        let n = limbs(n);
        if n.len() == 1 {
            return jacobi_1(a.first().copied().unwrap_or(0), n[0]);
        }
        jacobi_long(a, n)
    }

    /// A square root of `self` modulo the odd prime `p`: the smaller of the two, or `None` when
    /// `self` is not a square modulo `p`. Zero is its own root. The result is in the byte order
    /// of `self`.
    ///
    /// The result is checked, so a composite `p` never gives a wrong root, though it may give
    /// `None` for a number that has one.
    ///
    /// Like [`pow_mod`](Number::pow_mod), which it is built on, it is not constant-time.
    ///
    /// # Panics
    ///
    /// Panics if `p` is even, zero or negative.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let p = Number::new_from_u8(13, ByteOrder::BigEndian);
    /// // 6^2 = 36 = 10 modulo 13, and 7 = 13 - 6 is the other root.
    /// let r = Number::new_from_u8(10, ByteOrder::BigEndian).sqrt_mod(&p).unwrap();
    /// assert_eq!(r.value, [6]);
    /// assert!(Number::new_from_u8(5, ByteOrder::BigEndian).sqrt_mod(&p).is_none());
    /// ```
    #[track_caller]
    pub fn sqrt_mod(&self, p: &Number) -> Option<Number> {
        if p.negative || !p.bit(0) {
            odd_positive();
        }
        let order = self.byte_order;
        let a = with_order(self.rem_euclid(p), order);
        if a.is_zero() {
            return Some(a);
        }
        if a.jacobi(p) != 1 {
            return None;
        }
        let r = tonelli_shanks(&a, &with_order(p.clone(), order))?;
        let other = p - &r;
        let r = if other < r { other } else { r };
        Some(with_order(r, order))
    }
}

/// Panics for a modulus the symbol is not defined for.
#[cold]
#[inline(never)]
#[track_caller]
fn odd_positive() -> ! {
    panic!("the modulus must be odd and positive")
}

/// `x` in the byte order `order`.
fn with_order(x: Number, order: ByteOrder) -> Number {
    if x.byte_order == order {
        return x;
    }
    let mut value = x.value;
    value.reverse();
    Number {
        value,
        negative: x.negative,
        byte_order: order,
    }
}

/// A root of the non-zero square `a` below `p`, by Tonelli and Shanks: with `p - 1 = q 2^s`,
/// `r = a^((q + 1) / 2)` has `r^2 = a b` for `b = a^q`, whose order divides `2^s`, and each step
/// multiplies `r` and `b` by powers of a generator `c` of the group of order `2^s` until `b` is 1.
/// For `s = 1` it is the single power `a^((p + 1) / 4)`.
fn tonelli_shanks(a: &Number, p: &Number) -> Option<Number> {
    let one = Number::new_from_u8(1, a.byte_order);
    let p1 = p - &one;
    let s = p1.trailing_zeros()? as usize;
    let q = &p1 >> s;
    let half = (&q - &one) >> 1usize;
    let x = a.pow_mod(&half, p);
    let mut r = (a * &x).rem_euclid(p);
    let mut b = (&r * &x).rem_euclid(p);
    if s > 1 && b != one {
        let mut z = Number::new_from_u8(2, a.byte_order);
        loop {
            match z.jacobi(p) {
                -1 => break,
                // A factor of `p`, which is composite, and maybe the square of a number, whose
                // symbols are never -1.
                0 => return None,
                _ => z = &z + &one,
            }
        }
        let mut c = z.pow_mod(&q, p);
        let mut m = s;
        while b != one {
            let mut t = 0;
            let mut b2 = b.clone();
            while b2 != one {
                b2 = (&b2 * &b2).rem_euclid(p);
                t += 1;
                if t == m {
                    return None;
                }
            }
            let mut g = c;
            for _ in 0..m - t - 1 {
                g = (&g * &g).rem_euclid(p);
            }
            r = (&r * &g).rem_euclid(p);
            c = (&g * &g).rem_euclid(p);
            b = (&b * &c).rem_euclid(p);
            m = t;
        }
    }
    ((&r * &r).rem_euclid(p) == *a).then_some(r)
}

/// The limbs of the magnitude of `x`, the least significant first, without zeros above it.
fn limbs(x: &Number) -> Vec<u64> {
    let mut l: Vec<u64> = if x.byte_order == ByteOrder::LittleEndian {
        x.value
            .chunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(b)
            })
            .collect()
    } else {
        x.value
            .rchunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[8 - c.len()..].copy_from_slice(c);
                u64::from_be_bytes(b)
            })
            .collect()
    };
    trim(&mut l);
    l
}

/// Drops the zero limbs at the top.
fn trim(x: &mut Vec<u64>) {
    while x.last() == Some(&0) {
        x.pop();
    }
}

/// The Jacobi symbol `(a / n)` for `a < n`, `n` odd, by the binary method of [`jacobi_1`] on limbs.
fn jacobi_long(mut a: Vec<u64>, mut n: Vec<u64>) -> i32 {
    let mut t = 1;
    loop {
        trim(&mut a);
        if a.is_empty() {
            return if n == [1] { t } else { 0 };
        }
        if a.len() == 1 && n.len() == 1 {
            return t * jacobi_1(a[0], n[0]);
        }
        let z = shift_out_zeros(&mut a);
        if z & 1 == 1 && matches!(n[0] & 7, 3 | 5) {
            t = -t;
        }
        if compare(&a, &n) == Ordering::Less {
            if a[0] & n[0] & 2 != 0 {
                t = -t;
            }
            core::mem::swap(&mut a, &mut n);
        }
        subtract(&mut a, &n);
    }
}

/// Divides the non-zero `x` by the largest power of two that divides it, and gives the exponent.
fn shift_out_zeros(x: &mut Vec<u64>) -> u64 {
    let whole = x.iter().take_while(|&&l| l == 0).count();
    x.drain(..whole);
    let bits = x[0].trailing_zeros();
    if bits != 0 {
        for i in 0..x.len() {
            let above = x.get(i + 1).copied().unwrap_or(0);
            x[i] = (x[i] >> bits) | (above << (64 - bits));
        }
    }
    trim(x);
    whole as u64 * 64 + u64::from(bits)
}

/// The order of `a` and `b`, without zeros at the top.
fn compare(a: &[u64], b: &[u64]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// `a -= b`, for `a >= b`.
fn subtract(a: &mut [u64], b: &[u64]) {
    let mut borrow = false;
    for (i, x) in a.iter_mut().enumerate() {
        let y = b.get(i).copied().unwrap_or(0);
        let (d, b1) = x.overflowing_sub(y);
        let (d, b2) = d.overflowing_sub(u64::from(borrow));
        *x = d;
        borrow = b1 | b2;
    }
}

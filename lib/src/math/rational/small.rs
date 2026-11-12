//! The register paths of the arithmetic, for rationals whose parts all fit in 64 bits: the parts
//! are read as `u64`s with their signs, combined in `u128`s, and the results written back in the
//! byte order of the left operand, in its vectors when it is assigned to.

use super::Rational;
use crate::math::numbers::{ByteOrder, Magnitude, Number, fraction};

/// A rational with both parts of at most 64 bits: the sign, the numerator's magnitude and the
/// denominator.
#[derive(Clone, Copy)]
pub(crate) struct Small {
    pub(crate) negative: bool,
    pub(crate) num: u64,
    pub(crate) den: u64,
}

/// The magnitude `x` with the sign `negative` (not for zero), read from the word's bytes in place:
/// its significant ones, in the machine's byte order.
#[inline(always)]
pub(crate) fn magnitude(x: &u64, negative: bool) -> Magnitude<'_> {
    let len = 8 - (x.leading_zeros() / 8) as usize;
    // SAFETY: a `u64` is eight initialised bytes with no padding, and `[u8; 8]` has its size and an
    // alignment of 1; the reference lives as long as `x`'s.
    let bytes: &[u8; 8] = unsafe { &*core::ptr::from_ref(x).cast::<[u8; 8]>() };
    let big_endian = cfg!(target_endian = "big");
    Magnitude {
        bytes: if big_endian {
            &bytes[8 - len..]
        } else {
            &bytes[..len]
        },
        big_endian,
        negative: negative && len > 0,
    }
}

/// The magnitude of any number when it fits in 64 bits: a vector of up to 8 bytes read straight,
/// zero bytes on top or not, with no search for its significant ones; a longer one when they fit.
#[inline(always)]
pub(crate) fn word_of(n: &Number) -> Option<u64> {
    if n.value.len() <= 8 {
        return Some(short_vector(n));
    }
    any_word(n)
}

/// The value of the vector of `n`, of up to 8 bytes.
#[inline(always)]
pub(crate) fn short_vector(n: &Number) -> u64 {
    debug_assert!(n.value.len() <= 8);
    Magnitude {
        bytes: &n.value,
        big_endian: n.byte_order == ByteOrder::BigEndian,
        negative: false,
    }
    .small() as u64
}

/// The magnitude of a number of a vector of more than 8 bytes, when it fits in 64 bits.
#[cold]
#[inline(never)]
fn any_word(n: &Number) -> Option<u64> {
    let m = Magnitude::of(n);
    if m.bytes.len() <= 8 {
        Some(m.small() as u64)
    } else {
        None
    }
}

impl Small {
    /// The parts of `r`, when both fit in 64 bits: when it is in words.
    #[inline(always)]
    pub(super) fn of(r: &Rational) -> Option<Small> {
        r.word()
    }

    /// The integer `n` as a fraction over 1, when it fits in 64 bits.
    #[inline(always)]
    pub(super) fn integer(n: &Number) -> Option<Small> {
        let num = word_of(n)?;
        Some(Small {
            negative: n.negative && num != 0,
            num,
            den: 1,
        })
    }
}

/// A result of the register paths: the sign, the numerator and the denominator, in lowest terms,
/// the denominator positive; zero is `0/1` and not negative. The numerator of a sum can have 129
/// bits: `top` is its bit 128.
#[derive(Clone, Copy, Debug)]
pub(super) struct Wide {
    pub(super) negative: bool,
    pub(super) num: u128,
    pub(super) top: bool,
    pub(super) den: u128,
}

/// The greatest common divisor of two `u64`s, `gcd(x, 0) = x`, by the binary method (see
/// [`gcd_odd`]), after a remainder when one is longer than the other by 16 bits or more.
#[inline(always)]
pub(crate) fn gcd(x: u64, y: u64) -> u64 {
    if x == 0 || y == 0 {
        return x | y;
    }
    gcd_nonzero(x, y)
}

/// [`gcd`] of two numbers not zero.
#[inline(always)]
fn gcd_nonzero(x: u64, y: u64) -> u64 {
    let k = (x | y).trailing_zeros();
    let (u, v) = (x >> x.trailing_zeros(), y >> y.trailing_zeros());
    let (hi, lo) = if u >= v { (u, v) } else { (v, u) };
    if hi >> 16 > lo {
        return gcd_apart(hi, lo) << k;
    }
    gcd_odd(hi, lo) << k
}

/// The greatest common divisor of the odd `hi` and `lo`, `hi` longer by 16 bits or more: the
/// binary method takes off about a bit a step, a remainder all of the difference at once. Out of
/// line, as it is rare among the parts of a sum or a product.
#[inline(never)]
fn gcd_apart(hi: u64, lo: u64) -> u64 {
    let r = hi % lo;
    if r == 0 {
        return lo;
    }
    gcd_odd(r >> r.trailing_zeros(), lo)
}

/// The greatest common divisor of the odd `u` and `v` by the binary method: the larger is
/// replaced by the difference, whose zero bits at the low end are shifted out, until the two
/// meet, or until one is 1, which leaves 1 (FLINT's `n_gcd` stops there too). The difference, the
/// smaller one and the shift are found together without branches: four cycles a step on an
/// Apple M4, of which there are about `0.7` for each bit of the two.
#[inline(always)]
fn gcd_odd(mut u: u64, mut v: u64) -> u64 {
    debug_assert!(u & v & 1 == 1);
    while u != v && u > 1 && v > 1 {
        (u, v) = step(u, v);
    }
    if u == v { u } else { 1 }
}

/// A step of the binary method on the odd `u` and `v`: `|u - v|` without its zero bits at the
/// low end, and `min(u, v)`. Of `u = v` it makes `(0, v)`, then `(v, 0)`, which it keeps: a
/// finished gcd stays readable as `u | v` while another runs on in step with it.
#[inline(always)]
fn step(u: u64, v: u64) -> (u64, u64) {
    let (d, below) = u.overflowing_sub(v);
    let zeros = d.trailing_zeros();
    let m = if below { u } else { v };
    let a = if below { d.wrapping_neg() } else { d };
    (a.wrapping_shr(zeros), m)
}

/// The greatest common divisors of `(x1, y1)` and `(x2, y2)`, all four not zero, by the binary
/// method on both pairs in step, in one loop: the two chains of dependent steps run side by side,
/// and the loop's exit, the one branch that the predictor cannot know, is taken once rather than
/// twice. A pair whose parts are apart by 16 bits or more takes [`gcd`] on its own.
#[inline(always)]
fn gcd_pair(x1: u64, y1: u64, x2: u64, y2: u64) -> (u64, u64) {
    let k1 = (x1 | y1).trailing_zeros();
    let k2 = (x2 | y2).trailing_zeros();
    let (u1, v1) = (x1 >> x1.trailing_zeros(), y1 >> y1.trailing_zeros());
    let (u2, v2) = (x2 >> x2.trailing_zeros(), y2 >> y2.trailing_zeros());
    let (mut h1, mut l1) = if u1 >= v1 { (u1, v1) } else { (v1, u1) };
    let (mut h2, mut l2) = if u2 >= v2 { (u2, v2) } else { (v2, u2) };
    if h1 >> 16 > l1 || h2 >> 16 > l2 {
        return (gcd_nonzero(x1, y1), gcd_nonzero(x2, y2));
    }
    // A chain is done when its two meet or one is at most 1: see `step` for what it does then.
    let done = |u: u64, v: u64| (u == v) | (u <= 1) | (v <= 1);
    while !(done(h1, l1) & done(h2, l2)) {
        (h1, l1) = step(h1, l1);
        (h2, l2) = step(h2, l2);
        (h1, l1) = step(h1, l1);
        (h2, l2) = step(h2, l2);
    }
    let read = |u: u64, v: u64| if u.min(v) == 1 { 1 } else { u | v };
    (read(h1, l1) << k1, read(h2, l2) << k2)
}

/// `a + b`, `b` taken with the sign `negative_b`.
///
/// The sum `(a·d ± c·b) / (b·d)` of `a/b` and `c/d` is computed whole, without branches, before
/// the greatest common divisor `g = gcd(b, d)`, so that the processor computes it during the
/// gcd's loop: when `g = 1`, as for three pairs of random denominators in five, it is the result
/// in lowest terms. Otherwise it is divided by
/// `g·gcd(n/g, g)` (Knuth's reduction), which is `gcd(n mod g², g²)` for the whole numerator `n`,
/// in words when the parts fit, and by Knuth's steps in 128 bits when they do not.
#[inline(always)]
pub(super) fn add(a: Small, b: Small, negative_b: bool) -> Wide {
    let nb = (b.negative != negative_b) & (b.num != 0);
    let x = u128::from(a.num) * u128::from(b.den);
    let y = u128::from(b.num) * u128::from(a.den);
    let (negative, num, carry) = signed_sum(a.negative, x, nb, y);
    let den = u128::from(a.den) * u128::from(b.den);
    let g = gcd_nonzero(a.den, b.den);
    if g == 1 {
        // `num = 0` only for `n/1 - n/1`, over `1·1`.
        return Wide {
            negative,
            num,
            top: carry,
            den,
        };
    }
    if (num | den) >> 64 == 0 {
        // `g·gcd(n/g, g) = gcd(n, g²)`, as `n/g` is prime to `b/g` and to `d/g`. `g²` divides
        // `b·d`, so fits in 64 bits with it, and so do both products of the sum (`b` and `d` are
        // at least 2), which then did not carry.
        let (n, d) = (num as u64, den as u64);
        let gg = g * g;
        let h = gcd(n % gg, gg);
        return Wide {
            negative: negative && n != 0,
            num: u128::from(n / h),
            top: false,
            den: u128::from(d / h),
        };
    }
    add_knuth(a, b, nb, g)
}

/// [`add`] by Knuth's steps, for parts whose products pass 64 bits: `b` and `d` divided by
/// `g = gcd(b, d)` first, the numerator `a·(d/g) ± c·(b/g)` then by `g2 = gcd(it, g)`, over
/// `(b/g)·(d/g2)`.
#[inline(never)]
fn add_knuth(a: Small, b: Small, nb: bool, g: u64) -> Wide {
    let (s, t) = (a.den / g, b.den / g);
    let x = u128::from(a.num) * u128::from(t);
    let y = u128::from(b.num) * u128::from(s);
    let (negative, num, carry) = signed_sum(a.negative, x, nb, y);
    if carry {
        return carried(negative, num, g, s, b.den);
    }
    if num == 0 {
        return ZERO;
    }
    let g2 = gcd_wide(num, g);
    Wide {
        negative,
        num: div_wide(num, g2),
        top: false,
        den: u128::from(s) * u128::from(b.den / g2),
    }
}

/// `gcd(x, g)` for a `u128` and a non-zero `u64`.
#[inline]
fn gcd_wide(x: u128, g: u64) -> u64 {
    if g == 1 {
        return 1;
    }
    let r = if x >> 64 == 0 {
        x as u64 % g
    } else {
        (x % u128::from(g)) as u64
    };
    gcd(r, g)
}

/// `x / d` for a `u128` and a non-zero `u64`.
#[inline]
fn div_wide(x: u128, d: u64) -> u128 {
    if d == 1 {
        x
    } else if x >> 64 == 0 {
        u128::from(x as u64 / d)
    } else {
        x / u128::from(d)
    }
}

/// The sum of [`add_knuth`] whose numerator `2^128 + lo` passed 128 bits: reduced by `g2`, its
/// greatest common divisor with `g`, by Hensel's exact division modulo `2^128` (the quotient by
/// `g2 >= 2` is below `2^128`), over `s·(d/g2)`.
#[cold]
#[inline(never)]
fn carried(negative: bool, lo: u128, g: u64, s: u64, d: u64) -> Wide {
    let g2 = if g == 1 {
        1
    } else {
        // `(2^128 + lo) mod g`: `2^128 mod g` is `(2^128 - 1) mod g + 1`, at most `g`.
        let g128 = u128::from(g);
        let r = ((u128::MAX % g128) + 1 + lo % g128) % g128;
        gcd(r as u64, g)
    };
    if g2 == 1 {
        return Wide {
            negative,
            num: lo,
            top: true,
            den: u128::from(s) * u128::from(d),
        };
    }
    let z = g2.trailing_zeros();
    let odd = g2 >> z;
    // The numerator shifted right by the zero bits of `g2`, which it has: `2^(128 - z) + lo >> z`
    // for `z > 0`, and `lo` modulo `2^128` for `z = 0`.
    let shifted = if z == 0 {
        lo
    } else {
        (lo >> z) | (1u128 << (128 - z))
    };
    // The inverse of `odd` modulo `2^128`: modulo `2^64`, then one Newton step.
    let v = u128::from(fraction::inverse_word(odd));
    let v = v.wrapping_mul(2u128.wrapping_sub(u128::from(odd).wrapping_mul(v)));
    Wide {
        negative,
        num: shifted.wrapping_mul(v),
        top: false,
        den: u128::from(s) * u128::from(d / g2),
    }
}

/// `a · b`, or `a / b` when `divide` is set (`b` not zero).
///
/// `(a/b)·(c/d)` is `(a/g1)·(c/g2) / ((b/g2)·(d/g1))` for `g1 = gcd(a, d)` and `g2 = gcd(c, b)`,
/// the two gcds run in step (see [`gcd_pair`]); the whole products, computed beside them, are the
/// result when both are 1.
#[inline(always)]
pub(super) fn mul(a: Small, b: Small, divide: bool) -> Wide {
    let (bn, bd) = if divide {
        (b.den, b.num)
    } else {
        (b.num, b.den)
    };
    if a.num == 0 || bn == 0 {
        return ZERO;
    }
    let negative = a.negative != b.negative;
    let num = u128::from(a.num) * u128::from(bn);
    let den = u128::from(a.den) * u128::from(bd);
    let (g1, g2) = gcd_pair(a.num, bd, bn, a.den);
    if g1 == 1 && g2 == 1 {
        return Wide {
            negative,
            num,
            top: false,
            den,
        };
    }
    Wide {
        negative,
        num: u128::from(a.num / g1) * u128::from(bn / g2),
        top: false,
        den: u128::from(a.den / g2) * u128::from(bd / g1),
    }
}

/// The order of `a` and `b` by value: by the signs, then by `|a|·d` against `|c|·b`, without
/// branches.
#[inline(always)]
pub(super) fn cmp(a: Small, b: Small) -> core::cmp::Ordering {
    let x = u128::from(a.num) * u128::from(b.den);
    let y = u128::from(b.num) * u128::from(a.den);
    // The order of the magnitudes, -1, 0 or 1, turned over for two negative numbers; numbers of
    // different signs (zero is not negative) are in the order of their signs.
    let m = i8::from(x > y) - i8::from(x < y);
    let m = if a.negative { -m } else { m };
    let s = i8::from(b.negative) - i8::from(a.negative);
    let r = if a.negative == b.negative { m } else { s };
    r.cmp(&0)
}

/// Zero, `0/1`.
const ZERO: Wide = Wide {
    negative: false,
    num: 0,
    top: false,
    den: 1,
};

/// `±x ± y` as a sign, a magnitude modulo `2^128` and whether it carried out of 128 bits; zero is
/// not negative. Both the sum and the difference are made and one chosen, without branches: the
/// signs of random operands would mispredict one in two.
#[inline(always)]
fn signed_sum(nx: bool, x: u128, ny: bool, y: u128) -> (bool, u128, bool) {
    let (sum, carry) = x.overflowing_add(y);
    let (d, below) = x.overflowing_sub(y);
    let diff = if below { d.wrapping_neg() } else { d };
    let diff_negative = if below { ny } else { nx && d != 0 };
    let same = nx == ny;
    (
        if same { nx } else { diff_negative },
        if same { sum } else { diff },
        same && carry,
    )
}

/// Writes the magnitude `m` with the sign `negative` over `n`, in the byte order `order`, in the
/// shortest form, in `n`'s vector when it has room.
#[inline(always)]
pub(super) fn put(n: &mut Number, order: ByteOrder, negative: bool, m: u128) {
    let v = &mut n.value;
    if v.capacity() < 16 {
        let len = 16 - (m | 1).leading_zeros() as usize / 8;
        if v.capacity() < len {
            // A new vector with room for any result of the register paths, rather than a
            // reallocation of one that copies the bytes about to be replaced.
            *v = Vec::with_capacity(16);
            write16(v, order, m);
        } else {
            v.clear();
            match order {
                ByteOrder::BigEndian => v.extend_from_slice(&m.to_be_bytes()[16 - len..]),
                ByteOrder::LittleEndian => v.extend_from_slice(&m.to_le_bytes()[..len]),
            }
        }
    } else {
        write16(v, order, m);
    }
    n.negative = negative && m != 0;
    n.byte_order = order;
}

/// A new number of the magnitude `m` with the sign `negative`, in the byte order `order`, in the
/// shortest form, in a vector with room for any result of the register paths.
#[inline(always)]
pub(super) fn number(order: ByteOrder, negative: bool, m: u128) -> Number {
    let mut value = Vec::with_capacity(16);
    write16(&mut value, order, m);
    Number {
        value,
        negative: negative && m != 0,
        byte_order: order,
    }
}

/// Writes the magnitude `m` in the shortest form over `v`, which has room for 16 bytes: all 16
/// written at once, the significant ones first, and the length set to theirs.
#[inline(always)]
fn write16(v: &mut Vec<u8>, order: ByteOrder, m: u128) {
    debug_assert!(v.capacity() >= 16);
    let len = 16 - (m | 1).leading_zeros() as usize / 8;
    let bytes = match order {
        ByteOrder::BigEndian => (m << (8 * (16 - len))).to_be_bytes(),
        ByteOrder::LittleEndian => m.to_le_bytes(),
    };
    // SAFETY: the vector has room for 16 bytes, which the write initialises; `len` is at most 16.
    unsafe {
        v.as_mut_ptr().cast::<[u8; 16]>().write_unaligned(bytes);
        v.set_len(len);
    }
}

/// Writes the magnitude `2^128 + m` with the sign `negative` over `n`, in the byte order `order`,
/// 17 bytes, in `n`'s vector when it has room.
#[cold]
#[inline(never)]
fn put_top(n: &mut Number, order: ByteOrder, negative: bool, m: u128) {
    let v = &mut n.value;
    if v.capacity() < 17 {
        *v = Vec::with_capacity(24);
    }
    v.clear();
    match order {
        ByteOrder::BigEndian => {
            v.push(1);
            v.extend_from_slice(&m.to_be_bytes());
        }
        ByteOrder::LittleEndian => {
            v.extend_from_slice(&m.to_le_bytes());
            v.push(1);
        }
    }
    n.negative = negative;
    n.byte_order = order;
}

impl Wide {
    /// Whether both parts fit in 64 bits.
    #[inline(always)]
    fn short(self) -> bool {
        !self.top && (self.num | self.den) >> 64 == 0
    }

    /// Writes the result over `r`, in the byte order `order`: in words when they fit, and
    /// otherwise in its vectors when it has them with room.
    #[inline(always)]
    pub(super) fn write(self, r: &mut Rational, order: ByteOrder) {
        if self.short() {
            r.set_word(self.negative, self.num as u64, self.den as u64, order);
            return;
        }
        match r.numbers_mut() {
            Some((num, den)) => {
                if self.top {
                    put_top(num, order, self.negative, self.num);
                } else {
                    put(num, order, self.negative, self.num);
                }
                put(den, order, false, self.den);
            }
            None => *r = self.long(order),
        }
    }

    /// The result as a new rational in the byte order `order`.
    #[inline(always)]
    pub(super) fn rational(self, order: ByteOrder) -> Rational {
        if self.short() {
            return Rational::from_word(self.negative, self.num as u64, self.den as u64, order);
        }
        self.long(order)
    }

    /// The result, of a part of more than 64 bits, in new numbers: the common case of products of
    /// parts of a limb, not a rare one.
    #[inline(never)]
    fn long(self, order: ByteOrder) -> Rational {
        let mut num = number(order, self.negative, self.num);
        if self.top {
            put_top(&mut num, order, self.negative, self.num);
        }
        Rational::from_reduced(num, number(order, false, self.den))
    }
}

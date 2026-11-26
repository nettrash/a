//! The arithmetic of a rational in words with one in numbers: a short operand with a long one, as
//! `r + 1/k` in a sum of many terms.
//!
//! The steps are those of Knuth's reductions, as on limbs (see
//! [`fraction`](crate::math::numbers::fraction)), but with the words of the short operand in
//! registers: the greatest common divisor of a word and a long part is a remainder of the long
//! part by the word and the binary gcd of two words, and every product and exact quotient is a
//! single pass over a long part by a word. Each pass reads the long parts' bytes in place as
//! little-endian limbs and writes its result straight into the bytes of a part of the result, so
//! that nothing is copied between bytes and limbs and nothing is allocated but the two vectors of
//! the result (or nothing, when the result is written into the vectors of the operand assigned
//! to, which have room).
//!
//! A sum `a/b ± C/D` takes `r = D mod b`, `g = gcd(b, r)` and, when `g = 1` (about three times in
//! five for random parts), the numerator `a·D ± b·C` and the denominator `b·D` in one pass over
//! both long parts. Otherwise `T = D/g` by Hensel's division, the numerator `a·T ± (b/g)·C`, its
//! remainder by `b` on the powers already made for `D` and so `g₂ = gcd(num mod g, g)`, and the
//! denominator `T·((b/g)·(g/g₂))`, a product by one word. A product `(a/b)·(C/D)` takes
//! `g₁ = gcd(a, D mod a)` and `g₂ = gcd(b, C mod b)`, then the two products by words.
//!
//! The passes take the numbers' kernels on the bytes of the parts (`mul_1_bytes` and
//! `addmul_1_bytes`, and the remainder by a word eight limbs a step with the powers of `B` modulo
//! it, `rem_1`) from [`KERNEL_LIMBS`] limbs, and plain loops of `u128` products below and for the
//! few limbs above the parts' full ones; Hensel's exact division by a word is a loop here.
//!
//! This path is taken on little-endian targets, for a long operand in little-endian byte order
//! and a result in it; any other goes the way of [`fraction`](crate::math::numbers::fraction).

use super::Rational;
use super::small::Small;
use crate::math::numbers::nat::{
    Operand, Powers, REM_1_BYTES, REM_1_BYTES_WIDE, REM_1_MAX, REM_1_PREPARED, addmul_1_bytes,
    mul_1_bytes, rem_1 as rem_1_powers,
};
use crate::math::numbers::{ByteOrder, Number, divide_2by1, fraction, reciprocal};
use core::cell::RefCell;
use core::ptr;

/// The magnitude of a part, read in place: a pointer to its little-endian bytes and the number of
/// its significant ones (no zero byte on top; none for zero).
#[derive(Clone, Copy)]
struct Part {
    p: *const u8,
    len: usize,
}

impl Part {
    /// The magnitude of `n`, a number in little-endian byte order.
    #[inline(always)]
    fn of(n: &Number) -> Part {
        debug_assert!(n.byte_order == ByteOrder::LittleEndian);
        let v = &n.value;
        let mut len = v.len();
        while len > 0 && v[len - 1] == 0 {
            len -= 1;
        }
        Part { p: v.as_ptr(), len }
    }

    /// Its number of limbs.
    #[inline(always)]
    fn n(self) -> usize {
        self.len.div_ceil(8)
    }

    /// Whether it is 1.
    #[inline(always)]
    fn is_one(self) -> bool {
        // SAFETY: a part of one significant byte holds it.
        self.len == 1 && unsafe { *self.p } == 1
    }

    /// Its full limbs: those of eight significant bytes.
    #[inline(always)]
    fn full(self) -> usize {
        self.len / 8
    }

    /// Its full limb `i`, `i < full()`.
    ///
    /// # Safety
    ///
    /// The part's bytes must be valid for reads.
    #[inline(always)]
    unsafe fn at(self, i: usize) -> u64 {
        // SAFETY: as the caller guarantees; the limb is within the significant bytes.
        unsafe { u64::from_le(ptr::read_unaligned(self.p.add(8 * i).cast::<u64>())) }
    }

    /// Its limb `i`, zero above its top.
    ///
    /// # Safety
    ///
    /// The part's bytes must be valid for reads.
    #[inline(always)]
    unsafe fn limb(self, i: usize) -> u64 {
        let off = 8 * i;
        // SAFETY: a full limb is within the significant bytes; a partial one is read byte by byte.
        unsafe {
            if off + 8 <= self.len {
                u64::from_le(ptr::read_unaligned(self.p.add(off).cast::<u64>()))
            } else if off < self.len {
                let mut w = 0u64;
                for k in 0..self.len - off {
                    w |= u64::from(*self.p.add(off + k)) << (8 * k);
                }
                w
            } else {
                0
            }
        }
    }
}

/// A part being written: a vector whose capacity holds the limbs written, its bytes the limbs in
/// little-endian order.
struct Out<'a> {
    v: &'a mut Vec<u8>,
}

impl Out<'_> {
    /// The pointer to write limbs at.
    #[inline(always)]
    fn ptr(&mut self) -> *mut u8 {
        self.v.as_mut_ptr()
    }

    /// Sets the length to the significant bytes of the `n` limbs written; one byte, zero, for zero.
    ///
    /// # Safety
    ///
    /// The `n` limbs must have been written.
    #[inline(always)]
    unsafe fn finish(&mut self, n: usize) -> usize {
        let p = self.v.as_ptr();
        let mut n = n;
        // SAFETY: the limbs were written.
        unsafe {
            while n > 0 && ptr::read_unaligned(p.add(8 * (n - 1)).cast::<u64>()) == 0 {
                n -= 1;
            }
            let len = if n == 0 {
                *self.v.as_mut_ptr() = 0;
                1
            } else {
                let top = u64::from_le(ptr::read_unaligned(p.add(8 * (n - 1)).cast::<u64>()));
                8 * n - top.leading_zeros() as usize / 8
            };
            self.v.set_len(len);
            len
        }
    }
}

/// Makes room for `n` limbs in `v`, in its allocation when it has it, with an eighth to spare when
/// it grows. The contents are not kept.
#[inline(always)]
fn room(v: &mut Vec<u8>, n: usize) {
    let bytes = 8 * n;
    if v.capacity() < bytes {
        *v = Vec::with_capacity(bytes + bytes / 8 + 8);
    }
}

/// Makes room for `n` limbs in `v` keeping its contents, which an operation reads in place while
/// writing (each limb read before it is written).
#[inline(always)]
fn room_keep(v: &mut Vec<u8>, n: usize) {
    let bytes = 8 * n;
    if v.capacity() < bytes {
        v.reserve(bytes + bytes / 8 + 8 - v.len());
    }
}

/// Writes the limb `w` at index `i`.
///
/// # Safety
///
/// The vector's capacity must hold the limb.
#[inline(always)]
unsafe fn put(o: *mut u8, i: usize, w: u64) {
    // SAFETY: as the caller guarantees.
    unsafe { ptr::write_unaligned(o.add(8 * i).cast::<u64>(), w.to_le()) }
}

/// `x · m` into `o`, which may be the bytes of `x` (each limb is read before it is written);
/// returns the limbs written, `x.n() + 1`.
///
/// # Safety
///
/// `o` must hold `x.n() + 1` limbs and `x` be valid for reads.
unsafe fn mul_1(o: *mut u8, x: Part, m: u64) -> usize {
    let (n, full) = (x.n(), x.full());
    let mut carry = 0u64;
    // SAFETY: as the caller guarantees; the kernel reads and writes the full limbs, `o` being `x`
    // or apart from it.
    unsafe {
        if full >= KERNEL_LIMBS {
            carry = mul_1_bytes(o, x.p, full, m);
        } else {
            for i in 0..full {
                let t = u128::from(x.at(i)) * u128::from(m) + u128::from(carry);
                put(o, i, t as u64);
                carry = (t >> 64) as u64;
            }
        }
        if n > full {
            let t = u128::from(x.limb(full)) * u128::from(m) + u128::from(carry);
            put(o, full, t as u64);
            carry = (t >> 64) as u64;
        }
        put(o, n, carry);
    }
    n + 1
}

/// `x` itself, into `o`; returns the limbs written.
///
/// # Safety
///
/// `o` must hold `x.n()` limbs, apart from `x`.
unsafe fn copy(o: *mut u8, x: Part) -> usize {
    let n = x.n();
    // SAFETY: as the caller guarantees.
    unsafe {
        ptr::copy_nonoverlapping(x.p, o, x.len);
        ptr::write_bytes(o.add(x.len), 0, 8 * n - x.len);
    }
    n
}

/// `u·p + v·q`, or `u·p - v·q` with `sub`, into `o`, which may be the bytes of `p` or `q`: returns
/// the limbs written and whether the result is negative (it may be zero).
///
/// # Safety
///
/// `o` must hold `max(p.n(), q.n()) + 2` limbs and the parts be valid for reads.
#[inline(always)]
unsafe fn lincomb(o: *mut u8, u: u64, p: Part, v: u64, q: Part, sub: bool) -> (usize, bool) {
    // SAFETY: as the caller guarantees.
    unsafe {
        if sub {
            comb::<true, false>(o, u, p, v, q, ptr::null_mut(), 0)
        } else {
            comb::<false, false>(o, u, p, v, q, ptr::null_mut(), 0)
        }
    }
}

/// [`lincomb`], and `w·p` into `o2` in the same pass (which may be the bytes of `p`): the numerator
/// and the denominator of a sum whose denominators are prime to each other.
///
/// # Safety
///
/// As for [`lincomb`], and `o2` must hold `max(p.n(), q.n()) + 2` limbs.
#[inline(always)]
unsafe fn lincomb_mul(
    o: *mut u8,
    u: u64,
    p: Part,
    v: u64,
    q: Part,
    sub: bool,
    (o2, w): (*mut u8, u64),
) -> (usize, bool, usize) {
    let n2 = p.n().max(q.n()) + 1;
    // SAFETY: as the caller guarantees.
    let (k, neg) = unsafe {
        if sub {
            comb::<true, true>(o, u, p, v, q, o2, w)
        } else {
            comb::<false, true>(o, u, p, v, q, o2, w)
        }
    };
    (k, neg, n2)
}

/// The pass of [`lincomb`] and [`lincomb_mul`]: `u·p ± v·q`, and with `DEN` also `w·p` into `o2`.
///
/// # Safety
///
/// As for [`lincomb_mul`].
#[inline(always)]
unsafe fn comb<const SUB: bool, const DEN: bool>(
    o: *mut u8,
    u: u64,
    p: Part,
    v: u64,
    q: Part,
    o2: *mut u8,
    w: u64,
) -> (usize, bool) {
    let n = p.n().max(q.n());
    // Both parts' full limbs read straight; the few above them limb by limb.
    let both = p.full().min(q.full());
    if both >= KERNEL_LIMBS {
        // SAFETY: as the caller guarantees.
        return unsafe { comb_kernels::<SUB, DEN>(o, u, p, v, q, (o2, w), both) };
    }
    let (mut c1, mut c2, mut c3, mut c4) = (0u64, 0u64, 0u64, 0u64);
    // SAFETY: as the caller guarantees; each limb is read before it is written.
    unsafe {
        let mut step = |i: usize, pi: u64, qi: u64| {
            let t1 = u128::from(u) * u128::from(pi) + u128::from(c1);
            let t2 = u128::from(v) * u128::from(qi) + u128::from(c2);
            if DEN {
                let t3 = u128::from(w) * u128::from(pi) + u128::from(c4);
                put(o2, i, t3 as u64);
                c4 = (t3 >> 64) as u64;
            }
            let s = if SUB {
                let (s, b1) = (t1 as u64).overflowing_sub(t2 as u64);
                let (s, b2) = s.overflowing_sub(c3);
                c3 = u64::from(b1) + u64::from(b2);
                s
            } else {
                let (s, v1) = (t1 as u64).overflowing_add(t2 as u64);
                let (s, v2) = s.overflowing_add(c3);
                c3 = u64::from(v1) + u64::from(v2);
                s
            };
            put(o, i, s);
            c1 = (t1 >> 64) as u64;
            c2 = (t2 >> 64) as u64;
        };
        for i in 0..both {
            step(i, p.at(i), q.at(i));
        }
        for i in both..n {
            step(i, p.limb(i), q.limb(i));
        }
        if DEN {
            put(o2, n, c4);
        }
        if !SUB {
            let (s, v1) = c1.overflowing_add(c2);
            let (s, v2) = s.overflowing_add(c3);
            put(o, n, s);
            put(o, n + 1, u64::from(v1) + u64::from(v2));
            return (n + 2, false);
        }
        let top = i128::from(c1) - i128::from(c2) - i128::from(c3);
        put(o, n, top as u64);
        if top >= 0 {
            return (n + 1, false);
        }
        // Negative: the magnitude is the two's complement of the n + 1 limbs, with the limb above
        // them all ones.
        let mut carry = 1u64;
        for i in 0..=n {
            let x = ptr::read_unaligned(o.add(8 * i).cast::<u64>());
            let (s, c) = (!u64::from_le(x)).overflowing_add(carry);
            put(o, i, s);
            carry = u64::from(c);
        }
        (n + 1, true)
    }
}

/// From this many limbs that both parts have in full, [`comb`] and [`mul_1`] take the numbers'
/// kernels for them, and the limbs above them in the loop of `u128` products.
// Measured on an Apple M4 Max (sweep 3..=512 limbs, each way forced, a word of 64-bit parts with
// a long fraction or integer): the product by a word ([`mul_1`]) is faster by the kernels from 5
// limbs (by 1 to 2 % to 16 limbs, 7 % at 64, 24 % at 512), the pass of a sum ([`comb`]) by the
// loop to about 40 limbs in place (by 1 to 8 %) and to 512 for a new sum (by 1 to 7 %). Against
// 8, this value took 0.955 to 0.964 of the time of `+` and `+=` from 8 to 11 limbs and 1.002 to
// 1.016 of that of `·` (alternated, twice); 16 took 1.022 to 1.023 of it for `·` at 15 limbs.
const KERNEL_LIMBS: usize = 12;

thread_local! {
    /// The limbs of `v·q` that a difference subtracts, for [`comb_kernels`].
    static SCRATCH: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

/// [`comb`] for parts of at least [`KERNEL_LIMBS`] full limbs each: the low `both` limbs of
/// `u·p ± v·q` by the numbers' kernels (`v·q` first, into `o`, which may be the bytes of `q` but
/// never those of `p`; a difference subtracts `v·q` from `u·p` in the scratch), the limbs above in
/// the loop of `u128` products from the carries, and `w·p` last, into `o2`, which may be the bytes
/// of `p`.
///
/// # Safety
///
/// As for [`comb`]; `both` is at most the full limbs of each part, and `o` is not `p`'s bytes.
#[inline(never)]
unsafe fn comb_kernels<const SUB: bool, const DEN: bool>(
    o: *mut u8,
    u: u64,
    p: Part,
    v: u64,
    q: Part,
    (o2, w): (*mut u8, u64),
    both: usize,
) -> (usize, bool) {
    debug_assert!(!ptr::eq(o.cast_const(), p.p));
    // SAFETY: as the caller guarantees: the kernels read the `both` full limbs of each part and
    // write those of `o`; `q`'s are read before they are written, when `o` is `q`'s bytes.
    let (c1, c2, c3) = unsafe {
        if SUB {
            SCRATCH.with(|s| {
                let mut s = s.borrow_mut();
                if s.len() < both {
                    s.resize(both, 0);
                }
                let t = s.as_mut_ptr().cast::<u8>();
                let c2 = mul_1_bytes(t, q.p, both, v);
                let c1 = mul_1_bytes(o, p.p, both, u);
                // `o ← o - t` over the `both` limbs: the borrow is the third carry.
                let mut borrow = 0u64;
                for i in 0..both {
                    let x = u64::from_le(ptr::read_unaligned(o.add(8 * i).cast::<u64>()));
                    let (d, b1) = x.overflowing_sub(*s.get_unchecked(i));
                    let (d, b2) = d.overflowing_sub(borrow);
                    put(o, i, d);
                    borrow = u64::from(b1 | b2);
                }
                (c1, c2, borrow)
            })
        } else {
            let c2 = mul_1_bytes(o, q.p, both, v);
            let c1 = addmul_1_bytes(o, p.p, both, u);
            (c1, c2, 0)
        }
    };
    // SAFETY: as the caller guarantees; the limbs from `both` on are read before they are written.
    let r = unsafe { tail::<SUB>(o, u, p, v, q, both, (c1, c2, c3)) };
    if DEN {
        // SAFETY: `o2` holds the limbs of `w·p`; `p` was read in full above.
        unsafe {
            let k = mul_1(o2, p, w);
            for i in k..=p.n().max(q.n()) {
                put(o2, i, 0);
            }
        }
    }
    r
}

/// The limbs of `u·p ± v·q` from `from` on, with the carries `(c1, c2, c3)` of the products and the
/// sum below it, and the top limbs: the end of [`comb`]'s pass.
///
/// # Safety
///
/// As for [`comb`].
#[inline(always)]
unsafe fn tail<const SUB: bool>(
    o: *mut u8,
    u: u64,
    p: Part,
    v: u64,
    q: Part,
    from: usize,
    (mut c1, mut c2, mut c3): (u64, u64, u64),
) -> (usize, bool) {
    let n = p.n().max(q.n());
    // SAFETY: as the caller guarantees.
    unsafe {
        for i in from..n {
            let t1 = u128::from(u) * u128::from(p.limb(i)) + u128::from(c1);
            let t2 = u128::from(v) * u128::from(q.limb(i)) + u128::from(c2);
            let s = if SUB {
                let (s, b1) = (t1 as u64).overflowing_sub(t2 as u64);
                let (s, b2) = s.overflowing_sub(c3);
                c3 = u64::from(b1) + u64::from(b2);
                s
            } else {
                let (s, v1) = (t1 as u64).overflowing_add(t2 as u64);
                let (s, v2) = s.overflowing_add(c3);
                c3 = u64::from(v1) + u64::from(v2);
                s
            };
            put(o, i, s);
            c1 = (t1 >> 64) as u64;
            c2 = (t2 >> 64) as u64;
        }
        if !SUB {
            let (s, v1) = c1.overflowing_add(c2);
            let (s, v2) = s.overflowing_add(c3);
            put(o, n, s);
            put(o, n + 1, u64::from(v1) + u64::from(v2));
            return (n + 2, false);
        }
        let top = i128::from(c1) - i128::from(c2) - i128::from(c3);
        put(o, n, top as u64);
        if top >= 0 {
            return (n + 1, false);
        }
        let mut carry = 1u64;
        for i in 0..=n {
            let x = ptr::read_unaligned(o.add(8 * i).cast::<u64>());
            let (s, c) = (!u64::from_le(x)).overflowing_add(carry);
            put(o, i, s);
            carry = u64::from(c);
        }
        (n + 1, true)
    }
}

/// `x / d` for a word `d` (at least 2) that divides `x` exactly, into `o`, which may be the bytes
/// of `x`: Hensel's division from the low end, by the inverse of the odd part of `d` modulo `B`.
/// Returns the limbs written.
///
/// # Safety
///
/// `o` must hold `x.n()` limbs and `x` be valid for reads.
unsafe fn divexact_1(o: *mut u8, x: Part, d: u64) -> usize {
    let n = x.n();
    let z = d.trailing_zeros();
    let d = d >> z;
    let inv = fraction::inverse_word(d);
    // SAFETY: as the caller guarantees; limb `i + 1` is read before limb `i` is written.
    unsafe {
        let mut c = 0u64;
        let mut next = x.limb(0);
        for i in 0..n {
            let hi = if i + 1 < n { x.limb(i + 1) } else { 0 };
            let lo = if z == 0 {
                next
            } else {
                (next >> z) | (hi << (64 - z))
            };
            next = hi;
            let (s, b) = lo.overflowing_sub(c);
            let q = s.wrapping_mul(inv);
            put(o, i, q);
            c = ((u128::from(q) * u128::from(d)) >> 64) as u64 + u64::from(b);
        }
    }
    n
}

/// The remainder of `(u1 B + u0) mod d` for `u1 < d`, with `d` shifted by `s` to `dn` and its
/// reciprocal `v`.
#[inline(always)]
fn rem_2by1(u1: u64, u0: u64, s: u32, dn: u64, v: u64) -> u64 {
    if s == 0 {
        return divide_2by1(u1, u0, dn, v).1;
    }
    let hi = (u1 << s) | (u0 >> (64 - s));
    divide_2by1(hi, u0 << s, dn, v).1 >> s
}

/// A word divisor for remainders of long parts: `d` normalized with its reciprocal for the short
/// ones, a limb at a time, and for the long ones the powers of `B` modulo `d` of the numbers'
/// [`rem_1`](rem_1_powers), eight limbs a step, made the first time a long part asks for them,
/// then kept for the others.
struct ByWord {
    d: u64,
    s: u32,
    dn: u64,
    v: u64,
    powers: Option<Powers>,
}

impl ByWord {
    /// The divisor `d`, at least 2.
    #[inline(always)]
    fn new(d: u64) -> ByWord {
        let s = d.leading_zeros();
        let dn = d << s;
        ByWord {
            d,
            s,
            dn,
            v: reciprocal(dn),
            powers: None,
        }
    }

    /// `(u1 B + u0) mod d` for `u1 < d`.
    #[inline(always)]
    fn rem2(&self, u1: u64, u0: u64) -> u64 {
        rem_2by1(u1, u0, self.s, self.dn, self.v)
    }

    /// The remainder of `x`: by the powers from the lengths where the numbers' remainders take
    /// them, the powers' own cost included the first time.
    ///
    /// # Safety
    ///
    /// `x` must be valid for reads.
    unsafe fn rem(&mut self, x: Part) -> u64 {
        let n = x.n();
        if n == 0 {
            return 0;
        }
        let long = if self.powers.is_some() {
            REM_1_PREPARED
        } else if self.d <= REM_1_MAX {
            REM_1_BYTES
        } else {
            REM_1_BYTES_WIDE
        };
        // SAFETY: as the caller guarantees.
        unsafe {
            if x.len >= long {
                return self.rem_long(x);
            }
            let mut r = x.limb(n - 1) % self.d;
            for i in (0..n - 1).rev() {
                r = self.rem2(r, x.limb(i));
            }
            r
        }
    }

    /// [`rem`](ByWord::rem) of a long part, by the powers.
    ///
    /// # Safety
    ///
    /// `x` must be valid for reads.
    #[inline(never)]
    unsafe fn rem_long(&mut self, x: Part) -> u64 {
        let p = *self.powers.get_or_insert_with(|| Powers::of(self.d));
        // SAFETY: as the caller guarantees; the operand is the part's significant bytes,
        // little-endian.
        unsafe {
            rem_1_powers(
                Operand {
                    ptr: x.p,
                    len: x.len,
                    big_endian: false,
                    significant: x.len,
                },
                &p,
            )
        }
    }
}

/// The remainder of `x` by the word `d`, at least 2.
///
/// # Safety
///
/// `x` must be valid for reads.
#[inline(always)]
unsafe fn rem_1(x: Part, d: u64) -> u64 {
    // SAFETY: as the caller guarantees.
    unsafe { ByWord::new(d).rem(x) }
}

/// The longest part, in limbs, that this path takes: beyond, the operation goes the way of
/// [`fraction`](crate::math::numbers::fraction).
// Measured on an Apple M4 Max (sweep 1024..=262144 limbs, each way forced, a word of 64-bit parts
// with a long fraction): this path took 0.57 to 0.90 of the time of `fraction`'s for `+`, `-` and
// `·` at every length to this one, 0.73 to 0.94 for a sum of 32 terms by `+=`, and 0.94 to 1.07
// for `+=` and then `-=` of the same word in place (a common factor in the second); beyond, 0.55
// to 0.93 and 0.83 to 0.90 of it. The bound keeps the working memory of a difference
// ([`SCRATCH`], which the thread keeps) at 512 KiB.
const MIXED_LIMBS: usize = 1 << 16;

/// The long operand of an operation, read in place: the sign and the magnitudes of its parts.
#[derive(Clone, Copy)]
struct Long {
    negative: bool,
    num: Part,
    den: Part,
}

/// Whether the parts of `r` are little-endian numbers that this path reads: on little-endian
/// targets.
#[inline(always)]
fn readable(r: &Rational) -> Option<Long> {
    if cfg!(target_endian = "big") {
        return None;
    }
    let (n, d) = r.numbers()?;
    let longest = n.value.len().max(d.value.len());
    // Parts of up to 16 bytes take the `u128` registers of the limbs' path, which is faster.
    if n.byte_order != ByteOrder::LittleEndian
        || d.byte_order != ByteOrder::LittleEndian
        || longest <= 16
        || longest > 8 * MIXED_LIMBS
    {
        return None;
    }
    Some(Long {
        negative: n.negative,
        num: Part::of(n),
        den: Part::of(d),
    })
}

/// The two vectors a result is written into.
struct Target<'a> {
    num: &'a mut Vec<u8>,
    den: &'a mut Vec<u8>,
}

/// `±a/b + ±C/D` (the signs `sw` and `y.negative`) into `t`, which may be the vectors of `y`:
/// returns the sign of the result. `a` is not zero and `b` at least 1.
///
/// # Safety
///
/// The parts of `y` must be valid for reads; when they are the vectors of `t`, `t`'s vectors must
/// have room for the result's limbs (see [`sum_limbs`]) without moving.
unsafe fn sum_into(sw: bool, a: u64, b: u64, y: Long, t: Target) -> bool {
    let (c, d) = (y.num, y.den);
    // The sign of the result follows the larger term; `sub` when the signs differ, the result
    // then taken as `sw`'s term minus `y`'s.
    let sub = sw != y.negative;
    // SAFETY: as the caller guarantees; the denominator `D` is read by the numerator's pass before
    // the denominator is written.
    unsafe {
        let mut num = Out { v: t.num };
        let mut den = Out { v: t.den };
        if b == 1 {
            // `(a·D ± C)/D`, in lowest terms as `C/D` is.
            let (k, neg) = lincomb(num.ptr(), a, d, 1, c, sub);
            num.finish(k);
            if !ptr::eq(den.v.as_ptr(), d.p) {
                let k = copy(den.ptr(), d);
                den.finish(k);
            }
            return sw != neg;
        }
        let mut by = ByWord::new(b);
        let r = if d.is_one() { 1 } else { by.rem(d) };
        let g = fraction::gcd_words(b, r);
        if g == 1 {
            // The numerator and the denominator in one pass over `D` and `C`.
            let (k, neg, k2) = lincomb_mul(num.ptr(), a, d, b, c, sub, (den.ptr(), b));
            num.finish(k);
            den.finish(k2);
            return sw != neg;
        }
        // `s = b/g`, `T = D/g`: the numerator `a·T ± s·C` can share a factor only with `g`. `T` is
        // written into the denominator's vector, read by the numerator's pass, and then replaced.
        let s = b / g;
        let k = divexact_1(den.ptr(), d, g);
        den.finish(k);
        let tp = Part {
            p: den.v.as_ptr(),
            len: den.v.len(),
        };
        let (k, neg) = lincomb(num.ptr(), a, tp, s, c, sub);
        let len = num.finish(k);
        if len == 1 && *num.v.as_ptr() == 0 {
            // Zero is `0/1`.
            *den.v.as_mut_ptr() = 1;
            den.v.set_len(1);
            return false;
        }
        let np = Part {
            p: num.v.as_ptr(),
            len,
        };
        // `g₂ = gcd(num mod g, g)`, the remainder by `b` (a multiple of `g`) on its powers.
        let g2 = fraction::gcd_words(by.rem(np) % g, g);
        // The denominator `s·(D/g₂)`, from `T = D/g` as `T·(s·g/g₂)`, a word as `b` is.
        let k = mul_1(den.ptr(), tp, s * (g / g2));
        den.finish(k);
        if g2 != 1 {
            let k = divexact_1(num.ptr(), np, g2);
            num.finish(k);
        }
        sw != neg
    }
}

/// The limbs that a sum of a word with `y` writes in each part, at most.
#[inline(always)]
fn sum_limbs(y: Long) -> usize {
    y.num.n().max(y.den.n()) + 2
}

/// `(±a/b)·(±C/D)` into `t`, which may be the vectors of `y`. `a` is not zero.
///
/// # Safety
///
/// As for [`sum_into`], with room for [`product_limbs`].
unsafe fn product_into(a: u64, b: u64, y: Long, t: Target) {
    let (c, d) = (y.num, y.den);
    // SAFETY: as the caller guarantees.
    unsafe {
        let g1 = if a == 1 || d.is_one() {
            1
        } else {
            fraction::gcd_words(a, rem_1(d, a))
        };
        let g2 = if b == 1 {
            1
        } else {
            fraction::gcd_words(b, rem_1(c, b))
        };
        let mut num = Out { v: t.num };
        let mut den = Out { v: t.den };
        // The numerator `(C/g₂)·(a/g₁)`, in place in its pass.
        let k = if g2 == 1 {
            mul_1(num.ptr(), c, a / g1)
        } else {
            let k = divexact_1(num.ptr(), c, g2);
            let l = num.finish(k);
            mul_1(
                num.ptr(),
                Part {
                    p: num.v.as_ptr(),
                    len: l,
                },
                a / g1,
            )
        };
        num.finish(k);
        let k = if g1 == 1 {
            mul_1(den.ptr(), d, b / g2)
        } else {
            let k = divexact_1(den.ptr(), d, g1);
            let l = den.finish(k);
            mul_1(
                den.ptr(),
                Part {
                    p: den.v.as_ptr(),
                    len: l,
                },
                b / g2,
            )
        };
        den.finish(k);
    }
}

/// The limbs that a product of a word with `y` writes in each part, at most.
#[inline(always)]
fn product_limbs(y: Long) -> usize {
    y.num.n().max(y.den.n()) + 1
}

/// The kind of operation, with the short operand on the left or the right.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// A result in two new vectors, made a rational.
#[inline(always)]
fn rational(num: Vec<u8>, negative: bool, den: Vec<u8>) -> Rational {
    let num = Number {
        negative: negative && !(num.len() == 1 && num[0] == 0),
        value: num,
        byte_order: ByteOrder::LittleEndian,
    };
    let den = Number {
        value: den,
        negative: false,
        byte_order: ByteOrder::LittleEndian,
    };
    Rational::from_reduced(num, den)
}

/// `x op y` (`y op x` when `short_left` is not set) for `x` in words and `y` in numbers, in new
/// vectors, when this path reads `y` (and the result is little-endian); `None` otherwise. A
/// division by zero is the caller's to take apart.
#[inline]
pub(super) fn word_long(
    x: Small,
    y: &Rational,
    op: Op,
    short_left: bool,
    order: ByteOrder,
) -> Option<Rational> {
    if order != ByteOrder::LittleEndian {
        return None;
    }
    let ly = readable(y)?;
    let mut num = Vec::new();
    let mut den = Vec::new();
    // SAFETY: the parts of `y` are borrowed for the call and the new vectors have room.
    unsafe {
        match op {
            Op::Add | Op::Sub => {
                if x.num == 0 {
                    // `0 ± y`.
                    let neg = if op == Op::Sub && short_left {
                        !ly.negative
                    } else {
                        ly.negative
                    };
                    let mut r = y.clone();
                    if r.is_negative() != neg {
                        r.negate();
                    }
                    return Some(r);
                }
                // `x - y` is `x + (-y)`, and `y - x` is `y + (-x)`.
                let (sx, yl) = match (op, short_left) {
                    (Op::Sub, true) => (
                        x.negative,
                        Long {
                            negative: !ly.negative,
                            ..ly
                        },
                    ),
                    (Op::Sub, false) => (!x.negative, ly),
                    _ => (x.negative, ly),
                };
                let n = sum_limbs(yl);
                room(&mut num, n);
                room(&mut den, n);
                let neg = sum_into(
                    sx,
                    x.num,
                    x.den,
                    yl,
                    Target {
                        num: &mut num,
                        den: &mut den,
                    },
                );
                Some(rational(num, neg, den))
            }
            Op::Mul | Op::Div => {
                // `x·y`, `x/y = x·(D/C)`, `y/x = y·(b/a)`.
                let (a, b, l) = match (op, short_left) {
                    (Op::Mul, _) => (x.num, x.den, ly),
                    (Op::Div, true) => (
                        x.num,
                        x.den,
                        Long {
                            negative: ly.negative,
                            num: ly.den,
                            den: ly.num,
                        },
                    ),
                    (Op::Div, false) => (x.den, x.num, ly),
                    _ => unreachable!(),
                };
                if a == 0 {
                    return Some(Rational::from_word(false, 0, 1, order));
                }
                let n = product_limbs(l);
                room(&mut num, n);
                room(&mut den, n);
                product_into(
                    a,
                    b,
                    l,
                    Target {
                        num: &mut num,
                        den: &mut den,
                    },
                );
                Some(rational(num, x.negative != ly.negative, den))
            }
        }
    }
}

/// `y = y op x` (`y = x op y` when `short_left` is set) for `y` in numbers and `x` in words, in
/// the vectors of `y`, when this path reads `y` and `order` is little-endian; `false` otherwise,
/// with `y` as it was. A division by zero is the caller's to take apart.
#[inline]
pub(super) fn assign_long(
    y: &mut Rational,
    x: Small,
    op: Op,
    short_left: bool,
    order: ByteOrder,
) -> bool {
    if order != ByteOrder::LittleEndian {
        return false;
    }
    let Some(ly) = readable(y) else {
        return false;
    };
    match op {
        Op::Add | Op::Sub => {
            if x.num == 0 {
                if op == Op::Sub && short_left {
                    y.negate();
                }
                return true;
            }
            let (sx, yl) = match (op, short_left) {
                (Op::Sub, true) => (
                    x.negative,
                    Long {
                        negative: !ly.negative,
                        ..ly
                    },
                ),
                (Op::Sub, false) => (!x.negative, ly),
                _ => (x.negative, ly),
            };
            let n = sum_limbs(yl);
            let (yn, yd) = y.numbers_mut().unwrap_or_else(|| unreachable!());
            room_keep(&mut yn.value, n);
            room_keep(&mut yd.value, n);
            // The parts again, from the vectors that now have room.
            let yl = Long {
                negative: yl.negative,
                num: Part {
                    p: yn.value.as_ptr(),
                    len: yl.num.len,
                },
                den: Part {
                    p: yd.value.as_ptr(),
                    len: yl.den.len,
                },
            };
            // SAFETY: the vectors have room for the result's limbs, and each pass reads a limb
            // before it writes it.
            let neg = unsafe {
                sum_into(
                    sx,
                    x.num,
                    x.den,
                    yl,
                    Target {
                        num: &mut yn.value,
                        den: &mut yd.value,
                    },
                )
            };
            yn.negative = neg && !(yn.value.len() == 1 && yn.value[0] == 0);
            yd.negative = false;
            y.settle();
            true
        }
        Op::Mul | Op::Div => {
            let swap = op == Op::Div && short_left;
            let (a, b) = if op == Op::Div && !short_left {
                (x.den, x.num)
            } else {
                (x.num, x.den)
            };
            if a == 0 {
                *y = Rational::from_word(false, 0, 1, order);
                return true;
            }
            let negative = x.negative != ly.negative;
            if swap {
                y.invert();
            }
            let n = product_limbs(ly);
            let (yn, yd) = y.numbers_mut().unwrap_or_else(|| unreachable!());
            room_keep(&mut yn.value, n);
            room_keep(&mut yd.value, n);
            let l = Long {
                negative,
                num: Part::of(yn),
                den: Part::of(yd),
            };
            // SAFETY: as for the sum.
            unsafe {
                product_into(
                    a,
                    b,
                    l,
                    Target {
                        num: &mut yn.value,
                        den: &mut yd.value,
                    },
                );
            }
            yn.negative = negative;
            yd.negative = false;
            y.settle();
            true
        }
    }
}

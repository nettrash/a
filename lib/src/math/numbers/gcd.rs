//! Greatest common divisors of [`Number`]s: [`Number::gcd`], [`Number::lcm`],
//! [`Number::gcd_ext`] and [`Number::mod_inverse`].
//!
//! Operands whose vectors both hold at most 16 bytes take short paths in registers: the binary
//! method for the greatest common divisor, and for the cofactors Euclid's algorithm on two limbs,
//! with the steps that Lehmer's method finds from the top limbs, then on one.
//!
//! Longer operands are read into limbs of eight bytes, native `u64`s, the least significant first,
//! in working memory on the stack, or on the heap for the longest (see [`limbs`]). A much longer
//! operand is first divided by the shorter one, and the remainder takes its place. From there the
//! numbers are reduced by Lehmer's method: the quotients of many of Euclid's steps are found from
//! the top two limbs of the numbers by Möller's hgcd2, and their matrix is applied to the whole
//! numbers in one pass, two limbs read and two written a position. The greatest common divisor
//! alone shifts the zero bits at the low end out of both numbers first, which keeps it odd, and
//! finishes on two limbs by the binary method; the extended algorithm keeps a row of the matrices'
//! product, the cofactor of one number, and finishes on one limb by Euclid's divisions. Numbers of
//! [`GCD_DC_THRESHOLD`] limbs and more (of [`GCDEXT_DC_THRESHOLD`] for the cofactors) are first
//! reduced by Möller's subquadratic half gcd on the multiplication (see [`hgcd`]).
//!
//! `lcm` divides the shorter operand exactly by the greatest common divisor and multiplies the
//! quotient by the longer one; `mod_inverse` takes the extended algorithm on the operand modulo
//! the modulus and the modulus, and keeps the cofactor of the operand.

// The kernels under the name `kernels`, chosen as those of the multiplication are: assembly on
// 64-bit little-endian aarch64 where the system allows unaligned accesses, and portable Rust
// elsewhere and under Miri.
mod hgcd;
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
#[path = "gcd/aarch64.rs"]
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
#[path = "gcd/portable.rs"]
mod kernels;
mod limbs;

use super::add::{read_small, significant_len, small, write_small};
use super::mul::limbs::mul;
use super::mul::{Operand, Product, STACK_LIMBS, on_heap, on_stack, to_limbs};
use super::{ByteOrder, Number};
use core::mem::MaybeUninit;
use core::ptr;
use hgcd::{GCD_DC_THRESHOLD, GCDEXT_DC_THRESHOLD, gcd_dc, gcd_dc_itch, gcdext, gcdext_dc_itch};
use limbs::{
    Row, Step, add_to, divexact, divexact_itch, gcd_11, gcd_lehmer, gcd_u128, gcdext_1, hgcd2,
    lehmer_ext_itch, lshift, mul_bound, normalized, rem_1, rshift, sub_from, tdiv_itch, tdiv_qr,
    with_scratch,
};

impl Number {
    /// Returns the greatest common divisor of `self` and `other`, in the byte order of `self`, in
    /// the shortest form: the largest number that divides both, never negative.
    ///
    /// As for GMP's `mpz_gcd`, `gcd(x, 0) = |x|`, and `gcd(0, 0) = 0`. Signs do not count. Both
    /// operands are left as they were, and the result is built in one new vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i32(-84, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(36, ByteOrder::LittleEndian);
    /// let g = a.gcd(&b);
    /// assert_eq!(g.value, [12]);
    /// assert!(!g.is_negative());
    /// assert!(g.byte_order() == ByteOrder::BigEndian);
    /// ```
    pub fn gcd(&self, other: &Number) -> Number {
        if small(self, other) {
            // SAFETY: both vectors hold at most 16 bytes.
            let (x, y) = unsafe {
                (
                    read_small(&self.value, self.byte_order),
                    read_small(&other.value, other.byte_order),
                )
            };
            return small_number(self.byte_order, gcd_u128(x, y), false);
        }
        gcd_general(self, other)
    }

    /// Returns the least common multiple of `self` and `other`, in the byte order of `self`, in
    /// the shortest form: the smallest number that both divide, never negative, and zero when
    /// either is zero, as GMP's `mpz_lcm` gives it.
    ///
    /// It is computed as `|a| / gcd(a, b) · |b|`, the shorter operand divided exactly by the
    /// greatest common divisor, and the quotient multiplied by the longer one.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// let a = Number::new_from_i32(-4, ByteOrder::LittleEndian);
    /// let b = Number::new_from_i32(6, ByteOrder::BigEndian);
    /// let l = a.lcm(&b);
    /// assert_eq!(l.value, [12]);
    /// assert!(!l.is_negative());
    /// ```
    pub fn lcm(&self, other: &Number) -> Number {
        if small(self, other) {
            // SAFETY: both vectors hold at most 16 bytes.
            let (x, y) = unsafe {
                (
                    read_small(&self.value, self.byte_order),
                    read_small(&other.value, other.byte_order),
                )
            };
            return small_lcm(self.byte_order, x, y);
        }
        lcm_general(self, other)
    }

    /// Returns `(g, x, y)`: the greatest common divisor `g` of `self` and `other`, never negative,
    /// and cofactors with `g = self · x + other · y`, all three in the byte order of `self` and in
    /// the shortest form.
    ///
    /// The cofactors are those of GMP's `mpz_gcdext`: `|x| < |other| / 2g` and `|y| < |self| / 2g`,
    /// which define them uniquely, but for a few cases. When `|self| = |other|`, `x = 0` and `y =
    /// sgn(other)`; otherwise `x = sgn(self)` when `other = 0` or `|other| = 2g`, and `y =
    /// sgn(other)` when `self = 0` or `|self| = 2g`. So `x = 0` exactly when `g = |other|`, and
    /// `gcd_ext(0, 0) = (0, 0, 0)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // gcd(240, -46) = 2 = 240 · (-9) + (-46) · (-47).
    /// let a = Number::new_from_i32(240, ByteOrder::BigEndian);
    /// let b = Number::new_from_i32(-46, ByteOrder::BigEndian);
    /// let (g, x, y) = a.gcd_ext(&b);
    /// assert_eq!(g.value, [2]);
    /// assert_eq!((x.is_negative(), x.value), (true, vec![9]));
    /// assert_eq!((y.is_negative(), y.value), (true, vec![47]));
    /// ```
    pub fn gcd_ext(&self, other: &Number) -> (Number, Number, Number) {
        if small(self, other) {
            // SAFETY: both vectors hold at most 16 bytes.
            let (x, y) = unsafe {
                (
                    read_small(&self.value, self.byte_order),
                    read_small(&other.value, other.byte_order),
                )
            };
            return small_gcd_ext(self.byte_order, x, self.negative, y, other.negative);
        }
        gcd_ext_general(self, other)
    }

    /// Returns the inverse of `self` modulo `|modulus|`: the `r` with `0 <= r < |modulus|` and
    /// `self · r = 1` modulo `|modulus|`, in the byte order of `self` and in the shortest form, or
    /// `None` when there is none, which is when `self` and `modulus` have a common divisor besides
    /// 1. As for GMP's `mpz_invert`, every number has the inverse 0 modulo 1.
    ///
    /// # Panics
    ///
    /// When `modulus` is zero, with the message of `%`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::{ByteOrder, Number};
    ///
    /// // 3 · 4 = 12 = 1 modulo 11, and -3 · 7 = -21 = 1 modulo 11.
    /// let m = Number::new_from_i32(11, ByteOrder::BigEndian);
    /// let r = Number::new_from_i32(3, ByteOrder::BigEndian).mod_inverse(&m).unwrap();
    /// assert_eq!(r.value, [4]);
    /// let r = Number::new_from_i32(-3, ByteOrder::BigEndian).mod_inverse(&m).unwrap();
    /// assert_eq!(r.value, [7]);
    ///
    /// // 6 and 9 have the common divisor 3.
    /// let m = Number::new_from_i32(9, ByteOrder::BigEndian);
    /// assert!(Number::new_from_i32(6, ByteOrder::BigEndian).mod_inverse(&m).is_none());
    /// ```
    #[track_caller]
    pub fn mod_inverse(&self, modulus: &Number) -> Option<Number> {
        if small(self, modulus) {
            // SAFETY: both vectors hold at most 16 bytes.
            let (x, n) = unsafe {
                (
                    read_small(&self.value, self.byte_order),
                    read_small(&modulus.value, modulus.byte_order),
                )
            };
            if n == 0 {
                by_zero();
            }
            return small_inverse(x, self.negative, n)
                .map(|r| small_number(self.byte_order, r, false));
        }
        inverse_general(self, modulus)
    }
}

/// Panics for a modulus of zero, with the message of `%`.
#[cold]
#[inline(never)]
#[track_caller]
fn by_zero() -> ! {
    panic!("attempt to calculate the remainder with a divisor of zero")
}

// Numbers of up to 16 bytes

/// The magnitude `m` with the sign `negative`, which zero does not keep, as a number in one new
/// vector of exactly its length in the byte order `order`.
#[inline]
fn small_number(order: ByteOrder, m: u128, negative: bool) -> Number {
    let len = 16 - (m | 1).leading_zeros() as usize / 8;
    let mut v = Vec::with_capacity(len);
    // SAFETY: the new vector has room for the `len` bytes, from 1 to 16, all of which
    // `write_small` writes before the length is set.
    unsafe {
        write_small(v.as_mut_ptr(), order, m as u64, (m >> 64) as u64, len);
        v.set_len(len);
    }
    Number {
        value: v,
        negative: negative && m != 0,
        byte_order: order,
    }
}

/// The least common multiple of the magnitudes `x` and `y`, of up to 16 bytes each, as a number in
/// the byte order `order`: the one divided by the greatest common divisor, by the hardware's
/// division when both fit in a limb, and the quotient times the other, in four limbs.
fn small_lcm(order: ByteOrder, x: u128, y: u128) -> Number {
    let g = gcd_u128(x, y);
    if g == 0 {
        return small_number(order, 0, false);
    }
    let q = if (x | g) >> 64 == 0 {
        u128::from(x as u64 / g as u64)
    } else {
        x / g
    };
    let (q0, q1, y0, y1) = (q as u64, (q >> 64) as u64, y as u64, (y >> 64) as u64);
    let mul = |a: u64, b: u64| u128::from(a) * u128::from(b);
    let (p00, p01, p10, p11) = (mul(q0, y0), mul(q0, y1), mul(q1, y0), mul(q1, y1));
    let c1 = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
    let c2 = (c1 >> 64) + (p01 >> 64) + (p10 >> 64) + u128::from(p11 as u64);
    let c3 = (c2 >> 64) + (p11 >> 64);
    let limbs = [p00 as u64, c1 as u64, c2 as u64, c3 as u64];
    // SAFETY: the four limbs are valid for reads while the number is written.
    unsafe { limbs_number(limbs.as_ptr(), 4, false, order) }
}

/// A cofactor of up to two limbs: its magnitude and whether it is negative.
type Signed = (u128, bool);

/// `x - q y` for cofactors held as `i128`s, whose values all fit.
#[inline(always)]
fn less_times(x: i128, q: u128, y: i128) -> i128 {
    x.wrapping_sub((q as i128).wrapping_mul(y))
}

/// The greatest common divisor `g` of the magnitudes `x` and `y`, of up to two limbs each and not
/// zero, and the cofactors `s` and `t` of `g = s x + t y` that [`Number::gcd_ext`] promises:
/// Euclid's algorithm on two limbs, with the steps that [`hgcd2`] finds from the numbers shifted
/// together applied at once, then on one limb by [`gcdext_1`], keeping the cofactors of both
/// numbers.
fn gcdext_u128(x: u128, y: u128) -> (u128, i128, i128) {
    let (mut a, mut b) = (x, y);
    // a = sa x + ta y and b = sb x + tb y; every cofactor fits in an `i128`, as each is at most
    // half the other number.
    let (mut sa, mut ta, mut sb, mut tb) = (1i128, 0i128, 0i128, 1i128);
    while (a | b) >> 64 != 0 {
        let s = (a | b).leading_zeros();
        let (ah, bh) = (a << s, b << s);
        if let Some(m) = hgcd2((ah >> 64) as u64, ah as u64, (bh >> 64) as u64, bh as u64) {
            // The reduced numbers fit in two limbs, so they are right modulo 2^128, as are the
            // cofactors, which fit as well.
            let (u00, u01, u10, u11) = (
                u128::from(m.u00),
                u128::from(m.u01),
                u128::from(m.u10),
                u128::from(m.u11),
            );
            (a, b) = (
                u11.wrapping_mul(a).wrapping_sub(u01.wrapping_mul(b)),
                u00.wrapping_mul(b).wrapping_sub(u10.wrapping_mul(a)),
            );
            let (i00, i01, i10, i11) = (u00 as i128, u01 as i128, u10 as i128, u11 as i128);
            (sa, sb) = (
                i11.wrapping_mul(sa).wrapping_sub(i01.wrapping_mul(sb)),
                i00.wrapping_mul(sb).wrapping_sub(i10.wrapping_mul(sa)),
            );
            (ta, tb) = (
                i11.wrapping_mul(ta).wrapping_sub(i01.wrapping_mul(tb)),
                i00.wrapping_mul(tb).wrapping_sub(i10.wrapping_mul(ta)),
            );
            continue;
        }
        // One step of Euclid's: the larger number modulo the smaller.
        if a == b {
            return smaller_pair(a, (sa, ta), (sb, tb));
        }
        if a > b {
            let q = a / b;
            a -= q * b;
            if a == 0 {
                return (b, sb, tb);
            }
            sa = less_times(sa, q, sb);
            ta = less_times(ta, q, tb);
        } else {
            let q = b / a;
            b -= q * a;
            if b == 0 {
                return (a, sa, ta);
            }
            sb = less_times(sb, q, sa);
            tb = less_times(tb, q, ta);
        }
    }
    if a == b {
        return smaller_pair(a, (sa, ta), (sb, tb));
    }
    let (g, u, v) = gcdext_1(a as u64, b as u64);
    let (u, v) = (i128::from(u), i128::from(v));
    (
        u128::from(g),
        u.wrapping_mul(sa).wrapping_add(v.wrapping_mul(sb)),
        u.wrapping_mul(ta).wrapping_add(v.wrapping_mul(tb)),
    )
}

/// The greatest common divisor `g` when both numbers of [`gcdext_u128`] are `g`, with the pair of
/// cofactors of the two whose first cofactor is the smaller, or the positive one when they are
/// equal in magnitude: Euclid's.
#[inline]
fn smaller_pair(g: u128, a: (i128, i128), b: (i128, i128)) -> (u128, i128, i128) {
    let (ua, ub) = (a.0.unsigned_abs(), b.0.unsigned_abs());
    let pick_b = ub < ua || (ub == ua && b.0 > 0);
    let (s, t) = if pick_b { b } else { a };
    (g, s, t)
}

/// [`Number::gcd_ext`] of operands of up to 16 bytes, magnitudes `x` and `y` with their signs, in
/// the byte order `order`.
fn small_gcd_ext(
    order: ByteOrder,
    x: u128,
    negative_x: bool,
    y: u128,
    negative_y: bool,
) -> (Number, Number, Number) {
    let sgn = |m: u128, negative: bool| -> Signed { (u128::from(m != 0), negative) };
    let (g, s, t) = if y == 0 {
        (x, sgn(x, negative_x), (0, false))
    } else if x == 0 {
        (y, (0, false), sgn(y, negative_y))
    } else if x == y {
        (x, (0, false), (1, negative_y))
    } else {
        let (g, s, t) = gcdext_u128(x, y);
        (
            g,
            (s.unsigned_abs(), (s < 0) != negative_x),
            (t.unsigned_abs(), (t < 0) != negative_y),
        )
    };
    (
        small_number(order, g, false),
        small_number(order, s.0, s.1),
        small_number(order, t.0, t.1),
    )
}

/// The inverse of `x`, of sign `negative`, modulo `n`, not zero, of up to two limbs each, or
/// `None`.
fn small_inverse(x: u128, negative: bool, n: u128) -> Option<u128> {
    if n == 1 {
        return Some(0);
    }
    let x = x % n;
    if x == 0 {
        return None;
    }
    let (g, s, _) = gcdext_u128(x, n);
    if g != 1 {
        return None;
    }
    let r = if s < 0 {
        n - s.unsigned_abs()
    } else {
        s as u128
    };
    Some(if negative && r != 0 { n - r } else { r })
}

// Writing results

/// The `n` limbs at `r` as a number of sign `negative` in the byte order `order`, in one new vector
/// of exactly its length; zero, which is never negative, when they are all zero.
///
/// # Safety
///
/// The limbs must be valid for reads.
unsafe fn limbs_number(r: *const u64, n: usize, negative: bool, order: ByteOrder) -> Number {
    // SAFETY: as the caller guarantees.
    unsafe {
        let top = normalized(r, n);
        let p = if top == 0 {
            Product {
                r: ptr::null(),
                len: 1,
                negative: false,
            }
        } else {
            Product {
                r,
                len: 8 * top - (*r.add(top - 1)).leading_zeros() as usize / 8,
                negative,
            }
        };
        let mut v = Vec::with_capacity(p.len);
        p.write(v.as_mut_ptr(), order);
        v.set_len(p.len);
        Number {
            value: v,
            negative: p.negative,
            byte_order: order,
        }
    }
}

/// The magnitude of `n` with the sign `negative`, which zero does not keep, in the byte order
/// `order`, in the shortest form, in one new vector.
fn magnitude(n: &Number, negative: bool, order: ByteOrder) -> Number {
    let be = n.byte_order == ByteOrder::BigEndian;
    let len = if be {
        significant_len::<true>(&n.value)
    } else {
        significant_len::<false>(&n.value)
    };
    let bytes = if be {
        &n.value[n.value.len() - len..]
    } else {
        &n.value[..len]
    };
    let value = if len == 0 {
        vec![0]
    } else if n.byte_order == order {
        bytes.to_vec()
    } else {
        bytes.iter().rev().copied().collect()
    };
    Number {
        value,
        negative: negative && len != 0,
        byte_order: order,
    }
}

/// The sign of a number of sign `negative` and magnitude not zero when `nonzero`, as a number:
/// -1, 0 or 1, in the byte order `order`.
fn sign(nonzero: bool, negative: bool, order: ByteOrder) -> Number {
    small_number(order, u128::from(nonzero), negative)
}

// Working memory

/// Runs `f` on `need` limbs of working memory, on the stack up to [`STACK_LIMBS`], aligned to 16
/// bytes.
#[inline(always)]
fn working<R>(need: usize, f: impl FnOnce(*mut u64) -> R) -> R {
    let run = |s: *mut MaybeUninit<u64>| f(s.cast());
    if need <= STACK_LIMBS {
        on_stack(run)
    } else {
        on_heap(need, run)
    }
}

/// Shifts the zero bits at the low end out of the `n` limbs at `p`, not all zero, in place, and
/// returns the limbs left without those at the top that become zero, and the number of bits.
///
/// # Safety
///
/// The limbs must be valid for reads and writes.
unsafe fn strip(p: *mut u64, n: usize) -> (usize, usize) {
    // SAFETY: as the caller guarantees: a limb that is not zero comes before the end.
    unsafe {
        let mut skip = 0;
        while *p.add(skip) == 0 {
            skip += 1;
        }
        let bits = (*p.add(skip)).trailing_zeros();
        let m = n - skip;
        if skip > 0 {
            ptr::copy(p.add(skip), p, m);
        }
        if bits > 0 {
            rshift(p, p, m, bits);
        }
        (normalized(p, m), 64 * skip + bits as usize)
    }
}

// The greatest common divisor

/// `a.gcd(b)` for operands of which at least one has more than 16 bytes.
#[inline(never)]
fn gcd_general(a: &Number, b: &Number) -> Number {
    let order = a.byte_order;
    let (x, y) = (Operand::of(a), Operand::of(b));
    if x.significant == 0 {
        return magnitude(b, false, order);
    }
    if y.significant == 0 {
        return magnitude(a, false, order);
    }
    let (nx, ny) = (x.significant.div_ceil(8), y.significant.div_ceil(8));
    let (long, short) = (nx.max(ny), nx.min(ny));
    // Both numbers with a limb to spare, the result shifted with a limb above it and its zero
    // limbs, and the scratch of the first division and of the steps.
    let need = (nx + 1) + (ny + 1) + (long + 2) + gcd_itch(long, short);
    working(need, |s| {
        // SAFETY: the working memory holds the limbs counted above; the operands are borrowed.
        unsafe {
            let (pa, pb) = (s, s.add(nx + 1));
            let r = pb.add(ny + 1);
            let t = r.add(long + 2);
            to_limbs(x, pa);
            to_limbs(y, pb);
            let (na, za) = strip(pa, nx);
            let (nb, zb) = strip(pb, ny);
            let k = za.min(zb);
            let (pa, na, pb, nb) = if na >= nb {
                (pa, na, pb, nb)
            } else {
                (pb, nb, pa, na)
            };
            // The result's zero limbs, then the odd greatest common divisor shifted by the bits.
            let (zl, zbits) = (k / 64, (k % 64) as u32);
            let g = r.add(zl);
            let gn = gcd_limbs(g, pa, na, pb, nb, t, gcd_itch(long, short));
            ptr::write_bytes(r, 0, zl);
            *g.add(gn) = lshift(g, g, gn, zbits);
            limbs_number(r, zl + gn + 1, false, order)
        }
    })
}

/// The limbs of scratch space that [`gcd_limbs`] needs for numbers of `na >= nb` limbs.
fn gcd_itch(na: usize, nb: usize) -> usize {
    let lehmer = (na - nb + 1) + tdiv_itch(na, nb).max(nb + 1 + tdiv_itch(nb, 1));
    if nb >= GCD_DC_THRESHOLD {
        lehmer.max(gcd_dc_itch(nb))
    } else {
        lehmer
    }
}

/// The greatest common divisor of the odd numbers of `na >= nb` limbs at `a` and `b`, whose top
/// limbs are not zero, written at `g`; returns its number of limbs. The numbers are clobbered.
///
/// # Safety
///
/// The limbs at `a` and `b` must be valid for reads and writes, the `nb` at `g` for writes, and
/// the `avail` limbs at `t` for reads and writes, at least [`gcd_itch`] of them, all apart.
unsafe fn gcd_limbs(
    g: *mut u64,
    a: *mut u64,
    na: usize,
    b: *mut u64,
    nb: usize,
    t: *mut u64,
    avail: usize,
) -> usize {
    // SAFETY: as the caller guarantees.
    unsafe {
        if nb == 1 {
            let d = *b;
            let r = if na == 1 && *a < d {
                *a
            } else {
                rem_1(a, na, d, t)
            };
            *g = if r == 0 {
                d
            } else {
                gcd_11(r >> r.trailing_zeros(), d)
            };
            return 1;
        }
        if na > nb {
            // The scratch was planned for the lengths before the zero bits at the low end were
            // shifted out, which can leave the divisor much shorter: a longer quotient and a
            // division that needs more scratch take it from `with_scratch`.
            let qn = na - nb + 1;
            with_scratch(t, avail, qn + tdiv_itch(na, nb), |t| {
                tdiv_qr(t, a, na, b, nb, t.add(qn));
            });
            if normalized(a, nb) == 0 {
                ptr::copy_nonoverlapping(b, g, nb);
                return nb;
            }
        }
        let (mut a, mut b, mut n) = (a, b, nb);
        if n >= GCD_DC_THRESHOLD {
            match gcd_dc(&mut a, &mut b, n, t, avail) {
                Step::Size(k) => n = k,
                Step::Gcd(p, k) => {
                    ptr::copy(p, g, k);
                    return k;
                }
            }
        }
        gcd_lehmer(g, a, b, n, t, avail)
    }
}

// The least common multiple

/// `a.lcm(b)` for operands of which at least one has more than 16 bytes: the shorter one divided
/// exactly by the greatest common divisor, times the longer one.
#[inline(never)]
fn lcm_general(a: &Number, b: &Number) -> Number {
    let order = a.byte_order;
    let (x, y) = (Operand::of(a), Operand::of(b));
    if x.significant == 0 || y.significant == 0 {
        return small_number(order, 0, false);
    }
    let g = a.gcd(b);
    let one = g.value == [1];
    let mut l = if x.significant <= y.significant {
        let q = if one {
            magnitude(a, false, order)
        } else {
            a.div_exact(&g)
        };
        q * b
    } else {
        let q = if one {
            magnitude(b, false, b.byte_order)
        } else {
            b.div_exact(&g)
        };
        a * q
    };
    l.negative = false;
    l
}

// The extended algorithm

/// The scratch of the half gcd for the extended algorithm on numbers of `n` limbs: none below its
/// threshold, so that the shorter numbers keep their working memory on the stack.
fn ext_dc_itch(n: usize) -> usize {
    if n >= GCDEXT_DC_THRESHOLD {
        gcdext_dc_itch(n)
    } else {
        0
    }
}

/// `a.gcd_ext(b)` for operands of which at least one has more than 16 bytes.
#[inline(never)]
fn gcd_ext_general(a: &Number, b: &Number) -> (Number, Number, Number) {
    let order = a.byte_order;
    let (x, y) = (Operand::of(a), Operand::of(b));
    if y.significant == 0 {
        return (
            magnitude(a, false, order),
            sign(x.significant != 0, a.negative, order),
            sign(false, false, order),
        );
    }
    if x.significant == 0 {
        return (
            magnitude(b, false, order),
            sign(false, false, order),
            sign(true, b.negative, order),
        );
    }
    let (nx, ny) = (x.significant.div_ceil(8), y.significant.div_ceil(8));
    // The cofactor of the longer operand is the one computed, as GMP's `mpz_gcdext` does: it is
    // the shorter of the two.
    let swap = nx < ny;
    let (p, q, negative_p, negative_q) = if swap {
        (y, x, b.negative, a.negative)
    } else {
        (x, y, a.negative, b.negative)
    };
    let (np, nq) = if swap { (ny, nx) } else { (nx, ny) };
    let (g, cp, cq) = cofactors(p, np, q, nq, order);
    let cp = Number {
        negative: cp.negative != negative_p && cp.value != [0],
        ..cp
    };
    let cq = Number {
        negative: cq.negative != negative_q && cq.value != [0],
        ..cq
    };
    if swap { (g, cq, cp) } else { (g, cp, cq) }
}

/// The greatest common divisor of the magnitudes of the operands `p` and `q`, not zero, of `np >=
/// nq` limbs, and their cofactors, in the byte order `order`: the cofactor of `p` by Lehmer's
/// method on `p` modulo `q` and `q`, and the one of `q` from it, `(g - cp p) / q`, an exact
/// division.
fn cofactors(
    p: Operand,
    np: usize,
    q: Operand,
    nq: usize,
    order: ByteOrder,
) -> (Number, Number, Number) {
    let ualloc = nq + 4;
    // The numbers; the row; the greatest common divisor and the cofactor; the product of the
    // cofactor and `p`, the other cofactor; and the scratch.
    let scratch = lehmer_ext_itch(nq)
        .max(ext_dc_itch(nq))
        .max((np - nq + 1) + tdiv_itch(np, nq))
        .max(mul_bound(np, ualloc))
        .max(divexact_itch(np + ualloc + 1, nq));
    let need = (np + 1)
        + (nq + 1)
        + 2 * ualloc
        + nq
        + (ualloc + 1)
        + (np + ualloc + 1)
        + (np + 2)
        + scratch;
    working(need, |s| {
        // SAFETY: the working memory holds the limbs counted above, in that order; the operands are
        // borrowed.
        unsafe {
            let pa = s;
            let pb = pa.add(np + 1);
            let u0 = pb.add(nq + 1);
            let u1 = u0.add(ualloc);
            let g = u1.add(ualloc);
            let c = g.add(nq);
            let prod = c.add(ualloc + 1);
            let v = prod.add(np + ualloc + 1);
            let t = v.add(np + 2);
            to_limbs(p, pa);
            to_limbs(q, pb);
            if np > nq {
                let qn = np - nq + 1;
                tdiv_qr(t, pa, np, pb, nq, t.add(qn));
                if normalized(pa, nq) == 0 {
                    // `q` divides `p`: g = |q| = 0 |p| + 1 |q|.
                    let gq = limbs_number(pb, nq, false, order);
                    return (gq, sign(false, false, order), sign(true, false, order));
                }
            }
            ptr::write_bytes(u0, 0, 2 * ualloc);
            *u1 = 1;
            let mut row = Row { u0, u1, un: 1 };
            let (gn, cn, negative) = gcdext(g, c, pa, pb, nq, &mut row, t, scratch);
            let gq = limbs_number(g, gn, false, order);
            let cp = limbs_number(c, cn, negative, order);
            // The cofactor of `q`: (g - c |p|) / |q|, whose sign is the opposite of `c`'s.
            if cn == 0 {
                // g = |q|.
                return (gq, cp, sign(true, false, order));
            }
            to_limbs(p, pa);
            to_limbs(q, pb);
            let (long, short) = if np >= cn {
                ((pa, np), (c, cn))
            } else {
                ((c, cn), (pa, np))
            };
            let pn = np + cn;
            mul(prod, long.0, long.1, short.0, short.1, t);
            if negative {
                add_to(prod, prod, pn, g, gn);
            } else {
                sub_from(prod, prod, pn, g, gn);
            }
            let pn = normalized(prod, pn);
            if pn == 0 {
                return (gq, cp, sign(false, false, order));
            }
            divexact(v, prod, pn, pb, nq, t, scratch);
            let cq = limbs_number(v, pn - nq + 1, !negative, order);
            (gq, cp, cq)
        }
    })
}

// The inverse

/// `a.mod_inverse(n)` for operands of which at least one has more than 16 bytes.
#[inline(never)]
#[track_caller]
fn inverse_general(a: &Number, modulus: &Number) -> Option<Number> {
    let order = a.byte_order;
    let (x, m) = (Operand::of(a), Operand::of(modulus));
    if m.significant == 0 {
        by_zero();
    }
    let nm = m.significant.div_ceil(8);
    let nx = x.significant.div_ceil(8);
    let ualloc = nm + 4;
    let scratch = lehmer_ext_itch(nm)
        .max(ext_dc_itch(nm))
        .max((nx.max(nm) - nm + 1) + tdiv_itch(nx.max(nm), nm));
    let need = (nx.max(nm) + 1) + (nm + 1) + 2 * ualloc + nm + (ualloc + 1) + scratch;
    working(need, |s| {
        // SAFETY: the working memory holds the limbs counted above, in that order; the operands are
        // borrowed.
        unsafe {
            let pa = s;
            let pm = pa.add(nx.max(nm) + 1);
            let u0 = pm.add(nm + 1);
            let u1 = u0.add(ualloc);
            let g = u1.add(ualloc);
            let c = g.add(nm);
            let t = c.add(ualloc + 1);
            to_limbs(m, pm);
            if nm == 1 && *pm == 1 {
                return Some(small_number(order, 0, false));
            }
            if x.significant == 0 {
                return None;
            }
            to_limbs(x, pa);
            let mut na = nx;
            if nx >= nm {
                let qn = nx - nm + 1;
                tdiv_qr(t, pa, nx, pm, nm, t.add(qn));
                na = normalized(pa, nm);
                if na == 0 {
                    return None;
                }
            }
            ptr::write_bytes(pa.add(na), 0, nm - na);
            ptr::write_bytes(u0, 0, 2 * ualloc);
            *u1 = 1;
            let mut row = Row { u0, u1, un: 1 };
            let (gn, cn, negative) = gcdext(g, c, pa, pm, nm, &mut row, t, scratch);
            if gn != 1 || *g != 1 {
                return None;
            }
            // The inverse of |a| is c, of a it is c with the sign of a; below zero, |m| more.
            let negative = negative != a.negative;
            if negative {
                to_limbs(m, pm);
                sub_from(pm, pm, nm, c, cn);
                return Some(limbs_number(pm, nm, false, order));
            }
            Some(limbs_number(c, cn, false, order))
        }
    })
}

//! Prime fields known at compile time: a type implementing [`TypedPrime`] names the prime, and
//! [`TypedFp`] is an element, `Copy`, its `N` limbs and nothing else, so that values stay in
//! registers from one operation to the next. The constants of the arithmetic (`-1/p mod 2⁶⁴`,
//! `R mod p`, `R² mod p`, the exponents and the non-residue of the square roots) are computed by
//! `const fn`s when the type is used, and the products, sums and differences are the kernels of
//! the runtime form [`PrimeField`] takes for the same prime: on aarch64 the no-carry Montgomery
//! products of four and six limbs for a prime with a spare top bit, P-256's and P-384's own
//! reductions, the folds of `Pseudo`'s form for a prime `2^k − c` of four limbs with a small `c`,
//! the carry chains of the sums; the rows of Montgomery's product, or the numbers' kernel and
//! REDC from seven limbs, elsewhere. The limbs are the residue `x R mod p` in Montgomery's form;
//! in `Pseudo`'s form (on aarch64) any four limbs congruent to the value; for Goldilocks's prime
//! `2^64 − 2^32 + 1` the value, on every target, its products reduced by the prime's shape.
//!
//! The inverse and the Legendre symbol leave the form for the value, as in the runtime fields
//! ([`Fp::inverse`], [`Fp::legendre`]): through [`Number`] for two limbs and more, on the word for
//! a prime of one limb (the binary extended Euclid; the Jacobi symbol of words).

use super::prime::{Fp, PrimeField};
use super::prime_chains::{BLS12_381_P, BLS12_381_SQRT};
use super::prime_sqrt::Chain;
use crate::math::ec::field::{Mont, limbs_of, number_of};
use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// A prime of `N` 64-bit limbs known at compile time, the parameter of [`TypedFp`].
///
/// The prime must be odd, of exactly `N` limbs (the top one not zero), and `N` from 1 to 9: a
/// use of [`TypedFp`] with another one fails to compile. That it is prime is not checked at
/// compile time: [`TypedFp::modulus_is_prime`] tests it, and a test of the crate that declares
/// the type should call it (the example of [`typed_prime!`](crate::typed_prime) shows one). A
/// composite modulus gives a ring, whose inverses and roots are wrong.
///
/// [`typed_prime!`](crate::typed_prime) declares a type and its implementation from the prime in
/// hexadecimal; the named fields are in [`named`](super::named).
///
/// # Examples
///
/// ```
/// use a::math::field::{TypedFp, TypedPrime};
///
/// /// The prime 2^61 - 1.
/// #[derive(Clone, Copy, Debug)]
/// struct M61;
///
/// impl TypedPrime<1> for M61 {
///     const MODULUS: [u64; 1] = [(1 << 61) - 1];
///     const NAME: &'static str = "M61";
/// }
///
/// type F = TypedFp<M61, 1>;
/// assert!(F::modulus_is_prime());
/// let x = F::from_u64(1 << 60);
/// assert_eq!(x.double(), F::ONE); // 2^61 = 1
/// ```
pub trait TypedPrime<const N: usize>: 'static {
    /// The prime, the low limb first.
    const MODULUS: [u64; N];
    /// A name, for the documentation and errors.
    const NAME: &'static str;
}

/// Which product kernel a typed field takes besides Montgomery's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum TypedSpecial {
    /// Montgomery's rows (no-carry for a spare top bit).
    None,
    /// P-256's reduction by shifts.
    P256,
    /// P-384's reduction by shifts.
    P384,
}

/// The constants the kernels of a typed field take ([`Mont::typed_mul`] and the like).
#[derive(Clone, Copy, Debug)]
pub(crate) struct TypedKernel<const N: usize> {
    /// The prime.
    pub(crate) m: [u64; N],
    /// `-1 / m mod 2^64`.
    pub(crate) minv: u64,
    /// `-1 / m mod 2^128`, the low limb first.
    pub(crate) minv2: [u64; 2],
    /// Whether the top bit of `m` is clear (read by the aarch64 kernels' choice).
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_endian = "little", not(miri))),
        allow(dead_code)
    )]
    pub(crate) spare: bool,
    /// P-256 or P-384 (read by the aarch64 kernels' choice).
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_endian = "little", not(miri))),
        allow(dead_code)
    )]
    pub(crate) special: TypedSpecial,
}

// ---------------------------------------------------------------- constants at compile time

const P256: [u64; 4] = [0xffffffffffffffff, 0xffffffff, 0, 0xffffffff00000001];
const P384: [u64; 6] = [
    0xffffffff,
    0xffffffff00000000,
    0xfffffffffffffffe,
    0xffffffffffffffff,
    0xffffffffffffffff,
    0xffffffffffffffff,
];

/// Whether the limbs `a` and `b` are the same, of any lengths.
const fn same_limbs(a: &[u64], b: &[u64]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// `a ≥ b`.
const fn ge<const N: usize>(a: &[u64; N], b: &[u64; N]) -> bool {
    let mut i = N;
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return a[i] > b[i];
        }
    }
    true
}

/// `a - b mod 2^(64 N)`.
const fn sub_limbs<const N: usize>(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    let mut r = [0; N];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < N {
        let (d, b1) = a[i].overflowing_sub(b[i]);
        let (d, b2) = d.overflowing_sub(borrow);
        r[i] = d;
        borrow = (b1 | b2) as u64;
        i += 1;
    }
    r
}

/// `x - w` for a word `w` not above `x`.
const fn sub_word<const N: usize>(x: &[u64; N], w: u64) -> [u64; N] {
    let mut b = [0; N];
    b[0] = w;
    sub_limbs(x, &b)
}

/// `x >> k`, `k < 64`.
const fn shr<const N: usize>(x: &[u64; N], k: u32) -> [u64; N] {
    let mut r = [0; N];
    let mut i = 0;
    while i < N {
        r[i] = x[i] >> k;
        if k > 0 && i + 1 < N {
            r[i] |= x[i + 1] << (64 - k);
        }
        i += 1;
    }
    r
}

/// `x >> k` for any `k`.
const fn shr_any<const N: usize>(x: &[u64; N], k: u32) -> [u64; N] {
    let mut r = *x;
    let mut k = k;
    while k >= 63 {
        r = shr(&r, 63);
        k -= 63;
    }
    shr(&r, k)
}

/// `2x mod m`, for `x` below `m`.
const fn double_mod<const N: usize>(x: &[u64; N], m: &[u64; N]) -> [u64; N] {
    let mut r = [0; N];
    let mut carry = 0u64;
    let mut i = 0;
    while i < N {
        let v = x[i];
        r[i] = (v << 1) | carry;
        carry = v >> 63;
        i += 1;
    }
    if carry != 0 || ge(&r, m) {
        sub_limbs(&r, m)
    } else {
        r
    }
}

/// `2^k x mod m`, for `x` below `m`.
const fn shift_mod<const N: usize>(x: &[u64; N], m: &[u64; N], k: usize) -> [u64; N] {
    let mut r = *x;
    let mut i = 0;
    while i < k {
        r = double_mod(&r, m);
        i += 1;
    }
    r
}

/// `-1 / m mod 2^64` for an odd `m`, by Newton's iteration.
const fn minv_of(m: u64) -> u64 {
    let mut x = m;
    let mut i = 0;
    while i < 6 {
        x = x.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(x)));
        i += 1;
    }
    x.wrapping_neg()
}

/// `-1 / m mod 2^128` for an odd `m = m0 + m1 2^64`, the low limb first.
const fn minv2_of(m0: u64, m1: u64) -> [u64; 2] {
    let m = (m0 as u128) | ((m1 as u128) << 64);
    let mut x = m;
    let mut i = 0;
    while i < 7 {
        x = x.wrapping_mul(2u128.wrapping_sub(m.wrapping_mul(x)));
        i += 1;
    }
    let n = x.wrapping_neg();
    [n as u64, (n >> 64) as u64]
}

/// Montgomery's product `a b / R mod m` at compile time (the rows of `Mont`).
const fn mont_mul_const<const N: usize>(
    a: &[u64; N],
    b: &[u64; N],
    m: &[u64; N],
    minv: u64,
) -> [u64; N] {
    let mut t = [0u64; N];
    let mut top = 0u64;
    let mut i = 0;
    while i < N {
        let bi = b[i];
        let mut c = 0u64;
        let mut j = 0;
        while j < N {
            let p = (a[j] as u128) * (bi as u128) + (t[j] as u128) + (c as u128);
            t[j] = p as u64;
            c = (p >> 64) as u64;
            j += 1;
        }
        let s = (top as u128) + (c as u128);
        let (tn, tn1) = (s as u64, (s >> 64) as u64);
        let q = t[0].wrapping_mul(minv);
        let p = (q as u128) * (m[0] as u128) + (t[0] as u128);
        let mut c = (p >> 64) as u64;
        let mut j = 1;
        while j < N {
            let p = (q as u128) * (m[j] as u128) + (t[j] as u128) + (c as u128);
            t[j - 1] = p as u64;
            c = (p >> 64) as u64;
            j += 1;
        }
        let s = (tn as u128) + (c as u128);
        t[N - 1] = s as u64;
        top = tn1 + (s >> 64) as u64;
        i += 1;
    }
    if top != 0 || ge(&t, m) {
        sub_limbs(&t, m)
    } else {
        t
    }
}

/// `x^e` at compile time, for a residue `x`, `one` being `R mod m`: windows of four bits from the
/// top set bit, so that the evaluation stays short.
const fn pow_const<const N: usize>(
    x: &[u64; N],
    e: &[u64; N],
    m: &[u64; N],
    minv: u64,
    one: &[u64; N],
) -> [u64; N] {
    let mut table = [*one; 16];
    let mut k = 1;
    while k < 16 {
        table[k] = mont_mul_const(&table[k - 1], x, m, minv);
        k += 1;
    }
    let mut top = N;
    while top > 0 && e[top - 1] == 0 {
        top -= 1;
    }
    let mut r = *one;
    let mut i = 16 * top;
    while i > 0 {
        i -= 1;
        let mut j = 0;
        while j < 4 {
            r = mont_mul_const(&r, &r, m, minv);
            j += 1;
        }
        let d = ((e[i / 16] >> (4 * (i % 16))) & 15) as usize;
        if d != 0 {
            r = mont_mul_const(&r, &table[d], m, minv);
        }
    }
    r
}

/// The Jacobi symbol `(a / n)` of words, `n` odd.
const fn jacobi_words(mut a: u64, mut n: u64) -> i32 {
    let mut t = 1;
    a %= n;
    while a != 0 {
        while a.is_multiple_of(2) {
            a /= 2;
            if n % 8 == 3 || n % 8 == 5 {
                t = -t;
            }
        }
        core::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            t = -t;
        }
        a %= n;
    }
    if n == 1 { t } else { 0 }
}

/// The Legendre symbol `(z / p)` of a word `z ≥ 2` modulo the odd prime `p` of `N` limbs, by the
/// reciprocity of Jacobi symbols down to words: `(2 / p)` by `p mod 8`, then `(p mod z / z)`.
const fn legendre_word<const N: usize>(z: u64, p: &[u64; N]) -> i32 {
    let p8 = p[0] % 8;
    let mut a = z;
    let mut t = 1;
    while a.is_multiple_of(2) {
        a /= 2;
        if p8 == 3 || p8 == 5 {
            t = -t;
        }
    }
    if a == 1 {
        return t;
    }
    if a % 4 == 3 && p[0] % 4 == 3 {
        t = -t;
    }
    // p mod a, from the top limb.
    let mut r = 0u128;
    let mut i = N;
    while i > 0 {
        i -= 1;
        r = ((r << 64) | (p[i] as u128)) % (a as u128);
    }
    t * jacobi_words(r as u64, a)
}

/// The limbs of the prime `hex` (hexadecimal digits, an optional `0x`, `_` between digits), the
/// low limb first, at compile time, for [`TypedPrime::MODULUS`]: a `const` that does not fit `N`
/// limbs, or a digit that is not one, fails to compile.
///
/// # Panics
///
/// Panics (at compile time in a `const`) on a character that is not a hexadecimal digit or `_`,
/// or on more than `16 N` digits.
///
/// # Examples
///
/// ```
/// use a::math::field::limbs_from_hex;
///
/// const P: [u64; 2] = limbs_from_hex("0x1_0000_0000_0000_0003");
/// assert_eq!(P, [3, 1]);
/// ```
#[must_use]
pub const fn limbs_from_hex<const N: usize>(hex: &str) -> [u64; N] {
    let b = hex.as_bytes();
    let mut start = 0;
    if b.len() >= 2 && b[0] == b'0' && (b[1] == b'x' || b[1] == b'X') {
        start = 2;
    }
    let mut r = [0u64; N];
    let mut digits = 0usize;
    let mut i = b.len();
    while i > start {
        i -= 1;
        let c = b[i];
        let v = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            b'_' => continue,
            _ => panic!("not a hexadecimal digit"),
        };
        assert!(digits < 16 * N, "the prime does not fit its limbs");
        r[digits / 16] |= (v as u64) << (4 * (digits % 16));
        digits += 1;
    }
    r
}

/// The constants of the prime `P`, computed when they are first used for it.
struct C<P, const N: usize>(PhantomData<P>);

impl<P: TypedPrime<N>, const N: usize> C<P, N> {
    /// The properties the arithmetic needs, checked at compile time.
    const CHECK: () = {
        assert!(N >= 1 && N <= 9, "a typed prime has 1 to 9 limbs");
        assert!(
            P::MODULUS[N - 1] != 0,
            "the top limb of a typed prime is not zero"
        );
        assert!(P::MODULUS[0] & 1 == 1, "a typed prime is odd");
        assert!(N > 1 || P::MODULUS[0] > 2, "a typed prime is above 2");
    };
    const MINV: u64 = minv_of(P::MODULUS[0]);
    const KERNEL: TypedKernel<N> = TypedKernel {
        m: P::MODULUS,
        minv: Self::MINV,
        minv2: minv2_of(P::MODULUS[0], if N > 1 { P::MODULUS[1 % N] } else { 0 }),
        spare: P::MODULUS[N - 1] >> 63 == 0,
        special: if same_limbs(&P::MODULUS, &P256) {
            TypedSpecial::P256
        } else if same_limbs(&P::MODULUS, &P384) {
            TypedSpecial::P384
        } else {
            TypedSpecial::None
        },
    };
    /// `c` of a prime `2^k − c` of four limbs with `f = c 2^s < 2^62` (`s = 256 − k`), which
    /// takes `Pseudo`'s kernels on aarch64 (as the runtime field does): its elements are then any
    /// four limbs congruent to the value; 0 for Montgomery's form.
    const PSEUDO_C: u64 = {
        let k = Self::BITS as usize;
        let s = 64 * N - k;
        let mut two_k = [0u64; N];
        if k < 64 * N {
            two_k[k / 64] = 1 << (k % 64);
        }
        // 2^k − p, wrapping for k = 64 N.
        let c = sub_limbs(&two_k, &P::MODULUS);
        let mut small = true;
        let mut i = 1;
        while i < N {
            small &= c[i] == 0;
            i += 1;
        }
        if cfg!(all(
            target_arch = "aarch64",
            target_endian = "little",
            not(miri)
        )) && N == 4
            && small
            && s < 62
            && c[0] < 1 << (62 - s)
        {
            c[0]
        } else {
            0
        }
    };
    /// Whether the elements are in `Pseudo`'s form (`2^k − c`).
    const PSEUDO: bool = Self::PSEUDO_C != 0;
    /// Whether the prime is Goldilocks's `2^64 − 2^32 + 1`, whose elements are their values,
    /// below `p`, and whose products reduce by its shape (`2^64 = 2^32 − 1`, `2^96 = −1`).
    const GOLDILOCKS: bool = N == 1 && P::MODULUS[0] == 0xffff_ffff_0000_0001;
    /// Whether the elements are values (in `Pseudo`'s or Goldilocks's form), not residues.
    const PLAIN: bool = Self::PSEUDO || Self::GOLDILOCKS;
    /// Whether the prime is BLS12-381's base prime, whose square root takes the runtime field's
    /// plan of the power.
    const BLS12_381: bool = same_limbs(&P::MODULUS, &BLS12_381_P);
    /// `s = 64 N − k`.
    const PSEUDO_S: u32 = 64 * (N as u32) - Self::BITS;
    /// `f = c 2^s = 2^(64 N) mod p`.
    #[cfg_attr(
        not(all(target_arch = "aarch64", target_endian = "little", not(miri))),
        allow(dead_code)
    )]
    const PSEUDO_F: u64 = Self::PSEUDO_C << Self::PSEUDO_S;
    /// One in the elements' form: `R mod m`, or 1.
    const ONE_FORM: [u64; N] = if Self::PLAIN {
        let mut one = [0u64; N];
        one[0] = 1;
        one
    } else {
        Self::ONE
    };
    /// `R mod m`, one, from 1 doubled `64 N` times.
    const ONE: [u64; N] = {
        let mut one = [0u64; N];
        one[0] = 1;
        shift_mod(&one, &P::MODULUS, 64 * N)
    };
    /// `R² mod m`.
    const R2: [u64; N] = shift_mod(&Self::ONE, &P::MODULUS, 64 * N);
    /// The bits of the prime.
    const BITS: u32 = 64 * (N as u32) - P::MODULUS[N - 1].leading_zeros();
    /// The bytes of an encoding.
    const BYTES: usize = (Self::BITS as usize).div_ceil(8);
    /// `(p - 1) / 2`, the value bound of the smaller square root.
    const HALF: [u64; N] = shr(&P::MODULUS, 1);
    /// `s` of `p - 1 = q 2^s`, `q` odd.
    const S: u32 = {
        let pm1 = sub_word(&P::MODULUS, 1);
        let mut i = 0;
        while pm1[i] == 0 {
            i += 1;
        }
        64 * (i as u32) + pm1[i].trailing_zeros()
    };
    /// The exponent of the square root: `(p + 1)/4` for `s = 1`, `(p - 5)/8` for `s = 2`, else
    /// `(q - 1)/2`.
    const E: [u64; N] = match Self::S {
        1 => {
            // p = 4k + 3: (p + 1)/4 = k + 1.
            let k = shr(&P::MODULUS, 2);
            let mut one = [0u64; N];
            one[0] = 1;
            let mut r = [0u64; N];
            let mut carry = 0u64;
            let mut i = 0;
            while i < N {
                let (s, c1) = k[i].overflowing_add(one[i]);
                let (s, c2) = s.overflowing_add(carry);
                r[i] = s;
                carry = (c1 | c2) as u64;
                i += 1;
            }
            r
        }
        2 => shr(&P::MODULUS, 3),
        s => shr(&shr_any(&P::MODULUS, s), 1),
    };
    /// For Tonelli and Shanks's method (`s ≥ 3`): `c = z^q` for the least non-residue `z`.
    const TONELLI_C: [u64; N] = {
        let m = &P::MODULUS;
        let minv = Self::MINV;
        // The least non-residue, by Legendre symbols of words; then one power.
        let mut z = 2u64;
        while legendre_word(z, m) != -1 {
            z += 1;
        }
        let mut zl = [0u64; N];
        zl[0] = z;
        let zr = mont_mul_const(&zl, &Self::R2, m, minv);
        let q = shr_any(&sub_word(m, 1), Self::S);
        let c = pow_const(&zr, &q, m, minv, &Self::ONE);
        if Self::PLAIN {
            // Out of Montgomery's form: REDC of the residue.
            let mut one = [0u64; N];
            one[0] = 1;
            mont_mul_const(&c, &one, m, minv)
        } else {
            c
        }
    };
}

/// The unique element congruent to `a`, below `p = 2^k − c`, `s = 64 N − k` (`Pseudo::canonical`).
#[inline(always)]
fn pseudo_canonical<const N: usize>(a: &[u64; N], c: u64, s: u32) -> [u64; N] {
    let add_1 = |a: &[u64; N], x: u64| {
        let mut r = *a;
        let mut carry = x;
        for l in r.iter_mut() {
            let (v, o) = l.overflowing_add(carry);
            *l = v;
            carry = u64::from(o);
        }
        (r, carry != 0)
    };
    let mut r = *a;
    if s != 0 {
        // a = t 2^k + r = r + t c, below 2^k + f.
        let t = r[N - 1] >> (64 - s);
        r[N - 1] &= u64::MAX >> s;
        r = add_1(&r, t * c).0;
    }
    // At least p exactly when adding c reaches 2^k; then the sum less 2^k is the difference.
    let (mut b, carry) = add_1(&r, c);
    let reached = if s == 0 {
        carry
    } else {
        let top = b[N - 1] >> (64 - s) != 0;
        b[N - 1] &= u64::MAX >> s;
        top
    };
    if reached { b } else { r }
}

// ---------------------------------------------------------------- the elements

/// An element of the prime field of `P`, a prime of `N` limbs known at compile time
/// ([`TypedPrime`]): `Copy`, its `N` limbs in Montgomery's form and no reference to a field, for
/// arithmetic whose values stay in registers. `+`, `-`, `*`, `/` take owned and borrowed
/// operands, with their assignments; `-x` negates; equality and hashing are by the value.
///
/// It computes what [`Fp`] computes for the same prime, with the same kernels on aarch64: the
/// Montgomery forms (four and six limbs with a spare top bit, P-256, P-384) and, for a prime
/// `2^k − c` of four limbs (secp256k1's, 2²⁵⁵ − 19), the folds by `2^256 mod p`, whose elements are
/// then any four limbs congruent to the value; Goldilocks's prime `2^64 − 2^32 + 1` holds its
/// values and reduces its products by its shape, on every target. Other primes, and those on other
/// targets, are in Montgomery's form (P-521, which the runtime field holds in limbs of 58 bits,
/// among them). `to_fp` and `from_fp` convert
/// to and from the runtime field of the same prime.
///
/// # Examples
///
/// ```
/// use a::math::field::PrimeField;
/// use a::math::field::named::Bn254BaseFp;
/// use a::math::numbers::Number;
///
/// type F = Bn254BaseFp;
/// let x = F::from_u64(5);
/// let y = x * x + F::ONE; // values, copied
/// assert_eq!(y, F::from_u64(26));
/// assert_eq!((y / x) * x, y);
/// let r = F::from_u64(25).sqrt().unwrap();
/// assert_eq!(r, F::from_u64(5)); // the smaller root
///
/// // The runtime field of the same prime.
/// let f = PrimeField::bn254_base();
/// assert_eq!(y.to_fp(f), f.from_u64(26));
/// assert_eq!(F::from_fp(&f.from_u64(26)), y);
/// assert_eq!(y.value(), Number::from(26u8));
/// ```
pub struct TypedFp<P, const N: usize> {
    /// The residue `x R mod p`, below `p`, the low limb first; in `Pseudo`'s form, four limbs
    /// congruent to the value; for Goldilocks's prime, the value.
    l: [u64; N],
    p: PhantomData<fn() -> P>,
}

impl<P, const N: usize> Clone for TypedFp<P, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<P, const N: usize> Copy for TypedFp<P, N> {}

impl<P: TypedPrime<N>, const N: usize> PartialEq for TypedFp<P, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.unique() == other.unique()
    }
}

impl<P: TypedPrime<N>, const N: usize> Eq for TypedFp<P, N> {}

impl<P: TypedPrime<N>, const N: usize> Hash for TypedFp<P, N> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        // One to one with the value.
        self.unique().hash(h);
    }
}

impl<P: TypedPrime<N>, const N: usize> TypedFp<P, N> {
    const K: TypedKernel<N> = C::<P, N>::KERNEL;

    /// 0.
    pub const ZERO: Self = Self::of([0; N]);
    /// 1.
    pub const ONE: Self = Self::of(C::<P, N>::ONE_FORM);

    #[inline(always)]
    const fn of(l: [u64; N]) -> Self {
        let () = C::<P, N>::CHECK;
        TypedFp { l, p: PhantomData }
    }

    /// The element of the value with the limbs `x`, below `p`: one product by `R²` (or `x`
    /// itself in `Pseudo`'s form).
    #[inline]
    fn of_value(x: &[u64; N]) -> Self {
        if C::<P, N>::PLAIN {
            return Self::of(*x);
        }
        Self::of(Mont::<N>::typed_mul(&Self::K, x, &C::<P, N>::R2))
    }

    /// The limbs of the value, below `p`: one product by 1 (or the canonical limbs).
    #[inline]
    fn value_limbs(&self) -> [u64; N] {
        if C::<P, N>::GOLDILOCKS {
            return self.l;
        }
        if C::<P, N>::PSEUDO {
            return pseudo_canonical(&self.l, C::<P, N>::PSEUDO_C, C::<P, N>::PSEUDO_S);
        }
        let mut one = [0u64; N];
        one[0] = 1;
        Mont::<N>::typed_mul(&Self::K, &self.l, &one)
    }

    /// Limbs one to one with the value: the residue, or the canonical limbs in `Pseudo`'s form.
    #[inline(always)]
    fn unique(&self) -> [u64; N] {
        if C::<P, N>::PSEUDO {
            return pseudo_canonical(&self.l, C::<P, N>::PSEUDO_C, C::<P, N>::PSEUDO_S);
        }
        self.l
    }

    /// `a b` in the elements' form.
    #[inline(always)]
    fn mul_l(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        if C::<P, N>::GOLDILOCKS {
            let mut r = [0u64; N];
            r[0] = goldilocks_reduce(u128::from(a[0]) * u128::from(b[0]));
            return r;
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if C::<P, N>::PSEUDO {
            return Mont::<N>::typed_pseudo_mul(a, b, C::<P, N>::PSEUDO_F);
        }
        Mont::<N>::typed_mul(&Self::K, a, b)
    }

    /// `a^2` in the elements' form.
    #[inline(always)]
    fn sqr_l(a: &[u64; N]) -> [u64; N] {
        if C::<P, N>::GOLDILOCKS {
            return Self::mul_l(a, a);
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if C::<P, N>::PSEUDO {
            return Mont::<N>::typed_pseudo_sqr(a, C::<P, N>::PSEUDO_F);
        }
        Mont::<N>::typed_sqr(&Self::K, a)
    }

    /// `a + b` in the elements' form: the asm carry chains on aarch64, which an alternated A/B
    /// found as fast as plain Rust for sums and faster for differences, negations and doublings
    /// (1.5 to 3 times) in dependent chains.
    #[inline(always)]
    fn add_l(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if C::<P, N>::PSEUDO {
            return Mont::<N>::typed_pseudo_add(a, b, C::<P, N>::PSEUDO_F);
        }
        Mont::<N>::typed_add(&Self::K, a, b)
    }

    /// `a - b` in the elements' form.
    #[inline(always)]
    fn sub_l(a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
        if C::<P, N>::PSEUDO {
            return Mont::<N>::typed_pseudo_sub(a, b, C::<P, N>::PSEUDO_F);
        }
        Mont::<N>::typed_sub(&Self::K, a, b)
    }

    /// `-a` in the elements' form.
    #[inline(always)]
    fn neg_l(a: &[u64; N]) -> [u64; N] {
        if C::<P, N>::PSEUDO {
            return Self::sub_l(&[0; N], a);
        }
        Mont::<N>::typed_neg(&Self::K, a)
    }

    /// The prime, the low limb first.
    #[must_use]
    pub const fn modulus_limbs() -> [u64; N] {
        P::MODULUS
    }

    /// The prime as a number.
    #[must_use]
    pub fn modulus() -> Number {
        number_of(&P::MODULUS, ByteOrder::LittleEndian)
    }

    /// The bits of the prime.
    #[must_use]
    pub const fn bits() -> u32 {
        C::<P, N>::BITS
    }

    /// The bytes of an encoding, `⌈bits / 8⌉`.
    #[must_use]
    pub const fn byte_len() -> usize {
        C::<P, N>::BYTES
    }

    /// Whether the modulus is prime, by [`Number::is_probable_prime`]: the check a test of a
    /// declared [`TypedPrime`] should make, as the compile-time checks cannot.
    #[must_use]
    pub fn modulus_is_prime() -> bool {
        Self::modulus().is_probable_prime()
    }

    /// The element of a word.
    #[must_use]
    #[inline]
    pub fn from_u64(x: u64) -> Self {
        let mut l = [0u64; N];
        // A prime of two limbs or more is above every word.
        l[0] = if N == 1 { x % P::MODULUS[0] } else { x };
        Self::of_value(&l)
    }

    /// The element congruent to a number of any sign and length.
    #[must_use]
    pub fn from_number(x: &Number) -> Self {
        let v = x.rem_euclid(&Self::modulus());
        let l = limbs_of(&v);
        let mut a = [0u64; N];
        a[..l.len()].copy_from_slice(&l);
        Self::of_value(&a)
    }

    /// The element of the big-endian encoding of exactly [`byte_len`](TypedFp::byte_len) bytes,
    /// `None` for another length or a value not below `p`.
    #[must_use]
    pub fn from_bytes_be(b: &[u8]) -> Option<Self> {
        Self::decode(b, true)
    }

    /// The element of the little-endian encoding, as [`from_bytes_be`](TypedFp::from_bytes_be).
    #[must_use]
    pub fn from_bytes_le(b: &[u8]) -> Option<Self> {
        Self::decode(b, false)
    }

    fn decode(b: &[u8], be: bool) -> Option<Self> {
        let n = C::<P, N>::BYTES;
        if b.len() != n {
            return None;
        }
        let mut l = [0u64; N];
        for (j, limb) in l.iter_mut().enumerate() {
            let (lo, hi) = (8 * j, (8 * j + 8).min(n));
            if lo >= hi {
                break;
            }
            let mut w = [0u8; 8];
            *limb = if be {
                w[8 - (hi - lo)..].copy_from_slice(&b[n - hi..n - lo]);
                u64::from_be_bytes(w)
            } else {
                w[..hi - lo].copy_from_slice(&b[lo..hi]);
                u64::from_le_bytes(w)
            };
        }
        (compare(&l, &P::MODULUS) == Ordering::Less).then(|| Self::of_value(&l))
    }

    /// The big-endian encoding of the value in exactly [`byte_len`](TypedFp::byte_len) bytes.
    #[must_use]
    pub fn to_bytes_be(&self) -> Vec<u8> {
        let l = self.value_limbs();
        let mut b = vec![0u8; C::<P, N>::BYTES];
        // Eight bytes a limb from the least significant end; the top limb's low bytes in front.
        let (top, chunks) = b.as_rchunks_mut::<8>();
        for (dst, w) in chunks.iter_mut().rev().zip(&l) {
            *dst = w.to_be_bytes();
        }
        if !top.is_empty() {
            let n = top.len();
            top.copy_from_slice(&l[N - 1].to_be_bytes()[8 - n..]);
        }
        b
    }

    /// The little-endian encoding of the value.
    #[must_use]
    pub fn to_bytes_le(&self) -> Vec<u8> {
        let l = self.value_limbs();
        let mut b = vec![0u8; C::<P, N>::BYTES];
        let (chunks, top) = b.as_chunks_mut::<8>();
        for (dst, w) in chunks.iter_mut().zip(&l) {
            *dst = w.to_le_bytes();
        }
        if !top.is_empty() {
            let n = top.len();
            top.copy_from_slice(&l[N - 1].to_le_bytes()[..n]);
        }
        b
    }

    /// The value in `[0, p)`.
    #[must_use]
    pub fn value(&self) -> Number {
        number_of(&self.value_limbs(), ByteOrder::LittleEndian)
    }

    /// The element of `f`, the runtime field of the same prime.
    ///
    /// # Panics
    ///
    /// Panics if `f` has another prime.
    #[must_use]
    #[track_caller]
    pub fn to_fp<'f>(&self, f: &'f PrimeField) -> Fp<'f> {
        assert!(
            f.limbs == P::MODULUS,
            "the field has another prime than {}",
            P::NAME
        );
        // The runtime field's Montgomery forms of four and six limbs hold the same residues.
        if !C::<P, N>::PLAIN
            && let Some(x) = Fp::from_montgomery(f, &self.l)
        {
            return x;
        }
        f.from_bytes_le(&self.to_bytes_le()).expect("below p")
    }

    /// The element of the value of `x`, an element of the runtime field of the same prime.
    ///
    /// # Panics
    ///
    /// Panics if `x` is of another prime.
    #[must_use]
    #[track_caller]
    pub fn from_fp(x: &Fp<'_>) -> Self {
        assert!(
            x.field().limbs == P::MODULUS,
            "the element is of another prime than {}",
            P::NAME
        );
        if !C::<P, N>::PLAIN
            && let Some(l) = x.montgomery()
        {
            let mut a = [0u64; N];
            a.copy_from_slice(l);
            return Self::of(a);
        }
        Self::from_bytes_le(&x.to_bytes_le()).expect("below p")
    }

    /// Whether the element is 0.
    #[must_use]
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.unique().iter().fold(0, |x, l| x | l) == 0
    }

    /// Whether the element is 1.
    #[must_use]
    #[inline]
    pub fn is_one(&self) -> bool {
        self.unique() == C::<P, N>::ONE_FORM
    }

    /// The square.
    #[must_use]
    #[inline]
    pub fn square(&self) -> Self {
        Self::of(Self::sqr_l(&self.l))
    }

    /// Twice the element.
    #[must_use]
    #[inline]
    pub fn double(&self) -> Self {
        Self::of(Self::add_l(&self.l, &self.l))
    }

    /// The inverse, `None` for 0: for a prime of one limb the binary extended Euclid on the
    /// value's word; for longer primes [`Number::mod_inverse`] of the value (Lehmer's extended
    /// Euclid), as [`Fp::inverse`] takes it.
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        if N == 1 {
            // A prime of one word: the binary extended Euclid on words.
            let mut a = [0u64; N];
            a[0] = inverse_word(self.value_limbs()[0], P::MODULUS[0]);
            return Some(Self::of_value(&a));
        }
        let v = self.value().mod_inverse(&Self::modulus())?;
        let l = limbs_of(&v);
        let mut a = [0u64; N];
        a[..l.len()].copy_from_slice(&l);
        Some(Self::of_value(&a))
    }

    /// The `e`-th power; a negative `e` is a power of the inverse, and `0⁰ = 1`; an exponent more
    /// than 64 bits longer than `p` is first reduced modulo `p − 1` (for an element not 0).
    ///
    /// # Panics
    ///
    /// Panics if `e` is negative and the element is 0.
    #[must_use]
    #[track_caller]
    pub fn pow(&self, e: &Number) -> Self {
        if e.is_negative() {
            let inv = self.inverse().expect("0 has no inverse");
            return inv.pow(&-e);
        }
        if e.bits() > u64::from(Self::bits()) + 64 && !self.is_zero() {
            let pm1 = &Self::modulus() - &Number::from(1u8);
            return self.pow_limbs(&limbs_of(&e.rem_euclid(&pm1)));
        }
        self.pow_limbs(&limbs_of(e))
    }

    /// The `e`-th power.
    #[must_use]
    pub fn pow_u64(&self, e: u64) -> Self {
        self.pow_limbs(&[e])
    }

    /// `x^e` for the exponent's limbs, the low one first: windows of four bits from the top.
    fn pow_limbs(&self, e: &[u64]) -> Self {
        let bits = e
            .iter()
            .rposition(|&w| w != 0)
            .map_or(0, |i| 64 * i + 64 - e[i].leading_zeros() as usize);
        if bits == 0 {
            return Self::ONE;
        }
        let mut table = [Self::ONE; 16];
        table[1] = *self;
        for i in 2..16 {
            table[i] = table[i - 1] * *self;
        }
        let nibble = |i: usize| ((e[i / 16] >> (4 * (i % 16))) & 15) as usize;
        let top = bits.div_ceil(4);
        let mut r = table[nibble(top - 1)];
        for i in (0..top - 1).rev() {
            r = r.square().square().square().square();
            let d = nibble(i);
            if d != 0 {
                r *= table[d];
            }
        }
        r
    }

    /// `x^e` by the plan of a fixed power ([`Chain`], the runtime field's).
    fn chain_power(&self, c: &Chain) -> Self {
        let x2 = self.square();
        let count = usize::from(c.top).div_ceil(2);
        let mut odd = Vec::with_capacity(count);
        odd.push(*self);
        for i in 1..count {
            let t = odd[i - 1] * x2;
            odd.push(t);
        }
        let mut r = odd[usize::from(c.first) / 2];
        for &(s, v) in c.steps {
            for _ in 0..s {
                r = r.square();
            }
            if v != 0 {
                r *= odd[usize::from(v) / 2];
            }
        }
        r
    }

    /// The Legendre symbol: 1, −1, or 0 for 0: for a prime of one limb the Jacobi symbol of
    /// words (reciprocity and remainders of words); for longer primes [`Number::jacobi`] of the
    /// value, as [`Fp::legendre`] takes it.
    #[must_use]
    pub fn legendre(&self) -> i32 {
        if self.is_zero() {
            return 0;
        }
        if N == 1 {
            return jacobi_words(self.value_limbs()[0], P::MODULUS[0]);
        }
        self.value().jacobi(&Self::modulus())
    }

    /// Whether the element is a square (0 is).
    #[must_use]
    pub fn is_square(&self) -> bool {
        self.legendre() >= 0
    }

    /// The parity of the value, RFC 9380's `sgn0`.
    #[must_use]
    pub fn sgn0(&self) -> bool {
        self.value_limbs()[0] & 1 == 1
    }

    /// The smaller of the two square roots in `[0, p)`, `None` for a non-square: `a^((p + 1)/4)`
    /// for `p ≡ 3 (mod 4)`, Atkin's method for `p ≡ 5 (mod 8)`, Tonelli and Shanks's with the
    /// least non-residue's power `z^q` computed at compile time for every `p ≡ 1 (mod 8)`; each
    /// checked or self-checking. [`Fp::sqrt`] takes Cipolla's method instead where `s² > 12·b`
    /// (`p − 1 = q·2^s`, `b` bits; Goldilocks's prime among them): the root is the same, the
    /// smaller.
    #[must_use]
    pub fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(*self);
        }
        let e = &C::<P, N>::E;
        let r = match C::<P, N>::S {
            1 => {
                let r = if C::<P, N>::BLS12_381 {
                    self.chain_power(&BLS12_381_SQRT)
                } else {
                    self.pow_limbs(e)
                };
                (r.square() == *self).then_some(r)?
            }
            2 => {
                let a2 = self.double();
                let b = a2.pow_limbs(e);
                let i = a2 * b.square();
                let r = *self * b * (i - Self::ONE);
                (r.square() == *self).then_some(r)?
            }
            s => {
                // r = a^((q + 1)/2), b = a^q; each round halves the order of b.
                let w = self.pow_limbs(e);
                let mut r = *self * w;
                let mut b = r * w;
                let mut c = Self::of(C::<P, N>::TONELLI_C);
                let mut m = s;
                while !b.is_one() {
                    let mut i = 0;
                    let mut b2 = b;
                    while !b2.is_one() {
                        b2 = b2.square();
                        i += 1;
                        if i == m {
                            return None;
                        }
                    }
                    let mut g = c;
                    for _ in 0..m - i - 1 {
                        g = g.square();
                    }
                    r *= g;
                    c = g.square();
                    b *= c;
                    m = i;
                }
                r
            }
        };
        Some(
            if compare(&r.value_limbs(), &C::<P, N>::HALF) == Ordering::Greater {
                -r
            } else {
                r
            },
        )
    }

    /// `self / b`.
    #[track_caller]
    fn quotient(&self, b: &Self) -> Self {
        let inv = b.inverse().expect("division by zero");
        Self::of(Self::mul_l(&self.l, &inv.l))
    }

    /// The inverses of `xs`, zeros left as zeros: Montgomery's trick, one inversion and
    /// `3(n − 1)` products.
    #[must_use]
    pub fn batch_invert(xs: &[Self]) -> Vec<Self> {
        let mut acc = Vec::with_capacity(xs.len());
        let mut p = Self::ONE;
        for x in xs {
            acc.push(p);
            if !x.is_zero() {
                p *= *x;
            }
        }
        let mut inv = p.inverse().unwrap_or(Self::ONE);
        let mut r = vec![Self::ZERO; xs.len()];
        for i in (0..xs.len()).rev() {
            if xs[i].is_zero() {
                continue;
            }
            r[i] = inv * acc[i];
            inv *= xs[i];
        }
        r
    }
}

/// `x mod p` for Goldilocks's prime `p = 2^64 − 2^32 + 1` and `x < 2^128`: with `x = l + h 2^64`,
/// `h = h₁ 2^32 + h₀`, `x ≡ l − h₁ + h₀ (2^32 − 1)`, as `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1`.
#[inline(always)]
fn goldilocks_reduce(x: u128) -> u64 {
    const P: u64 = 0xffff_ffff_0000_0001;
    const EPS: u64 = 0xffff_ffff;
    let (lo, hi) = (x as u64, (x >> 64) as u64);
    let (h1, h0) = (hi >> 32, hi & EPS);
    // l − h₁: a borrow is 2^64 too few, which is 2^32 − 1 modulo p.
    let (mut t, borrow) = lo.overflowing_sub(h1);
    if borrow {
        t = t.wrapping_sub(EPS);
    }
    // + h₀ (2^32 − 1): a carry is 2^64 too many, 2^32 − 1 modulo p.
    let (mut r, carry) = t.overflowing_add(h0 * EPS);
    if carry {
        r = r.wrapping_add(EPS);
    }
    if r >= P { r - P } else { r }
}

/// `1/a mod p` for a word `a` in `1..p` and an odd prime `p` of one word: the binary extended
/// Euclid, `x₁ a ≡ u` and `x₂ a ≡ v (mod p)` kept as `u`, `v` lose their factors of two and the
/// larger takes the smaller off.
fn inverse_word(a: u64, p: u64) -> u64 {
    let half = |x: u64| -> u64 {
        if x & 1 == 0 {
            x >> 1
        } else {
            // (x + p)/2 for odd x and p, without the carry out of x + p.
            (x >> 1) + (p >> 1) + 1
        }
    };
    let (mut u, mut v, mut x1, mut x2) = (a, p, 1u64, 0u64);
    while u != 1 && v != 1 {
        while u & 1 == 0 {
            u >>= 1;
            x1 = half(x1);
        }
        while v & 1 == 0 {
            v >>= 1;
            x2 = half(x2);
        }
        if u >= v {
            u -= v;
            x1 = if x1 >= x2 { x1 - x2 } else { x1 + (p - x2) };
        } else {
            v -= u;
            x2 = if x2 >= x1 { x2 - x1 } else { x2 + (p - x1) };
        }
    }
    if u == 1 { x1 } else { x2 }
}

/// `a` against `b`, of the same length, from the top limb.
fn compare(a: &[u64], b: &[u64]) -> Ordering {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        match x.cmp(y) {
            Ordering::Equal => {}
            o => return o,
        }
    }
    Ordering::Equal
}

impl<P: TypedPrime<N>, const N: usize> Default for TypedFp<P, N> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<P: TypedPrime<N>, const N: usize> From<u64> for TypedFp<P, N> {
    fn from(x: u64) -> Self {
        Self::from_u64(x)
    }
}

impl<P: TypedPrime<N>, const N: usize> fmt::Debug for TypedFp<P, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value(), P::NAME)
    }
}

impl<P: TypedPrime<N>, const N: usize> fmt::Display for TypedFp<P, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value())
    }
}

/// A binary operator on owned and borrowed operands, and its assignment.
macro_rules! typed_binary {
    ($tr:ident, $m:ident, $atr:ident, $am:ident, |$a:ident, $b:ident| $body:expr) => {
        impl<P: TypedPrime<N>, const N: usize> $tr for TypedFp<P, N> {
            type Output = Self;
            #[inline(always)]
            #[track_caller]
            fn $m(self, rhs: Self) -> Self {
                let ($a, $b) = (&self, &rhs);
                $body
            }
        }
        impl<P: TypedPrime<N>, const N: usize> $tr<&TypedFp<P, N>> for TypedFp<P, N> {
            type Output = Self;
            #[inline(always)]
            #[track_caller]
            fn $m(self, rhs: &Self) -> Self {
                let ($a, $b) = (&self, rhs);
                $body
            }
        }
        impl<P: TypedPrime<N>, const N: usize> $tr<TypedFp<P, N>> for &TypedFp<P, N> {
            type Output = TypedFp<P, N>;
            #[inline(always)]
            #[track_caller]
            fn $m(self, rhs: TypedFp<P, N>) -> TypedFp<P, N> {
                let ($a, $b) = (self, &rhs);
                $body
            }
        }
        impl<P: TypedPrime<N>, const N: usize> $tr<&TypedFp<P, N>> for &TypedFp<P, N> {
            type Output = TypedFp<P, N>;
            #[inline(always)]
            #[track_caller]
            fn $m(self, rhs: &TypedFp<P, N>) -> TypedFp<P, N> {
                let ($a, $b) = (self, rhs);
                $body
            }
        }
        impl<P: TypedPrime<N>, const N: usize> $atr for TypedFp<P, N> {
            #[inline(always)]
            #[track_caller]
            fn $am(&mut self, rhs: Self) {
                let r = {
                    let ($a, $b) = (&*self, &rhs);
                    $body
                };
                *self = r;
            }
        }
        impl<P: TypedPrime<N>, const N: usize> $atr<&TypedFp<P, N>> for TypedFp<P, N> {
            #[inline(always)]
            #[track_caller]
            fn $am(&mut self, rhs: &Self) {
                let r = {
                    let ($a, $b) = (&*self, rhs);
                    $body
                };
                *self = r;
            }
        }
    };
}

typed_binary!(Add, add, AddAssign, add_assign, |a, b| TypedFp::of(
    TypedFp::<P, N>::add_l(&a.l, &b.l)
));
typed_binary!(Sub, sub, SubAssign, sub_assign, |a, b| TypedFp::of(
    TypedFp::<P, N>::sub_l(&a.l, &b.l)
));
typed_binary!(Mul, mul, MulAssign, mul_assign, |a, b| TypedFp::of(
    TypedFp::<P, N>::mul_l(&a.l, &b.l)
));
typed_binary!(Div, div, DivAssign, div_assign, |a, b| a.quotient(b));

impl<P: TypedPrime<N>, const N: usize> Neg for TypedFp<P, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self::of(Self::neg_l(&self.l))
    }
}

impl<P: TypedPrime<N>, const N: usize> Neg for &TypedFp<P, N> {
    type Output = TypedFp<P, N>;
    #[inline(always)]
    fn neg(self) -> TypedFp<P, N> {
        -*self
    }
}

/// Declares a [`TypedPrime`]: a unit struct (with the derives `Clone`, `Copy`, `Debug`,
/// `Default`, `PartialEq`, `Eq`, `Hash`) and its implementation, from the number of limbs and the
/// prime in hexadecimal ([`limbs_from_hex`](crate::math::field::limbs_from_hex), checked at
/// compile time: odd, of exactly that many limbs). A test should check that it is prime, as the
/// example's does.
///
/// # Examples
///
/// ```
/// use a::math::field::TypedFp;
///
/// a::typed_prime! {
///     /// 2^127 - 1.
///     pub M127, 2, "7fffffffffffffffffffffffffffffff"
/// }
///
/// type F = TypedFp<M127, 2>;
/// assert!(F::modulus_is_prime()); // the check a test of the declaration makes
/// let x = F::from_u64(3);
/// assert_eq!(x * x.inverse().unwrap(), F::ONE);
/// assert_eq!(F::bits(), 127);
/// ```
#[macro_export]
macro_rules! typed_prime {
    ($(#[$meta:meta])* $vis:vis $name:ident, $n:literal, $hex:literal) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        $vis struct $name;

        impl $crate::math::field::TypedPrime<$n> for $name {
            const MODULUS: [u64; $n] = $crate::math::field::limbs_from_hex($hex);
            const NAME: &'static str = stringify!($name);
        }
    };
}

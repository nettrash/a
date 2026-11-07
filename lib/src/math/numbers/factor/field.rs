//! The arithmetic of the methods: residues modulo the odd number `n` being factored, in
//! Montgomery's form, `x` held as `x R mod n`, below `n`, or below `2n` when lazily reduced.
//!
//! - One limb, two, and three to eight, in registers: the powers' [`Mont1`], [`Mont2`] and
//!   [`MontN`], each product reduced as it is formed ([`Reg`]); when the top limb is below `2^62`,
//!   so that `4n < R`, [`Lazy2`] and [`LazyN`], whose residues stay below `2n`, without the
//!   subtraction after each product.
//! - Nine limbs and more, in memory: the powers' [`Montgomery`], the multiplication's kernels then
//!   REDC by the kernel's passes, or by products from 100 limbs ([`Big`]).
//!
//! The operations write their result to a residue of their own, `r = a b`, or in place, `r = r b`,
//! so that the methods are written once for both: in registers the residues are copied values,
//! which the compiler keeps in registers; in memory they are vectors, which no operation allocates.
//!
//! A residue `x R` has the same common divisors with `n` as `x`, as `R` is a power of two and `n`
//! odd: the greatest common divisors are taken of the residues as they are.

use super::super::div::limbs::binvert_limb;
use super::super::mul::basecase;
use super::super::mul::limbs::{add_n, cmp_n, sub_n};
use super::super::pow::montgomery::{
    Mont1, Mont2, MontN, Montgomery, redc_inverse, redc_limbs, rem, rem_itch,
};
use super::super::pow::window::{Registers, Ring};
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Number};
use core::cell::UnsafeCell;
use core::cmp::Ordering;
use core::hint::select_unpredictable;

/// The arithmetic of residues modulo an odd `n > 1`, in Montgomery's form, below `n`, or below
/// `2n` when lazily reduced (then 0 is held as 0 or `n`).
pub(super) trait Arith {
    /// A residue.
    type V: Clone;

    /// Whether the residues are in memory rather than in registers.
    const MEMORY: bool = false;

    /// The limbs of `n`, the least significant first, the top one not zero.
    fn n(&self) -> &[u64];

    /// The residue of 0.
    fn zero(&self) -> Self::V;

    /// The residue of 1, `R mod n`.
    fn one(&self) -> Self::V;

    /// The residue of the plain `x`, below `n`, given by its limbs, at most those of `n`.
    fn residue(&self, x: &[u64]) -> Self::V;

    /// The residue of the plain `c`, reduced modulo `n` first.
    fn small(&self, c: u64) -> Self::V;

    /// The limbs of the residue as it is held, `a R mod n` for the number `a`, as many as `n`'s.
    fn raw(&self, a: &Self::V) -> Vec<u64>;

    /// The low limb of the residue as it is held.
    fn low(&self, a: &Self::V) -> u64;

    /// The residue held as the limbs `x`, below `n`: [`Arith::raw`]'s inverse.
    fn of_raw(&self, x: &[u64]) -> Self::V;

    /// `R^3 mod n` as a held residue: its product by the held `x` is `x R^2`.
    fn cube(&self) -> &Self::V;

    /// `r = a b`.
    fn mul(&self, r: &mut Self::V, a: &Self::V, b: &Self::V);

    /// `r = r b`.
    fn mul_in(&self, r: &mut Self::V, b: &Self::V);

    /// `r = a^2`.
    fn sqr(&self, r: &mut Self::V, a: &Self::V);

    /// `r = r^2`.
    fn sqr_in(&self, r: &mut Self::V);

    /// `r = a + b`.
    fn add(&self, r: &mut Self::V, a: &Self::V, b: &Self::V);

    /// `r = r + b`.
    fn add_in(&self, r: &mut Self::V, b: &Self::V);

    /// `r = a - b`.
    fn sub(&self, r: &mut Self::V, a: &Self::V, b: &Self::V);

    /// `r = r - b`.
    fn sub_in(&self, r: &mut Self::V, b: &Self::V);

    /// Whether `a` is the residue of 0.
    fn is_zero(&self, a: &Self::V) -> bool;

    /// Swaps `a` and `b` when `c` is set.
    fn cswap(&self, c: bool, a: &mut Self::V, b: &mut Self::V);
}

/// A computation on residues modulo `n`, run by [`with_field`] in the arithmetic for `n`'s length.
pub(super) trait Job {
    /// What it gives.
    type Out;

    /// Runs it in the arithmetic `f`.
    fn run<A: Arith>(self, f: &A) -> Self::Out;
}

/// Runs `job` in the arithmetic modulo the odd `n > 1` of the limbs `n`, the top one not zero:
/// in registers up to eight limbs, lazily reduced when the top limb is below `2^62`, in memory
/// beyond.
pub(super) fn with_field<J: Job>(n: &[u64], job: J) -> J::Out {
    debug_assert!(n[0] & 1 == 1 && *n.last().unwrap() != 0 && (n.len() > 1 || n[0] > 1));
    let lazy = *n.last().unwrap() < 1 << 62;
    match (n.len(), lazy) {
        (1, _) => job.run(&Reg::new(Mont1::new(n[0]))),
        (2, true) => job.run(&Reg::new(Lazy2::new([n[0], n[1]]))),
        (2, false) => job.run(&Reg::new(Mont2::new([n[0], n[1]]))),
        (3, true) => job.run(&Reg::new(LazyN::<3>::new(array(n)))),
        (3, false) => job.run(&Reg::new(MontN::<3>::new(array(n)))),
        (4, true) => job.run(&Reg::new(LazyN::<4>::new(array(n)))),
        (4, false) => job.run(&Reg::new(MontN::<4>::new(array(n)))),
        (5, true) => job.run(&Reg::new(LazyN::<5>::new(array(n)))),
        (5, false) => job.run(&Reg::new(MontN::<5>::new(array(n)))),
        (6, true) => job.run(&Reg::new(LazyN::<6>::new(array(n)))),
        (6, false) => job.run(&Reg::new(MontN::<6>::new(array(n)))),
        (7, true) => job.run(&Reg::new(LazyN::<7>::new(array(n)))),
        (7, false) => job.run(&Reg::new(MontN::<7>::new(array(n)))),
        (8, true) => job.run(&Reg::new(LazyN::<8>::new(array(n)))),
        (8, false) => job.run(&Reg::new(MontN::<8>::new(array(n)))),
        _ => job.run(&Big::new(n)),
    }
}

/// The limbs `x` as an array.
fn array<const N: usize>(x: &[u64]) -> [u64; N] {
    let mut a = [0; N];
    a.copy_from_slice(x);
    a
}

// Conversions

/// The limbs of the magnitude of `x`, the least significant first, without zero limbs at the top:
/// none for zero.
pub(super) fn limbs_of(x: &Number) -> Vec<u64> {
    let v = &x.value;
    let mut out = Vec::with_capacity(v.len().div_ceil(8));
    match x.byte_order {
        ByteOrder::LittleEndian => {
            for c in v.chunks(8) {
                let mut b = [0u8; 8];
                b[..c.len()].copy_from_slice(c);
                out.push(u64::from_le_bytes(b));
            }
        }
        ByteOrder::BigEndian => {
            for c in v.rchunks(8) {
                let mut b = [0u8; 8];
                b[8 - c.len()..].copy_from_slice(c);
                out.push(u64::from_be_bytes(b));
            }
        }
    }
    trim(&mut out);
    out
}

/// Drops the zero limbs at the top of `x`.
#[inline]
pub(super) fn trim(x: &mut Vec<u64>) {
    while x.last() == Some(&0) {
        x.pop();
    }
}

/// The non-negative number of the limbs `x`, the least significant first, in the byte order
/// `order`, in the shortest form.
pub(super) fn number_of(x: &[u64], order: ByteOrder) -> Number {
    let bytes: Vec<u8> = match order {
        ByteOrder::LittleEndian => x.iter().flat_map(|l| l.to_le_bytes()).collect(),
        ByteOrder::BigEndian => x.iter().rev().flat_map(|l| l.to_be_bytes()).collect(),
    };
    let mut n = Number::new(bytes, order);
    if n.value.is_empty() {
        n.value.push(0);
    }
    n
}

/// The greatest common divisor of the limbs `a` and `b`, without zero limbs at the top.
pub(super) fn gcd_limbs(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len() <= 1 && b.len() <= 1 {
        let g = gcd_u64(
            a.first().copied().unwrap_or(0),
            b.first().copied().unwrap_or(0),
        );
        let mut v = vec![g];
        trim(&mut v);
        return v;
    }
    let order = ByteOrder::LittleEndian;
    limbs_of(&number_of(a, order).gcd(&number_of(b, order)))
}

/// The greatest common divisor of `a` and `b`, by the binary method.
#[inline]
pub(super) fn gcd_u64(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let k = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            core::mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << k;
        }
    }
}

/// The greatest common divisor of the residue `a` with `n`, without zero limbs at the top.
pub(super) fn gcd_n<A: Arith>(f: &A, a: &A::V) -> Vec<u64> {
    let mut x = f.raw(a);
    trim(&mut x);
    gcd_limbs(&x, f.n())
}

/// The inverse of the residue `a`, or the greatest common divisor of `a` and `n` when there is
/// none: `t = 1 / (a R)` by the greatest common divisor's cofactors, then `t R^3 / R = R / a`.
pub(super) fn invert<A: Arith>(f: &A, a: &A::V) -> Result<A::V, Vec<u64>> {
    let n = f.n();
    // One limb and two: the extended Euclidean algorithm in registers.
    let t = match n.len() {
        1 => match inverse_u64(f.low(a), n[0]) {
            Ok(t) => vec![t],
            Err(g) => return Err(vec![g]),
        },
        2 => {
            let x = f.raw(a);
            let x = u128::from(x[0]) | (u128::from(x[1]) << 64);
            let m = u128::from(n[0]) | (u128::from(n[1]) << 64);
            match inverse_u128(x, m) {
                Ok(t) => vec![t as u64, (t >> 64) as u64],
                Err(g) => {
                    let mut g = vec![g as u64, (g >> 64) as u64];
                    trim(&mut g);
                    return Err(g);
                }
            }
        }
        _ => {
            let order = ByteOrder::LittleEndian;
            let mut x = f.raw(a);
            trim(&mut x);
            match number_of(&x, order).mod_inverse(&number_of(n, order)) {
                Some(t) => {
                    let mut t = limbs_of(&t);
                    t.resize(n.len(), 0);
                    t
                }
                None => return Err(gcd_n(f, a)),
            }
        }
    };
    let t = f.of_raw(&t);
    let mut r = f.zero();
    f.mul(&mut r, &t, f.cube());
    Ok(r)
}

/// The inverse of `a` modulo `m > 1`, by the extended Euclidean algorithm, or the greatest common
/// divisor of `a` and `m` when it is not 1 (`m` for `a = 0`).
pub(super) fn inverse_u64(a: u64, m: u64) -> Result<u64, u64> {
    let (mut r0, mut r1) = (m, a % m);
    // `t_i a = r_i mod m`; `|t_i| <= m`.
    let (mut t0, mut t1) = (0i128, 1i128);
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 - i128::from(q) * t1);
    }
    if r0 != 1 {
        return Err(r0);
    }
    Ok(if t0 < 0 {
        (t0 + i128::from(m)) as u64
    } else {
        t0 as u64
    })
}

/// The inverse of `a` modulo the odd `m > 1`, by the binary extended algorithm (shifts and
/// subtractions, no division), or the greatest common divisor of `a` and `m` when it is not 1.
pub(super) fn inverse_u128(a: u128, m: u128) -> Result<u128, u128> {
    let a = a % m;
    if a == 0 {
        return Err(m);
    }
    // `x1 a = u` and `x2 a = v` modulo `m`, all along.
    let (mut u, mut v) = (a, m);
    let (mut x1, mut x2) = (1u128, 0u128);
    let half = |x: u128| {
        if x & 1 == 0 {
            x >> 1
        } else {
            // `(x + m) / 2` without the carry of `x + m`.
            (x >> 1) + (m >> 1) + 1
        }
    };
    let sub = |x: u128, y: u128| if x >= y { x - y } else { x + (m - y) };
    while u != 1 && v != 1 {
        if u == 0 {
            return Err(v);
        }
        if v == 0 {
            return Err(u);
        }
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
            x1 = sub(x1, x2);
        } else {
            v -= u;
            x2 = sub(x2, x1);
        }
    }
    Ok(if u == 1 { x1 } else { x2 })
}

/// `r = a b`, out of line: for the paths that run rarely, so that each does not hold its own copy
/// of the product, which in registers is long.
#[inline(never)]
pub(super) fn mul_cold<A: Arith>(f: &A, r: &mut A::V, a: &A::V, b: &A::V) {
    f.mul(r, a, b);
}

/// `r = r b`, out of line, as [`mul_cold`].
#[inline(never)]
pub(super) fn mul_in_cold<A: Arith>(f: &A, r: &mut A::V, b: &A::V) {
    f.mul_in(r, b);
}

/// Sets each residue of `v` to its inverse, by Montgomery's trick: the products of the first `i`,
/// one inversion of the product of all, and three products a residue. When one of them has no
/// inverse, gives the greatest common divisor of their product with `n` instead, and leaves `v`
/// as it was.
pub(super) fn invert_all<A: Arith>(f: &A, v: &mut [A::V]) -> Result<(), Vec<u64>> {
    if v.is_empty() {
        return Ok(());
    }
    // `p[i]` is the product of `v[0..=i]`.
    let mut p: Vec<A::V> = Vec::with_capacity(v.len());
    p.push(v[0].clone());
    for i in 1..v.len() {
        let mut t = f.zero();
        mul_cold(f, &mut t, &p[i - 1], &v[i]);
        p.push(t);
    }
    let mut inv = invert(f, &p[v.len() - 1])?;
    let mut t = f.zero();
    for i in (1..v.len()).rev() {
        // `inv` is `1 / (v[0] ... v[i])`: `1 / v[i] = inv p[i - 1]`, and `inv v[i]` the next.
        mul_cold(f, &mut t, &inv, &p[i - 1]);
        mul_in_cold(f, &mut inv, &v[i]);
        core::mem::swap(&mut v[i], &mut t);
    }
    v[0] = inv;
    Ok(())
}

// In registers

/// The operations [`Reg`] needs of the powers' arithmetic in registers besides products and
/// squares.
pub(super) trait Lanes: Registers<V: Eq> {
    /// The limbs of the modulus.
    fn modulus(&self) -> &[u64];

    /// `a + b mod n`.
    fn add(&self, a: Self::V, b: Self::V) -> Self::V;

    /// `a - b mod n`.
    fn sub(&self, a: Self::V, b: Self::V) -> Self::V;

    /// The value of the limbs `x`, at most as many as the modulus's.
    fn value(x: &[u64]) -> Self::V;

    /// The limbs of `a`.
    fn limbs(a: Self::V) -> Vec<u64>;

    /// The low limb of `a`.
    fn low(a: Self::V) -> u64;

    /// `R^2 mod n`.
    fn r2(&self) -> Self::V;

    /// `a` when `c` is set, otherwise `b`, without a branch.
    fn select(c: bool, a: Self::V, b: Self::V) -> Self::V;

    /// Whether `a` is the residue of 0.
    #[inline(always)]
    fn is_zero(&self, a: Self::V) -> bool {
        a == Self::value(&[])
    }
}

impl Lanes for Mont1 {
    #[inline(always)]
    fn modulus(&self) -> &[u64] {
        core::slice::from_ref(&self.m)
    }

    #[inline(always)]
    fn add(&self, a: u64, b: u64) -> u64 {
        let (s, carry) = a.overflowing_add(b);
        let (d, borrow) = s.overflowing_sub(self.m);
        // `a + b - n` when it is not negative, counting the carry out of `a + b`.
        select_unpredictable(carry || !borrow, d, s)
    }

    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        let (d, borrow) = a.overflowing_sub(b);
        d.wrapping_add(if borrow { self.m } else { 0 })
    }

    #[inline(always)]
    fn value(x: &[u64]) -> u64 {
        x.first().copied().unwrap_or(0)
    }

    fn limbs(a: u64) -> Vec<u64> {
        vec![a]
    }

    #[inline(always)]
    fn low(a: u64) -> u64 {
        a
    }

    #[inline]
    fn r2(&self) -> u64 {
        // `B mod n`, then its square, divided.
        let r = self.m.wrapping_neg() % self.m;
        ((u128::from(r) * u128::from(r)) % u128::from(self.m)) as u64
    }

    #[inline(always)]
    fn select(c: bool, a: u64, b: u64) -> u64 {
        select_unpredictable(c, a, b)
    }
}

/// `a + b - m` or `a + b` for the `N`-limb residues `a` and `b`, below `m`: the sum reduced.
#[inline(always)]
fn add_mod<const N: usize>(a: [u64; N], b: [u64; N], m: &[u64; N]) -> [u64; N] {
    let mut s = [0; N];
    let mut carry = false;
    for j in 0..N {
        let (x, c1) = a[j].overflowing_add(b[j]);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        s[j] = x;
        carry = c1 | c2;
    }
    let mut d = [0; N];
    let mut borrow = false;
    for j in 0..N {
        let (x, b1) = s[j].overflowing_sub(m[j]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        d[j] = x;
        borrow = b1 | b2;
    }
    select_array(carry || !borrow, d, s)
}

/// `a - b mod m` for the `N`-limb residues `a` and `b`, below `m`.
#[inline(always)]
fn sub_mod<const N: usize>(a: [u64; N], b: [u64; N], m: &[u64; N]) -> [u64; N] {
    let mut d = [0; N];
    let mut borrow = false;
    for j in 0..N {
        let (x, b1) = a[j].overflowing_sub(b[j]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        d[j] = x;
        borrow = b1 | b2;
    }
    // `m` added when the difference is negative, as a mask, without a branch.
    let mask = u64::from(borrow).wrapping_neg();
    let mut carry = false;
    for j in 0..N {
        let (x, c1) = d[j].overflowing_add(m[j] & mask);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        d[j] = x;
        carry = c1 | c2;
    }
    d
}

/// `a` when `c` is set, otherwise `b`, limb by limb without a branch.
#[inline(always)]
fn select_array<const N: usize>(c: bool, a: [u64; N], b: [u64; N]) -> [u64; N] {
    let mut r = [0; N];
    for j in 0..N {
        r[j] = select_unpredictable(c, a[j], b[j]);
    }
    r
}

/// `R^2 mod m` for the `N`-limb `m`, `R = B^N`, by a division.
fn r2_of<const N: usize>(m: &[u64; N]) -> [u64; N] {
    let mut r = [0; N];
    // The dividend `B^(2N)`, of `2N + 1 <= 17` limbs, by a divisor of `N <= 8`.
    let mut buf = [0u64; 96];
    let itch = rem_itch(2 * N + 1, N);
    debug_assert!(itch <= buf.len());
    let s = Scratch::new(buf.as_mut_ptr(), itch);
    // SAFETY: the scratch holds the remainder's `rem_itch(2N + 1, N)` limbs, which it checks, and
    // `m` is `N` limbs with the top one not zero.
    unsafe { rem(r.as_mut_ptr(), &1, 1, 2 * N, m.as_ptr(), N, s) };
    r
}

impl Lanes for Mont2 {
    #[inline(always)]
    fn modulus(&self) -> &[u64] {
        &self.m
    }

    #[inline(always)]
    fn add(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        add_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        sub_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn value(x: &[u64]) -> [u64; 2] {
        let mut a = [0; 2];
        a[..x.len()].copy_from_slice(x);
        a
    }

    fn limbs(a: [u64; 2]) -> Vec<u64> {
        a.to_vec()
    }

    #[inline(always)]
    fn low(a: [u64; 2]) -> u64 {
        a[0]
    }

    fn r2(&self) -> [u64; 2] {
        r2_of(&self.m)
    }

    #[inline(always)]
    fn select(c: bool, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        select_array(c, a, b)
    }
}

impl<const N: usize> Lanes for MontN<N> {
    #[inline(always)]
    fn modulus(&self) -> &[u64] {
        &self.m
    }

    #[inline(always)]
    fn add(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        add_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        sub_mod(a, b, &self.m)
    }

    #[inline(always)]
    fn value(x: &[u64]) -> [u64; N] {
        let mut a = [0; N];
        a[..x.len()].copy_from_slice(x);
        a
    }

    fn limbs(a: [u64; N]) -> Vec<u64> {
        a.to_vec()
    }

    #[inline(always)]
    fn low(a: [u64; N]) -> u64 {
        a[0]
    }

    fn r2(&self) -> [u64; N] {
        r2_of(&self.m)
    }

    #[inline(always)]
    fn select(c: bool, a: [u64; N], b: [u64; N]) -> [u64; N] {
        select_array(c, a, b)
    }
}

/// Residues in registers, by the powers' arithmetic `M`.
pub(super) struct Reg<M: Lanes> {
    /// The arithmetic.
    m: M,
    /// `R mod n`.
    one: M::V,
    /// `R^2 mod n`.
    r2: M::V,
    /// `R^3 mod n`.
    r3: M::V,
}

impl<M: Lanes> Reg<M> {
    /// The residues of the arithmetic `m`.
    pub(super) fn new(m: M) -> Self {
        let r2 = m.r2();
        let r3 = m.sqr(r2);
        let mut unit = [0u64; 1];
        unit[0] = 1;
        let one = m.mul(r2, M::value(&unit));
        Reg { m, one, r2, r3 }
    }
}

impl<M: Lanes> Arith for Reg<M> {
    type V = M::V;

    #[inline(always)]
    fn n(&self) -> &[u64] {
        self.m.modulus()
    }

    #[inline(always)]
    fn zero(&self) -> M::V {
        M::value(&[])
    }

    #[inline(always)]
    fn one(&self) -> M::V {
        self.one
    }

    #[inline]
    fn residue(&self, x: &[u64]) -> M::V {
        self.m.mul(M::value(x), self.r2)
    }

    #[inline]
    fn small(&self, c: u64) -> M::V {
        let n = self.m.modulus();
        let c = if n.len() == 1 { c % n[0] } else { c };
        self.residue(&[c])
    }

    fn raw(&self, a: &M::V) -> Vec<u64> {
        M::limbs(*a)
    }

    #[inline(always)]
    fn low(&self, a: &M::V) -> u64 {
        M::low(*a)
    }

    #[inline(always)]
    fn of_raw(&self, x: &[u64]) -> M::V {
        M::value(x)
    }

    #[inline(always)]
    fn cube(&self) -> &M::V {
        &self.r3
    }

    #[inline(always)]
    fn mul(&self, r: &mut M::V, a: &M::V, b: &M::V) {
        *r = self.m.mul(*a, *b);
    }

    #[inline(always)]
    fn mul_in(&self, r: &mut M::V, b: &M::V) {
        *r = self.m.mul(*r, *b);
    }

    #[inline(always)]
    fn sqr(&self, r: &mut M::V, a: &M::V) {
        *r = self.m.sqr(*a);
    }

    #[inline(always)]
    fn sqr_in(&self, r: &mut M::V) {
        *r = self.m.sqr(*r);
    }

    #[inline(always)]
    fn add(&self, r: &mut M::V, a: &M::V, b: &M::V) {
        *r = Lanes::add(&self.m, *a, *b);
    }

    #[inline(always)]
    fn add_in(&self, r: &mut M::V, b: &M::V) {
        *r = Lanes::add(&self.m, *r, *b);
    }

    #[inline(always)]
    fn sub(&self, r: &mut M::V, a: &M::V, b: &M::V) {
        *r = Lanes::sub(&self.m, *a, *b);
    }

    #[inline(always)]
    fn sub_in(&self, r: &mut M::V, b: &M::V) {
        *r = Lanes::sub(&self.m, *r, *b);
    }

    #[inline(always)]
    fn is_zero(&self, a: &M::V) -> bool {
        Lanes::is_zero(&self.m, *a)
    }

    #[inline(always)]
    fn cswap(&self, c: bool, a: &mut M::V, b: &mut M::V) {
        let (x, y) = (*a, *b);
        *a = M::select(c, y, x);
        *b = M::select(c, x, y);
    }
}

// Two limbs, lazily reduced

/// Montgomery's arithmetic modulo an odd `m` of two limbs below `2^126`, `R = B^2`, with the
/// residues kept below `2m` rather than `m`: as `4m < R`, REDC of a product of two of them is below
/// `(4 m^2 + R m) / R < 2m` without a subtraction at the end, and a sum or a difference takes one
/// subtraction or addition of `2m`. REDC goes a limb at a time, each quotient limb `t_i (-1/m)
/// mod B` and its product by `m` added, the low limb of which is known to clear `t_i`: 8 products
/// against 12 for both quotient limbs at once.
#[derive(Clone, Copy)]
pub(super) struct Lazy2 {
    /// The modulus, the low limb first.
    m: [u64; 2],
    /// `2m`.
    m2: u128,
    /// `-1/m mod B`.
    minv: u64,
}

impl Lazy2 {
    /// The arithmetic modulo the odd `m < 2^126`.
    #[inline]
    pub(super) fn new(m: [u64; 2]) -> Self {
        debug_assert!(m[1] < 1 << 62 && m[0] & 1 == 1);
        Lazy2 {
            m,
            m2: (u128::from(m[0]) | (u128::from(m[1]) << 64)) << 1,
            minv: binvert_limb(m[0]).wrapping_neg(),
        }
    }

    /// `t / R mod m`, below `2m`, for `t < 4 m^2`, a limb at a time.
    #[inline(always)]
    fn redc(&self, t: [u64; 4]) -> [u64; 2] {
        let [m0, m1] = self.m;
        let [t0, t1, t2, t3] = t;
        // `t += q0 m`, `q0 = t0 (-1/m)`: the low limb becomes 0, with a carry unless `t0` was.
        let q0 = t0.wrapping_mul(self.minv);
        let p0 = u128::from(q0) * u128::from(m0);
        let p1 = u128::from(q0) * u128::from(m1);
        let s = u128::from(t1) + (p0 >> 64) + u128::from(p1 as u64) + u128::from(t0 != 0);
        let t1 = s as u64;
        let s = u128::from(t2) + (p1 >> 64) + (s >> 64);
        let t2 = s as u64;
        let t3 = t3 + (s >> 64) as u64;
        // `t += q1 m B`, the same for the next limb.
        let q1 = t1.wrapping_mul(self.minv);
        let p0 = u128::from(q1) * u128::from(m0);
        let p1 = u128::from(q1) * u128::from(m1);
        let s = u128::from(t2) + (p0 >> 64) + u128::from(p1 as u64) + u128::from(t1 != 0);
        let r0 = s as u64;
        let r1 = t3 + (p1 >> 64) as u64 + (s >> 64) as u64;
        [r0, r1]
    }

    /// The residue as a `u128`.
    #[inline(always)]
    fn wide(a: [u64; 2]) -> u128 {
        u128::from(a[0]) | (u128::from(a[1]) << 64)
    }

    /// The `u128` as limbs.
    #[inline(always)]
    fn limbs_of(x: u128) -> [u64; 2] {
        [x as u64, (x >> 64) as u64]
    }
}

impl Registers for Lazy2 {
    type V = [u64; 2];

    #[inline(always)]
    fn sqr(&self, a: [u64; 2]) -> [u64; 2] {
        let p00 = u128::from(a[0]) * u128::from(a[0]);
        let p01 = u128::from(a[0]) * u128::from(a[1]);
        let p11 = u128::from(a[1]) * u128::from(a[1]);
        // `a < 2^127`, so `2 p01 < 2^191` and the limbs carry no further than the top.
        let d = p01 << 1;
        let c1 = (p00 >> 64) + u128::from(d as u64);
        let c2 = (c1 >> 64) + (d >> 64) + u128::from(p11 as u64);
        let c3 = ((c2 >> 64) + (p11 >> 64)) as u64;
        self.redc([p00 as u64, c1 as u64, c2 as u64, c3])
    }

    #[inline(always)]
    fn mul(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let p00 = u128::from(a[0]) * u128::from(b[0]);
        let p01 = u128::from(a[0]) * u128::from(b[1]);
        let p10 = u128::from(a[1]) * u128::from(b[0]);
        let p11 = u128::from(a[1]) * u128::from(b[1]);
        let c1 = (p00 >> 64) + u128::from(p01 as u64) + u128::from(p10 as u64);
        let c2 = (c1 >> 64) + (p01 >> 64) + (p10 >> 64) + u128::from(p11 as u64);
        let c3 = ((c2 >> 64) + (p11 >> 64)) as u64;
        self.redc([p00 as u64, c1 as u64, c2 as u64, c3])
    }
}

impl Lanes for Lazy2 {
    #[inline(always)]
    fn modulus(&self) -> &[u64] {
        &self.m
    }

    #[inline(always)]
    fn add(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        // Below `4m < 2^128`.
        let s = Self::wide(a) + Self::wide(b);
        let (d, borrow) = s.overflowing_sub(self.m2);
        Self::limbs_of(select_unpredictable(borrow, s, d))
    }

    #[inline(always)]
    fn sub(&self, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        let (d, borrow) = Self::wide(a).overflowing_sub(Self::wide(b));
        Self::limbs_of(d.wrapping_add(if borrow { self.m2 } else { 0 }))
    }

    #[inline(always)]
    fn value(x: &[u64]) -> [u64; 2] {
        let mut a = [0; 2];
        a[..x.len()].copy_from_slice(x);
        a
    }

    fn limbs(a: [u64; 2]) -> Vec<u64> {
        a.to_vec()
    }

    #[inline(always)]
    fn low(a: [u64; 2]) -> u64 {
        a[0]
    }

    fn r2(&self) -> [u64; 2] {
        r2_of(&self.m)
    }

    #[inline(always)]
    fn select(c: bool, a: [u64; 2], b: [u64; 2]) -> [u64; 2] {
        select_array(c, a, b)
    }

    #[inline(always)]
    fn is_zero(&self, a: [u64; 2]) -> bool {
        a == [0, 0] || a == self.m
    }
}

// Three to eight limbs, lazily reduced

/// Montgomery's arithmetic modulo an odd `m` of `N` limbs below `R / 4`, `R = B^N`, residues kept
/// below `2m` as in [`Lazy2`]: the powers' coarsely integrated operand scanning, without the
/// subtraction at the end, as `(a b + q m) / R < 2m` for `a, b < 2m`.
#[derive(Clone, Copy)]
pub(super) struct LazyN<const N: usize> {
    /// The modulus, the low limb first.
    m: [u64; N],
    /// `2m`.
    m2: [u64; N],
    /// `-1/m mod B`.
    minv: u64,
}

impl<const N: usize> LazyN<N> {
    /// The arithmetic modulo the odd `m < R / 4`.
    #[inline]
    pub(super) fn new(m: [u64; N]) -> Self {
        let mut m2 = [0; N];
        let mut carry = 0;
        for j in 0..N {
            m2[j] = (m[j] << 1) | carry;
            carry = m[j] >> 63;
        }
        LazyN {
            m,
            m2,
            minv: binvert_limb(m[0]).wrapping_neg(),
        }
    }
}

impl<const N: usize> Registers for LazyN<N> {
    type V = [u64; N];

    #[inline(always)]
    fn sqr(&self, a: [u64; N]) -> [u64; N] {
        self.mul(a, a)
    }

    #[inline(always)]
    fn mul(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        let m = &self.m;
        let mut t = [0u64; N];
        let mut top = 0u64;
        for &bi in b.iter() {
            let mut c = 0u64;
            for j in 0..N {
                let p = u128::from(a[j]) * u128::from(bi) + u128::from(t[j]) + u128::from(c);
                t[j] = p as u64;
                c = (p >> 64) as u64;
            }
            let s = u128::from(top) + u128::from(c);
            let (tn, tn1) = (s as u64, (s >> 64) as u64);
            let q = t[0].wrapping_mul(self.minv);
            let p = u128::from(q) * u128::from(m[0]) + u128::from(t[0]);
            let mut c = (p >> 64) as u64;
            for j in 1..N {
                let p = u128::from(q) * u128::from(m[j]) + u128::from(t[j]) + u128::from(c);
                t[j - 1] = p as u64;
                c = (p >> 64) as u64;
            }
            let s = u128::from(tn) + u128::from(c);
            t[N - 1] = s as u64;
            top = tn1 + (s >> 64) as u64;
        }
        // Below `2m < R`: `top` is 0.
        debug_assert!(top == 0);
        t
    }
}

impl<const N: usize> Lanes for LazyN<N> {
    #[inline(always)]
    fn modulus(&self) -> &[u64] {
        &self.m
    }

    #[inline(always)]
    fn add(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        // The sum, below `4m < R`, less `2m` when that is not negative.
        let mut s = [0; N];
        let mut carry = false;
        for j in 0..N {
            let (x, c1) = a[j].overflowing_add(b[j]);
            let (x, c2) = x.overflowing_add(u64::from(carry));
            s[j] = x;
            carry = c1 | c2;
        }
        let mut d = [0; N];
        let mut borrow = false;
        for j in 0..N {
            let (x, b1) = s[j].overflowing_sub(self.m2[j]);
            let (x, b2) = x.overflowing_sub(u64::from(borrow));
            d[j] = x;
            borrow = b1 | b2;
        }
        select_array(borrow, s, d)
    }

    #[inline(always)]
    fn sub(&self, a: [u64; N], b: [u64; N]) -> [u64; N] {
        sub_mod(a, b, &self.m2)
    }

    #[inline(always)]
    fn value(x: &[u64]) -> [u64; N] {
        let mut a = [0; N];
        a[..x.len()].copy_from_slice(x);
        a
    }

    fn limbs(a: [u64; N]) -> Vec<u64> {
        a.to_vec()
    }

    #[inline(always)]
    fn low(a: [u64; N]) -> u64 {
        a[0]
    }

    fn r2(&self) -> [u64; N] {
        r2_of(&self.m)
    }

    #[inline(always)]
    fn select(c: bool, a: [u64; N], b: [u64; N]) -> [u64; N] {
        select_array(c, a, b)
    }

    #[inline(always)]
    fn is_zero(&self, a: [u64; N]) -> bool {
        a == [0; N] || a == self.m
    }
}

// In memory

/// Two limbs aligned to 16 bytes, as the kernels' working memory starts.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Pair([u64; 2]);

/// Residues of nine limbs and more in memory, by the powers' [`Montgomery`]: the kernels' product
/// or square, REDC, and one subtraction of `n` when the result is not below it, as REDC of a
/// product of two residues below `n` is below `2n`.
pub(super) struct Big {
    /// The modulus, which `mont` points to.
    m: Vec<u64>,
    /// The arithmetic.
    mont: Montgomery,
    /// The working memory `mont` keeps, `1/n mod R` when REDC takes products.
    _keep: Vec<Pair>,
    /// The scratch of the products, used by one operation at a time.
    work: UnsafeCell<Vec<Pair>>,
    /// `R mod n`.
    one: Vec<u64>,
    /// `R^2 mod n`.
    r2: Vec<u64>,
    /// `R^3 mod n`.
    r3: Vec<u64>,
    /// `-1/n mod B^2`, for the kernel's REDC called directly below [`BASECASE`] limbs.
    minv: [u64; 2],
}

/// Below this many limbs (Karatsuba's threshold), the products of [`Big`] call the schoolbook
/// kernels and the kernel's REDC directly, without the dispatch of the longer products.
const BASECASE: usize = 24;

impl Big {
    /// `r = a b` or `r = a^2` (with `b` null) by the kernels, into the scratch then REDC to `r`,
    /// below `n`.
    ///
    /// # Safety
    ///
    /// The limbs must be valid, `n` each; `r` may be `a` or `b`.
    #[inline(always)]
    unsafe fn product(&self, r: *mut u64, a: *const u64, b: *const u64) {
        let nn = self.m.len();
        let t = self.scratch();
        // SAFETY: as the caller guarantees; the product's `2n` limbs at the start of the scratch,
        // which `Montgomery::itch` gives.
        unsafe {
            if nn < BASECASE {
                let p = t.take(2 * nn);
                if b.is_null() {
                    basecase::sqr(p, a, nn);
                } else {
                    basecase::mul(p, a, b, nn);
                }
                if redc_limbs(r, p, self.m.as_ptr(), nn, self.minv) {
                    sub_n(r, r, self.m.as_ptr(), nn);
                }
            } else if b.is_null() {
                self.mont.sqr(r, a, t);
            } else {
                self.mont.mul(r, a, b, t);
            }
            self.mont.reduce(r);
        }
    }

    /// The residues modulo the odd `n` of the limbs `n`, at least three, the top one not zero.
    pub(super) fn new(n: &[u64]) -> Big {
        let nn = n.len();
        debug_assert!(nn >= 3);
        let m = n.to_vec();
        let keep_len = Montgomery::new_itch(nn);
        let mut keep = vec![Pair([0; 2]); keep_len.div_ceil(2)];
        // SAFETY: the modulus is odd, of `nn >= 3` limbs, and stays where it is with `m`, whose
        // heap buffer does not move; the working memory stays with `keep`.
        let mont = unsafe {
            Montgomery::new(
                m.as_ptr(),
                nn,
                Scratch::new(keep.as_mut_ptr().cast(), keep_len),
            )
        };
        let itch = Montgomery::itch(nn).max(rem_itch(2 * nn + 1, nn));
        let work = UnsafeCell::new(vec![Pair([0; 2]); itch.div_ceil(2)]);
        let mut big = Big {
            m,
            mont,
            _keep: keep,
            work,
            one: vec![0; nn],
            r2: vec![0; nn],
            r3: vec![0; nn],
            minv: redc_inverse(n[0], n[1]),
        };
        // SAFETY: the scratch holds `rem_itch(2nn + 1, nn)` limbs.
        unsafe {
            rem(
                big.r2.as_mut_ptr(),
                &1,
                1,
                2 * nn,
                big.m.as_ptr(),
                nn,
                big.scratch(),
            );
        }
        let mut r3 = vec![0; nn];
        big.sqr(&mut r3, &big.r2);
        big.r3 = r3;
        let mut unit = vec![0; nn];
        unit[0] = 1;
        let mut one = vec![0; nn];
        big.mul(&mut one, &big.r2, &unit);
        big.one = one;
        big
    }

    /// The working memory of the products.
    #[inline(always)]
    fn scratch(&self) -> Scratch {
        // SAFETY: the vector is only read here, for its pointer and length, and the scratch is
        // used by one operation at a time, which nothing else interleaves: `Big` is not `Sync`.
        let work = unsafe { &mut *self.work.get() };
        Scratch::new(work.as_mut_ptr().cast(), 2 * work.len())
    }
}

impl Arith for Big {
    type V = Vec<u64>;

    const MEMORY: bool = true;

    #[inline(always)]
    fn n(&self) -> &[u64] {
        &self.m
    }

    fn zero(&self) -> Vec<u64> {
        vec![0; self.m.len()]
    }

    fn one(&self) -> Vec<u64> {
        self.one.clone()
    }

    fn residue(&self, x: &[u64]) -> Vec<u64> {
        let mut a = vec![0; self.m.len()];
        a[..x.len()].copy_from_slice(x);
        let mut r = vec![0; self.m.len()];
        self.mul(&mut r, &a, &self.r2);
        r
    }

    fn small(&self, c: u64) -> Vec<u64> {
        self.residue(&[c])
    }

    fn raw(&self, a: &Vec<u64>) -> Vec<u64> {
        a.clone()
    }

    #[inline(always)]
    fn low(&self, a: &Vec<u64>) -> u64 {
        a[0]
    }

    fn of_raw(&self, x: &[u64]) -> Vec<u64> {
        let mut a = vec![0; self.m.len()];
        a[..x.len()].copy_from_slice(x);
        a
    }

    #[inline(always)]
    fn cube(&self) -> &Vec<u64> {
        &self.r3
    }

    #[inline]
    fn mul(&self, r: &mut Vec<u64>, a: &Vec<u64>, b: &Vec<u64>) {
        // SAFETY: the residues hold `nn` limbs each; `r` may be `a` or `b`, as the product goes to
        // the scratch first.
        unsafe { self.product(r.as_mut_ptr(), a.as_ptr(), b.as_ptr()) };
    }

    #[inline]
    fn mul_in(&self, r: &mut Vec<u64>, b: &Vec<u64>) {
        let p = r.as_mut_ptr();
        // SAFETY: as in `mul`, with `r` as an operand too.
        unsafe { self.product(p, p, b.as_ptr()) };
    }

    #[inline]
    fn sqr(&self, r: &mut Vec<u64>, a: &Vec<u64>) {
        // SAFETY: as in `mul`.
        unsafe { self.product(r.as_mut_ptr(), a.as_ptr(), core::ptr::null()) };
    }

    #[inline]
    fn sqr_in(&self, r: &mut Vec<u64>) {
        let p = r.as_mut_ptr();
        // SAFETY: as in `mul`.
        unsafe { self.product(p, p, core::ptr::null()) };
    }

    #[inline]
    fn add(&self, r: &mut Vec<u64>, a: &Vec<u64>, b: &Vec<u64>) {
        let nn = self.m.len();
        // SAFETY: the residues hold `nn` limbs each, `r` apart from `a` and `b`.
        unsafe {
            if add_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), nn)
                || cmp_n(r.as_ptr(), self.m.as_ptr(), nn) != Ordering::Less
            {
                sub_n(r.as_mut_ptr(), r.as_ptr(), self.m.as_ptr(), nn);
            }
        }
    }

    #[inline]
    fn add_in(&self, r: &mut Vec<u64>, b: &Vec<u64>) {
        let nn = self.m.len();
        let p = r.as_mut_ptr();
        // SAFETY: as in `add`; the kernels allow the result at an operand.
        unsafe {
            if add_n(p, p, b.as_ptr(), nn) || cmp_n(p, self.m.as_ptr(), nn) != Ordering::Less {
                sub_n(p, p, self.m.as_ptr(), nn);
            }
        }
    }

    #[inline]
    fn sub(&self, r: &mut Vec<u64>, a: &Vec<u64>, b: &Vec<u64>) {
        let nn = self.m.len();
        // SAFETY: as in `add`.
        unsafe {
            if sub_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), nn) {
                add_n(r.as_mut_ptr(), r.as_ptr(), self.m.as_ptr(), nn);
            }
        }
    }

    #[inline]
    fn sub_in(&self, r: &mut Vec<u64>, b: &Vec<u64>) {
        let nn = self.m.len();
        let p = r.as_mut_ptr();
        // SAFETY: as in `add_in`.
        unsafe {
            if sub_n(p, p, b.as_ptr(), nn) {
                add_n(p, p, self.m.as_ptr(), nn);
            }
        }
    }

    #[inline]
    fn is_zero(&self, a: &Vec<u64>) -> bool {
        a.iter().all(|&x| x == 0)
    }

    #[inline]
    fn cswap(&self, c: bool, a: &mut Vec<u64>, b: &mut Vec<u64>) {
        if c {
            core::mem::swap(a, b);
        }
    }
}

/// `x / p` for the odd `p` that divides `x`, by Hensel's exact division with `inv = 1/p mod B`;
/// `None` when `p` does not divide `x`, which the carry out of the top limb tells.
pub(super) fn div_exact_1(x: &[u64], p: u64, inv: u64) -> Option<Vec<u64>> {
    let mut q = Vec::with_capacity(x.len());
    let mut c = 0u64;
    for &l in x {
        let (s, b) = l.overflowing_sub(c);
        let qi = s.wrapping_mul(inv);
        q.push(qi);
        c = ((u128::from(qi) * u128::from(p)) >> 64) as u64 + u64::from(b);
    }
    if c != 0 {
        return None;
    }
    trim(&mut q);
    Some(q)
}

/// `1/p mod B` for the odd `p`.
#[inline]
pub(super) fn inverse_limb(p: u64) -> u64 {
    binvert_limb(p)
}

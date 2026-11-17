//! Kernels on polynomials modulo a word `p ≥ 2`, the coefficients words below `p`, the constant
//! first, as a [`ModPolynomial`](super::super::ModPolynomial) of one limb holds them: the loops of
//! the factorisation on the arithmetic of [`arith`](super::super::arith).
//!
//! The products are those of [`wmul`] (the schoolbook method with one reduction a coefficient,
//! the number-theoretic transforms on the coefficients); the remainders modulo a fixed `f`, the
//! products and the powers modulo it are those of [`WordRem`], which keeps the inverse of the
//! reversed `f`; a division by a polynomial met once is the delayed-reduction schoolbook division
//! of [`wmul`] for a short quotient or divisor, Newton's method beyond. What stays here is the
//! matrix of the baby steps of Brent and Kung's composition ([`Powers`]), which one `h` serves for
//! many compositions, its sums of products kept in one, two or three words and reduced once.

use super::super::arith::wmul;
use super::super::arith::wrem::{self, WordRem};
use crate::math::numbers::{ByteOrder, from_limbs};
use core::cell::RefCell;

pub(crate) use super::super::arith::nmod::WordMod as Wm;

/// From this many coefficients in both the quotient and the divisor, a division met once goes by
/// the Newton inverse of the reversed divisor; below, by the schoolbook method.
pub(crate) const NEWTON_DIV: usize = 256;

/// A sum of products of residues, unreduced.
pub(crate) trait Acc: Copy + Default {
    /// The accumulator holding `x`.
    fn of(x: u64) -> Self;
    /// `self += a·b`.
    fn madd(&mut self, a: u64, b: u64);
    /// The sum modulo `p`.
    fn red(self, w: &Wm) -> u64;
}

impl Acc for u64 {
    #[inline(always)]
    fn of(x: u64) -> Self {
        x
    }
    #[inline(always)]
    fn madd(&mut self, a: u64, b: u64) {
        // Exact by the class: both below 2^32, the sum below 2^64.
        *self = self.wrapping_add(a.wrapping_mul(b));
    }
    #[inline(always)]
    fn red(self, w: &Wm) -> u64 {
        w.reduce1(self)
    }
}

impl Acc for u128 {
    #[inline(always)]
    fn of(x: u64) -> Self {
        u128::from(x)
    }
    #[inline(always)]
    fn madd(&mut self, a: u64, b: u64) {
        *self = self.wrapping_add(u128::from(a) * u128::from(b));
    }
    #[inline(always)]
    fn red(self, w: &Wm) -> u64 {
        w.reduce_u128(self)
    }
}

/// An accumulator of three words.
#[derive(Clone, Copy, Default)]
pub(crate) struct A3 {
    lo: u128,
    hi: u64,
}

impl Acc for A3 {
    #[inline(always)]
    fn of(x: u64) -> Self {
        A3 {
            lo: u128::from(x),
            hi: 0,
        }
    }
    #[inline(always)]
    fn madd(&mut self, a: u64, b: u64) {
        let (s, c) = self.lo.overflowing_add(u128::from(a) * u128::from(b));
        self.lo = s;
        self.hi += u64::from(c);
    }
    #[inline(always)]
    fn red(self, w: &Wm) -> u64 {
        w.reduce3(self.hi, (self.lo >> 64) as u64, self.lo as u64)
    }
}

/// `⌈log₂ n⌉` for `n ≥ 1`.
#[inline]
pub(crate) fn clog2(n: usize) -> u32 {
    usize::BITS - (n.max(1) - 1).leading_zeros()
}

/// The accumulator a sum of `terms` products of residues modulo `w` and one residue needs: 1, 2
/// or 3 words.
#[inline]
pub(crate) fn class(w: &Wm, terms: usize) -> u32 {
    let need = 2 * w.bits() + clog2(terms + 1);
    if need <= 64 {
        1
    } else if need <= 128 {
        2
    } else {
        3
    }
}

/// Calls the kernel `$f` with the accumulator of the class of `$terms` products modulo `$w`.
macro_rules! by_class {
    ($w:expr, $terms:expr, $f:ident ( $($arg:expr),* )) => {
        match class($w, $terms) {
            1 => $f::<u64>($($arg),*),
            2 => $f::<u128>($($arg),*),
            _ => $f::<A3>($($arg),*),
        }
    };
}
pub(crate) use by_class;

/// Drops the zeros at the top.
#[inline]
pub(crate) fn trim(a: &mut Vec<u64>) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

/// Whether the transforms of the arith area run on NEON lanes here (elsewhere their products
/// keep the arith area's own choice of method).
const NEON: bool = cfg!(all(
    target_arch = "aarch64",
    target_endian = "little",
    target_pointer_width = "64",
    not(miri)
));

/// Up to this many coefficients in the shorter operand, a product modulo a `p` of `bits` bits
/// goes by the schoolbook method (the arith area's NEON and dot-product kernels): measured on an
/// Apple M4 against the transforms, whose length doubles past 64 coefficients.
fn school_len(bits: u32) -> usize {
    if cfg!(miri) {
        return 6;
    }
    // Kronecker substitution takes over first for the smallest moduli (see `ks_len`).
    match bits {
        0..=3 => 64,
        4..=6 => 80,
        7..=10 => 120,
        11..=16 => 150,
        17..=32 => 110,
        _ => 96,
    }
}

/// Up to this many coefficients in the longer operand, a product modulo a small `p` of `bits`
/// bits goes by Kronecker substitution into the crate's product of integers, whose fields of
/// `2 bits + log₂ len` bits are short; beyond, by the transforms, which cost the same for every
/// `p` of up to 28 bits.
fn ks_len(bits: u32) -> usize {
    if cfg!(miri) {
        return if bits <= 13 { 24 } else { 0 };
    }
    match bits {
        0..=3 => 2048,
        4..=6 => 1024,
        7..=10 => 512,
        11..=13 => 300,
        _ => 0,
    }
}

/// The product of `a` and `b` (the square of `a` when `b` is empty), both not empty, into `r` of
/// `la + lb - 1` coefficients: the arith area's kernels, by the schoolbook method, Kronecker
/// substitution for a small `p`, the transforms beyond.
pub(crate) fn mul_into(r: &mut [u64], a: &[u64], b: &[u64], w: &Wm) {
    let square = b.is_empty();
    let (la, lb) = (a.len(), if square { a.len() } else { b.len() });
    debug_assert!(la >= 1 && lb >= 1 && r.len() == la + lb - 1);
    if NEON || cfg!(miri) {
        let bits = w.bits();
        if la.min(lb) <= school_len(bits) {
            if square {
                wmul::school_sqr(r, a, w);
            } else {
                wmul::school(r, a, b, w);
            }
            return;
        }
        if la.max(lb) <= ks_len(bits) {
            wmul::ks(r, a, b, w, &mut Vec::new());
            return;
        }
    }
    if square {
        wmul::sqr_into(r, a, w);
    } else {
        wmul::mul_into(r, a, b, w);
    }
}

/// The product of `a` and `b`, each possibly with zeros at the top: `la + lb - 1` coefficients,
/// none when one is empty.
#[inline]
pub(crate) fn mul_raw(w: &Wm, a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = vec![0u64; a.len() + b.len() - 1];
    mul_into(&mut r, a, b, w);
    r
}

/// The square of `a`, possibly with zeros at the top.
#[inline]
pub(crate) fn sqr_raw(w: &Wm, a: &[u64]) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut r = vec![0u64; 2 * a.len() - 1];
    mul_into(&mut r, a, &[], w);
    r
}

/// The product, trimmed.
pub(crate) fn mul(w: &Wm, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut r = mul_raw(w, a, b);
    trim(&mut r);
    r
}

/// The inverse of `a` modulo `p`, `None` when it is not a unit: the extended Euclidean algorithm
/// on `(p, a)` with the cofactors of `a` kept as magnitudes (their signs alternate, and they stay
/// below `p`), the quotients 1 to 3 by subtractions.
pub(crate) fn inv(w: &Wm, a: u64) -> Option<u64> {
    let p = w.n;
    let (mut r0, mut r1) = (p, if a >= p { a % p } else { a });
    let (mut t0, mut t1) = (0u64, 1u64);
    // t1 is the cofactor of r1 with the sign (-1)^odd... tracked by `odd`.
    let mut odd = false;
    while r1 != 0 {
        let d = r0 - r1;
        let (q, r) = if d < r1 {
            (1, d)
        } else if d - r1 < r1 {
            (2, d - r1)
        } else if d - 2 * r1 < r1 {
            (3, d - 2 * r1)
        } else {
            (r0 / r1, r0 % r1)
        };
        (r0, r1) = (r1, r);
        (t0, t1) = (t1, t0 + q * t1);
        odd = !odd;
    }
    // r0 = ± t0·a mod p, the sign + after an odd number of steps.
    (r0 == 1).then(|| if odd || t0 == 0 { t0 } else { p - t0 })
}

/// The inverse of the leading coefficient of `f`, not empty.
#[track_caller]
fn lead_inv(w: &Wm, f: &[u64]) -> u64 {
    inv(w, f[f.len() - 1]).expect("the leading coefficient is not a unit modulo n")
}

/// The quotient and the remainder of `a` by `f` (trimmed, not zero): the schoolbook method with
/// delayed reduction for a short quotient or divisor, Newton's method beyond.
#[track_caller]
pub(crate) fn divrem(w: &Wm, a: &[u64], f: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let n = f.len();
    assert!(n > 0, "division by the zero polynomial");
    let inv = lead_inv(w, f);
    if a.len() < n {
        return (Vec::new(), a.to_vec());
    }
    let lq = a.len() + 1 - n;
    let (mut q, mut r) = if lq < NEWTON_DIV || n < NEWTON_DIV {
        wmul::div_rem_school(a, f, inv, w)
    } else {
        divrem_newton(w, a, f, inv)
    };
    trim(&mut q);
    trim(&mut r);
    (q, r)
}

/// The division by the Newton inverse of the reversed `f`: the quotient's reversal is that of
/// the top of `a` times it modulo `x^lq`.
fn divrem_newton(w: &Wm, a: &[u64], f: &[u64], inv: u64) -> (Vec<u64>, Vec<u64>) {
    let (la, n) = (a.len(), f.len());
    let lq = la + 1 - n;
    let rf: Vec<u64> = f.iter().rev().take(lq).copied().collect();
    let mut finv = wrem::inv_series(&rf, lq, inv, w);
    finv.resize(lq, 0);
    let ra: Vec<u64> = a[n - 1..].iter().rev().copied().collect();
    let mut q = mul_raw(w, &ra, &finv);
    q.truncate(lq);
    q.reverse();
    let mut r = a[..n - 1].to_vec();
    if n > 1 {
        let qf = mul_raw(w, &q, &f[..n - 1]);
        for (x, &y) in r.iter_mut().zip(&qf) {
            *x = w.sub(*x, y);
        }
    }
    (q, r)
}

/// The remainder of `a` by `f` (trimmed, not zero).
#[track_caller]
pub(crate) fn rem(w: &Wm, a: &[u64], f: &[u64]) -> Vec<u64> {
    let n = f.len();
    assert!(n > 0, "division by the zero polynomial");
    if a.len() < n {
        let mut r = a.to_vec();
        trim(&mut r);
        return r;
    }
    let lq = a.len() + 1 - n;
    if lq < NEWTON_DIV || n < NEWTON_DIV {
        let mut r = wmul::rem_school(a, f, lead_inv(w, f), w, None);
        trim(&mut r);
        return r;
    }
    divrem(w, a, f).1
}

/// The inverse of the power series `h` (`h[0]` a unit) modulo `xⁿ`: `n` coefficients, possibly
/// with zeros at the top.
#[track_caller]
pub(crate) fn inv_series(w: &Wm, h: &[u64], n: usize) -> Vec<u64> {
    let h0 = inv(w, h[0]).expect("the constant coefficient is not a unit modulo n");
    let mut g = wrem::inv_series(&h[..h.len().min(n)], n, h0, w);
    g.resize(n, 0);
    g
}

/// Up to this many coefficients in the shorter operand, [`gcd`] keeps the Euclid here.
const OWN_GCD: usize = 3;

/// The monic greatest common divisor of `a` and `b` (trimmed); empty when both are.
///
/// The gcd seam (word moduli): the gcd area's Euclid in Montgomery's form (fraction-free steps
/// for short divisors) and its half gcd past the crossover; up to [`OWN_GCD`] coefficients in
/// the shorter operand, Euclid's algorithm on the delayed-reduction schoolbook remainders here.
///
/// Measured (Apple M4): the gcd area's from degree 4 (2 to 4 times faster at degrees 8 to 64),
/// this one at degree 2 (by 20%).
pub(crate) fn gcd(w: &Wm, a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.len().min(b.len()) > OWN_GCD {
        return super::super::gcd::word_gcd(w.n, a, b);
    }
    let (mut a, mut b) = if a.len() >= b.len() {
        (a.to_vec(), b.to_vec())
    } else {
        (b.to_vec(), a.to_vec())
    };
    while !b.is_empty() {
        let mut r = wmul::rem_school(&a, &b, lead_inv(w, &b), w, None);
        trim(&mut r);
        a = b;
        b = r;
    }
    if let Some(&l) = a.last()
        && l != 1
    {
        let li = inv(w, l).expect("the leading coefficient is not a unit modulo n");
        for x in a.iter_mut() {
            *x = w.mul(*x, li);
        }
    }
    a
}

/// Up to this many bits in `p`, the products and remainders modulo a long `f` take the private
/// path of [`Modf`] (Kronecker substitution products, see [`ks_len`]); above, [`WordRem`]'s.
fn own_bits() -> u32 {
    if NEON || cfg!(miri) { 13 } else { 0 }
}

/// Up to this many coefficients in `f`, the remainders modulo `f` by the private
/// delayed-reduction schoolbook division (one, two or three words a sum, by the class of `p` and
/// the steps, no product by the leading coefficient's inverse: `f` is monic).
fn own_school() -> usize {
    if cfg!(miri) { 8 } else { 150 }
}

/// How the remainders modulo `f` go.
enum Path {
    /// The private schoolbook division.
    School,
    /// The private Newton division with the inverse of the reversed `f` modulo `x^(n-1)`.
    Own(Vec<u64>),
    /// [`WordRem`]'s, on working memory reused from call to call.
    Shared(Box<WordRem>, RefCell<wrem::Work>),
}

/// A monic modulus `f` of `n ≥ 2` coefficients with what its remainders, products and powers
/// need: the private schoolbook division for a short `f`; for a long one [`WordRem`] (the inverse
/// of the reversed `f` kept), or for a small `p` the same method on the products of [`mul_into`].
pub(crate) struct Modf {
    /// The arithmetic modulo `p`.
    w: Wm,
    /// `f`.
    f: Vec<u64>,
    /// The remainders.
    path: Path,
}

/// The delayed-reduction schoolbook remainder of `a` by the monic `f`: the low `n - 1`
/// coefficients, each sum in the accumulator `A`.
fn rem_school<A: Acc>(w: &Wm, a: &[u64], f: &[u64]) -> Vec<u64> {
    let (la, n) = (a.len(), f.len());
    let mut acc: Vec<A> = a.iter().map(|&x| A::of(x)).collect();
    for i in (n - 1..la).rev() {
        let r = acc[i].red(w);
        if r == 0 {
            continue;
        }
        let c = w.neg(r);
        let base = i + 1 - n;
        for (x, y) in acc[base..i].iter_mut().zip(&f[..n - 1]) {
            x.madd(c, *y);
        }
    }
    acc[..n - 1].iter().map(|x| x.red(w)).collect()
}

/// Up to this many coefficients in `f`, a product modulo `f` (a square when `square`) modulo a
/// `p` of `bits` bits in one pass: the sums of products of the operands left unreduced, the
/// schoolbook division run on those sums, one reduction a coefficient of the product in all
/// (instead of one for the product and one for the division). Measured on an Apple M4 against
/// the schoolbook product (whose squares take half the products) and division.
fn fused_max(bits: u32, square: bool) -> usize {
    if cfg!(miri) {
        return 6;
    }
    match (bits, square) {
        (0..=32, _) => 32,
        (_, true) => 48,
        (_, false) => 80,
    }
}

/// `a·b mod f` (`a` and `b` reduced, not empty, `f` monic) in one pass, the sums in `A`.
fn mul_fused<A: Acc>(w: &Wm, a: &[u64], b: &[u64], f: &[u64]) -> Vec<u64> {
    let (la, lb, lf) = (a.len(), b.len(), f.len());
    let len = la + lb - 1;
    let mut stack = [A::default(); 64];
    let mut heap = Vec::new();
    let need = len.max(lf - 1);
    let acc = if need <= 64 {
        &mut stack[..need]
    } else {
        heap.resize(need, A::default());
        &mut heap[..]
    };
    for (i, &x) in a.iter().enumerate() {
        if x == 0 {
            continue;
        }
        for (o, &y) in acc[i..].iter_mut().zip(b) {
            o.madd(x, y);
        }
    }
    let fl = &f[..lf - 1];
    for top in (lf - 1..len).rev() {
        let t = acc[top].red(w);
        if t == 0 {
            continue;
        }
        let nt = w.n - t;
        for (o, &y) in acc[top + 1 - lf..top].iter_mut().zip(fl) {
            o.madd(nt, y);
        }
    }
    let mut r: Vec<u64> = acc[..lf - 1].iter().map(|x| x.red(w)).collect();
    trim(&mut r);
    r
}

impl Modf {
    /// `a·b mod f` in one pass, for a short `f` (see [`fused_max`]).
    #[inline]
    fn fused(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        let w = &self.w;
        let terms = a.len().min(b.len()) + (a.len() + b.len()).saturating_sub(self.f.len());
        by_class!(w, terms, mul_fused(w, a, b, &self.f))
    }

    /// The modulus `f`, monic, of degree at least 1.
    pub(crate) fn new(w: &Wm, f: &[u64]) -> Modf {
        debug_assert!(f.len() >= 2 && f[f.len() - 1] == 1);
        let n = f.len();
        let path = if n <= own_school() {
            Path::School
        } else if w.bits() <= own_bits() {
            let rev: Vec<u64> = f.iter().rev().copied().collect();
            Path::Own(inv_series(w, &rev, n - 1))
        } else {
            Path::Shared(
                Box::new(WordRem::new(f, 1, *w, None)),
                RefCell::new(wrem::Work::default()),
            )
        };
        Modf {
            w: *w,
            f: f.to_vec(),
            path,
        }
    }

    /// The degree of `f`.
    #[inline]
    pub(crate) fn deg(&self) -> usize {
        self.f.len() - 1
    }

    /// The remainder of `a`, possibly with zeros at the top, by `f`, trimmed.
    pub(crate) fn rem(&self, a: &[u64]) -> Vec<u64> {
        if a.len() <= self.deg() {
            let mut v = a.to_vec();
            trim(&mut v);
            return v;
        }
        let mut v = match &self.path {
            Path::School => {
                let w = &self.w;
                by_class!(w, a.len() - self.deg(), rem_school(w, a, &self.f))
            }
            Path::Own(finv) => {
                let mut v = a.to_vec();
                self.own_rem(&mut v, finv);
                v
            }
            Path::Shared(r, work) => {
                let mut v = a.to_vec();
                r.rem(&mut v, &mut work.borrow_mut());
                v
            }
        };
        trim(&mut v);
        v
    }

    /// The private Newton remainder of `a` in place, its low `n - 1` coefficients left.
    fn own_rem(&self, a: &mut Vec<u64>, finv: &[u64]) {
        let w = &self.w;
        let f = &self.f;
        let lf = f.len();
        while a.len() > 2 * lf - 2 {
            // The top 2 lf - 2 coefficients reduced at a time.
            let cut = a.len() - (2 * lf - 2);
            let mut top = a.split_off(cut);
            self.own_rem(&mut top, finv);
            a.extend_from_slice(&top);
        }
        let la = a.len();
        if la < lf {
            return;
        }
        let lq = la + 1 - lf;
        let ra: Vec<u64> = a[lf - 1..].iter().rev().copied().collect();
        let mut q = mul_raw(w, &ra, &finv[..lq]);
        q.truncate(lq);
        q.reverse();
        let qf = mul_raw(w, &q, &f[..lf - 1]);
        a.truncate(lf - 1);
        for (x, &y) in a.iter_mut().zip(&qf) {
            *x = w.sub(*x, y);
        }
    }

    /// `a·b mod f` for `a` and `b` reduced.
    pub(crate) fn mulmod(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        if let Path::Shared(r, work) = &self.path {
            let mut out = Vec::new();
            r.mul(&mut out, a, b, &mut work.borrow_mut());
            trim(&mut out);
            return out;
        }
        if self.f.len() <= fused_max(self.w.bits(), false) {
            return self.fused(a, b);
        }
        self.rem(&mul_raw(&self.w, a, b))
    }

    /// `a² mod f` for `a` reduced.
    pub(crate) fn sqrmod(&self, a: &[u64]) -> Vec<u64> {
        if a.is_empty() {
            return Vec::new();
        }
        if let Path::Shared(r, work) = &self.path {
            let mut out = Vec::new();
            r.sqr(&mut out, a, &mut work.borrow_mut());
            trim(&mut out);
            return out;
        }
        if self.f.len() <= fused_max(self.w.bits(), true) {
            return self.fused(a, a);
        }
        self.rem(&sqr_raw(&self.w, a))
    }

    /// `x·a mod f` for `a` reduced.
    fn mulx(&self, a: &[u64]) -> Vec<u64> {
        if a.is_empty() {
            return Vec::new();
        }
        let w = &self.w;
        let f = &self.f;
        let mut r = Vec::with_capacity(a.len() + 1);
        r.push(0);
        r.extend_from_slice(a);
        if r.len() == f.len() {
            let t = r.pop().expect("not empty");
            if t != 0 {
                let nt = w.neg(t);
                for (x, y) in r.iter_mut().zip(f) {
                    *x = w.add(*x, w.mul(nt, *y));
                }
            }
        }
        trim(&mut r);
        r
    }

    /// `aᵉ mod f` for `a` reduced and the exponent of the limbs `e`, the least significant first.
    pub(crate) fn powmod(&self, a: &[u64], e: &[u64]) -> Vec<u64> {
        let bits = limb_bits(e);
        if bits == 0 {
            return self.rem(&[1]);
        }
        if a.is_empty() {
            return Vec::new();
        }
        if let Path::Shared(r, _) = &self.path {
            let mut r = r.pow(a, &from_limbs(e, ByteOrder::LittleEndian));
            trim(&mut r);
            return r;
        }
        let bit = |i: u64| (e[(i / 64) as usize] >> (i % 64)) & 1 == 1;
        if a == [0, 1] {
            // x^e by squares and shifts.
            let mut r = a.to_vec();
            for i in (0..bits - 1).rev() {
                r = self.sqrmod(&r);
                if bit(i) {
                    r = self.mulx(&r);
                }
            }
            return r;
        }
        let k: u64 = match bits {
            0..=8 => 1,
            9..=24 => 2,
            25..=80 => 3,
            81..=240 => 4,
            _ => 5,
        };
        // The odd powers a, a³, …, a^(2^k - 1).
        let a2 = self.sqrmod(a);
        let mut odd = vec![a.to_vec()];
        for i in 1..(1usize << (k - 1)) {
            let t = self.mulmod(&odd[i - 1], &a2);
            odd.push(t);
        }
        let mut r: Option<Vec<u64>> = None;
        let mut i = bits as i64 - 1;
        while i >= 0 {
            if !bit(i as u64) {
                if let Some(x) = r.as_mut() {
                    *x = self.sqrmod(x);
                }
                i -= 1;
                continue;
            }
            // The window from bit i down to the lowest set bit within k bits.
            let mut j = (i - k as i64 + 1).max(0);
            while !bit(j as u64) {
                j += 1;
            }
            let mut v = 0usize;
            for b in (j..=i).rev() {
                v = (v << 1) | usize::from(bit(b as u64));
            }
            r = Some(match r {
                None => odd[v >> 1].clone(),
                Some(mut x) => {
                    for _ in j..=i {
                        x = self.sqrmod(&x);
                    }
                    self.mulmod(&x, &odd[v >> 1])
                }
            });
            i = j - 1;
        }
        r.expect("a set bit")
    }

    /// `xᵉ mod f` for the exponent of the limbs `e`: squares and shifts.
    pub(crate) fn powmod_x(&self, e: &[u64]) -> Vec<u64> {
        let x = self.rem(&[0, 1]);
        self.powmod(&x, e)
    }
}

/// The number of bits of the limbs `e`.
#[inline]
pub(crate) fn limb_bits(e: &[u64]) -> u64 {
    match e.iter().rposition(|&l| l != 0) {
        None => 0,
        Some(i) => 64 * i as u64 + u64::from(64 - e[i].leading_zeros()),
    }
}

/// The powers `1, h, …, h^(m-1)` of a residue `h` modulo `f`, as the rows of a matrix of `deg f`
/// columns, and `h^m`: the baby steps of the Brent–Kung composition, kept for every composition
/// with the same `h`.
pub(crate) struct Powers {
    /// The number of rows, `m`.
    pub(crate) m: usize,
    /// The columns, `deg f`.
    cols: usize,
    /// The matrix transposed: the coefficients of `x^c` in `1, h, …, h^(m-1)` one after the other,
    /// for each column `c`.
    cols_t: Vec<u64>,
    /// `h^m mod f`.
    hm: Vec<u64>,
}

impl Powers {
    /// The powers of `h mod f` for compositions with polynomials of `len` coefficients, `m` of
    /// them (`⌈√len⌉` when `None`).
    pub(crate) fn new(h: &[u64], mf: &Modf, len: usize, m: Option<usize>) -> Powers {
        let cols = mf.deg();
        let m = m
            .unwrap_or_else(|| (len.max(1) as f64).sqrt().ceil() as usize)
            .max(1);
        let mut cols_t = vec![0u64; m * cols];
        let h = mf.rem(h);
        let mut p = mf.rem(&[1]);
        for i in 0..m {
            for (c, &x) in p.iter().enumerate() {
                cols_t[c * m + i] = x;
            }
            p = mf.mulmod(&p, &h);
        }
        Powers {
            m,
            cols,
            cols_t,
            hm: p,
        }
    }

    /// `g(h) mod f`.
    pub(crate) fn compose(&self, w: &Wm, g: &[u64], mf: &Modf) -> Vec<u64> {
        if g.is_empty() {
            return Vec::new();
        }
        let blocks = g.len().div_ceil(self.m);
        let c = by_class!(w, self.m, matmul(self, w, g, blocks));
        let mut r = c[blocks - 1].clone();
        trim(&mut r);
        for j in (0..blocks - 1).rev() {
            r = mf.mulmod(&r, &self.hm);
            let cj = &c[j];
            if r.len() < cj.len() {
                r.resize(cj.len(), 0);
            }
            for (x, y) in r.iter_mut().zip(cj) {
                *x = w.add(*x, *y);
            }
            trim(&mut r);
        }
        r
    }
}

/// The rows `Σᵢ g[jm + i]·hⁱ`, reduced, for each block `j` of `m` coefficients of `g`: each
/// coefficient a dot product of the block with a column of the powers, summed in registers.
fn matmul<A: Acc>(pw: &Powers, w: &Wm, g: &[u64], blocks: usize) -> Vec<Vec<u64>> {
    let (m, cols) = (pw.m, pw.cols);
    let mut gb = g.to_vec();
    gb.resize(blocks * m, 0);
    let mut out = vec![vec![0u64; cols]; blocks];
    for (c, col) in pw.cols_t.chunks_exact(m).enumerate() {
        for (o, b) in out.iter_mut().zip(gb.chunks_exact(m)) {
            let mut acc = A::default();
            for (x, y) in b.iter().zip(col) {
                acc.madd(*x, *y);
            }
            o[c] = acc.red(w);
        }
    }
    out
}

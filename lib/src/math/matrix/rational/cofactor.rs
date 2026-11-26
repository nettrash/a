//! Determinants of integer matrices of orders 2 to 4 by cofactors in fixed-width registers, and
//! the solutions and inverses of order 2 by Cramer's rule, for elements of up to 64 bits.
//!
//! Bareiss's elimination, in registers or on limbs, divides at every step; at these orders the
//! cofactor expansion is cheaper still, and needs no bound to stay exact: the minors of order 2
//! of the last two rows, those of order 3 from them, and the determinant from those, each a sum of
//! products of a minor by an element. With elements below `2^b`, a determinant of order `n` is
//! below `n!·2^(n·b)`: it is computed in an `i128` when that fits (`b ≤ 62` at order 2, `b ≤ 41`
//! at order 3, `b ≤ 30` at order 4), and otherwise in five limbs in two's complement, room for
//! `24·2^256`. Order 2 takes the two products of magnitudes in `u128`s and their sum or
//! difference with its 129th bit, for any elements of 64 bits.
//!
//! The solution and the inverse of order 2 are Cramer's: `x = (e·y₀ - b·y₁)/D` and
//! `y = (a·y₁ - c·y₀)/D` with `D = a·e - b·c`, and the inverse `(e, -b; -c, a)/D`, each element
//! reduced by one greatest common divisor in 128 bits; when `D` or a numerator passes 128 bits the
//! caller takes the elimination instead.

use super::super::Matrix;
use crate::math::numbers::{ByteOrder, Magnitude, Number, fraction};
use crate::math::rational::Rational;

/// An element: its sign and magnitude.
#[derive(Clone, Copy, Default)]
struct E {
    neg: bool,
    m: u64,
}

impl E {
    /// The element `x` when it fits in 64 bits.
    #[inline(always)]
    fn of(x: &Number) -> Option<E> {
        let m = Magnitude::of(x);
        if m.bytes.len() > 8 {
            return None;
        }
        let v = m.small() as u64;
        Some(E {
            neg: m.negative && v != 0,
            m: v,
        })
    }

    /// As an `i128`.
    #[inline(always)]
    fn i(self) -> i128 {
        let v = i128::from(self.m);
        if self.neg { -v } else { v }
    }
}

/// The elements of the `n × n` matrix `a`, `n ≤ 4`, row after row, and the bits of the largest
/// magnitude; `None` when one passes 64 bits.
#[inline(always)]
fn read(a: &Matrix<Number>, n: usize) -> Option<([[E; 4]; 4], u32)> {
    let mut e = [[E::default(); 4]; 4];
    let mut or = 0u64;
    for (i, row) in e.iter_mut().enumerate().take(n) {
        for (j, slot) in row.iter_mut().enumerate().take(n) {
            let v = E::of(&a[(i, j)])?;
            or |= v.m;
            *slot = v;
        }
    }
    Some((e, 64 - or.leading_zeros()))
}

/// The arithmetic of the cofactors: an `i128`, or five limbs.
trait Arith: Copy {
    /// `±x·y`.
    fn product(x: E, y: E) -> Self;
    /// `self + o`.
    fn add(self, o: Self) -> Self;
    /// `self - o`.
    fn sub(self, o: Self) -> Self;
    /// `self·y`.
    fn times(self, y: E) -> Self;
}

impl Arith for i128 {
    #[inline(always)]
    fn product(x: E, y: E) -> i128 {
        x.i() * y.i()
    }
    #[inline(always)]
    fn add(self, o: i128) -> i128 {
        self + o
    }
    #[inline(always)]
    fn sub(self, o: i128) -> i128 {
        self - o
    }
    #[inline(always)]
    fn times(self, y: E) -> i128 {
        self * y.i()
    }
}

/// A signed integer of five limbs in two's complement, the least significant first.
#[derive(Clone, Copy)]
struct W([u64; 5]);

impl W {
    /// Whether it is below zero.
    #[inline(always)]
    fn negative(self) -> bool {
        self.0[4] >> 63 == 1
    }

    /// `-self`.
    #[inline(always)]
    fn neg(self) -> W {
        let mut r = [0u64; 5];
        let mut c = 1u64;
        for i in 0..5 {
            let (s, o) = (!self.0[i]).overflowing_add(c);
            r[i] = s;
            c = u64::from(o);
        }
        W(r)
    }

    /// The sign and the magnitude.
    #[inline(always)]
    fn magnitude(self) -> (bool, [u64; 5]) {
        if self.negative() {
            (true, self.neg().0)
        } else {
            (false, self.0)
        }
    }

    /// The number, in the byte order `order`.
    fn number(self, order: ByteOrder) -> Number {
        let (neg, m) = self.magnitude();
        number_of(neg, &m, order)
    }
}

impl Arith for W {
    #[inline(always)]
    fn product(x: E, y: E) -> W {
        let p = u128::from(x.m) * u128::from(y.m);
        let w = W([p as u64, (p >> 64) as u64, 0, 0, 0]);
        if x.neg != y.neg { w.neg() } else { w }
    }
    #[inline(always)]
    fn add(self, o: W) -> W {
        let mut r = [0u64; 5];
        let mut c = false;
        for i in 0..5 {
            let (s, o1) = self.0[i].overflowing_add(o.0[i]);
            let (s, o2) = s.overflowing_add(u64::from(c));
            r[i] = s;
            c = o1 | o2;
        }
        W(r)
    }
    #[inline(always)]
    fn sub(self, o: W) -> W {
        let mut r = [0u64; 5];
        let mut b = false;
        for i in 0..5 {
            let (s, o1) = self.0[i].overflowing_sub(o.0[i]);
            let (s, o2) = s.overflowing_sub(u64::from(b));
            r[i] = s;
            b = o1 | o2;
        }
        W(r)
    }
    #[inline(always)]
    fn times(self, y: E) -> W {
        let (neg, m) = self.magnitude();
        let mut r = [0u64; 5];
        let mut c = 0u64;
        for i in 0..5 {
            let t = u128::from(m[i]) * u128::from(y.m) + u128::from(c);
            r[i] = t as u64;
            c = (t >> 64) as u64;
        }
        let w = W(r);
        if neg != y.neg { w.neg() } else { w }
    }
}

/// The number of the sign `neg` and the magnitude `m` (limbs, the least significant first), in the
/// byte order `order`, in a vector of its significant bytes.
fn number_of(neg: bool, m: &[u64], order: ByteOrder) -> Number {
    let mut n = m.len();
    while n > 0 && m[n - 1] == 0 {
        n -= 1;
    }
    if n == 0 {
        return Number::new_from_u8(0, order);
    }
    let len = 8 * n - m[n - 1].leading_zeros() as usize / 8;
    let mut v = Vec::with_capacity(len);
    for &w in &m[..n] {
        v.extend_from_slice(&w.to_le_bytes());
    }
    v.truncate(len);
    if order == ByteOrder::BigEndian {
        v.reverse();
    }
    Number {
        value: v,
        negative: neg,
        byte_order: order,
    }
}

/// The determinant of order 3 or 4 of the elements `e`, in the arithmetic `T`.
#[inline(always)]
fn expand<T: Arith>(e: &[[E; 4]; 4], n: usize) -> T {
    let m2 = |r: usize, j: usize, k: usize| {
        T::product(e[r][j], e[r + 1][k]).sub(T::product(e[r][k], e[r + 1][j]))
    };
    if n == 3 {
        let (m12, m02, m01) = (m2(1, 1, 2), m2(1, 0, 2), m2(1, 0, 1));
        return m12
            .times(e[0][0])
            .sub(m02.times(e[0][1]))
            .add(m01.times(e[0][2]));
    }
    let (m01, m02, m03) = (m2(2, 0, 1), m2(2, 0, 2), m2(2, 0, 3));
    let (m12, m13, m23) = (m2(2, 1, 2), m2(2, 1, 3), m2(2, 2, 3));
    let r = &e[1];
    let c0 = m23.times(r[1]).sub(m13.times(r[2])).add(m12.times(r[3]));
    let c1 = m23.times(r[0]).sub(m03.times(r[2])).add(m02.times(r[3]));
    let c2 = m13.times(r[0]).sub(m03.times(r[1])).add(m01.times(r[3]));
    let c3 = m12.times(r[0]).sub(m02.times(r[1])).add(m01.times(r[2]));
    let r = &e[0];
    c0.times(r[0])
        .sub(c1.times(r[1]))
        .add(c2.times(r[2]))
        .sub(c3.times(r[3]))
}

/// `±p ∓ q` for products of magnitudes `p = |a·e|`, `q = |b·c|` with the signs `sp` and `sq` of
/// the products: the sign and the magnitude, in 129 bits.
#[inline(always)]
fn difference(sp: bool, p: u128, sq: bool, q: u128) -> (bool, u128, bool) {
    if sp != sq {
        // Of the sign of `p`: `|p| + |q|`.
        let (s, c) = p.overflowing_add(q);
        (sp, s, c)
    } else if p >= q {
        (sp, p - q, false)
    } else {
        (!sp, q - p, false)
    }
}

/// `x·y - z·w` for elements: its sign, magnitude and 129th bit.
#[inline(always)]
fn det2(x: E, y: E, z: E, w: E) -> (bool, u128, bool) {
    let p = u128::from(x.m) * u128::from(y.m);
    let q = u128::from(z.m) * u128::from(w.m);
    let r = difference(x.neg != y.neg, p, z.neg != w.neg, q);
    // Zero is not negative.
    (r.0 && (r.1 != 0 || r.2), r.1, r.2)
}

/// The determinant of the `n × n` matrix of integers `a`, `2 ≤ n ≤ 4`, when its elements fit in 64
/// bits; `None` otherwise.
pub(super) fn det(a: &Matrix<Number>, n: usize, order: ByteOrder) -> Option<Number> {
    debug_assert!((2..=4).contains(&n));
    let (e, bits) = read(a, n)?;
    if n == 2 {
        let (neg, m, top) = det2(e[0][0], e[1][1], e[0][1], e[1][0]);
        return Some(number_of(
            neg,
            &[m as u64, (m >> 64) as u64, u64::from(top)],
            order,
        ));
    }
    let small = if n == 3 { bits <= 41 } else { bits <= 30 };
    if small {
        let d: i128 = expand(&e, n);
        let m = d.unsigned_abs();
        return Some(number_of(d < 0, &[m as u64, (m >> 64) as u64], order));
    }
    Some(expand::<W>(&e, n).number(order))
}

/// `x / g` for a divisor `g` of `x`, both not zero.
#[inline(always)]
fn quo(x: u128, g: u128) -> u128 {
    if g == 1 {
        x
    } else if (x | g) >> 64 == 0 {
        u128::from(x as u64 / g as u64)
    } else {
        x / g
    }
}

/// The rational `±n / d` in lowest terms, `d` not zero, in the byte order `order`.
#[inline(always)]
fn fraction(neg: bool, n: u128, d: u128, order: ByteOrder) -> Rational {
    if n == 0 {
        return Rational::from_word(false, 0, 1, order);
    }
    let g = fraction::gcd_wide(n, d);
    Rational::from_u128(neg, quo(n, g), quo(d, g), order)
}

/// The determinant `D` of order 2 of `a` when it fits in 128 bits: its sign and magnitude; `None`
/// when an element passes 64 bits or `D` passes 128.
#[inline(always)]
fn det_128(a: &Matrix<Number>) -> Option<([[E; 4]; 4], bool, u128)> {
    let (e, _) = read(a, 2)?;
    let (neg, m, top) = det2(e[0][0], e[1][1], e[0][1], e[1][0]);
    if top {
        return None;
    }
    Some((e, neg, m))
}

/// The solution `X` of `A·X = B` for `A` of order 2, by Cramer's rule: `Some(None)` for a singular
/// `A`, and `None` when an element passes 64 bits or a determinant 128 (the elimination's).
pub(super) fn solve(
    a: &Matrix<Number>,
    b: &Matrix<Number>,
    order: ByteOrder,
) -> Option<Option<Vec<Rational>>> {
    let (e, dneg, d) = det_128(a)?;
    if d == 0 {
        return Some(None);
    }
    let k = b.cols();
    let mut ys = Vec::with_capacity(2 * k);
    for j in 0..k {
        ys.push((E::of(&b[(0, j)])?, E::of(&b[(1, j)])?));
    }
    let mut x = Vec::with_capacity(2 * k);
    for (y0, y1) in ys {
        // `e·y₀ - b·y₁` and `a·y₁ - c·y₀`.
        let (n0, m0, t0) = det2(e[1][1], y0, e[0][1], y1);
        let (n1, m1, t1) = det2(e[0][0], y1, e[1][0], y0);
        if t0 || t1 {
            return None;
        }
        x.push(fraction(n0 != dneg, m0, d, order));
        x.push(fraction(n1 != dneg, m1, d, order));
    }
    Some(Some(x))
}

/// The inverse of `A` of order 2 as rationals, column after column, by Cramer's rule: `Some(None)`
/// for a singular `A`, and `None` when an element passes 64 bits or the determinant 128 bits.
pub(super) fn inverse(a: &Matrix<Number>, order: ByteOrder) -> Option<Option<Vec<Rational>>> {
    let (e, dneg, d) = det_128(a)?;
    if d == 0 {
        return Some(None);
    }
    // `(e, -b; -c, a) / D`, column after column: `e, -c, -b, a`.
    let f = |x: E, flip: bool| fraction((x.neg != flip) != dneg, u128::from(x.m), d, order);
    Some(Some(vec![
        f(e[1][1], false),
        f(e[1][0], true),
        f(e[0][1], true),
        f(e[0][0], false),
    ]))
}

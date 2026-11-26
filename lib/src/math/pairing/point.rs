//! The arithmetic that `G1` and `G2` share: points of `y² = x³ + b` in Jacobian coordinates over a
//! coordinate field, `F_p` for `G1` and `F_p²` for `G2`.

use crate::math::field::{Fp, Fp2};
use crate::math::numbers::{ByteOrder, Number};

/// What the point arithmetic needs of its coordinate field.
pub(super) trait Coordinate: Clone + PartialEq {
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn square(&self) -> Self;
    fn double(&self) -> Self;
    fn neg(&self) -> Self;
    fn is_zero(&self) -> bool;
    /// 0 of the element's field.
    fn zero(&self) -> Self;
    /// 1 of the element's field.
    fn one(&self) -> Self;
}

impl Coordinate for Fp<'_> {
    fn add(&self, o: &Self) -> Self {
        self + o
    }
    fn sub(&self, o: &Self) -> Self {
        self - o
    }
    fn mul(&self, o: &Self) -> Self {
        self * o
    }
    fn square(&self) -> Self {
        Fp::square(self)
    }
    fn double(&self) -> Self {
        Fp::double(self)
    }
    fn neg(&self) -> Self {
        -self
    }
    fn is_zero(&self) -> bool {
        Fp::is_zero(self)
    }
    fn zero(&self) -> Self {
        self.field().zero()
    }
    fn one(&self) -> Self {
        self.field().one()
    }
}

impl Coordinate for Fp2<'_> {
    fn add(&self, o: &Self) -> Self {
        self + o
    }
    fn sub(&self, o: &Self) -> Self {
        self - o
    }
    fn mul(&self, o: &Self) -> Self {
        self * o
    }
    fn square(&self) -> Self {
        Fp2::square(self)
    }
    fn double(&self) -> Self {
        Fp2::double(self)
    }
    fn neg(&self) -> Self {
        -self
    }
    fn is_zero(&self) -> bool {
        Fp2::is_zero(self)
    }
    fn zero(&self) -> Self {
        Fp2::zero(self.tower())
    }
    fn one(&self) -> Self {
        Fp2::one(self.tower())
    }
}

/// A point `(X : Y : Z)` of `y² = x³ + b`, `x = X/Z²`, `y = Y/Z³`, the point at infinity for
/// `Z = 0`.
#[derive(Clone)]
pub(super) struct Jacobian<F> {
    pub(super) x: F,
    pub(super) y: F,
    pub(super) z: F,
}

impl<F: Coordinate> Jacobian<F> {
    /// The affine point `(x, y)`.
    pub(super) fn from_affine(x: F, y: F) -> Self {
        let z = x.one();
        Jacobian { x, y, z }
    }

    /// The point at infinity, `(1 : 1 : 0)`, in the field of `like`.
    pub(super) fn identity(like: &F) -> Self {
        Jacobian {
            x: like.one(),
            y: like.one(),
            z: like.zero(),
        }
    }

    pub(super) fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    pub(super) fn neg(&self) -> Self {
        Jacobian {
            x: self.x.clone(),
            y: self.y.neg(),
            z: self.z.clone(),
        }
    }

    /// `2P`, by "dbl-2009-l" of the Explicit-Formulas Database (Lange), for `a = 0`: two products
    /// and five squares.
    pub(super) fn double(&self) -> Self {
        if self.is_identity() || self.y.is_zero() {
            return Jacobian::identity(&self.x);
        }
        let a = self.x.square();
        let b = self.y.square();
        let c = b.square();
        let d = self.x.add(&b).square().sub(&a).sub(&c).double();
        let e = a.double().add(&a);
        let f = e.square();
        let x3 = f.sub(&d.double());
        let c8 = c.double().double().double();
        let y3 = e.mul(&d.sub(&x3)).sub(&c8);
        let z3 = self.y.mul(&self.z).double();
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `P + Q`, by "add-2007-bl" of the Explicit-Formulas Database: eleven products and five
    /// squares, with the doubling and the opposite points apart.
    pub(super) fn add(&self, o: &Self) -> Self {
        if self.is_identity() {
            return o.clone();
        }
        if o.is_identity() {
            return self.clone();
        }
        let z1z1 = self.z.square();
        let z2z2 = o.z.square();
        let u1 = self.x.mul(&z2z2);
        let u2 = o.x.mul(&z1z1);
        let s1 = self.y.mul(&o.z).mul(&z2z2);
        let s2 = o.y.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&u1);
        let r = s2.sub(&s1).double();
        if h.is_zero() {
            return if r.is_zero() {
                self.double()
            } else {
                Jacobian::identity(&self.x)
            };
        }
        let i = h.double().square();
        let j = h.mul(&i);
        let v = u1.mul(&i);
        let x3 = r.square().sub(&j).sub(&v.double());
        let y3 = r.mul(&v.sub(&x3)).sub(&s1.mul(&j).double());
        let z3 = self.z.add(&o.z).square().sub(&z1z1).sub(&z2z2).mul(&h);
        Jacobian {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `k P` for any integer `k`, by the width-4 non-adjacent form of `|k|` with the odd multiples
    /// `P, 3P, 5P, 7P`: about `bits(k)` doublings and `bits(k)/5` additions.
    pub(super) fn mul(&self, k: &Number) -> Self {
        let digits = wnaf(k, 4);
        if digits.is_empty() || self.is_identity() {
            return Jacobian::identity(&self.x);
        }
        let p2 = self.double();
        let mut odd = vec![self.clone()];
        for i in 1..4 {
            let next = odd[i - 1].add(&p2);
            odd.push(next);
        }
        let mut r = Jacobian::identity(&self.x);
        for &d in digits.iter().rev() {
            r = r.double();
            if d > 0 {
                r = r.add(&odd[(d as usize) / 2]);
            } else if d < 0 {
                r = r.add(&odd[(-d as usize) / 2].neg());
            }
        }
        if k.is_negative() { r.neg() } else { r }
    }

    /// Whether two points are the same, without an inversion.
    pub(super) fn equals(&self, o: &Self) -> bool {
        match (self.is_identity(), o.is_identity()) {
            (true, true) => true,
            (false, false) => {
                let z1z1 = self.z.square();
                let z2z2 = o.z.square();
                self.x.mul(&z2z2) == o.x.mul(&z1z1)
                    && self.y.mul(&o.z).mul(&z2z2) == o.y.mul(&self.z).mul(&z1z1)
            }
            _ => false,
        }
    }
}

/// The 64-bit limbs of `|k|`, least significant first.
pub(super) fn limbs(k: &Number) -> Vec<u64> {
    let mut bytes = k.value.clone();
    if k.byte_order == ByteOrder::BigEndian {
        bytes.reverse();
    }
    bytes
        .chunks(8)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        })
        .collect()
}

/// The width-`w` non-adjacent form of `|k|`, least significant digit first: odd digits below
/// `2^(w−1)` in absolute value, each followed by at least `w − 1` zeros.
pub(super) fn wnaf(k: &Number, w: u32) -> Vec<i8> {
    let mut l = limbs(k);
    l.push(0);
    let window = 1u64 << w;
    let half = window >> 1;
    let mut out = Vec::with_capacity(l.len() * 64 + 1);
    while l.iter().any(|&x| x != 0) {
        if l[0] & 1 == 1 {
            let m = l[0] & (window - 1);
            if m >= half {
                // k − (m − 2^w) = k + (2^w − m).
                add_small(&mut l, window - m);
                out.push((m as i64 - window as i64) as i8);
            } else {
                l[0] -= m; // m ≤ l[0], so there is no borrow.
                out.push(m as i8);
            }
        } else {
            out.push(0);
        }
        // l >>= 1.
        for i in 0..l.len() {
            let next = if i + 1 < l.len() { l[i + 1] << 63 } else { 0 };
            l[i] = (l[i] >> 1) | next;
        }
    }
    out
}

/// The digits in `{−1, 0, 1}` of `|k|` for a Miller loop, the least significant first: the
/// non-adjacent form when it has fewer nonzero digits than the binary form (BN254's `6x + 2`: 21
/// additions instead of 36), the binary form otherwise (BLS12-381's `|x|`, where the non-adjacent
/// form is a doubling longer for the same five additions).
pub(super) fn loop_digits(k: &Number) -> Vec<i8> {
    let k = &k.abs();
    let n = wnaf(k, 2);
    let weight = n.iter().filter(|&&d| d != 0).count() as u64;
    if weight < k.count_ones() {
        n
    } else {
        (0..k.bits()).map(|i| i8::from(k.bit(i))).collect()
    }
}

/// `l += x`, with room for the carry in the top limb.
fn add_small(l: &mut [u64], x: u64) {
    let mut carry = x;
    for limb in l.iter_mut() {
        let (s, c) = limb.overflowing_add(carry);
        *limb = s;
        if !c {
            return;
        }
        carry = 1;
    }
}

/// The operators of a group of points (`G1` or `G2`): `+`, `-`, unary `-`, and `*` by a
/// [`Number`] on either side, for owned and borrowed operands, with their assignments; and
/// `PartialEq`, `Eq`, `Hash`, `Debug` and `Display`.
macro_rules! group_ops {
    ($ty:ident) => {
        impl<'c> core::ops::AddAssign<&$ty<'c>> for $ty<'c> {
            #[track_caller]
            fn add_assign(&mut self, rhs: &$ty<'c>) {
                *self = self.sum(rhs);
            }
        }
        impl<'c> core::ops::AddAssign<$ty<'c>> for $ty<'c> {
            #[track_caller]
            fn add_assign(&mut self, rhs: $ty<'c>) {
                *self += &rhs;
            }
        }
        impl<'c> core::ops::SubAssign<&$ty<'c>> for $ty<'c> {
            #[track_caller]
            fn sub_assign(&mut self, rhs: &$ty<'c>) {
                *self = self.sum(&rhs.negated());
            }
        }
        impl<'c> core::ops::SubAssign<$ty<'c>> for $ty<'c> {
            #[track_caller]
            fn sub_assign(&mut self, rhs: $ty<'c>) {
                *self -= &rhs;
            }
        }
        $crate::math::pairing::point::group_binary!($ty, Add, add, AddAssign, add_assign);
        $crate::math::pairing::point::group_binary!($ty, Sub, sub, SubAssign, sub_assign);

        impl<'c> core::ops::Neg for &$ty<'c> {
            type Output = $ty<'c>;
            fn neg(self) -> $ty<'c> {
                self.negated()
            }
        }
        impl<'c> core::ops::Neg for $ty<'c> {
            type Output = $ty<'c>;
            fn neg(self) -> $ty<'c> {
                -&self
            }
        }

        impl<'c> core::ops::MulAssign<&Number> for $ty<'c> {
            fn mul_assign(&mut self, k: &Number) {
                *self = self.times(k);
            }
        }
        impl<'c> core::ops::MulAssign<Number> for $ty<'c> {
            fn mul_assign(&mut self, k: Number) {
                *self *= &k;
            }
        }
        impl<'c> core::ops::Mul<&Number> for &$ty<'c> {
            type Output = $ty<'c>;
            fn mul(self, k: &Number) -> $ty<'c> {
                self.times(k)
            }
        }
        impl<'c> core::ops::Mul<Number> for &$ty<'c> {
            type Output = $ty<'c>;
            fn mul(self, k: Number) -> $ty<'c> {
                self * &k
            }
        }
        impl<'c> core::ops::Mul<&Number> for $ty<'c> {
            type Output = $ty<'c>;
            fn mul(self, k: &Number) -> $ty<'c> {
                &self * k
            }
        }
        impl<'c> core::ops::Mul<Number> for $ty<'c> {
            type Output = $ty<'c>;
            fn mul(self, k: Number) -> $ty<'c> {
                &self * &k
            }
        }
        impl<'c> core::ops::Mul<&$ty<'c>> for &Number {
            type Output = $ty<'c>;
            fn mul(self, p: &$ty<'c>) -> $ty<'c> {
                p * self
            }
        }
        impl<'c> core::ops::Mul<$ty<'c>> for &Number {
            type Output = $ty<'c>;
            fn mul(self, p: $ty<'c>) -> $ty<'c> {
                &p * self
            }
        }
        impl<'c> core::ops::Mul<&$ty<'c>> for Number {
            type Output = $ty<'c>;
            fn mul(self, p: &$ty<'c>) -> $ty<'c> {
                p * &self
            }
        }
        impl<'c> core::ops::Mul<$ty<'c>> for Number {
            type Output = $ty<'c>;
            fn mul(self, p: $ty<'c>) -> $ty<'c> {
                &p * &self
            }
        }

        /// Two points are equal when they are the same point, in whatever projective form each is
        /// held; this needs no inversion.
        impl PartialEq for $ty<'_> {
            #[track_caller]
            fn eq(&self, other: &Self) -> bool {
                self.same_point(other)
            }
        }

        impl Eq for $ty<'_> {}

        /// By the affine coordinates, the same for equal points: each hash takes an inversion.
        impl core::hash::Hash for $ty<'_> {
            fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
                self.coordinates().hash(state);
            }
        }

        impl core::fmt::Debug for $ty<'_> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                core::fmt::Display::fmt(self, f)
            }
        }

        /// `(x, y)` in decimal, or `infinity`.
        impl core::fmt::Display for $ty<'_> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self.coordinates() {
                    Some((x, y)) => write!(f, "({x}, {y})"),
                    None => f.write_str("infinity"),
                }
            }
        }
    };
}

/// A binary operator of points for the four combinations of owned and borrowed operands, from its
/// assignment.
macro_rules! group_binary {
    ($ty:ident, $tr:ident, $m:ident, $atr:ident, $am:ident) => {
        impl<'c> core::ops::$tr<&$ty<'c>> for $ty<'c> {
            type Output = $ty<'c>;
            #[track_caller]
            fn $m(mut self, rhs: &$ty<'c>) -> $ty<'c> {
                core::ops::$atr::$am(&mut self, rhs);
                self
            }
        }
        impl<'c> core::ops::$tr<$ty<'c>> for $ty<'c> {
            type Output = $ty<'c>;
            #[track_caller]
            fn $m(mut self, rhs: $ty<'c>) -> $ty<'c> {
                core::ops::$atr::$am(&mut self, &rhs);
                self
            }
        }
        impl<'c> core::ops::$tr<&$ty<'c>> for &$ty<'c> {
            type Output = $ty<'c>;
            #[track_caller]
            fn $m(self, rhs: &$ty<'c>) -> $ty<'c> {
                let mut r = self.clone();
                core::ops::$atr::$am(&mut r, rhs);
                r
            }
        }
        impl<'c> core::ops::$tr<$ty<'c>> for &$ty<'c> {
            type Output = $ty<'c>;
            #[track_caller]
            fn $m(self, rhs: $ty<'c>) -> $ty<'c> {
                let mut r = self.clone();
                core::ops::$atr::$am(&mut r, &rhs);
                r
            }
        }
    };
}

pub(super) use group_binary;
pub(super) use group_ops;

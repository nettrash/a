//! Points of the curve and of its twist on the engine's arithmetic, in Jacobian coordinates, and
//! the products by scalars: one routine for both groups, over a [`Coord`] of `F_p` or `F_p²`.
//!
//! The formulas are those of the public points, "dbl-2009-l" and "add-2007-bl" of the
//! Explicit-Formulas Database (for `a = 0`), with "madd-2007-bl" when the second point is affine.
//! Products by several scalars at once share their doublings (Straus's method), each scalar in the
//! width-`w` non-adjacent form over a table of odd multiples.

use super::{Engine, Fp, Fp2};

/// A coordinate: an element of `F_p` (for `G1`) or of `F_p²` (for `G2`), held in a store of
/// `[u64; 6]` per element of `F_p`.
pub(crate) trait Coord: Copy + PartialEq {
    /// The elements of `F_p` in one coordinate.
    const PARTS: usize;
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn square(&self) -> Self;
    fn double(&self) -> Self;
    fn neg(&self) -> Self;
    fn inverse(&self) -> Option<Self>;
    /// The element stored in `s[0..PARTS]`.
    fn load(s: &[[u64; 6]]) -> Self;
    /// Stores the element in `s[0..PARTS]`.
    fn store(&self, s: &mut [[u64; 6]]);
}

impl<E: Engine<N>, const N: usize> Coord for Fp<E, N> {
    const PARTS: usize = 1;
    #[inline(always)]
    fn zero() -> Self {
        Fp::zero()
    }
    #[inline(always)]
    fn one() -> Self {
        Fp::one()
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        Fp::is_zero(self)
    }
    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Fp::add(self, o)
    }
    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Fp::sub(self, o)
    }
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        Fp::mul(self, o)
    }
    #[inline(always)]
    fn square(&self) -> Self {
        Fp::square(self)
    }
    #[inline(always)]
    fn double(&self) -> Self {
        Fp::double(self)
    }
    #[inline(always)]
    fn neg(&self) -> Self {
        Fp::neg(self)
    }
    fn inverse(&self) -> Option<Self> {
        Fp::inverse(self)
    }
    #[inline(always)]
    fn load(s: &[[u64; 6]]) -> Self {
        let mut l = [0u64; N];
        l.copy_from_slice(&s[0][..N]);
        Fp::raw(l)
    }
    #[inline(always)]
    fn store(&self, s: &mut [[u64; 6]]) {
        s[0] = [0; 6];
        s[0][..N].copy_from_slice(&self.0);
    }
}

impl<E: Engine<N>, const N: usize> Coord for Fp2<E, N> {
    const PARTS: usize = 2;
    #[inline(always)]
    fn zero() -> Self {
        Fp2::zero()
    }
    #[inline(always)]
    fn one() -> Self {
        Fp2::one()
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        Fp2::is_zero(self)
    }
    #[inline(always)]
    fn add(&self, o: &Self) -> Self {
        Fp2::add(self, o)
    }
    #[inline(always)]
    fn sub(&self, o: &Self) -> Self {
        Fp2::sub(self, o)
    }
    #[inline(always)]
    fn mul(&self, o: &Self) -> Self {
        Fp2::mul(self, o)
    }
    #[inline(always)]
    fn square(&self) -> Self {
        Fp2::square(self)
    }
    #[inline(always)]
    fn double(&self) -> Self {
        Fp2::double(self)
    }
    #[inline(always)]
    fn neg(&self) -> Self {
        Fp2::neg(self)
    }
    fn inverse(&self) -> Option<Self> {
        Fp2::inverse(self)
    }
    #[inline(always)]
    fn load(s: &[[u64; 6]]) -> Self {
        Fp2::new(
            <Fp<E, N> as Coord>::load(&s[0..1]),
            <Fp<E, N> as Coord>::load(&s[1..2]),
        )
    }
    #[inline(always)]
    fn store(&self, s: &mut [[u64; 6]]) {
        Coord::store(&self.c0, &mut s[0..1]);
        Coord::store(&self.c1, &mut s[1..2]);
    }
}

/// A point `(X : Y : Z)`, `x = X/Z²`, `y = Y/Z³`, at infinity when `Z = 0`.
#[derive(Clone, Copy)]
pub(crate) struct Jac<C> {
    pub(crate) x: C,
    pub(crate) y: C,
    pub(crate) z: C,
}

impl<C: Coord> Jac<C> {
    #[inline(always)]
    pub(crate) fn identity() -> Self {
        Jac {
            x: C::one(),
            y: C::one(),
            z: C::zero(),
        }
    }

    #[inline(always)]
    pub(crate) fn from_affine(x: C, y: C) -> Self {
        Jac { x, y, z: C::one() }
    }

    #[inline(always)]
    pub(crate) fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    /// The point in its store: 3 coordinates of [`Coord::PARTS`] elements.
    pub(crate) fn load(s: &[[u64; 6]]) -> Self {
        let k = C::PARTS;
        Jac {
            x: C::load(&s[0..k]),
            y: C::load(&s[k..2 * k]),
            z: C::load(&s[2 * k..3 * k]),
        }
    }

    pub(crate) fn store(&self, s: &mut [[u64; 6]]) {
        let k = C::PARTS;
        self.x.store(&mut s[0..k]);
        self.y.store(&mut s[k..2 * k]);
        self.z.store(&mut s[2 * k..3 * k]);
    }

    #[inline(always)]
    pub(crate) fn neg(&self) -> Self {
        Jac {
            x: self.x,
            y: self.y.neg(),
            z: self.z,
        }
    }

    /// `2P`, "dbl-2009-l": 2M + 5S.
    #[inline]
    pub(crate) fn double(&self) -> Self {
        if self.is_identity() || self.y.is_zero() {
            return Self::identity();
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
        Jac {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `P + Q`, "add-2007-bl": 11M + 5S, with the cases of equal and opposite points.
    #[inline]
    pub(crate) fn add(&self, o: &Self) -> Self {
        if self.is_identity() {
            return *o;
        }
        if o.is_identity() {
            return *self;
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
                Self::identity()
            };
        }
        let i = h.double().square();
        let j = h.mul(&i);
        let v = u1.mul(&i);
        let x3 = r.square().sub(&j).sub(&v.double());
        let y3 = r.mul(&v.sub(&x3)).sub(&s1.mul(&j).double());
        let z3 = self.z.add(&o.z).square().sub(&z1z1).sub(&z2z2).mul(&h);
        Jac {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `self + (x₂, y₂)` for an affine point: "madd-2007-bl" of the Explicit-Formulas Database,
    /// seven products and four squares, the doubling for equal points.
    pub(crate) fn add_affine(&self, x2: &C, y2: &C) -> Self {
        if self.is_identity() {
            return Self::from_affine(*x2, *y2);
        }
        let z1z1 = self.z.square();
        let u2 = x2.mul(&z1z1);
        let s2 = y2.mul(&self.z).mul(&z1z1);
        let h = u2.sub(&self.x);
        let r = s2.sub(&self.y).double();
        if h.is_zero() {
            return if r.is_zero() {
                self.double()
            } else {
                Self::identity()
            };
        }
        let hh = h.square();
        let i = hh.double().double();
        let j = h.mul(&i);
        let v = self.x.mul(&i);
        let x3 = r.square().sub(&j).sub(&v.double());
        let y3 = r.mul(&v.sub(&x3)).sub(&self.y.mul(&j).double());
        let z3 = self.z.add(&h).square().sub(&z1z1).sub(&hh);
        Jac {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// Whether two points are equal, whatever their `Z`.
    pub(crate) fn equals(&self, o: &Self) -> bool {
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

    /// The affine coordinates, `None` at infinity.
    pub(crate) fn to_affine(self) -> Option<(C, C)> {
        let zi = self.z.inverse()?;
        let zi2 = zi.square();
        Some((self.x.mul(&zi2), self.y.mul(&zi2.mul(&zi))))
    }

    /// Whether `(x, y)` is on `y² = x³ + b`.
    pub(crate) fn on_curve(x: &C, y: &C, b: &C) -> bool {
        y.square() == x.square().mul(x).add(b)
    }

    /// The product by the nonnegative integer whose bits are `bits` (the most significant last):
    /// double and add, for short scalars of few ones such as `|x|`.
    pub(crate) fn mul_bits(&self, bits: &[bool]) -> Self {
        let mut r = Self::identity();
        for &b in bits.iter().rev() {
            r = r.double();
            if b {
                r = r.add(self);
            }
        }
        r
    }

    /// The product by signed digits (least significant first, in `{−1, 0, 1}`).
    pub(crate) fn mul_signed(&self, digits: &[i8]) -> Self {
        let neg = self.neg();
        let mut r = Self::identity();
        for &d in digits.iter().rev() {
            r = r.double();
            match d {
                1 => r = r.add(self),
                -1 => r = r.add(&neg),
                _ => {}
            }
        }
        r
    }
}

/// The width-`w` non-adjacent form of the nonnegative number of the limbs `k`, least significant
/// digit first.
pub(crate) fn wnaf_limbs(k: &[u64], w: u32) -> Vec<i8> {
    let mut l = k.to_vec();
    l.push(0);
    let window = 1u64 << w;
    let half = window >> 1;
    let mut out = Vec::with_capacity(l.len() * 64 + 1);
    while l.iter().any(|&x| x != 0) {
        if l[0] & 1 == 1 {
            let m = l[0] & (window - 1);
            if m >= half {
                let mut carry = window - m;
                for limb in l.iter_mut() {
                    let (s, c) = limb.overflowing_add(carry);
                    *limb = s;
                    if !c {
                        break;
                    }
                    carry = 1;
                }
                out.push((m as i64 - window as i64) as i8);
            } else {
                l[0] -= m;
                out.push(m as i8);
            }
        } else {
            out.push(0);
        }
        for i in 0..l.len() {
            let next = if i + 1 < l.len() { l[i + 1] << 63 } else { 0 };
            l[i] = (l[i] >> 1) | next;
        }
    }
    out
}

/// `Σ kᵢ Pᵢ` by Straus's method: every scalar in width-`w` non-adjacent form, one run of
/// doublings for all, each point's odd multiples `P, 3P, …, (2^(w−1) − 1)P` precomputed.
/// `terms` are the points with the limbs of their nonnegative scalars.
pub(crate) fn straus<C: Coord>(terms: &[(Jac<C>, &[u64])], w: u32) -> Jac<C> {
    let digits: Vec<Vec<i8>> = terms.iter().map(|(_, k)| wnaf_limbs(k, w)).collect();
    let len = digits.iter().map(Vec::len).max().unwrap_or(0);
    if len == 0 {
        return Jac::identity();
    }
    let size = 1usize << (w - 2);
    let tables: Vec<Vec<Jac<C>>> = terms
        .iter()
        .map(|(p, _)| {
            let mut t = Vec::with_capacity(size);
            t.push(*p);
            if size > 1 {
                let p2 = p.double();
                for i in 1..size {
                    let next = t[i - 1].add(&p2);
                    t.push(next);
                }
            }
            t
        })
        .collect();
    let mut r = Jac::identity();
    // The affine tables cost an inversion and about 7 products a point; each of the additions
    // then saves 4 products and a square. The inversion counts as 125 products in `F_p` and 45 in
    // `F_p²`. Measured on an Apple M4 Max, products by scalars of 4 to 160 bits at width 4 with
    // each kind of table forced, alternated (21 and 31 samples): the affine tables become the
    // faster at about 150 bits on BLS12-381's `G1` (1.05 times the projective ones' time at 80
    // bits, 1.00 at 128, 0.99 at 160), and at 78 and 66 bits on BLS12-381's and BN254's `G2`
    // (BN254's `G1` takes them in every product, by GLV, whose two terms are long).
    let points: usize = tables.iter().map(Vec::len).sum();
    let adds: usize = digits
        .iter()
        .map(|d| d.iter().filter(|&&x| x != 0).count())
        .sum();
    let inversion = if C::PARTS == 1 { 125 } else { 45 };
    let affine = if 5 * adds >= 7 * points + inversion {
        to_affine_all(&tables)
    } else {
        None
    };
    if let Some(affine) = affine {
        // Mixed additions of the affine tables, after one inversion for all of them.
        for i in (0..len).rev() {
            r = r.double();
            for (t, d) in affine.iter().zip(&digits) {
                let Some(&d) = d.get(i) else { continue };
                if d > 0 {
                    let (x, y) = &t[(d as usize) / 2];
                    r = r.add_affine(x, y);
                } else if d < 0 {
                    let (x, y) = &t[(-d as usize) / 2];
                    r = r.add_affine(x, &y.neg());
                }
            }
        }
        return r;
    }
    for i in (0..len).rev() {
        r = r.double();
        for (t, d) in tables.iter().zip(&digits) {
            let Some(&d) = d.get(i) else { continue };
            if d > 0 {
                r = r.add(&t[(d as usize) / 2]);
            } else if d < 0 {
                r = r.add(&t[(-d as usize) / 2].neg());
            }
        }
    }
    r
}

/// The affine forms of all the tables' points by Montgomery's trick (one inversion, three
/// products a point, then a square and two products), or `None` if one is at infinity.
fn to_affine_all<C: Coord>(tables: &[Vec<Jac<C>>]) -> Option<Vec<Vec<(C, C)>>> {
    let points: Vec<&Jac<C>> = tables.iter().flatten().collect();
    if points.iter().any(|p| p.is_identity()) {
        return None;
    }
    // prefix[i] = Z₀···Z_{i−1}.
    let mut prefix = Vec::with_capacity(points.len());
    let mut acc = C::one();
    for p in &points {
        prefix.push(acc);
        acc = acc.mul(&p.z);
    }
    let mut inv = acc.inverse()?;
    let mut flat = vec![(C::zero(), C::zero()); points.len()];
    for (i, p) in points.iter().enumerate().rev() {
        // inv = (Z₀···Z_i)⁻¹ here: 1/Z_i = inv·prefix[i].
        let zi = inv.mul(&prefix[i]);
        inv = inv.mul(&p.z);
        let zi2 = zi.square();
        flat[i] = (p.x.mul(&zi2), p.y.mul(&zi2.mul(&zi)));
    }
    let mut it = flat.into_iter();
    Some(
        tables
            .iter()
            .map(|t| it.by_ref().take(t.len()).collect())
            .collect(),
    )
}

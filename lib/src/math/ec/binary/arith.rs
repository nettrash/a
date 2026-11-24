//! The group law of `y² + x·y = x³ + a·x² + b` over `GF(2ᵐ)` in López–Dahab coordinates.
//!
//! A point `(X : Y : Z)` stands for `x = X / Z`, `y = Y / Z²` (López and Dahab, "Improved
//! algorithms for elliptic curve arithmetic in GF(2ⁿ)", SAC 1998), and `Z = 0` for the point at
//! infinity. The formulas, with `M` a product and `S` a square of the field:
//!
//! - doubling, 4M + 5S (3M + 5S when `b = 1`): `Z₃ = X₁²·Z₁²`, `X₃ = X₁⁴ + b·Z₁⁴`,
//!   `Y₃ = b·Z₁⁴·Z₃ + X₃·(a·Z₃ + Y₁² + b·Z₁⁴)`;
//! - the mixed addition of an affine point, 8M + 5S for `a ∈ {0, 1}` (Al-Daboul, Lim, Kim and Lee,
//!   as in Hankerson, Menezes and Vanstone, *Guide to Elliptic Curve Cryptography*, §3.2.3):
//!   `A = Y₁ + y₂·Z₁²`, `B = X₁ + x₂·Z₁`, `C = Z₁·B`, `D = B²·(C + a·Z₁²)`, `Z₃ = C²`,
//!   `E = A·C`, `X₃ = A² + D + E`, `Y₃ = (E + Z₃)·(X₃ + x₂·Z₃) + (x₂ + y₂)·Z₃²`;
//! - the addition of two projective points, 13M + 5S, from the affine chord law over the common
//!   denominator `C = Z₁·Z₂·(X₁·Z₂ + X₂·Z₁)`;
//! - the negation `−(X : Y : Z) = (X : X·Z + Y : Z)`, from `−(x, y) = (x, x + y)`;
//! - the Frobenius map `(X : Y : Z) → (X² : Y² : Z²)`, an endomorphism when `a` and `b` are in
//!   `GF(2)`, as on Koblitz curves.
//!
//! The additions fall back to the doubling for equal points and give the point at infinity for
//! opposite ones, so they are correct on every pair of points.

use crate::math::field::{BinaryField, F2m};

/// What a product by a coefficient costs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Small {
    /// It is 0.
    Zero,
    /// It is 1.
    One,
    /// Any other element: a product.
    Other,
}

impl Small {
    /// The kind of `x`.
    pub(super) fn of(x: &F2m<'_>) -> Small {
        if x.is_zero() {
            Small::Zero
        } else if x.is_one() {
            Small::One
        } else {
            Small::Other
        }
    }
}

/// A point in López–Dahab coordinates: `x = X / Z`, `y = Y / Z²`, at infinity when `Z = 0`.
#[derive(Clone)]
pub(super) struct Ld<'f> {
    /// `X`.
    pub(super) x: F2m<'f>,
    /// `Y`.
    pub(super) y: F2m<'f>,
    /// `Z`.
    pub(super) z: F2m<'f>,
}

/// A point in affine coordinates, never the point at infinity.
#[derive(Clone)]
pub(super) struct Affine<'f> {
    /// `x`.
    pub(super) x: F2m<'f>,
    /// `y`.
    pub(super) y: F2m<'f>,
}

/// The arithmetic of a curve's points, with its coefficients as elements of its field.
pub(super) struct Ops<'f> {
    /// The field.
    pub(super) f: &'f BinaryField,
    /// `a`.
    pub(super) a: F2m<'f>,
    /// `b`.
    pub(super) b: F2m<'f>,
    /// What a product by `a` costs.
    a_kind: Small,
    /// What a product by `b` costs.
    b_kind: Small,
}

impl<'f> Ops<'f> {
    /// The arithmetic of the curve with the coefficients `a` and `b`, as words of `f`.
    pub(super) fn new(f: &'f BinaryField, a: &[u64], b: &[u64]) -> Ops<'f> {
        let (a, b) = (f.element(a), f.element(b));
        Ops {
            f,
            a_kind: Small::of(&a),
            b_kind: Small::of(&b),
            a,
            b,
        }
    }

    /// `a·t`.
    pub(super) fn times_a(&self, t: &F2m<'f>) -> F2m<'f> {
        match self.a_kind {
            Small::Zero => self.f.zero(),
            Small::One => t.clone(),
            Small::Other => &self.a * t,
        }
    }

    /// `b·t`.
    pub(super) fn times_b(&self, t: &F2m<'f>) -> F2m<'f> {
        match self.b_kind {
            Small::Zero => self.f.zero(),
            Small::One => t.clone(),
            Small::Other => &self.b * t,
        }
    }

    /// Whether `(x, y)` satisfies `y² + x·y = x³ + a·x² + b`.
    pub(super) fn satisfied(&self, x: &F2m<'f>, y: &F2m<'f>) -> bool {
        let x2 = x.square();
        let left = &(y + x) * y;
        let right = &(&(x + &self.a) * &x2) + &self.b;
        left == right
    }

    /// The point at infinity, `(1 : 0 : 0)`.
    pub(super) fn identity(&self) -> Ld<'f> {
        Ld {
            x: self.f.one(),
            y: self.f.zero(),
            z: self.f.zero(),
        }
    }

    /// The affine point `p` in López–Dahab coordinates.
    pub(super) fn of_affine(&self, p: &Affine<'f>) -> Ld<'f> {
        Ld {
            x: p.x.clone(),
            y: p.y.clone(),
            z: self.f.one(),
        }
    }

    /// `−p = (X : X·Z + Y : Z)`.
    pub(super) fn neg(&self, p: &Ld<'f>) -> Ld<'f> {
        if p.z.is_zero() {
            return p.clone();
        }
        Ld {
            x: p.x.clone(),
            y: &(&p.x * &p.z) + &p.y,
            z: p.z.clone(),
        }
    }

    /// `−(x, y) = (x, x + y)`.
    pub(super) fn neg_affine(&self, p: &Affine<'f>) -> Affine<'f> {
        Affine {
            x: p.x.clone(),
            y: &p.x + &p.y,
        }
    }

    /// `2 p`, 4M + 5S.
    pub(super) fn double(&self, p: &Ld<'f>) -> Ld<'f> {
        if p.z.is_zero() {
            return p.clone();
        }
        // `X = 0` is the point `(0, √b)` of order 2: `Z₃` comes out 0.
        let x2 = p.x.square();
        let z2 = p.z.square();
        let z3 = &x2 * &z2;
        let bz4 = self.times_b(&z2.square());
        let x3 = &x2.square() + &bz4;
        let inner = &(&self.times_a(&z3) + &p.y.square()) + &bz4;
        let y3 = &(&bz4 * &z3) + &(&x3 * &inner);
        Ld {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `p + q` for the affine `q`, 8M + 5S for `a ∈ {0, 1}`.
    pub(super) fn add_affine(&self, p: &Ld<'f>, q: &Affine<'f>) -> Ld<'f> {
        if p.z.is_zero() {
            return self.of_affine(q);
        }
        let z1s = p.z.square();
        let a = &p.y + &(&q.y * &z1s);
        let b = &p.x + &(&q.x * &p.z);
        if b.is_zero() {
            // The same `x`: the same point, or its opposite.
            return if a.is_zero() {
                self.double(p)
            } else {
                self.identity()
            };
        }
        let c = &p.z * &b;
        let d = &b.square() * &(&c + &self.times_a(&z1s));
        let z3 = c.square();
        let e = &a * &c;
        let x3 = &(&a.square() + &d) + &e;
        let f = &x3 + &(&q.x * &z3);
        let g = &(&q.x + &q.y) * &z3.square();
        let y3 = &(&(&e + &z3) * &f) + &g;
        Ld {
            x: x3,
            y: y3,
            z: z3,
        }
    }

    /// `p + q`, or `p − q` with `negate`, for the affine `q`, or the point at infinity for `None`.
    pub(super) fn add_option(&self, p: &Ld<'f>, q: Option<&Affine<'f>>, negate: bool) -> Ld<'f> {
        match q {
            None => p.clone(),
            Some(q) if negate => self.add_affine(p, &self.neg_affine(q)),
            Some(q) => self.add_affine(p, q),
        }
    }

    /// `p + q`, 13M + 5S.
    ///
    /// With `B₁ = X₁·Z₂`, `B = B₁ + X₂·Z₁`, `A₁ = Y₁·Z₂²`, `A = A₁ + Y₂·Z₁²`, `F = Z₁·Z₂`,
    /// `C = F·B` and `D = C·B`, the slope is `A / C`, `x₁ = B₁·D / C²` and `y₁ = A₁·D² / C⁴`, so
    /// that `Z₃ = C²`, `X₃ = A² + A·C + D·(B + a·F)` and
    /// `Y₃ = A·C·(B₁·D + X₃) + X₃·C² + A₁·D²`.
    pub(super) fn add(&self, p: &Ld<'f>, q: &Ld<'f>) -> Ld<'f> {
        if p.z.is_zero() {
            return q.clone();
        }
        if q.z.is_zero() {
            return p.clone();
        }
        let z1s = p.z.square();
        let z2s = q.z.square();
        let a1 = &p.y * &z2s;
        let a = &a1 + &(&q.y * &z1s);
        let b1 = &p.x * &q.z;
        let b = &b1 + &(&q.x * &p.z);
        if b.is_zero() {
            return if a.is_zero() {
                self.double(p)
            } else {
                self.identity()
            };
        }
        let f = &p.z * &q.z;
        let c = &f * &b;
        let d = &c * &b;
        let ac = &a * &c;
        let x3 = &(&a.square() + &ac) + &(&d * &(&b + &self.times_a(&f)));
        let c2 = c.square();
        let y3 = &(&(&ac * &(&(&b1 * &d) + &x3)) + &(&x3 * &c2)) + &(&a1 * &d.square());
        Ld {
            x: x3,
            y: y3,
            z: c2,
        }
    }

    /// The Frobenius map `(X : Y : Z) → (X² : Y² : Z²)`.
    pub(super) fn frobenius(&self, p: &Ld<'f>) -> Ld<'f> {
        Ld {
            x: p.x.square(),
            y: p.y.square(),
            z: p.z.square(),
        }
    }

    /// Whether `p` and `q` are the same point: `X₁·Z₂ = X₂·Z₁` and `Y₁·Z₂² = Y₂·Z₁²`.
    pub(super) fn equal(&self, p: &Ld<'f>, q: &Ld<'f>) -> bool {
        match (p.z.is_zero(), q.z.is_zero()) {
            (true, true) => true,
            (false, false) => {
                &p.x * &q.z == &q.x * &p.z && &p.y * &q.z.square() == &q.y * &p.z.square()
            }
            _ => false,
        }
    }

    /// The affine coordinates of each point, `None` for the point at infinity, with one inversion
    /// for all of them, by Montgomery's trick, and five products for each.
    pub(super) fn affine_all(&self, p: &[Ld<'f>]) -> Vec<Option<Affine<'f>>> {
        let mut running = Vec::with_capacity(p.len());
        let mut product = self.f.one();
        for q in p {
            running.push(product.clone());
            if !q.z.is_zero() {
                product = &product * &q.z;
            }
        }
        let mut inverse = product
            .inverse()
            .unwrap_or_else(|| unreachable!("a product of elements that are not zero"));
        let mut out = vec![None; p.len()];
        for (i, q) in p.iter().enumerate().rev() {
            if q.z.is_zero() {
                continue;
            }
            let zi = &inverse * &running[i];
            inverse = &inverse * &q.z;
            out[i] = Some(Affine {
                x: &q.x * &zi,
                y: &q.y * &zi.square(),
            });
        }
        out
    }

    /// The affine coordinates of `p`, `None` for the point at infinity.
    pub(super) fn affine(&self, p: &Ld<'f>) -> Option<Affine<'f>> {
        let zi = p.z.inverse()?;
        Some(Affine {
            x: &p.x * &zi,
            y: &p.y * &zi.square(),
        })
    }
}

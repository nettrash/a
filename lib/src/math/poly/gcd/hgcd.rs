//! The half greatest common divisor (Thull and Yap's formulation of the Knuth–Schönhage
//! algorithm), and the greatest common divisor, the extended one and the resultant built on it,
//! for any coefficient ring with a division the [`Engine`] provides.
//!
//! For `a` of the degree `n > deg b` and `m = ⌈n/2⌉`, [`hgcd`] finds the matrix `M`, the product
//! of the quotient matrices `[[q, 1], [1, 0]]` of Euclid's steps, and the consecutive remainders
//! `(c, d) = M⁻¹(a, b)` of the remainder sequence of `a` and `b` with `deg c ≥ m > deg d`. The
//! quotients of the top halves `a div xᵐ`, `b div xᵐ` are those of `a` and `b` as long as the
//! divisor keeps at least half the degree of the top halves, so the first half of the steps comes
//! from a recursive call on the top halves, one step is taken by a division, and the second half
//! from a recursive call on the top parts of the new pair; each call works on half the degree, and
//! the whole takes `O(M(n) log n)` operations on coefficients.

/// The ring operations, the division and the base-case kernels of an engine of polynomials.
pub(crate) trait Engine {
    /// A polynomial, the constant first, trimmed.
    type P: Clone;
    /// A coefficient.
    type C: Clone;

    /// The number of coefficients up to the leading one.
    fn len(&self, a: &Self::P) -> usize;
    /// The zero polynomial.
    fn zero(&self) -> Self::P;
    /// The constant 1.
    fn one(&self) -> Self::P;
    /// The leading coefficient of a non-zero polynomial.
    fn lead(&self, a: &Self::P) -> Self::C;
    /// `a + b`.
    fn add(&self, a: &Self::P, b: &Self::P) -> Self::P;
    /// `a - b`.
    fn sub(&self, a: &Self::P, b: &Self::P) -> Self::P;
    /// `-a`.
    fn neg(&self, a: &Self::P) -> Self::P {
        self.sub(&self.zero(), a)
    }
    /// `a·b`.
    fn mul(&self, a: &Self::P, b: &Self::P) -> Self::P;
    /// The quotient and the remainder by `b`, not zero.
    fn divrem(&self, a: &Self::P, b: &Self::P) -> (Self::P, Self::P);
    /// `a div xᵏ`.
    fn shr(&self, a: &Self::P, k: usize) -> Self::P;
    /// `a mod xᵏ`, trimmed.
    fn low(&self, a: &Self::P, k: usize) -> Self::P;
    /// `hi·xᵏ + lo`.
    fn shl_add(&self, hi: &Self::P, k: usize, lo: &Self::P) -> Self::P;
    /// `x·y`.
    fn cmul(&self, x: &Self::C, y: &Self::C) -> Self::C;
    /// `-x`.
    fn cneg(&self, x: &Self::C) -> Self::C;
    /// `xᵉ`.
    fn cpow(&self, x: &Self::C, e: u64) -> Self::C;
    /// 1.
    fn cone(&self) -> Self::C;
    /// 0.
    fn czero(&self) -> Self::C;
    /// The last non-zero remainder of Euclid's algorithm on `a` and `b`; with `rec`, the degree
    /// (plus `shift`) and leading coefficient of each divisor, stopping at a constant one.
    fn euclid(
        &self,
        a: Self::P,
        b: Self::P,
        rec: Option<&mut Vec<(usize, Self::C)>>,
        shift: usize,
    ) -> Self::P;
    /// A kernel of Euclid's algorithm with the product of the quotient matrices, `(M, g)` with
    /// `(a, b) = M (g, 0)`, for `len a > len b > 0`, when the engine has one.
    fn euclid_mat(&self, _a: &Self::P, _b: &Self::P) -> Option<(Mat<Self::P>, Self::P)> {
        None
    }
    /// Euclid's steps of [`hgcd_base`] from `(a, b)` while the degree of the divisor is at least
    /// `m` (without a record), by a kernel of the engine's own, when it has one.
    fn hgcd_steps(&self, _a: &Self::P, _b: &Self::P, _m: usize) -> Option<Steps<Self::P>> {
        None
    }
    /// Below this degree [`hgcd`] takes Euclid's steps one by one.
    fn hgcd_cutoff(&self) -> usize;
    /// Below this degree the greatest common divisor is Euclid's.
    fn gcd_cutoff(&self) -> usize;
    /// Below this degree the extended greatest common divisor is Euclid's.
    fn xgcd_cutoff(&self) -> usize;
}

/// What Euclid's steps of [`hgcd_base`] give: the product of their quotient matrices and the two
/// remainders.
pub(crate) type Steps<P> = (Mat<P>, P, P);

/// A product of quotient matrices `[[a, b], [c, d]]`, with its determinant, `-1` when `neg`.
pub(crate) struct Mat<P> {
    pub(crate) a: P,
    pub(crate) b: P,
    pub(crate) c: P,
    pub(crate) d: P,
    pub(crate) neg: bool,
}

impl<P: Clone> Mat<P> {
    /// The identity.
    fn identity<E: Engine<P = P>>(e: &E) -> Mat<P> {
        Mat {
            a: e.one(),
            b: e.zero(),
            c: e.zero(),
            d: e.one(),
            neg: false,
        }
    }

    /// `self · [[q, 1], [1, 0]]`.
    fn mul_q<E: Engine<P = P>>(&mut self, e: &E, q: &P) {
        let a = e.add(&e.mul(&self.a, q), &self.b);
        let c = e.add(&e.mul(&self.c, q), &self.d);
        self.b = core::mem::replace(&mut self.a, a);
        self.d = core::mem::replace(&mut self.c, c);
        self.neg = !self.neg;
    }

    /// `self · other`.
    fn mul<E: Engine<P = P>>(&self, e: &E, o: &Mat<P>) -> Mat<P> {
        Mat {
            a: e.add(&e.mul(&self.a, &o.a), &e.mul(&self.b, &o.c)),
            b: e.add(&e.mul(&self.a, &o.b), &e.mul(&self.b, &o.d)),
            c: e.add(&e.mul(&self.c, &o.a), &e.mul(&self.d, &o.c)),
            d: e.add(&e.mul(&self.c, &o.b), &e.mul(&self.d, &o.d)),
            neg: self.neg != o.neg,
        }
    }

    /// `self⁻¹ (x, y)`: `det·(d x - b y, a y - c x)`.
    fn apply_inv<E: Engine<P = P>>(&self, e: &E, x: &P, y: &P) -> (P, P) {
        let u = e.sub(&e.mul(&self.d, x), &e.mul(&self.b, y));
        let v = e.sub(&e.mul(&self.a, y), &e.mul(&self.c, x));
        if self.neg {
            (e.neg(&u), e.neg(&v))
        } else {
            (u, v)
        }
    }
}

/// The record of the divisors of a remainder sequence: their degrees and leading coefficients.
type Rec<C> = Vec<(usize, C)>;

/// Euclid's steps from `(a, b)` while the degree of the divisor is at least `m`: the matrix and
/// the remainders.
fn hgcd_base<E: Engine>(
    e: &E,
    a: &E::P,
    b: &E::P,
    m: usize,
    mut rec: Option<&mut Rec<E::C>>,
    shift: usize,
) -> (Mat<E::P>, E::P, E::P) {
    if rec.is_none()
        && let Some(r) = e.hgcd_steps(a, b, m)
    {
        return r;
    }
    let mut mat = Mat::identity(e);
    let (mut x, mut y) = (a.clone(), b.clone());
    while e.len(&y) > m {
        if let Some(rec) = rec.as_deref_mut() {
            rec.push((e.len(&y) - 1 + shift, e.lead(&y)));
        }
        let (q, r) = e.divrem(&x, &y);
        mat.mul_q(e, &q);
        x = y;
        y = r;
    }
    (mat, x, y)
}

/// The half greatest common divisor of `a` and `b`, `deg a > deg b`: the matrix (when `want`)
/// and the remainders `(c, d)`, `deg c ≥ ⌈deg a / 2⌉ > deg d`; with `rec`, the divisors of the
/// steps, their degrees plus `shift`.
pub(crate) fn hgcd<E: Engine>(
    e: &E,
    a: &E::P,
    b: &E::P,
    mut rec: Option<&mut Rec<E::C>>,
    shift: usize,
    want: bool,
) -> (Option<Mat<E::P>>, E::P, E::P) {
    let la = e.len(a);
    debug_assert!(la > e.len(b));
    let n = la - 1;
    let m = n.div_ceil(2);
    if e.len(b) <= m {
        return (want.then(|| Mat::identity(e)), a.clone(), b.clone());
    }
    if n < e.hgcd_cutoff() {
        let (mat, x, y) = hgcd_base(e, a, b, m, rec, shift);
        return (want.then_some(mat), x, y);
    }
    // The first half from the top halves.
    let (a1, b1) = (e.shr(a, m), e.shr(b, m));
    let (r, c1, d1) = hgcd(e, &a1, &b1, rec.as_deref_mut(), shift + m, true);
    let mut r = r.expect("asked for");
    let (u, v) = r.apply_inv(e, &e.low(a, m), &e.low(b, m));
    let x = e.shl_add(&c1, m, &u);
    let y = e.shl_add(&d1, m, &v);
    if e.len(&y) <= m {
        return (want.then_some(r), x, y);
    }
    // One step.
    if let Some(rec) = rec.as_deref_mut() {
        rec.push((e.len(&y) - 1 + shift, e.lead(&y)));
    }
    let (q, z) = e.divrem(&x, &y);
    r.mul_q(e, &q);
    let (x, y) = (y, z);
    if e.len(&y) <= m {
        return (want.then_some(r), x, y);
    }
    // The second half from the top parts of the new pair.
    let l = e.len(&x) - 1;
    let k = 2 * m - l;
    let (a2, b2) = (e.shr(&x, k), e.shr(&y, k));
    let (s, c2, d2) = hgcd(e, &a2, &b2, rec, shift + k, true);
    let s = s.expect("asked for");
    let (u, v) = s.apply_inv(e, &e.low(&x, k), &e.low(&y, k));
    let c = e.shl_add(&c2, k, &u);
    let d = e.shl_add(&d2, k, &v);
    (want.then(|| r.mul(e, &s)), c, d)
}

/// The last non-zero remainder of the sequence of `a` and `b`, not made monic; zero when both
/// are zero.
pub(crate) fn gcd<E: Engine>(e: &E, a: &E::P, b: &E::P) -> E::P {
    let (mut x, mut y) = if e.len(a) >= e.len(b) {
        (a.clone(), b.clone())
    } else {
        (b.clone(), a.clone())
    };
    if e.len(&y) == 0 {
        return x;
    }
    if e.len(&x) == e.len(&y) {
        let (_, r) = e.divrem(&x, &y);
        x = y;
        y = r;
    }
    loop {
        if e.len(&y) == 0 {
            return x;
        }
        if e.len(&y) <= e.gcd_cutoff() {
            return e.euclid(x, y, None, 0);
        }
        let (_, c, d) = hgcd(e, &x, &y, None, 0, false);
        if e.len(&d) == 0 {
            return c;
        }
        let (_, r) = e.divrem(&c, &d);
        x = d;
        y = r;
    }
}

/// `(g, s, t)`, `g = s·a + t·b` the last non-zero remainder of the sequence of `a` and `b`, not
/// made monic, `s` and `t` its cofactors from the sequence; `(0, 0, 0)` when both are zero.
pub(crate) fn xgcd<E: Engine>(e: &E, a: &E::P, b: &E::P) -> (E::P, E::P, E::P) {
    if e.len(a) < e.len(b) {
        let (g, t, s) = xgcd(e, b, a);
        return (g, s, t);
    }
    if e.len(b) == 0 {
        if e.len(a) == 0 {
            return (e.zero(), e.zero(), e.zero());
        }
        return (a.clone(), e.one(), e.zero());
    }
    // (a, b) = M (x, y) throughout; `None` for the identity.
    let mut mat: Option<Mat<E::P>> = None;
    let (mut x, mut y) = (a.clone(), b.clone());
    if e.len(&x) == e.len(&y) {
        let (q, r) = e.divrem(&x, &y);
        let mut m = Mat::identity(e);
        m.mul_q(e, &q);
        mat = Some(m);
        x = y;
        y = r;
    }
    let times = |m: Option<Mat<E::P>>, n: Mat<E::P>| match m {
        None => n,
        Some(m) => m.mul(e, &n),
    };
    while e.len(&y) != 0 {
        if e.len(&y) <= e.xgcd_cutoff() {
            let (m2, c) = match e.euclid_mat(&x, &y) {
                Some(r) => r,
                None => {
                    let (m2, c, _) = hgcd_base(e, &x, &y, 0, None, 0);
                    (m2, c)
                }
            };
            mat = Some(times(mat, m2));
            x = c;
            break;
        }
        let (r, c, d) = hgcd(e, &x, &y, None, 0, true);
        let mut m = times(mat, r.expect("asked for"));
        if e.len(&d) == 0 {
            mat = Some(m);
            x = c;
            break;
        }
        let (q, z) = e.divrem(&c, &d);
        m.mul_q(e, &q);
        mat = Some(m);
        x = d;
        y = z;
    }
    let mat = mat.unwrap_or_else(|| Mat::identity(e));
    // (x, 0) = M⁻¹ (a, b): x = det·(d a - b b').
    let (s, t) = if mat.neg {
        (e.neg(&mat.d), mat.b)
    } else {
        (mat.d, e.neg(&mat.b))
    };
    (x, s, t)
}

/// The resultant of `a` and `b`, both not constant, from the degrees and leading coefficients
/// of the remainders of their sequence: `res(r₀, r₁) = (-1)^(d₀d₁) l₁^(d₀ - d₂) res(r₁, r₂)`, and
/// `res(r, c) = c^deg r` for a constant `c`; 0 when the sequence ends before a constant.
pub(crate) fn resultant<E: Engine>(e: &E, a: &E::P, b: &E::P) -> E::C {
    let mut rec: Rec<E::C> = vec![(e.len(a) - 1, e.lead(a))];
    let (mut x, mut y) = (a.clone(), b.clone());
    loop {
        if e.len(&y) == 0 {
            return e.czero();
        }
        if e.len(&y) == 1 {
            rec.push((0, e.lead(&y)));
            break;
        }
        if e.len(&y) <= e.gcd_cutoff() {
            let g = e.euclid(x, y, Some(&mut rec), 0);
            if e.len(&g) != 1 {
                return e.czero();
            }
            break;
        }
        if e.len(&x) == e.len(&y) {
            // One step to make the degrees differ.
            rec.push((e.len(&y) - 1, e.lead(&y)));
            let (_, r) = e.divrem(&x, &y);
            x = y;
            y = r;
            continue;
        }
        let (_, c, d) = hgcd(e, &x, &y, Some(&mut rec), 0, false);
        if e.len(&d) == 0 {
            return e.czero();
        }
        if e.len(&d) == 1 {
            rec.push((0, e.lead(&d)));
            break;
        }
        rec.push((e.len(&d) - 1, e.lead(&d)));
        let (_, r) = e.divrem(&c, &d);
        x = d;
        y = r;
    }
    // The product over the divisors.
    let mut res = e.cone();
    let mut neg = false;
    for i in 1..rec.len() {
        let (d0, _) = rec[i - 1];
        let (d1, ref l) = rec[i];
        let d2 = rec.get(i + 1).map_or(0, |r| r.0);
        if d0 & d1 & 1 == 1 {
            neg = !neg;
        }
        res = e.cmul(&res, &e.cpow(l, (d0 - d2) as u64));
    }
    if neg { e.cneg(&res) } else { res }
}

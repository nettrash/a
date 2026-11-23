//! The fast phase of BKZ, in fplll's design: Schnorr and Euchner's tours on a basis whose rows,
//! transformation and Gram matrix are kept exactly, in integers, and whose Gram–Schmidt
//! orthogonalisation is kept in floating point across insertions and tours, a row at a time by
//! the Cholesky factorisation from the exact Gram matrix (as L² computes it, Nguyen and Stehlé,
//! SIAM J. Comput. 39, 2009); after an insertion a local L² (lazy size reduction, Lovász's test,
//! no certificate) reduces the rows of the block alone, and the orthogonalisation is computed
//! again from the first row that changed.
//!
//! Nothing here is proved: each step is an exact unimodular change of the basis, so `U·B = B'`
//! holds throughout, but the floats only steer. [`bkz`](super::bkz) then reduces the result by
//! the certified LLL and finishes with its exact tours, which give the guarantees. A run that
//! the floats cannot carry (a coefficient that is not finite, a norm that is not positive, a
//! size reduction that does not end, too many iterations or tours) stops early, and the exact
//! tours do all the work.

use super::enum_float::FloatLevels;
use super::enum_gso::{dot, scale};
use crate::math::numbers::{ByteOrder, Number};

/// `2^e` for `−1022 ≤ e ≤ 1023`.
fn pow2(e: i64) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// A float with an exponent of its own, `m·2^e` with `1/2 ≤ |m| < 1` or `m = 0`, so that the
/// orthogonalisation of rows of any size fits: 53 bits, every operation rounded once in the
/// mantissa.
#[derive(Clone, Copy, Debug)]
struct Xf {
    m: f64,
    e: i64,
}

const ZERO: Xf = Xf { m: 0.0, e: 0 };

impl Xf {
    /// `x·2^e`, normalised; a zero or a value that is not finite keeps the exponent 0.
    fn new(x: f64, e: i64) -> Xf {
        if x == 0.0 || !x.is_finite() {
            return Xf { m: x, e: 0 };
        }
        let bits = x.to_bits();
        let ex = ((bits >> 52) & 0x7ff) as i64;
        if ex == 0 {
            // Subnormal: scaled into the normal range first.
            return Xf::new(x * pow2(64), e - 64);
        }
        Xf {
            m: f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52)),
            e: e + ex - 1022,
        }
    }

    /// The nearest `Xf` to an integer of any size, within a relative `2⁻⁵²`: beyond 1000 bits
    /// from its top 16 bytes, read in place.
    fn of(n: &Number) -> Xf {
        let b = n.bits() as i64;
        if b <= 1000 {
            return Xf::new(n.to_f64(), 0);
        }
        let v = &n.value;
        let len = v.len();
        // The bytes from the most significant down, whatever the order they are kept in.
        let at = |i: usize| -> u8 {
            match n.byte_order {
                ByteOrder::BigEndian => v[i],
                ByteOrder::LittleEndian => v[len - 1 - i],
            }
        };
        let mut top: u128 = 0;
        for i in 0..16 {
            top = (top << 8) | u128::from(at(i));
        }
        let x = top as f64;
        Xf::new(if n.is_negative() { -x } else { x }, 8 * (len as i64 - 16))
    }

    fn finite(self) -> bool {
        self.m.is_finite()
    }

    fn mul(self, o: Xf) -> Xf {
        Xf::new(self.m * o.m, self.e + o.e)
    }

    fn div(self, o: Xf) -> Xf {
        Xf::new(self.m / o.m, self.e - o.e)
    }

    fn add(self, o: Xf) -> Xf {
        if self.m == 0.0 {
            return o;
        }
        if o.m == 0.0 {
            return self;
        }
        let d = self.e - o.e;
        if d > 64 {
            self
        } else if d < -64 {
            o
        } else if d >= 0 {
            Xf::new(self.m + o.m * pow2(-d), self.e)
        } else {
            Xf::new(self.m * pow2(d) + o.m, o.e)
        }
    }

    fn sub(self, o: Xf) -> Xf {
        self.add(Xf { m: -o.m, e: o.e })
    }

    /// As an `f64`: `±∞` beyond its range, towards zero below it.
    fn to_f64(self) -> f64 {
        if self.m == 0.0 || !self.m.is_finite() {
            return self.m;
        }
        scale(self.m, self.e)
    }

    /// The nearest integer.
    fn rint(self) -> Int {
        if self.e <= 53 {
            Int::S(self.to_f64().round() as i64)
        } else {
            // The 53 bits of the mantissa as an integer, shifted into place: exact.
            Int::B(Number::from((self.m * pow2(53)) as i64) << (self.e - 53) as usize)
        }
    }
}

/// The floating-point numbers of the orthogonalisation: [`Xf`], with an exponent of its own,
/// for bases of any size, and a plain `f64` holding the squared norms and products scaled by
/// one power of two, `2⁻ˢ`, fixed for the run (the coefficients `μ` are not scaled), which is
/// several times faster where the scaled values fit.
trait Fl: Copy {
    const ZERO: Self;
    /// An integer of the Gram matrix (`scale` is `s`) or a coefficient (`scale` is 0).
    fn of_int(x: &Int, scale: i64) -> Self;
    fn of_f64(x: f64) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    /// The value as kept, as an `f64`: ratios of squared norms and coefficients are right.
    fn value(self) -> f64;
    /// The nearest integer to a coefficient.
    fn nearest(self) -> Int;
    fn is_finite(self) -> bool;
    fn is_positive(self) -> bool;
}

impl Fl for Xf {
    const ZERO: Xf = ZERO;
    fn of_int(x: &Int, _: i64) -> Xf {
        x.xf()
    }
    fn of_f64(x: f64) -> Xf {
        Xf::new(x, 0)
    }
    fn mul(self, o: Xf) -> Xf {
        Xf::mul(self, o)
    }
    fn div(self, o: Xf) -> Xf {
        Xf::div(self, o)
    }
    fn add(self, o: Xf) -> Xf {
        Xf::add(self, o)
    }
    fn sub(self, o: Xf) -> Xf {
        Xf::sub(self, o)
    }
    fn value(self) -> f64 {
        self.to_f64()
    }
    fn nearest(self) -> Int {
        self.rint()
    }
    fn is_finite(self) -> bool {
        self.finite()
    }
    fn is_positive(self) -> bool {
        self.m > 0.0
    }
}

impl Fl for f64 {
    const ZERO: f64 = 0.0;
    fn of_int(x: &Int, s: i64) -> f64 {
        let v = x.xf();
        if v.m == 0.0 { 0.0 } else { scale(v.m, v.e - s) }
    }
    fn of_f64(x: f64) -> f64 {
        x
    }
    fn mul(self, o: f64) -> f64 {
        self * o
    }
    fn div(self, o: f64) -> f64 {
        self / o
    }
    fn add(self, o: f64) -> f64 {
        self + o
    }
    fn sub(self, o: f64) -> f64 {
        self - o
    }
    fn value(self) -> f64 {
        self
    }
    fn nearest(self) -> Int {
        if self.abs() < 4.0e18 {
            Int::S(self.round() as i64)
        } else {
            Xf::new(self, 0).rint()
        }
    }
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    fn is_positive(self) -> bool {
        self > 0.0
    }
}

/// An integer in a machine word while it fits, a [`Number`] beyond: fplll's choice of `long`
/// for the entries that allow it, here per entry. The rows of a reduced basis, its
/// transformation and most of its Gram matrix are small, and a `Number` costs an allocation
/// for every result.
#[derive(Clone, Debug)]
enum Int {
    S(i64),
    B(Number),
}

impl Default for Int {
    fn default() -> Int {
        Int::S(0)
    }
}

impl Int {
    fn of(n: &Number) -> Int {
        match i64::try_from(n) {
            Ok(v) => Int::S(v),
            Err(_) => Int::B(n.clone()),
        }
    }

    fn of_number(n: Number) -> Int {
        match i64::try_from(&n) {
            Ok(v) => Int::S(v),
            Err(_) => Int::B(n),
        }
    }

    fn of_i128(v: i128) -> Int {
        match i64::try_from(v) {
            Ok(v) => Int::S(v),
            Err(_) => Int::B(Number::from(v)),
        }
    }

    fn number(&self) -> Number {
        match self {
            Int::S(v) => Number::from(*v),
            Int::B(n) => n.clone(),
        }
    }

    fn is_zero(&self) -> bool {
        match self {
            Int::S(v) => *v == 0,
            Int::B(n) => n.is_zero(),
        }
    }

    fn xf(&self) -> Xf {
        match self {
            Int::S(v) => Xf::new(*v as f64, 0),
            Int::B(n) => Xf::of(n),
        }
    }

    fn neg(&self) -> Int {
        match self {
            Int::S(v) => Int::of_i128(-(*v as i128)),
            Int::B(n) => Int::of_number(-n),
        }
    }

    fn mul(&self, o: &Int) -> Int {
        match (self, o) {
            (Int::S(a), Int::S(b)) => Int::of_i128(*a as i128 * *b as i128),
            (Int::B(a), Int::S(b)) | (Int::S(b), Int::B(a)) => Int::of_number(a * *b),
            (Int::B(a), Int::B(b)) => Int::of_number(a * b),
        }
    }

    fn add(&self, o: &Int) -> Int {
        match (self, o) {
            (Int::S(a), Int::S(b)) => Int::of_i128(*a as i128 + *b as i128),
            (Int::B(a), Int::S(b)) | (Int::S(b), Int::B(a)) => Int::of_number(a + *b),
            (Int::B(a), Int::B(b)) => Int::of_number(a + b),
        }
    }

    fn sub(&self, o: &Int) -> Int {
        self.add(&o.neg())
    }

    /// `self ← self − p`.
    fn sub_assign(&mut self, p: &Int) {
        match (&mut *self, p) {
            (Int::S(a), Int::S(b)) => *self = Int::of_i128(*a as i128 - *b as i128),
            (Int::B(a), Int::S(b)) => {
                *a -= *b;
                if a.bits() < 63 {
                    *self = Int::of(a);
                }
            }
            (Int::B(a), Int::B(b)) => {
                *a -= b;
                if a.bits() < 63 {
                    *self = Int::of(a);
                }
            }
            (Int::S(a), Int::B(b)) => *self = Int::of_number(&Number::from(*a) - b),
        }
    }

    /// `self ← self − x·y`.
    fn sub_mul(&mut self, x: &Int, y: &Int) {
        if let (Int::S(s), Int::S(a), Int::S(b)) = (&*self, x, y) {
            // |s − a·b| < 2^63 + 2^126: no overflow in an i128.
            *self = Int::of_i128(*s as i128 - *a as i128 * *b as i128);
            return;
        }
        self.sub_assign(&x.mul(y));
    }
}

/// `v[k] ← v[k] − x·v[i]` for rows `i ≠ k`.
fn sub_mul(v: &mut [Vec<Int>], k: usize, i: usize, x: &Int) {
    let (dst, src) = if i < k {
        let (lo, hi) = v.split_at_mut(k);
        (&mut hi[0], &lo[i])
    } else {
        let (lo, hi) = v.split_at_mut(i);
        (&mut lo[k], &hi[0])
    };
    for (a, b) in dst.iter_mut().zip(src.iter()) {
        if !b.is_zero() {
            a.sub_mul(x, b);
        }
    }
}

/// How the fast phase ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    /// A tour of the near-strict kind inserted nothing.
    Converged,
    /// The limit of tours was reached.
    Stopped,
    /// The floats could not carry the basis, or the tours did not settle; the basis is still a
    /// basis of the lattice, with its transformation.
    Unsettled,
}

/// The state of the fast phase: the rows and the transformation (exact), the Gram matrix
/// (exact, its lower triangle), and the floating-point orthogonalisation of the rows before
/// `valid`.
struct Fast<F> {
    b: Vec<Vec<Int>>,
    u: Vec<Vec<Int>>,
    /// The lower triangle of the Gram matrix: `g[i][j] = ⟨bᵢ, bⱼ⟩` for `j ≤ i`.
    g: Vec<Vec<Int>>,
    /// `mu[i][j] ≈ μᵢⱼ` for `j < i`.
    mu: Vec<Vec<F>>,
    /// `rr[i][j] ≈ ⟨bᵢ, b*ⱼ⟩` for `j < i`, and `rr[i][i] ≈ |b*ᵢ|²`.
    rr: Vec<Vec<F>>,
    valid: usize,
    /// `s`: the Gram matrix and the squared norms are kept times `2⁻ˢ`.
    scale: i64,
    delta: f64,
    eta: f64,
}

impl<F: Fl> Fast<F> {
    /// The fast phase on the independent rows `b` with the transformation `u`, testing Lovász's
    /// condition with `delta` and size-reducing to `max(eta, 0.51)`: room above 1/2 for the
    /// rounding of the floats, as L² needs; the certified reduction afterwards meets `η`.
    fn new(b: Vec<Vec<Number>>, u: Vec<Vec<Number>>, delta: f64, eta: f64) -> Fast<F> {
        let n = b.len();
        let g = (0..n)
            .map(|i| (0..=i).map(|j| Int::of_number(dot(&b[i], &b[j]))).collect())
            .collect();
        let ints = |v: Vec<Vec<Number>>| -> Vec<Vec<Int>> {
            v.iter().map(|r| r.iter().map(Int::of).collect()).collect()
        };
        Fast {
            b: ints(b),
            u: ints(u),
            g,
            mu: vec![vec![F::ZERO; n]; n],
            rr: vec![vec![F::ZERO; n]; n],
            valid: 0,
            scale: 0,
            delta,
            eta: eta.max(0.51),
        }
    }

    /// The same exact state with another float type and scale; the orthogonalisation is
    /// computed again.
    fn retype<G: Fl>(self, scale: i64) -> Fast<G> {
        let n = self.b.len();
        Fast {
            b: self.b,
            u: self.u,
            g: self.g,
            mu: vec![vec![G::ZERO; n]; n],
            rr: vec![vec![G::ZERO; n]; n],
            valid: 0,
            scale,
            delta: self.delta,
            eta: self.eta,
        }
    }

    /// The scale for plain floats, if every squared norm of the rows fits with room: the
    /// largest `|bᵢ|²` brought to about `2⁴⁰⁰`, and the smallest `|b*ᵢ|²`, at least
    /// `(δ − η²)^n` of the first and not below `|bᵢ|²/2^(2·bits)`, kept above `2⁻⁹⁰⁰`.
    fn plain_scale(&self) -> Option<i64> {
        let bits = |x: &Int| -> i64 {
            match x {
                Int::S(v) => 64 - v.unsigned_abs().leading_zeros() as i64,
                Int::B(n) => n.bits() as i64,
            }
        };
        let n = self.g.len();
        let hi = (0..n).map(|i| bits(&self.g[i][i])).max()?;
        let lo = (0..n).map(|i| bits(&self.g[i][i])).min()?;
        // An LLL-reduced basis loses at most about 0.5 bit of |b*ᵢ|² a row against |bᵢ|².
        let span = hi - lo + n as i64 + 64;
        (span < 1200).then_some(hi - 400)
    }

    /// The rows and the transformation.
    fn into_parts(self) -> (Vec<Vec<Number>>, Vec<Vec<Number>>) {
        let numbers = |v: Vec<Vec<Int>>| -> Vec<Vec<Number>> {
            v.iter()
                .map(|r| r.iter().map(Int::number).collect())
                .collect()
        };
        (numbers(self.b), numbers(self.u))
    }

    /// `⟨bᵢ, bⱼ⟩`.
    fn gram(&self, i: usize, j: usize) -> &Int {
        if j <= i { &self.g[i][j] } else { &self.g[j][i] }
    }

    fn gram_mut(&mut self, i: usize, j: usize) -> &mut Int {
        if j <= i {
            &mut self.g[i][j]
        } else {
            &mut self.g[j][i]
        }
    }

    /// The row `i` of the orthogonalisation from the exact Gram matrix and the rows before it;
    /// `false` if a coefficient is not finite. `|b*ᵢ|²` may come out below zero by cancellation
    /// while the row is not size-reduced; the Lovász test checks it after.
    fn gso_row(&mut self, i: usize) -> bool {
        for j in 0..i {
            let mut t = F::of_int(&self.g[i][j], self.scale);
            for l in 0..j {
                t = t.sub(self.mu[j][l].mul(self.rr[i][l]));
            }
            let m = t.div(self.rr[j][j]);
            if !m.is_finite() {
                return false;
            }
            self.rr[i][j] = t;
            self.mu[i][j] = m;
        }
        let mut t = F::of_int(&self.g[i][i], self.scale);
        for l in 0..i {
            t = t.sub(self.mu[i][l].mul(self.rr[i][l]));
        }
        self.rr[i][i] = t;
        t.is_finite()
    }

    /// The orthogonalisation of the rows up to `upto`.
    fn refresh(&mut self, upto: usize) -> bool {
        while self.valid < upto {
            let i = self.valid;
            if !self.gso_row(i) || !self.rr[i][i].is_positive() {
                return false;
            }
            self.valid += 1;
        }
        true
    }

    /// `bₖ ← bₖ − x·bᵢ`, with the transformation and the Gram matrix.
    fn sub_row(&mut self, k: usize, i: usize, x: &Int) {
        sub_mul(&mut self.b, k, i, x);
        sub_mul(&mut self.u, k, i, x);
        // |bₖ − x·bᵢ|² = Gₖₖ − x·(2·Gₖᵢ − x·Gᵢᵢ), with the old Gₖᵢ.
        let gki = self.gram(k, i).clone();
        let t = gki.add(&gki).sub(&x.mul(self.gram(i, i)));
        for j in 0..self.g.len() {
            if j == k {
                continue;
            }
            let p = x.mul(self.gram(i, j));
            if !p.is_zero() {
                self.gram_mut(k, j).sub_assign(&p);
            }
        }
        self.g[k][k].sub_mul(x, &t);
        self.valid = self.valid.min(k);
    }

    /// `(rᵢ, rⱼ) ← (p·rᵢ + q·rⱼ, s·rᵢ + t·rⱼ)` for `i ≠ j`, with the transformation and the Gram
    /// matrix.
    fn combine(&mut self, i: usize, j: usize, c: [&Number; 4]) {
        let [p, q, s, t] = c.map(Int::of);
        for v in [&mut self.b, &mut self.u] {
            let a = core::mem::take(&mut v[i]);
            let b = core::mem::take(&mut v[j]);
            v[i] = a
                .iter()
                .zip(&b)
                .map(|(x, y)| p.mul(x).add(&q.mul(y)))
                .collect();
            v[j] = a
                .iter()
                .zip(&b)
                .map(|(x, y)| s.mul(x).add(&t.mul(y)))
                .collect();
        }
        let (aii, ajj, aij) = (
            self.gram(i, i).clone(),
            self.gram(j, j).clone(),
            self.gram(i, j).clone(),
        );
        for l in 0..self.g.len() {
            if l == i || l == j {
                continue;
            }
            let (gi, gj) = (self.gram(i, l).clone(), self.gram(j, l).clone());
            *self.gram_mut(i, l) = p.mul(&gi).add(&q.mul(&gj));
            *self.gram_mut(j, l) = s.mul(&gi).add(&t.mul(&gj));
        }
        let quad = |x: &Int, y: &Int| -> Int {
            let xy = x.mul(y).mul(&aij);
            x.mul(x)
                .mul(&aii)
                .add(&xy)
                .add(&xy)
                .add(&y.mul(y).mul(&ajj))
        };
        let gii = quad(&p, &q);
        let gjj = quad(&s, &t);
        let gij = p
            .mul(&s)
            .mul(&aii)
            .add(&p.mul(&t).add(&q.mul(&s)).mul(&aij))
            .add(&q.mul(&t).mul(&ajj));
        *self.gram_mut(i, i) = gii;
        *self.gram_mut(j, j) = gjj;
        *self.gram_mut(i, j) = gij;
        self.valid = self.valid.min(i.min(j));
    }

    /// Moves the row `from` to the place `to`, the rows between shifting by one, with the lower
    /// triangle of the Gram matrix: the entries of the rows `lo..=hi` taken out once each and
    /// put where they now belong, the columns `lo..=hi` of the rows after them rotated.
    fn move_row(&mut self, from: usize, to: usize) {
        if from == to {
            return;
        }
        for v in [&mut self.b, &mut self.u] {
            let r = v.remove(from);
            v.insert(to, r);
        }
        let (lo, hi) = (from.min(to), from.max(to));
        // The old index of the row now at `p`.
        let old = |p: usize| -> usize {
            if p < lo || p > hi {
                p
            } else if p == to {
                from
            } else if from > to {
                p - 1
            } else {
                p + 1
            }
        };
        let mut window: Vec<Vec<Int>> =
            (lo..=hi).map(|i| core::mem::take(&mut self.g[i])).collect();
        #[allow(
            clippy::needless_range_loop,
            reason = "`p` also picks the old row and the length of the new one"
        )]
        for p in lo..=hi {
            let a = old(p);
            self.g[p] = (0..=p)
                .map(|q| {
                    let c = old(q);
                    let (x, y) = if c <= a { (a, c) } else { (c, a) };
                    core::mem::take(&mut window[x - lo][y])
                })
                .collect();
        }
        for row in self.g.iter_mut().skip(hi + 1) {
            if from > to {
                row[lo..=hi].rotate_right(1);
            } else {
                row[lo..=hi].rotate_left(1);
            }
        }
        self.valid = self.valid.min(lo);
    }

    /// The lazy size reduction of L² (Nguyen and Stehlé, Figure 5): until every
    /// `|μₖⱼ| ≤ η`, the coefficients rounded from the top down, each correcting those below it,
    /// and the row reduced exactly. The rows before `k` must be orthogonalised; the row `k` is
    /// after it.
    fn size_reduce(&mut self, k: usize) -> bool {
        // Each pass takes about 50 bits off the coefficients, as L² allows for.
        let bits = match &self.g[k][k] {
            Int::S(_) => 64,
            Int::B(n) => n.bits(),
        };
        for _ in 0..64 + 2 * bits {
            if !self.gso_row(k) {
                return false;
            }
            if (0..k).all(|j| self.mu[k][j].value().abs() <= self.eta) {
                return true;
            }
            let mut xs: Vec<(usize, Int)> = Vec::new();
            for j in (0..k).rev() {
                let x = self.mu[k][j].nearest();
                if x.is_zero() {
                    continue;
                }
                let xf = F::of_int(&x, 0);
                for l in 0..j {
                    self.mu[k][l] = self.mu[k][l].sub(xf.mul(self.mu[j][l]));
                }
                xs.push((j, x));
            }
            if xs.is_empty() {
                return true;
            }
            for (j, x) in xs {
                self.sub_row(k, j, &x);
            }
        }
        false
    }

    /// L² on the rows `start..end`, the rows before `start` taken as reduced and no row moved
    /// below `kmin`: each row size-reduced, then moved down past every place where Lovász's
    /// condition fails.
    fn lll(&mut self, kmin: usize, start: usize, end: usize) -> bool {
        if !self.refresh(start) {
            return false;
        }
        let cap = 1000 + 100 * (end * end) as u64;
        let mut iterations = 0u64;
        let mut k = start;
        while k < end {
            iterations += 1;
            if iterations > cap {
                return false;
            }
            if k == 0 {
                if !self.gso_row(0) {
                    return false;
                }
                self.valid = self.valid.max(1);
                k = 1;
                continue;
            }
            // A row whose orthogonalisation is current and already size-reduced is left as it is.
            let current = k < self.valid && (0..k).all(|j| self.mu[k][j].value().abs() <= self.eta);
            if !current && !self.size_reduce(k) {
                return false;
            }
            // s = |π_place(bₖ)|², growing as the place goes down. At `k` it may be wrong by
            // cancellation, below zero even, where the row is far shorter projected than whole;
            // Lovász's test fails there all the same, and the row moves down to where it is
            // right: there it must be positive (at 0 it is the exact Gₖₖ).
            let mut s = self.rr[k][k];
            let mut place = k;
            while place > kmin {
                let i = place - 1;
                let si = s.add(self.mu[k][i].mul(self.rr[k][i]));
                if si.sub(F::of_f64(self.delta).mul(self.rr[i][i])).value() < 0.0 {
                    s = si;
                    place = i;
                } else {
                    break;
                }
            }
            if !s.is_positive() || !s.is_finite() {
                return false;
            }
            if place == k {
                self.valid = self.valid.max(k + 1);
                k += 1;
            } else {
                self.move_row(k, place);
                k = place;
            }
        }
        true
    }

    /// A vector of the block `[k, end)` whose projection is shorter than `factor·|b*ₖ|²` by
    /// the floats, the shortest the float enumeration finds, as coordinates on the block's rows.
    fn block_svp(&self, k: usize, end: usize, factor: f64) -> Option<Vec<i64>> {
        let nb = end - k;
        let rk = self.rr[k][k];
        let r: Vec<f64> = (k..end)
            .map(|l| {
                let v = self.rr[l][l].div(rk).value();
                if v.is_finite() {
                    v.clamp(1.0e-300, 1.0e300)
                } else {
                    1.0e300
                }
            })
            .collect();
        let mut mu = vec![0.0f64; nb * nb];
        for j in 0..nb {
            for l in 0..j {
                let v = self.mu[k + j][k + l].value();
                if !v.is_finite() {
                    return None;
                }
                mu[j * nb + l] = v;
            }
        }
        let mmax = FloatLevels::column_maxima(nb, &mu);
        let fl = FloatLevels {
            n: nb,
            mu,
            r,
            t: vec![0.0; nb],
            mmax,
            symmetric: true,
        };
        let mut best: Option<Vec<i64>> = None;
        let mut radius = factor;
        let out = fl.run(factor, &mut |x: &[f64]| {
            // The squared norm of the projection, by the floats.
            let mut norm = 0.0;
            for l in 0..nb {
                let c: f64 = (l + 1..nb).map(|j| x[j] * fl.mu[j * nb + l]).sum();
                let d = x[l] + c;
                norm += d * d * fl.r[l];
            }
            if norm < radius {
                radius = norm;
                best = Some(x.iter().map(|&v| v as i64).collect());
                return Some(norm);
            }
            None
        });
        if out.is_err() {
            return None;
        }
        best
    }

    /// `bᵣ ← −bᵣ`, with the transformation and the Gram matrix.
    fn negate(&mut self, r: usize) {
        for v in [&mut self.b, &mut self.u] {
            for x in v[r].iter_mut() {
                *x = x.neg();
            }
        }
        for j in 0..self.g.len() {
            if j != r {
                *self.gram_mut(r, j) = self.gram(r, j).neg();
            }
        }
        self.valid = self.valid.min(r);
    }

    /// Makes `Σ xᵢ·b_{k+i}` the row `k`, with the Gram matrix. With a coordinate `±1` (the
    /// common case), the vector replaces that row and moves to `k`, the other rows unchanged;
    /// otherwise by the unimodular gcd steps of [`insert`](super::bkz).
    fn insert(&mut self, k: usize, x: &[i64]) {
        if let Some(i) = x.iter().rposition(|v| v.abs() == 1) {
            let r = k + i;
            if x[i] < 0 {
                self.negate(r);
            }
            for (j, &xj) in x.iter().enumerate() {
                if j != i && xj != 0 {
                    self.sub_row(r, k + j, &Int::S(-xj));
                }
            }
            self.move_row(r, k);
            return;
        }
        let mut y: Vec<Number> = x.iter().map(|&v| Number::from(v)).collect();
        for j in (1..y.len()).rev() {
            if y[j].is_zero() {
                continue;
            }
            if y[j - 1].is_zero() {
                y.swap(j - 1, j);
                self.move_row(k + j, k + j - 1);
                continue;
            }
            let (g, a, b) = y[j - 1].gcd_ext(&y[j]);
            let p = &y[j - 1] / &g;
            let q = &y[j] / &g;
            let nb = -&b;
            self.combine(k + j - 1, k + j, [&p, &q, &nb, &a]);
            y[j - 1] = g;
            y[j] = Number::default();
        }
        self.valid = self.valid.min(k);
    }

    /// The LLL reduction of every row, then tours with blocks of `beta`, counted in `tours`
    /// against `max_tours`: first inserting a
    /// vector only below `δ·|b*ₖ|²`, then, once a tour inserts nothing, below `(1 − 10⁻⁹)·|b*ₖ|²`,
    /// until a tour of that kind inserts nothing. At most `4n + 64` tours for this start.
    fn tours(&mut self, beta: usize, max_tours: Option<usize>, tours: &mut usize) -> Outcome {
        let n = self.b.len();
        if !self.lll(0, 0, n) {
            return Outcome::Unsettled;
        }
        let mut strict = false;
        for _ in 0..4 * n + 64 {
            if max_tours.is_some_and(|t| *tours >= t) {
                return Outcome::Stopped;
            }
            *tours += 1;
            let factor = if strict { 1.0 - 1.0e-9 } else { self.delta };
            let mut inserted = false;

            for k in 0..n - 1 {
                let end = (k + beta).min(n);
                if !self.lll(0, k, end) || !self.refresh(end) {
                    return Outcome::Unsettled;
                }
                if let Some(x) = self.block_svp(k, end, factor) {
                    self.insert(k, &x);
                    if !self.lll(k, k, end) {
                        return Outcome::Unsettled;
                    }
                    inserted = true;
                }
            }
            if !inserted {
                if strict {
                    return Outcome::Converged;
                }
                strict = true;
            }
        }
        Outcome::Unsettled
    }
}

/// The fast phase on the independent rows `b` with the transformation `u`: the LLL reduction
/// of the rows and the tours (counted in `tours` against `max_tours`), in plain scaled floats
/// where the reduced basis allows them and in [`Xf`] otherwise or if they fail. Returns the rows,
/// the transformation and how the tours ended.
pub(super) fn run(
    b: Vec<Vec<Number>>,
    u: Vec<Vec<Number>>,
    (delta, eta): (f64, f64),
    beta: usize,
    max_tours: Option<usize>,
    tours: &mut usize,
) -> (Vec<Vec<Number>>, Vec<Vec<Number>>, Outcome) {
    let mut wide = Fast::<Xf>::new(b, u, delta, eta);
    let n = wide.b.len();
    if !wide.lll(0, 0, n) {
        let (b, u) = wide.into_parts();
        return (b, u, Outcome::Unsettled);
    }
    if let Some(s) = wide.plain_scale() {
        let mut plain = wide.retype::<f64>(s);
        let outcome = plain.tours(beta, max_tours, tours);
        if outcome != Outcome::Unsettled {
            let (b, u) = plain.into_parts();
            return (b, u, outcome);
        }
        wide = plain.retype::<Xf>(0);
    }
    let outcome = wide.tours(beta, max_tours, tours);
    let (b, u) = wide.into_parts();
    (b, u, outcome)
}

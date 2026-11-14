//! Hypergeometric series by binary splitting, in the form of y-cruncher's formula files:
//!
//! `S = (c_Q + c_P Σ_(k ≥ 1) t_k) / c_D`, `t_k = P(k)/Q(k) Π_(j < k) R(j)/Q(j)`,
//!
//! for integer polynomials `P`, `Q`, `R` with `Q(k) ≠ 0` for `k ≥ 1`. Over a range `[a, b)` of
//! terms, `T(a, b) = Σ_(a ≤ k < b) P(k) Π_(a ≤ j < k) R(j) Π_(k < j < b) Q(j)`,
//! `Q(a, b) = Π Q(k)` and `R(a, b) = Π R(k)` make `Σ_(a ≤ k < b) t_k / Π_(j < a) R(j)/Q(j)` equal to
//! `T/Q`, and a range split at `m` gives `T = T1 Q2 + R1 T2`, `Q = Q1 Q2`, `R = R1 R2`, with `R` not
//! needed on the right spine of the tree. The leaves are blocks of terms evaluated from the right
//! in limbs, `T ← P(k) Q_s + R(k) T`, `Q_s ← Q(k) Q_s`, `R_s ← R(k) R_s`.
//!
//! A series whose polynomials are positive at every `k ≥ 1` may be split with its numbers cut to a
//! working number of bits once they outgrow it ([`Cut`]): only the top bits of the tree's large
//! numbers count for the quotient `T/Q`.

use super::super::limbs;
use super::num::{Cut, Num};
use crate::math::numbers::nat;

/// The value of a polynomial at an integer: a signed magnitude of up to four limbs.
#[derive(Clone, Copy, Debug)]
pub(super) struct Val {
    /// The sign.
    neg: bool,
    /// The magnitude, the least significant limb first.
    m: [u64; 4],
    /// The limbs of the magnitude without the zero ones at the top.
    n: usize,
}

impl Val {
    /// The value `v`.
    fn of(v: i128) -> Val {
        let a = v.unsigned_abs();
        let mut r = Val {
            neg: v < 0,
            m: [a as u64, (a >> 64) as u64, 0, 0],
            n: 2,
        };
        r.trim();
        r
    }

    #[inline]
    fn trim(&mut self) {
        while self.n > 0 && self.m[self.n - 1] == 0 {
            self.n -= 1;
        }
        if self.n == 0 {
            self.neg = false;
        }
    }

    /// The limbs of the magnitude.
    #[inline]
    fn limbs(&self) -> &[u64] {
        &self.m[..self.n]
    }

    /// `self·k`.
    fn mul(&mut self, k: u64) {
        let mut carry = 0u128;
        for x in &mut self.m {
            let t = u128::from(*x) * u128::from(k) + carry;
            *x = t as u64;
            carry = t >> 64;
        }
        assert!(carry == 0, "a polynomial value beyond 256 bits");
        self.n = 4;
        self.trim();
    }

    /// `self + c`.
    fn add(&mut self, c: i128) {
        let c = Val::of(c);
        if c.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = c;
            return;
        }
        if self.neg == c.neg {
            let mut carry = false;
            for i in 0..4 {
                let (s, c1) = self.m[i].overflowing_add(c.m[i]);
                let (s, c2) = s.overflowing_add(u64::from(carry));
                self.m[i] = s;
                carry = c1 || c2;
            }
            assert!(!carry, "a polynomial value beyond 256 bits");
        } else {
            let (big, small, neg) = if self.m.iter().rev().cmp(c.m.iter().rev()).is_ge() {
                (self.m, c.m, self.neg)
            } else {
                (c.m, self.m, c.neg)
            };
            let mut borrow = false;
            for i in 0..4 {
                let (s, b1) = big[i].overflowing_sub(small[i]);
                let (s, b2) = s.overflowing_sub(u64::from(borrow));
                self.m[i] = s;
                borrow = b1 || b2;
            }
            self.neg = neg;
        }
        self.n = 4;
        self.trim();
    }
}

/// An integer polynomial by its coefficients, the constant first.
#[derive(Clone, Debug)]
pub(super) struct Poly(pub(super) Vec<i128>);

impl Poly {
    /// The value at `k`, by Horner's rule.
    pub(super) fn at(&self, k: u64) -> Val {
        let c = &self.0;
        let mut v = Val::of(c[c.len() - 1]);
        for &ci in c[..c.len() - 1].iter().rev() {
            v.mul(k);
            v.add(ci);
        }
        v
    }

    /// The value at `k` in floating point and a bound of its relative error: Horner's rule in
    /// `f64` is within `γ_(2d) Σ |c_i| k^i` of the value for the degree `d` (Higham, Accuracy and
    /// Stability of Numerical Algorithms, 5.1), `γ_m = m u / (1 - m u)`, `u = 2^-53`, and the
    /// coefficients' own rounding adds `u Σ |c_i| k^i`: `(2d + 2)·2^-52 Σ |c_i| k^i / |v|` bounds
    /// both while `k < 2^53`.
    fn at_f64(&self, k: u64) -> (f64, f64) {
        let x = k as f64;
        let v = self.0.iter().rev().fold(0.0, |v, &c| v * x + c as f64);
        let s = self
            .0
            .iter()
            .rev()
            .fold(0.0, |v, &c| v * x + (c as f64).abs());
        let d = (self.0.len() - 1) as f64;
        (v.abs(), (2.0 * d + 2.0) * f64::EPSILON * s / v.abs())
    }

    /// Whether the polynomial is the constant 1.
    fn is_one(&self) -> bool {
        self.0 == [1]
    }
}

/// A series in the form of the [module](self).
#[derive(Clone, Debug)]
pub(super) struct Series {
    pub(super) p: Poly,
    pub(super) q: Poly,
    pub(super) r: Poly,
    /// A bound, below 1, of `|t_(k+1) / t_k|` for every `k ≥ 1`, proved for each series: the tail
    /// after `N` terms is at most `|t_(N+1)| / (1 - ratio)`.
    pub(super) ratio: f64,
    /// Whether `P(k)`, `Q(k)` and `R(k)` are positive for every `k ≥ 1`: the tree's numbers may be
    /// cut.
    pub(super) positive: bool,
}

/// A node of the tree: `T`, `Q`, and `R` where it is needed.
struct Node {
    t: Num,
    q: Num,
    r: Option<Num>,
}

/// The terms of a leaf.
const LEAF: u64 = 16;

/// A signed magnitude of the leaves.
#[derive(Default)]
struct Signed {
    neg: bool,
    m: Vec<u64>,
}

impl Signed {
    /// The value `v`.
    fn of(v: &Val) -> Signed {
        Signed {
            neg: v.neg,
            m: v.limbs().to_vec(),
        }
    }

    /// `self·v` in place; `tmp` and `scratch` are working memory.
    fn mul(&mut self, v: &Val, tmp: &mut Vec<u64>, scratch: &mut Vec<u64>) {
        if self.m.is_empty() || v.n == 0 {
            self.m.clear();
            self.neg = false;
            return;
        }
        self.neg ^= v.neg;
        if v.n == 1 {
            let y = v.m[0];
            let mut carry = 0u64;
            for x in &mut self.m {
                let t = u128::from(*x) * u128::from(y) + u128::from(carry);
                *x = t as u64;
                carry = (t >> 64) as u64;
            }
            if carry != 0 {
                self.m.push(carry);
            }
        } else {
            nat::mul_into(tmp, &self.m, v.limbs(), scratch);
            limbs::trim(tmp);
            core::mem::swap(&mut self.m, tmp);
        }
    }

    /// `self·v` into `out`.
    fn mul_into(&self, v: &Val, out: &mut Signed, scratch: &mut Vec<u64>) {
        out.m.clear();
        out.neg = self.neg ^ v.neg;
        if self.m.is_empty() || v.n == 0 {
            out.neg = false;
            return;
        }
        nat::mul_into(&mut out.m, &self.m, v.limbs(), scratch);
        limbs::trim(&mut out.m);
    }

    /// `self + y` in place.
    fn add(&mut self, y: &Signed) {
        if y.m.is_empty() {
            return;
        }
        if self.m.is_empty() {
            self.neg = y.neg;
            self.m.clear();
            self.m.extend_from_slice(&y.m);
            return;
        }
        if self.neg == y.neg {
            if self.m.len() < y.m.len() {
                self.m.resize(y.m.len(), 0);
            }
            if nat::add_assign(&mut self.m, &y.m) {
                self.m.push(1);
            }
            return;
        }
        match limbs::cmp(&self.m, &y.m) {
            core::cmp::Ordering::Less => {
                self.m = limbs::sub(&y.m, &self.m);
                self.neg = y.neg;
            }
            core::cmp::Ordering::Equal => {
                self.m.clear();
                self.neg = false;
            }
            core::cmp::Ordering::Greater => {
                let borrow = nat::sub_assign(&mut self.m, &y.m);
                debug_assert!(!borrow);
                limbs::trim(&mut self.m);
            }
        }
    }

    /// The exact [`Num`], its zero bits at the bottom moved to the exponent.
    fn num(self) -> Num {
        let mut n = Num::exact(self.neg, self.m);
        if !n.is_zero() {
            let z = limbs::trailing_zeros(&n.m);
            if z > 0 {
                n.m = limbs::shr(&n.m, z);
                n.e = z;
            }
        }
        n
    }
}

impl Series {
    /// The least number of terms `N` whose tail `|Σ_(k > N) t_k|` is at most `2^-target`.
    ///
    /// `log2 |t_k|` is bounded from above from the polynomials' values in floating point, each
    /// with a bound `ρ` of its relative error ([`Poly::at_f64`]): a quotient of two of them is
    /// within `1 + 2(ρ1 + ρ2)` of its value while `ρ1, ρ2 < 2^-10` (asserted), less than
    /// `3(ρ1 + ρ2)` in its logarithm in base 2, and the logarithm itself and the sums are rounded,
    /// by less than `2^-40` each for these magnitudes (below `2^12`): the bound adds all of these,
    /// and half a bit more.
    pub(super) fn terms(&self, target: f64) -> u64 {
        let geometric = -(1.0 - self.ratio).log2();
        // `log2 Π_(j < k) |R(j)/Q(j)|`, and the slack of its rounding errors.
        let (mut s, mut slack) = (0.0f64, 0.5f64);
        let mut k = 1u64;
        loop {
            let (qk, eq) = self.q.at_f64(k);
            let (pk, ep) = self.p.at_f64(k);
            let (rk, er) = self.r.at_f64(k);
            assert!(eq < 1e-3 && ep < 1e-3 && er < 1e-3);
            let lt = (pk / qk).log2() + s + 3.0 * (ep + eq) + slack;
            if lt + geometric <= -target {
                return k - 1;
            }
            s += (rk / qk).log2();
            slack += 3.0 * (er + eq) + 2f64.powi(-40);
            k += 1;
        }
    }

    /// `(T, Q)` of the first `n` terms, `k = 1..=n`, `Σ t_k = T/Q`, with the numbers cut to `w`
    /// bits (a positive series only) or exact.
    pub(super) fn sum(&self, n: u64, w: Option<u64>) -> (Num, Num) {
        debug_assert!(w.is_none() || self.positive);
        if n == 0 {
            return (Num::default(), Num::exact(false, vec![1]));
        }
        let mut cut = Cut::new(w);
        let node = self.split(1, n + 1, false, &mut cut);
        (node.t, node.q)
    }

    /// The node of the terms `a..b`, with `R` when `need_r`.
    fn split(&self, a: u64, b: u64, need_r: bool, cut: &mut Cut) -> Node {
        if b - a <= LEAF {
            return self.leaf(a, b, need_r, cut);
        }
        let leaves = (b - a).div_ceil(LEAF);
        let m = a + leaves / 2 * LEAF;
        let r_one = self.r.is_one();
        let left = self.split(a, m, !r_one, cut);
        let right = self.split(m, b, need_r && !r_one, cut);
        let t1q2 = cut.mul(&left.t, &right.q);
        let t = match &left.r {
            Some(r1) => {
                let r1t2 = cut.mul(r1, &right.t);
                cut.add(t1q2, r1t2)
            }
            None => cut.add(t1q2, right.t),
        };
        let q = cut.mul(&left.q, &right.q);
        let r = match (need_r && !r_one, &left.r, &right.r) {
            (true, Some(r1), Some(r2)) => Some(cut.mul(r1, r2)),
            _ => None,
        };
        Node { t, q, r }
    }

    /// The node of the terms `a..b`, exactly, then cut.
    fn leaf(&self, a: u64, b: u64, need_r: bool, cut: &mut Cut) -> Node {
        let scratch = &mut cut.scratch;
        let (p_one, r_one) = (self.p.is_one(), self.r.is_one());
        let need_r = need_r && !r_one;
        let k = b - 1;
        let mut t = Signed::of(&self.p.at(k));
        let mut qs = Signed::of(&self.q.at(k));
        let mut rs = Signed::of(&self.r.at(k));
        let (mut pq, mut tmp) = (Signed::default(), Vec::new());
        for k in (a..b - 1).rev() {
            // `T ← P(k) Q_s + R(k) T`.
            if !r_one {
                let rk = self.r.at(k);
                t.mul(&rk, &mut tmp, scratch);
                if need_r {
                    rs.mul(&rk, &mut tmp, scratch);
                }
            }
            if p_one {
                t.add(&qs);
            } else {
                qs.mul_into(&self.p.at(k), &mut pq, scratch);
                t.add(&pq);
            }
            qs.mul(&self.q.at(k), &mut tmp, scratch);
        }
        if self.positive {
            debug_assert!(
                !t.neg && !qs.neg && !rs.neg,
                "a positive series has positive numbers"
            );
        }
        let mut node = Node {
            t: t.num(),
            q: qs.num(),
            r: need_r.then(|| rs.num()),
        };
        cut.cut(&mut node.t);
        cut.cut(&mut node.q);
        if let Some(r) = &mut node.r {
            cut.cut(r);
        }
        node
    }
}

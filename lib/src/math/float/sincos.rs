//! The kernels of the circular functions, on fixed-point numbers of a working precision: the
//! Taylor series of the sine and the cosine summed by rectangular splitting, the doubling of the
//! angle, the series of the arctangent, and the arctangent by Newton's iteration on the sine and
//! the cosine. Each result carries a bound of its error, which the functions of
//! [`trig`](super::trig) and [`atan`](super::atan) hand to Ziv's rounding.
//!
//! A fixed-point number of `F = 64·fl` bits after the point is a magnitude `M` of limbs, the value
//! `M·2^-F`; `F` a multiple of 64, so that a product drops whole limbs. Errors are counted in
//! units of the last place (ulps) of the value they belong to.

use super::limbs;
use super::short;
use super::split;
use crate::math::numbers::nat;

/// What each operation on a [`Lg`] adds to the logarithm: far above the rounding errors of the
/// `f64` arithmetic and of `log2`/`exp2` on the values it meets (below `2^41` in magnitude), so
/// that a bound stays a bound.
const SLACK: f64 = 1.0 / 256.0;

/// An upper bound `2^l` of a non-negative quantity, held as `l`, rounded up at each operation;
/// `-∞` for zero.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lg(pub(super) f64);

impl Lg {
    /// The bound of zero.
    pub(super) const ZERO: Lg = Lg(f64::NEG_INFINITY);

    /// The bound `x`, itself an upper bound of the quantity.
    #[inline]
    pub(super) fn of(x: f64) -> Lg {
        if x > 0.0 {
            Lg(x.log2() + SLACK)
        } else {
            Lg::ZERO
        }
    }

    /// The bound `2^e`.
    #[inline]
    pub(super) fn pow2(e: i128) -> Lg {
        Lg(e as f64 + SLACK)
    }

    /// The bound of a sum.
    #[inline]
    pub(super) fn add(self, o: Lg) -> Lg {
        let (a, b) = if self.0 >= o.0 {
            (self.0, o.0)
        } else {
            (o.0, self.0)
        };
        if b == f64::NEG_INFINITY {
            return Lg(a);
        }
        Lg(a + (1.0 + (b - a).exp2()).log2() + SLACK)
    }

    /// The bound of a product.
    #[inline]
    pub(super) fn mul(self, o: Lg) -> Lg {
        if self.0 == f64::NEG_INFINITY || o.0 == f64::NEG_INFINITY {
            return Lg::ZERO;
        }
        Lg(self.0 + o.0 + SLACK)
    }

    /// The bound times `2^k`.
    #[inline]
    pub(super) fn scale(self, k: i128) -> Lg {
        if self.0 == f64::NEG_INFINITY {
            return self;
        }
        Lg(self.0 + k as f64 + SLACK)
    }

    /// The least integer `e` with `2^e` at least the bound; `None` for zero.
    #[inline]
    pub(super) fn ceil(self) -> Option<i128> {
        if self.0 == f64::NEG_INFINITY {
            None
        } else {
            Some(self.0.ceil() as i128)
        }
    }
}

/// An upper bound of the magnitude `m` as a [`Lg`], `m < (t + 1)·2^(bits - 64)` for its top 64
/// bits `t`.
pub(super) fn lg_up(m: &[u64]) -> Lg {
    let bits = limbs::bit_len(m);
    if bits == 0 {
        return Lg::ZERO;
    }
    let t = top64(m, bits);
    Lg((t as f64 + 1.0).log2() + SLACK + (bits as f64 - 64.0))
}

/// A lower bound `log2 m` of the magnitude `m`, which is not zero: `m >= t·2^(bits - 64)`.
pub(super) fn lg_down(m: &[u64]) -> f64 {
    let bits = limbs::bit_len(m);
    debug_assert!(bits > 0);
    let t = top64(m, bits);
    (t as f64).log2() - SLACK + (bits as f64 - 64.0)
}

/// The top 64 bits of a magnitude of `bits` bits, the bits below cut off (as the top bits, shifted
/// up, when there are fewer than 64).
fn top64(m: &[u64], bits: u64) -> u64 {
    let n = m.len();
    let top = m[n - 1];
    let lz = top.leading_zeros();
    if bits <= 64 {
        top << lz
    } else if lz == 0 {
        top
    } else {
        top << lz | m[n - 2] >> (64 - lz)
    }
}

/// A value `±m·2^lsb` whose error is at most `2^lsb·2^err`: `err` in units of its last place.
#[derive(Clone, Debug)]
pub(super) struct Fx {
    /// The sign.
    pub(super) negative: bool,
    /// The magnitude, without zero limbs at the top.
    pub(super) m: Vec<u64>,
    /// The exponent of the magnitude's last bit.
    pub(super) lsb: i128,
    /// The bound of the error, in units of `2^lsb`.
    pub(super) err: Lg,
}

impl Fx {
    /// The exact value `+m·2^lsb`.
    pub(super) fn exact(m: Vec<u64>, lsb: i128) -> Fx {
        Fx {
            negative: false,
            m,
            lsb,
            err: Lg::ZERO,
        }
    }

    /// The bound of the absolute error, as the exponent of a power of two (a very low one for an
    /// exact value).
    pub(super) fn abs_err(&self) -> Lg {
        self.err.scale(self.lsb)
    }

    /// The exponent `e` of the value's magnitude, `2^(e - 1) <= |v| < 2^e`, of the approximation.
    pub(super) fn exponent(&self) -> i128 {
        self.lsb + i128::from(limbs::bit_len(&self.m))
    }

    /// The value cut to `bits` significant bits at most: the bits below dropped, the error grown
    /// by one unit of the new last place when they were not zero.
    pub(super) fn cut(mut self, bits: u64) -> Fx {
        let b = limbs::bit_len(&self.m);
        if b > bits {
            let d = b - bits;
            let lost = !limbs::zero_below(&self.m, d);
            self.m = limbs::shr(&self.m, d);
            self.err = self.err.scale(-i128::from(d));
            self.lsb += i128::from(d);
            if lost {
                self.err = self.err.add(Lg(0.0));
            }
        }
        self
    }

    /// The value as a fixed-point number of `f` bits after the point, rounded down, and the bound
    /// of its error in units of `2^-f`.
    pub(super) fn to_fixed(&self, f: u64) -> (Vec<u64>, Lg) {
        let s = self.lsb + i128::from(f);
        let err = self.err.scale(s);
        if s >= 0 {
            (limbs::shl(&self.m, s as u64), err)
        } else {
            let d = s.unsigned_abs().min(u128::from(u64::MAX)) as u64;
            let lost = !self.m.is_empty() && !limbs::zero_below(&self.m, d);
            let m = if d >= 64 * self.m.len() as u64 {
                Vec::new()
            } else {
                limbs::shr(&self.m, d)
            };
            (m, if lost { err.add(Lg(0.0)) } else { err })
        }
    }
}

/// The signed sum of terms, exactly at the lowest of their last places, the errors added.
pub(super) fn sum(terms: &[&Fx]) -> Fx {
    let lsb = terms.iter().map(|t| t.lsb).min().expect("terms");
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    let mut err = Lg::ZERO;
    for t in terms {
        let m = limbs::shl(&t.m, (t.lsb - lsb) as u64);
        if t.negative {
            neg = limbs::add(&neg, &m);
        } else {
            pos = limbs::add(&pos, &m);
        }
        err = err.add(t.err.scale(t.lsb - lsb));
    }
    limbs::trim(&mut pos);
    limbs::trim(&mut neg);
    let (negative, m) = match limbs::cmp(&pos, &neg) {
        core::cmp::Ordering::Less => (true, limbs::sub(&neg, &pos)),
        _ => (false, limbs::sub(&pos, &neg)),
    };
    Fx {
        negative,
        m,
        lsb,
        err,
    }
}

/// The quotient `a/b`, of about `bits` significant bits, with the bound of its error: of the
/// errors of both and of the quotient rounded down. `None` when `b`'s error is not below half of
/// it, so that nothing is known of the quotient.
pub(super) fn quotient(a: &Fx, b: &Fx, bits: u64) -> Option<Fx> {
    if b.m.is_empty() {
        return None;
    }
    let (ba, bb) = (limbs::bit_len(&a.m), limbs::bit_len(&b.m));
    // The bound: with `|ã - a| <= α`, `|b̃ - b| <= β <= |b̃|/2`,
    // `|a/b - ã/b̃| <= (|ã| + α)·β/(|b̃|·|b|) + α/|b̃| <= 2(|ã| + α)·β/|b̃|^2 + α/|b̃|`.
    let lb = lg_down(&b.m) + b.lsb as f64;
    let beta = b.abs_err();
    if beta.0 > lb - 1.0 {
        return None;
    }
    let alpha = a.abs_err();
    let la = lg_up(&a.m).scale(a.lsb).add(alpha);
    let mut err = Lg(1.0 + la.0 + beta.0 - 2.0 * lb + SLACK);
    if beta.0 == f64::NEG_INFINITY {
        err = Lg::ZERO;
    }
    err = err.add(Lg(alpha.0 - lb + SLACK));
    if a.m.is_empty() {
        return Some(Fx {
            negative: a.negative != b.negative,
            m: Vec::new(),
            lsb: a.lsb - b.lsb,
            err: err.scale(b.lsb - a.lsb),
        });
    }
    let s = (bits + 2 + bb).saturating_sub(ba);
    let q = div(&limbs::shl(&a.m, s), &b.m);
    let lsb = a.lsb - i128::from(s) - b.lsb;
    Some(Fx {
        negative: a.negative != b.negative,
        m: q,
        lsb,
        err: err.scale(-lsb).add(Lg(0.0)),
    })
}

// Fixed-point arithmetic.

/// `floor(a·b·2^(-64·fl))`.
#[inline]
pub(super) fn fmul(a: &[u64], b: &[u64], fl: usize) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = limbs::mul(a, b);
    drop_limbs(&mut r, fl);
    r
}

/// `floor(a²·2^(-64·fl))`.
#[inline]
pub(super) fn fsqr(a: &[u64], fl: usize) -> Vec<u64> {
    fmul(a, a, fl)
}

/// Drops the `k` lowest limbs.
#[inline]
fn drop_limbs(r: &mut Vec<u64>, k: usize) {
    if k >= r.len() {
        r.clear();
    } else {
        r.drain(..k);
    }
}

/// `a·c`.
pub(super) fn mul_small(a: &[u64], c: u64) -> Vec<u64> {
    let mut r = Vec::with_capacity(a.len() + 1);
    let mut carry = 0u64;
    for &x in a {
        let p = u128::from(x) * u128::from(c) + u128::from(carry);
        r.push(p as u64);
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        r.push(carry);
    }
    limbs::trim(&mut r);
    r
}

/// `a ← a·c`, in place.
fn mul_small_in(a: &mut Vec<u64>, c: u64) {
    let mut carry = 0u64;
    for x in a.iter_mut() {
        let p = u128::from(*x) * u128::from(c) + u128::from(carry);
        *x = p as u64;
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        a.push(carry);
    }
    limbs::trim(a);
}

/// `a ← a + b`, in place.
fn add_in(a: &mut Vec<u64>, b: &[u64]) {
    if a.len() < b.len() {
        a.resize(b.len(), 0);
    }
    if nat::add_assign(a, b) {
        a.push(1);
    }
}

/// `a ← a - b` for `a >= b`, in place.
fn sub_in(a: &mut Vec<u64>, b: &[u64]) {
    let borrow = nat::sub_assign(a, b);
    debug_assert!(!borrow);
    limbs::trim(a);
}

/// `a ← a·2^s` for `s < 64`, in place.
fn shl_in(a: &mut Vec<u64>, s: u32) {
    debug_assert!(s > 0 && s < 64);
    let mut carry = 0u64;
    for x in a.iter_mut() {
        let v = *x;
        *x = v << s | carry;
        carry = v >> (64 - s);
    }
    if carry != 0 {
        a.push(carry);
    }
}

/// `a ← floor(a/d)` for `d > 0`, in place, as [`div_small`].
fn div_small_in(a: &mut Vec<u64>, d: u64) {
    let n = a.len();
    if n == 0 {
        return;
    }
    let s = d.leading_zeros();
    let dn = d << s;
    let v = nat::invert_limb(dn);
    let mut r = if s == 0 { 0 } else { a[n - 1] >> (64 - s) };
    for i in (0..n).rev() {
        let u0 = if s == 0 {
            a[i]
        } else {
            let lo = if i > 0 { a[i - 1] >> (64 - s) } else { 0 };
            a[i] << s | lo
        };
        let (qi, ri) = nat::div_2by1(r, u0, dn, v);
        a[i] = qi;
        r = ri;
    }
    limbs::trim(a);
}

/// `floor(a/d)` for `d > 0`, by the reciprocal of `d` normalized (Möller and Granlund).
pub(super) fn div_small(a: &[u64], d: u64) -> Vec<u64> {
    debug_assert!(d != 0);
    let n = a.len();
    if n == 0 {
        return Vec::new();
    }
    let s = d.leading_zeros();
    let dn = d << s;
    let v = nat::invert_limb(dn);
    let mut q = vec![0u64; n];
    let mut r = if s == 0 { 0 } else { a[n - 1] >> (64 - s) };
    for i in (0..n).rev() {
        let u0 = if s == 0 {
            a[i]
        } else {
            let lo = if i > 0 { a[i - 1] >> (64 - s) } else { 0 };
            a[i] << s | lo
        };
        let (qi, ri) = nat::div_2by1(r, u0, dn, v);
        q[i] = qi;
        r = ri;
    }
    limbs::trim(&mut q);
    q
}

/// `floor(a / Π t)` for the integers `t` of `range`, each below `2^32`: by as many of them at a
/// time as a limb holds, one division by a limb each, as `floor(floor(a/x)/y) = floor(a/(x·y))`.
fn div_range(mut a: Vec<u64>, range: core::ops::RangeInclusive<u64>) -> Vec<u64> {
    let mut d = 1u64;
    for t in range {
        match d.checked_mul(t) {
            Some(p) => d = p,
            None => {
                div_small_in(&mut a, d);
                d = t;
            }
        }
    }
    if d > 1 {
        div_small_in(&mut a, d);
    }
    a
}

/// `floor(a/b)` for `b` not zero.
pub(super) fn div(a: &[u64], b: &[u64]) -> Vec<u64> {
    if limbs::cmp(a, b) == core::cmp::Ordering::Less {
        return Vec::new();
    }
    if b.len() == 1 {
        return div_small(a, b[0]);
    }
    div_rem(a, b).0
}

/// `floor(a/b)` and the remainder, for `b` not zero.
pub(super) fn div_rem(a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    use core::cmp::Ordering;
    if limbs::cmp(a, b) == Ordering::Less {
        return (Vec::new(), a.to_vec());
    }
    let (ba, bb) = (limbs::bit_len(a), limbs::bit_len(b));
    if bb > 64 && ba - bb <= 62 {
        // A quotient below `2^63` from the top bits: with `s = bits(b) - 64`, `a' = floor(a/2^s)`
        // and `b' = floor(b/2^s)`, `q' = floor(a'/b')` is within `[q - 1, q + 2]` of the
        // quotient `q` (`a'/(b' + 1) < a/b < (a' + 1)/b'`, `q < 2^63 <= b'`), then corrected.
        let s = bb - 64;
        let bt = limbs::shr(b, s)[0];
        let at = limbs::shr(a, s);
        let at = u128::from(at[0]) | u128::from(at.get(1).copied().unwrap_or(0)) << 64;
        let mut q = (at / u128::from(bt)).min(u128::from(u64::MAX)) as u64;
        let mut prod = mul_small(b, q);
        while limbs::cmp(&prod, a) == Ordering::Greater {
            q -= 1;
            prod = limbs::sub(&prod, b);
        }
        let mut r = limbs::sub(a, &prod);
        while limbs::cmp(&r, b) != Ordering::Less {
            q += 1;
            r = limbs::sub(&r, b);
        }
        let mut qv = vec![q];
        limbs::trim(&mut qv);
        return (qv, r);
    }
    let (mut q, mut r) = (Vec::new(), Vec::new());
    nat::div_rem_into(&mut q, &mut r, a, b, &mut Vec::new());
    (q, r)
}

/// `floor(sqrt(a))`.
pub(super) fn isqrt(a: &[u64]) -> Vec<u64> {
    if a.is_empty() {
        return Vec::new();
    }
    let mut s = Vec::new();
    nat::sqrt_into(&mut s, a, &mut Vec::new());
    s
}

/// `2^(64·fl)`, one in fixed point.
#[inline]
pub(super) fn one(fl: usize) -> Vec<u64> {
    let mut v = vec![0u64; fl + 1];
    v[fl] = 1;
    v
}

/// The value `m·2^-f` as an `f64` rounded up, for the bounds; `m` at most `2^f`.
fn up(m: &[u64], f: u64) -> f64 {
    if m.is_empty() {
        return 0.0;
    }
    lg_up(m).scale(-i128::from(f)).0.exp2() * (1.0 + 1e-12)
}

/// The value `m·2^-f` as an `f64`, about (its top 64 bits, rounded to 53), for estimates.
fn to_f64(m: &[u64], f: u64) -> f64 {
    let bits = limbs::bit_len(m);
    if bits == 0 {
        return 0.0;
    }
    top64(m, bits) as f64 * (bits as f64 - 64.0 - f as f64).exp2()
}

// The series.

/// A lower bound of `log2 t` for an integer `t >= 1`.
#[inline]
fn log2_floor(t: u64) -> u64 {
    u64::from(63 - t.leading_zeros())
}

/// The block size of rectangular splitting for `n` terms: even, about `sqrt(n)`, at least 2.
#[inline]
fn block(n: u64) -> u64 {
    let r = (n as f64).sqrt().round() as u64;
    (r + (r & 1)).max(2)
}

/// The powers `y^1..=y^m` of the fixed-point `y < 1`, each rounded down; `p[0]` is one. `p[i]` is
/// within `i - 1` ulps below `y^i`: a product of `p[i - 1]` and `y` adds at most one to the error
/// of `p[i - 1]`, as `y < 1`, and a square of `p[i]` doubles it and adds one, `2(i - 1) + 1 =
/// 2i - 1`.
fn powers(y: &[u64], m: u64, fl: usize) -> Vec<Vec<u64>> {
    let mut p = Vec::with_capacity(m as usize + 1);
    p.push(one(fl));
    p.push(y.to_vec());
    for i in 2..=m as usize {
        let next = if i % 2 == 0 {
            fsqr(&p[i / 2], fl)
        } else {
            fmul(&p[i - 1], y, fl)
        };
        p.push(next);
    }
    p
}

/// `Σ_(n < N) (-1)^n y^n / (2n + δ)!` for the fixed-point `y = Y·2^-F`, `F = 64·fl`, `0 <= y <
/// 1/4`, and `δ` 1 (the sine over its argument) or 2 (one less the cosine, over the square of its
/// argument), with `N` the terms that leave a tail below half an ulp; and the bound of the error in
/// ulps, which also holds the tail.
///
/// Rectangular splitting (Smith, 1989): the powers `y^1..y^m`, `m` even, are computed once, and
/// the terms in blocks of `m`, `n = jm + i`. With `f(n) = (2n + δ)!`, block `j`'s terms are
/// `y^(jm) T_j / f(jm + m - 1)` for the integer combination `T_j = Σ_i (-1)^i y^i c_i`, `c_i =
/// f(jm + m - 1) / f(jm + i)`, summed by `B ← B·g + (-1)^i y^i` with the small factors `g =
/// c_i / c_(i + 1)`, exactly. Horner's rule over the blocks, `Q_j = T_j + y^m Q_(j+1) / D_j` with
/// `D_j = f(jm + 2m - 1) / f(jm + m - 1)`, a product of `2m` integers divided out a limb at a
/// time, gives the sum `Q_0 / f(m - 1)`.
///
/// The error: the powers are within `i - 1` ulps, and an error in `y^i` of block `j` counts in the
/// sum with the weight `c_i / f(jm + m - 1) = 1/f(jm + i)`, which sums below 0.01 over the `i >=
/// 2`; the two roundings of each Horner step count `1/f(jm + m - 1) <= 1/6` each, below 0.34 over
/// all `j` (the factorials grow faster than geometrically); the error of `y^m` times `Q_(j+1) <=
/// 2 c_0` counts `2(m - 1)/f(jm + m) <= m/50`; the last division one ulp; the tail half an ulp.
/// Below `2 + m/50` ulps in all, bounded by `3 + m/16`.
fn series_sin_cos(y: &[u64], fl: usize, delta: u64) -> (Vec<u64>, f64) {
    let f = 64 * fl as u64;
    let yb = limbs::bit_len(y);
    if yb == 0 {
        let mut r = one(fl);
        if delta == 2 {
            r = limbs::shr(&r, 1);
        }
        return (r, 0.0);
    }
    debug_assert!(yb + 2 <= f, "a series argument below 1/4");
    // `y < 2^(yb - F)`: the tail after `N` terms is below `y^N / f(N) < 2^(-N·gap - lf(N))`.
    let gap = f - yb;
    let mut n = 1u64;
    let mut lf: u64 = (2..=delta).map(log2_floor).sum();
    loop {
        // `lf` = a lower bound of `log2 f(n)`.
        lf += log2_floor(2 * n + delta - 1) + log2_floor(2 * n + delta);
        if n * gap + lf > f {
            break;
        }
        n += 1;
    }
    let m = block(n);
    let blocks = n.div_ceil(m);
    let p = powers(y, m, fl);
    let mut q: Vec<u64> = Vec::new();
    for j in (0..blocks).rev() {
        let base = 2 * j * m;
        let mut b = p[0].clone();
        for i in 1..m {
            let g = (base + 2 * (i - 1) + delta + 1) * (base + 2 * (i - 1) + delta + 2);
            mul_small_in(&mut b, g);
            let term = &p[i as usize];
            if i % 2 == 1 {
                sub_in(&mut b, term);
            } else {
                add_in(&mut b, term);
            }
        }
        if j + 1 < blocks {
            let x = fmul(&p[m as usize], &q, fl);
            let x = div_range(x, base + 2 * m - 1 + delta..=base + 4 * m - 2 + delta);
            add_in(&mut b, &x);
        }
        q = b;
    }
    let s = div_range(q, 2..=2 * m - 2 + delta);
    (s, 3.0 + m as f64 / 16.0)
}

/// `Σ_(n < N) (-1)^n y^n / (2n + 1)` for the fixed-point `y`, `0 <= y < 1/4`, the arctangent over
/// its argument, `N` terms leaving a tail below half an ulp; and the bound of the error in ulps.
///
/// Rectangular splitting as for [`series_sin_cos`], but the coefficients are no factorials: each
/// term of a block is `floor(y^i / (2n + 1))`, and the blocks are summed by Horner's rule, `Q_j =
/// T_j + y^m Q_(j+1)` (`m` even). The error of each term is at most `(i - 1)/(2n + 1) + 1 < 1.5`
/// ulps, each product of the rule adds one ulp and `m - 1` ulps of `y^m` times `Q_(j+1) <= 1.1`:
/// at most `J (2.6 m + 1)` ulps over `J` blocks, and half an ulp of tail; bounded by `J (3m + 1)
/// + 1`.
fn series_atan(y: &[u64], fl: usize) -> (Vec<u64>, f64) {
    let f = 64 * fl as u64;
    let yb = limbs::bit_len(y);
    if yb == 0 {
        return (one(fl), 0.0);
    }
    debug_assert!(yb + 2 <= f, "a series argument below 1/4");
    let gap = f - yb;
    let mut n = 1u64;
    while n * gap + log2_floor(2 * n + 1) <= f {
        n += 1;
    }
    let m = block(n);
    let blocks = n.div_ceil(m);
    let p = powers(y, m, fl);
    let mut q: Vec<u64> = Vec::new();
    for j in (0..blocks).rev() {
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        for i in 0..m {
            let t = div_small(&p[i as usize], 2 * (j * m + i) + 1);
            if i % 2 == 0 {
                pos = limbs::add(&pos, &t);
            } else {
                neg = limbs::add(&neg, &t);
            }
        }
        limbs::trim(&mut pos);
        limbs::trim(&mut neg);
        let mut t = limbs::sub(&pos, &neg);
        if j + 1 < blocks {
            t = limbs::add(&t, &fmul(&p[m as usize], &q, fl));
        }
        q = t;
    }
    limbs::trim(&mut q);
    (q, blocks as f64 * (3.0 * m as f64 + 1.0) + 1.0)
}

/// The exponent `-K` to which an argument is halved before the series of the cosine, at `F` bits
/// after the point: the doublings back cost about `K` squares and the series `sqrt(F/(2K))`
/// products, which balance near `K = (F/2)^(1/3)`.
pub(super) fn halvings(f: u64) -> i128 {
    ((f as f64 / 2.0).cbrt().round() as i128).max(3)
}

/// The sine and the cosine of `r = R·2^-g`, `0 < |r| <= 1`, of which the sign plays no part (the
/// caller gives the sine its sign): each only when asked for, as [`Fx`]s with bounds of their
/// errors, at `F = 64·fl` bits after the point or, for the sine of an argument below `2^-K`
/// ([`halvings`]), relatively to it.
///
/// - For `|r| < 2^-K`, `sin r = r·u(r²)` and `cos r = 1 - r²·v(r²)` by the series of
///   [`series_sin_cos`], `u` with `δ = 1` and `v` with `δ = 2`; `r` cut to `F + 64` bits.
/// - Otherwise the argument is halved `k` times, to `r' = r/2^k` of the exponent `-K`; `t_0 = 1 -
///   cos r' = r'^2 v(r'^2)` by the series, and `k` doublings `t' = 1 - cos 2a = 4t - 2t²`. An error
///   `η` of `t` becomes at most `η (4 + 4t + 2η) + 2` ulps, which grows as `t` does, so that the
///   relative error stays about that of `t_0`, times `Π (1 + t_i) <= e^(4/3·0.46) < 1.85`. Then
///   `cos r = 1 - t` and `sin r = sqrt(t (2 - t))`, whose error is at most `η / sqrt(t_min) + 1`
///   ulps, as `d sin / dt = (1 - t)/sin r <= 1/sqrt(t)` from `sin² = t(2 - t) >= t`.
pub(super) fn sin_cos(
    r: &[u64],
    g: i128,
    fl: usize,
    want_sin: bool,
    want_cos: bool,
) -> (Option<Fx>, Option<Fx>) {
    if fl <= short::MAX_LIMBS {
        return short::sin_cos(r, g, fl, want_sin, want_cos);
    }
    let f = 64 * fl as u64;
    let fi = i128::from(f);
    let k_halve = halvings(f);
    // `r` cut to `F + 64` bits: an error below one unit of the new last place, `2^-g`.
    let rb = limbs::bit_len(r);
    let (r, g, cut) = if rb > f + 64 {
        let d = rb - (f + 64);
        (limbs::shr(r, d), g - i128::from(d), true)
    } else {
        (r.to_vec(), g, false)
    };
    let er = i128::from(limbs::bit_len(&r)) - g;
    debug_assert!(er <= 1 && !r.is_empty());
    let r2 = limbs::mul(&r, &r);
    // `y = r'^2` at `F` bits for `r' = r/2^k`: within one ulp below when bits are dropped.
    let square = |k: i128| -> Vec<u64> {
        let s = fi - 2 * g - 2 * k;
        if s >= 0 {
            limbs::shl(&r2, s as u64)
        } else {
            limbs::shr(&r2, s.unsigned_abs() as u64)
        }
    };
    if er <= -k_halve {
        let y = square(0);
        let sin = want_sin.then(|| {
            let (u, eu) = series_sin_cos(&y, fl, 1);
            // `|du/dy| <= 1/6`, so the ulp of `y` counts below one of `u`; times `R`, in units of
            // `2^-(g + F)`; the cut of `r` one unit of `2^-g`, `2^F` of those.
            let m = limbs::mul(&r, &u);
            let mut err = lg_up(&r).mul(Lg::of(eu + 1.0));
            if cut {
                err = err.add(Lg::pow2(fi));
            }
            Fx {
                negative: false,
                m,
                lsb: -(g + fi),
                err,
            }
        });
        let cos = want_cos.then(|| {
            let (v, ev) = series_sin_cos(&y, fl, 2);
            // `t = y·v`: `|dt/dy| <= 1/2` for the ulp of `y`, `y·ev < ev`, the product's floor one
            // ulp, the cut of `r` below one more.
            let t = fmul(&y, &v, fl);
            Fx {
                negative: false,
                m: limbs::sub(&one(fl), &t),
                lsb: -fi,
                err: Lg::of(ev + 3.0),
            }
        });
        return (sin, cos);
    }
    if fl >= BS_LIMBS {
        return split::sin_cos(&r, g, cut, fl, want_sin, want_cos);
    }
    let k = er + k_halve;
    let y = square(k);
    let (v, ev) = series_sin_cos(&y, fl, 2);
    let mut t = fmul(&y, &v, fl);
    // The error of `t_0`, as for the cosine above; then of the doublings, `h = η / 4^i`.
    let mut h = ev + 3.0;
    for i in 0..k {
        let tu = up(&t, f);
        let scale = (2.0 * i as f64 - f as f64).exp2();
        h = h * (1.0 + tu + h * scale * 0.5 + 1e-12) + 2.0 * 0.25f64.powi(i as i32 + 1) + 1e-300;
        let mut t2 = fsqr(&t, fl);
        shl_in(&mut t, 2);
        shl_in(&mut t2, 1);
        sub_in(&mut t, &t2);
    }
    // `η = h·4^k` ulps.
    let eta = Lg::of(h).scale(2 * k);
    let cos = want_cos.then(|| Fx {
        negative: false,
        m: limbs::sub(&one(fl), &t),
        lsb: -fi,
        err: eta,
    });
    let sin = want_sin.then(|| {
        // `t_min = t - η` from below, in `log2`.
        let lt = lg_down(&t) - fi as f64;
        let le = eta.0 - fi as f64;
        let lt_min = if le < lt - 1.0 {
            lt + (1.0 - (le - lt).exp2()).log2() - SLACK
        } else {
            f64::NEG_INFINITY
        };
        let two_minus = limbs::sub(&limbs::shl(&one(fl), 1), &t);
        let s = isqrt(&limbs::mul(&t, &two_minus));
        let err = if lt_min == f64::NEG_INFINITY {
            Lg(fi as f64)
        } else {
            Lg(eta.0 - lt_min / 2.0 + SLACK).add(Lg(0.0))
        };
        Fx {
            negative: false,
            m: s,
            lsb: -fi,
            err,
        }
    });
    (sin, cos)
}

/// From this many limbs after the point (about 410,000 bits), the sine and the cosine of an
/// argument above `2^-K` by binary splitting ([`split`]) instead of the doublings: measured on an
/// Apple M4 Max, a sine of 200,000 bits took 28 ms by the doublings and 68 ms by binary splitting,
/// of 400,000 bits 129 ms by either, and of 1,000,000 bits 434 ms against 324 ms.
pub(super) const BS_LIMBS: usize = 6400;

/// `atan(z)` for the fixed-point `z = Z·2^-F`, `0 < z <= 1`, `F = 64·fl`, by Newton's iteration on
/// `tan y = z`, `y ← y - cos y (sin y - z cos y)`, at doubling precisions from the `f64`
/// arctangent; and the bound of the error in ulps.
///
/// The bound comes after the fact from the last step's own sine and cosine, whatever the
/// precision of the step before: for `y = θ + ε`, `tan θ = z`, `sin ε = (sin y - z cos y)·cos θ`,
/// so `|ε| <= (π/2)|sin y - z cos y|` (for `|ε| <= π/2`, which `0 <= y <= 1` makes sure). The
/// exact step gives `θ + ε - cos(θ + ε) sin ε / cos θ - θ = ε - sin(2ε)/2 + z sin² ε`, at most
/// `z ε² + (2/3)|ε|³`. The step computed adds the errors of `cos y (sin y - z cos y)`: `|c̃ - c|·|ρ̃|
/// + |ρ̃ - ρ|` and the floor of the product.
pub(super) fn atan_newton(z: &[u64], fl: usize) -> (Vec<u64>, Lg) {
    let f = 64 * fl as u64;
    let fi = i128::from(f);
    let mut y = if fl <= 2 {
        // The `f64` arctangent of `z`'s top bits: about 50 bits right, which the step doubles.
        let zf = to_f64(z, f).min(1.0);
        let a = zf.atan();
        let y0 = (a * (1u64 << 62) as f64) as u64;
        limbs::shl(&[y0], f - 62)
    } else {
        let fl2 = (f / 2 + 32).div_ceil(64) as usize;
        let d = 64 * (fl - fl2) as u64;
        let (y2, _) = atan_newton(&limbs::shr(z, d), fl2);
        limbs::shl(&y2, d)
    };
    limbs::trim(&mut y);
    // `0 <= y <= 1`, so that `|ε| <= 1`.
    if limbs::cmp(&y, &one(fl)) == core::cmp::Ordering::Greater {
        y = one(fl);
    }
    if y.is_empty() {
        y = limbs::shr(&one(fl), 1);
    }
    let (s, c) = sin_cos(&y, fi, fl, true, true);
    let (s, es) = s.expect("asked").to_fixed(f);
    let (c, ec) = c.expect("asked").to_fixed(f);
    let zc = fmul(z, &c, fl);
    // `ρ = s - z c`, within `es + z ec + 1` ulps.
    let (rho_neg, rho) = match limbs::cmp(&s, &zc) {
        core::cmp::Ordering::Less => (true, limbs::sub(&zc, &s)),
        _ => (false, limbs::sub(&s, &zc)),
    };
    let drho = es.add(ec).add(Lg(0.0));
    let eps = lg_up(&rho)
        .add(drho)
        .mul(Lg((core::f64::consts::FRAC_PI_2).log2()));
    let corr = fmul(&c, &rho, fl);
    let y = if rho_neg {
        limbs::add(&y, &corr)
    } else if limbs::cmp(&corr, &y) == core::cmp::Ordering::Greater {
        return (Vec::new(), Lg(fi as f64));
    } else {
        limbs::sub(&y, &corr)
    };
    let err = Lg(2.0 * eps.0 - fi as f64 + SLACK)
        .add(Lg(3.0 * eps.0 - 2.0 * fi as f64 + SLACK))
        .add(ec.mul(lg_up(&rho)).scale(-fi))
        .add(drho)
        .add(Lg(0.0));
    (y, err)
}

/// `atan(z)` for `0 < z <= 1` (an [`Fx`] with its error), at `F = 64·fl` bits after the point, or
/// relatively to `z` for `z < 2^-K`: then by the series `z·Σ (-1)^n z^(2n)/(2n + 1)`
/// ([`series_atan`]), otherwise by [`atan_newton`]. The error of `z` passes as it is, `|atan' |
/// <= 1`.
pub(super) fn atan(z: &Fx, fl: usize) -> Fx {
    let f = 64 * fl as u64;
    let fi = i128::from(f);
    let z = z.clone().cut(f + 64);
    let ez = z.exponent();
    if ez <= -halvings(f) {
        let r2 = limbs::mul(&z.m, &z.m);
        let s = fi + 2 * z.lsb;
        let y = if s >= 0 {
            limbs::shl(&r2, s as u64)
        } else {
            limbs::shr(&r2, s.unsigned_abs().min(u128::from(u64::MAX)) as u64)
        };
        let (u, eu) = series_atan(&y, fl);
        // `|du/dy| <= 1/3` for the ulp of `y`; the error of `z` times `u <= 1`, in units of
        // `2^(lsb_z - F)`, `2^F` of `z`'s units.
        let err = lg_up(&z.m).mul(Lg::of(eu + 1.0)).add(z.err.scale(fi));
        return Fx {
            negative: false,
            m: limbs::mul(&z.m, &u),
            lsb: z.lsb - fi,
            err,
        };
    }
    let (zf, ez) = z.to_fixed(f);
    let (y, ey) = atan_newton(&zf, fl);
    Fx {
        negative: false,
        m: y,
        lsb: -fi,
        err: ey.add(ez),
    }
}

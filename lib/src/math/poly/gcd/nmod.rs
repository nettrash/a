//! Residues modulo a word and the kernels of the greatest common divisors on polynomials of word
//! coefficients: remainders that reduce once a coefficient for two rows of the division (by the
//! reciprocal, or in Montgomery's form, or as pseudo-remainders without inverses), Euclid's
//! algorithm with the product of the quotient matrices; the products are the arithmetic area's.
//!
//! A polynomial is a vector of its coefficients, each below `n`, the constant first, without
//! zeros at the top, as [`ModPolynomial`](super::super::ModPolynomial) holds those of a modulus
//! below `2^64`.

use super::super::arith::nmod::WordMod;
use super::super::arith::wmul;
use super::hgcd::Mat;
use crate::math::numbers::nat::{self, div_2by1, invert_limb};

/// The modulus `n ≥ 1`, prepared for remainders by it: shifted so that its top bit is set, with
/// its reciprocal (Möller and Granlund), and the shift.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Nmod {
    /// `n`.
    pub(crate) n: u64,
    /// `n << s`, its top bit set.
    d: u64,
    /// The reciprocal of `d`.
    v: u64,
    /// The leading zeros of `n`.
    s: u32,
}

impl Nmod {
    /// The modulus `n ≥ 1`.
    #[inline]
    pub(crate) fn new(n: u64) -> Nmod {
        debug_assert!(n >= 1);
        let s = n.leading_zeros();
        let d = n << s;
        Nmod {
            n,
            d,
            v: invert_limb(d),
            s,
        }
    }

    /// Whether two products of residues and a residue sum to below `n·2^64`, as the kernels that
    /// reduce once for two rows need: `n < 2^63`.
    #[inline(always)]
    pub(crate) fn two_rows(&self) -> bool {
        self.s >= 1
    }

    /// `(hi·2^64 + lo) mod n` for `hi < n`.
    #[inline(always)]
    pub(crate) fn red2(&self, hi: u64, lo: u64) -> u64 {
        debug_assert!(hi < self.n);
        if self.s == 0 {
            div_2by1(hi, lo, self.d, self.v).1
        } else {
            let u1 = (hi << self.s) | (lo >> (64 - self.s));
            div_2by1(u1, lo << self.s, self.d, self.v).1 >> self.s
        }
    }

    /// `x mod n` for `x < n·2^64`.
    #[inline(always)]
    pub(crate) fn red(&self, x: u128) -> u64 {
        self.red2((x >> 64) as u64, x as u64)
    }

    /// `x mod n` for any word.
    #[inline(always)]
    pub(crate) fn red1(&self, x: u64) -> u64 {
        if x < self.n { x } else { self.red2(0, x) }
    }

    /// `(t2·2^128 + t1·2^64 + t0) mod n`.
    #[inline(always)]
    pub(crate) fn red3(&self, t2: u64, t1: u64, t0: u64) -> u64 {
        let r = self.red1(t2);
        let r = self.red2(r, t1);
        self.red2(r, t0)
    }

    /// `a·b mod n` for residues.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        self.red(u128::from(a) * u128::from(b))
    }

    /// `a + b mod n` for residues.
    #[inline(always)]
    pub(crate) fn add(&self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_add(b);
        if c || s >= self.n {
            s.wrapping_sub(self.n)
        } else {
            s
        }
    }

    /// `a - b mod n` for residues.
    #[inline(always)]
    pub(crate) fn sub(&self, a: u64, b: u64) -> u64 {
        let (s, b) = a.overflowing_sub(b);
        if b { s.wrapping_add(self.n) } else { s }
    }

    /// `-a mod n` for a residue.
    #[inline(always)]
    pub(crate) fn neg(&self, a: u64) -> u64 {
        if a == 0 { 0 } else { self.n - a }
    }

    /// The inverse of the residue `a`, `None` when it is not a unit.
    #[inline]
    pub(crate) fn inv(&self, a: u64) -> Option<u64> {
        nat::inverse_word(a, self.n)
    }

    /// The inverse of the residue `a`, which the division needs.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not a unit.
    #[inline]
    #[track_caller]
    pub(crate) fn inv_lead(&self, a: u64) -> u64 {
        self.inv(a)
            .expect("the leading coefficient is not a unit modulo n")
    }

    /// `aᵉ mod n`.
    pub(crate) fn pow(&self, a: u64, mut e: u64) -> u64 {
        let mut r = self.red1(1);
        let mut b = a;
        while e > 0 {
            if e & 1 == 1 {
                r = self.mul(r, b);
            }
            e >>= 1;
            if e > 0 {
                b = self.mul(b, b);
            }
        }
        r
    }
}

/// Drops the zeros at the top of `a`.
#[inline]
pub(crate) fn trim(a: &mut Vec<u64>) {
    while a.last() == Some(&0) {
        a.pop();
    }
}

/// `r = r mod b` in place, for `b` not empty with the inverse `binv` of its leading coefficient:
/// the division from the top, two coefficients of the quotient at a time with one reduction a
/// coefficient for both rows when `n < 2^63`, one at a time otherwise; `r` trimmed below the length
/// of `b`. With `q`, the quotient's coefficients as well, at their places (`q` as long as `r` is
/// longer than `b`, by one).
pub(crate) fn divrem_in(
    m: &Nmod,
    r: &mut Vec<u64>,
    b: &[u64],
    binv: u64,
    mut q: Option<&mut [u64]>,
) {
    let lb = b.len();
    debug_assert!(lb >= 1);
    if r.len() < lb {
        trim(r);
        return;
    }
    let n = m.n;
    let mut i = r.len() - 1;
    let two = m.two_rows() && lb >= 2;
    // `i` is the top coefficient still to clear, at or above `lb - 1`.
    loop {
        if two && i >= lb {
            // Two rows: the quotient's coefficients at `i` and `i - 1` (of `x^(i - lb + 1)` and
            // `x^(i - lb)`), then the coefficients `i - lb ..= i - 2` reduced once each.
            let q1 = m.mul(r[i], binv);
            let t = m.sub(r[i - 1], m.mul(q1, b[lb - 2]));
            let q0 = m.mul(t, binv);
            let (nq1, nq0) = (u128::from(n - q1), u128::from(n - q0));
            let base = i - lb;
            let w = &mut r[base..i - 1];
            // Position `base + j`: row q0 brings b[j], row q1 b[j - 1].
            w[0] = m.red(u128::from(w[0]) + nq0 * u128::from(b[0]));
            for j in 1..lb - 1 {
                let s = u128::from(w[j]) + nq0 * u128::from(b[j]) + nq1 * u128::from(b[j - 1]);
                w[j] = m.red(s);
            }
            if let Some(q) = q.as_deref_mut() {
                q[i - lb + 1] = q1;
                q[i - lb] = q0;
            }
            r[i] = 0;
            r[i - 1] = 0;
            if i < lb + 1 {
                break;
            }
            i -= 2;
            if i < lb - 1 {
                break;
            }
        } else {
            let q1 = m.mul(r[i], binv);
            let nq = u128::from(n - q1);
            let base = i + 1 - lb;
            for (x, &y) in r[base..i].iter_mut().zip(b) {
                *x = m.red(u128::from(*x) + nq * u128::from(y));
            }
            if let Some(q) = q.as_deref_mut() {
                q[base] = q1;
            }
            r[i] = 0;
            if i < lb {
                break;
            }
            i -= 1;
        }
    }
    r.truncate(lb - 1);
    trim(r);
}

/// The quotient and the remainder of `a` by `b`, not empty, its leading coefficient a unit.
pub(crate) fn divrem(m: &Nmod, a: &[u64], b: &[u64]) -> (Vec<u64>, Vec<u64>) {
    let mut r = a.to_vec();
    if a.len() < b.len() {
        return (Vec::new(), r);
    }
    let binv = m.inv_lead(b[b.len() - 1]);
    let mut q = vec![0u64; a.len() - b.len() + 1];
    divrem_in(m, &mut r, b, binv, Some(&mut q));
    trim(&mut q);
    (q, r)
}

/// `a + b`.
pub(crate) fn add(m: &Nmod, a: &[u64], b: &[u64]) -> Vec<u64> {
    let (l, s) = if a.len() >= b.len() { (a, b) } else { (b, a) };
    let mut r = l.to_vec();
    for (x, &y) in r.iter_mut().zip(s) {
        *x = m.add(*x, y);
    }
    trim(&mut r);
    r
}

/// `a - b`.
pub(crate) fn sub(m: &Nmod, a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut r = a.to_vec();
    if r.len() < b.len() {
        r.resize(b.len(), 0);
    }
    for (x, &y) in r.iter_mut().zip(b) {
        *x = m.sub(*x, y);
    }
    trim(&mut r);
    r
}

/// The product `a·b`, trimmed, by the arithmetic area's word products (schoolbook, Kronecker
/// substitution or the number-theoretic transforms, by the lengths).
pub(crate) fn mul(w: &WordMod, a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut r = if core::ptr::eq(a, b) {
        wmul::sqr(a, w)
    } else {
        wmul::mul(a, b, w)
    };
    trim(&mut r);
    r
}

/// Montgomery's form modulo the odd `n`: `x` held as `x·2^64 mod n`, and products reduced by
/// REDC, two multiplications.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Mont {
    /// `n`, odd.
    n: u64,
    /// `n⁻¹ mod 2^64`.
    ninv: u64,
    /// `2^128 mod n`.
    r2: u64,
}

impl Mont {
    /// The form modulo the odd `n` of `m`.
    pub(crate) fn new(m: &Nmod) -> Mont {
        debug_assert!(m.n & 1 == 1);
        let r = m.red2(1, 0);
        Mont {
            n: m.n,
            ninv: nat::binvert(m.n),
            r2: m.mul(r, r),
        }
    }

    /// `(hi·2^64 + lo)·2^-64 mod n`, for `hi < n`, in `0..n`.
    #[inline(always)]
    pub(crate) fn redc(&self, hi: u64, lo: u64) -> u64 {
        let q = lo.wrapping_mul(self.ninv);
        let t = ((u128::from(q) * u128::from(self.n)) >> 64) as u64;
        let (r, b) = hi.overflowing_sub(t);
        if b { r.wrapping_add(self.n) } else { r }
    }

    /// `a·b·2^-64 mod n`.
    #[inline(always)]
    pub(crate) fn mul(&self, a: u64, b: u64) -> u64 {
        let p = u128::from(a) * u128::from(b);
        self.redc((p >> 64) as u64, p as u64)
    }

    /// The form of the residue `x`.
    #[inline(always)]
    pub(crate) fn to(&self, x: u64) -> u64 {
        self.mul(x, self.r2)
    }

    /// The residue of the form `x`.
    #[inline(always)]
    pub(crate) fn from(&self, x: u64) -> u64 {
        self.redc(0, x)
    }

    /// `a - b mod n`.
    #[inline(always)]
    fn sub(&self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_sub(b);
        if c { s.wrapping_add(self.n) } else { s }
    }
}

/// `r = r mod b` in place in Montgomery's form, `binv` the form of the inverse of `b`'s leading
/// coefficient: two coefficients of the quotient at a time, the two products of a coefficient
/// summed and reduced by one REDC, for `n < 2^63`; one at a time otherwise.
fn rem_mont(mo: &Mont, r: &mut Vec<u64>, b: &[u64], binv: u64, mut qo: Option<&mut Vec<u64>>) {
    let lb = b.len();
    if let Some(q) = qo.as_deref_mut() {
        q.clear();
        q.resize((r.len() + 1).saturating_sub(lb), 0);
    }
    if r.len() < lb {
        trim(r);
        return;
    }
    let two = mo.n < (1 << 63) && lb >= 2;
    let mut i = r.len() - 1;
    loop {
        if two && i >= lb {
            let q1 = mo.mul(r[i], binv);
            let t = mo.sub(r[i - 1], mo.mul(q1, b[lb - 2]));
            let q0 = mo.mul(t, binv);
            let base = i - lb;
            let w = &mut r[base..i - 1];
            w[0] = mo.sub(w[0], mo.mul(q0, b[0]));
            for ((x, &b1), &b0) in w[1..].iter_mut().zip(&b[..lb - 2]).zip(&b[1..lb - 1]) {
                let s = u128::from(q1) * u128::from(b1) + u128::from(q0) * u128::from(b0);
                *x = mo.sub(*x, mo.redc((s >> 64) as u64, s as u64));
            }
            if let Some(q) = qo.as_deref_mut() {
                q[i - lb + 1] = q1;
                q[i - lb] = q0;
            }
            r[i] = 0;
            r[i - 1] = 0;
            if i < lb + 1 {
                break;
            }
            i -= 2;
        } else {
            let q = mo.mul(r[i], binv);
            let base = i + 1 - lb;
            for (x, &y) in r[base..i].iter_mut().zip(b) {
                *x = mo.sub(*x, mo.mul(q, y));
            }
            if let Some(qv) = qo.as_deref_mut() {
                qv[base] = q;
            }
            r[i] = 0;
            if i < lb {
                break;
            }
            i -= 1;
        }
    }
    r.truncate(lb - 1);
    trim(r);
}

/// The last non-zero remainder of Euclid's algorithm on `a` and `b`, not made monic; with `rec`,
/// the degree (plus `shift`) and the leading coefficient of each divisor, in order, stopping at a
/// constant one. For an odd modulus in Montgomery's form, converted on the way in and out.
pub(crate) fn euclid(
    m: &Nmod,
    mut a: Vec<u64>,
    mut b: Vec<u64>,
    mut rec: Option<&mut Vec<(usize, u64)>>,
    shift: usize,
) -> Vec<u64> {
    if m.n & 1 == 1 && m.n > 1 {
        return euclid_mont(m, a, b, rec, shift);
    }
    while !b.is_empty() {
        let l = b[b.len() - 1];
        if let Some(rec) = rec.as_deref_mut() {
            rec.push((b.len() - 1 + shift, l));
            if b.len() == 1 {
                return b;
            }
        }
        let binv = m.inv_lead(l);
        divrem_in(m, &mut a, &b, binv, None);
        core::mem::swap(&mut a, &mut b);
    }
    a
}

/// Below this length of the divisor a step of [`euclid_mont`] with a quotient of one or two
/// coefficients is taken without the inverse of the leading coefficient, as a pseudo-remainder:
/// the inverse costs about as much as a pass over 500 coefficients.
pub(crate) const FRACTION_FREE: usize = 600;

/// Euclid's algorithm in Montgomery's form for the odd `n`, see [`euclid`]. The steps whose
/// quotient has one or two coefficients and whose divisor is short (below [`FRACTION_FREE`]) take
/// the pseudo-remainder `lc(b)^(δ+1)·a - q·b` instead, three products a coefficient summed and
/// reduced by one REDC (for `n < 2^62`), and no inverse; each remainder is then the true one times
/// a known factor, `c_(i+1) = λᵢ c_(i-1)`, and the leading coefficients recorded are divided by
/// those factors at the end, all inverted at once (Montgomery's trick), which also checks that
/// every leading coefficient was a unit.
fn euclid_mont(
    m: &Nmod,
    mut a: Vec<u64>,
    mut b: Vec<u64>,
    rec: Option<&mut Vec<(usize, u64)>>,
    shift: usize,
) -> Vec<u64> {
    let mo = Mont::new(m);
    for x in a.iter_mut().chain(b.iter_mut()) {
        *x = mo.to(*x);
    }
    let one = mo.to(m.red1(1));
    let ff = m.n < (1 << 62);
    // The factors of a and b over the true remainders, and the record in the form: (degree, the
    // leading coefficient held, its factor).
    let (mut ca, mut cb) = (one, one);
    let mut held: Vec<(usize, u64, u64)> = Vec::new();
    // The product of the leading coefficients the steps did not invert, to check they are units.
    let mut units = one;
    while !b.is_empty() {
        let lb = b.len();
        let l = b[lb - 1];
        if rec.is_some() {
            held.push((lb - 1 + shift, l, cb));
            if lb == 1 {
                a = b;
                break;
            }
        }
        let delta = a.len().wrapping_sub(lb);
        if ff && (2..FRACTION_FREE).contains(&lb) && (delta == 0 || delta == 1) && a.len() >= lb {
            let lam = pseudo_step(&mo, &mut a, &b);
            units = mo.mul(units, l);
            // a becomes λ·(a mod b): its factor λ·c_a.
            ca = mo.mul(lam, ca);
        } else {
            let binv = mo.to(m.inv_lead(mo.from(l)));
            rem_mont(&mo, &mut a, &b, binv, None);
        }
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut ca, &mut cb);
    }
    if units != one {
        // Panics unless every leading coefficient was a unit.
        m.inv_lead(mo.from(units));
    }
    if let Some(rec) = rec {
        // The true leading coefficients lc / c, the factors inverted together.
        let n = held.len();
        let mut pre = Vec::with_capacity(n);
        let mut acc = one;
        for &(_, _, c) in &held {
            pre.push(acc);
            acc = mo.mul(acc, c);
        }
        let mut inv = mo.to(m.inv_lead(mo.from(acc)));
        // inv = 1 / Π c; walk back: 1/cᵢ = inv·pre[i], then inv *= cᵢ.
        let mut out = vec![0u64; n];
        for i in (0..n).rev() {
            let (_, l, c) = held[i];
            let ci = mo.mul(inv, pre[i]);
            inv = mo.mul(inv, c);
            out[i] = mo.from(mo.mul(l, ci));
        }
        rec.extend(held.iter().zip(out).map(|(&(d, _, _), l)| (d, l)));
    }
    for x in &mut a {
        *x = mo.from(*x);
    }
    a
}

/// One pseudo-remainder step of Montgomery's form, `len a ∈ {len b, len b + 1}`, `len b ≥ 2`, `n <
/// 2^62`: `a ← lc(b)^(δ+1)·a - q·b` with the quotient's `δ + 1` coefficients chosen to clear the
/// top of `a`, trimmed below the length of `b`; returns the factor `lc(b)^(δ+1)`.
fn pseudo_step(mo: &Mont, a: &mut Vec<u64>, b: &[u64]) -> u64 {
    let lb = b.len();
    let n = mo.n;
    let lc = b[lb - 1];
    let lam = if a.len() == lb + 1 {
        let i = lb;
        let q1 = mo.mul(lc, a[i]);
        let q0 = mo.sub(mo.mul(lc, a[i - 1]), mo.mul(a[i], b[lb - 2]));
        let l2 = mo.mul(lc, lc);
        let (nq1, nq0) = (u128::from(n - q1), u128::from(n - q0));
        let lw = u128::from(l2);
        // a[j] ← L a[j] - q1 b[j-1] - q0 b[j], j = 0 ..= lb - 2.
        let p0 = lw * u128::from(a[0]) + nq0 * u128::from(b[0]);
        a[0] = mo.redc((p0 >> 64) as u64, p0 as u64);
        for ((x, &b1), &b0) in a[1..lb - 1].iter_mut().zip(&b[..lb - 2]).zip(&b[1..lb - 1]) {
            let s = lw * u128::from(*x) + nq1 * u128::from(b1) + nq0 * u128::from(b0);
            *x = mo.redc((s >> 64) as u64, s as u64);
        }
        l2
    } else {
        let i = lb - 1;
        let q = a[i];
        let nq = u128::from(n - q);
        let lw = u128::from(lc);
        for (x, &y) in a[..lb - 1].iter_mut().zip(b) {
            let s = lw * u128::from(*x) + nq * u128::from(y);
            *x = mo.redc((s >> 64) as u64, s as u64);
        }
        lc
    };
    a.truncate(lb - 1);
    trim(a);
    lam
}

/// `d += q·s` in Montgomery's form for the short `q`: each coefficient's products summed, two at a
/// time, and reduced by one REDC (for `n < 2^63`).
fn addmul_mont(mo: &Mont, d: &mut Vec<u64>, s: &[u64], q: &[u64]) {
    if s.is_empty() || q.is_empty() {
        return;
    }
    let len = s.len() + q.len() - 1;
    if d.len() < len {
        d.resize(len, 0);
    }
    let two = mo.n < (1 << 63);
    let mut k = 0;
    while k < q.len() {
        if two && k + 1 < q.len() {
            let (q0, q1) = (q[k], q[k + 1]);
            // d[k + j] += q0 s[j] + q1 s[j - 1].
            let w = &mut d[k..];
            w[0] = mo.add(w[0], mo.mul(q0, s[0]));
            for ((x, &s1), &s0) in w[1..].iter_mut().zip(s).zip(&s[1..]) {
                let p = u128::from(q1) * u128::from(s1) + u128::from(q0) * u128::from(s0);
                *x = mo.add(*x, mo.redc((p >> 64) as u64, p as u64));
            }
            let top = k + s.len();
            d[top] = mo.add(d[top], mo.mul(q1, s[s.len() - 1]));
            k += 2;
        } else {
            let qk = q[k];
            for (x, &y) in d[k..].iter_mut().zip(s) {
                *x = mo.add(*x, mo.mul(qk, y));
            }
            k += 1;
        }
    }
    trim(d);
}

impl Mont {
    /// `a + b mod n`.
    #[inline(always)]
    fn add(&self, a: u64, b: u64) -> u64 {
        let (s, c) = a.overflowing_add(b);
        if c || s >= self.n {
            s.wrapping_sub(self.n)
        } else {
            s
        }
    }
}

/// Euclid's algorithm on `a` and `b`, `len a > len b > 0`, with the product of the quotient
/// matrices: `(M, g)` with `(a, b) = M (g, 0)`, in Montgomery's form for an odd modulus.
pub(crate) fn euclid_mat(m: &Nmod, a: &[u64], b: &[u64]) -> (Mat<Vec<u64>>, Vec<u64>) {
    let mo = Mont::new(m);
    let to = |v: &[u64]| -> Vec<u64> { v.iter().map(|&x| mo.to(x)).collect() };
    let (mut x, mut y) = (to(a), to(b));
    // The columns (c1, c2) of M: M · [[q, 1], [1, 0]] = (c1 q + c2, c1).
    let one = mo.to(m.red1(1));
    let (mut c11, mut c21) = (vec![one], Vec::new());
    let (mut c12, mut c22) = (Vec::new(), vec![one]);
    let mut neg = false;
    let mut q = Vec::new();
    while !y.is_empty() {
        let l = mo.from(y[y.len() - 1]);
        let binv = mo.to(m.inv_lead(l));
        rem_mont(&mo, &mut x, &y, binv, Some(&mut q));
        trim(&mut q);
        addmul_mont(&mo, &mut c12, &c11, &q);
        addmul_mont(&mo, &mut c22, &c21, &q);
        core::mem::swap(&mut c11, &mut c12);
        core::mem::swap(&mut c21, &mut c22);
        neg = !neg;
        core::mem::swap(&mut x, &mut y);
    }
    let from = |v: Vec<u64>| -> Vec<u64> { v.into_iter().map(|x| mo.from(x)).collect() };
    (
        Mat {
            a: from(c11),
            b: from(c12),
            c: from(c21),
            d: from(c22),
            neg,
        },
        from(x),
    )
}

/// The engine of the half greatest common divisor on word coefficients.
pub(crate) struct Word {
    /// The modulus.
    pub(crate) m: Nmod,
    /// The arithmetic area's form of the modulus, for the products.
    pub(crate) wm: WordMod,
    /// [`Engine::hgcd_cutoff`].
    pub(crate) hgcd: usize,
    /// [`Engine::gcd_cutoff`].
    pub(crate) gcd: usize,
    /// [`Engine::xgcd_cutoff`].
    pub(crate) xgcd: usize,
}

/// Below this degree the half gcd on words takes Euclid's steps.
pub(crate) const HGCD_CUTOFF: usize = 100;
/// Below this degree the gcd on words is Euclid's, for a modulus of 35 to 63 bits.
pub(crate) const GCD_CUTOFF: usize = 3000;
/// Below this degree the extended gcd on words is Euclid's, for a modulus of 35 to 63 bits.
pub(crate) const XGCD_CUTOFF: usize = 1500;

/// The degrees below which the gcd and the extended gcd modulo `n` are Euclid's. The half gcd's
/// products cost with the number of transform primes, which grows with the bits of `n`, while a
/// step of Euclid's costs about the same for every `n` below 2^63 and more from 2^63. Measured on an
/// Apple M4 Max at 17, 24, 30, 40, 50, 63 and 64 bits.
fn cutoffs(n: u64) -> (usize, usize) {
    match 64 - n.leading_zeros() {
        0..=26 => (1000, 600),
        27..=34 => (1900, 950),
        35..=63 => (GCD_CUTOFF, XGCD_CUTOFF),
        _ => (2000, 850),
    }
}

impl Word {
    /// The engine modulo `n`, with the crossovers for its size.
    pub(crate) fn new(n: u64) -> Word {
        let (gcd, xgcd) = cutoffs(n);
        Word {
            m: Nmod::new(n),
            wm: WordMod::new(n),
            hgcd: HGCD_CUTOFF,
            gcd,
            xgcd,
        }
    }
}

impl super::hgcd::Engine for Word {
    type P = Vec<u64>;
    type C = u64;

    #[inline]
    fn len(&self, a: &Vec<u64>) -> usize {
        a.len()
    }
    fn zero(&self) -> Vec<u64> {
        Vec::new()
    }
    fn one(&self) -> Vec<u64> {
        let o = self.m.red1(1);
        if o == 0 { Vec::new() } else { vec![o] }
    }
    fn lead(&self, a: &Vec<u64>) -> u64 {
        a[a.len() - 1]
    }
    fn add(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        add(&self.m, a, b)
    }
    fn sub(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        sub(&self.m, a, b)
    }
    fn neg(&self, a: &Vec<u64>) -> Vec<u64> {
        a.iter().map(|&x| self.m.neg(x)).collect()
    }
    fn mul(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        mul(&self.wm, a, b)
    }
    fn divrem(&self, a: &Vec<u64>, b: &Vec<u64>) -> (Vec<u64>, Vec<u64>) {
        divrem(&self.m, a, b)
    }
    fn shr(&self, a: &Vec<u64>, k: usize) -> Vec<u64> {
        a.get(k..).map_or_else(Vec::new, <[u64]>::to_vec)
    }
    fn low(&self, a: &Vec<u64>, k: usize) -> Vec<u64> {
        let mut r = a[..k.min(a.len())].to_vec();
        trim(&mut r);
        r
    }
    fn shl_add(&self, hi: &Vec<u64>, k: usize, lo: &Vec<u64>) -> Vec<u64> {
        if hi.is_empty() {
            return lo.clone();
        }
        let mut r = vec![0u64; (k + hi.len()).max(lo.len())];
        r[..lo.len()].copy_from_slice(lo);
        for (x, &y) in r[k..].iter_mut().zip(hi) {
            *x = self.m.add(*x, y);
        }
        trim(&mut r);
        r
    }
    fn cmul(&self, x: &u64, y: &u64) -> u64 {
        self.m.mul(*x, *y)
    }
    fn cneg(&self, x: &u64) -> u64 {
        self.m.neg(*x)
    }
    fn cpow(&self, x: &u64, e: u64) -> u64 {
        self.m.pow(*x, e)
    }
    fn cone(&self) -> u64 {
        self.m.red1(1)
    }
    fn czero(&self) -> u64 {
        0
    }
    fn euclid(
        &self,
        a: Vec<u64>,
        b: Vec<u64>,
        rec: Option<&mut Vec<(usize, u64)>>,
        shift: usize,
    ) -> Vec<u64> {
        euclid(&self.m, a, b, rec, shift)
    }
    fn euclid_mat(&self, a: &Vec<u64>, b: &Vec<u64>) -> Option<(Mat<Vec<u64>>, Vec<u64>)> {
        (self.m.n & 1 == 1 && self.m.n > 1).then(|| euclid_mat(&self.m, a, b))
    }
    fn hgcd_cutoff(&self) -> usize {
        self.hgcd
    }
    fn gcd_cutoff(&self) -> usize {
        self.gcd
    }
    fn xgcd_cutoff(&self) -> usize {
        self.xgcd
    }
}

//! Residues modulo an integer of `k ≥ 2` limbs and the kernels of the greatest common divisors on
//! polynomials of such coefficients, packed `k` limbs apiece as
//! [`ModPolynomial`](super::super::ModPolynomial) holds them: remainders that sum the products of
//! two rows of the division in `2k + 1` limbs and reduce once, schoolbook products that reduce once
//! a coefficient; the products are the arithmetic area's.

use super::super::arith::lred::LimbRed;
use super::super::arith::mmul;
use crate::math::numbers::{ByteOrder, Number, from_limbs, limbs_of, nat};
use core::cell::RefCell;

/// The modulus `n` of `k` limbs, shifted so that its top bit is set, for the crate's division.
#[derive(Clone, Debug)]
pub(crate) struct Mmod {
    /// `k`.
    pub(crate) k: usize,
    /// The limbs of `n`.
    pub(crate) n: Vec<u64>,
    /// `n` as a number.
    num: Number,
    /// The arithmetic area's division by `n`.
    pub(crate) red: LimbRed,
}

/// Working memory of the reductions.
#[derive(Default)]
pub(crate) struct Scr {
    /// The division's scratch.
    s: Vec<u64>,
    /// A product.
    p: Vec<u64>,
    /// A sum of products.
    acc: Vec<u64>,
}

/// `r += a` on limbs, `r` at least as long, the carry rippling up `r`.
#[inline]
fn add_in(r: &mut [u64], a: &[u64]) {
    if nat::add_assign(&mut r[..a.len()], a) {
        for x in &mut r[a.len()..] {
            let (s, c) = x.overflowing_add(1);
            *x = s;
            if !c {
                break;
            }
        }
    }
}

impl Mmod {
    /// `out = x mod n`, `out` of `k` limbs, `x` of any length: by the arithmetic area's
    /// division by `n` (two registers for two limbs).
    #[inline]
    pub(crate) fn reduce(&self, out: &mut [u64], x: &[u64], _w: &mut Scr) {
        self.red.reduce(x, out);
    }
    /// `out = a·b mod n`.
    pub(crate) fn mul(&self, out: &mut [u64], a: &[u64], b: &[u64], w: &mut Scr) {
        let mut p = core::mem::take(&mut w.p);
        p.clear();
        p.resize(2 * self.k, 0);
        nat::mul_to(&mut p, a, b, &mut w.s);
        self.reduce(out, &p, w);
        w.p = p;
    }

    /// `r = r + a mod n`.
    #[inline]
    pub(crate) fn add(&self, r: &mut [u64], a: &[u64]) {
        let carry = nat::add_assign(r, a);
        if carry || cmp(r, &self.n) != core::cmp::Ordering::Less {
            nat::sub_assign(r, &self.n);
        }
    }

    /// `r = r - a mod n`.
    #[inline]
    pub(crate) fn sub(&self, r: &mut [u64], a: &[u64]) {
        if nat::sub_assign(r, a) {
            nat::add_assign(r, &self.n);
        }
    }

    /// `r = n - r` for `r` not zero, 0 for 0.
    #[inline]
    pub(crate) fn neg(&self, r: &mut [u64]) {
        if r.iter().any(|&t| t != 0) {
            nat::sub_from(r, &self.n);
        }
    }

    /// `n - r` as limbs: `n` for `r = 0` (a multiple of `n`, which the sums of products take).
    #[inline]
    fn complement(&self, out: &mut [u64], r: &[u64]) {
        out.copy_from_slice(&self.n);
        nat::sub_assign(out, r);
    }

    /// The inverse of the residue `a`, which the division needs.
    ///
    /// # Panics
    ///
    /// Panics if `a` is not a unit.
    #[track_caller]
    pub(crate) fn inv_lead(&self, a: &[u64]) -> Vec<u64> {
        let x = from_limbs(a, ByteOrder::LittleEndian)
            .mod_inverse(&self.num)
            .expect("the leading coefficient is not a unit modulo n");
        let mut l = limbs_of(&x);
        l.resize(self.k, 0);
        l
    }

    /// `aᵉ mod n`.
    pub(crate) fn pow(&self, a: &[u64], mut e: u64) -> Vec<u64> {
        let mut w = Scr::default();
        let mut r = vec![0u64; self.k];
        r[0] = 1;
        let mut b = a.to_vec();
        let mut t = vec![0u64; self.k];
        while e > 0 {
            if e & 1 == 1 {
                self.mul(&mut t, &r, &b, &mut w);
                r.copy_from_slice(&t);
            }
            e >>= 1;
            if e > 0 {
                self.mul(&mut t, &b, &b, &mut w);
                b.copy_from_slice(&t);
            }
        }
        r
    }
}

/// `a` compared with `b`, of the same length, from the top.
#[inline]
fn cmp(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        if x != y {
            return x.cmp(y);
        }
    }
    core::cmp::Ordering::Equal
}

/// Drops the zero coefficients at the top of the packed `a`.
#[inline]
pub(crate) fn trim(a: &mut Vec<u64>, k: usize) {
    while a.len() >= k && a[a.len() - k..].iter().all(|&t| t == 0) {
        a.truncate(a.len() - k);
    }
}

/// `r = r mod b` in place for the packed `b` not empty, with the inverse `binv` of its leading
/// coefficient: two coefficients of the quotient at a time, the products of both rows summed in
/// `2k + 1` limbs a coefficient and reduced once; `r` trimmed below the length of `b`. With `q`,
/// the quotient's coefficients too, at their places.
pub(crate) fn divrem_in(
    m: &Mmod,
    r: &mut Vec<u64>,
    b: &[u64],
    binv: &[u64],
    mut q: Option<&mut [u64]>,
    w: &mut Scr,
) {
    let k = m.k;
    let lb = b.len() / k;
    debug_assert!(lb >= 1);
    let lr = r.len() / k;
    if lr < lb {
        trim(r, k);
        return;
    }
    let co = |i: usize| i * k..(i + 1) * k;
    let mut q1 = vec![0u64; k];
    let mut q0 = vec![0u64; k];
    let mut t = vec![0u64; k];
    let mut u = vec![0u64; k];
    let mut nq1 = vec![0u64; k];
    let mut nq0 = vec![0u64; k];
    let mut acc = core::mem::take(&mut w.acc);
    let mut p = vec![0u64; 2 * k];
    let mut i = lr - 1;
    loop {
        if lb >= 2 && i >= lb {
            m.mul(&mut q1, &r[co(i)], binv, w);
            m.mul(&mut u, &q1, &b[co(lb - 2)], w);
            t.copy_from_slice(&r[co(i - 1)]);
            m.sub(&mut t, &u);
            m.mul(&mut q0, &t, binv, w);
            m.complement(&mut nq1, &q1);
            m.complement(&mut nq0, &q0);
            let base = i - lb;
            for j in 0..lb - 1 {
                acc.clear();
                acc.resize(2 * k + 1, 0);
                acc[..k].copy_from_slice(&r[co(base + j)]);
                nat::mul_to(&mut p, &nq0, &b[co(j)], &mut w.s);
                add_in(&mut acc, &p);
                if j >= 1 {
                    nat::mul_to(&mut p, &nq1, &b[co(j - 1)], &mut w.s);
                    add_in(&mut acc, &p);
                }
                let mut out = core::mem::take(&mut t);
                m.reduce(&mut out, &acc, w);
                r[co(base + j)].copy_from_slice(&out);
                t = out;
            }
            if let Some(q) = q.as_deref_mut() {
                q[co(i - lb + 1)].copy_from_slice(&q1);
                q[co(i - lb)].copy_from_slice(&q0);
            }
            r[co(i)].fill(0);
            r[co(i - 1)].fill(0);
            if i < lb + 1 {
                break;
            }
            i -= 2;
        } else {
            m.mul(&mut q1, &r[co(i)], binv, w);
            m.complement(&mut nq1, &q1);
            let base = i + 1 - lb;
            for j in 0..lb - 1 {
                acc.clear();
                acc.resize(2 * k + 1, 0);
                acc[..k].copy_from_slice(&r[co(base + j)]);
                nat::mul_to(&mut p, &nq1, &b[co(j)], &mut w.s);
                add_in(&mut acc, &p);
                m.reduce(&mut t, &acc, w);
                r[co(base + j)].copy_from_slice(&t);
            }
            if let Some(q) = q.as_deref_mut() {
                q[co(base)].copy_from_slice(&q1);
            }
            r[co(i)].fill(0);
            if i < lb {
                break;
            }
            i -= 1;
        }
    }
    w.acc = acc;
    r.truncate((lb - 1) * k);
    trim(r, k);
}

/// The engine of the half greatest common divisor on coefficients of `k ≥ 2` limbs.
pub(crate) struct Limbs {
    /// The modulus.
    pub(crate) m: Mmod,
    /// The working memory.
    pub(crate) w: RefCell<Scr>,
    /// [`Engine::hgcd_cutoff`](super::hgcd::Engine::hgcd_cutoff).
    pub(crate) hgcd: usize,
    /// [`Engine::gcd_cutoff`](super::hgcd::Engine::gcd_cutoff).
    pub(crate) gcd: usize,
    /// [`Engine::xgcd_cutoff`](super::hgcd::Engine::xgcd_cutoff).
    pub(crate) xgcd: usize,
}

/// Below this degree the half gcd on limbs takes Euclid's steps.
pub(crate) const HGCD_CUTOFF: usize = 60;
/// Below this degree the gcd on limbs is Euclid's.
pub(crate) const GCD_CUTOFF: usize = 250;
/// Below this degree the extended gcd on limbs is Euclid's.
pub(crate) const XGCD_CUTOFF: usize = 80;

impl Limbs {
    /// The engine modulo the `m` of several limbs, its division by `n` taken over.
    pub(crate) fn of(m: &crate::math::poly::modular::Modulus) -> Limbs {
        let red = m.l.clone().expect("a modulus of several limbs");
        Limbs {
            m: Mmod {
                k: m.d.len(),
                n: m.d.clone(),
                num: m.n.clone(),
                red,
            },
            w: RefCell::new(Scr::default()),
            hgcd: HGCD_CUTOFF,
            gcd: GCD_CUTOFF,
            xgcd: XGCD_CUTOFF,
        }
    }

    /// `g` divided by its leading coefficient; the zero polynomial stays.
    pub(crate) fn monic(&self, mut g: Vec<u64>) -> Vec<u64> {
        let k = self.m.k;
        if g.is_empty() {
            return g;
        }
        let l = &g[g.len() - k..];
        if l[0] == 1 && l[1..].iter().all(|&x| x == 0) {
            return g;
        }
        let inv = self.m.inv_lead(l);
        let w = &mut *self.w.borrow_mut();
        let mut t = vec![0u64; k];
        for c in g.chunks_exact_mut(k) {
            self.m.mul(&mut t, c, &inv, w);
            c.copy_from_slice(&t);
        }
        g
    }

    /// The product of the packed `a` and `b`, trimmed, by the arithmetic area's products of
    /// residues of several limbs.
    pub(crate) fn product(&self, a: &[u64], b: &[u64]) -> Vec<u64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let b = if core::ptr::eq(a, b) { &[][..] } else { b };
        let mut r = mmul::mul(a, b, &self.m.n, &self.m.red);
        trim(&mut r, self.m.k);
        r
    }
    /// The last non-zero remainder of Euclid's algorithm on the packed `a` and `b`; with `rec`,
    /// the divisors' degrees (plus `shift`) and leading coefficients, to a constant one.
    pub(crate) fn euclid_in(
        &self,
        mut a: Vec<u64>,
        mut b: Vec<u64>,
        rec: Option<&mut Vec<(usize, Vec<u64>)>>,
        shift: usize,
    ) -> Vec<u64> {
        let k = self.m.k;
        let w = &mut *self.w.borrow_mut();
        let one = {
            let mut o = vec![0u64; k];
            o[0] = 1;
            o
        };
        // The factors of a and b over the true remainders, the record held (degree, leading
        // coefficient, factor), and the product of the leading coefficients not inverted.
        let (mut ca, mut cb) = (one.clone(), one.clone());
        let mut held: Vec<(usize, Vec<u64>, Vec<u64>)> = Vec::new();
        let mut units = one.clone();
        let mut t = vec![0u64; k];
        while !b.is_empty() {
            let lb = b.len() / k;
            let l = b[b.len() - k..].to_vec();
            if rec.is_some() {
                held.push((lb - 1 + shift, l.clone(), cb.clone()));
                if lb == 1 {
                    a = b;
                    break;
                }
            }
            let delta = (a.len() / k).wrapping_sub(lb);
            if (2..FRACTION_FREE).contains(&lb) && (delta == 0 || delta == 1) {
                let lam = pseudo_step(&self.m, &mut a, &b, w);
                self.m.mul(&mut t, &units, &l, w);
                units.copy_from_slice(&t);
                self.m.mul(&mut t, &lam, &ca, w);
                ca.copy_from_slice(&t);
            } else {
                let binv = self.m.inv_lead(&l);
                let (_, mut r) = mmul::div_rem_school(&a, &b, &binv, &self.m.n, &self.m.red);
                trim(&mut r, k);
                a = r;
            }
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut ca, &mut cb);
        }
        if units != one {
            // Panics unless every leading coefficient was a unit.
            self.m.inv_lead(&units);
        }
        if let Some(rec) = rec {
            // The true leading coefficients lc / c, the factors inverted together.
            let n = held.len();
            let mut pre = Vec::with_capacity(n);
            let mut acc = one.clone();
            for (_, _, c) in &held {
                pre.push(acc.clone());
                self.m.mul(&mut t, &acc, c, w);
                acc.copy_from_slice(&t);
            }
            let mut inv = self.m.inv_lead(&acc);
            let mut out = vec![Vec::new(); n];
            for i in (0..n).rev() {
                let (_, ref l, ref c) = held[i];
                let mut ci = vec![0u64; k];
                self.m.mul(&mut ci, &inv, &pre[i], w);
                self.m.mul(&mut t, &inv, c, w);
                inv.copy_from_slice(&t);
                let mut lt = vec![0u64; k];
                self.m.mul(&mut lt, l, &ci, w);
                out[i] = lt;
            }
            rec.extend(held.iter().zip(out).map(|((d, _, _), l)| (*d, l)));
        }
        a
    }
}

/// Below this length of the divisor, a step of [`Limbs::euclid_in`] whose quotient has one or two
/// coefficients is a pseudo-remainder, without the inverse of the leading coefficient (an
/// inverse of a few limbs by the crate's extended gcd costs a pass over a few hundred
/// coefficients).
pub(crate) const FRACTION_FREE: usize = 24;

/// One pseudo-remainder step on limbs, `len a ∈ {len b, len b + 1}`, `len b ≥ 2`:
/// `a ← lc(b)^(δ+1)·a - q·b`, the three products of a coefficient summed in `2k + 2` limbs and
/// reduced once; returns the factor `lc(b)^(δ+1)`.
fn pseudo_step(m: &Mmod, a: &mut Vec<u64>, b: &[u64], w: &mut Scr) -> Vec<u64> {
    let k = m.k;
    let lb = b.len() / k;
    let co = |i: usize| i * k..(i + 1) * k;
    let lc = b[co(lb - 1)].to_vec();
    let mut acc = vec![0u64; 2 * k + 2];
    let mut p = vec![0u64; 2 * k];
    let mut t = vec![0u64; k];
    let mut out = vec![0u64; k];
    let lam = if a.len() / k == lb + 1 {
        let i = lb;
        let mut q1 = vec![0u64; k];
        m.mul(&mut q1, &lc, &a[co(i)], w);
        let mut q0 = vec![0u64; k];
        m.mul(&mut q0, &lc, &a[co(i - 1)], w);
        m.mul(&mut t, &a[co(i)], &b[co(lb - 2)], w);
        m.sub(&mut q0, &t);
        let mut l2 = vec![0u64; k];
        m.mul(&mut l2, &lc, &lc, w);
        let (mut nq1, mut nq0) = (vec![0u64; k], vec![0u64; k]);
        m.complement(&mut nq1, &q1);
        m.complement(&mut nq0, &q0);
        for j in 0..lb - 1 {
            acc.fill(0);
            nat::mul_to(&mut p, &l2, &a[co(j)], &mut w.s);
            add_in(&mut acc, &p);
            nat::mul_to(&mut p, &nq0, &b[co(j)], &mut w.s);
            add_in(&mut acc, &p);
            if j >= 1 {
                nat::mul_to(&mut p, &nq1, &b[co(j - 1)], &mut w.s);
                add_in(&mut acc, &p);
            }
            m.reduce(&mut out, &acc, w);
            a[co(j)].copy_from_slice(&out);
        }
        l2
    } else {
        let i = lb - 1;
        let mut nq = vec![0u64; k];
        m.complement(&mut nq, &a[co(i)]);
        for j in 0..lb - 1 {
            acc.fill(0);
            nat::mul_to(&mut p, &lc, &a[co(j)], &mut w.s);
            add_in(&mut acc, &p);
            nat::mul_to(&mut p, &nq, &b[co(j)], &mut w.s);
            add_in(&mut acc, &p);
            m.reduce(&mut out, &acc, w);
            a[co(j)].copy_from_slice(&out);
        }
        lc
    };
    a.truncate((lb - 1) * k);
    trim(a, k);
    lam
}

impl super::hgcd::Engine for Limbs {
    type P = Vec<u64>;
    type C = Vec<u64>;

    #[inline]
    fn len(&self, a: &Vec<u64>) -> usize {
        a.len() / self.m.k
    }
    fn zero(&self) -> Vec<u64> {
        Vec::new()
    }
    fn one(&self) -> Vec<u64> {
        let mut o = vec![0u64; self.m.k];
        o[0] = 1;
        o
    }
    fn lead(&self, a: &Vec<u64>) -> Vec<u64> {
        a[a.len() - self.m.k..].to_vec()
    }
    fn add(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let k = self.m.k;
        let (l, s) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let mut r = l.clone();
        for (x, y) in r.chunks_exact_mut(k).zip(s.chunks_exact(k)) {
            self.m.add(x, y);
        }
        trim(&mut r, k);
        r
    }
    fn sub(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let k = self.m.k;
        let mut r = a.clone();
        if r.len() < b.len() {
            r.resize(b.len(), 0);
        }
        for (x, y) in r.chunks_exact_mut(k).zip(b.chunks_exact(k)) {
            self.m.sub(x, y);
        }
        trim(&mut r, k);
        r
    }
    fn neg(&self, a: &Vec<u64>) -> Vec<u64> {
        let mut r = a.clone();
        for x in r.chunks_exact_mut(self.m.k) {
            self.m.neg(x);
        }
        r
    }
    fn mul(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        self.product(a, b)
    }
    fn divrem(&self, a: &Vec<u64>, b: &Vec<u64>) -> (Vec<u64>, Vec<u64>) {
        let k = self.m.k;
        let mut r = a.clone();
        if a.len() < b.len() {
            return (Vec::new(), r);
        }
        let binv = self.m.inv_lead(&b[b.len() - k..]);
        let mut q = vec![0u64; a.len() - b.len() + k];
        divrem_in(
            &self.m,
            &mut r,
            b,
            &binv,
            Some(&mut q),
            &mut self.w.borrow_mut(),
        );
        trim(&mut q, k);
        (q, r)
    }
    fn shr(&self, a: &Vec<u64>, s: usize) -> Vec<u64> {
        a.get(s * self.m.k..).map_or_else(Vec::new, <[u64]>::to_vec)
    }
    fn low(&self, a: &Vec<u64>, s: usize) -> Vec<u64> {
        let mut r = a[..(s * self.m.k).min(a.len())].to_vec();
        trim(&mut r, self.m.k);
        r
    }
    fn shl_add(&self, hi: &Vec<u64>, s: usize, lo: &Vec<u64>) -> Vec<u64> {
        let k = self.m.k;
        if hi.is_empty() {
            return lo.clone();
        }
        let mut r = vec![0u64; (s * k + hi.len()).max(lo.len())];
        r[..lo.len()].copy_from_slice(lo);
        for (x, y) in r[s * k..].chunks_exact_mut(k).zip(hi.chunks_exact(k)) {
            self.m.add(x, y);
        }
        trim(&mut r, k);
        r
    }
    fn cmul(&self, x: &Vec<u64>, y: &Vec<u64>) -> Vec<u64> {
        let mut o = vec![0u64; self.m.k];
        self.m.mul(&mut o, x, y, &mut self.w.borrow_mut());
        o
    }
    fn cneg(&self, x: &Vec<u64>) -> Vec<u64> {
        let mut o = x.clone();
        self.m.neg(&mut o);
        o
    }
    fn cpow(&self, x: &Vec<u64>, e: u64) -> Vec<u64> {
        self.m.pow(x, e)
    }
    fn cone(&self) -> Vec<u64> {
        self.one()
    }
    fn czero(&self) -> Vec<u64> {
        vec![0u64; self.m.k]
    }
    fn euclid(
        &self,
        a: Vec<u64>,
        b: Vec<u64>,
        rec: Option<&mut Vec<(usize, Vec<u64>)>>,
        shift: usize,
    ) -> Vec<u64> {
        self.euclid_in(a, b, rec, shift)
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

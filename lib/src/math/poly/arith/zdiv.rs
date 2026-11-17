//! Pseudo-division and exact division of polynomials over the integers by the schoolbook method
//! on limbs: every coefficient a magnitude and a sign in a slot of a width fixed in advance by a
//! bound, the operations in place, no integer allocated a step.
//!
//! The pseudo-division `l^k a = q d + r` (`k` the quotient's length) runs the classic recurrence
//! `R ← l R - t x^i d` (`t` the top coefficient of `R`) lazily: a coefficient keeps the number of
//! the factors `l` it has had, and takes the ones it misses in one product by a power of `l` when a
//! step changes it, so each step multiplies only the `deg d` coefficients it subtracts from, and
//! the quotient's coefficients `t l^(k-1-s)` take their powers once at the end (the classic method
//! multiplies every coefficient below the top, and the quotient too, by `l` at every step).
//!
//! Past the Kronecker crossover the pseudo-division is first tried with the fewest factors of `l`
//! (FLINT's `fmpz_poly_pseudo_divrem`): a step whose top coefficient `l` divides takes the
//! quotient `t / l` and no factor, so an exact or nearly exact division keeps the size of its
//! operands, and its results take the missing power `l^(k - level)` at the end; it gives up after
//! a few factors, the general case, which the substitution does faster.
//!
//! The exact division takes each quotient coefficient as the top coefficient divided by `l`,
//! `None` at the first one that does not divide, or when a value outgrows Mignotte's bound on the
//! coefficients of a divisor's cofactor, which an exact division never does.

use super::pack;
use crate::math::numbers::{ByteOrder, Number, from_limbs, nat};

/// Signed integers in slots of `w` limbs each: the magnitudes, their lengths (no zero limb at the
/// top) and their signs.
struct Slots {
    w: usize,
    mag: Vec<u64>,
    len: Vec<usize>,
    neg: Vec<bool>,
}

/// The value of a slot or a temporary: a magnitude without zero limbs at the top, and a sign.
#[derive(Clone, Copy)]
struct Val<'a> {
    mag: &'a [u64],
    neg: bool,
}

/// `x` without its zero limbs at the top.
#[inline]
fn trimmed(x: &[u64]) -> usize {
    x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1)
}

/// `a` against `b`, both without zero limbs at the top.
#[inline]
fn cmp(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// A value over in its slot: the bound was too small (exact division: not a divisor).
struct Overflow;

impl Slots {
    /// The integers `c`, each in a slot of `w` limbs (`None` when one does not fit).
    fn new(c: &[Number], w: usize) -> Option<Slots> {
        let mut s = Slots {
            w,
            mag: vec![0; c.len() * w],
            len: vec![0; c.len()],
            neg: vec![false; c.len()],
        };
        let mut buf = Vec::new();
        for (j, x) in c.iter().enumerate() {
            pack::limbs(x, &mut buf);
            if buf.len() > w {
                return None;
            }
            s.mag[j * w..j * w + buf.len()].copy_from_slice(&buf);
            s.len[j] = buf.len();
            s.neg[j] = x.is_negative() && !buf.is_empty();
        }
        Some(s)
    }

    /// The value in slot `j`.
    #[inline]
    fn get(&self, j: usize) -> Val<'_> {
        Val {
            mag: &self.mag[j * self.w..j * self.w + self.len[j]],
            neg: self.neg[j],
        }
    }

    /// Slot `j` times the positive `p` (the limbs of `p`, not zero), with `tmp` for the product.
    #[inline]
    fn scale(&mut self, j: usize, p: &[u64], tmp: &mut Vec<u64>) -> Result<(), Overflow> {
        let xl = self.len[j];
        if xl == 0 || (p.len() == 1 && p[0] == 1) {
            return Ok(());
        }
        let w = self.w;
        tmp.clear();
        tmp.resize(xl + p.len(), 0);
        let x = &self.mag[j * w..j * w + xl];
        let mut scratch = Vec::new();
        nat::mul_to(tmp, x, p, &mut scratch);
        let n = trimmed(tmp);
        if n > w {
            return Err(Overflow);
        }
        self.mag[j * w..j * w + n].copy_from_slice(&tmp[..n]);
        self.len[j] = n;
        Ok(())
    }

    /// Slot `j` minus `t·y`, with `tmp` for the product.
    #[inline]
    fn sub_mul(&mut self, j: usize, t: Val, y: Val, tmp: &mut Vec<u64>) -> Result<(), Overflow> {
        if t.mag.is_empty() || y.mag.is_empty() {
            return Ok(());
        }
        tmp.clear();
        tmp.resize(t.mag.len() + y.mag.len(), 0);
        if y.mag.len() == 1 {
            tmp[t.mag.len()] = nat::addmul_1(&mut tmp[..t.mag.len()], t.mag, y.mag[0]);
        } else {
            nat::mul_to(tmp, t.mag, y.mag, &mut Vec::new());
        }
        let pl = trimmed(tmp);
        // x - (sign) p: the magnitudes added when the signs of x and -t y agree.
        self.add_signed(j, &tmp[..pl], !(t.neg ^ y.neg))
    }

    /// Slot `j` plus the value of magnitude `p` (no zero limbs at the top) and sign `neg`.
    #[inline]
    fn add_signed(&mut self, j: usize, p: &[u64], neg: bool) -> Result<(), Overflow> {
        let w = self.w;
        let xl = self.len[j];
        let pl = p.len();
        if pl > w {
            return Err(Overflow);
        }
        let slot = &mut self.mag[j * w..(j + 1) * w];
        if xl == 0 {
            slot[..pl].copy_from_slice(p);
            self.len[j] = pl;
            self.neg[j] = neg;
            return Ok(());
        }
        if self.neg[j] == neg {
            // The slot's limbs above its length are zero, so the sum runs over the longer.
            let mut n = xl.max(pl);
            let carry = nat::add_assign(&mut slot[..n], p);
            if carry {
                if n >= w {
                    return Err(Overflow);
                }
                slot[n] = 1;
                n += 1;
            }
            self.len[j] = n;
            return Ok(());
        }
        // Different signs: the larger magnitude less the smaller, with its sign.
        match cmp(&slot[..xl], p) {
            core::cmp::Ordering::Less => {
                // p - x over pl limbs, x zero-extended in the slot.
                nat::sub_from(&mut slot[..pl], p);
                self.len[j] = trimmed(&slot[..pl]);
                self.neg[j] = neg;
            }
            _ => {
                nat::sub_assign(&mut slot[..xl], p);
                self.len[j] = trimmed(&slot[..xl]);
                if self.len[j] == 0 {
                    self.neg[j] = false;
                }
            }
        }
        Ok(())
    }

    /// Slot `j` as a number.
    fn number(&self, j: usize) -> Number {
        let v = self.get(j);
        to_number(v.mag, v.neg)
    }
}

/// The number of the magnitude `m` and the sign `neg`.
fn to_number(m: &[u64], neg: bool) -> Number {
    let mut c = from_limbs(m, ByteOrder::LittleEndian);
    c.negative = neg && !c.is_zero();
    c
}

/// The bits of `n ≥ 1`.
#[inline]
fn bits_of(n: usize) -> usize {
    (usize::BITS - n.leading_zeros()) as usize
}

/// The pseudo-quotient of `a` (not empty) by the constant `l`, not zero: `l^(m-1) a` for `a` of
/// `m` coefficients, the pseudo-remainder being 0 (`l^m a = (l^(m-1) a)·l`): the power once, then
/// one product a coefficient.
pub(crate) fn pseudo_by_constant(a: &[Number], l: &Number) -> Vec<Number> {
    let m = a.len();
    if m <= 1 || crate::math::rational::is_one(l) {
        return a.to_vec();
    }
    let lk = l.pow((m - 1) as u32);
    a.iter().map(|c| c * &lk).collect()
}

/// The pseudo-quotient and pseudo-remainder of `a` by `d` (`d` of at least two coefficients, `a`
/// at least as long, no zeros at the top), as `Polynomial::pseudo_div_rem` defines them: `l^k a
/// = q d + r`, `k = a.len() - d.len() + 1`. The remainder has `d.len() - 1` coefficients, not
/// trimmed.
pub(crate) fn pseudo_div_rem(a: &[Number], d: &[Number]) -> (Vec<Number>, Vec<Number>) {
    let (m, n) = (a.len(), d.len());
    debug_assert!(n >= 1 && m >= n);
    let lq = m - n + 1;
    // A negative leading coefficient: divide by -d, then (-1)^(k+1) q and (-1)^k r.
    let flip = d[n - 1].is_negative();
    let mut buf = Vec::new();
    let dv: Vec<(Vec<u64>, bool)> = d
        .iter()
        .map(|c| {
            pack::limbs(c, &mut buf);
            (buf.clone(), (c.is_negative() != flip) && !buf.is_empty())
        })
        .collect();
    let lmag = dv[n - 1].0.clone();
    let hb = super::zmul::max_bits(d);
    let ab = super::zmul::max_bits(a);
    // |R| grows by at most a factor |l| + |t|H/|t| <= 2 max(|l|, H) a step.
    let w = (ab + lq * (hb + 1) + 2).div_ceil(64) + 1;
    let mut r = Slots::new(a, w).expect("the coefficients fit their slots");
    // Powers of |l|, l^0 .. l^lq.
    let unit = lmag.len() == 1 && lmag[0] == 1;
    let mut lp: Vec<Vec<u64>> = vec![vec![1]];
    if !unit {
        let mut scratch = Vec::new();
        for e in 1..=lq {
            let prev = &lp[e - 1];
            let mut p = vec![0u64; prev.len() + lmag.len()];
            nat::mul_to(&mut p, prev, &lmag, &mut scratch);
            p.truncate(trimmed(&p));
            lp.push(p);
        }
    }
    let pw = |e: usize| -> &[u64] { if unit { &lp[0] } else { &lp[e] } };
    let mut level = vec![0usize; m];
    let mut q: Vec<(Vec<u64>, bool)> = vec![(Vec::new(), false); lq];
    let mut tmp = Vec::new();
    for s in 0..lq {
        let i = lq - 1 - s;
        let h = i + n - 1;
        let gap = s - level[h];
        r.scale(h, pw(gap), &mut tmp)
            .unwrap_or_else(|_| unreachable!("the bound holds"));
        level[h] = s;
        let t = r.get(h);
        let (tm, tn) = (t.mag.to_vec(), t.neg);
        if !tm.is_empty() {
            let tv = Val { mag: &tm, neg: tn };
            for j in i..h {
                let gap = s + 1 - level[j];
                r.scale(j, pw(gap), &mut tmp)
                    .and_then(|()| {
                        let (dm, dn) = &dv[j - i];
                        r.sub_mul(j, tv, Val { mag: dm, neg: *dn }, &mut tmp)
                    })
                    .unwrap_or_else(|_| unreachable!("the bound holds"));
                level[j] = s + 1;
            }
        }
        q[i] = (tm, tn);
    }
    // r_j = R_j l^(k - level), q_i = t l^(k - 1 - s) for i = k - 1 - s.
    let k_odd = lq % 2 == 1;
    let rem: Vec<Number> = (0..n - 1)
        .map(|j| {
            let gap = lq - level[j];
            r.scale(j, pw(gap), &mut tmp)
                .unwrap_or_else(|_| unreachable!("the bound holds"));
            let mut c = r.number(j);
            if flip && k_odd && !c.is_zero() {
                c.negative = !c.negative;
            }
            c
        })
        .collect();
    let mut scratch = Vec::new();
    let quo: Vec<Number> = q
        .into_iter()
        .enumerate()
        .map(|(i, (tm, tn))| {
            if tm.is_empty() {
                return Number::default();
            }
            // Step s = k - 1 - i took l^(k - 1 - s) = l^i.
            let p = pw(i);
            let mut c = if p.len() == 1 && p[0] == 1 {
                to_number(&tm, tn)
            } else {
                let mut v = vec![0u64; tm.len() + p.len()];
                nat::mul_to(&mut v, &tm, p, &mut scratch);
                to_number(&v, tn)
            };
            // (-1)^(k+1): negated when k is even.
            if flip && !k_odd && !c.is_zero() {
                c.negative = !c.negative;
            }
            c
        })
        .collect();
    (quo, rem)
}

/// [`pseudo_div_rem`] when at most `max_d` of its steps need a factor `l` (see [`pseudo_lazy`]),
/// `None` otherwise (after at most `max_d + 1` steps that do): for a caller with a faster method
/// for the general case, which an exact or nearly exact division would make far too large.
pub(crate) fn pseudo_div_rem_lazy(
    a: &[Number],
    d: &[Number],
    max_d: usize,
) -> Option<(Vec<Number>, Vec<Number>)> {
    pseudo_lazy(a, d, max_d)
}

/// A signed integer being reduced: its magnitude, without zero limbs at the top, and its sign.
struct Coef {
    mag: Vec<u64>,
    neg: bool,
}

impl Coef {
    /// `self · p` for the positive `p` (`[1]` for 1).
    fn scale(&mut self, p: &[u64], tmp: &mut Vec<u64>) {
        if self.mag.is_empty() || (p.len() == 1 && p[0] == 1) {
            return;
        }
        tmp.clear();
        tmp.resize(self.mag.len() + p.len(), 0);
        nat::mul_to(tmp, &self.mag, p, &mut Vec::new());
        tmp.truncate(trimmed(tmp));
        core::mem::swap(&mut self.mag, tmp);
    }

    /// `self - t·y` for the values `(t, tneg)` and `(y, yneg)`.
    fn sub_mul(&mut self, t: &[u64], tneg: bool, y: &[u64], yneg: bool, tmp: &mut Vec<u64>) {
        if t.is_empty() || y.is_empty() {
            return;
        }
        tmp.clear();
        tmp.resize(t.len() + y.len(), 0);
        if y.len() == 1 {
            tmp[t.len()] = nat::addmul_1(&mut tmp[..t.len()], t, y[0]);
        } else {
            nat::mul_to(tmp, t, y, &mut Vec::new());
        }
        tmp.truncate(trimmed(tmp));
        let pneg = !(tneg ^ yneg);
        let p = &tmp[..];
        if self.mag.is_empty() {
            self.mag.extend_from_slice(p);
            self.neg = pneg;
            return;
        }
        if self.neg == pneg {
            let n = self.mag.len().max(p.len());
            self.mag.resize(n, 0);
            if nat::add_assign(&mut self.mag, p) {
                self.mag.push(1);
            }
            return;
        }
        match cmp(&self.mag, p) {
            core::cmp::Ordering::Less => {
                let n = p.len();
                self.mag.resize(n, 0);
                nat::sub_from(&mut self.mag, p);
                self.neg = pneg;
            }
            _ => {
                nat::sub_assign(&mut self.mag, p);
            }
        }
        let n = trimmed(&self.mag);
        self.mag.truncate(n);
        if n == 0 {
            self.neg = false;
        }
    }
}

/// The powers of `|l|`, computed as they are asked for.
struct Powers {
    l: Vec<u64>,
    p: Vec<Option<Vec<u64>>>,
}

impl Powers {
    /// `|l|^e`.
    fn get(&mut self, e: usize) -> &[u64] {
        if self.p.len() <= e {
            self.p.resize(e + 1, None);
        }
        if self.p[e].is_none() {
            let v = if e == 0 {
                vec![1]
            } else {
                // From the largest power below already known, by products of l.
                let mut b = e - 1;
                while b > 0 && self.p[b].is_none() {
                    b -= 1;
                }
                let mut cur = if b == 0 {
                    vec![1u64]
                } else {
                    self.p[b].clone().expect("known")
                };
                let mut scratch = Vec::new();
                for _ in b..e {
                    let mut t = vec![0u64; cur.len() + self.l.len()];
                    nat::mul_to(&mut t, &cur, &self.l, &mut scratch);
                    t.truncate(trimmed(&t));
                    cur = t;
                }
                cur
            };
            self.p[e] = Some(v);
        }
        self.p[e].as_deref().expect("set")
    }
}

/// The pseudo-division by the schoolbook method on limbs with the fewest factors of `l`, as
/// FLINT's `fmpz_poly_pseudo_divrem` computes it: `l^D a = Q d + R` with `D` the number of steps
/// whose top coefficient `l` does not divide; a step whose top coefficient `t` it divides takes
/// `t / l` into the quotient and needs no factor (so that an exact division, or one whose
/// remainder takes few steps, keeps the size of its operands). Every coefficient keeps the level
/// `D` it was last brought to and takes the factors it misses in one product by a power of `l`
/// when a step changes it; at the end each one is brought to the level `k`, `l^(k - level)` (the
/// powers in increasing order, each from the one before), which gives the pseudo-division's
/// `l^k a = q d + r` exactly. `None` once more than `max_d` steps need a factor.
fn pseudo_lazy(a: &[Number], d: &[Number], max_d: usize) -> Option<(Vec<Number>, Vec<Number>)> {
    lazy_core(a, d, max_d, true).map(|(q, r, _)| (q, r))
}

/// The remainder of `a` by `d` (at least two coefficients, `a` at least as long) over the
/// rationals as `r / |l|^D`: the pseudo-division with the fewest factors of `l` (see
/// [`pseudo_lazy`]), its remainder left at the level `D` it reached (not `k`), `d - 1`
/// coefficients, not trimmed.
pub(crate) fn rem_fewest(a: &[Number], d: &[Number]) -> (Vec<Number>, usize) {
    let (_, r, big_d) = lazy_core(a, d, usize::MAX, false).expect("no limit on the factors of l");
    (r, big_d)
}

/// The steps of [`pseudo_lazy`]; the results brought to the level `k` when `to_k` (with the
/// pseudo-division's signs for a negative `l`), otherwise the remainder to the level `D` reached
/// and its magnitude's signs as they are (`|l|^D a ≡ r` modulo `d`), the quotient left empty.
fn lazy_core(
    a: &[Number],
    d: &[Number],
    max_d: usize,
    to_k: bool,
) -> Option<(Vec<Number>, Vec<Number>, usize)> {
    let (m, n) = (a.len(), d.len());
    debug_assert!(n >= 1 && m >= n);
    let lq = m - n + 1;
    // A negative leading coefficient: divide by -d, then (-1)^(k+1) q and (-1)^k r.
    let flip = d[n - 1].is_negative();
    let mut buf = Vec::new();
    let dv: Vec<(Vec<u64>, bool)> = d
        .iter()
        .map(|c| {
            pack::limbs(c, &mut buf);
            (buf.clone(), (c.is_negative() != flip) && !buf.is_empty())
        })
        .collect();
    let lmag = dv[n - 1].0.clone();
    let unit = lmag.len() == 1 && lmag[0] == 1;
    let mut r: Vec<Coef> = a
        .iter()
        .map(|c| {
            pack::limbs(c, &mut buf);
            Coef {
                mag: buf.clone(),
                neg: c.is_negative() && !buf.is_empty(),
            }
        })
        .collect();
    let mut pw = Powers {
        l: lmag.clone(),
        p: Vec::new(),
    };
    let mut level = vec![0usize; m];
    let mut big_d = 0usize;
    let mut q: Vec<(Coef, usize)> = Vec::with_capacity(lq);
    let (mut tmp, mut qt, mut rt, mut scratch) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for s in 0..lq {
        let i = lq - 1 - s;
        let h = i + n - 1;
        let gap = big_d - level[h];
        r[h].scale(pw.get(gap), &mut tmp);
        level[h] = big_d;
        let t = core::mem::replace(
            &mut r[h],
            Coef {
                mag: Vec::new(),
                neg: false,
            },
        );
        if t.mag.is_empty() {
            q.push((t, big_d));
            continue;
        }
        // t / l when l divides t: no factor; otherwise one more factor of l for everything.
        let tau = if unit {
            Some(t.mag.clone())
        } else {
            nat::div_rem_into(&mut qt, &mut rt, &t.mag, &lmag, &mut scratch);
            rt.is_empty().then(|| qt.clone())
        };
        let tv = match tau {
            Some(v) => v,
            None => {
                big_d += 1;
                if big_d > max_d {
                    return None;
                }
                t.mag.clone()
            }
        };
        for j in i..h {
            let gap = big_d - level[j];
            r[j].scale(pw.get(gap), &mut tmp);
            level[j] = big_d;
            let (dm, dn) = &dv[j - i];
            r[j].sub_mul(&tv, t.neg, dm, *dn, &mut tmp);
        }
        q.push((
            Coef {
                mag: tv,
                neg: t.neg,
            },
            big_d,
        ));
    }
    if !to_k {
        // Dividing by -d when l < 0 leaves the remainder as it is: |l|^D a = (-q)(d) + r.
        let mut rem = Vec::with_capacity(n - 1);
        for j in 0..n - 1 {
            let e = big_d - level[j];
            r[j].scale(pw.get(e), &mut tmp);
            rem.push(to_number(&r[j].mag, r[j].neg));
        }
        return Some((Vec::new(), rem, big_d));
    }
    // q was pushed from the top: q[s] is the coefficient of x^(lq - 1 - s).
    q.reverse();
    // Every coefficient to the level k: the factor l^(k - level), the powers in increasing order.
    let k_odd = lq % 2 == 1;
    let mut outs: Vec<(usize, usize, bool)> = Vec::with_capacity(lq + n - 1);
    for (i, (_, lv)) in q.iter().enumerate() {
        outs.push((lq - lv, i, true));
    }
    for (j, lv) in level.iter().enumerate().take(n - 1) {
        outs.push((lq - lv, j, false));
    }
    outs.sort_unstable();
    let mut quo = vec![Number::default(); lq];
    let mut rem = vec![Number::default(); n - 1];
    let mut cur: Vec<u64> = vec![1];
    let mut cur_e = 0usize;
    for (e, idx, is_q) in outs {
        let c = if is_q { &mut q[idx].0 } else { &mut r[idx] };
        if c.mag.is_empty() {
            continue;
        }
        if !unit && e > cur_e {
            let step = pw.get(e - cur_e).to_vec();
            let mut t = vec![0u64; cur.len() + step.len()];
            nat::mul_to(&mut t, &cur, &step, &mut scratch);
            t.truncate(trimmed(&t));
            cur = t;
            cur_e = e;
        }
        c.scale(&cur, &mut tmp);
        let mut v = to_number(&c.mag, c.neg);
        if flip && !v.is_zero() && (if is_q { !k_odd } else { k_odd }) {
            v.negative = !v.negative;
        }
        if is_q {
            quo[idx] = v;
        } else {
            rem[idx] = v;
        }
    }
    Some((quo, rem, big_d))
}

/// The quotient of `a` by `d` (`d` not empty, `a` at least as long, no zeros at the top) when it
/// is exact over the integers, `None` otherwise.
pub(crate) fn div_exact(a: &[Number], d: &[Number]) -> Option<Vec<Number>> {
    let (m, n) = (a.len(), d.len());
    debug_assert!(n >= 1 && m >= n);
    let lq = m - n + 1;
    let mut buf = Vec::new();
    let dv: Vec<(Vec<u64>, bool)> = d
        .iter()
        .map(|c| {
            pack::limbs(c, &mut buf);
            (buf.clone(), c.is_negative() && !buf.is_empty())
        })
        .collect();
    let (lmag, lneg) = dv[n - 1].clone();
    let unit = lmag.len() == 1 && lmag[0] == 1;
    let hb = super::zmul::max_bits(d);
    let ab = super::zmul::max_bits(a);
    // Mignotte: a quotient's coefficients are below 2^(deg q) ‖a‖₂; the values of R below
    // |a| + n H |q|.
    let qbits = lq + ab + bits_of(m) / 2 + 1;
    let w = (ab.max(qbits + hb + bits_of(n)) + 2).div_ceil(64) + 1;
    let mut r = Slots::new(a, w)?;
    let mut quo = vec![Number::default(); lq];
    let mut tmp = Vec::new();
    let (mut qt, mut rt, mut scratch) = (Vec::new(), Vec::new(), Vec::new());
    for i in (0..lq).rev() {
        let h = i + n - 1;
        let top = r.get(h);
        if top.mag.is_empty() {
            continue;
        }
        let tneg = top.neg ^ lneg;
        if unit {
            qt.clear();
            qt.extend_from_slice(top.mag);
        } else {
            nat::div_rem_into(&mut qt, &mut rt, top.mag, &lmag, &mut scratch);
            if !rt.is_empty() {
                return None;
            }
        }
        if qt.len() * 64 > qbits + 64 {
            return None;
        }
        let tv = Val {
            mag: &qt,
            neg: tneg,
        };
        for j in i..h {
            let (dm, dn) = &dv[j - i];
            if r.sub_mul(j, tv, Val { mag: dm, neg: *dn }, &mut tmp)
                .is_err()
            {
                return None;
            }
        }
        quo[i] = to_number(&qt, tneg);
    }
    (0..n - 1).all(|j| r.len[j] == 0).then_some(quo)
}

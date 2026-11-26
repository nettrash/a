//! Real root isolation of polynomials over the integers: Descartes's rule of signs with Taylor
//! shifts (the Vincent–Collins–Akritas bisection, as Rouillier and Zimmermann and FLINT's
//! `fmpz_poly_isolate_real_roots` run it) and the continued fractions of Vincent, Akritas and
//! Strzeboński (VAS, with the local-max-quadratic bound), a choice between them by the input,
//! counting by Sturm's sequences when they collapse, then the refinement of an isolating
//! interval to a correctly rounded float by Newton's method on dyadic points in fixed point.

use super::fixed::{Fixed, Pt, RECT_N, RECT_W};
use super::flat::{Approx, Flat};
use super::yun;
use crate::math::float::{BigFloat, Round};
use crate::math::numbers::Number;
use crate::math::poly::Polynomial;
use crate::math::rational::Rational;
use core::cell::OnceCell;
use core::cmp::Ordering;

/// An isolated real root: exactly a rational, or the only root in an open interval whose ends are
/// not roots.
#[derive(Clone, Debug)]
pub(crate) enum Iso {
    Point(Rational),
    Open(Rational, Rational),
}

impl Iso {
    fn lo(&self) -> &Rational {
        match self {
            Iso::Point(r) => r,
            Iso::Open(a, _) => a,
        }
    }
    fn hi(&self) -> &Rational {
        match self {
            Iso::Point(r) => r,
            Iso::Open(_, b) => b,
        }
    }
    fn neg(self) -> Iso {
        match self {
            Iso::Point(r) => Iso::Point(-r),
            Iso::Open(a, b) => Iso::Open(-b, -a),
        }
    }
}

/// The method of isolation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Descartes,
    ContinuedFractions,
    /// For each of the positive and the negative roots, continued fractions where the bound
    /// of the roots is large or the degree small, else Descartes's rule with its deep nodes
    /// by continued fractions (see `positive`).
    Auto,
}

/// The method `real_root_intervals`, `count_real_roots` and `real_roots` take.
pub(crate) const DEFAULT: Method = Method::Auto;

/// For `Method::Auto`: continued fractions when the bound `2^s` of the positive roots has
/// `s ≥ CF_BOUND` (roots far apart in a long interval, as Wilkinson's, `s ≥ 8` from the degree
/// 10; the bound is at most 5 for Chebyshev's polynomials to the degree 200 and 4 for random
/// ones); by measurement. (Below the degree 100 continued fractions are also faster on random
/// polynomials, by 1.1 to 1.9 times, but slower on Chebyshev's, 1.5 times at 20 and 3 at 50, so
/// that the degree alone does not choose.)
const CF_BOUND: i64 = 7;

/// The positive roots of the square-free `f` (`f(0) ≠ 0`) by the method: intervals in no
/// particular order (increasing for Descartes's rule alone), or only their number.
fn positive(f: &[Number], method: Method, out: &mut Vec<Iso>, count_only: bool) -> usize {
    match method {
        Method::Descartes => vca_positive(f, out, count_only, false),
        Method::ContinuedFractions => cf_positive(f, out, count_only),
        Method::Auto => {
            let big = root_bound(&mags_of(f)).is_some_and(|s| s >= CF_BOUND);
            if big {
                cf_positive(f, out, count_only)
            } else {
                vca_positive(f, out, count_only, true)
            }
        }
    }
}

/// `2^k` as a number.
fn pow2(k: u64) -> Number {
    Number::from(1u8) << k as usize
}

/// The exponent `b` of an upper bound `2^b` of the positive roots of the polynomial of the signs
/// and the bit lengths `m` (sign, bits of the magnitude, the constant first), with room: every
/// positive root is below `2^b` strictly. `None` when there is no positive root (no sign change).
/// Kioustelidis's bound, `2·max (|aᵢ|/aₙ)^(1/(n−i))` over the `aᵢ` of the sign opposite to `aₙ`.
fn positive_bound(m: &[(i32, u64)]) -> Option<i64> {
    let n = m.len() - 1;
    let (sn, bn) = m[n];
    let mut best: Option<i64> = None;
    for (i, &(s, b)) in m[..n].iter().enumerate() {
        if s != 0 && s != sn {
            // |aᵢ|/|aₙ| < 2^(b − bn + 1).
            let e = (b as i64 - bn as i64 + 1).div_euclid(1);
            let k = (n - i) as i64;
            let t = e.div_euclid(k) + i64::from(e.rem_euclid(k) != 0);
            best = Some(best.map_or(t, |x: i64| x.max(t)));
        }
    }
    best.map(|t| t + 2)
}

/// The exponent `e` of an upper bound `2^e` of the positive roots by the local-max-quadratic bound
/// of Akritas, Strzeboński and Vigklas, over the signs and bit lengths `m` (the constant first),
/// rounded up; `None` when there is no positive root.
fn lmq_bound(m: &[(i32, u64)]) -> Option<i64> {
    let n = m.len() - 1;
    let sn = m[n].0;
    let mut used = vec![0i64; n + 1];
    let mut best: Option<i64> = None;
    for i in (0..n).rev() {
        let (s, b) = m[i];
        if s == 0 || s == sn {
            continue;
        }
        let mut low: Option<(i64, usize)> = None;
        for j in i + 1..=n {
            let (sj, bj) = m[j];
            if sj != sn {
                continue;
            }
            // (2^used · |aᵢ| / aⱼ)^(1/(j−i)) < 2^((used + b − bj + 1)/(j − i)).
            let e = used[j] + 1 + b as i64 - bj as i64 + 1;
            let k = (j - i) as i64;
            let t = e.div_euclid(k) + i64::from(e.rem_euclid(k) != 0);
            if low.is_none_or(|(x, _)| t < x) {
                low = Some((t, j));
            }
        }
        if let Some((t, j)) = low {
            used[j] += 1;
            best = Some(best.map_or(t, |x: i64| x.max(t)));
        }
    }
    best
}

/// The signs and exact bit lengths of the coefficients of `q`.
fn mags(q: &Flat) -> Vec<(i32, u64)> {
    (0..q.len)
        .map(|i| {
            let s = q.sign(i);
            let x = q.coeff(i);
            let b = if s >= 0 {
                bits_of(x)
            } else {
                let mut y = x.to_vec();
                let mut carry = true;
                for l in y.iter_mut() {
                    let (v, c) = (!*l).overflowing_add(carry as u64);
                    *l = v;
                    carry = c;
                }
                bits_of(&y)
            };
            (s, b)
        })
        .collect()
}

fn bits_of(x: &[u64]) -> u64 {
    for k in (0..x.len()).rev() {
        if x[k] != 0 {
            return k as u64 * 64 + 64 - x[k].leading_zeros() as u64;
        }
    }
    0
}

/// The signs and bit lengths of numbers.
fn mags_of(c: &[Number]) -> Vec<(i32, u64)> {
    c.iter()
        .map(|x| {
            let s = if x.is_zero() {
                0
            } else if x.is_negative() {
                -1
            } else {
                1
            };
            (s, x.bits())
        })
        .collect()
}

/// A node of the bisection: the polynomial `q` of the interval `(c/2^k, (c+1)/2^k)` (of the
/// scaled variable), mapped to `(0, 1)`, or an exact root found at a midpoint. With `shift`, the
/// node's polynomial is `q(x + 1)` (divided by `x` with `drop`), computed only if the node is
/// split or its test needs it exactly.
enum Node {
    Poly {
        q: Flat,
        c: Number,
        k: u64,
        shift: bool,
        drop: bool,
    },
    Point(Number, u64),
}

/// Makes the polynomial of a node exact: the shift it waits for, done.
fn settle(q: &mut Flat, shift: &mut bool, drop: bool) {
    if *shift {
        let n = q.len as u64 - 1;
        q.room(n + 1);
        q.taylor1();
        if drop {
            q.drop_constant();
        }
        q.remove_pow2();
        q.room(0);
        *shift = false;
    }
}

/// The sign variations of the Taylor shift of the reversal of a node's polynomial, as far as 2:
/// in double precision with a bound of its error when that decides every sign, exactly
/// otherwise.
fn descartes_test(q: &mut Flat, shift: &mut bool, drop: bool, a: &mut Approx) -> u32 {
    let n = q.len as u64 - 1;
    let ops = if *shift { 2 } else { 1 };
    if q.approx(ops, a) {
        if *shift {
            a.taylor1();
            if drop {
                a.v.remove(0);
            }
        }
        if let Some(v) = a.test(if *shift { n } else { 0 }) {
            return v;
        }
    } else if *shift {
        // Too long for two shifts in double precision: the exact shift, then one.
        settle(q, shift, drop);
        if q.approx(1, a)
            && let Some(v) = a.test(0)
        {
            return v;
        }
    }
    settle(q, shift, drop);
    let n = q.len as u64 - 1;
    let mut t = q.clone();
    t.reverse();
    t.room(n + 1);
    t.taylor1_variations()
}

/// `c·2^(s−k)` as a rational.
fn dyadic(c: &Number, s: i64, k: u64) -> Rational {
    let e = s - k as i64;
    if e >= 0 {
        Rational::from(c << e as usize)
    } else {
        Rational::new(c.clone(), pow2(e.unsigned_abs()))
    }
}

/// The exponent of a bound `2^b` of the positive roots of `f`, every positive root below it
/// strictly: the smaller of Kioustelidis's and the local-max-quadratic bound.
fn root_bound(m: &[(i32, u64)]) -> Option<i64> {
    let k = positive_bound(m)?;
    Some(match lmq_bound(m) {
        Some(l) => k.min(l + 1),
        None => k,
    })
}

/// The depth of the bisection from which a node of a polynomial of the degree `n` goes on by
/// continued fractions: past the depths that separate the roots of random and Chebyshev's
/// polynomials (about `2·log2 n`).
fn deep_level(n: usize) -> u64 {
    2 * u64::from(n.max(2).ilog2()) + DEEP_EXTRA
}

/// The levels past `2·log2 n` of `deep_level`, by measurement.
const DEEP_EXTRA: u64 = 16;

/// The positive roots of the square-free `f` (`f(0) ≠ 0`, degree at least 1) by the bisection of
/// Descartes's rule: the isolating intervals in increasing order, or, with `count_only`, only
/// their number.
fn vca_positive(f: &[Number], out: &mut Vec<Iso>, count_only: bool, hybrid: bool) -> usize {
    let n = f.len() - 1;
    let Some(s) = root_bound(&mags_of(f)) else {
        return 0;
    };
    let mut q = Flat::from_numbers(f, s.unsigned_abs() * n as u64);
    if s >= 0 {
        q.scale_pow2(s as u64);
    } else {
        q.scale_inv_pow2(s.unsigned_abs());
    }
    q.remove_pow2();
    q.room(0);
    let deep = if hybrid { deep_level(n) } else { u64::MAX };
    let mut ap = Approx::default();
    let mut found = 0;
    let mut stack = vec![Node::Poly {
        q,
        c: Number::from(0u8),
        k: 0,
        shift: false,
        drop: false,
    }];
    while let Some(node) = stack.pop() {
        let (mut q, c, k, mut shift, drop) = match node {
            Node::Point(c, k) => {
                found += 1;
                if !count_only {
                    out.push(Iso::Point(dyadic(&c, s, k)));
                }
                continue;
            }
            Node::Poly {
                q,
                c,
                k,
                shift,
                drop,
            } => (q, c, k, shift, drop),
        };
        if q.len - usize::from(shift && drop) <= 1 {
            continue;
        }
        let v = descartes_test(&mut q, &mut shift, drop, &mut ap);
        if v == 0 {
            continue;
        }
        if v == 1 {
            found += 1;
            if !count_only {
                out.push(Iso::Open(dyadic(&c, s, k), dyadic(&(&c + 1u8), s, k)));
            }
            continue;
        }
        settle(&mut q, &mut shift, drop);
        let n = q.len as u64 - 1;
        if k >= deep {
            // Deep in the tree, as for roots very close together (Mignotte's): the node's roots
            // by continued fractions, whose shifts by the bounds of the roots jump the levels
            // the bisection would take one by one. The roots in (0, 1) of q are those of
            // (y + 1)ⁿ·q(1/(y + 1)) in (0, ∞), and x = 1/(y + 1) is 2^(s−k)·(c·y + c + 1)/(y + 1)
            // in the scaled variable's interval.
            let mut t = q;
            t.reverse();
            t.room(n + 1);
            t.taylor1();
            if t.sign(0) == 0 {
                // q(1) = 0: the right end, a root found at a midpoint above.
                t.drop_constant();
            }
            t.remove_pow2();
            t.room(0);
            let e = s - k as i64;
            let c1 = &c + 1u8;
            let m = if e >= 0 {
                let p = pow2(e as u64);
                Mobius {
                    a: &c * &p,
                    b: &c1 * &p,
                    c: Number::from(1u8),
                    d: Number::from(1u8),
                }
            } else {
                let p = pow2(e.unsigned_abs());
                Mobius {
                    a: c,
                    b: c1,
                    c: p.clone(),
                    d: p,
                }
            };
            found += cf_run(vec![(t, m)], out, count_only);
            continue;
        }
        let mut l = q;
        l.room(n);
        l.scale_half();
        let mid = l.sum_is_zero();
        l.remove_pow2();
        l.room(0);
        let c2 = &c << 1usize;
        let c3 = &c2 + 1u8;
        stack.push(Node::Poly {
            q: l.clone(),
            c: c3.clone(),
            k: k + 1,
            shift: true,
            drop: mid,
        });
        if mid {
            stack.push(Node::Point(c3, k + 1));
        }
        stack.push(Node::Poly {
            q: l,
            c: c2,
            k: k + 1,
            shift: false,
            drop: false,
        });
    }
    found
}

/// A Möbius transformation `(a·x + b)/(c·x + d)` with `a, b, c, d ≥ 0`, `d > 0`.
#[derive(Clone)]
struct Mobius {
    a: Number,
    b: Number,
    c: Number,
    d: Number,
}

impl Mobius {
    fn at(&self, x: &Number) -> Rational {
        Rational::new(&(&self.a * x) + &self.b, &(&self.c * x) + &self.d)
    }
    fn at0(&self) -> Rational {
        Rational::new(self.b.clone(), self.d.clone())
    }
}

/// The positive roots of the square-free `f` (`f(0) ≠ 0`, degree at least 1) by continued
/// fractions (VAS): isolating intervals in no particular order, or only their number.
fn cf_positive(f: &[Number], out: &mut Vec<Iso>, count_only: bool) -> usize {
    let q = Flat::from_numbers(f, 0);
    let one = Number::from(1u8);
    let zero = Number::from(0u8);
    let m = Mobius {
        a: one.clone(),
        b: zero.clone(),
        c: zero,
        d: one,
    };
    cf_run(vec![(q, m)], out, count_only)
}

/// The continued fractions from the nodes of `stack`, each a polynomial whose positive roots
/// the Möbius transformation maps to those wanted.
fn cf_run(mut stack: Vec<(Flat, Mobius)>, out: &mut Vec<Iso>, count_only: bool) -> usize {
    let mut found = 0;
    let mut emit = |iso: Iso, out: &mut Vec<Iso>| {
        found += 1;
        if !count_only {
            out.push(iso);
        }
    };
    let interval = |q: &Flat, m: &Mobius| -> Iso {
        let lo = m.at0();
        let hi = if m.c.is_zero() {
            let e = positive_bound(&mags(q)).expect("a positive root");
            if e >= 0 {
                m.at(&pow2(e as u64))
            } else {
                // (a·2^e + b)/d with e < 0: (a + b·2^−e)/(d·2^−e).
                let t = pow2(e.unsigned_abs());
                Rational::new(&m.a + &(&m.b * &t), &m.d * &t)
            }
        } else {
            Rational::new(m.a.clone(), m.c.clone())
        };
        if lo <= hi {
            Iso::Open(lo, hi)
        } else {
            Iso::Open(hi, lo)
        }
    };
    while let Some((mut q, mut m)) = stack.pop() {
        if q.len <= 1 {
            continue;
        }
        let v = q.variations();
        if v == 0 {
            continue;
        }
        if v == 1 {
            emit(interval(&q, &m), out);
            continue;
        }
        // Shift by the lower bound of the positive roots when it is at least 1.
        let mut rev = mags(&q);
        rev.reverse();
        if let Some(e) = lmq_bound(&rev)
            && e <= 0
        {
            let s = e.unsigned_abs();
            let n = q.len as u64 - 1;
            q.room((s + 1) * n + 1);
            q.taylor_pow2(s);
            m.b += &(&m.a << s as usize);
            m.d += &(&m.c << s as usize);
            q.remove_pow2();
            q.room(0);
            if q.sign(0) == 0 {
                emit(Iso::Point(m.at0()), out);
                q.drop_constant();
            }
            let v = q.variations();
            if v == 0 {
                continue;
            }
            if v == 1 {
                emit(interval(&q, &m), out);
                continue;
            }
        }
        let n = q.len as u64 - 1;
        let v = q.variations();
        // The roots above 1: q(x + 1), M(x + 1).
        let mut q1 = q.clone();
        q1.room(n + 1);
        q1.taylor1();
        let m1 = Mobius {
            a: m.a.clone(),
            b: &m.a + &m.b,
            c: m.c.clone(),
            d: &m.c + &m.d,
        };
        let root1 = q1.sign(0) == 0;
        if root1 {
            emit(Iso::Point(m1.at0()), out);
            q1.drop_constant();
        }
        q1.remove_pow2();
        q1.room(0);
        let v1 = q1.variations();
        // The roots in (0, 1): (x + 1)ⁿ·q(1/(x + 1)), M(1/(x + 1)); none when the shift kept every
        // variation (Var(q(x+1)) + Var((x+1)ⁿ q(1/(x+1))) ≤ Var(q)).
        if v1 + u32::from(root1) < v {
            let mut q2 = q;
            q2.reverse();
            q2.room(n + 1);
            q2.taylor1();
            if root1 {
                q2.drop_constant();
            }
            q2.remove_pow2();
            q2.room(0);
            let m2 = Mobius {
                a: m.b.clone(),
                b: &m.a + &m.b,
                c: m.d.clone(),
                d: &m.c + &m.d,
            };
            stack.push((q2, m2));
        }
        if v1 > 0 {
            stack.push((q1, m1));
        }
    }
    found
}

/// The real roots of the square-free `f` of a degree at least 1 with `f(0) ≠ 0`: isolating
/// intervals in increasing order.
pub(crate) fn isolate_squarefree(f: &[Number], method: Method) -> Vec<Iso> {
    if f.len() == 2 {
        // One rational root, exactly.
        return vec![Iso::Point(Rational::new(-&f[0], f[1].clone()))];
    }
    let neg: Vec<Number> = f
        .iter()
        .enumerate()
        .map(|(i, c)| if i % 2 == 1 { -c } else { c.clone() })
        .collect();
    let mut pos = Vec::new();
    let mut negs = Vec::new();
    positive(f, method, &mut pos, false);
    positive(&neg, method, &mut negs, false);
    // Continued fractions (also below the deep nodes of the bisection) give them in no order.
    pos.sort_by(|x, y| x.lo().cmp(y.lo()));
    negs.sort_by(|x, y| x.lo().cmp(y.lo()));
    let mut out: Vec<Iso> = negs.into_iter().rev().map(Iso::neg).collect();
    out.extend(pos);
    out
}

/// The largest degree of the first Sturm remainder `−rem(f, f')` for which counting takes
/// Sturm's sequence (otherwise of the degree `n − 2`), and the least degree of `f` that tries it.
const STURM_TAIL: usize = 16;
const STURM_FROM: usize = 12;

/// The polynomial divided by its content, which is positive: the signs stay.
fn divide_content(r: Polynomial<Number>) -> Polynomial<Number> {
    let g = r.content();
    if crate::math::rational::is_one(&g) {
        return r;
    }
    r.map(|c| c / &g)
}

/// The pseudo-remainder `l^(n−m+1)·a mod b` (`l` the leading coefficient of `b`, `n ≥ m` the
/// degrees) for a short `b`: the `m + 1` coefficients under the top in a window, each step
/// `R ← l·R − q·x^k·b` on the window alone, and each coefficient of `a` taken in times the power
/// of `l` the steps before it would have given it: `O(n·m)` operations, not the schoolbook's
/// `O(n²)` products by `l`.
fn prem_short(a: &[Number], b: &[Number]) -> Vec<Number> {
    let n = a.len() - 1;
    let m = b.len() - 1;
    if n < m {
        return a.to_vec();
    }
    let l = &b[m];
    // The window holds R[k−m..=k], the constant first; the power is l^(n−k).
    let mut win: Vec<Number> = a[n - m..].to_vec();
    let mut pw = Number::from(1u8);
    let mut k = n;
    loop {
        // The step at k: q = R[k], R ← l·R − q·x^(k−m)·b on the window.
        let q = win[m].clone();
        for (j, x) in win.iter_mut().enumerate().take(m) {
            *x = &(&*x * l) - &(&q * &b[j]);
        }
        if k == m {
            win.truncate(m);
            return win;
        }
        k -= 1;
        // Move the window down: R[k−m] enters with l^(n−k).
        pw = &pw * l;
        win.pop();
        win.insert(0, &a[k - m] * &pw);
    }
}

/// The number of real roots of the square-free `f` (a degree at least 2) by Sturm's theorem,
/// when its sequence collapses at once: the first remainder `−rem(f, f')` of a degree at most
/// `STURM_TAIL` (as for `xⁿ` plus a polynomial of a low degree, Mignotte's), so that the rest of
/// the sequence is short. `None` otherwise, after one pseudo-division of `O(n)` operations.
/// The sequence is of pseudo-remainders with the signs of the true remainders (the pseudo
/// remainder's factor `lᵏ` taken out by its sign), each divided by its content.
fn sturm_count(f: &[Number]) -> Option<usize> {
    let n = f.len() - 1;
    let p = Polynomial::new(f.to_vec());
    let dp = p.derivative();
    let mut seq = vec![p, dp];
    loop {
        let k = seq.len();
        let (a, b) = (&seq[k - 2], &seq[k - 1]);
        let db = b.degree().expect("not zero");
        if db == 0 {
            break;
        }
        let e = a.degree().expect("not zero") - db + 1;
        let r = if db <= STURM_TAIL {
            Polynomial::new(prem_short(&a.c, &b.c))
        } else {
            a.pseudo_div_rem(b).1
        };
        // Square-free: the last remainder is a nonzero constant, never 0.
        r.degree()?;
        if k == 2 && r.degree().expect("not zero") > STURM_TAIL.min(n / 4) {
            return None;
        }
        let flip = b.leading().expect("not zero").is_negative() && e % 2 == 1;
        let r = if flip { r } else { -r };
        seq.push(divide_content(r));
    }
    let variations = |minus: bool| {
        let mut v = 0;
        let mut last = 0;
        for q in &seq {
            let mut s = if q.leading().expect("not zero").is_negative() {
                -1
            } else {
                1
            };
            if minus && q.degree().expect("not zero") % 2 == 1 {
                s = -s;
            }
            if last != 0 && s != last {
                v += 1;
            }
            last = s;
        }
        v
    };
    Some(variations(true) - variations(false))
}

/// The number of real roots of the square-free `f` of a degree at least 1, `f(0) ≠ 0`: by
/// Sturm's theorem when its sequence collapses at once, else by the method.
pub(crate) fn count_squarefree(f: &[Number], method: Method) -> usize {
    if f.len() > STURM_FROM
        && let Some(c) = sturm_count(f)
    {
        return c;
    }
    let neg: Vec<Number> = f
        .iter()
        .enumerate()
        .map(|(i, c)| if i % 2 == 1 { -c } else { c.clone() })
        .collect();
    let mut sink = Vec::new();
    positive(f, method, &mut sink, true) + positive(&neg, method, &mut sink, true)
}

/// The sign of `f(r)`, exactly: −1, 0 or 1.
pub(crate) fn sign_at(f: &[Number], r: &Rational) -> i32 {
    // q^n·f(p/q) = Σ aᵢ·pⁱ·q^(n−i), by Horner's rule, q > 0.
    let p = r.numer();
    let q = r.denom();
    let mut v = f[f.len() - 1].clone();
    let mut qk = Number::from(1u8);
    let one = crate::math::rational::is_one(q);
    for c in f[..f.len() - 1].iter().rev() {
        v = &v * p;
        if one {
            v += c;
        } else {
            qk = &qk * q;
            v += &(c * &qk);
        }
    }
    if v.is_zero() {
        0
    } else if v.is_negative() {
        -1
    } else {
        1
    }
}

/// The derivative's coefficients.
fn derivative(f: &[Number]) -> Vec<Number> {
    f.iter()
        .enumerate()
        .skip(1)
        .map(|(i, c)| c * &Number::from(i as u64))
        .collect()
}

/// An isolated root of a square-free factor, with what its refinement needs.
#[derive(Clone)]
pub(crate) struct Root {
    /// The index of the factor.
    pub(crate) factor: usize,
    pub(crate) iso: Iso,
    /// The sign of the factor just right of the left end, 0 until it is needed; just left of
    /// the right end it is the other, as the one root inside is simple.
    sa: i32,
}

/// A square-free factor of the polynomial, its derivative and multiplicity.
pub(crate) struct Factor {
    pub(crate) f: Vec<Number>,
    pub(crate) df: Vec<Number>,
    /// The factor and its derivative (for a degree at least 2) prepared for evaluations in
    /// fixed point.
    /// Built when first needed (isolation alone rarely needs them).
    fx: OnceCell<Fixed>,
    dfx: OnceCell<Option<Fixed>>,
    pub(crate) mult: u32,
}

impl Factor {
    fn new(f: Vec<Number>, mult: u32) -> Factor {
        let df = derivative(&f);
        Factor {
            f,
            df,
            fx: OnceCell::new(),
            dfx: OnceCell::new(),
            mult,
        }
    }

    /// The factor prepared for evaluations in fixed point.
    fn fx(&self) -> &Fixed {
        self.fx.get_or_init(|| Fixed::new(&self.f))
    }

    /// The derivative prepared so, for a factor of a degree at least 2.
    fn dfx(&self) -> Option<&Fixed> {
        self.dfx
            .get_or_init(|| (self.df.len() >= 2).then(|| Fixed::new(&self.df)))
            .as_ref()
    }

    /// The sign of the factor at `r`: in fixed point at a dyadic `r`, exactly otherwise.
    fn sign(&self, r: &Rational) -> i32 {
        let d = r.denom();
        if d.trailing_zeros().is_some_and(|t| d.bits() == t + 1) {
            let e = d.bits() as i64 - 1;
            self.fx().sign(&Pt::dyadic(r.numer(), -e))
        } else {
            sign_at(&self.f, r)
        }
    }
}

impl Root {
    fn new(factor: usize, iso: Iso) -> Root {
        Root { factor, iso, sa: 0 }
    }

    /// The sign of the factor just right of the left end (that of the derivative where the end
    /// is a root).
    fn side(&mut self, fs: &[Factor]) -> i32 {
        if self.sa == 0
            && let Iso::Open(a, _) = &self.iso
        {
            let f = &fs[self.factor];
            self.sa = match f.sign(a) {
                0 => sign_at(&f.df, a),
                s => s,
            };
        }
        self.sa
    }

    /// Halves the interval, exactly: the root is then the midpoint, or in one of the halves. An
    /// interval of one sign over more than two binades is halved in its exponent instead, at
    /// the power of two between its ends' (so that a root in a long interval, as from the bound
    /// of the roots, is reached in the logarithm of its length's binades, not in its binades).
    fn bisect(&mut self, fs: &[Factor]) {
        let Iso::Open(a, b) = &self.iso else {
            return;
        };
        let lg = |r: &Rational| r.numer().bits() as i64 - r.denom().bits() as i64;
        let same = !a.is_zero() && !b.is_zero() && a.is_negative() == b.is_negative();
        let (ea, eb) = (lg(&a.abs()), lg(&b.abs()));
        let m = if same && (ea - eb).abs() > 2 {
            let e = (ea + eb).div_euclid(2);
            let p = if e >= 0 {
                Rational::from(pow2(e as u64))
            } else {
                Rational::new(Number::from(1u8), pow2(e.unsigned_abs()))
            };
            if a.is_negative() { -p } else { p }
        } else {
            (a + b) / 2u32
        };
        if &m > a && &m < b {
            self.split_at(m, fs);
        } else {
            let m = (a + b) / 2u32;
            self.split_at(m, fs);
        }
    }

    /// Cuts the interval at `m`, inside it.
    fn split_at(&mut self, m: Rational, fs: &[Factor]) {
        if !matches!(self.iso, Iso::Open(..)) {
            return;
        }
        let s = fs[self.factor].sign(&m);
        let left = self.side(fs);
        let Iso::Open(a, b) = &self.iso else {
            return;
        };
        if s == 0 {
            self.iso = Iso::Point(m);
        } else if s == left {
            self.iso = Iso::Open(m, b.clone());
        } else {
            self.iso = Iso::Open(a.clone(), m);
        }
    }

    fn width(&self) -> Rational {
        self.iso.hi() - self.iso.lo()
    }
}

/// Whether two isolations overlap: open intervals that share more than an end, or a point inside
/// an open interval.
fn overlap(x: &Iso, y: &Iso) -> bool {
    match (x, y) {
        (Iso::Point(r), Iso::Point(s)) => r == s,
        // A point at an end too: the ends must not be roots.
        (Iso::Point(r), Iso::Open(a, b)) | (Iso::Open(a, b), Iso::Point(r)) => a <= r && r <= b,
        (Iso::Open(a, b), Iso::Open(c, d)) => a.max(c) < b.min(d),
    }
}

/// The square-free factors of `p` (a polynomial over the integers, not zero) and the
/// multiplicity of the root 0.
pub(crate) fn factors(p: &Polynomial<Number>) -> (Vec<Factor>, usize) {
    let (g, k) = yun::strip_zeros(p);
    let mut fs = Vec::new();
    if g.degree().unwrap_or(0) > 0 {
        for (h, m) in yun::squarefree(&g) {
            fs.push(Factor::new(h.c, m));
        }
    }
    (fs, k)
}

/// The isolated real roots of `p` (not zero): the roots of its square-free factors, refined until
/// no two overlap, in increasing order, and the factors.
pub(crate) fn isolate(p: &Polynomial<Number>, method: Method) -> (Vec<Root>, Vec<Factor>) {
    let (mut fs, k) = factors(p);
    let mut roots = Vec::new();
    for (i, f) in fs.iter().enumerate() {
        for iso in isolate_squarefree(&f.f, method) {
            roots.push(Root::new(i, iso));
        }
    }
    if k > 0 {
        // The root 0, of the factor x^k.
        fs.push(Factor::new(
            vec![Number::from(0u8), Number::from(1u8)],
            k as u32,
        ));
        roots.push(Root::new(fs.len() - 1, Iso::Point(Rational::default())));
    }
    // Refine until the isolations of the different factors are disjoint.
    loop {
        roots.sort_by(|x, y| x.iso.lo().cmp(y.iso.lo()).then(x.iso.hi().cmp(y.iso.hi())));
        let mut changed = false;
        let mut i = 0;
        while i + 1 < roots.len() {
            // Any later interval that starts before this one ends may overlap it.
            let mut j = i + 1;
            while j < roots.len() && roots[j].iso.lo() <= roots[i].iso.hi() {
                if overlap(&roots[i].iso, &roots[j].iso) {
                    let (x, y) = (roots[i].width(), roots[j].width());
                    if x >= y {
                        roots[i].bisect(&fs);
                    } else {
                        roots[j].bisect(&fs);
                    }
                    changed = true;
                }
                j += 1;
            }
            i += 1;
        }
        if !changed {
            break;
        }
    }
    (roots, fs)
}

/// The isolating intervals with multiplicities.
pub(crate) fn intervals(p: &Polynomial<Number>, method: Method) -> Vec<(Rational, Rational, u32)> {
    let (roots, fs) = isolate(p, method);
    roots
        .into_iter()
        .map(|r| {
            let m = fs[r.factor].mult;
            match r.iso {
                Iso::Point(x) => (x.clone(), x, m),
                Iso::Open(a, b) => (a, b, m),
            }
        })
        .collect()
}

/// The number of distinct real roots of `p` (not zero).
pub(crate) fn count(p: &Polynomial<Number>, method: Method) -> usize {
    let (fs, k) = factors(p);
    fs.iter()
        .map(|f| count_squarefree(&f.f, method))
        .sum::<usize>()
        + usize::from(k > 0)
}

/// The neighbours of a float halfway to the floats of its precision next to it: `(lo, hi)`,
/// the ends of the interval of the reals that round to it, as exact floats.
fn cell(x: &BigFloat) -> (BigFloat, BigFloat) {
    let p = x.prec();
    let (m, e) = x.mantissa_exponent().expect("a finite float not zero");
    let am = m.abs();
    let two = Number::from(2u8);
    let up = &(&am * &two) + 1u8;
    let is_pow2 = am.bits() == p && am == pow2(p - 1);
    let (down, de) = if is_pow2 {
        (&(&am * &Number::from(4u8)) - 1u8, e - 2)
    } else {
        (&(&am * &two) - 1u8, e - 1)
    };
    let hi = BigFloat::from_parts(&up, e - 1, p + 2, Round::Nearest).0;
    let lo = BigFloat::from_parts(&down, de, p + 2, Round::Nearest).0;
    if m.is_negative() {
        (-hi, -lo)
    } else {
        (lo, hi)
    }
}

/// The limbs of the evaluation in fixed point that give `p + kappa` bits of a value relative to
/// the bound of its magnitude, for a factor of the degree `n` (see `fixed`: `64W − 68 −
/// log2(2n)` bits at least).
fn width(n: usize, p: u64, kappa: u64) -> usize {
    (p + kappa + 72 + u64::from(n.max(1).ilog2())).div_ceil(64) as usize
}

/// The limbs of the fraction of a point of about `2^ex` in magnitude held to `p` bits.
fn frac_limbs(ex: i64, p: u64) -> usize {
    (p as i64 - ex).max(0).div_euclid(64) as usize + 1
}

/// The outcome of a Newton step.
enum Step {
    /// The evaluation is not precise enough for a step to `p` bits: so many more bits.
    Noise(u64),
    /// `x` is a root, exactly.
    Root,
    /// The derivative vanishes (to its precision).
    Flat,
    /// The next point, of the same fraction, the correction's bits (`None` for 0), and the
    /// correction in units of the fraction.
    Next(Pt, Option<u64>, Number),
}

/// One Newton step at `x` towards `p` correct bits, the value evaluated to `p + kappa` bits
/// relative to its bound and the derivative to about half that.
fn newton(fac: &Factor, x: &Pt, p: u64, kappa: u64) -> Step {
    let fx = fac.fx();
    let n = fx.degree();
    let w = width(n, p, kappa);
    // The derivative to about half the precision (its bound is n times the value's): the step's
    // error is then of the order of the square of the point's. Wide, both by rectangular
    // splitting; narrow, together by Horner's rule.
    let wd = width(n, p / 2 + 8, kappa + u64::from(n.ilog2()) + 1).min(w);
    let (v, d) = match fac.dfx() {
        Some(dfx) if w >= RECT_W && n >= RECT_N => (fx.value(x, w), dfx.value(&x.top(wd), wd)),
        _ => {
            let (v, d) = fx.eval(x, w, wd);
            (v, d.expect("the derivative"))
        }
    };
    if v.exact && v.v.iter().all(|&l| l == 0) {
        return Step::Root;
    }
    // The value within its error of 0 and 0 modulo the test prime: most likely x is a root
    // exactly (as the integers for Wilkinson's), which only an exact evaluation shows.
    if v.sign().is_none() && fx.zero_mod_p(x) && fx.sign(x) == 0 {
        return Step::Root;
    }
    let db = d.bits();
    if db == 0 {
        return Step::Flat;
    }
    // The uncertainty of the correction v/d, log2 (err·2^(64u)/|d|), against the step wanted,
    // |x|·2^−p: more bits when it is not well below.
    if !v.exact {
        let lerr = if v.err.is_finite() {
            v.err.max(1.0).log2().ceil() as i64
        } else {
            64 * w as i64
        };
        let unc = lerr + 64 * v.u - (db as i64 - 1 + 64 * d.u);
        let want = if x.mag.is_empty() {
            unc - 1
        } else {
            x.mag_bits() as i64 - 1 - 64 * x.f as i64 - p as i64 - 4
        };
        if unc > want || v.sign().is_none() {
            return Step::Noise((unc - want).max(16) as u64);
        }
    }
    // x − v/d with v = V·2^(64u), d = D·2^(64u'), in units of 2^(−64f):
    // C = V·2^(64(u − u' + f))/D.
    let sh = 64 * (v.u - d.u + x.f as i64);
    let vn = v.number();
    let num = if sh >= 0 {
        vn << sh as usize
    } else {
        vn >> sh.unsigned_abs() as usize
    };
    let c = &num / &d.number();
    let bits = if c.is_zero() { None } else { Some(c.bits()) };
    let y = &x.number() - &c;
    Step::Next(Pt::of_number(&y, x.f), bits, c)
}

/// Whether the root of `root` lies within `2|c|` of `y` (`c` in units of `y`'s fraction), by
/// the signs of the factor at the ends of that bracket (those inside the interval): if so, the
/// interval becomes the bracket.
fn bracket(root: &mut Root, fs: &[Factor], y: &Pt, c: &Number) -> bool {
    let Iso::Open(lo, hi) = root.iso.clone() else {
        return true;
    };
    let d = c.abs() << 1usize;
    let yn = y.number();
    let below = Pt::of_number(&(&yn - &d), y.f);
    let above = Pt::of_number(&(&yn + &d), y.f);
    let left = root.side(fs);
    let fx = fs[root.factor].fx();
    let inside_lo = below.cmp_rational(&lo) == Ordering::Greater;
    let inside_hi = above.cmp_rational(&hi) == Ordering::Less;
    let sl = if inside_lo { fx.sign(&below) } else { left };
    let sh = if inside_hi { fx.sign(&above) } else { -left };
    if sl == 0 {
        root.iso = Iso::Point(below.rational());
        return true;
    }
    if sh == 0 {
        root.iso = Iso::Point(above.rational());
        return true;
    }
    if sl == sh {
        return false;
    }
    // The root is between: the interval narrowed to the bracket (its side stays: sl = left).
    let (a, b) = (
        if inside_lo { below.rational() } else { lo },
        if inside_hi { above.rational() } else { hi },
    );
    root.iso = Iso::Open(a, b);
    root.sa = sl;
    true
}

/// The root of `r` rounded to nearest at `prec` bits, correctly.
fn refine(root: &mut Root, fs: &[Factor], prec: u64, kappa: &mut u64) -> BigFloat {
    let f = &fs[root.factor];
    if let Iso::Point(x) = &root.iso {
        return BigFloat::from_rational(x, prec, Round::Nearest).0;
    }
    let fx = f.fx();
    // Newton's method works above the precision the interval's width needs (2^−32 of it).
    let mut target = (prec + 32).max(96);
    // `kappa`: the bits the evaluation needs beyond those of the point, where the polynomial is
    // ill conditioned (as near ±1 for Chebyshev's polynomials), found as it goes and passed on
    // from one root of the factor to the next.
    loop {
        // An interval over many binades is first cut down to a few by its exponents.
        for _ in 0..4096 {
            let Iso::Open(a, b) = &root.iso else {
                break;
            };
            let lg = |r: &Rational| r.numer().bits() as i64 - r.denom().bits() as i64;
            let same = !a.is_zero() && !b.is_zero() && a.is_negative() == b.is_negative();
            if !(same && (lg(&a.abs()) - lg(&b.abs())).abs() > 2) {
                break;
            }
            root.bisect(fs);
        }
        let Iso::Open(a, b) = root.iso.clone() else {
            return BigFloat::from_rational(root.iso.lo(), prec, Round::Nearest).0;
        };
        let mid = (&a + &b) / 2u32;
        // The bits the interval already gives: log2(|mid| / width), roughly; Newton's method
        // starts above them, and the root must be inside the interval only at the end.
        let width_r = &b - &a;
        let lg = |r: &Rational| r.numer().bits() as i64 - r.denom().bits() as i64;
        let ex = lg(&mid.abs()).max(lg(&a.abs())).max(lg(&b.abs()));
        let known = (lg(&mid.abs()) - lg(&width_r)).max(0) as u64;
        // At least 32 bits finer than the interval, so that its midpoint is inside it.
        let mut p = (known + 64).min(target.max(known + 32));
        let at = |r: &Rational, p: u64| {
            let fl = frac_limbs(ex, p);
            let num = r.numer() << (64 * fl);
            Pt::of_number(&(&num / r.denom()), fl)
        };
        let mut x = at(&mid, p);
        let mut ok = true;
        // Newton's method safeguarded by bisection at the starting precision, the interval cut
        // when an iterate leaves it, until the steps are small; then one step at each doubled
        // precision up to the target.
        let mut converged = false;
        // The bits of the last correction: a step that shrinks it less than 4 times is linear
        // convergence (towards a close pair of roots).
        let mut prev: Option<u64> = None;
        // Whether a step converged only linearly: then the last iterate is proven by a bracket
        // (Newton's steps are small long before it is near a root of a close pair).
        let mut linear_seen = false;
        for _ in 0..400 {
            let Iso::Open(lo, hi) = root.iso.clone() else {
                return BigFloat::from_rational(root.iso.lo(), prec, Round::Nearest).0;
            };
            let next = match newton(f, &x, p, *kappa) {
                Step::Noise(more) => {
                    *kappa += more;
                    continue;
                }
                Step::Root => {
                    converged = true;
                    break;
                }
                Step::Flat => None,
                Step::Next(y, bits, c) => {
                    if y.cmp_rational(&lo) == Ordering::Greater
                        && y.cmp_rational(&hi) == Ordering::Less
                    {
                        // Small: below 2^(−p/2) of the point; unless the step was quadratic, proven
                        // by the signs at y ± 2|c| (towards a close pair of roots, Mignotte's,
                        // Newton's steps are small long before the iterate is near one).
                        let small = bits.is_none_or(|c| c + p / 2 <= y.mag_bits());
                        if small {
                            // Quadratic: |c| ≲ |c_prev|²/|y|.
                            let quadratic = !linear_seen
                                && matches!((bits, prev), (Some(cb), Some(pb))
                                    if cb + y.mag_bits() <= 2 * pb + 4);
                            if quadratic || bracket(root, fs, &y, &c) {
                                x = y;
                                converged = true;
                                break;
                            }
                            // Not yet: the interval cut at y, Newton's method on from there.
                            root.split_at(y.rational(), fs);
                            if !matches!(root.iso, Iso::Open(..)) {
                                converged = true;
                                break;
                            }
                        } else if let (Some(c), Some(pc)) = (bits, prev)
                            && c + 2 > pc
                        {
                            linear_seen = true;
                        }
                        prev = bits;
                        Some(y)
                    } else {
                        None
                    }
                }
            };
            // A Newton step inside the interval is taken as it is; one outside it halves the
            // interval and starts again from the middle.
            x = match next {
                Some(y) => y,
                None => {
                    root.bisect(fs);
                    let Iso::Open(lo, hi) = &root.iso else {
                        continue;
                    };
                    at(&((lo + hi) / 2u32), p)
                }
            };
        }
        let Iso::Open(a, b) = root.iso.clone() else {
            return BigFloat::from_rational(root.iso.lo(), prec, Round::Nearest).0;
        };
        if !converged {
            ok = false;
        }
        let mut exact_root = false;
        let mut linear = false;
        'up: while ok && p < target {
            p = (2 * p).min(target);
            let ex = if x.mag.is_empty() { ex } else { x.log2_up() };
            let fl = frac_limbs(ex, p);
            if fl > x.f {
                let mut y = x.clone();
                let more = fl - x.f;
                // (0 stays empty: limbs of zeros under nothing would be a magnitude of 0 that
                // is not trimmed.)
                if !y.mag.is_empty() {
                    y.mag.splice(0..0, core::iter::repeat_n(0u64, more));
                }
                y.f = fl;
                x = y;
            }
            let mut steps = 1;
            while steps > 0 {
                match newton(f, &x, p, *kappa) {
                    Step::Noise(more) => *kappa += more,
                    Step::Root => {
                        exact_root = true;
                        break 'up;
                    }
                    Step::Flat => {
                        ok = false;
                        break 'up;
                    }
                    Step::Next(y, bits, _) => {
                        // At this level the step is of the size of the last level's error,
                        // 2^(−p/2) of the point; one much larger is linear convergence: from the
                        // (narrowed) interval again.
                        if bits.is_some_and(|c| c + p / 2 > y.mag_bits() + 8) {
                            linear = true;
                            break 'up;
                        }
                        x = y;
                        steps -= 1;
                    }
                }
            }
        }
        if linear {
            // Bisection gains a bit a step for sure until the interval is below the distance to
            // the other root of the pair; then Newton's method is quadratic.
            for _ in 0..16 {
                root.bisect(fs);
            }
            continue;
        }
        if ok && !(x.cmp_rational(&a) == Ordering::Greater && x.cmp_rational(&b) == Ordering::Less)
        {
            ok = false;
        }
        if !ok {
            for _ in 0..8 {
                root.bisect(fs);
            }
            target += 32;
            continue;
        }
        if exact_root {
            return x.float(prec, Round::Nearest);
        }
        // Verify the rounding: the root is in the cell of the rounded value.
        let r = x.float(prec, Round::Nearest);
        if r.is_zero() {
            target *= 2;
            continue;
        }
        let (lo, hi) = cell(&r);
        let slo = if lo > a {
            fx.sign(&Pt::of_float(&lo))
        } else {
            root.side(fs)
        };
        let shi = if hi < b {
            fx.sign(&Pt::of_float(&hi))
        } else {
            -root.side(fs)
        };
        if slo == 0 {
            return lo.prec_round(prec, Round::Nearest).0;
        }
        if shi == 0 {
            return hi.prec_round(prec, Round::Nearest).0;
        }
        if slo != shi {
            return r;
        }
        // The root is outside the cell: narrow the interval at the end of the cell and go on.
        let cut = if slo == root.side(fs) { hi } else { lo };
        let cut = Rational::try_from(&cut).expect("finite");
        if cut > a && cut < b {
            root.split_at(cut, fs);
        }
        target += 32;
    }
}

/// The real roots of `p` (not zero) rounded to nearest at `prec` bits, with their
/// multiplicities, in increasing order.
pub(crate) fn real_roots(p: &Polynomial<Number>, prec: u64) -> Vec<(BigFloat, u32)> {
    let (mut roots, fs) = isolate(p, DEFAULT);
    let mut kappa = vec![0u64; fs.len()];
    roots
        .iter_mut()
        .map(|r| {
            let m = fs[r.factor].mult;
            (refine(r, &fs, prec, &mut kappa[r.factor]), m)
        })
        .collect()
}

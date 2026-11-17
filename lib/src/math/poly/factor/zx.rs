//! Factorisation over the integers: the square-free decomposition by Yun's method, and the
//! factors of a square-free primitive polynomial by Zassenhaus's method — its factorisation
//! modulo a good prime (the one of several with the fewest factors), lifted by Hensel's lemma
//! (quadratically, all the factors at once in a binary tree), and recombined, the subsets of the
//! lifted factors pruned by the degrees the other primes allow, by the trace test and by the
//! constant-coefficient test before a trial division.

use super::super::modular::{Modulus, Work};
use super::super::{ModPolynomial, Polynomial};
use super::fp::Fp;
use super::{ff, nmod, zn};
use crate::math::numbers::{Number, is_prime_u64, limbs_of, nat};
use core::cmp::Ordering;
use std::sync::Arc;

/// The good primes the factorisation tries, keeping the one with the fewest factors and the
/// degrees all of them allow: [`PRIMES`], and up to [`MORE_PRIMES`] while the fewest factors are
/// more than [`MANY`], which makes the recombination long; all [`MORE_PRIMES`] when the bound of
/// the coefficients of a factor has [`WIDE_BOUND`] bits a degree or more, where the lifting is
/// long against the distinct-degree factorisations and the degree sets of several primes prove
/// most inputs irreducible without it. Measured on an Apple M4: a second and third prime's
/// distinct-degree factorisations cost more than their pruning saves on most inputs (random
/// polynomials and products of two of degree 100 to 300 take half the time with one), but not
/// with large coefficients (random ones of 1,000 bits at degree 10 and 30 take a tenth of the
/// time with six, of 256 bits at degree 20 a third) or many local factors (`x^210 - 1` takes ten
/// times as long with one prime only).
const PRIMES: usize = 1;
/// See [`PRIMES`].
const WIDE_BOUND: u64 = 12;
/// See [`PRIMES`].
const MORE_PRIMES: usize = 6;
/// See [`PRIMES`].
const MANY: usize = 12;

/// `f` with integer coefficients compared by degree, then by the coefficients from the leading
/// one down.
pub(crate) fn cmp_poly(a: &Polynomial<Number>, b: &Polynomial<Number>) -> Ordering {
    a.length().cmp(&b.length()).then_with(|| {
        for (x, y) in a.coeffs().iter().rev().zip(b.coeffs().iter().rev()) {
            match x.cmp(y) {
                Ordering::Equal => {}
                o => return o,
            }
        }
        Ordering::Equal
    })
}

/// Sorts factors with multiplicities by the factor, then the multiplicity.
pub(crate) fn sort_factors(v: &mut [(Polynomial<Number>, u32)]) {
    v.sort_by(|a, b| cmp_poly(&a.0, &b.0).then(a.1.cmp(&b.1)));
}

/// `f` divided by `d`, which divides it.
fn exact(f: &Polynomial<Number>, d: &Polynomial<Number>) -> Polynomial<Number> {
    f.div_exact(d).expect("an exact division")
}

/// Up to this degree of a candidate factor the recombination's trial division is the schoolbook
/// one, which stops at the first coefficient of the quotient that does not divide; beyond, the
/// arith area's exact division (Kronecker substitution past 8 coefficients of the quotient).
fn short_div() -> usize {
    4
}

/// `f / d` when `d` divides `f` over the integers.
fn div_exact(f: &Polynomial<Number>, d: &Polynomial<Number>) -> Option<Polynomial<Number>> {
    let (n, m) = (d.length(), f.length());
    if n == 0 || m < n || n - 1 > short_div() {
        return f.div_exact(d);
    }
    let l = d.leading().expect("not zero");
    let one = crate::math::rational::is_one(l);
    let dc = d.coeffs();
    let mut r = f.coeffs().to_vec();
    let mut q = vec![Number::default(); m - n + 1];
    for i in (0..=m - n).rev() {
        let top = &r[i + n - 1];
        let t = if one {
            top.clone()
        } else {
            let (t, rem) = top.div_rem(l);
            if !rem.is_zero() {
                return None;
            }
            t
        };
        if !t.is_zero() {
            for (x, y) in r[i..i + n - 1].iter_mut().zip(dc) {
                *x -= &t * y;
            }
        }
        q[i] = t;
    }
    r[..n - 1]
        .iter()
        .all(Number::is_zero)
        .then(|| Polynomial::new(q))
}

/// The greatest common divisor over the integers of Yun's method.
///
/// The gcd seam (integers): the gcd area's `Polynomial<Number>::gcd` (multimodular, the heuristic
/// gcd for the shortest). Reached only for inputs that are not square-free (the quick test modulo
/// primes above `2^31` settles the others).
fn gcd_z(a: &Polynomial<Number>, b: &Polynomial<Number>) -> Polynomial<Number> {
    a.gcd(b)
}

/// The Bézout coefficients `s·a + t·b = 1` of coprime `a` and `b` modulo a prime, for the
/// Hensel tree.
///
/// The gcd seam (extended gcd modulo a word prime): the gcd area's `ModPolynomial::xgcd` (Euclid
/// with the matrix in Montgomery's form, the half gcd past the crossover).
fn xgcd_mod(a: &ModPolynomial, b: &ModPolynomial) -> (ModPolynomial, ModPolynomial) {
    let (g, s, t) = a.xgcd(b);
    debug_assert!(g.is_one());
    (s, t)
}

/// The square-free decomposition of the primitive `f` of a positive leading coefficient and
/// degree at least 1, by Yun's method: the primitive square-free `aᵢ`, pairwise coprime, of
/// positive leading coefficients, with `f = Π aᵢ^i`, those not constant.
pub(crate) fn yun(f: &Polynomial<Number>) -> Vec<(Polynomial<Number>, u32)> {
    let mut out = Vec::new();
    let df = f.derivative();
    let a0 = gcd_z(f, &df);
    let mut b = exact(f, &a0);
    let mut c = exact(&df, &a0);
    let mut i = 1u32;
    loop {
        let d = &c - &b.derivative();
        let a = gcd_z(&b, &d);
        if !a.is_constant() {
            out.push((a.clone(), i));
        }
        b = exact(&b, &a);
        if b.is_constant() {
            break;
        }
        c = exact(&d, &a);
        i += 1;
    }
    out
}

/// Whether `f`, modulo the word prime `p` and of the same degree, is square-free.
fn squarefree_mod(f: &ModPolynomial, p: u64) -> bool {
    let w = nmod::Wm::new(p);
    let d: Vec<u64> =
        f.c.iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| w.mul(c, w.reduce1(i as u64)))
            .collect();
    let mut d = d;
    nmod::trim(&mut d);
    nmod::gcd(&w, &f.c, &d) == [1]
}

/// A prime `p` that does not divide `lc` and modulo which `f` stays square-free, from `from`
/// up: `p` and `f mod p`, `None` when the primes run out below `2^62`.
fn next_good_prime(
    f: &Polynomial<Number>,
    lc: &Number,
    from: &mut u64,
) -> Option<(u64, ModPolynomial)> {
    while *from < 1 << 62 {
        let p = *from;
        *from += 1;
        if !is_prime_u64(p) {
            continue;
        }
        let pn = Number::from(p);
        if lc.rem_euclid(&pn).is_zero() {
            continue;
        }
        let fp = ModPolynomial::from_polynomial(f, &pn);
        if squarefree_mod(&fp, p) {
            return Some((p, fp));
        }
    }
    None
}

/// The first primes above `2^31`, of the quick test of square-freeness.
const SQUAREFREE_PRIMES: [u64; 6] = [
    2_147_483_659,
    2_147_483_693,
    2_147_483_713,
    2_147_483_743,
    2_147_483_777,
    2_147_483_783,
];

/// Whether the primitive `f` is square-free: by primes from `2^31` up, unlikely to divide its
/// discriminant, modulo one of which it stays so, or by Yun's method when three of them do not
/// tell.
pub(crate) fn is_squarefree(f: &Polynomial<Number>) -> bool {
    let lc = f.leading().expect("not zero").clone();
    let mut tried = 0;
    for p in SQUAREFREE_PRIMES {
        let pn = Number::from(p);
        if lc.rem_euclid(&pn).is_zero() {
            continue;
        }
        if squarefree_mod(&ModPolynomial::from_polynomial(f, &pn), p) {
            return true;
        }
        tried += 1;
        if tried == 3 {
            break;
        }
    }
    let y = yun(f);
    y.len() == 1 && y[0].1 == 1
}

/// The subset sums of the degrees `d`, as a set of `0..=n`.
fn degree_set(d: &[usize], n: usize) -> Vec<bool> {
    let mut s = vec![false; n + 1];
    s[0] = true;
    for &x in d {
        for i in (x..=n).rev() {
            if s[i - x] {
                s[i] = true;
            }
        }
    }
    s
}

/// `⌊√x⌋ + 1`, an upper bound of `√x`.
fn sqrt_up(x: &Number) -> Number {
    &x.isqrt() + &Number::from(1u8)
}

/// The binomial coefficient `C(n, k)`.
fn binomial(n: usize, k: usize) -> Number {
    let mut r = Number::from(1u8);
    for i in 0..k {
        r = &(&r * &Number::from((n - i) as u64)) / &Number::from((i + 1) as u64);
    }
    r
}

/// The bounds of the recombination of the factors of `f` of degree `n`: `B`, of the coefficients
/// of `lc(f)·g/lc(g)` for any factor `g` of a degree up to `n/2` (the recombination tries the
/// complement of a subset of a degree above half), and `T`, of its coefficient below the leading
/// one for a factor of any degree, the trace test's, both by Mignotte's bound `|gⱼ| ≤ C(k-1, j)·‖f‖
/// + C(k-1, j-1)·|lc(f)|` for a factor `g` of degree `k`: `B = |lc|·C(K, ⌊K/2⌋)·(‖f‖ + |lc|)` with
/// `K = ⌊n/2⌋`, and `T = |lc|·(‖f‖ + (n-1)·|lc|)`, `‖f‖` rounded up.
fn bounds(f: &Polynomial<Number>) -> (Number, Number) {
    let n = f.degree().expect("not constant");
    let lc = f.leading().expect("not zero").abs();
    let mut s2 = Number::default();
    for c in f.coeffs() {
        s2 += &(c * c);
    }
    let norm = sqrt_up(&s2);
    let k = (n / 2).max(1);
    let b = &(&binomial(k, k / 2) * &(&norm + &lc)) * &lc;
    let t = &(&norm + &(&Number::from((n - 1) as u64) * &lc)) * &lc;
    (b, t)
}

/// A node of the tree of Hensel's lifting: a product of local factors, and the Bézout
/// coefficients of its two children, `s·left + t·right ≡ 1`.
struct Node {
    v: ModPolynomial,
    kids: Option<Box<(Node, Node, ModPolynomial, ModPolynomial)>>,
}

/// The tree of the factors `fs` modulo `p`, halved by degree.
fn build(fs: &[ModPolynomial]) -> Node {
    if fs.len() == 1 {
        return Node {
            v: fs[0].clone(),
            kids: None,
        };
    }
    let total: usize = fs.iter().map(|f| f.degree().unwrap_or(0)).sum();
    let mut acc = 0;
    let mut cut = 1;
    for (i, f) in fs.iter().enumerate() {
        acc += f.degree().unwrap_or(0);
        if 2 * acc >= total {
            cut = (i + 1).clamp(1, fs.len() - 1);
            break;
        }
    }
    let l = build(&fs[..cut]);
    let r = build(&fs[cut..]);
    let v = &l.v * &r.v;
    let (s, t) = xgcd_mod(&l.v, &r.v);
    Node {
        v,
        kids: Some(Box::new((l, r, s, t))),
    }
}

/// The polynomial `a` with the modulus `m`, its coefficients the same integers.
fn remod(a: &ModPolynomial, m: &Arc<Modulus>) -> ModPolynomial {
    let k = m.k();
    let mut c = vec![0u64; a.length() * k];
    for i in 0..a.length() {
        let l = a.limbs(i);
        let n = l.len().min(k);
        c[i * k..i * k + n].copy_from_slice(&l[..n]);
    }
    ModPolynomial::of(m.clone(), c)
}

/// From this many limbs of `pᵃ` a step of Hensel's lifting makes its corrections modulo `m1`
/// (measured on an Apple M4: from two limbs on, at half the precision, they pay for the
/// divisions by `m` that the errors take).
const SPLIT_LIFT: usize = 2;

/// The moduli of one step of the lifting: from `m` to `m2 = m·m1`, `m1` dividing `m`.
struct Step {
    /// The limbs of `m`, odd, and the inverse of the lowest modulo `2^64`.
    ml: Vec<u64>,
    mv: u64,
    /// `m1 = m2/m`, the modulus of the corrections.
    m1: Arc<Modulus>,
    /// `m2`.
    m2: Arc<Modulus>,
}

/// The coefficients of `a` reduced modulo `to`.
fn reduce_to(a: &ModPolynomial, to: &Arc<Modulus>) -> Vec<u64> {
    let k = to.k();
    let mut c = vec![0u64; a.length() * k];
    let mut work = Work::default();
    for i in 0..a.length() {
        to.reduce(&mut c[i * k..(i + 1) * k], a.limbs(i), &mut work);
    }
    zn::trim(&mut c, k);
    c
}

/// The packed `e` modulo `m2`, each coefficient a multiple of `m`, divided by `m`: modulo `m1`,
/// by the exact division from the low end (`m` is odd, the quotient below `m1`).
fn divide(e: &[u64], st: &Step) -> Vec<u64> {
    let (k2, k1) = (st.m2.k(), st.m1.k());
    let mut c = Vec::with_capacity(e.len() / k2 * k1);
    let mut x = vec![0u64; k2];
    let mut q = Vec::with_capacity(k1);
    for xs in e.chunks_exact(k2) {
        x.copy_from_slice(xs);
        let qn = k1.min(k2);
        nat::bdiv_into(&mut q, qn, &mut x, &st.ml, st.mv);
        q.resize(k1, 0);
        c.extend_from_slice(&q);
    }
    zn::trim(&mut c, k1);
    c
}

/// `a ± m·x` modulo `m2` for `a` modulo `m` and the packed `x` modulo `m1`: below `m2` for `+`
/// as `a < m` and `x < m1`, and `m2` added to a negative difference for `-`.
fn shift_add(a: &ModPolynomial, x: &[u64], st: &Step, minus: bool) -> ModPolynomial {
    let (k1, k2) = (st.m1.k(), st.m2.k());
    let ka = a.m.k();
    debug_assert!(ka <= k2);
    let n = a.length().max(x.len() / k1);
    let mut out = vec![0u64; n * k2];
    let km = st.ml.len();
    let mut t = vec![0u64; km + k1];
    let mut scratch = Vec::new();
    for (i, o) in out.chunks_exact_mut(k2).enumerate() {
        if i < a.length() {
            o[..ka].copy_from_slice(a.limbs(i));
        }
        let Some(xi) = x.get(i * k1..(i + 1) * k1) else {
            continue;
        };
        if xi.iter().all(|&l| l == 0) {
            continue;
        }
        nat::mul_to(&mut t, &st.ml, xi, &mut scratch);
        let tl = t.iter().rposition(|&l| l != 0).map_or(0, |j| j + 1);
        debug_assert!(tl <= k2);
        if minus {
            if nat::sub_assign(o, &t[..tl]) {
                nat::add_assign(o, &st.m2.d);
            }
        } else {
            let carry = nat::add_assign(o, &t[..tl]);
            debug_assert!(!carry);
        }
    }
    ModPolynomial::of(st.m2.clone(), out)
}

/// Lifts the node from its factorisation `v = left·right` modulo `m` (with its Bézout
/// coefficients) to one of `target` modulo `m2 = m·m1`, `target ≡ v mod m`; the Bézout coefficients
/// too when `bezout`. Von zur Gathen and Gerhard's Algorithm 15.10 for monic factors, the
/// corrections modulo `m1` from the errors divided by `m`: only `g·h` and `s·g* + t·h*` take the
/// full precision.
fn lift(node: &mut Node, target: ModPolynomial, st: &Step, bezout: bool) {
    node.v = target;
    let Some(kids) = node.kids.as_mut() else {
        return;
    };
    let (l, r, s, t) = &mut **kids;
    let (m1, m2) = (&*st.m1, &*st.m2);
    let (g, h) = (&l.v, &r.v);
    // e = f - g·h ≡ 0 mod m, and δ = e/m mod m1.
    let gh = zn::mul(m2, &remod(g, &st.m2).c, &remod(h, &st.m2).c);
    let delta = divide(&zn::sub(m2, &node.v.c, &gh), st);
    let (g1, h1) = (reduce_to(g, &st.m1), reduce_to(h, &st.m1));
    let (s1, t1) = (reduce_to(s, &st.m1), reduce_to(t, &st.m1));
    let dh = zn::Div::new(m1, &h1);
    // s·δ = q·h + r: g* = g + m·(t·δ + q·g), h* = h + m·r.
    let (q, rr) = dh.divrem(m1, &zn::mul(m1, &s1, &delta), true);
    let x = zn::add(m1, &zn::mul(m1, &t1, &delta), &zn::mul(m1, &q, &g1));
    let gs = shift_add(g, &x, st, false);
    let hs = shift_add(h, &rr, st, false);
    if bezout {
        // b = s·g* + t·h* - 1 ≡ 0 mod m, b1 = b/m mod m1; s·b1 = c·h* + d (h* ≡ h mod m1):
        // s* = s - m·d, t* = t - m·(t·b1 + c·g*).
        let mut one = vec![0u64; m2.k()];
        one[0] = 1;
        let sg = zn::mul(m2, &remod(s, &st.m2).c, &gs.c);
        let th = zn::mul(m2, &remod(t, &st.m2).c, &hs.c);
        let b1 = divide(&zn::sub(m2, &zn::add(m2, &sg, &th), &one), st);
        let (c, d) = dh.divrem(m1, &zn::mul(m1, &s1, &b1), true);
        let y = zn::add(m1, &zn::mul(m1, &t1, &b1), &zn::mul(m1, &c, &g1));
        *s = shift_add(s, &d, st, true);
        *t = shift_add(t, &y, st, true);
    }
    lift(l, gs, st, bezout);
    lift(r, hs, st, bezout);
}

/// Lifts the node from its factorisation `v = left·right` modulo `m` (with its Bézout
/// coefficients) to one of `target` modulo `m2`, a divisor of `m²`, `target ≡ v mod m`; the
/// Bézout coefficients too when `bezout`: every product at the full precision.
fn lift_full(node: &mut Node, target: ModPolynomial, m2: &Arc<Modulus>, bezout: bool) {
    node.v = target;
    let Some(kids) = node.kids.as_mut() else {
        return;
    };
    let (l, r, s, t) = &mut **kids;
    let m = &**m2;
    let (g, h) = (remod(&l.v, m2).c, remod(&r.v, m2).c);
    let (s2, t2) = (remod(s, m2).c, remod(t, m2).c);
    let e = zn::sub(m, &node.v.c, &zn::mul(m, &g, &h));
    let dh = zn::Div::new(m, &h);
    let (q, rr) = dh.divrem(m, &zn::mul(m, &s2, &e), true);
    let gs = zn::add(
        m,
        &zn::add(m, &g, &zn::mul(m, &t2, &e)),
        &zn::mul(m, &q, &g),
    );
    let hs = zn::add(m, &h, &rr);
    if bezout {
        let mut one = vec![0u64; m.k()];
        one[0] = 1;
        let b = zn::sub(
            m,
            &zn::add(m, &zn::mul(m, &s2, &gs), &zn::mul(m, &t2, &hs)),
            &one,
        );
        let (c, d) = zn::Div::new(m, &hs).divrem(m, &zn::mul(m, &s2, &b), true);
        *s = ModPolynomial::of(m2.clone(), zn::sub(m, &s2, &d));
        *t = ModPolynomial::of(
            m2.clone(),
            zn::sub(
                m,
                &zn::sub(m, &t2, &zn::mul(m, &t2, &b)),
                &zn::mul(m, &c, &gs),
            ),
        );
    }
    lift_full(l, ModPolynomial::of(m2.clone(), gs), m2, bezout);
    lift_full(r, ModPolynomial::of(m2.clone(), hs), m2, bezout);
}

/// The leaves of the tree, in order.
fn leaves(node: Node, out: &mut Vec<ModPolynomial>) {
    match node.kids {
        None => out.push(node.v),
        Some(k) => {
            let (l, r, _, _) = *k;
            leaves(l, out);
            leaves(r, out);
        }
    }
}

/// The monic factors `fs` modulo `p` of `f/lc(f)` lifted to monic factors modulo `p^a`.
fn hensel(f: &Polynomial<Number>, p: u64, a: u32, fs: &[ModPolynomial]) -> Vec<ModPolynomial> {
    if fs.len() == 1 || a == 1 {
        return fs.to_vec();
    }
    let pn = Number::from(p);
    // The exponents of the steps, from a down to 1, each at most twice the next.
    let mut es = vec![a];
    while *es.last().expect("not empty") > 1 {
        let e = es.last().expect("not empty").div_ceil(2);
        es.push(e);
    }
    es.reverse();
    let mut tree = build(fs);
    let lc = f.leading().expect("not zero").clone();
    for (i, w) in es.windows(2).enumerate() {
        let m = pn.pow(w[0]);
        let m2n = pn.pow(w[1]);
        let ml = limbs_of(&m);
        let st = Step {
            m1: Arc::new(Modulus::new(&pn.pow(w[1] - w[0]))),
            m2: Arc::new(Modulus::new(&m2n)),
            mv: nat::binvert(ml[0]),
            ml,
        };
        let inv = lc
            .mod_inverse(&m2n)
            .expect("p does not divide the leading coefficient");
        let target = ModPolynomial::from_polynomial(f, &m2n).scale(&inv);
        let target = ModPolynomial::of(st.m2.clone(), target.c);
        // The corrections modulo m1 pay for the divisions of the errors by m from two limbs up.
        if st.m2.k() >= SPLIT_LIFT {
            lift(&mut tree, target, &st, i + 2 < es.len());
        } else {
            lift_full(&mut tree, target, &st.m2, i + 2 < es.len());
        }
    }
    let mut out = Vec::with_capacity(fs.len());
    leaves(tree, &mut out);
    out
}

/// A prime of the factorisation: its field, the distinct-degree factorisation of `f` modulo it,
/// `x^p mod f` and the number of factors.
type Choice = (Fp, Vec<(Vec<u64>, u32)>, Vec<u64>, usize);

/// The irreducible factors over the integers of the square-free primitive `f` of a positive
/// leading coefficient, degree at least 1 and `f(0) ≠ 0`, primitive with positive leading
/// coefficients, in no order.
pub(crate) fn factor_squarefree(f: &Polynomial<Number>) -> Vec<Polynomial<Number>> {
    let n = f.degree().expect("not zero");
    if n <= 1 {
        return vec![f.clone()];
    }
    let lc = f.leading().expect("not zero").clone();
    // Good primes: the one with the fewest factors, and the degrees all allow.
    let mut from = 3u64;
    // For each prime only the distinct-degree factorisation, which tells the number of factors
    // and their degrees; the equal-degree splitting for the chosen one.
    let mut best: Option<Choice> = None;
    let mut allowed = vec![true; n + 1];
    // Every prime when the lifting goes far: their degree sets prove most such inputs
    // irreducible without it.
    let (b, t) = bounds(f);
    let primes = if b.bits() >= WIDE_BOUND * n as u64 {
        MORE_PRIMES
    } else {
        PRIMES
    };
    for i in 0..MORE_PRIMES {
        if i >= primes && best.as_ref().is_some_and(|b| b.3 <= MANY) {
            break;
        }
        let Some((_, fm)) = next_good_prime(f, &lc, &mut from) else {
            break;
        };
        let fpx = Fp::new(fm.m.clone());
        let monic = fm.to_monic();
        let (parts, sigma) = ff::ddf(&fpx, &monic.c);
        let ds: Vec<usize> = parts
            .iter()
            .flat_map(|(g, d)| {
                let d = *d as usize;
                core::iter::repeat_n(d, fpx.deg(g) / d)
            })
            .collect();
        if ds.len() == 1 {
            return vec![f.clone()];
        }
        let set = degree_set(&ds, n);
        for (a, b) in allowed.iter_mut().zip(&set) {
            *a &= *b;
        }
        if allowed[1..n].iter().all(|&x| !x) {
            return vec![f.clone()];
        }
        if best.as_ref().is_none_or(|(_, _, _, r)| ds.len() < *r) {
            best = Some((fpx, parts, sigma, ds.len()));
        }
    }
    let (fpx, parts, sigma, _) = best.expect("a good prime");
    let p = fpx.m.word().expect("a word prime");
    let mut state = ff::SEED;
    let mut fs = Vec::new();
    for (g, d) in &parts {
        let mut out = Vec::new();
        ff::edf(&fpx, g, *d as usize, Some(&sigma), &mut state, &mut out);
        fs.extend(out.into_iter().map(|c| fpx.wrap(c)));
    }
    // p^a > 2B.
    let pn = Number::from(p);
    let two_b = &b << 1usize;
    let mut a = 1u32;
    let mut pa = pn.clone();
    while pa <= two_b {
        pa = &pa * &pn;
        a += 1;
    }
    let lifted = hensel(f, p, a, &fs);
    recombine(f, lifted, &pa, &allowed, &t)
}

/// A test of the recombination: the values of the active factors by position, and the residues
/// `bound` and `pa - bound` between which a sum's residue fails.
type Test = (Vec<Vec<u64>>, Vec<u64>, Vec<u64>);

/// An upper bound of the absolute values of the complex roots of `f`, Fujiwara's: `2·max(|f_(n-i)
/// / f_n|^(1/i), |f_0 / (2 f_n)|^(1/n))`, each root rounded up.
fn root_bound(f: &Polynomial<Number>) -> Number {
    let n = f.degree().expect("not constant");
    let lc = f.leading().expect("not zero").abs();
    let one = Number::from(1u8);
    let mut r = Number::default();
    for i in 1..=n {
        let c = f.coeff(n - i).abs();
        if c.is_zero() {
            continue;
        }
        let den = if i == n { &lc << 1usize } else { lc.clone() };
        let q = (&(&c + &den) - &one) / &den;
        let x = &q.nth_root(i as u32) + &one;
        if x > r {
            r = x;
        }
    }
    &r << 1usize
}

/// Zassenhaus's recombination of the monic factors `gs` modulo `pa` of `f/lc(f)`: the subsets of
/// one factor, then of two, …, each tested by its degree against `allowed`; by the power sums
/// `p₁` and `p₂` of the roots of the subset, the sums of those of its factors, scaled to integers
/// by `lc` and `lc²` and bounded by those of the roots of `f` (`t` and Fujiwara's bound: the
/// first is the trace test, the second catches the even factors of an even polynomial, where the
/// trace is 0); by the constant coefficient, dividing `lc·f(0)`; and then by a trial division.
fn recombine(
    f: &Polynomial<Number>,
    gs: Vec<ModPolynomial>,
    pa: &Number,
    allowed: &[bool],
    t: &Number,
) -> Vec<Polynomial<Number>> {
    let pm = gs[0].m.clone();
    let k = pm.k();
    let n = f.degree().expect("not constant");
    let half = pa >> 1usize;
    let sym = |x: Number| if x > half { &x - pa } else { x };
    let limbs = |x: &Number| pm.residue(x);
    // The sums on two words when pa is below 2^127, on limbs otherwise.
    let narrow = pa.bits() < 127;
    let wide = |l: &[u64]| u128::from(l[0]) | (u128::from(*l.get(1).unwrap_or(&0)) << 64);
    let p128 = if narrow { wide(&pm.d) } else { 0 };
    let le = |a: &[u64], b: &[u64]| {
        for (x, y) in a.iter().rev().zip(b.iter().rev()) {
            if x != y {
                return x < y;
            }
        }
        true
    };
    let rb = root_bound(f);
    let nn = Number::from(n as u64);
    // p₁ and p₂ of each local factor: -g_(d-1) and g_(d-1)² - 2g_(d-2).
    let p1: Vec<Number> = gs
        .iter()
        .map(|g| {
            let d = g.degree().unwrap_or(0);
            (pa - &g.coeff(d - 1)).rem_euclid(pa)
        })
        .collect();
    let p2: Vec<Number> = gs
        .iter()
        .map(|g| {
            let d = g.degree().unwrap_or(0);
            let a = g.coeff(d - 1);
            let b = if d >= 2 {
                g.coeff(d - 2)
            } else {
                Number::default()
            };
            (&(&a * &a) - &(&b << 1usize)).rem_euclid(pa)
        })
        .collect();
    let mut out = Vec::new();
    let mut fcur = f.clone();
    let mut active: Vec<usize> = (0..gs.len()).collect();
    let degs: Vec<usize> = gs.iter().map(|g| g.degree().unwrap_or(0)).collect();
    let consts: Vec<Number> = gs.iter().map(|g| g.coeff(0)).collect();
    let mut s = 1usize;
    'sizes: while 2 * s <= active.len() {
        let lc = fcur.leading().expect("not zero").clone();
        let lc2 = &lc * &lc;
        let lc_f0 = &lc * &fcur.coeff(0);
        let r = active.len();
        // The tests: the values of the active factors by position, and the bound of a sum's
        // symmetric residue; a test whose bound is not below pa/2 tells nothing.
        let b1 = {
            let a = &(&lc * &nn) * &rb;
            if &a < t { a } else { t.clone() }
        };
        let b2 = &(&(&lc2 * &nn) * &rb) * &rb;
        let mut tests: Vec<Test> = Vec::new();
        for (vals, scale, bound) in [(&p1, &lc, b1), (&p2, &lc2, b2)] {
            if bound >= half {
                continue;
            }
            let v: Vec<Vec<u64>> = active.iter().map(|&i| limbs(&(scale * &vals[i]))).collect();
            tests.push((v, limbs(&bound), limbs(&(pa - &bound))));
        }
        let nt = tests.len();
        let t128: Vec<(Vec<u128>, u128, u128)> = if narrow {
            tests
                .iter()
                .map(|(v, lo, hi)| (v.iter().map(|l| wide(l)).collect(), wide(lo), wide(hi)))
                .collect()
        } else {
            Vec::new()
        };
        // The combinations of s of the active factors, as positions in `active`, with the
        // partial sums of their degrees and of the tests' values. When 2s is the number of
        // factors, a subset and its complement are the same split: only those with the first.
        let mut idx: Vec<usize> = (0..s).collect();
        let mut psum = vec![vec![vec![0u64; k]; s + 1]; nt];
        let mut psum128 = vec![vec![0u128; s + 1]; nt];
        let mut pdeg = vec![0usize; s + 1];
        let mut from = 0;
        let last = fcur.length() - 1;
        loop {
            for j in from..s {
                pdeg[j + 1] = pdeg[j] + degs[active[idx[j]]];
                for q in 0..nt {
                    if narrow {
                        let v = psum128[q][j] + t128[q].0[idx[j]];
                        psum128[q][j + 1] = if v >= p128 { v - p128 } else { v };
                    } else {
                        let (lo, hi) = psum[q].split_at_mut(j + 1);
                        hi[0].copy_from_slice(&lo[j]);
                        pm.add(&mut hi[0], &tests[q].0[idx[j]]);
                    }
                }
            }
            let deg = pdeg[s];
            let mut pass = allowed[deg] && deg < last;
            for q in 0..nt {
                if !pass {
                    break;
                }
                pass = if narrow {
                    let v = psum128[q][s];
                    v <= t128[q].1 || v >= t128[q].2
                } else {
                    le(&psum[q][s], &tests[q].1) || le(&tests[q].2, &psum[q][s])
                };
            }
            if pass {
                let mut c0 = lc.rem_euclid(pa);
                for &i in &idx {
                    c0 = (&c0 * &consts[active[i]]).rem_euclid(pa);
                }
                let c0 = sym(c0);
                if !c0.is_zero() && lc_f0.rem_euclid(&c0).is_zero() {
                    // The product of the subset, or of its complement when the subset's
                    // degree is above half: the precision bounds the factors up to half.
                    let inside: Vec<usize> = idx.iter().map(|&i| active[i]).collect();
                    let complement = 2 * deg > last;
                    let chosen: Vec<usize> = if complement {
                        active
                            .iter()
                            .copied()
                            .filter(|i| !inside.contains(i))
                            .collect()
                    } else {
                        inside.clone()
                    };
                    let mut prod = gs[chosen[0]].scale(&lc).c;
                    for &i in &chosen[1..] {
                        prod = zn::mul(&pm, &prod, &gs[i].c);
                    }
                    let cand = ModPolynomial::of(pm.clone(), prod)
                        .to_symmetric()
                        .primitive_part();
                    if let Some(q) = div_exact(&fcur, &cand) {
                        if complement {
                            out.push(q.primitive_part());
                            fcur = cand;
                        } else {
                            out.push(cand);
                            fcur = q;
                        }
                        active.retain(|i| !inside.contains(i));
                        continue 'sizes;
                    }
                }
            }
            // The next combination.
            let mut i = s;
            loop {
                if i == 0 || (i == 1 && 2 * s == r) {
                    s += 1;
                    continue 'sizes;
                }
                i -= 1;
                if idx[i] < r - s + i {
                    idx[i] += 1;
                    for j in i + 1..s {
                        idx[j] = idx[j - 1] + 1;
                    }
                    from = i;
                    break;
                }
            }
        }
    }
    if !fcur.is_constant() {
        out.push(fcur.primitive_part());
    }
    out
}

/// The content with the sign of the leading coefficient, and the primitive part of a positive
/// leading coefficient.
pub(crate) fn split_content(f: &Polynomial<Number>) -> (Number, Polynomial<Number>) {
    let mut c = f.content();
    if f.leading().is_some_and(Number::is_negative) {
        c = -c;
    }
    (c, f.primitive_part())
}

/// The complete factorisation over the integers: the content with the sign of the leading
/// coefficient, and the primitive irreducible factors of positive leading coefficients with their
/// multiplicities, sorted.
pub(crate) fn factor(f: &Polynomial<Number>) -> (Number, Vec<(Polynomial<Number>, u32)>) {
    if f.is_zero() {
        return (Number::default(), Vec::new());
    }
    let (c, g) = split_content(f);
    let mut out = Vec::new();
    // The power of x.
    let z = g.coeffs().iter().take_while(|x| x.is_zero()).count();
    let g = g.shift_right(z);
    if z > 0 {
        out.push((Polynomial::x(), z as u32));
    }
    if !g.is_constant() {
        let parts = if is_squarefree(&g) {
            vec![(g, 1)]
        } else {
            yun(&g)
        };
        for (h, e) in parts {
            for q in factor_squarefree(&h) {
                out.push((q, e));
            }
        }
    }
    sort_factors(&mut out);
    (c, out)
}

/// Whether the polynomial over the integers is irreducible over the rationals: of degree at least
/// 1 and not a product of two of lower degrees (the content is a unit there).
pub(crate) fn is_irreducible(f: &Polynomial<Number>) -> bool {
    let Some(n) = f.degree() else {
        return false;
    };
    if n == 0 {
        return false;
    }
    if n == 1 {
        return true;
    }
    let (_, g) = split_content(f);
    if g.coeff(0).is_zero() || !is_squarefree(&g) {
        return false;
    }
    factor_squarefree(&g).len() == 1
}

/// The modulus check of the methods over `F_p`: panics unless `n` is prime.
#[track_caller]
pub(crate) fn assert_prime(m: &Modulus) {
    let prime = match m.word() {
        Some(p) => is_prime_u64(p),
        None => m.n.is_probable_prime(),
    };
    assert!(prime, "the modulus is not prime");
}

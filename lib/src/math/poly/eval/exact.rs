//! The paths of the integers and the rationals: evaluation point by point on limbs (Horner's rule
//! on word blocks, divide and conquer), the integer product tree on the polynomials' products
//! (`<Number as Coefficient>::poly_mul`), and the rationals brought to the integers by clearing
//! denominators.

use super::tree::{self, Mul};
use crate::math::numbers::{ByteOrder, Number, from_limbs, limbs_of, nat};
use crate::math::poly::Polynomial;
use crate::math::poly::arith::zmul;
use crate::math::rational::Rational;

/// The products of the integer trees.
const ZMUL: Mul<Number> = number_mul;

/// The product of integer polynomials of the trees: the polynomials' products
/// (`<Number as Coefficient>::poly_mul`, a square when both are the same).
fn number_mul(a: &[Number], b: &[Number]) -> Vec<Number> {
    if core::ptr::eq(a, b) {
        zmul::sqr(a)
    } else {
        zmul::mul(a, b)
    }
}

/// The values of the integer polynomial `c` at the integers `xs`, each by [`horner_limbs`]: over
/// the integers the values have about `n·bits(x)` bits, so that the subproduct tree, whose
/// remainders are as large at every level, costs as much as evaluating at each point by divide
/// and conquer, and loses to it by its constants (by 10x at 300 points on an Apple M4).
pub(crate) fn number_eval_many(c: &[Number], xs: &[Number]) -> Vec<Number> {
    horner_limbs(c, xs)
}

/// A signed integer as the limbs of its magnitude, without zeros at the top, and its sign.
struct Signed {
    m: Vec<u64>,
    neg: bool,
}

impl Signed {
    /// The integer `x`.
    fn of(x: &Number) -> Signed {
        Signed {
            m: limbs_of(x),
            neg: x.is_negative(),
        }
    }

    /// The integer as a number.
    fn number(&self) -> Number {
        let x = from_limbs(&self.m, ByteOrder::LittleEndian);
        if self.neg { -x } else { x }
    }

    /// `self += a`.
    fn add(&mut self, a: &Signed) {
        if a.m.is_empty() {
            return;
        }
        if self.m.is_empty() {
            self.m.extend_from_slice(&a.m);
            self.neg = a.neg;
            return;
        }
        if self.neg == a.neg {
            if self.m.len() < a.m.len() {
                self.m.resize(a.m.len(), 0);
            }
            if nat::add_assign(&mut self.m, &a.m) {
                self.m.push(1);
            }
            return;
        }
        // Opposite signs: the larger magnitude less the smaller.
        let ge = self.m.len() > a.m.len()
            || (self.m.len() == a.m.len() && cmp_limbs(&self.m, &a.m) != core::cmp::Ordering::Less);
        if ge {
            nat::sub_assign(&mut self.m, &a.m);
        } else {
            let mut t = a.m.clone();
            nat::sub_assign(&mut t, &self.m);
            self.m = t;
            self.neg = a.neg;
        }
        while self.m.last() == Some(&0) {
            self.m.pop();
        }
    }
}

/// `a` compared with `b`, of the same length.
fn cmp_limbs(a: &[u64], b: &[u64]) -> core::cmp::Ordering {
    for (x, y) in a.iter().rev().zip(b.iter().rev()) {
        if x != y {
            return x.cmp(y);
        }
    }
    core::cmp::Ordering::Equal
}

/// Divide and conquer at a point `x` pays from about this many bits of `xⁿ`, `n` the length:
/// below it, and within it below this many bits of a part, Horner's rule (measured on an Apple
/// M4: Horner is 2x faster at 30 coefficients and a word point, divide and conquer 2.4x faster at
/// 1000).
const Z_DC_BITS: u64 = 12_000;

impl Signed {
    /// `self · x` into `out`, with the working memory `s`.
    fn mul_into(&self, x: &Signed, out: &mut Signed, s: &mut Vec<u64>) {
        out.neg = self.neg ^ x.neg;
        if self.m.is_empty() || x.m.is_empty() {
            out.m.clear();
            return;
        }
        if x.m.len() == 1 {
            out.m.clear();
            out.m.resize(self.m.len(), 0);
            let c = nat::addmul_1(&mut out.m, &self.m, x.m[0]);
            if c != 0 {
                out.m.push(c);
            }
            return;
        }
        if self.m.len() == 1 {
            out.m.clear();
            out.m.resize(x.m.len(), 0);
            let c = nat::addmul_1(&mut out.m, &x.m, self.m[0]);
            if c != 0 {
                out.m.push(c);
            }
            return;
        }
        nat::mul_into(&mut out.m, &self.m, &x.m, s);
        while out.m.last() == Some(&0) {
            out.m.pop();
        }
    }
}

/// The value of the polynomial of the coefficients `c`, not empty, at the point of the powers
/// `pw` (`pw[k] = x^(2^k)`): Horner's rule up to `base` coefficients, otherwise
/// `lo(x) + x^h · hi(x)` with `h` the largest power of two below the length, the two halves in
/// turn.
fn eval_dc(
    c: &[Signed],
    pw: &[Signed],
    base: usize,
    blk: Option<Block>,
    s: &mut Vec<u64>,
    t: &mut Signed,
) -> Signed {
    let len = c.len();
    if len <= base {
        return horner_base(c, &pw[0], blk, s, t);
    }
    let k = (usize::BITS - 1 - (len - 1).leading_zeros()) as usize;
    let h = 1usize << k;
    let lo = eval_dc(&c[..h], pw, base, blk, s, t);
    let hi = eval_dc(&c[h..], pw, base, blk, s, t);
    let mut r = Signed {
        m: Vec::new(),
        neg: false,
    };
    hi.mul_into(&pw[k], &mut r, s);
    r.add(&lo);
    r
}

/// Horner's rule on blocks of `k` coefficients at a point `x` of a word, the coefficients of a
/// word each: `v ← v·xᵏ + Σ c_{j+t} xᵗ`, the sum of a block in 128 bits, so that the long value
/// is multiplied once a block.
#[derive(Clone, Copy)]
struct Block {
    k: usize,
    /// `|x|ᵏ` and whether `xᵏ` is negative.
    y: u64,
    yneg: bool,
    /// `x`.
    x: i128,
}

impl Block {
    /// The blocks for the point `x`, when it is a word of at most 30 bits, not 0 or ±1, and the
    /// coefficients are words (`words`).
    fn of(x: &Signed, words: bool) -> Option<Block> {
        if !words || x.m.len() != 1 || x.m[0] < 2 {
            return None;
        }
        let b = 64 - u64::from(x.m[0].leading_zeros());
        if b > 30 {
            return None;
        }
        // k·b ≤ 64 for the word |x|^k, (k - 1)·b ≤ 60 for the sums in 128 bits.
        let k = (64 / b).min(60 / b + 1) as usize;
        let y = x.m[0].pow(k as u32);
        let xs = if x.neg {
            -i128::from(x.m[0])
        } else {
            i128::from(x.m[0])
        };
        Some(Block {
            k,
            y,
            yneg: x.neg && k % 2 == 1,
            x: xs,
        })
    }
}

/// The value of `c`, not empty, at the point `x`, by Horner's rule, on blocks when `blk`.
fn horner_base(
    c: &[Signed],
    x: &Signed,
    blk: Option<Block>,
    s: &mut Vec<u64>,
    t: &mut Signed,
) -> Signed {
    let len = c.len();
    let word = |a: &Signed| -> i128 {
        let m = i128::from(a.m.first().copied().unwrap_or(0));
        if a.neg { -m } else { m }
    };
    if let Some(b) = blk {
        let k = b.k;
        // The top partial block, then the full ones from the top down.
        let r = len % k;
        let mut v = Signed {
            m: Vec::new(),
            neg: false,
        };
        let sum = |part: &[Signed]| -> i128 {
            let mut acc = 0i128;
            for a in part.iter().rev() {
                acc = acc * b.x + word(a);
            }
            acc
        };
        let groups = c.chunks_exact(k);
        let top = groups.remainder();
        let mut groups = groups.rev();
        if r > 0 {
            v.set_i128(sum(top));
        } else if let Some(g) = groups.next() {
            v.set_i128(sum(g));
        }
        for g in groups {
            v.mul_word(b.y, b.yneg, t);
            let add = Signed::from_i128(sum(g));
            v.add(&add);
        }
        return v;
    }
    let top = &c[len - 1];
    let mut v = Signed {
        m: top.m.clone(),
        neg: top.neg,
    };
    for a in c[..len - 1].iter().rev() {
        if x.m.len() == 1 {
            v.mul_word(x.m[0], x.neg, t);
        } else {
            v.mul_into(x, t, s);
            core::mem::swap(&mut v, t);
        }
        v.add(a);
    }
    v
}

impl Signed {
    /// The integer `x`.
    fn from_i128(x: i128) -> Signed {
        let mut v = Signed {
            m: Vec::new(),
            neg: false,
        };
        v.set_i128(x);
        v
    }

    /// `self = x`.
    fn set_i128(&mut self, x: i128) {
        let u = x.unsigned_abs();
        self.m.clear();
        if u != 0 {
            self.m.push(u as u64);
            if u >> 64 != 0 {
                self.m.push((u >> 64) as u64);
            }
        }
        self.neg = x < 0;
    }

    /// `self ·= ±y`: in place for a short value, by the crate's kernel into `t` for a long one.
    fn mul_word(&mut self, y: u64, neg: bool, t: &mut Signed) {
        self.neg ^= neg;
        if self.m.is_empty() {
            return;
        }
        if self.m.len() < 24 {
            let y = u128::from(y);
            let mut carry = 0u64;
            for l in &mut self.m {
                let p = u128::from(*l) * y + u128::from(carry);
                *l = p as u64;
                carry = (p >> 64) as u64;
            }
            if carry != 0 {
                self.m.push(carry);
            }
            return;
        }
        t.m.clear();
        t.m.resize(self.m.len(), 0);
        let c = nat::addmul_1(&mut t.m, &self.m, y);
        if c != 0 {
            t.m.push(c);
        }
        core::mem::swap(&mut self.m, &mut t.m);
    }
}

/// The values at each point, by [`eval_dc`]: the coefficients converted once, the powers
/// `x^(2^k)` of each point by squaring.
fn horner_limbs(c: &[Number], xs: &[Number]) -> Vec<Number> {
    if c.is_empty() {
        return vec![Number::default(); xs.len()];
    }
    if c.len() == 1 {
        // A constant: its value, in the form of the other values, at every point.
        let v = Signed::of(&c[0]).number();
        return vec![v; xs.len()];
    }
    let cs: Vec<Signed> = c.iter().map(Signed::of).collect();
    let words = cs.iter().all(|a| a.m.len() <= 1);
    let mut s = Vec::new();
    let mut t = Signed {
        m: Vec::new(),
        neg: false,
    };
    let len = c.len();
    xs.iter()
        .map(|x| {
            let bits = x.bits().max(1);
            // Horner's rule throughout when short; otherwise parts of `base` coefficients.
            let base = if (len as u64).saturating_mul(bits) < Z_DC_BITS {
                len
            } else {
                ((Z_DC_BITS / bits) as usize).max(8)
            };
            let levels = if len <= base {
                1
            } else {
                (usize::BITS - (len - 1).leading_zeros()) as usize
            };
            let mut pw = vec![Signed::of(x)];
            while pw.len() < levels {
                let last = pw.last().expect("not empty");
                let mut q = Signed {
                    m: Vec::new(),
                    neg: false,
                };
                if !last.m.is_empty() {
                    nat::sqr_into(&mut q.m, &last.m, &mut s);
                    while q.m.last() == Some(&0) {
                        q.m.pop();
                    }
                }
                pw.push(q);
            }
            let blk = Block::of(&pw[0], words);
            let mut v = eval_dc(&cs, &pw, base, blk, &mut s, &mut t);
            if v.m.is_empty() {
                v.neg = false;
            }
            v.number()
        })
        .collect()
}

/// `Π (x - rᵢ)` over the integers.
///
/// Blocks of [`Z_LINEAR_ROOTS`] roots one linear factor after the other on limbs, then their
/// product tree.
pub(crate) fn number_from_roots(r: &[Number]) -> Vec<Number> {
    if r.is_empty() {
        return vec![Number::from(1u8)];
    }
    if let [x] = r {
        // x - r, in the form of the other products.
        let mut v = Signed::of(x);
        v.neg = !v.neg && !v.m.is_empty();
        return vec![v.number(), Number::from(1u8)];
    }
    let blocks: Vec<Vec<Number>> = r.chunks(Z_LINEAR_ROOTS).map(linear_limbs).collect();
    tree::product(blocks, ZMUL)
}

/// The roots a block of [`number_from_roots`] takes one after the other (measured on an Apple
/// M4 against the product tree of single factors).
const Z_LINEAR_ROOTS: usize = 32;

/// `Π (x - rᵢ)` one linear factor after the other on limbs, `r ← r·(x - a)`: two vectors of the
/// coefficients in turn, `new[j] = old[j-1] - a·old[j]`, without allocating past the first.
fn linear_limbs(r: &[Number]) -> Vec<Number> {
    if let Some(v) = linear_words(r) {
        return v;
    }
    let zero = || Signed {
        m: Vec::new(),
        neg: false,
    };
    let mut old: Vec<Signed> = (0..=r.len()).map(|_| zero()).collect();
    let mut new: Vec<Signed> = (0..=r.len()).map(|_| zero()).collect();
    old[0].m.push(1);
    let mut t = zero();
    let mut s = Vec::new();
    for (i, a) in r.iter().enumerate() {
        let mut a = Signed::of(a);
        a.neg = !a.neg;
        // new = old·(x - a) = x·old + (-a)·old, old of i + 1 coefficients.
        for j in 0..=i + 1 {
            let n = &mut new[j];
            n.m.clear();
            n.neg = false;
            if j <= i && !old[j].m.is_empty() && !a.m.is_empty() {
                if a.m.len() == 1 {
                    n.m.extend_from_slice(&old[j].m);
                    n.neg = old[j].neg;
                    n.mul_word(a.m[0], a.neg, &mut t);
                } else {
                    old[j].mul_into(&a, n, &mut s);
                }
            }
            if j > 0 {
                n.add(&old[j - 1]);
            }
        }
        core::mem::swap(&mut old, &mut new);
    }
    old.iter()
        .map(|c| {
            if c.m.is_empty() {
                Number::default()
            } else {
                c.number()
            }
        })
        .collect()
}

/// [`linear_limbs`] for roots below `2^64` in magnitude: the coefficients in two's complement
/// in `l` limbs each, one vector, `l` from the bound `Π (1 + |rᵢ|)` on them, each step
/// `r_j ← r_(j-1) ∓ |a|·r_j` from the top by one pass of products by a word; `None` for a
/// larger root.
fn linear_words(r: &[Number]) -> Option<Vec<Number>> {
    let mut roots = Vec::with_capacity(r.len());
    let mut bits = 1u64;
    for x in r {
        if x.bits() > 64 {
            return None;
        }
        let m = limbs_of(x).first().copied().unwrap_or(0);
        bits += 65 - u64::from(m.leading_zeros()).min(64);
        roots.push((m, x.is_negative()));
    }
    let l = bits.div_ceil(64) as usize;
    let len = r.len() + 1;
    let mut c = vec![0u64; len * l];
    c[0] = 1;
    // The limbs the coefficients take so far (the bound of the product of the roots so far),
    // the others the sign's extension once they are needed.
    let (mut cur, mut sofar) = (1usize, 1u64);
    for (i, &(a, neg)) in roots.iter().enumerate() {
        sofar += 65 - u64::from(a.leading_zeros()).min(64);
        let need = sofar.div_ceil(64) as usize;
        if need > cur {
            for x in c[..(i + 1) * l].chunks_exact_mut(l) {
                let ext = if x[cur - 1] >> 63 == 1 { u64::MAX } else { 0 };
                x[cur..need].fill(ext);
            }
            cur = need;
        }
        // r·(x - a): r_(i+1) = r_i, r_j = r_(j-1) - a r_j, r_0 = -a r_0, modulo 2^(64 cur).
        c.copy_within(i * l..(i + 1) * l, (i + 1) * l);
        for j in (0..=i).rev() {
            let (lo, hi) = c.split_at_mut(j * l);
            let rj = &mut hi[..cur];
            // rj = -a·rj (modulo 2^(64 l), two's complement), then + r_(j-1).
            let mut carry = 0u64;
            for x in rj.iter_mut() {
                let t = u128::from(*x) * u128::from(a) + u128::from(carry);
                *x = t as u64;
                carry = (t >> 64) as u64;
            }
            if !neg {
                negate(rj);
            }
            if j > 0 {
                let mut c = false;
                for (x, &y) in rj.iter_mut().zip(&lo[(j - 1) * l..]) {
                    let (t, c1) = x.overflowing_add(y);
                    let (t, c2) = t.overflowing_add(u64::from(c));
                    *x = t;
                    c = c1 | c2;
                }
            }
        }
    }
    for x in c.chunks_exact_mut(l) {
        let ext = if x[cur - 1] >> 63 == 1 { u64::MAX } else { 0 };
        x[cur..].fill(ext);
    }
    Some(
        c.chunks_exact_mut(l)
            .map(|x| {
                let neg = x[l - 1] >> 63 == 1;
                if neg {
                    negate(x);
                }
                let v = from_limbs(x, ByteOrder::LittleEndian);
                if neg { -v } else { v }
            })
            .collect(),
    )
}

/// `x = -x` modulo `2^(64 len)`.
#[inline]
fn negate(x: &mut [u64]) {
    let mut c = true;
    for v in x.iter_mut() {
        let (t, o) = (!*v).overflowing_add(u64::from(c));
        *v = t;
        c = o;
    }
}

/// The least common multiple of the denominators of `xs`, 1 for none.
fn lcm_denoms(xs: &[Rational]) -> Number {
    let mut l = Number::from(1u8);
    for x in xs {
        let d = x.denom();
        if !crate::math::rational::is_one(d) && !(&l % d).is_zero() {
            l = l.lcm(d);
        }
    }
    l
}

/// The values of the rational polynomial `c` at the rationals `xs`: `c = k·P` for a primitive
/// integer `P`; at integer points `k·P(x)` by the integer trees; at points of the common
/// denominator `L`, `k·Q(L x) / L^d` for the integer `Q(z) = L^d P(z / L)`; otherwise, by Horner's
/// rule on the homogenised `P`, `k·Σ Pⱼ aʲ b^(d-j) / b^d` for each `a / b`.
pub(crate) fn rational_eval_many(c: &[Rational], xs: &[Rational]) -> Vec<Rational> {
    if c.is_empty() || xs.is_empty() {
        return vec![Rational::default(); xs.len()];
    }
    if c.len() == 1 {
        return vec![c[0].clone(); xs.len()];
    }
    let (k, p) = Polynomial::from_slice(c).to_primitive();
    let p = p.into_coeffs();
    let d = p.len() - 1;
    let l = lcm_denoms(xs);
    let maxden = xs.iter().map(|x| x.denom().bits()).max().unwrap_or(0);
    if crate::math::rational::is_one(&l) {
        let pts: Vec<Number> = xs.iter().map(|x| x.numer().clone()).collect();
        return number_eval_many(&p, &pts)
            .into_iter()
            .map(|v| &k * &Rational::from(v))
            .collect();
    }
    if l.bits() <= 2 * maxden + 64 {
        // Q_j = P_j L^(d - j), the points L x.
        let mut q = vec![Number::default(); d + 1];
        let mut pw = Number::from(1u8);
        for j in (0..=d).rev() {
            q[j] = &p[j] * &pw;
            if j > 0 {
                pw = &pw * &l;
            }
        }
        // pw = L^d now.
        let pts: Vec<Number> = xs.iter().map(|x| x.numer() * &(&l / x.denom())).collect();
        return number_eval_many(&q, &pts)
            .into_iter()
            .map(|v| &k * &Rational::new(v, pw.clone()))
            .collect();
    }
    xs.iter()
        .map(|x| {
            let (a, b) = (x.numer(), x.denom());
            // v = Σ P_j a^j b^(d-j), by Horner: v ← v·a + P_j·b^(d-j), the powers of b kept.
            let mut v = p[d].clone();
            let mut bp = Number::from(1u8);
            for j in (0..d).rev() {
                bp = &bp * b;
                v = &(&v * a) + &(&p[j] * &bp);
            }
            &k * &Rational::new(v, bp)
        })
        .collect()
}

/// `Π (x - rᵢ)` over the rationals: `Π (bᵢ x - aᵢ) / Π bᵢ` for `rᵢ = aᵢ / bᵢ`, the product by the
/// integer tree.
pub(crate) fn rational_from_roots(r: &[Rational]) -> Vec<Rational> {
    if r.is_empty() {
        return vec![Rational::from(1u8)];
    }
    let lin: Vec<Vec<Number>> = r
        .iter()
        .map(|x| vec![-x.numer(), x.denom().clone()])
        .collect();
    let p = tree::product(lin, ZMUL);
    let lead = p.last().expect("not empty").clone();
    p.into_iter()
        .map(|c| Rational::new(c, lead.clone()))
        .collect()
}

/// The interpolating polynomial over the rationals of the points `xs`, distinct, and the values
/// `ys`: with `L` the common denominator of the points, the integer points `Xᵢ = L xᵢ`, their tree
/// and the weights `Wᵢ = M'(Xᵢ)` by the integer scaled remainder tree, then the linear
/// combination `Σ yᵢ/Wᵢ · M/(z - Xᵢ)` up the tree as integer polynomials over a denominator each,
/// and `P(x) = Q(L x)`.
#[track_caller]
pub(crate) fn rational_interpolate(xs: &[Rational], ys: &[Rational]) -> Vec<Rational> {
    let m = xs.len();
    if m == 0 {
        return Vec::new();
    }
    let l = lcm_denoms(xs);
    let pts: Vec<Number> = if crate::math::rational::is_one(&l) {
        xs.iter().map(|x| x.numer().clone()).collect()
    } else {
        xs.iter().map(|x| x.numer() * &(&l / x.denom())).collect()
    };
    let (num, den) = if m == 1 {
        (vec![ys[0].numer().clone()], ys[0].denom().clone())
    } else {
        let t = tree::build(&pts, ZMUL);
        let w = weights(&pts);
        // The leaves y_i / W_i as n_i / d_i.
        let leaves: Vec<(Number, Number)> = ys
            .iter()
            .zip(&w)
            .enumerate()
            .map(|(i, (y, w))| {
                if w.is_zero() {
                    panic!("the point {i} is repeated: the points must be distinct");
                }
                let q = y / &Rational::from(w.clone());
                (q.numer().clone(), q.denom().clone())
            })
            .collect();
        combine_fractions(&t, leaves)
    };
    // P_j = Q_j L^j / D.
    let mut pw = Number::from(1u8);
    let one = crate::math::rational::is_one(&l);
    num.into_iter()
        .map(|c| {
            let r = Rational::new(&c * &pw, den.clone());
            if !one {
                pw = &pw * &l;
            }
            r
        })
        .collect()
}

/// The weights `Wᵢ = Π_{j≠i} (Xᵢ - Xⱼ)` of the integer points `pts` (`M'(Xᵢ)`): for each point
/// the differences multiplied together in a word while their product fits it, each such word
/// into a limb vector in place while it is short, its parts then by a product tree; 0 for a
/// repeated point. Over the integers this is cheaper than the values of `M'` by the trees, whose
/// coefficients are as large as the weights at every level.
fn weights(pts: &[Number]) -> Vec<Number> {
    let small: Option<Vec<i64>> = pts
        .iter()
        .map(|x| {
            let b = x.bits();
            (b <= 62).then(|| {
                let m = limbs_of(x).first().copied().unwrap_or(0) as i64;
                if x.is_negative() { -m } else { m }
            })
        })
        .collect();
    let mut s = Vec::new();
    let mut acc: Vec<u64> = Vec::new();
    (0..pts.len())
        .map(|i| {
            let mut parts: Vec<Vec<u64>> = Vec::new();
            acc.clear();
            acc.push(1);
            let mut neg = false;
            // acc ← acc·w in place; a part when it is long.
            let mul1 = |acc: &mut Vec<u64>, parts: &mut Vec<Vec<u64>>, w: u64| {
                let mut c = 0u64;
                for l in acc.iter_mut() {
                    let t = u128::from(*l) * u128::from(w) + u128::from(c);
                    *l = t as u64;
                    c = (t >> 64) as u64;
                }
                if c != 0 {
                    acc.push(c);
                }
                if acc.len() >= 32 {
                    parts.push(core::mem::replace(acc, vec![1]));
                }
            };
            let mut w = 1u64;
            for j in 0..pts.len() {
                if j == i {
                    continue;
                }
                match &small {
                    Some(v) => {
                        let d = v[i] - v[j];
                        if d == 0 {
                            return Number::default();
                        }
                        neg ^= d < 0;
                        let a = d.unsigned_abs();
                        match w.checked_mul(a) {
                            Some(t) => w = t,
                            None => {
                                mul1(&mut acc, &mut parts, w);
                                w = a;
                            }
                        }
                    }
                    None => {
                        let d = &pts[i] - &pts[j];
                        if d.is_zero() {
                            return Number::default();
                        }
                        neg ^= d.is_negative();
                        let l = limbs_of(&d);
                        if l.len() == 1 {
                            mul1(&mut acc, &mut parts, l[0]);
                        } else {
                            parts.push(l);
                        }
                    }
                }
            }
            mul1(&mut acc, &mut parts, w);
            parts.push(acc.clone());
            // The product of the parts, pairwise.
            while parts.len() > 1 {
                let mut next = Vec::with_capacity(parts.len().div_ceil(2));
                for p in parts.chunks(2) {
                    match p {
                        [a, b] => {
                            let mut r = Vec::new();
                            nat::mul_into(&mut r, a, b, &mut s);
                            while r.last() == Some(&0) {
                                r.pop();
                            }
                            next.push(r);
                        }
                        [a] => next.push(a.clone()),
                        _ => unreachable!(),
                    }
                }
                parts = next;
            }
            let x = from_limbs(&parts[0], ByteOrder::LittleEndian);
            if neg { -x } else { x }
        })
        .collect()
}

/// `Σ (nᵢ / dᵢ) · M / (z - Xᵢ)` as an integer polynomial over a positive denominator: going up
/// the tree `t`, `N_v / D_v = (N_u (D/D_u) M_w + N_w (D/D_w) M_u) / D` with `D = lcm(D_u, D_w)`.
fn combine_fractions(
    t: &tree::Tree<Number>,
    leaves: Vec<(Number, Number)>,
) -> (Vec<Number>, Number) {
    let unit = |f: &Number| crate::math::rational::is_one(f);
    let mut level: Vec<(Vec<Number>, Number)> =
        leaves.into_iter().map(|(n, d)| (vec![n], d)).collect();
    let mut l = 0;
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut it = level.into_iter().enumerate();
        while let Some((u, (nu, du))) = it.next() {
            let Some((w, (nw, dw))) = it.next() else {
                next.push((nu, du));
                break;
            };
            let (mu, mw) = (t.node(l, u), t.node(l, w));
            let g = du.gcd(&dw);
            let (fu, fw) = (&dw / &g, &du / &g);
            let d = &du * &fu;
            let dv = mu.len() + mw.len() - 2;
            let mut a = ZMUL(&nu, &mw);
            let b = ZMUL(&nw, &mu);
            debug_assert!(a.len() == dv && b.len() == dv);
            for (x, y) in a.iter_mut().zip(&b) {
                if !unit(&fu) {
                    *x *= &fu;
                }
                if unit(&fw) {
                    *x += y;
                } else {
                    *x += &(y * &fw);
                }
            }
            next.push((a, d));
        }
        level = next;
        l += 1;
    }
    level.pop().expect("the root")
}

//! Dixon's p-adic lifting (Dixon 1982), for the solutions of large systems of integers with a few
//! right-hand sides: `A` is factored once modulo a prime `p` of 62 bits, and each step solves
//! `A·x ≡ r (mod p)` by the factors and lifts the right-hand side, `r ← (r - A·x) / p`, exactly;
//! after `L` steps `X = Σ x_i·pⁱ` is the solution modulo `p^L`, and each element is the one
//! fraction `y / d` with `|y| ≤ N` and `0 < d ≤ D`, Hadamard's bounds on Cramer's numerators and
//! the determinant, that it is congruent to (rational reconstruction), once `p^L > 2·N·D`.
//!
//! The work is `n³/3` products modulo `p` for the factors and, for each of the `L` steps and
//! each right-hand side, `n²` products modulo `p` and `n²` products of the matrix's integers by
//! words: `O(n³ + n²·L)` operations on words, against Bareiss's `n³/3` products of numbers as
//! long as the minors. The denominators are found by reconstructing the first element alone,
//! as a rule, and the others from the common denominator so far.

use super::{Acc, B, Exact, Int, S, Work, combine, mul, trim};
use crate::math::numbers::{ByteOrder, Number, nat};
use core::cmp::Ordering;

/// Primes just below `2^62`: their products of two fit in 124 bits, sums of four of those below
/// `p·2^64`, as Montgomery's reduction needs.
const PRIMES: [u64; 4] = [
    0x3fff_ffff_ffff_ffc7,
    0x3fff_ffff_ffff_ffa9,
    0x3fff_ffff_ffff_ff8b,
    0x3fff_ffff_ffff_ff71,
];

/// The bits each step gains: `p > 2^61`.
const STEP_BITS: u64 = 61;

/// The integers modulo the prime `p`, in Montgomery's form `x·2^64 mod p`.
#[derive(Clone, Copy)]
struct Field {
    p: u64,
    /// `-p⁻¹ mod 2^64`.
    neg_inv: u64,
    /// `2^128 mod p`.
    r2: u64,
}

impl Field {
    fn new(p: u64) -> Field {
        let r = ((1u128 << 64) % p as u128) as u64;
        Field {
            p,
            neg_inv: super::inverse(p).wrapping_neg(),
            r2: ((r as u128 * r as u128) % p as u128) as u64,
        }
    }

    /// `t·2^-64 mod p` for `t < p·2^64`.
    #[inline(always)]
    fn redc(self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.neg_inv);
        // `t + m·p < 2^126 + 2^126`, and divisible by `2^64`.
        let u = ((t + m as u128 * self.p as u128) >> 64) as u64;
        if u >= self.p { u - self.p } else { u }
    }

    #[inline(always)]
    fn mul(self, a: u64, b: u64) -> u64 {
        self.redc(a as u128 * b as u128)
    }

    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        if a >= b { a - b } else { a + self.p - b }
    }

    /// The form of `a < p`.
    #[inline(always)]
    fn to(self, a: u64) -> u64 {
        self.mul(a, self.r2)
    }

    /// The value of the form `a`.
    #[inline(always)]
    fn from(self, a: u64) -> u64 {
        self.redc(a as u128)
    }

    /// The inverse of the form `a`, not zero, as a form: `a^(p-2)`.
    fn inv(self, a: u64) -> u64 {
        let mut e = self.p - 2;
        let mut base = a;
        let mut x = self.to(1);
        while e > 0 {
            if e & 1 == 1 {
                x = self.mul(x, base);
            }
            base = self.mul(base, base);
            e >>= 1;
        }
        x
    }

    /// The form of the integer `x` modulo `p`.
    fn of(self, x: &Int) -> u64 {
        match x {
            S(v) => self.to(v.rem_euclid(self.p as i64) as u64),
            B(b) => {
                // Horner from the top: `acc ← acc·2^64 + limb`, in the forms, `acc·2^64` being
                // the product by the form of `2^64`, `2^128 mod p`.
                let mut acc = 0;
                for &l in b.d.iter().rev() {
                    acc = self.mul(acc, self.r2);
                    let l = self.to(l % self.p);
                    acc = if acc + l >= self.p {
                        acc + l - self.p
                    } else {
                        acc + l
                    };
                }
                if b.neg && acc != 0 { self.p - acc } else { acc }
            }
        }
    }

    /// `Σ a_j·b_j` of forms, the products summed four at a time before a reduction.
    #[inline(always)]
    fn dot(self, a: &[u64], b: &[u64]) -> u64 {
        let mut s = 0u64;
        let n = a.len().min(b.len());
        let (ca, ra) = a[..n].as_chunks::<4>();
        let (cb, rb) = b[..n].as_chunks::<4>();
        for (x, y) in ca.iter().zip(cb) {
            let t = x[0] as u128 * y[0] as u128
                + x[1] as u128 * y[1] as u128
                + x[2] as u128 * y[2] as u128
                + x[3] as u128 * y[3] as u128;
            let r = self.redc(t);
            s = if s + r >= self.p {
                s + r - self.p
            } else {
                s + r
            };
        }
        let mut t = 0u128;
        for (x, y) in ra.iter().zip(rb) {
            t += *x as u128 * *y as u128;
        }
        let r = self.redc(t);
        if s + r >= self.p {
            s + r - self.p
        } else {
            s + r
        }
    }
}

/// The factors `P·A = L·U` of a square matrix modulo `p`, in the forms, row after row: `L` below
/// the diagonal with a unit diagonal, `U` on and above, the inverses of `U`'s diagonal, and the
/// row each row of `P·A` came from.
struct Factors {
    f: Field,
    n: usize,
    lu: Vec<u64>,
    inv: Vec<u64>,
    perm: Vec<usize>,
    /// Whether the rows were swapped an odd number of times.
    odd: bool,
}

impl Factors {
    /// The factors of the `n × n` matrix whose rows start every `stride` elements of `a`, or
    /// `None` when it is singular modulo `p`.
    fn new(f: Field, a: &[Int], n: usize, stride: usize) -> Option<Factors> {
        let mut lu: Vec<u64> = (0..n)
            .flat_map(|i| a[i * stride..i * stride + n].iter().map(|x| f.of(x)))
            .collect();
        let mut perm: Vec<usize> = (0..n).collect();
        let mut inv = Vec::with_capacity(n);
        let mut odd = false;
        for k in 0..n {
            let p = (k..n).find(|&i| lu[i * n + k] != 0)?;
            if p != k {
                let (x, y) = lu.split_at_mut(p * n);
                x[k * n..(k + 1) * n].swap_with_slice(&mut y[..n]);
                perm.swap(p, k);
                odd = !odd;
            }
            let d = f.inv(lu[k * n + k]);
            inv.push(d);
            let (top, rest) = lu.split_at_mut((k + 1) * n);
            let pivot = &top[k * n + k + 1..(k + 1) * n];
            for row in rest.chunks_exact_mut(n) {
                let l = f.mul(row[k], d);
                row[k] = l;
                if l != 0 {
                    for (x, &u) in row[k + 1..].iter_mut().zip(pivot) {
                        *x = f.sub(*x, f.mul(l, u));
                    }
                }
            }
        }
        Some(Factors {
            f,
            n,
            lu,
            inv,
            perm,
            odd,
        })
    }

    /// The determinant modulo `p`, below `p`.
    fn det(&self) -> u64 {
        let (f, n) = (self.f, self.n);
        let d = (0..n).fold(f.to(1), |d, i| f.mul(d, self.lu[i * n + i]));
        let d = f.from(d);
        if self.odd && d != 0 { f.p - d } else { d }
    }

    /// The solution of `A·x ≡ b`, `b` in the forms, written over `x` as values below `p`.
    fn solve(&self, b: &[u64], y: &mut [u64]) {
        let (f, n) = (self.f, self.n);
        for i in 0..n {
            let s = f.dot(&self.lu[i * n..i * n + i], &y[..i]);
            y[i] = f.sub(b[self.perm[i]], s);
        }
        for i in (0..n).rev() {
            let s = f.dot(&self.lu[i * n + i + 1..(i + 1) * n], &y[i + 1..]);
            y[i] = f.mul(f.sub(y[i], s), self.inv[i]);
        }
        for x in y.iter_mut() {
            *x = f.from(*x);
        }
    }
}

/// A row of the matrix for the lifting: its integers when all are in registers, or else their
/// magnitudes in `w` limbs each and their signs, for sums of products by words limb by limb.
enum Row {
    Words(Vec<i64>),
    Limbs {
        w: usize,
        mags: Vec<u64>,
        negs: Vec<bool>,
    },
}

impl Row {
    fn new(a: &[Int]) -> Row {
        if a.iter().all(|x| matches!(x, S(_))) {
            return Row::Words(
                a.iter()
                    .map(|x| if let S(v) = x { *v } else { 0 })
                    .collect(),
            );
        }
        let w = a
            .iter()
            .map(|x| if let B(b) = x { b.d.len() } else { 1 })
            .max()
            .unwrap_or(1);
        let mut mags = vec![0u64; a.len() * w];
        let mut negs = Vec::with_capacity(a.len());
        for (j, x) in a.iter().enumerate() {
            let mut buf = [0];
            let (neg, m) = x.mag(&mut buf);
            mags[j * w..j * w + m.len()].copy_from_slice(m);
            negs.push(neg);
        }
        Row::Limbs { w, mags, negs }
    }
}

/// Working memory of the lifting: the positive and the negative sums of a row.
#[derive(Default)]
struct Sums {
    pos: Vec<u64>,
    neg: Vec<u64>,
}

/// `acc += a·x` for the `w` limbs of `a` and the word `x`, `acc` of `w + 2` limbs.
#[inline(always)]
fn addmul(acc: &mut [u64], a: &[u64], x: u64) {
    let w = a.len();
    let carry = if w <= 2 {
        let mut carry = 0u128;
        for (s, &l) in acc.iter_mut().zip(a) {
            let t = l as u128 * x as u128 + *s as u128 + carry;
            *s = t as u64;
            carry = t >> 64;
        }
        carry as u64
    } else {
        crate::math::numbers::nat::addmul_1(&mut acc[..w], a, x)
    };
    let (t, c) = acc[w].overflowing_add(carry);
    acc[w] = t;
    acc[w + 1] += c as u64;
}

/// `r ← (r - Σ a_j·x_j) / p` for a row `a` of the matrix's integers, the words `x` below `p` and
/// `r` an integer, the division exact.
fn lift(r: &mut Int, a: &Row, x: &[u64], p: &mut Exact, w: &mut Work, sums: &mut Sums) {
    match a {
        Row::Words(a) => {
            // Each product is below `2^125`; the sum is checked.
            let mut acc = Acc::of(r);
            for (&a, &x) in a.iter().zip(x) {
                acc.sub(&S(a), &S(x as i64), w);
            }
            *r = acc.div(p, w);
        }
        Row::Limbs {
            w: width,
            mags,
            negs,
        } => {
            let len = width + 2;
            sums.pos.clear();
            sums.pos.resize(len, 0);
            sums.neg.clear();
            sums.neg.resize(len, 0);
            for ((m, &neg), &x) in mags.chunks_exact(*width).zip(negs).zip(x) {
                if x != 0 {
                    addmul(if neg { &mut sums.neg } else { &mut sums.pos }, m, x);
                }
            }
            // `r - pos + neg`.
            trim(&mut sums.pos);
            trim(&mut sums.neg);
            let mut t = match r {
                S(v) => {
                    let m = v.unsigned_abs();
                    (*v < 0, if m == 0 { Vec::new() } else { vec![m] })
                }
                B(b) => (b.neg, core::mem::take(&mut b.d)),
            };
            if !sums.pos.is_empty() {
                t.0 = if t.1.is_empty() {
                    t.1.extend_from_slice(&sums.pos);
                    true
                } else {
                    combine(&mut t.1, t.0, &sums.pos, false)
                };
            }
            if !sums.neg.is_empty() {
                t.0 = if t.1.is_empty() {
                    t.1.extend_from_slice(&sums.neg);
                    false
                } else {
                    combine(&mut t.1, t.0, &sums.neg, true)
                };
            }
            *r = if t.1.is_empty() {
                S(0)
            } else {
                p.limbs(&mut t.1, t.0, Vec::new(), w)
            };
        }
    }
}

/// The squared Euclidean lengths of the columns of the `rows × n` part of `a` held row after row
/// every `stride` elements, from column `from`, as numbers.
fn squares(a: &[Int], rows: usize, stride: usize, from: usize, n: usize) -> Vec<Number> {
    (0..n)
        .map(|j| {
            (0..rows).fold(Number::from(0u8), |s, i| {
                let x = a[i * stride + from + j].number(ByteOrder::LittleEndian);
                s + &x * &x
            })
        })
        .collect()
}

/// The solution of `A·X = B` for the `n × n` matrix of integers `A` beside the `n × k` one `B`,
/// held row after row in `m`: its elements column after column as fractions `y / d` with `d`
/// positive, not always in lowest terms, and the least common multiple of their denominators,
/// a divisor of `det(A)`; or `None` when
/// `A` is singular modulo each of the primes tried, which the caller then leaves to Bareiss's
/// elimination (a singular `A` is singular modulo every prime).
pub(super) fn solve(m: &[Int], n: usize, k: usize) -> Option<(Vec<(Int, Int)>, Int)> {
    let stride = n + k;
    let factors = PRIMES
        .iter()
        .find_map(|&p| Factors::new(Field::new(p), m, n, stride))?;
    let rhs: Vec<Vec<Int>> = (0..k)
        .map(|c| (0..n).map(|i| m[i * stride + n + c].clone()).collect())
        .collect();
    solve_by(&factors, m, n, stride, rhs)
}

/// The determinant of the `n × n` matrix of integers held row after row every `stride` elements
/// of `m`: the least common multiple `l` of the denominators of the solution of
/// `A·x = b` for a fixed pseudorandom `b`, which divides `det(A)` and is most of it as a rule,
/// then `det(A) / l` from the determinants modulo primes, by the Chinese remainder theorem, until
/// their product passes twice Hadamard's bound over `l` (one prime, as a rule). `None` when `A` is
/// singular modulo each of the first primes tried, which the caller leaves to Bareiss's
/// elimination.
pub(super) fn det(m: &[Int], n: usize, stride: usize) -> Option<Int> {
    let factors = PRIMES
        .iter()
        .find_map(|&p| Factors::new(Field::new(p), m, n, stride))?;
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let b: Vec<Int> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            S((state % 19) as i64 - 9)
        })
        .collect();
    let (_, l) = solve_by(&factors, m, n, stride, vec![b])?;
    let sq = squares(m, n, stride, 0, n);
    let dbound = sq.iter().fold(Number::from(1u8), |s, x| s * x).isqrt() + 1u8;
    let ln = l.number(ByteOrder::LittleEndian);
    let bound = &dbound / &ln + 1u8;
    // `c = det / l` modulo the primes, combined.
    let residue = |fs: &Factors| {
        let f = fs.f;
        let li = f.inv(f.of(&l));
        f.from(f.mul(f.to(fs.det()), li))
    };
    let mut c = Number::from(residue(&factors));
    let mut q = Number::from(factors.f.p);
    let mut p = Number::from(*PRIMES.last().unwrap());
    while q <= &bound * 2u8 {
        p = p.prev_prime()?;
        let pw = u64::try_from(&p).ok()?;
        let f = Field::new(pw);
        if f.of(&l) == 0 {
            continue;
        }
        let r = match Factors::new(f, m, n, stride) {
            Some(fs) => residue(&fs),
            None => 0,
        };
        // Garner: `c + q·((r - c)·q⁻¹ mod p)`.
        let cm = f.of(&Int::of(&c));
        let qi = f.inv(f.of(&Int::of(&q)));
        let t = f.from(f.mul(f.sub(f.to(r), cm), qi));
        c = &c + &(&q * t);
        q *= pw;
    }
    let half = &q >> 1usize;
    let c = if c > half { c - &q } else { c };
    Some(mul(&Int::of(&c), &l))
}

/// The solution of `A·X = B` with the factors of `A` modulo a prime, for the right-hand sides
/// `rhs`, as [`solve`] gives it.
fn solve_by(
    factors: &Factors,
    m: &[Int],
    n: usize,
    stride: usize,
    rhs: Vec<Vec<Int>>,
) -> Option<(Vec<(Int, Int)>, Int)> {
    let k = rhs.len();
    let p = factors.f.p;
    // Hadamard's bounds by columns: `D² = Π |a_j|²` for the determinant, `N_c² = D²·|b_c|² /
    // min |a_j|²` for the numerators of column `c`; then the steps for `p^L > 2·N·D`.
    let sq = squares(m, n, stride, 0, n);
    let d2 = sq.iter().fold(Number::from(1u8), |s, x| s * x);
    // `min |a_j|² ≥ 2^(least - 1)`, so `min |a_j| ≥ 2^((least - 1) / 2)`.
    let least = sq.iter().map(Number::bits).min().unwrap_or(1).max(1);
    let b2 = rhs
        .iter()
        .map(|c| {
            c.iter().fold(Number::from(0u8), |s, x| {
                let x = x.number(ByteOrder::LittleEndian);
                s + &x * &x
            })
        })
        .max()
        .unwrap_or_default();
    let dbound = d2.isqrt() + 1u8;
    let nbound = (&d2 * &b2).isqrt() / (Number::from(1u8) << ((least - 1) / 2) as usize) + 1u8;
    // `p^L > 2^(61 L) ≥ 2^(bits(N) + bits(D) + 1) > 2·N·D`.
    let steps = (nbound.bits() + dbound.bits() + 1).div_ceil(STEP_BITS) as usize;
    let mut r = rhs;
    // The digits of each column, step after step.
    let mut digits: Vec<Vec<u64>> = vec![Vec::with_capacity(steps * n); k];
    let mut exact = Exact::new(&S(p as i64));
    let mut w = Work::default();
    let (mut b, mut x) = (vec![0u64; n], vec![0u64; n]);
    let rows: Vec<Row> = (0..n)
        .map(|i| Row::new(&m[i * stride..i * stride + n]))
        .collect();
    let mut sums = Sums::default();
    for _ in 0..steps {
        for c in 0..k {
            for (bi, ri) in b.iter_mut().zip(&r[c]) {
                *bi = factors.f.of(ri);
            }
            factors.solve(&b, &mut x);
            for (ri, row) in r[c].iter_mut().zip(&rows) {
                lift(ri, row, &x, &mut exact, &mut w, &mut sums);
            }
            digits[c].extend_from_slice(&x);
        }
    }
    // `X = Σ x_i·pⁱ` modulo `M = p^L`, element by element, then the fractions.
    let modulus = (0..steps).fold(Number::from(1u8), |s, _| s * p);
    let mut den = Number::from(1u8);
    let mut powers = vec![vec![p]];
    while 1 << powers.len() < steps {
        let last = powers.last().unwrap();
        let mut sq = Vec::new();
        nat::mul_into(&mut sq, last, last, &mut w.s);
        trim(&mut sq);
        powers.push(sq);
    }
    let mut ys: Vec<Number> = Vec::with_capacity(n * k);
    let mut column = vec![0u64; steps];
    for digits in &digits {
        for i in 0..n {
            for (s, d) in column.iter_mut().enumerate() {
                *d = digits[s * n + i];
            }
            let v = value(&column, &powers, &mut w.s);
            ys.push(Int::of_limbs(false, v).number(ByteOrder::LittleEndian));
        }
    }
    // The fractions: each element times the common denominator so far, symmetric modulo `M`,
    // when it is within `N` times that denominator; otherwise reconstructed, with the bounds
    // `N·den` and `D / den`, under which the fraction is unique as `M > 2·N·D`.
    let half = &modulus >> 1usize;
    let mut out = Vec::with_capacity(n * k);
    for x in ys {
        let t = (&x * &den).rem_euclid(&modulus);
        let ts = if t > half { &t - &modulus } else { t.clone() };
        if ts.abs() <= &nbound * &den {
            out.push((Int::of(&ts), Int::of(&den)));
            continue;
        }
        let dmax = &dbound / &den;
        let (y, e) = reconstruct(&t, &modulus, &(&nbound * &den), &dmax)?;
        let g = y.gcd(&e);
        let (y, e) = if g == 1 {
            (y, e)
        } else {
            (y.div_exact(&g), e.div_exact(&g))
        };
        den *= &e;
        out.push((Int::of(&y), Int::of(&den)));
    }
    Some((out, Int::of(&den)))
}

/// The value `Σ d_i·pⁱ` of the digits `d` below `p`, for `powers[j] = p^(2^j)`: by halves, the
/// low half plus the high half times a power, down to 16 digits by Horner's rule; `O(M(L) log L)`
/// for `L` digits, against `L²/2` limb products by Horner's rule alone.
fn value(d: &[u64], powers: &[Vec<u64>], s: &mut Vec<u64>) -> Vec<u64> {
    if d.len() <= 16 {
        let p = powers[0][0];
        let mut v: Vec<u64> = Vec::with_capacity(d.len() + 1);
        for &digit in d.iter().rev() {
            let mut carry = digit as u128;
            for l in v.iter_mut() {
                let t = *l as u128 * p as u128 + carry;
                *l = t as u64;
                carry = t >> 64;
            }
            if carry != 0 {
                v.push(carry as u64);
            }
        }
        return v;
    }
    // The largest power of two below the length.
    let j = (usize::BITS - 1 - (d.len() - 1).leading_zeros()) as usize;
    let h = 1 << j;
    let low = value(&d[..h], powers, s);
    let high = value(&d[h..], powers, s);
    if high.is_empty() {
        return low;
    }
    let mut v = Vec::new();
    nat::mul_into(&mut v, &high, &powers[j], s);
    if low.len() > v.len() {
        v.resize(low.len(), 0);
    }
    if nat::add_assign(&mut v, &low) {
        v.push(1);
    }
    trim(&mut v);
    v
}

/// The fraction `y / d` with `|y| ≤ nb`, `0 < d ≤ db` and `y ≡ d·t (mod m)`, `0 ≤ t < m`, by the
/// extended Euclidean algorithm on `m` and `t`, stopped at the first remainder within `nb`
/// (Wang's rational reconstruction); `None` when the bounds admit none. Far from the stop, the
/// steps go by Lehmer's method (Knuth's Algorithm L): as long as the quotients of the top 63 bits
/// are those of the numbers, which two of them tell, they are found in registers and applied to
/// the numbers as one matrix of cofactors, about 30 steps at a time.
fn reconstruct(t: &Number, m: &Number, nb: &Number, db: &Number) -> Option<(Number, Number)> {
    let (mut r0, mut r1) = (m.clone(), t.clone());
    let (mut s0, mut s1) = (Number::from(0u8), Number::from(1u8));
    let stop = nb.bits() + 130;
    while r1.bits() > stop {
        let shift = (r0.bits() - 63) as usize;
        let top = |x: &Number| u64::try_from(&(x >> shift)).unwrap_or(0) as i128;
        let (mut x, mut y) = (top(&r0), top(&r1));
        let (mut a, mut b, mut c, mut d) = (1i128, 0i128, 0i128, 1i128);
        while y + c != 0 && y + d != 0 {
            let q = (x + a) / (y + c);
            if q != (x + b) / (y + d) {
                break;
            }
            (a, c) = (c, a - q * c);
            (b, d) = (d, b - q * d);
            (x, y) = (y, x - q * y);
        }
        if b == 0 {
            // No quotient was certain: one step on the numbers.
            let (q, r) = r0.div_rem(&r1);
            r0 = core::mem::replace(&mut r1, r);
            let s = &s0 - &(&q * &s1);
            s0 = core::mem::replace(&mut s1, s);
            continue;
        }
        // Each cofactor is below `2^63` in magnitude.
        let (a, b, c, d) = (a as i64, b as i64, c as i64, d as i64);
        let combine = |u: &Number, v: &Number, p: i64, q: i64| &(u * p) + &(v * q);
        (r0, r1) = (combine(&r0, &r1, a, b), combine(&r0, &r1, c, d));
        (s0, s1) = (combine(&s0, &s1, a, b), combine(&s0, &s1, c, d));
    }
    while &r1 > nb {
        let (q, r) = r0.div_rem(&r1);
        r0 = core::mem::replace(&mut r1, r);
        let s = &s0 - &(&q * &s1);
        s0 = core::mem::replace(&mut s1, s);
    }
    let (y, d) = if s1.is_negative() {
        (-r1, -s1)
    } else {
        (r1, s1)
    };
    if d.is_zero() || d.cmp(db) == Ordering::Greater {
        return None;
    }
    Some((y, d))
}

//! Exact algorithms on matrices of rationals, and the rational solutions of systems of integers:
//! determinants, ranks, solutions, inverses and null spaces, by fraction-free elimination.
//!
//! A matrix of rationals is first made a matrix of integers: each row is multiplied by the least
//! common multiple of its denominators, which leaves its solutions and its null space as they
//! were and multiplies its determinant by a known factor. Bareiss's elimination brings that
//! matrix, with the right-hand sides beside it, to an echelon form whose elements are minors, so
//! that every division in it is exact; a fraction-free back substitution then gives `D·x` for
//! the determinant `D`, integers by Cramer's rule; and each element of the result is reduced
//! once, by one greatest common divisor. No fraction is formed inside the loops.
//!
//! The integers of the elimination are held in registers while their magnitudes are below
//! `2^63`, where a step `(a·p - b·c) / d` is two products in 128 bits, a subtraction and a
//! product by the inverse of `d` modulo `2^64`; beyond, as limbs, each element in a vector of its
//! own that the steps write over, the products in two buffers of working memory, and the exact
//! quotient by Hensel's method, so that a step allocates nothing once the vectors have grown.
//!
//! Large systems with few right-hand sides, and large determinants, go by Dixon's p-adic lifting
//! instead ([`dixon`]), whose work is `O(n³)` operations on words. The product of matrices of
//! rationals clears the rows of the left one and the columns of the right one the same way, and
//! multiplies integers.

use super::Matrix;

mod dixon;
use crate::math::numbers::{ByteOrder, Magnitude, Number, nat};
use crate::math::rational::{Rational, in_order};
use core::cmp::Ordering;
use core::mem;

/// An integer beyond the registers: a sign and a magnitude of limbs, the least significant
/// first, without zero limbs at the top, not zero.
#[derive(Clone)]
struct Big {
    /// Whether it is below zero.
    neg: bool,
    /// The magnitude.
    d: Vec<u64>,
}

/// An integer of the eliminations: in a register while its magnitude is below `2^63`, so that a
/// product of two fits in an `i128` with room for a difference, and as limbs beyond.
#[derive(Clone)]
enum Int {
    /// An integer of magnitude below `2^63`: never `i64::MIN`.
    S(i64),
    /// An integer of magnitude `2^63` or more.
    B(Big),
}

use Int::{B, S};

/// Drops the zero limbs at the top of `d`.
#[inline]
fn trim(d: &mut Vec<u64>) {
    while d.last() == Some(&0) {
        d.pop();
    }
}

impl Int {
    /// The integer of the sign `neg` and the magnitude `d`, trimmed: in a register when it fits.
    #[inline]
    fn of_limbs(neg: bool, mut d: Vec<u64>) -> Int {
        trim(&mut d);
        match d.len() {
            0 => S(0),
            1 if d[0] < 1 << 63 => S(if neg { -(d[0] as i64) } else { d[0] as i64 }),
            _ => B(Big { neg, d }),
        }
    }

    /// The number `n`.
    fn of(n: &Number) -> Int {
        let m = Magnitude::of(n);
        let b = m.bytes;
        if b.len() <= 8 {
            let v = m.small() as u64;
            if v < 1 << 63 {
                let v = v as i64;
                return S(if m.negative { -v } else { v });
            }
        }
        let mut d = vec![0u64; b.len().div_ceil(8)];
        for (k, byte) in b.iter().enumerate() {
            // The `k`-th byte from the least significant end.
            let i = if m.big_endian { b.len() - 1 - k } else { k };
            d[i / 8] |= (*byte as u64) << (8 * (i % 8));
        }
        Int::of_limbs(m.negative, d)
    }

    /// The integer `x`.
    #[inline]
    fn wide(x: i128) -> Int {
        if x.unsigned_abs() < 1 << 63 {
            S(x as i64)
        } else {
            let m = x.unsigned_abs();
            Int::of_limbs(x < 0, vec![m as u64, (m >> 64) as u64])
        }
    }

    /// Whether it is zero.
    #[inline]
    fn is_zero(&self) -> bool {
        matches!(self, S(0))
    }

    /// Whether it is below zero.
    #[inline]
    fn is_negative(&self) -> bool {
        match self {
            S(x) => *x < 0,
            B(b) => b.neg,
        }
    }

    /// Whether it is 1.
    #[inline]
    fn is_one(&self) -> bool {
        matches!(self, S(1))
    }

    /// The number, in the byte order `order`.
    fn number(&self, order: ByteOrder) -> Number {
        match self {
            S(x) => Number::new_from_i64(*x, order),
            B(b) => {
                let mut v: Vec<u8> = b.d.iter().flat_map(|l| l.to_le_bytes()).collect();
                while v.last() == Some(&0) {
                    v.pop();
                }
                if order == ByteOrder::BigEndian {
                    v.reverse();
                }
                Number {
                    value: v,
                    negative: b.neg,
                    byte_order: order,
                }
            }
        }
    }

    /// `-x`.
    #[inline]
    fn neg(self) -> Int {
        match self {
            S(x) => S(-x),
            B(b) => B(Big {
                neg: !b.neg,
                d: b.d,
            }),
        }
    }

    /// The sign and the magnitude, the magnitude of a register in `buf`.
    #[inline(always)]
    fn mag<'a>(&'a self, buf: &'a mut [u64; 1]) -> (bool, &'a [u64]) {
        match self {
            S(x) => {
                buf[0] = x.unsigned_abs();
                (*x < 0, &buf[..])
            }
            B(b) => (b.neg, &b.d),
        }
    }

    /// The vector of the magnitude, to write a result in, or an empty one.
    #[inline]
    fn take_vec(&mut self) -> Vec<u64> {
        match mem::replace(self, S(0)) {
            B(b) => b.d,
            S(_) => Vec::new(),
        }
    }
}

/// Working memory of the steps beyond the registers: two products and the scratch of the longer
/// multiplications.
#[derive(Default)]
struct Work {
    u: Vec<u64>,
    v: Vec<u64>,
    s: Vec<u64>,
}

/// `buf = |a|·|b|`, trimmed, for `a` and `b` not zero: the sign of the product.
#[inline]
fn product_into(buf: &mut Vec<u64>, a: &Int, b: &Int, s: &mut Vec<u64>) -> bool {
    buf.clear();
    if let (S(x), S(y)) = (a, b) {
        let p = x.unsigned_abs() as u128 * y.unsigned_abs() as u128;
        buf.push(p as u64);
        if p >> 64 != 0 {
            buf.push((p >> 64) as u64);
        }
        return (*x < 0) != (*y < 0);
    }
    let (mut ta, mut tb) = ([0], [0]);
    let (na, ma) = a.mag(&mut ta);
    let (nb, mb) = b.mag(&mut tb);
    nat::mul_into(buf, ma, mb, s);
    trim(buf);
    na != nb
}

/// Compares two trimmed magnitudes.
#[inline]
fn cmp_mag(a: &[u64], b: &[u64]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// `u ← (±u) - (±v)` for the trimmed magnitudes `u` and `v` with the signs `nu` and `nv`: the
/// sign of the result, whose magnitude is left trimmed in `u` (empty for zero).
#[inline]
fn combine(u: &mut Vec<u64>, nu: bool, v: &[u64], nv: bool) -> bool {
    if nu != nv {
        if u.len() < v.len() {
            u.resize(v.len(), 0);
        }
        if nat::add_assign(u, v) {
            u.push(1);
        }
        return nu;
    }
    match cmp_mag(u, v) {
        Ordering::Less => {
            u.resize(v.len(), 0);
            nat::sub_from(u, v);
            trim(u);
            !nu
        }
        _ => {
            nat::sub_assign(u, v);
            trim(u);
            nu
        }
    }
}

/// `a · b`.
#[inline]
fn mul(a: &Int, b: &Int) -> Int {
    match (a, b) {
        (S(x), S(y)) => Int::wide(*x as i128 * *y as i128),
        _ if a.is_zero() || b.is_zero() => S(0),
        _ => {
            let mut d = Vec::new();
            let neg = product_into(&mut d, a, b, &mut Vec::new());
            Int::of_limbs(neg, d)
        }
    }
}

/// The inverse of the odd `a` modulo `2^64`, by Newton's iteration from a guess right to 5 bits.
#[inline]
fn inverse(a: u64) -> u64 {
    let mut x = a.wrapping_mul(3) ^ 2;
    for _ in 0..4 {
        x = x.wrapping_mul(2u64.wrapping_sub(a.wrapping_mul(x)));
    }
    x
}

/// The greatest common divisor of two words, by Stein's binary method.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 {
        return b;
    }
    if b == 0 {
        return a;
    }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b {
            mem::swap(&mut a, &mut b);
        }
        b -= a;
        if b == 0 {
            return a << shift;
        }
    }
}

/// A divisor prepared to divide exactly: for one in a register, its power of two, the inverse of
/// its odd part modulo `2^64` and the largest dividend in 128 bits whose quotient fits in a
/// register; for every divisor, its sign, the limbs of its odd part, their inverse modulo `2^64`
/// and its power of two, for the dividends beyond.
struct Exact {
    /// `(shift, inverse, limit)` when the divisor is in a register.
    small: Option<(u32, u64, u128)>,
    /// The divisor in a register, or 0.
    word: i64,
    /// Whether the divisor is below zero.
    neg: bool,
    /// The magnitude of the divisor shifted right by its zero bits at the low end, odd, when it
    /// is beyond a register; empty for one in a register, whose odd part is `odd1`.
    odd: Vec<u64>,
    /// The odd part of the magnitude of a divisor in a register.
    odd1: u64,
    /// The inverse of the low limb of `odd` modulo `2^64`.
    inv: u64,
    /// The zero bits at the low end of the divisor.
    zeros: usize,
    /// The inverse of `odd` modulo `2^(64·len)`, for the quotients of at least [`INVERSE_FROM`]
    /// limbs, when one has been asked for.
    hinv: Vec<u64>,
}

/// From this many limbs in the quotient, an exact division is a product by the inverse of the
/// divisor modulo a power of `2^64`, computed once for all the divisions by it, rather than
/// Hensel's quadratic method.
const INVERSE_FROM: usize = 150;

impl Exact {
    /// The divisor `d`, which is not zero.
    fn new(d: &Int) -> Exact {
        if let S(v) = *d {
            let bits = v.trailing_zeros();
            let odd1 = v.unsigned_abs() >> bits;
            return Exact {
                small: Some((
                    bits,
                    inverse((v >> bits) as u64),
                    i64::MAX as u128 * v.unsigned_abs() as u128,
                )),
                word: v,
                neg: v < 0,
                odd: Vec::new(),
                odd1,
                inv: inverse(odd1),
                zeros: bits as usize,
                hinv: Vec::new(),
            };
        }
        let mut buf = [0];
        let (neg, m) = d.mag(&mut buf);
        let limbs = m.iter().take_while(|&&x| x == 0).count();
        let bits = m[limbs].trailing_zeros();
        let mut odd = m[limbs..].to_vec();
        if bits != 0 {
            rshift(&mut odd, bits);
            trim(&mut odd);
        }
        let inv = nat::binvert(odd[0]);
        Exact {
            small: None,
            word: 0,
            neg,
            odd,
            odd1: 0,
            inv,
            zeros: 64 * limbs + bits as usize,
            hinv: Vec::new(),
        }
    }

    /// 1.
    fn one() -> Exact {
        Exact::new(&S(1))
    }

    /// `x / d` for a multiple `x` of the divisor.
    #[inline(always)]
    fn wide(&mut self, x: i128) -> Int {
        if let Some((shift, inv, limit)) = self.small {
            if x.unsigned_abs() <= limit {
                // `q·(d >> shift) = x >> shift` holds modulo `2^64` too, and `q` fits.
                return S(((x >> shift) as u64).wrapping_mul(inv) as i64);
            }
            return Int::wide(x / self.word as i128);
        }
        self.cold_wide(x)
    }

    /// `x / d` for a multiple `x` of a divisor beyond the registers.
    #[cold]
    #[inline(never)]
    fn cold_wide(&mut self, x: i128) -> Int {
        let m = x.unsigned_abs();
        let mut t = vec![m as u64, (m >> 64) as u64];
        trim(&mut t);
        if t.is_empty() {
            return S(0);
        }
        self.limbs(&mut t, x < 0, Vec::new(), &mut Work::default())
    }

    /// `t / d` for the multiple of the divisor of the sign `neg` and the trimmed magnitude `t`,
    /// not zero, which this changes; the quotient's limbs are written in `out`, with the vectors
    /// `v` and `s` of `w` as working memory.
    fn limbs(&mut self, t: &mut Vec<u64>, neg: bool, mut out: Vec<u64>, w: &mut Work) -> Int {
        let neg = neg != self.neg;
        if self.word == 1 {
            mem::swap(&mut out, t);
            return Int::of_limbs(neg, out);
        }
        if self.zeros != 0 {
            let limbs = self.zeros / 64;
            t.drain(..limbs);
            let bits = (self.zeros % 64) as u32;
            if bits != 0 {
                rshift(t, bits);
            }
            trim(t);
        }
        let odd = if self.odd.is_empty() {
            core::slice::from_ref(&self.odd1)
        } else {
            &self.odd
        };
        let qn = t.len() + 1 - odd.len();
        if qn >= INVERSE_FROM {
            if self.hinv.len() < qn {
                // With room for the longer quotients of the divisions to come by it.
                let l = qn + qn / 4;
                nat::inverse_into(&mut self.hinv, odd, self.inv, l, &mut w.v, &mut w.s);
            }
            nat::mul_into(&mut out, &t[..qn], &self.hinv[..qn], &mut w.s);
            out.truncate(qn);
        } else {
            nat::bdiv_into(&mut out, qn, t, odd, self.inv);
        }
        Int::of_limbs(neg, out)
    }
}

/// Shifts the limbs of `x` right by `s` bits, `0 < s < 64`, in place.
#[inline]
fn rshift(x: &mut [u64], s: u32) {
    let n = x.len();
    for i in 0..n - 1 {
        x[i] = (x[i] >> s) | (x[i + 1] << (64 - s));
    }
    x[n - 1] >>= s;
}

/// The step of Bareiss's elimination for one row: `x ← (x·p - b·y) / d` for each element `x` of
/// the row with `y` the element of the pivot's row in its column, `p` the pivot, `b` the row's
/// element in the pivot's column and `d` the previous pivot.
#[inline]
fn update_row(row: &mut [Int], pivot_row: &[Int], p: &Int, b: &Int, d: &mut Exact, w: &mut Work) {
    if let (S(pw), S(bw), Some(_)) = (p, b, d.small) {
        let (pw, bw) = (*pw as i128, *bw as i128);
        for (x, y) in row.iter_mut().zip(pivot_row) {
            if let (S(a), S(c)) = (&*x, y) {
                // Each magnitude is below `2^63`, so each product below `2^126`, and the
                // difference fits.
                *x = d.wide(*a as i128 * pw - bw * *c as i128);
            } else {
                step_big(x, p, b, y, d, w);
            }
        }
    } else {
        for (x, y) in row.iter_mut().zip(pivot_row) {
            step_big(x, p, b, y, d, w);
        }
    }
}

/// `x ← (x·p - b·y) / d` beyond the registers.
#[inline(never)]
fn step_big(x: &mut Int, p: &Int, b: &Int, y: &Int, d: &mut Exact, w: &mut Work) {
    if let (S(a), S(pp), S(bb), S(c)) = (&*x, p, b, y) {
        *x = d.wide(*a as i128 * *pp as i128 - *bb as i128 * *c as i128);
        return;
    }
    let mut nu = false;
    w.u.clear();
    if !x.is_zero() {
        nu = product_into(&mut w.u, x, p, &mut w.s);
    }
    if !b.is_zero() && !y.is_zero() {
        let nv = product_into(&mut w.v, b, y, &mut w.s);
        if w.u.is_empty() {
            mem::swap(&mut w.u, &mut w.v);
            nu = !nv;
        } else {
            nu = combine(&mut w.u, nu, &w.v, nv);
        }
    }
    if w.u.is_empty() {
        *x = S(0);
        return;
    }
    let out = x.take_vec();
    let mut u = mem::take(&mut w.u);
    *x = d.limbs(&mut u, nu, out, w);
    w.u = u;
}

/// What an elimination found: the rank, the number of row swaps and the pivots' columns.
struct Echelon {
    /// The number of pivots.
    rank: usize,
    /// The number of row swaps.
    swaps: usize,
    /// The column of each pivot, the pivot of row `k` being in column `pivots[k]`; empty for an
    /// elimination that stops at a column without a pivot, whose pivots are on the diagonal.
    pivots: Vec<usize>,
}

/// Bareiss's fraction-free elimination of the `rows × cols` matrix `m`, held row after row, to
/// an echelon form in place, the pivots sought in the first `search` columns, each the first
/// element that is not zero at or below its row; every element is then a minor of the matrix.
/// `None` at the first column without a pivot when `stop` is set.
fn eliminate(
    m: &mut [Int],
    rows: usize,
    cols: usize,
    search: usize,
    stop: bool,
) -> Option<Echelon> {
    let mut previous = Exact::one();
    let mut r = 0;
    let mut swaps = 0;
    // A square elimination that stops has its pivots on the diagonal.
    let mut pivots = Vec::with_capacity(if stop { 0 } else { rows.min(search) });
    let mut w = Work::default();
    for c in 0..search {
        if r == rows {
            break;
        }
        let Some(p) = (r..rows).find(|&i| !m[i * cols + c].is_zero()) else {
            if stop {
                return None;
            }
            continue;
        };
        if p != r {
            let (a, b) = m.split_at_mut(p * cols);
            a[r * cols..(r + 1) * cols].swap_with_slice(&mut b[..cols]);
            swaps += 1;
        }
        let (top, rest) = m.split_at_mut((r + 1) * cols);
        let pivot_row = &top[r * cols + c + 1..];
        let pivot = &top[r * cols + c];
        for row in rest.chunks_exact_mut(cols) {
            let b = mem::replace(&mut row[c], S(0));
            update_row(
                &mut row[c + 1..],
                pivot_row,
                pivot,
                &b,
                &mut previous,
                &mut w,
            );
        }
        if !stop {
            pivots.push(c);
        }
        r += 1;
        if r < rows {
            previous = Exact::new(pivot);
        }
    }
    Some(Echelon {
        rank: r,
        swaps,
        pivots,
    })
}

/// A sum of products in the back substitution: in 128 bits while it fits, as limbs beyond.
enum Acc {
    W(i128),
    N(bool, Vec<u64>),
}

impl Acc {
    /// `a · b`.
    #[inline]
    fn product(a: &Int, b: &Int) -> Acc {
        match (a, b) {
            (S(x), S(y)) => Acc::W(*x as i128 * *y as i128),
            _ if a.is_zero() || b.is_zero() => Acc::W(0),
            _ => {
                let mut d = Vec::new();
                let neg = product_into(&mut d, a, b, &mut Vec::new());
                Acc::N(neg, d)
            }
        }
    }

    /// `self -= a · b`.
    #[inline(always)]
    fn sub(&mut self, a: &Int, b: &Int, w: &mut Work) {
        if let (Acc::W(acc), S(x), S(y)) = (&mut *self, a, b)
            && let Some(v) = acc.checked_sub(*x as i128 * *y as i128)
        {
            *acc = v;
            return;
        }
        self.big(a, b, true, w);
    }

    /// `self += a · b`.
    #[inline(always)]
    fn add(&mut self, a: &Int, b: &Int, w: &mut Work) {
        if let (Acc::W(acc), S(x), S(y)) = (&mut *self, a, b)
            && let Some(v) = acc.checked_add(*x as i128 * *y as i128)
        {
            *acc = v;
            return;
        }
        self.big(a, b, false, w);
    }

    /// `self -= a · b`, or `self += a · b` when `subtract` is not set, beyond 128 bits.
    #[inline(never)]
    fn big(&mut self, a: &Int, b: &Int, subtract: bool, w: &mut Work) {
        if a.is_zero() || b.is_zero() {
            return;
        }
        if let Acc::W(x) = *self {
            let m = x.unsigned_abs();
            let mut d = vec![m as u64, (m >> 64) as u64];
            trim(&mut d);
            *self = Acc::N(x < 0, d);
        }
        let Acc::N(neg, d) = self else { unreachable!() };
        let nv = product_into(&mut w.v, a, b, &mut w.s) != !subtract;
        if d.is_empty() {
            d.extend_from_slice(&w.v);
            *neg = !nv;
        } else {
            *neg = combine(d, *neg, &w.v, nv);
        }
    }

    /// The sum as an integer.
    #[inline]
    fn into_int(self) -> Int {
        match self {
            Acc::W(x) => Int::wide(x),
            Acc::N(neg, d) => Int::of_limbs(neg, d),
        }
    }

    /// The integer `x`.
    #[inline]
    fn of(x: &Int) -> Acc {
        match x {
            S(v) => Acc::W(*v as i128),
            B(b) => Acc::N(b.neg, b.d.clone()),
        }
    }

    /// `-self`.
    #[inline]
    fn neg(self) -> Acc {
        match self {
            // `-(a·b)` of two magnitudes below `2^63` fits.
            Acc::W(w) => Acc::W(-w),
            Acc::N(neg, d) => Acc::N(!neg, d),
        }
    }

    /// `self / d`, which divides exactly.
    #[inline]
    fn div(self, d: &mut Exact, w: &mut Work) -> Int {
        match self {
            Acc::W(x) => d.wide(x),
            Acc::N(_, m) if m.is_empty() => S(0),
            Acc::N(neg, mut m) => d.limbs(&mut m, neg, Vec::new(), w),
        }
    }
}

/// The fraction-free back substitution: for the echelon form `u` of `[A | B]` of `n` rows and
/// `cols` columns, row after row, with `A` square and its pivots on the diagonal, `y = D·x` for
/// each right-hand side `k` in `n..cols`, `D` the last pivot: `y_i = (D·b_i - Σ u_ij·y_j) / u_ii`
/// from the bottom, each division exact as `D·x` is an integer by Cramer's rule. Gives each
/// column to `f(k - n, y)`.
fn back_substitute(u: &[Int], n: usize, cols: usize, mut f: impl FnMut(usize, &[Int])) {
    let mut divisors: Vec<Exact> = (0..n).map(|i| Exact::new(&u[i * cols + i])).collect();
    let d = &u[(n - 1) * cols + n - 1];
    let mut y = vec![S(0); n];
    let mut w = Work::default();
    for k in n..cols {
        y[n - 1] = u[(n - 1) * cols + k].clone();
        for i in (0..n - 1).rev() {
            let row = &u[i * cols..(i + 1) * cols];
            let mut acc = Acc::product(d, &row[k]);
            for (a, b) in row[i + 1..n].iter().zip(&y[i + 1..]) {
                acc.sub(a, b, &mut w);
            }
            y[i] = acc.div(&mut divisors[i], &mut w);
        }
        f(k - n, &y);
    }
}

/// `n` with the sign `negative` in the byte order `order`, in the shortest form.
#[inline]
fn word_number(order: ByteOrder, negative: bool, m: u64) -> Number {
    let mut n = Number::new_from_u64(m, order);
    n.negative = negative && m != 0;
    n
}

/// The rational `y / d`, `d` not zero and `dn` its number, in lowest terms, in the byte order
/// `order`: by one greatest common divisor, in registers when both fit.
fn reduce(y: &Int, d: &Int, dn: Option<&Number>, order: ByteOrder) -> Rational {
    if let (S(a), S(b)) = (y, d) {
        let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
        if ua == 0 {
            return Rational::from_reduced(
                word_number(order, false, 0),
                word_number(order, false, 1),
            );
        }
        let g = gcd(ua, ub);
        let (ua, ub) = if g == 1 { (ua, ub) } else { (ua / g, ub / g) };
        return Rational::from_reduced(
            word_number(order, (*a < 0) != (*b < 0), ua),
            word_number(order, false, ub),
        );
    }
    if y.is_zero() {
        return reduce(&S(0), &S(1), None, order);
    }
    if d.is_one() {
        return Rational::from_reduced(y.number(order), word_number(order, false, 1));
    }
    let den = match dn {
        Some(n) => n.clone(),
        None => d.number(order),
    };
    Rational::new(y.number(order), den)
}

/// The least common multiple of the positive `l` and `d`.
fn lcm(l: Int, d: &Int) -> Int {
    if let (S(x), S(y)) = (&l, d) {
        let (x, y) = (*x as u64, *y as u64);
        if let Some(m) = (x / gcd(x, y)).checked_mul(y)
            && m < 1 << 63
        {
            return S(m as i64);
        }
    }
    Int::of(
        &l.number(ByteOrder::LittleEndian)
            .lcm(&d.number(ByteOrder::LittleEndian)),
    )
}

/// The rows of the matrices `parts`, side by side, each times the least common multiple of its
/// denominators: the integers, row after row, and the multiples.
fn clear(parts: &[&Matrix<Rational>]) -> (Vec<Int>, Vec<Int>) {
    let rows = parts[0].rows();
    let cols: usize = parts.iter().map(|p| p.cols()).sum();
    let mut m = Vec::with_capacity(rows * cols);
    let multiples = (0..rows)
        .map(|i| clear_into(&mut m, || parts.iter().flat_map(move |p| p.row(i))))
        .collect();
    (m, multiples)
}

/// Pushes the rationals of `xs()` onto `m`, each times the least common multiple of their
/// denominators, which it returns.
fn clear_into<'a, I: Iterator<Item = &'a Rational>>(m: &mut Vec<Int>, xs: impl Fn() -> I) -> Int {
    let start = m.len();
    let mut l = S(1);
    for x in xs() {
        if !x.is_integer() {
            l = lcm(l, &Int::of(x.denom()));
        }
        m.push(Int::of(x.numer()));
    }
    if !l.is_one() {
        // Each numerator times `l` over its denominator.
        for (n, x) in m[start..].iter_mut().zip(xs()) {
            if n.is_zero() {
                continue;
            }
            let f = if x.is_integer() {
                l.clone()
            } else {
                match (&l, Int::of(x.denom())) {
                    (S(l), S(d)) => S(l / d),
                    (l, d) => Int::of(
                        &l.number(ByteOrder::LittleEndian)
                            .div_exact(&d.number(ByteOrder::LittleEndian)),
                    ),
                }
            };
            *n = mul(n, &f);
        }
    }
    l
}

/// `c = a·b` for the column-major `m × k` matrix of rationals `a` and `k × n` one `b`, the
/// elements of `c` little-endian as the zeros it holds: the rows of `a` and the columns of `b`
/// multiplied by the least common multiples `l_i` and `r_j` of their denominators, the product of
/// those integers, then one reduction of each element by `l_i·r_j`: `m·k·n` products of integers
/// and `m·n` greatest common divisors, against `m·k·n` products and sums of rationals, each with
/// its own. The sums of products are in 64-bit registers when the elements' lengths and `k` make
/// it safe, in 128 bits when that does, and checked beyond. Products of two terms, and of four
/// with a denominator beyond 32 bits, are the sums of products of rationals, which the rationals'
/// register paths make faster there.
pub(super) fn product(
    c: &mut [Rational],
    a: &[Rational],
    b: &[Rational],
    m: usize,
    k: usize,
    n: usize,
) {
    if k == 0 {
        return;
    }
    let order = ByteOrder::LittleEndian;
    let long = |x: &Rational| Magnitude::of(x.denom()).bytes.len() > 4;
    if k <= 2 || (k <= 4 && (a.iter().any(long) || b.iter().any(long))) {
        for j in 0..n {
            for i in 0..m {
                let mut s = &a[i] * &b[j * k];
                for p in 1..k {
                    s += &a[i + p * m] * &b[p + j * k];
                }
                // Little-endian, as the zero that the sum would start from.
                let s = if s.byte_order() == order {
                    s
                } else {
                    let (x, y) = s.into_parts();
                    Rational::from_reduced(in_order(x, order), in_order(y, order))
                };
                c[i + j * m] = s;
            }
        }
        return;
    }
    let mut ai = Vec::with_capacity(m * k);
    let l: Vec<Int> = (0..m)
        .map(|i| clear_into(&mut ai, || (0..k).map(move |p| &a[i + p * m])))
        .collect();
    let mut bi = Vec::with_capacity(k * n);
    let r: Vec<Int> = (0..n)
        .map(|j| clear_into(&mut bi, || b[j * k..(j + 1) * k].iter()))
        .collect();
    let mut put = |i: usize, j: usize, y: Int| {
        let d = if r[j].is_one() {
            l[i].clone()
        } else {
            mul(&l[i], &r[j])
        };
        c[i + j * m] = reduce(&y, &d, None, order);
    };
    // The bits of the largest magnitudes, when every element is in a register.
    let bits = |v: &[Int]| {
        v.iter().try_fold(0, |b, x| match x {
            S(x) => Some(b.max(64 - x.unsigned_abs().leading_zeros())),
            B(_) => None,
        })
    };
    let sum = usize::BITS - (k - 1).leading_zeros();
    match (bits(&ai), bits(&bi)) {
        (Some(x), Some(y)) if x + y + sum <= 62 => {
            let (av, bv) = (words(&ai), words(&bi));
            for (j, bj) in bv.chunks_exact(k).enumerate() {
                for (i, ar) in av.chunks_exact(k).enumerate() {
                    let s: i64 = ar.iter().zip(bj).map(|(x, y)| x * y).sum();
                    put(i, j, S(s));
                }
            }
        }
        (Some(x), Some(y)) if x + y + sum <= 126 => {
            let (av, bv) = (words(&ai), words(&bi));
            for (j, bj) in bv.chunks_exact(k).enumerate() {
                for (i, ar) in av.chunks_exact(k).enumerate() {
                    let s: i128 = ar
                        .iter()
                        .zip(bj)
                        .map(|(&x, &y)| x as i128 * y as i128)
                        .sum();
                    put(i, j, Int::wide(s));
                }
            }
        }
        _ => {
            let mut w = Work::default();
            for (j, bj) in bi.chunks_exact(k).enumerate() {
                for (i, ar) in ai.chunks_exact(k).enumerate() {
                    let mut acc = Acc::W(0);
                    for (x, y) in ar.iter().zip(bj) {
                        acc.add(x, y, &mut w);
                    }
                    put(i, j, acc.into_int());
                }
            }
        }
    }
}

/// The values of integers all in registers.
fn words(v: &[Int]) -> Vec<i64> {
    v.iter()
        .map(|x| match x {
            S(x) => *x,
            B(_) => unreachable!("an integer beyond the registers"),
        })
        .collect()
}

/// The integers of the matrices `parts`, side by side, row after row.
fn integers(parts: &[&Matrix<Number>]) -> Vec<Int> {
    let rows = parts[0].rows();
    let cols: usize = parts.iter().map(|p| p.cols()).sum();
    let mut m = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for p in parts {
            m.extend(p.row(i).map(Int::of));
        }
    }
    m
}

/// The identity matrix of order `n` beside the `n × n` matrix `a`, held row after row: `[A | I]`.
fn beside_identity(a: Vec<Int>, n: usize) -> Vec<Int> {
    let mut m = Vec::with_capacity(2 * n * n);
    let mut a = a.into_iter();
    for i in 0..n {
        m.extend(a.by_ref().take(n));
        m.extend((0..n).map(|j| S((i == j) as i64)));
    }
    m
}

/// The solution of the square system `[A | B]`, held row after row in `m` with `n` rows and
/// `n + k` columns, as a rational `n × k` matrix in the byte order `order`, each column `j`
/// multiplied by `scale[j]` when it is given; `None` when `A` is singular.
fn solve_rows(
    mut m: Vec<Int>,
    n: usize,
    k: usize,
    scale: Option<&[Int]>,
    order: ByteOrder,
) -> Option<Matrix<Rational>> {
    let cols = n + k;
    if k <= n / 8
        && dixon_pays(&m, n, DIXON_SOLVE)
        && let Some((y, _)) = dixon::solve(&m, n, k)
    {
        let mut x = Vec::with_capacity(n * k);
        for (j, y) in y.chunks_exact(n).enumerate() {
            match scale.map(|s| &s[j]) {
                Some(s) if !s.is_one() => {
                    x.extend(y.iter().map(|(y, d)| reduce(&mul(y, s), d, None, order)))
                }
                _ => x.extend(y.iter().map(|(y, d)| reduce(y, d, None, order))),
            }
        }
        return Some(Matrix::from_column_major(n, k, x));
    }
    eliminate(&mut m, n, cols, n, true)?;
    if n == 0 {
        return Some(Matrix::from_fn(0, k, |_, _| unreachable!()));
    }
    let d = m[(n - 1) * cols + n - 1].clone();
    let dn = d.number(order);
    let mut x = Vec::with_capacity(n * k);
    back_substitute(&m, n, cols, |j, y| match scale.map(|s| &s[j]) {
        Some(s) if !s.is_one() => {
            x.extend(y.iter().map(|y| reduce(&mul(y, s), &d, Some(&dn), order)))
        }
        _ => x.extend(y.iter().map(|y| reduce(y, &d, Some(&dn), order))),
    });
    Some(Matrix::from_column_major(n, k, x))
}

/// The determinant of the `n × n` matrix of integers held row after row in `m`, which this
/// changes: from the orders of [`DIXON_DET`] by Dixon's lifting and the residues of the quotient
/// by the divisor it finds ([`dixon::det`]), and otherwise, or when that fails, by Bareiss's
/// elimination.
fn det_rows(m: &mut [Int], n: usize) -> Int {
    if dixon_pays(m, n, DIXON_DET)
        && let Some(d) = dixon::det(m, n, n)
    {
        return d;
    }
    let Some(e) = eliminate(m, n, n, n, true) else {
        return S(0);
    };
    let d = m[n * n - 1].clone();
    if e.swaps % 2 == 1 { d.neg() } else { d }
}

/// The orders from which a system with at most an eighth as many right-hand sides is solved by
/// Dixon's p-adic lifting ([`dixon`]) rather than Bareiss's elimination: when every integer is
/// in a register, and when one is not. Measured on an Apple M4 Max, solving `A·x = b`: for
/// integers in `-99..=99`, 24 rows took 64.1 µs so against 67.5 by the elimination, 20 rows 46.8
/// against 36.1; for fractions of parts up to 99, whose rows' common denominators pass the
/// registers, 16 rows 102 µs against 118, 12 rows 55.5 against 43.3; for parts of 64 bits, 16
/// rows 6.18 ms against 7.59.
const DIXON_SOLVE: (usize, usize) = (24, 16);

/// The orders from which a determinant is Dixon's divisor times residues ([`dixon::det`]), as
/// [`DIXON_SOLVE`]: for integers in `-99..=99`, 28 rows took 95.2 µs against 96.0 by the
/// elimination, 24 rows 71.4 against 50.6; for fractions, 24 rows 358 µs against 426, 20 rows
/// 236 against 196; for parts of 64 bits, 20 rows 10.6 ms against 16.9, 16 rows 4.63 against
/// 4.41.
const DIXON_DET: (usize, usize) = (28, 22);

/// Whether Dixon's lifting pays for the `n` rows of the matrix of integers `m` by the orders
/// `from`: with every integer in a register, and with one beyond.
fn dixon_pays(m: &[Int], n: usize, from: (usize, usize)) -> bool {
    n >= from.0 || (n >= from.1 && m.iter().any(|x| matches!(x, B(_))))
}

/// The byte order of the results: that of the top left element, little-endian for an empty
/// matrix.
fn order_of<T>(m: &Matrix<T>, f: impl Fn(&T) -> ByteOrder) -> ByteOrder {
    m.as_slice().first().map_or(ByteOrder::LittleEndian, f)
}

impl Matrix<Rational> {
    /// The determinant, exactly: the rows multiplied by the least common multiples `l_i` of
    /// their denominators, the determinant of that matrix of integers, and one division,
    /// `det(A) = det(L·A) / ∏ l_i`, with no fraction in between. The determinant of the integers
    /// is Bareiss's fraction-free elimination, about `n³/3` steps of two products and an exact
    /// division of integers no longer than the minors; from 28 rows (22 when an integer passes
    /// 63 bits) it is the least common multiple of the denominators of the solution of a system
    /// by Dixon's p-adic lifting, a divisor of the determinant, times the quotient's residues
    /// modulo primes, `O(n³)` operations on words (FLINT's method). The determinant of the
    /// `0 × 0` matrix is 1; the result is in the byte order of the top left element.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::rational::Rational;
    ///
    /// let half = Rational::from(1) / 2u32;
    /// let a = Matrix::from_rows(&[[half.clone(), Rational::from(1)], [Rational::from(1), half]]);
    /// assert_eq!(a.det().to_string(), "-3/4");
    /// ```
    #[track_caller]
    pub fn det(&self) -> Rational {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let order = order_of(self, Rational::byte_order);
        if n == 0 {
            return reduce(&S(1), &S(1), None, order);
        }
        let (mut m, multiples) = clear(&[self]);
        let l = multiples.iter().fold(S(1), |l, x| mul(&l, x));
        reduce(&det_rows(&mut m, n), &l, None, order)
    }

    /// The rank, exactly: the number of pivots of Bareiss's elimination of the rows multiplied
    /// by the least common multiples of their denominators.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::rational::Rational;
    ///
    /// let a = Matrix::from_rows(&[[1, 2], [2, 4], [3, 6]]).map(|&x| Rational::from(x) / 3u32);
    /// assert_eq!(a.rank(), 1);
    /// ```
    pub fn rank(&self) -> usize {
        let (rows, cols) = self.shape();
        if rows == 0 || cols == 0 {
            return 0;
        }
        let (mut m, _) = clear(&[self]);
        eliminate(&mut m, rows, cols, cols, false).map_or(0, |e| e.rank)
    }

    /// The solution `X` of `A·X = B`, exactly, or `None` when the matrix is singular: the rows
    /// of `[A | B]` multiplied by the least common multiples of their denominators, which leaves
    /// the solution as it is, Bareiss's elimination of the integers to an echelon form with the
    /// last pivot `D = ±det`, a fraction-free back substitution to the integers `D·X` (Cramer's
    /// rule), and one reduction of each element by a greatest common divisor. From 24 rows (16
    /// when an integer passes 63 bits), with at most an eighth as many columns in `B`, Dixon's
    /// p-adic lifting instead: `A` factored once modulo a prime of 62 bits, the solution lifted
    /// to a power of it past Hadamard's bounds, and each element reconstructed as a fraction,
    /// `O(n³)` operations on words. The result is in the byte order of the top left element of
    /// `A`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, or `B` has not as many rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::rational::Rational;
    ///
    /// let a = Matrix::from_rows(&[[2, 1], [1, 3]]).map(|&x| Rational::from(x));
    /// let b = Matrix::from_rows(&[[1], [2]]).map(|&x| Rational::from(x) / 2u32);
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(&a * &x, b);
    /// assert_eq!(x[(0, 0)].to_string(), "1/10");
    /// ```
    #[track_caller]
    pub fn solve(&self, b: &Matrix<Rational>) -> Option<Matrix<Rational>> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        assert_eq!(b.rows(), n, "the right-hand side has not as many rows");
        let order = order_of(self, Rational::byte_order);
        if n == 0 {
            return Some(Matrix::zeros(0, b.cols()));
        }
        let (m, _) = clear(&[self, b]);
        solve_rows(m, n, b.cols(), None, order)
    }

    /// The inverse `A⁻¹`, exactly, or `None` when the matrix is singular: with `L` the diagonal
    /// matrix of the least common multiples of the rows' denominators, `A⁻¹ = (L·A)⁻¹·L`, the
    /// inverse of the integers `L·A` by the elimination and back substitution of
    /// [`solve`](Matrix::solve) on `[L·A | I]`, and one reduction of each element. The result is
    /// in the byte order of the top left element.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::rational::Rational;
    ///
    /// let a = Matrix::from_rows(&[[1, 2], [3, 4]]).map(|&x| Rational::from(x) / 5u32);
    /// let inv = a.inverse().unwrap();
    /// assert_eq!(&a * &inv, Matrix::identity(2));
    /// assert_eq!(inv[(0, 0)].to_string(), "-10");
    /// ```
    #[track_caller]
    pub fn inverse(&self) -> Option<Matrix<Rational>> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let order = order_of(self, Rational::byte_order);
        if n == 0 {
            return Some(Matrix::zeros(0, 0));
        }
        let (m, multiples) = clear(&[self]);
        let scale = (!multiples.iter().all(Int::is_one)).then_some(multiples.as_slice());
        solve_rows(beside_identity(m, n), n, n, scale, order)
    }

    /// A basis of the null space `{x : A·x = 0}`, as the columns of an `n × (n - rank)` matrix:
    /// one vector for each column `f` without a pivot in the fraction-free echelon form, with
    /// `x_f = 1` and the other such coordinates 0, which makes the basis the one of the reduced
    /// row echelon form, one for each subspace. Its pivot coordinates come from a fraction-free
    /// back substitution and one reduction each.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::rational::Rational;
    ///
    /// let a = Matrix::from_rows(&[[1, 2, 3], [2, 4, 7]]).map(|&x| Rational::from(x) / 2u32);
    /// let null = a.null_space();
    /// assert_eq!(null, Matrix::from_rows(&[[-2], [1], [0]]).map(|&x| Rational::from(x)));
    /// ```
    pub fn null_space(&self) -> Matrix<Rational> {
        let (rows, cols) = self.shape();
        let order = order_of(self, Rational::byte_order);
        if rows == 0 || cols == 0 {
            return Matrix::from_fn(cols, cols, |i, j| {
                reduce(&S((i == j) as i64), &S(1), None, order)
            });
        }
        let (mut m, _) = clear(&[self]);
        let e = eliminate(&mut m, rows, cols, cols, false).expect("an elimination that goes on");
        null_space_of(&m, cols, &e, order)
    }
}

/// The vectors of the null space from the echelon form `e` of `cols` columns, held row after row
/// in `m`: for each column `f` without a pivot, `f(f, y, d)` with `d` the last pivot and `y` the
/// integers `d·x` at the pivots' columns of the solution `x` with `x_f = 1` and the other columns
/// without a pivot 0, by a fraction-free back substitution.
fn null_vectors(m: &[Int], cols: usize, e: &Echelon, mut f: impl FnMut(usize, &[Int], &Int)) {
    let r = e.rank;
    let mut is_pivot = vec![false; cols];
    for &p in &e.pivots {
        is_pivot[p] = true;
    }
    let free = (0..cols).filter(|&c| !is_pivot[c]);
    if r == 0 {
        for c in free {
            f(c, &[], &S(1));
        }
        return;
    }
    let d = &m[(r - 1) * cols + e.pivots[r - 1]];
    let mut w = Work::default();
    let mut divisors: Vec<Exact> = (0..r)
        .map(|i| Exact::new(&m[i * cols + e.pivots[i]]))
        .collect();
    let mut y = vec![S(0); r];
    for c in free {
        for i in (0..r).rev() {
            let row = &m[i * cols..(i + 1) * cols];
            let mut acc = Acc::product(d, &row[c]).neg();
            for (j, yj) in (i + 1..r).zip(&y[i + 1..]) {
                acc.sub(&row[e.pivots[j]], yj, &mut w);
            }
            y[i] = acc.div(&mut divisors[i], &mut w);
        }
        f(c, &y, d);
    }
}

/// The null space from the echelon form `e` of `cols` columns, held row after row in `m`, as in
/// [`Matrix::<Rational>::null_space`].
fn null_space_of(m: &[Int], cols: usize, e: &Echelon, order: ByteOrder) -> Matrix<Rational> {
    let mut basis = Vec::with_capacity(cols * (cols - e.rank));
    let dn = e
        .rank
        .checked_sub(1)
        .map(|r| m[r * cols + e.pivots[r]].number(order));
    null_vectors(m, cols, e, |f, y, d| {
        let start = basis.len();
        basis.extend((0..cols).map(|c| reduce(&S((c == f) as i64), &S(1), None, order)));
        for (yi, &p) in y.iter().zip(&e.pivots) {
            basis[start + p] = reduce(yi, d, dn.as_ref(), order);
        }
    });
    Matrix::from_column_major(cols, cols - e.rank, basis)
}

/// The determinant of the square matrix of integers `a`, by the elimination of this module, in
/// the byte order of its top left element.
pub(super) fn det_of_integers(a: &Matrix<Number>) -> Number {
    let n = a.rows();
    let order = order_of(a, Number::byte_order);
    if n == 0 {
        return Number::new_from_u8(1, order);
    }
    det_rows(&mut integers(&[a]), n).number(order)
}

/// The rank of the matrix of integers `a`, by the elimination of this module.
pub(super) fn rank_of_integers(a: &Matrix<Number>) -> usize {
    let (rows, cols) = a.shape();
    if rows == 0 || cols == 0 {
        return 0;
    }
    let mut m = integers(&[a]);
    eliminate(&mut m, rows, cols, cols, false).map_or(0, |e| e.rank)
}

/// The adjugate and the determinant of the square matrix of integers `a`, or `None` when it is
/// singular: the integers `D·A⁻¹` of the back substitution on `[A | I]`, `D = ±det(A)` its last
/// pivot, which are the adjugate with the sign of the row swaps; in the byte order of the top left
/// element.
pub(super) fn inverse_of_integers(a: &Matrix<Number>) -> Option<(Matrix<Number>, Number)> {
    let n = a.rows();
    let order = order_of(a, Number::byte_order);
    if n == 0 {
        return Some((Matrix::zeros(0, 0), Number::new_from_u8(1, order)));
    }
    let mut m = beside_identity(integers(&[a]), n);
    let e = eliminate(&mut m, n, 2 * n, n, true)?;
    let odd = e.swaps % 2 == 1;
    let mut adj = Vec::with_capacity(n * n);
    back_substitute(&m, n, 2 * n, |_, y| {
        adj.extend(y.iter().map(|y| {
            let y = if odd { y.clone().neg() } else { y.clone() };
            y.number(order)
        }));
    });
    let d = m[n * 2 * n - n - 1].clone();
    let det = if odd { d.neg() } else { d };
    Some((Matrix::from_column_major(n, n, adj), det.number(order)))
}

/// A basis of the null space of the matrix of integers `a` as `Matrix::<Number>::null_space`
/// gives it: the vectors of the reduced echelon form, each times the smallest positive integer
/// that makes it one of integers; in the byte order of the top left element.
pub(super) fn null_space_of_integers(a: &Matrix<Number>) -> Matrix<Number> {
    let (rows, cols) = a.shape();
    let order = order_of(a, Number::byte_order);
    if rows == 0 || cols == 0 {
        return Matrix::from_fn(cols, cols, |i, j| {
            Number::new_from_u8((i == j) as u8, order)
        });
    }
    let mut m = integers(&[a]);
    let e = eliminate(&mut m, rows, cols, cols, false).expect("an elimination that goes on");
    let mut basis = Vec::with_capacity(cols * (cols - e.rank));
    null_vectors(&m, cols, &e, |f, y, d| {
        // `d·x`, of integers, then divided by the greatest common divisor of its elements, with
        // the sign that makes its element in column `f` positive.
        let mut v = vec![Number::new_from_u8(0, order); cols];
        v[f] = d.number(order);
        for (yi, &p) in y.iter().zip(&e.pivots) {
            v[p] = yi.number(order);
        }
        let mut g = v
            .iter()
            .fold(Number::new_from_u8(0, order), |g, x| g.gcd(x));
        if d.is_negative() {
            g = -g;
        }
        basis.extend(
            v.into_iter()
                .map(|x| if g == 1 { x } else { x.div_exact(&g) }),
        );
    });
    Matrix::from_column_major(cols, cols - e.rank, basis)
}

impl Matrix<Number> {
    /// The solution `X` of `A·X = B` over the rationals, exactly, or `None` when the matrix is
    /// singular: Bareiss's elimination of `[A | B]` to an echelon form with the last pivot
    /// `D = ±det(A)`, a fraction-free back substitution to the integers `D·X` (Cramer's rule),
    /// and one reduction of each element by a greatest common divisor; for the larger systems
    /// with few right-hand sides Dixon's p-adic lifting, as for
    /// [`Matrix::<Rational>::solve`](Matrix::<Rational>::solve). The result is in the byte order
    /// of the top left element of `A`.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square, or `B` has not as many rows.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    ///
    /// let a = Matrix::from_rows(&[[2, 1], [1, 3]]).map(|&x| Number::from(x));
    /// let b = Matrix::from_rows(&[[1], [1]]).map(|&x| Number::from(x));
    /// let x = a.solve(&b).unwrap();
    /// assert_eq!(x[(0, 0)].to_string(), "2/5");
    /// assert_eq!(x[(1, 0)].to_string(), "1/5");
    /// ```
    #[track_caller]
    pub fn solve(&self, b: &Matrix<Number>) -> Option<Matrix<Rational>> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        assert_eq!(b.rows(), n, "the right-hand side has not as many rows");
        let order = order_of(self, Number::byte_order);
        if n == 0 {
            return Some(Matrix::zeros(0, b.cols()));
        }
        solve_rows(integers(&[self, b]), n, b.cols(), None, order)
    }

    /// The inverse `A⁻¹` as a matrix of rationals, exactly, or `None` when the matrix is
    /// singular: [`solve`](Matrix::<Number>::solve) of `A·X = I`, each element reduced once.
    /// [`inverse`](Matrix::<Number>::inverse) gives it as the adjugate and the determinant.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::numbers::Number;
    /// use a::math::rational::Rational;
    ///
    /// let a = Matrix::from_rows(&[[2, 1], [1, 3]]).map(|&x| Number::from(x));
    /// let inv = a.inverse_rational().unwrap();
    /// assert_eq!(&a.map(|x| Rational::from(x)) * &inv, Matrix::identity(2));
    /// assert_eq!(inv[(0, 1)].to_string(), "-1/5");
    /// ```
    #[track_caller]
    pub fn inverse_rational(&self) -> Option<Matrix<Rational>> {
        assert!(self.is_square(), "the matrix is not square");
        let n = self.rows();
        let order = order_of(self, Number::byte_order);
        if n == 0 {
            return Some(Matrix::zeros(0, 0));
        }
        solve_rows(beside_identity(integers(&[self]), n), n, n, None, order)
    }
}

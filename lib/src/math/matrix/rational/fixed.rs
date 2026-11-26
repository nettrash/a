//! Bareiss's elimination on the stack, for integers whose minors fit in a fixed number of limbs.
//!
//! Every element is a minor of the matrix once the elimination has reached it, so Hadamard's bound
//! on the minors bounds them all: with at most `W` limbs for each, the elements are arrays of `W`
//! limbs and a sign in one array on the stack, the products of a step in `2W` limbs, and the exact
//! division by the previous pivot Hensel's, from the low end, of the quotient's own limbs only.
//! The steps take the lengths the elements have, short at the first steps and growing: no
//! vector, no allocation, and no kernel written for any length.

use super::{B, Int, S};

/// The largest order taken here.
pub(super) const MAX_ORDER: usize = 16;

/// An element: its sign and magnitude, of `len` limbs (zero for zero), the least significant
/// first, the limbs past `len` zero.
#[derive(Clone, Copy)]
struct E<const W: usize> {
    neg: bool,
    len: usize,
    d: [u64; W],
}

impl<const W: usize> E<W> {
    const ZERO: E<W> = E {
        neg: false,
        len: 0,
        d: [0; W],
    };

    /// The integer `x`, which fits.
    fn of(x: &Int) -> E<W> {
        let mut e = E::ZERO;
        match x {
            S(v) => {
                if *v != 0 {
                    e.neg = *v < 0;
                    e.d[0] = v.unsigned_abs();
                    e.len = 1;
                }
            }
            B(b) => {
                e.neg = b.neg;
                e.len = b.d.len();
                e.d[..e.len].copy_from_slice(&b.d);
            }
        }
        e
    }

    /// As an integer of the eliminations.
    fn int(&self) -> Int {
        Int::of_limbs(self.neg, self.d[..self.len].to_vec())
    }
}

/// `|x|·|y|` into `r`, of `x.len + y.len` limbs (the top one may be zero); `r` is overwritten.
#[inline(always)]
fn mul_into<const W: usize>(r: &mut [u64], x: &E<W>, y: &E<W>) -> usize {
    let n = x.len + y.len;
    r[..n].fill(0);
    for i in 0..x.len {
        let xi = u128::from(x.d[i]);
        let mut c = 0u64;
        for j in 0..y.len {
            let t = xi * u128::from(y.d[j]) + u128::from(r[i + j]) + u128::from(c);
            r[i + j] = t as u64;
            c = (t >> 64) as u64;
        }
        r[i + y.len] = c;
    }
    n
}

/// The length of `r` without its zero limbs at the top.
#[inline(always)]
fn trimmed(r: &[u64], mut n: usize) -> usize {
    while n > 0 && r[n - 1] == 0 {
        n -= 1;
    }
    n
}

/// `a ← a - b` for `a >= b`, `a` of `n` limbs, `b` of `m <= n`.
#[inline(always)]
fn sub_in(a: &mut [u64], n: usize, b: &[u64], m: usize) {
    let mut borrow = false;
    for i in 0..n {
        let bi = if i < m { b[i] } else { 0 };
        let (s, b1) = a[i].overflowing_sub(bi);
        let (s, b2) = s.overflowing_sub(u64::from(borrow));
        a[i] = s;
        borrow = b1 | b2;
        if i >= m && !borrow {
            break;
        }
    }
}

/// `a ← a + b`, `a` of `n` limbs with room for one more, `b` of `m <= n`: the new length.
#[inline(always)]
fn add_in(a: &mut [u64], n: usize, b: &[u64], m: usize) -> usize {
    let mut carry = false;
    for i in 0..n {
        let bi = if i < m { b[i] } else { 0 };
        let (s, c1) = a[i].overflowing_add(bi);
        let (s, c2) = s.overflowing_add(u64::from(carry));
        a[i] = s;
        carry = c1 | c2;
        if i >= m && !carry {
            break;
        }
    }
    if carry {
        a[n] = 1;
        n + 1
    } else {
        n
    }
}

/// Whether `a < b` for magnitudes of the lengths given, without zero limbs at the top.
#[inline(always)]
fn less(a: &[u64], na: usize, b: &[u64], nb: usize) -> bool {
    if na != nb {
        return na < nb;
    }
    for i in (0..na).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// The previous pivot, prepared to divide exactly by: its sign, its power of two, its odd part and
/// the inverse of the odd part's low limb modulo `2^64`.
struct Divisor<const W: usize> {
    neg: bool,
    shift: usize,
    odd: [u64; W],
    len: usize,
    inv: u64,
    /// Whether the magnitude is 1: the pivot is 1 or -1, and the quotient is the dividend with
    /// the sign of their product.
    one: bool,
}

impl<const W: usize> Divisor<W> {
    fn new(p: &E<W>) -> Divisor<W> {
        let limbs = p.d.iter().take_while(|&&x| x == 0).count();
        let bits = p.d[limbs].trailing_zeros() as usize;
        let shift = 64 * limbs + bits;
        let mut odd = [0u64; W];
        let n = p.len - limbs;
        for i in 0..n {
            let lo = p.d[limbs + i] >> bits;
            let hi = if bits != 0 && limbs + i + 1 < p.len {
                p.d[limbs + i + 1] << (64 - bits)
            } else {
                0
            };
            odd[i] = lo | hi;
        }
        let len = trimmed(&odd, n);
        Divisor {
            neg: p.neg,
            shift,
            odd,
            len,
            inv: crate::math::numbers::nat::binvert(odd[0]),
            one: len == 1 && odd[0] == 1 && shift == 0,
        }
    }

    /// `t / d` for the multiple `t` of `n` limbs (in place, as working memory) of the divisor, of
    /// the sign `neg`, whose quotient fits in `W` limbs: shifted right by the power of two, then
    /// Hensel's division from the low end, the quotient's `W` limbs only. The quotient's sign is
    /// that of the dividend times the divisor's, by either path.
    #[inline(always)]
    fn divide(&self, t: &mut [u64], n: usize, neg: bool) -> E<W> {
        let neg = neg != self.neg;
        let mut q = E::<W>::ZERO;
        if self.one {
            debug_assert!(n <= W);
            q.d[..n].copy_from_slice(&t[..n]);
            q.len = trimmed(&q.d, n);
            q.neg = neg && q.len != 0;
            return q;
        }
        // The shift, exact: the bits shifted out are zero.
        let (limbs, bits) = (self.shift / 64, (self.shift % 64) as u32);
        let m = n - limbs.min(n);
        for i in 0..m {
            let lo = t[limbs + i] >> bits;
            let hi = if bits != 0 && limbs + i + 1 < n {
                t[limbs + i + 1] << (64 - bits)
            } else {
                0
            };
            t[i] = lo | hi;
        }
        let m = trimmed(t, m);
        let qn = W.min(m);
        for i in 0..qn {
            let qi = t[i].wrapping_mul(self.inv);
            q.d[i] = qi;
            // t ← t - qi·odd·B^i, over the limbs below `qn` that later quotients read.
            let mut c = 0u64;
            let mut borrow = false;
            for j in 0..self.len.min(qn - i) {
                let p = u128::from(qi) * u128::from(self.odd[j]) + u128::from(c);
                c = (p >> 64) as u64;
                let (s, b1) = t[i + j].overflowing_sub(p as u64);
                let (s, b2) = s.overflowing_sub(u64::from(borrow));
                t[i + j] = s;
                borrow = b1 | b2;
            }
            // The carries past `qn - i` limbs reach no quotient limb; those below do.
            let mut k = i + self.len.min(qn - i);
            let mut c = u128::from(c) + u128::from(borrow);
            while c != 0 && k < qn {
                let (s, b) = t[k].overflowing_sub(c as u64);
                t[k] = s;
                c = (c >> 64) + u128::from(b);
                k += 1;
            }
        }
        q.len = trimmed(&q.d, qn);
        q.neg = neg && q.len != 0;
        q
    }
}

/// The determinant of the `n × n` matrix of integers `m`, row after row, `n <= MAX_ORDER`, by
/// Bareiss's elimination in elements of `W` limbs; the caller makes sure that every minor fits.
pub(super) fn det<const W: usize>(m: &[Int], n: usize) -> Int {
    debug_assert!(n <= MAX_ORDER);
    let mut a = [E::<W>::ZERO; MAX_ORDER * MAX_ORDER];
    for (slot, x) in a.iter_mut().zip(m) {
        *slot = E::of(x);
    }
    // The two products of a step, and their difference, in `2W` limbs and one to spare; `t` is
    // also the working memory of the division.
    let mut t = [0u64; 64];
    let mut u = [0u64; 64];
    debug_assert!(2 * W + 2 <= 64);
    let mut prev = Divisor::new(&E::<W>::of(&S(1)));
    let mut negate = false;
    for k in 0..n {
        // A pivot: the first element that is not zero at or below the row, in its column.
        let Some(p) = (k..n).find(|&i| a[i * n + k].len != 0) else {
            return S(0);
        };
        if p != k {
            for j in k..n {
                a.swap(k * n + j, p * n + j);
            }
            negate = !negate;
        }
        if k + 1 == n {
            break;
        }
        let pivot = a[k * n + k];
        for i in k + 1..n {
            let b = a[i * n + k];
            for j in k + 1..n {
                let (x, y) = (&a[i * n + j], &a[k * n + j]);
                // x·p - b·y, signed, in `t`, zeros above it.
                let (mut tn, mut tneg) = (0usize, false);
                if x.len != 0 {
                    let k = mul_into(&mut t, x, &pivot);
                    tn = trimmed(&t, k);
                    tneg = x.neg != pivot.neg;
                }
                if b.len != 0 && y.len != 0 {
                    let k = mul_into(&mut u, &b, y);
                    let un = trimmed(&u, k);
                    // The term `-b·y`, of the sign `uneg`.
                    let uneg = b.neg == y.neg;
                    if tn == 0 {
                        t[..un].copy_from_slice(&u[..un]);
                        tn = un;
                        tneg = uneg;
                    } else if tneg == uneg {
                        tn = add_in(&mut t, tn.max(un), &u, un);
                    } else if less(&t, tn, &u, un) {
                        // |u| - |t|, of the sign of the term.
                        sub_in(&mut u, un, &t, tn);
                        t[..un].copy_from_slice(&u[..un]);
                        tn = trimmed(&t, un);
                        tneg = uneg;
                    } else {
                        sub_in(&mut t, tn, &u, un);
                        tn = trimmed(&t, tn);
                    }
                }
                a[i * n + j] = if tn == 0 {
                    E::ZERO
                } else {
                    prev.divide(&mut t, tn, tneg)
                };
                t[..2 * W + 2].fill(0);
            }
        }
        prev = Divisor::new(&pivot);
    }
    let d = a[n * n - 1];
    let d = if negate {
        E {
            neg: !d.neg && d.len != 0,
            ..d
        }
    } else {
        d
    };
    d.int()
}

/// The limbs of every minor of the `n × n` matrix of integers `m`, by Hadamard's bound with a bit
/// for the sign: `None` past `max`.
pub(super) fn limbs(m: &[Int], n: usize, max: usize) -> Option<usize> {
    // |minor| <= Π over the rows of their Euclidean norms <= Π √n · 2^(bits of the largest).
    let mut bits = 0u64;
    for row in m.chunks_exact(n) {
        bits += row
            .iter()
            .map(|x| match x {
                S(v) => 64 - u64::from(v.unsigned_abs().leading_zeros()),
                B(b) => {
                    64 * b.d.len() as u64 - u64::from(b.d.last().map_or(64, |t| t.leading_zeros()))
                }
            })
            .max()
            .unwrap_or(0);
    }
    let lg = u64::from(usize::BITS - n.leading_zeros());
    bits += (n as u64 * lg).div_ceil(2) + 1;
    let w = bits.div_ceil(64) as usize;
    (w <= max).then_some(w)
}

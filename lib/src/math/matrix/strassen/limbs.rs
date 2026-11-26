//! The classical product of matrices of long numbers with each sum of products accumulated in
//! limbs: the operands read into limbs once, each product written by the limb kernels into a
//! buffer and added to one of two accumulators (the positive and the negative products), and one
//! number made from their difference. A sum of `k` products so costs `k` products of limbs and
//! additions, without the number each product and each partial sum of `c += a·b` would allocate.

use crate::math::numbers::nat::{add_assign, mul_to, sub_assign};
use crate::math::numbers::{ByteOrder, Magnitude, Number, from_limbs};
use core::cmp::Ordering;

/// Numbers in limbs: element `e`'s magnitude `d[at[e]..at[e + 1]]`, its sign `neg[e]`.
pub(super) struct Limbs {
    d: Vec<u64>,
    at: Vec<usize>,
    neg: Vec<bool>,
    /// The length of the longest magnitude.
    pub(super) longest: usize,
}

impl Limbs {
    /// The numbers `x(e)` for `e < len`, in that order.
    pub(super) fn new<'a>(len: usize, x: impl Fn(usize) -> &'a Number) -> Limbs {
        let mut d = Vec::new();
        let mut at = Vec::with_capacity(len + 1);
        let mut neg = Vec::with_capacity(len);
        let mut longest = 0;
        at.push(0);
        for e in 0..len {
            let m = Magnitude::of(x(e));
            let start = d.len();
            let w = m.bytes.len().div_ceil(8);
            d.resize(start + w, 0);
            for (k, &byte) in m.bytes.iter().enumerate() {
                // The `k`-th byte from the least significant end.
                let i = if m.big_endian {
                    m.bytes.len() - 1 - k
                } else {
                    k
                };
                d[start + i / 8] |= (byte as u64) << (8 * (i % 8));
            }
            longest = longest.max(w);
            at.push(d.len());
            neg.push(m.negative);
        }
        Limbs {
            d,
            at,
            neg,
            longest,
        }
    }

    #[inline(always)]
    pub(super) fn get(&self, e: usize) -> (bool, &[u64]) {
        (self.neg[e], &self.d[self.at[e]..self.at[e + 1]])
    }
}

/// `|x| - |y|` compared, for magnitudes of the same length.
fn cmp(x: &[u64], y: &[u64]) -> Ordering {
    for (a, b) in x.iter().rev().zip(y.iter().rev()) {
        match a.cmp(b) {
            Ordering::Equal => {}
            o => return o,
        }
    }
    Ordering::Equal
}

/// The working memory of the sums: the two accumulators, the product buffer and the limb
/// kernels' scratch.
pub(super) struct Sums {
    pos: Vec<u64>,
    neg: Vec<u64>,
    prod: Vec<u64>,
    scratch: Vec<u64>,
}

impl Sums {
    /// For sums of `k` products of magnitudes of at most `la` and `lb` limbs.
    pub(super) fn new(la: usize, lb: usize) -> Sums {
        let w = la + lb + 2;
        Sums {
            pos: vec![0; w],
            neg: vec![0; w],
            prod: vec![0; la + lb],
            scratch: Vec::new(),
        }
    }

    /// `start + Σ_p la[a(p)]·lb[b(p)]` for `p < k` as a number, little-endian.
    #[inline]
    pub(super) fn dot(
        &mut self,
        start: Option<&Number>,
        k: usize,
        a: impl Fn(usize) -> usize,
        la: &Limbs,
        b: impl Fn(usize) -> usize,
        lb: &Limbs,
    ) -> Number {
        self.pos.fill(0);
        self.neg.fill(0);
        if let Some(s) = start {
            let l = Limbs::new(1, |_| s);
            let (n, d) = l.get(0);
            if d.len() + 1 > self.pos.len() {
                self.pos.resize(d.len() + 1, 0);
                self.neg.resize(d.len() + 1, 0);
            }
            add_assign(if n { &mut self.neg } else { &mut self.pos }, d);
        }
        for p in 0..k {
            let (na, x) = la.get(a(p));
            let (nb, y) = lb.get(b(p));
            if x.is_empty() || y.is_empty() {
                continue;
            }
            let r = &mut self.prod[..x.len() + y.len()];
            mul_to(r, x, y, &mut self.scratch);
            // The accumulators hold k products of these lengths with a limb to spare.
            add_assign(
                if na != nb {
                    &mut self.neg
                } else {
                    &mut self.pos
                },
                r,
            );
        }
        // pos − neg.
        let (negative, mag) = match cmp(&self.pos, &self.neg) {
            Ordering::Equal => return Number::from(0u8),
            Ordering::Greater => {
                sub_assign(&mut self.pos, &self.neg);
                (false, &self.pos)
            }
            Ordering::Less => {
                sub_assign(&mut self.neg, &self.pos);
                (true, &self.neg)
            }
        };
        let mut x = from_limbs(mag, ByteOrder::LittleEndian);
        x.negative = negative;
        x
    }
}

/// `C = A·B` or `C += A·B` for the row-major views `a` (`m × k`, leading dimension `lda`), `b`
/// (`k × n`) and `c` (`m × n`), the sums in limbs.
#[allow(clippy::too_many_arguments)]
pub(super) fn classical(
    c: &mut [Number],
    ldc: usize,
    a: &[Number],
    lda: usize,
    b: &[Number],
    ldb: usize,
    (m, k, n): (usize, usize, usize),
    accumulate: bool,
) {
    let la = Limbs::new(m * k, |e| &a[(e / k) * lda + e % k]);
    let lb = Limbs::new(k * n, |e| &b[(e / n) * ldb + e % n]);
    let mut sums = Sums::new(la.longest, lb.longest);
    for i in 0..m {
        for j in 0..n {
            let at = i * ldc + j;
            let start = if accumulate {
                Some(c[at].clone())
            } else {
                None
            };
            c[at] = sums.dot(start.as_ref(), k, |p| i * k + p, &la, |p| p * n + j, &lb);
        }
    }
}

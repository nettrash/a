//! Sums and dot products of any number of terms, rounded once: the exact sum of the terms, or of
//! the exact products, as MPFR's `mpfr_sum` and `mpfr_dot`.
//!
//! The terms are added exactly into one fixed-point accumulator in two's complement, as long as
//! their bits span a modest range (the usual case): one product into a scratch buffer and one
//! shifted addition for each term, then one rounding. Terms whose bits span a range too wide for
//! that are sorted by their exponents and added exactly from the largest down, until the terms
//! left are too small to matter but through their sign: their sum then lies below a quarter of
//! the last place of the result and below the last bit of the sum so far, and only its sign is
//! needed, which the same accumulation gives at one bit, where the first such stop decides it.
//!
//! A sum of numbers (not products) is first tried in a window, as Lefèvre's algorithm for MPFR's
//! `mpfr_sum` has it: only the bits of the terms from `2^lo`, `prec + log2 n + 2 + 64` bits below
//! the largest exponent, go into the accumulator (on the stack when it is short), each term cut
//! below `2^lo` counted by its sign. The `c` terms cut move the sum by less than `c·2^lo`; when no
//! number of the precision and no point halfway between two lies within that of the sum of the
//! window, the sum of the window rounds as the exact sum does, in the same direction. Otherwise
//! (a cancellation that leaves too few bits, or a sum too near such a point), the exact sum. Terms
//! with products (dot products, the eliminations of matrices) take the same window, each product
//! computed whole into a buffer, when the terms reach more than twice as far below as the window
//! keeps; a sum it decides is then given as the window's top `need + 1` bits with a sticky bit.

use super::limbs;
use super::round::round_mag;
use super::{BigFloat, Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// One term of a sum: `±x`, or `±x·y` when `y` is given.
#[derive(Clone, Copy)]
pub(crate) struct Term<'a> {
    pub(crate) x: &'a BigFloat,
    pub(crate) y: Option<&'a BigFloat>,
    pub(crate) negate: bool,
}

impl<'a> Term<'a> {
    /// The term `x`.
    #[inline]
    pub(crate) fn one(x: &'a BigFloat, negate: bool) -> Term<'a> {
        Term { x, y: None, negate }
    }

    /// The term `x·y`.
    #[inline]
    pub(crate) fn product(x: &'a BigFloat, y: &'a BigFloat, negate: bool) -> Term<'a> {
        Term {
            x,
            y: Some(y),
            negate,
        }
    }
}

/// How a sum whose exact value is zero came to be: all of its terms zeros of one sign, zeros of
/// both signs, or terms that are not all zero cancelling exactly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Zeros {
    /// Every term `-0`.
    Negative,
    /// Every term `+0` (or no terms).
    Positive,
    /// Zeros of both signs, or terms that cancel.
    Mixed,
}

impl Zeros {
    /// Whether the zero is `-0` in the mode `round`: as IEEE 754's sums of zeros and MPFR's
    /// `mpfr_sum`, `-0` when every term is `-0`, and for mixed zeros or a cancellation only when
    /// rounding down.
    #[inline]
    pub(crate) fn negative(self, round: Round) -> bool {
        match self {
            Zeros::Negative => true,
            Zeros::Positive => false,
            Zeros::Mixed => round == Round::Down,
        }
    }

    /// The zeros of two sums of zeros together.
    #[inline]
    fn and(self, other: Zeros) -> Zeros {
        if self == other { self } else { Zeros::Mixed }
    }
}

/// The value of a sum before its rounding.
#[derive(Debug)]
pub(crate) enum Summed {
    /// NaN: a NaN term, `0·∞`, or infinities of both signs.
    Nan,
    /// An infinity of the sign.
    Inf(bool),
    /// An exact zero.
    Zero(Zeros),
    /// `±m·2^lsb`, `m` not zero and without zero limbs at the top, exactly; or with `sticky`,
    /// strictly between `±m·2^lsb` and `±(m + 1)·2^lsb`, `m` having more bits than the precision
    /// that the sum was asked for.
    Val {
        negative: bool,
        m: Vec<u64>,
        lsb: i128,
        sticky: bool,
    },
}

/// The number of bits of a bound `2^b ≥ n` of `n ≥ 1`: `ceil(log2 n)`.
#[inline]
fn log2_ceil(n: usize) -> i128 {
    if n <= 1 {
        0
    } else {
        i128::from(usize::BITS - (n - 1).leading_zeros())
    }
}

/// The number of zero limbs at the bottom of a mantissa.
#[inline]
fn lead(m: &[u64]) -> usize {
    m.iter().position(|&x| x != 0).unwrap_or(0)
}

/// The significant limbs of a finite number that is not zero and the exponent of their lowest
/// bit, the zero limbs below the precision left out.
#[inline]
fn significant(x: &BigFloat) -> (bool, &[u64], i128) {
    let (negative, m, lsb) = x.parts().expect("a finite number that is not zero");
    let z = lead(m);
    (negative, &m[z..], lsb + 64 * z as i128)
}

/// The buffers of the fixed-point accumulation, to keep across sums: the product of a term, its
/// shifted copy, the working memory of long products, and the accumulator itself (which a sum
/// rounded by [`round_summed_into`] gives back).
#[derive(Default)]
pub(crate) struct Work {
    product: Vec<u64>,
    shifted: Vec<u64>,
    scratch: Vec<u64>,
    acc: Vec<u64>,
}

/// The sum of `terms`, exactly or with a sticky bit below `need` bits: NaN, an infinity, an exact
/// zero, or the value.
pub(crate) fn sum_terms(terms: &[Term<'_>], need: u64) -> Summed {
    sum_terms_in(terms, need, &mut Work::default())
}

/// [`sum_terms`] with the buffers of `work`.
pub(crate) fn sum_terms_in(terms: &[Term<'_>], need: u64, work: &mut Work) -> Summed {
    // The special values first, as `mpfr_mul` and `mpfr_sum` have them.
    let mut inf: Option<bool> = None;
    let mut zeros: Option<Zeros> = None;
    let mut finite = 0usize;
    for t in terms {
        let (x, y) = (t.x, t.y);
        let negative =
            (x.is_sign_negative() != y.is_some_and(|y| y.is_sign_negative())) != t.negate;
        let nan = x.is_nan() || y.is_some_and(|y| y.is_nan());
        let infinite = x.is_infinite() || y.is_some_and(|y| y.is_infinite());
        let zero = x.is_zero() || y.is_some_and(|y| y.is_zero());
        if nan || infinite && zero {
            return Summed::Nan;
        }
        if infinite {
            match inf {
                Some(s) if s != negative => return Summed::Nan,
                _ => inf = Some(negative),
            }
        } else if zero {
            let z = if negative {
                Zeros::Negative
            } else {
                Zeros::Positive
            };
            zeros = Some(zeros.map_or(z, |w| w.and(z)));
        } else {
            finite += 1;
        }
    }
    if let Some(negative) = inf {
        return Summed::Inf(negative);
    }
    if finite == 0 {
        return Summed::Zero(zeros.unwrap_or(Zeros::Positive));
    }
    // The span of the bits: below `2^top` and from `2^low`.
    let mut top = i128::MIN;
    let mut low = i128::MAX;
    let mut longest = 0u64;
    for t in terms {
        if !t.x.is_normal() || t.y.is_some_and(|y| !y.is_normal()) {
            continue;
        }
        let (_, mx, lx) = significant(t.x);
        let (mut e, mut l, mut bits) = (lx + 64 * mx.len() as i128, lx, 64 * mx.len() as u64);
        if let Some(y) = t.y {
            let (_, my, ly) = significant(y);
            e += ly + 64 * my.len() as i128;
            l += ly;
            bits += 64 * my.len() as u64;
        }
        top = top.max(e);
        low = low.min(l);
        longest = longest.max(bits);
    }
    // The bits within reach of the result first (see the module), when the terms reach more
    // than twice as far below (so that the window leaves out more than it keeps, and pays for its
    // test); then the exact sums. Measured on an Apple M4 Max (`dot_round` of 4 to 64 terms at 53
    // to 4,000 bits, operands of the result's precision, twice and four times as long; products and
    // solves of `Matrix<BigFloat>` of order 4 to 16 at 53 to 1,000 bits; factors 0 to 4): once (the
    // window whenever it leaves bits out) takes the dot products of operands of the precision 0.73
    // to 0.96 times from 113 bits but costs them at 53 bits 8 to 11 % and the solves 4 to 14 %;
    // three and four times cost the dot products of longer operands 2 to 45 %.
    if top - low > 2 * (i128::from(need) + log2_ceil(finite) + 2 + SUM_GUARD)
        && let Some(s) = window_terms(terms, top, low, finite, need, work)
    {
        return s;
    }
    let span = top - low;
    if span <= 4 * i128::from(longest + need) + (1 << 16) {
        fixed(terms, top, low, finite, work)
    } else {
        sorted(terms, need)
    }
}

/// The exact sum by a two's-complement accumulator from `2^low` to past `2^top`, for `finite`
/// terms that are finite and not zero.
fn fixed(terms: &[Term<'_>], top: i128, low: i128, finite: usize, work: &mut Work) -> Summed {
    // `|sum| < finite·2^top`, and a sign bit above.
    let bits = (top - low) + log2_ceil(finite) + 2;
    let n = (bits as usize).div_ceil(64) + 1;
    let mut acc = core::mem::take(&mut work.acc);
    acc.clear();
    acc.resize(n, 0);
    let Work {
        product,
        shifted,
        scratch,
        ..
    } = work;
    for t in terms {
        if !t.x.is_normal() || t.y.is_some_and(|y| !y.is_normal()) {
            continue;
        }
        let (nx, mx, lx) = significant(t.x);
        let (m, lsb, negative) = match t.y {
            None => (mx, lx, nx != t.negate),
            Some(y) => {
                let (ny, my, ly) = significant(y);
                if core::ptr::eq(mx, my) {
                    nat::sqr_into(product, mx, scratch);
                } else {
                    nat::mul_into(product, mx, my, scratch);
                }
                (&product[..], lx + ly, (nx != ny) != t.negate)
            }
        };
        let off = (lsb - low) as u64;
        let (q, r) = ((off / 64) as usize, (off % 64) as u32);
        let m = if r == 0 {
            m
        } else {
            shifted.clear();
            let mut carry = 0u64;
            for &x in m {
                shifted.push(x << r | carry);
                carry = x >> (64 - r);
            }
            shifted.push(carry);
            &shifted[..]
        };
        // The carry or borrow out of the top is the two's complement wrapping.
        if negative {
            nat::sub_assign(&mut acc[q..], m);
        } else {
            nat::add_assign(&mut acc[q..], m);
        }
    }
    let negative = acc[n - 1] >> 63 == 1;
    if negative {
        // `-acc = !acc + 1`.
        let mut carry = true;
        for x in &mut acc {
            let (s, c) = (!*x).overflowing_add(u64::from(carry));
            *x = s;
            carry = c;
        }
    }
    limbs::trim(&mut acc);
    if acc.is_empty() {
        return Summed::Zero(Zeros::Mixed);
    }
    Summed::Val {
        negative,
        m: acc,
        lsb: low,
        sticky: false,
    }
}

/// An exact signed magnitude `±m·2^lsb`, `m` not zero and without zero limbs at the top.
struct Val {
    negative: bool,
    m: Vec<u64>,
    lsb: i128,
}

impl Val {
    /// The exponent of a bound: `|v| < 2^top`.
    #[inline]
    fn top(&self) -> i128 {
        self.lsb + i128::from(limbs::bit_len(&self.m))
    }
}

/// `a + b` exactly, `None` when they cancel.
fn add_exact(a: &Val, b: &Val) -> Option<Val> {
    let s = a.lsb.min(b.lsb);
    let x = limbs::shl(&a.m, (a.lsb - s) as u64);
    let y = limbs::shl(&b.m, (b.lsb - s) as u64);
    if a.negative == b.negative {
        return Some(Val {
            negative: a.negative,
            m: limbs::add(&x, &y),
            lsb: s,
        });
    }
    match limbs::cmp(&x, &y) {
        Ordering::Equal => None,
        Ordering::Greater => Some(Val {
            negative: a.negative,
            m: limbs::sub(&x, &y),
            lsb: s,
        }),
        Ordering::Less => Some(Val {
            negative: b.negative,
            m: limbs::sub(&y, &x),
            lsb: s,
        }),
    }
}

/// The scale below which the terms after the sum `s` so far change it only through their sign,
/// for a result of `need` bits: `min(lsb, top - need - 3)`, as [`add_mags`](super::arith::add_mags).
#[inline]
fn negligible_below(s: &Val, need: u64) -> i128 {
    s.lsb.min(s.top() - i128::from(need) - 3)
}

/// The sum of terms spread too widely for [`fixed`]: from the largest down, exactly, until the
/// rest is negligible (see the [module](self)).
fn sorted(terms: &[Term<'_>], need: u64) -> Summed {
    let mut vals: Vec<Val> = terms
        .iter()
        .filter(|t| t.x.is_normal() && t.y.is_none_or(|y| y.is_normal()))
        .map(|t| {
            let (nx, mx, lx) = significant(t.x);
            match t.y {
                None => Val {
                    negative: nx != t.negate,
                    m: mx.to_vec(),
                    lsb: lx,
                },
                Some(y) => {
                    let (ny, my, ly) = significant(y);
                    let mut m = limbs::mul(mx, my);
                    limbs::trim(&mut m);
                    Val {
                        negative: (nx != ny) != t.negate,
                        m,
                        lsb: lx + ly,
                    }
                }
            }
        })
        .collect();
    vals.sort_by_key(|v| core::cmp::Reverse(v.top()));
    let n = vals.len();
    let mut acc: Option<Val> = None;
    for i in 0..n {
        if let Some(s) = &acc {
            let k = negligible_below(s, need);
            // `|Σ_{j ≥ i} v_j| < (n - i)·2^top(v_i)`.
            if vals[i].top() + log2_ceil(n - i) <= k {
                let m = limbs::shl(&s.m, (s.lsb - k) as u64);
                let m = match tail_sign(&vals[i..]) {
                    None => {
                        return Summed::Val {
                            negative: s.negative,
                            m,
                            lsb: k,
                            sticky: false,
                        };
                    }
                    // The sum lies strictly between `m·2^k` and `(m ± 1)·2^k`.
                    Some(negative) if negative == s.negative => m,
                    Some(_) => limbs::sub(&m, &[1]),
                };
                return Summed::Val {
                    negative: s.negative,
                    m,
                    lsb: k,
                    sticky: true,
                };
            }
        }
        acc = match acc {
            None => Some(core::mem::replace(
                &mut vals[i],
                Val {
                    negative: false,
                    m: Vec::new(),
                    lsb: 0,
                },
            )),
            Some(s) => add_exact(&s, &vals[i]),
        };
    }
    match acc {
        None => Summed::Zero(Zeros::Mixed),
        Some(s) => Summed::Val {
            negative: s.negative,
            m: s.m,
            lsb: s.lsb,
            sticky: false,
        },
    }
}

/// The sign of the sum of `vals`, sorted by their exponents from the largest, `None` for zero:
/// once the sum so far `s` makes the rest negligible at one bit, the rest is below a quarter of
/// `|s|` and cannot turn its sign over.
fn tail_sign(vals: &[Val]) -> Option<bool> {
    let n = vals.len();
    let mut acc: Option<Val> = None;
    for (i, v) in vals.iter().enumerate() {
        if let Some(s) = &acc
            && v.top() + log2_ceil(n - i) <= negligible_below(s, 1)
        {
            return Some(s.negative);
        }
        acc = match acc {
            None => Some(Val {
                negative: v.negative,
                m: v.m.clone(),
                lsb: v.lsb,
            }),
            Some(s) => add_exact(&s, v),
        };
    }
    acc.map(|s| s.negative)
}

/// The most limbs of an accumulator that [`window_sum`] keeps on the stack (4,096 bits). Measured
/// on an Apple M4 Max (`sum_round` of 4 to 100 terms at 1,000 to 16,000 bits, `dot_round`, the
/// complex squares and norms; bounds 32 to 256): on the stack past 64 limbs the window takes 0.89
/// to 0.99 of an allocated one's time, but the buffers are made zero whole, and a bound of 96 costs
/// the sums of 17 to 64 limbs up to 7 % (128: 12 %).
const SUM_STACK: usize = 64;

/// The bits that [`window_sum`] keeps below the last place of the result (past the `log2 n` bits of
/// the carries), by which a sum must clear the numbers of the precision and the points halfway
/// between them for the terms it cuts not to matter. Measured on an Apple M4 Max (sums of 4 to 100
/// terms at 53 to 4,000 bits: terms of the precision and four times as long, cancellations of 16 to
/// 200 bits; dot products of operands twice as long; the complex squares and norms; 8 to 256 bits):
/// below 48 the window leaves more sums of terms of the precision undecided (up to 1.76 times the
/// time) and cancellations of 16 bits (up to 3.5), 48 those of 48 bits; 96 and 128 decide
/// cancellations of 48 to 100 bits (0.30 to 0.94 times) and long terms at 1,000 bits (0.64 to
/// 0.75), but cost long terms at 53 to 256 and at 4,000 bits, cancellations of 200 bits at 113 and
/// dot products at 53 bits 3 to 25 %; from 192 the complex squares' short products no longer fit
/// the window (twice the time).
const SUM_GUARD: i128 = 64;

/// `Σ xs` rounded to `prec` bits in the mode `round` from the bits of the terms in a window (see
/// the [module](self)), for terms that are finite, one at least not zero; `None` for the others
/// and when the window does not decide.
fn window_sum(xs: &[BigFloat], prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    // The largest exponent (`|x| < 2^top`) and the lowest bit of the terms.
    let mut top = i64::MIN;
    let mut low = i128::MAX;
    let mut finite = 0usize;
    for x in xs {
        match &x.kind {
            Kind::Finite { exp, mant, .. } => {
                top = top.max(*exp);
                low = low.min(i128::from(*exp) - 64 * mant.len() as i128);
                finite += 1;
            }
            Kind::Zero { .. } => {}
            Kind::Nan | Kind::Inf { .. } => return None,
        }
    }
    if finite == 0 {
        return None;
    }
    let top = i128::from(top);
    let carries = log2_ceil(finite);
    let lo = low.max(top - i128::from(prec) - carries - 2 - SUM_GUARD);
    // `|Σ| < finite·2^top`, and a sign bit above.
    let n = ((top - lo + carries + 2) as usize).div_ceil(64);
    // The accumulator and a buffer for a shifted term, on the stack (as few limbs made zero as will
    // do) or both in one vector. Measured on an Apple M4 Max (`sum_round` of 2 to 100 terms at 53
    // to 2,000 bits; first tiers of 0 to 32 limbs, and 10 in builds of its own): ten limbs would
    // take the sums of 9 and 10 limbs (500 to 570 bits) 0.79 to 0.93 times the time of the larger
    // buffer, but cost those of two to four terms at 113 bits 2 to 4 % in every build tried (1.00
    // to 1.02 times in one binary); from 12 the shortest sums (two to four terms of 53 to 113 bits)
    // lose 3 to 19 %.
    if n <= 8 {
        let mut s = [[0u64; 8]; 2];
        let [a, b] = &mut s;
        accumulate(xs, lo, prec, round, &mut a[..n], &mut b[..n])
    } else if n <= SUM_STACK {
        let mut s = [[0u64; SUM_STACK]; 2];
        let [a, b] = &mut s;
        accumulate(xs, lo, prec, round, &mut a[..n], &mut b[..n])
    } else {
        let mut v = vec![0; 2 * n];
        let (a, b) = v.split_at_mut(n);
        accumulate(xs, lo, prec, round, a, b)
    }
}

/// The sum of the bits of the terms from `2^lo` in the two's-complement accumulator `acc`
/// (zeros, as many limbs as it takes), `buf` as long, and its rounding when it decides: the rest
/// of [`window_sum`].
fn accumulate(
    xs: &[BigFloat],
    lo: i128,
    prec: u64,
    round: Round,
    acc: &mut [u64],
    buf: &mut [u64],
) -> Option<(BigFloat, Ordering)> {
    let mut cuts = Cuts::default();
    for x in xs {
        if let Kind::Finite {
            negative,
            exp,
            mant,
        } = &x.kind
        {
            let m: &[u64] = mant;
            let lsb = i128::from(*exp) - 64 * m.len() as i128;
            cuts.add(acc, buf, *negative, m, lsb, lo);
        }
    }
    match decide_window(acc, cuts, prec)? {
        Window::Zero => {
            let kind = Kind::Zero {
                negative: Zeros::Mixed.negative(round),
            };
            Some((BigFloat { prec, kind }, Ordering::Equal))
        }
        Window::Exact(negative, len) | Window::Cut(negative, len, _) => {
            Some(round_mag(negative, &acc[..len], lo, false, prec, round))
        }
    }
}

/// One term of [`round_approx_sum`]: `±M·2^lsb` known from below in magnitude, the exact term's
/// magnitude in `[M·2^lsb, M·2^lsb + 2^(lsb + err))`; exactly `±M·2^lsb` for `err = 0`.
#[derive(Clone, Copy)]
pub(crate) struct Approx<'a> {
    pub(crate) negative: bool,
    pub(crate) m: &'a [u64],
    pub(crate) lsb: i128,
    pub(crate) err: u64,
}

/// The sum of the exact terms that `parts` approximate (see [`Approx`]; not zero, aligned at
/// their top bits), rounded to `prec` bits in the mode `round` into `out`, from the window of
/// [`window_sum`]: each part's error below the window's `2^lo` (else `None`), so that the exact
/// term exceeds its bits in the window by less than two units of `2^lo` (its bits below `lo`, and
/// its error), counted twice among the cuts. `None` when the window does not decide, `out` then
/// untouched.
pub(crate) fn round_approx_sum(
    parts: &[Approx<'_>],
    prec: u64,
    round: Round,
    out: &mut Vec<u64>,
) -> Option<(Kind, Ordering)> {
    let top = parts.iter().map(|p| p.lsb + 64 * p.m.len() as i128).max()?;
    let low = parts.iter().map(|p| p.lsb).min()?;
    let carries = log2_ceil(parts.len());
    let lo = low.max(top - i128::from(prec) - carries - 2 - SUM_GUARD);
    // Exact parts (`err = 0`) are cut as any term; an error must lie below the window.
    if parts
        .iter()
        .any(|p| p.err > 0 && p.lsb + i128::from(p.err) > lo)
    {
        return None;
    }
    let n = ((top - lo + carries + 2) as usize).div_ceil(64);
    if n <= SUM_STACK {
        let mut s = [[0u64; SUM_STACK]; 2];
        let [a, b] = &mut s;
        approx_in(parts, lo, prec, round, &mut a[..n], &mut b[..n], out)
    } else {
        let mut v = vec![0u64; 2 * n];
        let (a, b) = v.split_at_mut(n);
        approx_in(parts, lo, prec, round, a, b, out)
    }
}

/// The rest of [`round_approx_sum`], into the zeros `acc` (and `buf` as long).
fn approx_in(
    parts: &[Approx<'_>],
    lo: i128,
    prec: u64,
    round: Round,
    acc: &mut [u64],
    buf: &mut [u64],
    out: &mut Vec<u64>,
) -> Option<(Kind, Ordering)> {
    let mut cuts = Cuts::default();
    for p in parts {
        cuts.add(acc, buf, p.negative, p.m, p.lsb, lo);
        // The error: one more unit of `2^lo` at most, in the term's direction.
        match (p.err, p.negative) {
            (0, _) => {}
            (_, true) => cuts.down += 1,
            (_, false) => cuts.up += 1,
        }
    }
    let (negative, len) = match decide_window(acc, cuts, prec)? {
        Window::Zero => return None,
        Window::Exact(negative, len) | Window::Cut(negative, len, _) => (negative, len),
    };
    Some(super::round::round_into(
        negative,
        &acc[..len],
        lo,
        false,
        prec,
        round,
        core::mem::take(out),
    ))
}

/// The terms of a window cut below `2^lo` (their bits below it not all zero), positive and
/// negative: each moves the sum of the window by less than `2^lo`, up for a positive term, down
/// for a negative one.
#[derive(Default, Clone, Copy)]
struct Cuts {
    up: u64,
    down: u64,
}

impl Cuts {
    /// Adds the bits from `2^lo` of `±m·2^lsb` (`m` not zero) to the window `acc` (`buf` as long),
    /// counting the term when it is cut.
    #[inline]
    fn add(
        &mut self,
        acc: &mut [u64],
        buf: &mut [u64],
        negative: bool,
        m: &[u64],
        lsb: i128,
        lo: i128,
    ) {
        let off = lsb - lo;
        let lost = if off >= 0 {
            let (q, r) = ((off / 64) as usize, (off % 64) as u32);
            add_shifted(&mut acc[q..], m, 0, r, negative, buf);
            false
        } else {
            let cut = (-off) as u64;
            if cut >= 64 * m.len() as u64 {
                // All of it below the window.
                true
            } else {
                let (q, b) = ((cut / 64) as usize, (cut % 64) as u32);
                let lost = m[..q].iter().any(|&w| w != 0) || m[q] & ((1u64 << b) - 1) != 0;
                if b == 0 {
                    add_shifted(acc, &m[q..], 0, 0, negative, buf);
                } else {
                    // `m >> cut`: the limbs from `q + 1` shifted up by `64 - b`, `m[q]` below.
                    add_shifted(acc, &m[q + 1..], m[q], 64 - b, negative, buf);
                }
                lost
            }
        };
        if lost {
            if negative {
                self.down += 1;
            } else {
                self.up += 1;
            }
        }
    }
}

/// What the window says of the exact sum: zero, exactly `±B` (the `len` limbs of the window, its
/// sign apart), or within the cuts of `±B` and rounding as `B` does at `need` bits, `t =
/// bits(B) - need ≥ 2` bits below them.
enum Window {
    Zero,
    Exact(bool, usize),
    Cut(bool, usize, u64),
}

/// The magnitude of the two's-complement window `acc` in place and what it decides (see
/// [`Window`]); `None` when the cuts leave the rounding at `need` bits open.
fn decide_window(acc: &mut [u64], cuts: Cuts, need: u64) -> Option<Window> {
    let n = acc.len();
    let negative = acc[n - 1] >> 63 == 1;
    if negative {
        // `-acc = !acc + 1`.
        let mut carry = true;
        for x in acc.iter_mut() {
            let (s, c) = (!*x).overflowing_add(u64::from(carry));
            *x = s;
            carry = c;
        }
    }
    let len = acc.iter().rposition(|&w| w != 0).map_or(0, |i| i + 1);
    let cuts = cuts.up.max(cuts.down);
    if len == 0 {
        // Exact cancellation, or a sum of the window of zero, which cannot decide.
        return (cuts == 0).then_some(Window::Zero);
    }
    if cuts == 0 {
        return Some(Window::Exact(negative, len));
    }
    // The exact sum is within `cuts` units of `2^lo` of the sum of the window `±B`: the rounding
    // is that of `B` when no multiple of `2^(t-1)`, `t = bits(B) - need` (the numbers of `need`
    // bits and the points halfway between them), lies within `cuts` of `B`, which then is none of
    // those itself, and the rounding is inexact either way.
    let m = &acc[..len];
    let t = limbs::bit_len(m).checked_sub(need)?;
    if t < 2
        || !low_bits_at_least(m, t - 1, cuts + 1, false)
        || !low_bits_at_least(m, t - 1, cuts, true)
    {
        return None;
    }
    Some(Window::Cut(negative, len, t))
}

/// [`sum_terms_in`] in a window, as [`window_sum`] has it, for terms and products that are finite
/// (`finite` of them not zero), the largest of exponent `top` (`|x·y| < 2^top`) and the lowest
/// bit at `2^low`: the products into `work`'s buffer one at a time. A cut sum is given as `M` of
/// `need + 1` bits with a sticky bit, as no number of `need` bits and no point halfway between
/// two lies between the exact sum and `B`. `None` when the window does not decide.
fn window_terms(
    terms: &[Term<'_>],
    top: i128,
    low: i128,
    finite: usize,
    need: u64,
    work: &mut Work,
) -> Option<Summed> {
    let carries = log2_ceil(finite);
    let lo = low.max(top - i128::from(need) - carries - 2 - SUM_GUARD);
    let n = ((top - lo + carries + 2) as usize).div_ceil(64);
    if n <= 8 {
        let mut s = [[0u64; 8]; 2];
        let [a, b] = &mut s;
        window_terms_in(terms, lo, need, &mut a[..n], &mut b[..n], work)
    } else if n <= SUM_STACK {
        let mut s = [[0u64; SUM_STACK]; 2];
        let [a, b] = &mut s;
        window_terms_in(terms, lo, need, &mut a[..n], &mut b[..n], work)
    } else {
        let mut v = vec![0; 2 * n];
        let (a, b) = v.split_at_mut(n);
        window_terms_in(terms, lo, need, a, b, work)
    }
}

/// The rest of [`window_terms`], into the zeros `acc` (and `buf` as long).
fn window_terms_in(
    terms: &[Term<'_>],
    lo: i128,
    need: u64,
    acc: &mut [u64],
    buf: &mut [u64],
    work: &mut Work,
) -> Option<Summed> {
    let mut cuts = Cuts::default();
    for t in terms {
        if !t.x.is_normal() || t.y.is_some_and(|y| !y.is_normal()) {
            continue;
        }
        let (nx, mx, lx) = significant(t.x);
        match t.y {
            None => cuts.add(acc, buf, nx != t.negate, mx, lx, lo),
            Some(y) => {
                let (ny, my, ly) = significant(y);
                let Work {
                    product, scratch, ..
                } = &mut *work;
                if core::ptr::eq(mx, my) {
                    nat::sqr_into(product, mx, scratch);
                } else {
                    nat::mul_into(product, mx, my, scratch);
                }
                let k = product.iter().rposition(|&w| w != 0).map_or(0, |i| i + 1);
                cuts.add(acc, buf, (nx != ny) != t.negate, &product[..k], lx + ly, lo);
            }
        }
    }
    let w = decide_window(acc, cuts, need)?;
    let mut v = core::mem::take(&mut work.acc);
    v.clear();
    Some(match w {
        Window::Zero => Summed::Zero(Zeros::Mixed),
        Window::Exact(negative, len) => {
            v.extend_from_slice(&acc[..len]);
            Summed::Val {
                negative,
                m: v,
                lsb: lo,
                sticky: false,
            }
        }
        Window::Cut(negative, len, t) => {
            // `M = floor(B / 2^(t-1))`, the exact sum strictly between `M` and `M + 1` units.
            let m = limbs::shr(&acc[..len], t - 1);
            Summed::Val {
                negative,
                m,
                lsb: lo + i128::from(t - 1),
                sticky: true,
            }
        }
    })
}

/// Whether the bits `0..k` of `m` (`k ≥ 1`), or of `!m` when `flip`, are at least `d` as a number.
#[inline]
fn low_bits_at_least(m: &[u64], k: u64, d: u64, flip: bool) -> bool {
    let f = if flip { u64::MAX } else { 0 };
    let (q, b) = ((k / 64) as usize, (k % 64) as u32);
    // The limbs above the lowest, whole, then the bits of a part limb: any set is at least `2^64`.
    let high = (1..q.min(m.len())).any(|i| m[i] ^ f != 0)
        || (q >= 1 && b != 0 && (m.get(q).copied().unwrap_or(0) ^ f) & ((1u64 << b) - 1) != 0);
    if high {
        return true;
    }
    let w0 = m[0] ^ f;
    let w0 = if q == 0 { w0 & ((1u64 << b) - 1) } else { w0 };
    w0 >= d
}

/// The fewest limbs of a term that [`add_shifted`] shifts into its buffer and adds by the
/// integers' kernels rather than limb by limb in one pass. Measured on an Apple M4 Max (`sum_round`
/// of 4 to 100 terms of 1 to 20 limbs, each way for all): limb by limb takes 0.78 to 1.01 of the
/// kernels' time to 7 limbs, 0.96 to 1.05 at 8 and 1.11 to 1.56 from 9.
const SHIFTED_KERNEL: usize = 8;

/// Adds (or subtracts, when `negative`) `m·2^r + (prev >> (64 - r))` (`m` alone when `r = 0`) to
/// the two's-complement number `acc`, which holds it, the carry or borrow running to its top:
/// a short term limb by limb, a longer one shifted into `buf` (as long as `acc`), then added.
#[inline]
fn add_shifted(acc: &mut [u64], m: &[u64], prev: u64, r: u32, negative: bool, buf: &mut [u64]) {
    if m.len() < SHIFTED_KERNEL {
        if negative {
            shifted_op::<true>(acc, m, prev, r);
        } else {
            shifted_op::<false>(acc, m, prev, r);
        }
        return;
    }
    let s = if r == 0 {
        m
    } else {
        let k = m.len();
        let s = &mut buf[..=k];
        s[0] = m[0] << r | prev >> (64 - r);
        for (d, w) in s[1..k].iter_mut().zip(m.windows(2)) {
            *d = w[1] << r | w[0] >> (64 - r);
        }
        s[k] = m[k - 1] >> (64 - r);
        &s[..]
    };
    // The carry or borrow out of the top is the two's complement wrapping.
    if negative {
        nat::sub_assign(acc, s);
    } else {
        nat::add_assign(acc, s);
    }
}

/// [`add_shifted`] for one sign.
#[inline]
fn shifted_op<const SUB: bool>(acc: &mut [u64], m: &[u64], prev: u64, r: u32) {
    #[inline(always)]
    fn step<const SUB: bool>(a: &mut u64, v: u64, c: bool) -> bool {
        if SUB {
            let (d, b1) = a.overflowing_sub(v);
            let (d, b2) = d.overflowing_sub(u64::from(c));
            *a = d;
            b1 | b2
        } else {
            let (s, c1) = a.overflowing_add(v);
            let (s, c2) = s.overflowing_add(u64::from(c));
            *a = s;
            c1 | c2
        }
    }
    let mut c = false;
    let mut i = m.len();
    if r == 0 {
        for (a, &w) in acc.iter_mut().zip(m) {
            c = step::<SUB>(a, w, c);
        }
    } else {
        let mut p = prev;
        for (a, &w) in acc.iter_mut().zip(m) {
            c = step::<SUB>(a, w << r | p >> (64 - r), c);
            p = w;
        }
        c = step::<SUB>(&mut acc[i], p >> (64 - r), c);
        i += 1;
    }
    while c && i < acc.len() {
        c = step::<SUB>(&mut acc[i], 0, true);
        i += 1;
    }
}

/// [`round_summed`] into `out` at its precision, in its vector, the sum's vector given back to
/// `work` for the next sum: the direction of the rounding.
pub(crate) fn round_summed_into(
    s: Summed,
    out: &mut BigFloat,
    round: Round,
    work: &mut Work,
) -> Ordering {
    let prec = out.prec;
    let (kind, o) = match s {
        Summed::Nan => (Kind::Nan, Ordering::Equal),
        Summed::Inf(negative) => (Kind::Inf { negative }, Ordering::Equal),
        Summed::Zero(z) => (
            Kind::Zero {
                negative: z.negative(round),
            },
            Ordering::Equal,
        ),
        Summed::Val {
            negative,
            m,
            lsb,
            sticky,
        } => {
            let r =
                super::round::round_into(negative, &m, lsb, sticky, prec, round, out.take_mant());
            if m.capacity() > work.acc.capacity() {
                work.acc = m;
            }
            r
        }
    };
    out.kind = kind;
    o
}

/// The sum rounded to `prec` bits in the mode `round`, and the direction of the rounding.
pub(crate) fn round_summed(s: Summed, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let kind = match s {
        Summed::Nan => Kind::Nan,
        Summed::Inf(negative) => Kind::Inf { negative },
        Summed::Zero(z) => Kind::Zero {
            negative: z.negative(round),
        },
        Summed::Val {
            negative,
            m,
            lsb,
            sticky,
        } => return round_mag(negative, &m, lsb, sticky, prec, round),
    };
    (BigFloat { prec, kind }, Ordering::Equal)
}

/// The square root of a sum that is not below zero (NaN below), rounded to `prec` bits in the
/// mode `round`; the sum asked for with `2·prec + 4` bits at least.
///
/// A sticky sum lies strictly between `M·2^lsb` and `(M + 1)·2^lsb`, `M` of at least
/// `2·prec + 4` bits: the squares of the numbers of `prec` bits near its root and of the points
/// halfway between them are multiples of `2^(2(e - prec - 1))`, `e` the exponent of the root,
/// which is at least `lsb + 2`, so none lies in that interval, and the root of
/// `(2M + 1)·2^(lsb - 1)` rounds as the root of the sum, inexactly as it does.
pub(crate) fn sqrt_summed(s: Summed, prec: u64, round: Round) -> (BigFloat, Ordering) {
    match s {
        Summed::Val {
            negative: false,
            m,
            lsb,
            sticky,
        } => {
            if sticky {
                let mut m = limbs::shl(&m, 1);
                m[0] |= 1;
                super::arith::sqrt_mag(&m, lsb - 1, prec, round)
            } else {
                super::arith::sqrt_mag(&m, lsb, prec, round)
            }
        }
        Summed::Val { .. } | Summed::Inf(true) => (BigFloat::nan(prec), Ordering::Equal),
        Summed::Zero(z) => (
            BigFloat {
                prec,
                kind: Kind::Zero {
                    negative: z.negative(round),
                },
            },
            Ordering::Equal,
        ),
        s => round_summed(s, prec, round),
    }
}

impl BigFloat {
    /// The sum of `terms` rounded once to `prec` bits in the mode `round`, and the direction of
    /// the rounding, as MPFR's `mpfr_sum`: the exact sum, however many terms and however far
    /// apart their exponents. NaN when a term is NaN or two are infinities of opposite signs; an
    /// exact zero `+0`, or `-0` when every term is `-0`, or when the terms are zeros of both
    /// signs or cancel and the mode rounds down; the sum of no terms `+0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let big = BigFloat::from(1u64 << 60);
    /// let terms = [big.clone(), BigFloat::from(1), -big];
    /// // Added one by one at 53 bits, the 1 would be lost; the sum is exact.
    /// let (s, o) = BigFloat::sum_round(&terms, 53, Round::Nearest);
    /// assert_eq!(s, BigFloat::from(1));
    /// assert_eq!(o, Ordering::Equal);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn sum_round(terms: &[BigFloat], prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = window_sum(terms, prec, round) {
            return r;
        }
        let terms: Vec<Term<'_>> = terms.iter().map(|x| Term::one(x, false)).collect();
        round_summed(sum_terms(&terms, prec), prec, round)
    }

    /// The dot product `Σ xs[i]·ys[i]` rounded once to `prec` bits in the mode `round`, and the
    /// direction of the rounding, as MPFR's `mpfr_dot`: the exact sum of the exact products. The
    /// special values of each product are those of [`mul_round`](BigFloat::mul_round) (`0·∞` is
    /// NaN), and those of the sum those of [`sum_round`](BigFloat::sum_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let x = [BigFloat::from(3), BigFloat::from(-5)];
    /// let y = [BigFloat::from(7), BigFloat::from(4)];
    /// assert_eq!(BigFloat::dot_round(&x, &y, 10, Round::Nearest), (BigFloat::from(1), Ordering::Equal));
    /// ```
    ///
    /// # Panics
    ///
    /// When the slices are not of the same length, or `prec` is not in
    /// [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn dot_round(
        xs: &[BigFloat],
        ys: &[BigFloat],
        prec: u64,
        round: Round,
    ) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        assert_eq!(xs.len(), ys.len(), "the slices are not of the same length");
        let terms: Vec<Term<'_>> = xs
            .iter()
            .zip(ys)
            .map(|(x, y)| Term::product(x, y, false))
            .collect();
        round_summed(sum_terms(&terms, prec), prec, round)
    }
}

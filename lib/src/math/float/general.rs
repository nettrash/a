//! The arithmetic of numbers of any length on limbs, each operation computing only the limbs that
//! its rounding needs and deciding the rest from what it knows of them, as MPFR's `mpfr_add1`,
//! `mpfr_sub1`, `mpfr_mul`, `mpfr_div` and `mpfr_sqrt` do; the exact computation is the fallback
//! for the rare operands that leave the rounding undecided.
//!
//! An operand is a mantissa aligned at its top bit, `0.m·2^exp`. A result of `n` limbs is computed
//! on a window of `n + 1` limbs or more, so that its bits below the precision hold the round bit,
//! and what lies below the window is known as a sticky bit, or only within a few units of the
//! window's lowest limb: then [`decided`] tells whether every value in that range rounds alike,
//! which it does unless the lowest limb is within those units of a limb boundary or of a halfway
//! point.

use super::limbs::{self, Work};
use super::round::{round_into, round_vec};
use super::same::{self, Put};
use super::small;
use super::{BigFloat, Kind, Round};
use crate::math::numbers::nat;
use core::cmp::Ordering;

/// An operand: the sign, the mantissa aligned at its top bit (not empty) and the exponent `e` of
/// `0.m·2^e`.
#[derive(Clone, Copy)]
pub(crate) struct Op<'a> {
    pub(crate) negative: bool,
    pub(crate) m: &'a [u64],
    pub(crate) exp: i128,
}

impl<'a> Op<'a> {
    /// The operand without the zero limbs at the bottom of its mantissa.
    #[inline]
    pub(crate) fn strip(self) -> Op<'a> {
        Op {
            m: &self.m[limbs::low_zeros(self.m)..],
            ..self
        }
    }

    /// The exponent of the lowest bit of the mantissa.
    #[inline]
    fn lsb(&self) -> i128 {
        self.exp - 64 * self.m.len() as i128
    }
}

/// Whether every integer from `q - lo` to `q + hi` rounds to `prec` bits as `q` with a sticky bit
/// does, so that a value known only to lie among them (or between them and one more) rounds as
/// that: the low limb of each is not zero and the others are those of `q`, so that their bits
/// above the round bit, the round bit and the sticky bit (set, as the bits below the round bit
/// are not all zero) are the same. `q` has at least two limbs, the top one not zero.
#[inline]
pub(crate) fn decided(q: &[u64], lo: u64, hi: u64, prec: u64) -> bool {
    let g = q[0];
    if q.len() < 2 || g <= lo || g > u64::MAX - hi {
        return false;
    }
    let bits = limbs::bit_len(q);
    if bits < prec + 2 {
        return false;
    }
    // The round bit.
    let r = bits - prec - 1;
    if r >= 64 {
        return true;
    }
    let (a, b) = (g - lo, g + hi);
    a >> r == b >> r && a & ((1u64 << r) - 1) != 0
}

/// Sets `dst` to the bits of `y` from bit `o` up (`dst[i]` is bits `o + 64i..` of `y` as an
/// integer), zero outside `y`.
#[inline]
fn window(y: &[u64], o: i128, dst: &mut [u64]) {
    dst.fill(0);
    let (n, ly) = (dst.len() as i128, y.len() as i128);
    let k = o.div_euclid(64);
    let b = o.rem_euclid(64) as u32;
    // `dst[i]` takes `y[k + i] >> b` and `y[k + i + 1] << (64 - b)`.
    let range = |shift: i128| -> (usize, usize) {
        let lo = (-(k + shift)).clamp(0, n);
        let hi = (ly - (k + shift)).clamp(lo, n);
        (lo as usize, hi as usize)
    };
    let (a, z) = range(0);
    let base = |i: usize, shift: i128| (k + shift + i as i128) as usize;
    if b == 0 {
        if a < z {
            dst[a..z].copy_from_slice(&y[base(a, 0)..base(z, 0)]);
        }
        return;
    }
    if a < z {
        for (d, &v) in dst[a..z].iter_mut().zip(&y[base(a, 0)..base(z, 0)]) {
            *d = v >> b;
        }
    }
    let (a, z) = range(1);
    if a < z {
        for (d, &v) in dst[a..z].iter_mut().zip(&y[base(a, 1)..base(z, 1)]) {
            *d |= v << (64 - b);
        }
    }
}

/// `x ± y` exactly, then rounded: both at the scale of the lower of their lowest bits. `x` is the
/// larger, in magnitude for a difference, and the length is bounded by the operands' and the
/// window's, as only a term that reaches into the other or into the window comes here.
#[cold]
fn exact_sum(x: Op<'_>, y: Op<'_>, prec: u64, round: Round, out: Vec<u64>) -> (Kind, Ordering) {
    let (lx, ly) = (x.lsb(), y.lsb());
    let s = lx.min(ly);
    let len = ((x.exp - s) / 64 + 2) as usize;
    let mut e = vec![0u64; len];
    window(x.m, s - lx, &mut e);
    if shift_add(&mut e, y.m, s - ly, x.negative != y.negative) {
        unreachable!("an exact sum carries out of its limbs");
    }
    limbs::trim(&mut e);
    if e.is_empty() {
        return (
            Kind::Zero {
                negative: round == Round::Down,
            },
            Ordering::Equal,
        );
    }
    round_into(x.negative, &e, s, false, prec, round, out)
}

/// `x + y` for operands of signs of their own (a difference when they differ), rounded to `prec`
/// bits in the mode `round`, into `out`.
///
/// Operands of the result's length (three or more limbs) are added by [`same::add`]; operands of
/// at most four limbs, or any for a result of at most two, are tried in a word first
/// ([`small::add_short`]). Otherwise:
///
/// The larger term `x` (in exponent; in magnitude for a difference of one exponent) sets a window
/// of `w` limbs below its top bit, `w = n + 1` for `n = ceil(prec / 64)`. Its limbs in the window,
/// plus or minus the bits of `y` there (shifted into place), make `T`, with a carry limb for a
/// sum. What lies below the window, the tails, decides the rest:
///
/// - no tails: `T` is exact;
/// - one tail: `T` is the floor of the sum (`T - 1` for a difference whose tail is `y`'s), and the
///   tail, not zero, a sticky bit;
/// - both tails: the sum is within `[T, T + 2)` units of the window's lowest limb (a difference
///   within `(T - 1, T + 1)`), and [`decided`] tells when that is enough; otherwise the exact sum;
/// - `y` below the lowest bit of `x` and below the window: `y` changes the result only as a sticky
///   bit (and a difference `T` by one unit when `x` has no bits below the window).
///
/// A difference of exponents more than one apart cancels at most one bit, and the window keeps
/// `64w - 2`. One of exponents at most one apart can cancel any number: the top limbs where `x`
/// and `y` agree cancel exactly (the first that differs takes any borrow from below), the window
/// of `w = n + 2` limbs starts below them, and when it still holds fewer than `prec + 64` bits
/// the difference is computed exactly.
pub(crate) fn add(
    x: Op<'_>,
    y: Op<'_>,
    prec: u64,
    round: Round,
    mut out: Vec<u64>,
) -> (Kind, Ordering) {
    let n = prec.div_ceil(64) as usize;
    if n >= 3 && x.m.len() == n && y.m.len() == n {
        out.clear();
        out.resize(n, 0);
        return match same::add(
            x.negative, x.m, x.exp, y.negative, y.m, y.exp, prec, round, &mut out,
        ) {
            Put::Finite(negative, exp, o) => (
                Kind::Finite {
                    negative,
                    exp,
                    mant: out,
                },
                o,
            ),
            Put::Other(kind, o) => (kind, o),
        };
    }
    let short = x.m.len() <= 4 && y.m.len() <= 4;
    if (n <= 2 || short)
        && word_exp(x.exp)
        && word_exp(y.exp)
        && let Some(r) = small::with_vec(prec, &mut out, |o| {
            small::add_short(
                x.negative,
                x.m,
                x.exp as i64,
                y.negative,
                y.m,
                y.exp as i64,
                prec,
                round,
                o,
            )
        })
    {
        return r;
    }
    let (x, y) = (x.strip(), y.strip());
    let sub = x.negative != y.negative;
    let (x, y) = if x.exp > y.exp {
        (x, y)
    } else if x.exp < y.exp {
        (y, x)
    } else if sub {
        match limbs::cmp_top(x.m, y.m) {
            Ordering::Greater => (x, y),
            Ordering::Less => (y, x),
            Ordering::Equal => {
                return (
                    Kind::Zero {
                        negative: round == Round::Down,
                    },
                    Ordering::Equal,
                );
            }
        }
    } else {
        (x, y)
    };
    let d = x.exp - y.exp;
    let near = sub && d <= 1;
    let la = x.m.len();
    let lx = x.lsb();
    let ly = y.lsb();
    // The limbs at the top where `x` and `y` agree, for a difference that may cancel.
    let mut skip = 0usize;
    if near {
        while skip < la
            && x.m[la - 1 - skip] == limbs::extract(y.m, x.exp - 64 * (skip as i128 + 1) - ly)
        {
            skip += 1;
        }
    }
    let wl = n + 1 + usize::from(near);
    // The lowest bit of the window.
    let u = x.exp - 64 * (skip + wl) as i128;
    let far = y.exp <= lx.min(u);
    // The window `T`, in the vector of the result, with room for a carry.
    out.clear();
    out.reserve(wl + 1);
    let xtop = la - skip;
    let xtail = xtop > wl;
    if xtop >= wl {
        out.extend_from_slice(&x.m[xtop - wl..xtop]);
    } else {
        out.resize(wl - xtop, 0);
        out.extend_from_slice(&x.m[..xtop]);
    }
    let x_tail_nonzero = || x.m[..xtop.saturating_sub(wl)].iter().any(|&v| v != 0);
    let negative = x.negative;
    if far {
        // `0 < y < 2^lx` and `y < 2^u`: for a sum, the bits of `x` below the window and `y` stay
        // below one unit of it; for a difference too when those bits of `x` are not all zero,
        // and otherwise the difference takes one unit off.
        if sub && !(xtail && x_tail_nonzero()) {
            let borrow = nat::sub_assign(&mut out, &[1]);
            debug_assert!(!borrow);
        }
        return round_vec(negative, out, u, true, prec, round);
    }
    // The bits of `y` in the window: limb `i` is bits `o + 64i..` of `y`'s mantissa.
    let o = u - ly;
    let ytail = ly < u;
    if shift_add(&mut out, y.m, o, sub) {
        debug_assert!(!sub);
        out.push(1);
    }
    limbs::trim(&mut out);
    if out.is_empty() || (near && limbs::bit_len(&out) < prec + 64) {
        // Cancelled beyond the window.
        return exact_sum(x, y, prec, round, out);
    }
    let y_tail_nonzero = || limbs::any_below(y.m, o);
    let sticky = match (xtail, ytail) {
        (false, false) => false,
        (true, false) => decided(&out, 0, 0, prec) || x_tail_nonzero(),
        (false, true) if !sub => decided(&out, 0, 0, prec) || y_tail_nonzero(),
        (false, true) => {
            if decided(&out, 1, 0, prec) {
                true
            } else if y_tail_nonzero() {
                let borrow = nat::sub_assign(&mut out, &[1]);
                debug_assert!(!borrow);
                limbs::trim(&mut out);
                true
            } else {
                false
            }
        }
        (true, true) => {
            let (lo, hi) = if sub { (1, 0) } else { (0, 1) };
            if decided(&out, lo, hi, prec) {
                true
            } else {
                return exact_sum(x, y, prec, round, out);
            }
        }
    };
    round_vec(negative, out, u, sticky, prec, round)
}

/// `t ± y·2^-o`, the bits of `y` from bit `o` up added to (subtracted from) `t` (`t[i]` takes
/// bits `o + 64i..` of `y`), with the carry (borrow) carried through `t`; returns the carry
/// (borrow) out of it. Short windows in one pass, long ones by the shifted limbs and the crate's
/// kernels.
fn shift_add(t: &mut [u64], y: &[u64], o: i128, sub: bool) -> bool {
    let (n, ly) = (t.len() as i128, y.len() as i128);
    let k = o.div_euclid(64);
    let b = o.rem_euclid(64) as u32;
    // The limbs of `t` that bits of `y` reach.
    let lo = (if b > 0 { -k - 1 } else { -k }).max(0);
    let hi = (ly - k).min(n);
    if lo >= hi {
        return false;
    }
    let (lo, hi) = (lo as usize, hi as usize);
    if hi - lo > 32 {
        return limbs::with_work(|w| {
            let s = &mut w.a;
            s.clear();
            s.resize(hi - lo, 0);
            window(y, o + 64 * lo as i128, s);
            if sub {
                nat::sub_assign(&mut t[lo..], s)
            } else {
                nat::add_assign(&mut t[lo..], s)
            }
        });
    }
    // The shifted limbs in a buffer of their own, then one pass of the sum.
    let mut v = [0u64; 33];
    window(y, o + 64 * lo as i128, &mut v[..hi - lo]);
    let mut c = false;
    if sub {
        for (ti, &vi) in t[lo..hi].iter_mut().zip(&v[..hi - lo]) {
            let (r, b1) = ti.overflowing_sub(vi);
            let (r, b2) = r.overflowing_sub(u64::from(c));
            *ti = r;
            c = b1 | b2;
        }
    } else {
        for (ti, &vi) in t[lo..hi].iter_mut().zip(&v[..hi - lo]) {
            let (r, c1) = ti.overflowing_add(vi);
            let (r, c2) = r.overflowing_add(u64::from(c));
            *ti = r;
            c = c1 | c2;
        }
    }
    if !c || hi == t.len() {
        return c;
    }
    if sub {
        nat::sub_assign(&mut t[hi..], &[1])
    } else {
        nat::add_assign(&mut t[hi..], &[1])
    }
}

/// `x·y` of the sign `negative`, rounded to `prec` bits in the mode `round`, into `out`.
///
/// Operands of at most four limbs for a result of three or more are multiplied in a word first
/// ([`small::mul_short`]). Otherwise:
///
/// Operands longer than `n + 2` limbs, `n = ceil(prec / 64)`, are cut to their top `n + 2`: the
/// product of the cut operands, `P`, is below the exact one by less than `B^l` units of its lowest
/// limb (`B = 2^64`) for one cut operand, `l` the other's length, and `2B^l + 1` for two, `l` their
/// length: so the exact product's limbs from `l` up are those of `P`, or one more (two more for
/// two cut operands), and [`decided`] tells when that is enough. Otherwise, and for operands not
/// cut, the exact product, by the crate's multiplication (Karatsuba, Toom-Cook, transforms).
pub(crate) fn mul(
    negative: bool,
    x: Op<'_>,
    y: Op<'_>,
    prec: u64,
    round: Round,
    w: &mut Work,
    out: Vec<u64>,
) -> (Kind, Ordering) {
    let same = core::ptr::eq(x.m, y.m);
    let mut out = out;
    if prec > 128
        && x.m.len() <= 4
        && y.m.len() <= 4
        && word_exp(x.exp)
        && word_exp(y.exp)
        && let Some(r) = small::with_vec(prec, &mut out, |o| {
            small::mul_short(
                negative,
                x.m,
                x.exp as i64,
                y.m,
                y.exp as i64,
                prec,
                round,
                o,
            )
        })
    {
        return r;
    }
    let (x, y) = (x.strip(), y.strip());
    let n = prec.div_ceil(64) as usize;
    let keep = n + 2;
    let cut = |m: &'_ [u64]| -> bool { m.len() > keep };
    let (ta, tb) = (cut(x.m), cut(y.m));
    let xt = if ta { &x.m[x.m.len() - keep..] } else { x.m };
    let yt = if tb { &y.m[y.m.len() - keep..] } else { y.m };
    let Work { a: p, scratch, .. } = w;
    let product = |p: &mut Vec<u64>, s: &mut Vec<u64>, a: &[u64], b: &[u64]| {
        if same {
            nat::sqr_into(p, a, s);
        } else {
            nat::mul_into(p, a, b, s);
        }
    };
    product(p, scratch, xt, yt);
    let lsb = (x.exp - 64 * xt.len() as i128) + (y.exp - 64 * yt.len() as i128);
    if ta || tb {
        let (l, hi) = match (ta, tb) {
            (true, true) => (keep, 2),
            (true, false) => (yt.len(), 1),
            _ => (xt.len(), 1),
        };
        let q = &p[l..];
        if decided(q, 0, hi, prec) {
            return round_into(negative, q, lsb + 64 * l as i128, true, prec, round, out);
        }
        product(p, scratch, x.m, y.m);
        let lsb = x.lsb() + y.lsb();
        limbs::trim(p);
        return round_into(negative, p, lsb, false, prec, round, out);
    }
    limbs::trim(p);
    round_into(negative, p, lsb, false, prec, round, out)
}

/// The longest divisor, in limbs, whose quotient [`div`] computes alone, with a shrinking divisor
/// for its last limbs (about half the products of the schoolbook method), before it takes the
/// remainder too: measured on an Apple M4 Max against the division with the remainder (divide and
/// conquer from 40 limbs), equal-length operands of 20,000 bits took 0.85 of its time, of 30,000
/// 0.95, of 38,000 1.04.
pub(crate) const SHORT_DIV: usize = 480;

/// Whether an exponent is one of a float's, which the kernels in words take as an `i64`: the
/// magnitudes that the other modules pass may have exponents of `i128`s beyond, and within the
/// range the sums and differences of two exponents are still `i64`s.
#[inline(always)]
fn word_exp(e: i128) -> bool {
    (i128::from(BigFloat::EXP_MIN)..=i128::from(BigFloat::EXP_MAX)).contains(&e)
}

/// `x/y` of the sign `negative` for a result of at most 62 bits, from the top limbs of the
/// operands: each lies between its top limb `T` and `T` plus one unit of it, strictly when its
/// other limbs are not all zero, so the quotient lies strictly between `X/(Y + 1)` and
/// `(X + 1)/Y` (each end one only when that operand has a tail), whose roundings by
/// [`small::div_fit1`] settle its rounding when [`small::between`] says so. `None` for the
/// general path.
#[inline]
fn div_short(
    negative: bool,
    x: Op<'_>,
    y: Op<'_>,
    prec: u64,
    round: Round,
) -> Option<small::Fit<u64>> {
    use small::Term;
    let (lx, ly) = (x.m.len(), y.m.len());
    let tx = x.m[..lx - 1].iter().any(|&v| v != 0);
    let ty = y.m[..ly - 1].iter().any(|&v| v != 0);
    let term = |o: Op<'_>| Term {
        negative: false,
        exp: o.exp as i64,
        m: o.m[o.m.len() - 1],
    };
    let (a, b) = (term(x), term(y));
    let sh = (64 - prec) as u32;
    let q =
        |a: Term<u64>, b: Term<u64>| small::div_fit1(negative, a.exp, a.m, b.exp, b.m, sh, round);
    if !(tx || ty) {
        return q(a, b);
    }
    // The ends of the magnitude, then in the order of the signed values.
    let small_end = q(a, if ty { b.bump() } else { b })?;
    let large_end = q(if tx { a.bump() } else { a }, b)?;
    let (lo, hi) = if negative {
        (large_end, small_end)
    } else {
        (small_end, large_end)
    };
    small::between(lo, hi)
}

/// `x/y` of the sign `negative`, rounded to `prec` bits in the mode `round`, into `out`.
///
/// With `x` taken as `0.x·2^-c`, `c = 1` when `0.x ≥ 0.y`, the quotient `F = 0.x·2^-c / 0.y` lies
/// in `[1/2, 1)`, and `Q = floor(F·B^q)` has `64q` bits for `q = n + 1` limbs. It comes from:
///
/// - for a divisor of more than `q` limbs, the top `2q` limbs `N'` of the dividend and the top `q`
///   limbs `D'` of the divisor: `Q' = floor(N'/D')` and `Q' - 2 < F·B^q < Q' + 1` (the bound of
///   the crate's quotient from the top limbs, `div_q_top`), so [`decided`] tells whether `Q'`
///   rounds as the quotient;
/// - otherwise, for a divisor of 3 to [`SHORT_DIV`] limbs, the quotient alone of the dividend's
///   top `d + q` limbs, `d` the divisor's length, exactly, its last limbs from a shrinking divisor:
///   when its bits below the round bit are not all zero, the remainder cannot change the rounding
///   and is a sticky bit;
/// - otherwise, or when that leaves it undecided, the exact division of those limbs with its
///   remainder (and the dividend's limbs below those, if any) for the sticky bit: Newton's method
///   for the longest.
///
/// A result of at most 62 bits from long operands is first tried from their top limbs, by
/// [`div_short`].
pub(crate) fn div(
    negative: bool,
    x: Op<'_>,
    y: Op<'_>,
    prec: u64,
    round: Round,
    w: &mut Work,
    out: Vec<u64>,
) -> (Kind, Ordering) {
    if prec <= 62
        && word_exp(x.exp)
        && word_exp(y.exp)
        && let Some(r) = div_short(negative, x, y, prec, round)
    {
        let mut out = out;
        out.clear();
        out.push(r.m);
        let kind = Kind::Finite {
            negative: r.negative,
            exp: r.exp,
            mant: out,
        };
        return (kind, r.o);
    }
    let (x, y) = (x.strip(), y.strip());
    let n = prec.div_ceil(64) as usize;
    let qn = n + 1;
    let c = u32::from(limbs::cmp_top(x.m, y.m) != Ordering::Less);
    let lsb = x.exp - y.exp + i128::from(c) - 64 * qn as i128;
    let dn = y.m.len();
    let Work {
        a: num,
        b: q,
        scratch,
        ..
    } = w;
    q.clear();
    q.resize(qn, 0);
    if dn > qn {
        num.clear();
        num.resize(2 * qn, 0);
        limbs::top_limbs(x.m, c, num);
        let high = nat::div_qr(q, num, &y.m[dn - qn..], scratch);
        if !high && decided(q, 2, 0, prec) {
            return round_into(negative, q, lsb, true, prec, round, out);
        }
    }
    let nn = dn + qn;
    num.clear();
    num.resize(nn, 0);
    let mut tail = limbs::top_limbs(x.m, c, num);
    if (3..=SHORT_DIV).contains(&dn) {
        // The quotient alone, exactly: when its bits below the round bit are not all zero, the
        // remainder cannot change the rounding, and is a sticky bit.
        if let Some(high) = nat::div_q(q, num, y.m, scratch) {
            debug_assert!(!high);
            if decided(q, 0, 0, prec) {
                return round_into(negative, q, lsb, true, prec, round, out);
            }
        }
        tail = limbs::top_limbs(x.m, c, num);
    }
    let high = nat::div_qr(q, num, y.m, scratch);
    debug_assert!(!high);
    let sticky = tail || num[..dn].iter().any(|&v| v != 0);
    round_into(negative, q, lsb, sticky, prec, round, out)
}

/// Up to this many limbs, a square root with fewer than 16 bits below its precision in its own
/// limbs takes the root of those limbs with its remainder rather than a limb more; beyond, the
/// remainder costs more than the limb (Apple M4 Max: 10^6 bits 1.76 ms against 1.55).
const SHORT_ROOT: usize = 64;

/// The square root of `x`, positive, rounded to `prec` bits in the mode `round`, into `out`.
///
/// With `c` the parity of the exponent, the top `2q` limbs `R` of `0.x·2^-c` (`q` the limbs of
/// `prec + 16` bits, so that 16 bits or more lie below the round bit, and 2 at least; beyond
/// [`SHORT_ROOT`] limbs, one more than the precision's) have a root
/// `S` of `64q` bits, by the crate's Karatsuba square root; the floor of the root of the truncated
/// `R` is that of the exact one, as `s^2 ≤ R` holds for an integer `s` exactly when it holds for
/// `floor(R)`. The remainder, and the limbs of `x` below `R`, matter only when the bits of `S`
/// below its round bit are all zero (a root halfway between two numbers of the precision is one
/// of them): then it is computed with the remainder. When fewer than 16 bits lie below the
/// precision in its own `n` limbs (up to [`SHORT_ROOT`]), the root of `n` limbs is taken with its
/// remainder instead, which decides exactly; for `64n` bits the round bit lies below the root,
/// and `√R ≥ S + 1/2` exactly when the remainder is above `S` (`R = S^2 + S + 1/4` is not an
/// integer), a tail leaving only the remainder `S` itself to the root of a limb more.
pub(crate) fn sqrt(
    x: Op<'_>,
    prec: u64,
    round: Round,
    w: &mut Work,
    out: Vec<u64>,
) -> (Kind, Ordering) {
    let x = x.strip();
    let c = (x.exp & 1) as u32;
    let Work {
        a: r,
        b: s,
        c: rem,
        scratch,
    } = w;
    let q0 = prec.div_ceil(64) as usize;
    if 64 * q0 as u64 - prec < 16 && (2..=SHORT_ROOT).contains(&q0) {
        // Fewer than 16 bits below the precision in `q0` limbs: the root of `q0` limbs and its
        // remainder decide, as MPFR's `mpfr_sqrt` takes them, rather than a limb more.
        r.clear();
        r.resize(2 * q0, 0);
        let tail = limbs::top_limbs(x.m, c, r);
        nat::sqrt_rem(s, Some(rem), r, scratch);
        let lsb = (x.exp + i128::from(c)) / 2 - 64 * q0 as i128;
        let sticky = tail || !rem.is_empty();
        if prec < 64 * q0 as u64 {
            return round_into(false, s, lsb, sticky, prec, round, out);
        }
        // The round bit below `S`: `√R ≥ S + 1/2` exactly when the remainder is above `S` (never
        // equal), but with a tail when it equals `S`, which a limb more decides.
        let above = limbs::cmp(rem, s);
        if !(above == Ordering::Equal && tail) {
            // `2S` or `2S + 1`, the half bit in place of the round bit, with the sticky bit.
            let mut carry = u64::from(above == Ordering::Greater);
            for v in s.iter_mut() {
                let next = *v >> 63;
                *v = *v << 1 | carry;
                carry = next;
            }
            s.push(carry);
            return round_into(false, s, lsb - 1, sticky, prec, round, out);
        }
    }
    // A limb more beyond [`SHORT_ROOT`], where the lengths of the integer root's halves matter
    // more than a limb (10^4 bits: 4.48 µs against 4.62 for the 157 limbs of `prec + 16` bits).
    let qn = if q0 > SHORT_ROOT {
        q0 + 1
    } else {
        (prec + 16).div_ceil(64).max(2) as usize
    };
    let lsb = (x.exp + i128::from(c)) / 2 - 64 * qn as i128;
    r.clear();
    r.resize(2 * qn, 0);
    let tail = limbs::top_limbs(x.m, c, r);
    nat::sqrt_rem(s, None, r, scratch);
    if decided(s, 0, 0, prec) {
        return round_into(false, s, lsb, true, prec, round, out);
    }
    nat::sqrt_rem(s, Some(rem), r, scratch);
    let sticky = tail || !rem.is_empty();
    round_into(false, s, lsb, sticky, prec, round, out)
}

/// `x·y + z` rounded once, the product of the sign `negative` exact: aligned at its top bit, then
/// added as an operand of [`add`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn fma(
    negative: bool,
    x: Op<'_>,
    y: Op<'_>,
    z: Op<'_>,
    prec: u64,
    round: Round,
    w: &mut Work,
    out: Vec<u64>,
) -> (Kind, Ordering) {
    let same = core::ptr::eq(x.m, y.m);
    let (x, y) = (x.strip(), y.strip());
    let mut p = core::mem::take(&mut w.c);
    if same {
        nat::sqr_into(&mut p, x.m, &mut w.scratch);
    } else {
        nat::mul_into(&mut p, x.m, y.m, &mut w.scratch);
    }
    let mut exp = x.exp + y.exp;
    if p[p.len() - 1] >> 63 == 0 {
        // The top bit one below: shifted into place, which carries nothing out.
        let mut carry = 0u64;
        for v in p.iter_mut() {
            let next = *v >> 63;
            *v = *v << 1 | carry;
            carry = next;
        }
        exp -= 1;
    }
    let r = add(
        Op {
            negative,
            m: &p,
            exp,
        },
        z,
        prec,
        round,
        out,
    );
    w.c = p;
    r
}

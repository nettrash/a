//! The partition numbers `p(n)`: up to 416 from a table, beyond by Hardy, Ramanujan and
//! Rademacher's series, as Johansson does it ("Efficient implementation of the
//! Hardy–Ramanujan–Rademacher formula", 2012).
//!
//! `p(n) = Σ_(k=1)^N T_k + R(n, N)`, `T_k = √(3/k) (4 / (24n - 1)) A_k(n) U(C/k)` with
//! `C = (π/6) √(24n - 1)` and `U(x) = cosh(x) - sinh(x)/x`, the remainder bounded by Rademacher's
//! `|R(n, N)| < 44π^2 / (225√3) N^-1/2 + (π√2 / 75) (N / (n - 1))^1/2 sinh((π/N) √(2n/3))`. `N` is
//! the least number of terms for which that bound is at most 1/4. Each `A_k(n)` is a product of
//! cosines ([`expsum`](mod@expsum)); each term is computed in double precision when that is
//! accurate enough ([`double`]), and otherwise at a precision that falls as `k` grows, up to 1248
//! bits in fixed point on limbs on the stack ([`mid`]) and beyond in floating point on numbers
//! ([`real`]), each with a bound of its error. The terms are summed in fixed point, and the sum
//! rounded to the nearest integer once the bounds of all errors add up to less than 1/2.
//!
//! The first term's `e^C`, on numbers from `n` of about 10^5, is reduced by `ln 2` first; the
//! other terms on numbers take `e^(C/k)`, `k = 2^j m` with `m` odd, as the `m`-th root of `e^C` by
//! Newton's method and `j` square roots, and their cosines of denominators `2^s 3^t q`, `q` 1 or 5,
//! from closed forms and the half- and triple-angle formulas ([`real::cos_pi_algebraic`]), as do
//! the terms on 9 limbs and more. The roots `√(3 sp / (k sq))` take four values, kept for the call.
//!
//! Measured on an Apple M4 Max, single-threaded, on a quiet machine, in cycles against FLINT 3.6's
//! `partitions_fmpz_ui` (which keeps `π`, logarithms and tables between calls): `p(500)` 11,000
//! against 16,300, `p(10^4)` 42,900 against 103,000, `p(10^6)` 601,000 against 873,000, `p(10^7)`
//! 2.93 M against 3.23 M; `p(10^9)` 20.2 ms against 16.6.

mod double;
mod expsum;
mod mid;
mod real;

use super::super::root::small_number;
use super::super::{ByteOrder, Number};
use double::U as UNIT;
use expsum::{Sieve, Sum, expsum};
use mid::Mid;
use real::{
    Float, cos_pi, cos_pi_algebraic, cos_pi_newton, exp, inverse_root, log2_above, pow2, shift,
    sqrt,
};
pub(crate) use real::{LN2_TABLE, PI_FULL, PI_TABLE, pi};

/// `p(n)` for `n` up to 416, the last below `2^64`, by Euler's pentagonal recurrence
/// `p(n) = Σ_(k>=1) (-1)^(k+1) (p(n - k(3k-1)/2) + p(n - k(3k+1)/2))`.
const TABLE: [u64; 417] = {
    let mut t = [0u64; 417];
    t[0] = 1;
    let mut n = 1;
    while n < 417 {
        let mut s: i128 = 0;
        let mut k = 1;
        loop {
            let g = k * (3 * k - 1) / 2;
            if g > n {
                break;
            }
            let sign: i128 = if k % 2 == 1 { 1 } else { -1 };
            s += sign * t[n - g] as i128;
            let h = k * (3 * k + 1) / 2;
            if h <= n {
                s += sign * t[n - h] as i128;
            }
            k += 1;
        }
        t[n] = s as u64;
        n += 1;
    }
    t
};

/// `p(n)`, in the byte order `order`.
pub(in super::super) fn partitions(n: u64, order: ByteOrder) -> Number {
    if n < TABLE.len() as u64 {
        return small_number(order, u128::from(TABLE[n as usize]), false);
    }
    let le = hrr(n, 0);
    let mut x = le;
    if order != ByteOrder::LittleEndian {
        x.value.reverse();
        x.byte_order = order;
    }
    x
}

/// The bits of the fixed-point sum below the point.
const FRACTION: u64 = 40;

/// The precision up to which a term is computed on limbs on the stack ([`mid`]) rather than on
/// [`Number`]s ([`real`]).
const MID_BITS: u64 = 64 * (mid::MAX as u64 - 1) - 160;

/// `44π^2 / (225√3)`, `π√2 / 75` and `π √(2/3)`, each the least double above.
const HRR_A: f64 = f64::from_bits(0x3FF1_D43F_7658_55DC);
const HRR_B: f64 = f64::from_bits(0x3FAE_5480_2DD9_67EC);
const HRR_C: f64 = f64::from_bits(0x4004_8552_F880_91A9);

/// `x` made larger by a few units of its last place, for upper bounds computed in doubles.
fn up(x: f64) -> f64 {
    x * (1.0 + 8.0 * UNIT)
}

/// An upper bound of Rademacher's bound of the remainder after `N` terms for `n`.
fn remainder(n: f64, terms: f64) -> f64 {
    let first = up(HRR_A / terms.sqrt());
    let y = up(up(HRR_C * up(n.sqrt())) / terms);
    if y > 700.0 {
        return f64::INFINITY;
    }
    // `sinh(y) < e^y / 2`; the exponential within `16 u`.
    let sinh = up(up(double::exp(y) * (1.0 + 16.0 * UNIT)) / 2.0);
    let root = up(up(terms / (n * (1.0 - UNIT) - 1.0)).sqrt());
    up(first + up(up(HRR_B * root) * sinh))
}

/// The least number of terms whose remainder's bound is at most 1/4.
fn needed_terms(n: f64) -> u64 {
    // The bound falls with `N` over the range searched, from the first `N` where its sinh is
    // below the rest: bisect for the least `N` at most 1/4.
    let (mut lo, mut hi) = (1u64, 2u64);
    while remainder(n, hi as f64) > 0.25 {
        hi *= 2;
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if remainder(n, mid as f64) <= 0.25 {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// `p(n)` by the series, little-endian, with `extra` bits of precision more than the schedule, which
/// a failed check of the bound of the error raises.
fn hrr(n: u64, extra: u64) -> Number {
    let nf = n as f64;
    let terms = needed_terms(nf);
    let r = remainder(nf, terms as f64);
    // Each term to within `2^-g`, so that all of them are within 1/16.
    let g = (terms as f64).log2().ceil() as u64 + 4 + extra;
    let n24 = 24 * u128::from(n) - 1;
    let n24f = n24 as f64;
    let n24f_err = if n24 < 1 << 53 { 0.0 } else { UNIT };
    let cf = core::f64::consts::PI / 6.0 * n24f.sqrt();
    // The precision of the first term, and of `C` and `π` with room to spare.
    let log2_n24 = n24f.log2();
    let lt1 = log2_bound(&Sum::one(), 1, cf, log2_n24);
    let w_first = (lt1 + g as f64 + 16.0).ceil().max(64.0) as u64;
    // Room for the fixed point of the terms on the stack, up to 128 bits more than their precision.
    let fc = w_first + 176;
    let root_bits = (n24f.log2() / 2.0).ceil() as u64 + 2;
    let fp = fc + root_bits + 16;
    let pi_f = pi(fp);
    // `C = π √(24n - 1) / 6` within a unit of `2^-fc`: `π` within 4 units of `2^-fp`, the root
    // within one of `2^-fp`.
    let root =
        (Number::new_from_u128(n24, ByteOrder::LittleEndian) << real::shift_count(2 * fp)).isqrt();
    let c_fix = &(&(&pi_f * &root) >> real::shift_count(2 * fp - fc)) / 6u32;
    let c_err = 2.0;
    let c_f64 = Float::of_fixed(c_fix.clone(), fc).to_f64();
    // `e^C` to the first term's precision, of which the others' are roots, when the first term is
    // computed on `Number`s.
    let first = if w_first > MID_BITS || n24 >= 1 << 64 {
        Some(exp(&c_fix, fc, c_err, w_first))
    } else {
        None
    };
    // `1 / C`, so that `1 / x = k / C` takes no division: within 4 units of `2^-fc` relatively.
    let one = Float {
        m: Number::new_from_u8(1, ByteOrder::LittleEndian),
        e: 0,
    };
    let inv_c = one.div(&Float::of_fixed(c_fix.clone(), fc).cut(fc), fc);
    let inv_c_fix = inv_c.to_fixed(fc);
    let mut sieve = Sieve::new(terms, n);
    // `C` and `1/C` in the fixed point of each length, once each.
    let mut constants: Constants = Vec::new();
    let mut roots: Roots = Vec::new();
    let mut float_roots: FloatRoots = Vec::new();
    let mut factors = Vec::with_capacity(16);
    let mut big = Number::new_from_u8(0, ByteOrder::LittleEndian);
    let mut small: i128 = 0;
    let mut err = 0.0f64;
    for k in 1..=terms {
        sieve.factors(k, &mut factors);
        let sum = expsum(k, &factors, &mut sieve);
        if sum.prefactor == 0 {
            continue;
        }
        let x = c_f64 / k as f64;
        let lt = log2_bound(&sum, k, cf, log2_n24);
        // The double precision's bound, about: `(60 + 13 cosines + 2x) u` relatively.
        let low = lt + log2_above(60.0 + 13.0 * sum.len as f64 + 2.0 * x) - 53.0;
        if low <= -(g as f64) - 2.0 {
            let (v, e) = double::term(&sum, k, n24f, n24f_err, c_f64, 1.01 * UNIT);
            // `v 2^FRACTION`, exactly, rounded to an integer: half a unit of `2^-FRACTION`.
            small += (v * pow2(FRACTION as i64)).round() as i128;
            err += e + pow2(-(FRACTION as i64) - 1);
        } else if (lt + g as f64 + 16.0) <= MID_BITS as f64 && n24 < 1 << 64 {
            let w = (lt + g as f64 + 16.0).ceil().max(64.0) as u64;
            // Room for the bits that the fixed point loses: two for each doubling of the
            // exponential and of the cosines (about `2.2 (w + 64)^(1/3)` in all), the factor
            // `4 / (24n - 1)`, and 24 to spare.
            let room = 4.4 * ((w + 64) as f64).cbrt() + log2_n24 + 24.0;
            let l = (w + room.ceil() as u64).div_ceil(64) as usize + 1;
            let fixed = (&c_fix, fc, c_err, &inv_c_fix);
            let n24 = n24 as u64;
            let cs = (&mut constants, &mut roots);
            let t = match l {
                2 => mid_term(Mid::<2> { l }, cs, &sum, k, n24, fixed),
                3 => mid_term(Mid::<3> { l }, cs, &sum, k, n24, fixed),
                4 => mid_term(Mid::<4> { l }, cs, &sum, k, n24, fixed),
                5 => mid_term(Mid::<5> { l }, cs, &sum, k, n24, fixed),
                6 => mid_term(Mid::<6> { l }, cs, &sum, k, n24, fixed),
                7 => mid_term(Mid::<7> { l }, cs, &sum, k, n24, fixed),
                8 => mid_term(Mid::<8> { l }, cs, &sum, k, n24, fixed),
                9..=16 => mid_term(Mid::<16> { l }, cs, &sum, k, n24, fixed),
                _ => mid_term(Mid::<{ mid::MAX }> { l }, cs, &sum, k, n24, fixed),
            };
            if let Some((t, e)) = t {
                big += t;
                err += e;
            }
        } else {
            let w = (lt + g as f64 + 16.0).ceil().max(64.0) as u64;
            let first = first.as_ref().expect("e^C for the terms on numbers");
            let (t, e) = high_term(
                &mut float_roots,
                &sum,
                k,
                n24,
                (&c_fix, fc, c_err, &inv_c),
                w,
                (&pi_f, fp),
                (first, w_first),
            );
            big += t;
            err += e + pow2(-(FRACTION as i64));
        }
    }
    big += Number::new_from_i128(small, ByteOrder::LittleEndian);
    let total = up(up(err) + r);
    if total >= 0.49 {
        // Not reached with the schedule's margins (the bounds of the terms' errors are far below
        // their share); more precision if it ever is.
        assert!(
            extra < 64,
            "the bound of the error of the partition series did not fall"
        );
        return hrr(n, extra + 16);
    }
    // The nearest integer.
    &(&big + &(Number::new_from_u8(1, ByteOrder::LittleEndian) << real::shift_count(FRACTION - 1)))
        >> real::shift_count(FRACTION)
}

/// An upper bound of `log2 |T_k|` with `U(x) < e^x`: `log2 |prefactor| + log2 √(3 sp / (k sq)) + 2
/// - log2(24n - 1) + x log2(e)`, with a bit to spare for the doubles' roundings.
fn log2_bound(sum: &Sum, k: u64, cf: f64, log2_n24: f64) -> f64 {
    let x = cf / k as f64;
    // Upper bounds of the logarithms, within 0.09 of them.
    log2_above(sum.prefactor.unsigned_abs() as f64)
        + 0.5 * log2_above((3 * sum.sp) as f64 / k as f64 / sum.sq as f64)
        + 2.0
        - log2_n24
        + x * core::f64::consts::LOG2_E
        + 1.0
}

/// `C` and `1/C` in the fixed point of [`Mid`] for the lengths of its values, as they are first
/// needed: few, as the terms' precisions fall with `k`.
type Constants = Vec<(usize, mid::Fx<{ mid::MAX }>, mid::Fx<{ mid::MAX }>)>;

/// The square roots `√(num / den)` of the terms on limbs, by the reduced fraction, with the number
/// of limbs of their fixed point, as they are first needed: `3 sp / (k sq)` takes few values (3,
/// 3/2, 1 and 1/2, by the factors of 2 and 3 of `k`), the terms' precisions fall with `k`, and a root
/// to more limbs is cut to fewer.
type Roots = Vec<((u128, u128), usize, mid::Fx<{ mid::MAX }>)>;

/// `√(num / den)` in the fixed point of at least `l` limbs, within a unit of its last place, and
/// the number of its limbs: from `roots`, or computed to `l` limbs and kept there.
fn root_of(roots: &mut Roots, num: u128, den: u128, l: usize) -> (usize, mid::Fx<{ mid::MAX }>) {
    let g = {
        let (mut x, mut y) = (num, den);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    let key = (num / g, den / g);
    let at = roots.iter().position(|(k, _, _)| *k == key);
    if let Some(i) = at
        && roots[i].1 >= l
    {
        return (roots[i].1, roots[i].2);
    }
    // `√(num den 2^(2f)) / den`, rounded down twice: within a unit.
    let m = Mid::<{ mid::MAX }> { l };
    let f = m.f();
    let square =
        Number::new_from_u128(key.0 * key.1, ByteOrder::LittleEndian) << real::shift_count(2 * f);
    let root = &square.isqrt() / &Number::new_from_u128(key.1, ByteOrder::LittleEndian);
    let r = m.of_number(&root, f);
    match at {
        Some(i) => roots[i] = (key, l, r),
        None => roots.push((key, l, r)),
    }
    (l, r)
}

/// The square roots `√(num / den)` of the terms on numbers, by the reduced fraction, with their
/// precision, as they are first needed (see [`Roots`]).
type FloatRoots = Vec<((u128, u128), u64, Float)>;

/// `√(num / den)` as a floating number of at least `w` bits within 6 units of its last place, and
/// its precision: from `roots`, or computed to `w` bits (the root within 2 units, the quotient
/// within 4) and kept there.
fn float_root_of(roots: &mut FloatRoots, num: u128, den: u128, w: u64) -> (u64, Float) {
    let g = {
        let (mut x, mut y) = (num, den);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    let key = (num / g, den / g);
    let at = roots.iter().position(|(k, _, _)| *k == key);
    if let Some(i) = at
        && roots[i].1 >= w
    {
        return (roots[i].1, roots[i].2.clone());
    }
    let root = sqrt(
        &Float {
            m: Number::new_from_u128(key.0 * key.1, ByteOrder::LittleEndian),
            e: 0,
        },
        w,
    );
    let r = root.div(
        &Float {
            m: Number::new_from_u128(key.1, ByteOrder::LittleEndian),
            e: 0,
        },
        w,
    );
    match at {
        Some(i) => roots[i] = (key, w, r.clone()),
        None => roots.push((key, w, r.clone())),
    }
    (w, r)
}

/// The term `T_k` on limbs on the stack by [`Mid`], in fixed point with [`FRACTION`] bits, and a
/// bound of its absolute error, or `None` when it is zero; `C` and `1/C` in the arithmetic's fixed
/// point are kept in `constants`, once for each length.
fn mid_term<const CAP: usize>(
    mid: Mid<CAP>,
    (constants, roots): (&mut Constants, &mut Roots),
    sum: &Sum,
    k: u64,
    n24: u64,
    (c_fix, fc, c_err, inv_c_fix): (&Number, u64, f64, &Number),
) -> Option<(Number, f64)> {
    let f = mid.f();
    let i = match constants.iter().position(|e| e.0 == mid.l) {
        Some(i) => i,
        None => {
            let m = Mid::<{ mid::MAX }> { l: mid.l };
            constants.push((mid.l, m.of_number(c_fix, fc), m.of_number(inv_c_fix, fc)));
            constants.len() - 1
        }
    };
    let (_, c, ic) = &constants[i];
    // The first `CAP` limbs: those above `l` are zero.
    let c: mid::Fx<CAP> = core::array::from_fn(|i| c[i]);
    let ic: mid::Fx<CAP> = core::array::from_fn(|i| ic[i]);
    let ec = c_err * pow2(f as i64 - fc as i64) + 1.0;
    // `1/C` within 4 units of `2^-fc` relatively, `1/C < 1`, and the cut.
    let eic = 4.0 * pow2(f as i64 - fc as i64) + 2.0;
    // `√(3 sp / (k sq))` within two units: from the cache, cut to `l` limbs.
    let (rl, root) = root_of(
        roots,
        3 * u128::from(sum.sp),
        u128::from(k) * u128::from(sum.sq),
        mid.l,
    );
    let root: mid::Fx<CAP> =
        core::array::from_fn(|i| if i < mid.l { root[rl - mid.l + i] } else { 0 });
    // The cosines with closed forms and half-angle chains, from numbers, from 9 limbs: below, the
    // series in registers took less time than the allocations of the numbers' square roots.
    let mut cosines = [None; 16];
    for (i, &(a, b)) in sum.cos[..sum.len].iter().enumerate() {
        if mid.l <= 8 {
            break;
        }
        match real::cos_pi_fixed(a, b, f, false) {
            Some(Some((v, negative, e))) => {
                cosines[i] = Some((mid.of_number(&v, f), negative, e + 1.0))
            }
            Some(None) => return None,
            None => {}
        }
    }
    let (v, negative, q, e) = mid.term(
        sum,
        k,
        n24,
        &c,
        ec,
        &ic,
        eic,
        (&root, 2.0),
        &cosines[..sum.len],
    )?;
    Some((
        mid.to_number(&v, negative, q + FRACTION as i64),
        e * pow2(q - f as i64) + pow2(-(FRACTION as i64)),
    ))
}

/// The term `T_k` in multiple precision, to `w` bits: in fixed point with [`FRACTION`] bits, and a
/// bound of its absolute error.
#[allow(clippy::too_many_arguments)]
fn high_term(
    float_roots: &mut FloatRoots,
    sum: &Sum,
    k: u64,
    n24: u128,
    (c_fix, fc, c_err, inv_c): (&Number, u64, f64, &Float),
    w: u64,
    (pi_f, fp): (&Number, u64),
    first: (&(Float, f64), u64),
) -> (Number, f64) {
    const LE: ByteOrder = ByteOrder::LittleEndian;
    let one = Float {
        m: Number::new_from_u8(1, LE),
        e: 0,
    };
    // `x = C / k` within `c_err + 1` units of `2^-fc`.
    let x_fix = c_fix / k;
    let ex = c_err + 1.0;
    // `e^x`, and `e^-x` when it is computed first.
    let ((e1, r1), w1) = first;
    // `k = 2^j m`, `m` odd: `e^(C/k)` is the `2^j`-th root of `e^(C/m)`, by `j` square roots, each of
    // which halves the relative error and loses two units.
    let (twos, odd) = (k.trailing_zeros(), k >> k.trailing_zeros());
    let roots = |mut y: Float, mut r: f64| {
        for _ in 0..twos {
            y = sqrt(&y, w);
            r = r / 2.0 + 2.0;
        }
        (y, r)
    };
    let (y, ry, z) = if k == 1 {
        (e1.clone(), *r1, None)
    } else if odd == 1 {
        // The cut of `e^C` to `w` bits: its error in units of `2^-w`, and one.
        let (y, r) = roots(e1.clone().cut(w), *r1 * pow2(w as i64 - w1 as i64) + 1.0);
        (y, r, None)
    } else if root_is_cheaper(odd, w) {
        let (z, rz) = inverse_root(e1, *r1, w1, odd, w);
        let (y, r) = roots(one.div(&z, w), rz + 4.0);
        let z = (twos == 0).then_some((z, rz));
        (y, r, z)
    } else {
        let (y, ry) = exp(&x_fix, fc, ex, w);
        (y, ry, None)
    };
    let xf = Float::of_fixed(x_fix, fc).cut(w);
    // `x` within `2` units of `2^-w` relatively (the cut; the fixed point's error is far below).
    let rx = 3.0;
    // `x > 2`: `x - 1` within twice `x`'s error, `x + 1` within it, and a cut each.
    let xm1 = sub(&xf, &one, w);
    let xp1 = xf.add(&one, w);
    let a = y.mul(&xm1, w);
    let mut ra = ry + 2.0 * rx + 4.0;
    let s = if 2.0 * xf.log2().exp2() * core::f64::consts::LOG2_E > w as f64 + 4.0 {
        // `e^-x (x + 1) < 2^-(w+4) e^x (x - 1)`: dropped, a unit.
        ra += 1.0;
        a
    } else {
        let (z, rz) = z.unwrap_or_else(|| (one.div(&y, w), ry + 4.0));
        let b = z.mul(&xp1, w);
        ra = ra.max(rz + rx + 4.0);
        a.add(&b, w)
    };
    let rs = ra + 2.0;
    // `U = s / (2x) = s (k / C) / 2`, `1 / C` within 4 units of `2^-(w + 8)` (a quotient and a cut,
    // `C`'s own error far below), then a cut.
    let inv_x = Float {
        m: &inv_c.m * k,
        e: inv_c.e,
    }
    .cut(w);
    let mut uv = s.mul(&inv_x, w);
    uv.e -= 1;
    let ru = rs + 1.0 + 2.0 + 2.0;
    // `√(3 sp / (k sq)) = √(3 sp k sq) / (k sq)`, within 6 units of its precision `wr >= w`, from
    // the cache; cut to `w` bits, a unit more.
    let (wr, root) = float_root_of(
        float_roots,
        3 * u128::from(sum.sp),
        u128::from(k) * u128::from(sum.sq),
        w,
    );
    let mut t = root.cut(w);
    let mut rel = 6.0 * pow2(w as i64 - wr as i64) + 1.0;
    // Times the prefactor and 4, exactly, over `24n - 1`.
    t.m = &t.m * &Number::new_from_u64(sum.prefactor.unsigned_abs(), LE);
    t.e += 2;
    t = t.div(
        &Float {
            m: Number::new_from_u128(n24, LE),
            e: 0,
        },
        w,
    );
    rel += 4.0 + 2.0;
    let mut negative = sum.prefactor < 0;
    for &(a, b) in &sum.cos[..sum.len] {
        let c = if let Some(c) = cos_pi_algebraic(a, b, w) {
            c
        } else if newton_is_cheaper(a, b, w) {
            cos_pi_newton(a, b, w).unwrap_or_else(|()| cos_pi(a, b, w, pi_f, fp))
        } else {
            cos_pi(a, b, w, pi_f, fp)
        };
        let Some((c, neg, rc)) = c else {
            return (Number::new_from_u8(0, LE), 0.0);
        };
        t = t.mul(&c, w);
        negative ^= neg;
        rel += rc + 2.0;
    }
    t = t.mul(&uv, w);
    rel += ru + 2.0;
    let mut v = t.to_fixed(FRACTION);
    if negative {
        v = -v;
    }
    // `|T| rel 2^-w`, in doubles through its logarithm.
    let e = (t.log2() + 1.0 + rel.log2() - w as f64).exp2();
    (v, e)
}

/// Whether `cos(π a / b)` to `w` bits takes fewer products by Newton's method on a Chebyshev
/// polynomial than by its series: about `3 (2 log2 m + 3 popcount m + 3)` for the polynomial of
/// degree `m` against [`real::cos_pi`]'s.
fn newton_is_cheaper(a: u64, b: u64, w: u64) -> bool {
    let g = {
        let (mut x, mut y) = (a, b);
        while y != 0 {
            (x, y) = (y, x % y);
        }
        x
    };
    let b = b / g.max(1);
    if b > real::NEWTON_DENOMINATOR {
        return false;
    }
    let m = if b.is_multiple_of(2) { b / 2 } else { b };
    let newton = 3 * (2 * u64::from(64 - m.leading_zeros()) + 3 * u64::from(m.count_ones()) + 3);
    let s = real::reduction(w);
    let t = real::terms(2 * s, w + 2 * s);
    let series = 2 * (t as f64).sqrt() as u64 + s + t / 8;
    newton < series
}

/// Whether `e^(-C/k)` takes fewer products as a root of `e^C` than `e^(C/k)` by its series, to `w`
/// bits: Newton's last step and the check take about `2 (log2 k + popcount k) + 2` products and the
/// steps before as many again; the series [`real::exp`]'s.
fn root_is_cheaper(k: u64, w: u64) -> bool {
    let bits = u64::from(64 - k.leading_zeros()) + u64::from(k.count_ones());
    let root = 4 * bits + 8;
    let s0 = real::reduction(w);
    let t = real::terms(s0, w + 2 * s0);
    let series = 2 * (t as f64).sqrt() as u64 + s0 + 16 + t / 8;
    root < series
}

/// `a - b` for floating numbers with `a > b > 0`, cut to `w` bits.
fn sub(a: &Float, b: &Float, w: u64) -> Float {
    let e = a.e.min(b.e);
    Float {
        m: &shift(&a.m, a.e - e) - &shift(&b.m, b.e - e),
        e,
    }
    .cut(w)
}

//! The logarithm: `ln`, `log2`, `log10` and `ln_1p`, correctly rounded.
//!
//! The kernel writes `x = 2^e u` with `2/3 <= u < 4/3` and takes `ln x = e ln 2 + ln u`. Up to
//! [`LN_LIMBS`] limbs after the point `ln u` comes from two tables, `ln(d/128)` and `ln(1 + j/2^14)`,
//! and a short series (see [`ln_small`](super::small_fix::ln_small)), `log2` and `log10` from it
//! times a constant; beyond, by Newton's iteration on the exponential, `y ← y + L(u e^-y - 1)` with
//! `L(v) = v - v²/2 + v³/3`, which multiplies the bits that are right by four, from the `f64`
//! logarithm and at a precision four times larger at each step, so that the steps together cost
//! about a third more than the last; only the last step's error counts, and it is bounded from the
//! `v` it finds. From [`AGM_BITS`] on, by the arithmetic-geometric mean on numbers of limbs
//! ([`ln_agm`]).

use super::ball::{Ball, Rad, round_near, ziv_fast};
use super::consts::ln2_approx;
use super::exp::exp_fix;
use super::fix::{self, SFix, Scratch};
use super::limbs;
use super::small_fix::{Base, LN_LIMBS, log_small};
use super::ziv::Approx;

use super::{BigFloat, Kind, Round};
use crate::math::numbers::{Number, nat};
use core::cmp::Ordering;

/// `ln u` for the fixed-point `u` (`n` limbs of fraction, `W = 64 n`) in about `[2/3, 4/3]`: the
/// signed fixed-point `y` and a bound of `|ln u - y|`.
///
/// The last step computes `v = u e^(-y) - 1` from `e^(-y)` within a relative `ε` (the product and
/// the scaling by `2^k` truncated, two units), so `|v - v_t| <= δ_v = (1 + |v|) ε + 2u` for the
/// exact `v_t = u e^(-y) - 1`, and `ln u = y + ln(1 + v_t)` exactly. With `|v| <= 1/4` and `δ_v <=
/// 1/16`, `|ln(1 + v_t) - ln(1 + v)| <= 2 δ_v`, `|ln(1 + v) - L(v)| <= |v|^4 / (4 (1 - |v|)) <=
/// |v|^4 / 2`, and `L(v)` computed with three truncations and a division by 3, five units: `|ln u -
/// y'| <= 2 δ_v + 5u + |v|^4 / 2`.
pub(crate) fn ln_fix(u: &[u64], n: usize) -> (SFix, Rad) {
    if n <= LN_LIMBS
        && let Some((neg, y, e)) = super::small_fix::ln_small::<11>(u, n)
    {
        let mut m = y.to_vec();
        limbs::trim(&mut m);
        return (SFix { neg, m }, e);
    }
    let w = 64 * n as u64;
    let one = {
        let mut v = vec![0u64; n + 1];
        v[n] = 1;
        v
    };
    let d = SFix {
        neg: false,
        m: u.to_vec(),
    }
    .sub(&SFix { neg: false, m: one });
    if d.is_zero() {
        return (SFix::zero(), Rad::ZERO);
    }
    // The levels of the iteration, in limbs: four times as many at each.
    let mut levels = vec![n];
    while *levels.last().expect("a level") > 3 {
        let l = *levels.last().expect("a level");
        levels.push(l.div_ceil(4) + 1);
    }
    levels.reverse();
    let n0 = levels[0];
    // The seed: the `f64` logarithm, or `d` itself when it is too small for an `f64`.
    let df = d.to_f64(n);
    let mut y = if df.abs() > 1e-300 {
        let y0 = df.ln_1p();
        let bits = y0.abs().to_bits();
        let (mant, e) = if bits >> 52 == 0 {
            (bits << 1, -1075i128)
        } else {
            (
                (bits & ((1 << 52) - 1)) | 1 << 52,
                ((bits >> 52) as i128) - 1075,
            )
        };
        let (m, _) = fix::of_mag(&[mant], e, n0);
        SFix { neg: y0 < 0.0, m }
    } else {
        SFix {
            neg: d.neg,
            m: fix::shr(&d.m, 64 * (n - n0) as u64),
        }
    };
    let mut scratch = Scratch::default();
    let mut prev = n0;
    let mut bound = Rad::int(1);
    for &l in &levels {
        // `y` from the level before, exactly.
        if l > prev {
            y.m = limbs::shl(&y.m, 64 * (l - prev) as u64);
        }
        prev = l;
        let wl = 64 * l as u64;
        let ul = Rad::pow2(-i128::from(wl));
        let uj = fix::shr(u, 64 * (n - l) as u64);
        let (e, k, eps) = exp_fix(!y.neg && !y.is_zero(), &y.m, l);
        let mut p = fix::mul(&uj, &e, l, l + 2, &mut scratch);
        if k < 0 {
            p = fix::shr(&p, k.unsigned_abs());
        } else if k > 0 {
            p = limbs::shl(&p, k as u64);
        }
        let mut onel = vec![0u64; l + 1];
        onel[l] = 1;
        let v = SFix { neg: false, m: p }.sub(&SFix {
            neg: false,
            m: onel,
        });
        let v2 = fix::sqr(&v.m, l, l + 1, &mut scratch);
        let mut v3 = fix::mul(&v2, &v.m, l, l + 1, &mut scratch);
        fix::div1(&mut v3, 3);
        let half = fix::shr(&v2, 1);
        let poly = v
            .sub(&SFix {
                neg: false,
                m: half,
            })
            .add(&SFix { neg: v.neg, m: v3 });
        y = y.add(&poly);
        if l == n {
            let vb = fix::bits(&v.m);
            let vmag = if vb == 0 {
                Rad::ZERO
            } else {
                Rad::pow2(vb as i128 - i128::from(wl))
            };
            let dv = eps.mul(Rad::int(1).add(vmag)).add(ul.shl(1));
            if !vmag.le(Rad::pow2(-2)) || !dv.le(Rad::pow2(-4)) {
                return (y, Rad::int(1));
            }
            let rem = vmag.mul(vmag).mul(vmag).mul(vmag).shl(-1);
            bound = dv.shl(1).add(ul.mul(Rad::int(5))).add(rem);
        }
    }
    debug_assert!(w == 64 * n as u64);
    (y, bound)
}

/// `ln 2` in fixed point with `W + g` bits of fraction (`n` limbs and `g` bits), and the bound of
/// its error.
pub(crate) fn ln2_fix(n: usize, g: u64) -> (Vec<u64>, Rad) {
    let w = 64 * n as u64;
    let a = ln2_approx(w + g + 8);
    let (lf, dropped) = fix::of_mag(&a.m, a.lsb + i128::from(g), n);
    let mut d = Rad::pow2(a.err);
    if dropped {
        d = d.add(Rad::pow2(-i128::from(w + g)));
    }
    (lf, d)
}

/// The natural logarithm of the ball `x` with an error about `2^-w` of the result, `None` when the
/// ball holds a number not above zero.
///
/// With `x = 2^e u`, `ln x = e ln 2 + ln u`: for `e = 0` the result is `ln u`, near `u - 1`, and
/// the fixed point takes as many more bits as `u - 1` has leading zeros; otherwise `|ln x| >= ln 2 -
/// ln(3/2) > 1/4`. The midpoint truncated to the fixed point is within `u` below, and `ln` within
/// `u / (2/3) < 2u`; a radius `ρ` relative to the midpoint changes `ln` by at most `ρ / (1 - ρ) <=
/// 2ρ` for `ρ <= 1/2`; `e ln 2` is within `|e| δ_L + u`.
pub(crate) fn ln_ball(x: &Ball, w: u64) -> Option<Ball> {
    let (neg, m, lsb, err) = ln_raw(x, w)?;
    Some(Ball::of_limbs(neg, &m, lsb, err))
}

/// [`ln_ball`] in the raw: the sign, the magnitude `M`, the exponent of its last bit and the
/// bound of the error, for an approximation without the rounding of a ball's midpoint.
pub(crate) fn ln_raw(x: &Ball, w: u64) -> Option<(bool, Vec<u64>, i128, Rad)> {
    let of_ball = |b: Ball| -> Option<(bool, Vec<u64>, i128, Rad)> {
        let (neg, m, lsb) = b.mid.parts()?;
        Some((neg, m.to_vec(), lsb, b.rad))
    };
    let lower = x.lower()?;
    if x.mid.is_sign_negative() {
        return None;
    }
    let (_, m, lsb) = x.mid.parts()?;
    let ex = x.mid.exponent().expect("finite");
    let (e, extra) = reduce(m, ex);
    if extra > w + 8 {
        // `|ln(1 + d) - d| <= d² / (2 (1 - |d|)) <= d²` for `d = x - 1`, `|d| < 2^-(w + 8)`.
        let d = x
            .mid
            .sub_round(&BigFloat::from(1), x.mid.prec + 2, Round::Nearest)
            .0;
        let (_, hd) = super::ball::mag_bounds(&d);
        let mut err = hd.mul(hd);
        if !x.rad.is_zero() {
            let rho = x.rad.div(lower);
            if !rho.le(Rad::pow2(-1)) {
                return None;
            }
            err = err.add(rho.shl(1));
        }
        return of_ball(Ball::new(d, err));
    }
    if w + extra >= AGM_BITS
        && let Some((neg, m, lsb, mut err)) = ln_agm(m, lsb, w, extra)
    {
        if !x.rad.is_zero() {
            let rho = x.rad.div(lower);
            if !rho.le(Rad::pow2(-1)) {
                return None;
            }
            err = err.add(rho.shl(1));
        }
        return Some((neg, m, lsb, err));
    }
    let n = (w + extra + 16).div_ceil(64) as usize;
    let wb = 64 * n as u64;
    if n <= LN_LIMBS
        && let Some((neg, v, mut err)) = log_tab(m, lsb, e, n, Base::E)
    {
        if !x.rad.is_zero() {
            let rho = x.rad.div(lower);
            if !rho.le(Rad::pow2(-1)) {
                return None;
            }
            err = err.add(rho.shl(1));
        }
        return Some((neg, v, -i128::from(wb), err));
    }
    let u = Rad::pow2(-i128::from(wb));
    let (uf, dropped) = fix::of_mag(m, lsb - i128::from(e), n);
    let (mut y, mut err) = ln_fix(&uf, n);
    if dropped {
        err = err.add(u.shl(1));
    }
    if e != 0 {
        let ea = e.unsigned_abs();
        let g = 64 - (ea + 1).leading_zeros() as u64 + 4;
        let (lf, dl) = ln2_fix(n, g);
        let el = fix::shr(&fix::mul1(&lf, ea), g);
        y = y.add(&SFix { neg: e < 0, m: el });
        err = err.add(dl.mul(Rad::int(ea))).add(u);
    }
    if !x.rad.is_zero() {
        let rho = x.rad.div(lower);
        if !rho.le(Rad::pow2(-1)) {
            return None;
        }
        err = err.add(rho.shl(1));
    }
    let mut m = y.m;
    limbs::trim(&mut m);
    Some((y.neg, m, -i128::from(wb), err))
}

/// The exponent `e` of the reduction `x = 2^e u`, `u` in `[2/3, 4/3)` about, of the mantissa `m`
/// and the exponent `ex` of `x`, and the bits of cancellation of `ln u` near zero: for `e = 0`,
/// `|u - 1| >= 2^-(run + 1)` for the run of the bits below the top bit equal to the next.
fn reduce(m: &[u64], ex: i64) -> (i64, u64) {
    let top = m[m.len() - 1];
    let (e, run) = if top < 0xAAAA_AAAA_AAAA_AAAA {
        // `u = 2 u0` in `[1, 4/3)`: `u - 1` has as many leading zeros as the mantissa below its top.
        (ex - 1, leading_run(m, false))
    } else {
        (ex, leading_run(m, true))
    };
    (e, if e == 0 { run } else { 0 })
}

/// The approximation for Ziv's loop of a fixed-point result `±y·2^lsb` within `err`; one that
/// never decides when the error is as large as the value.
fn fixed_approx(negative: bool, m: Vec<u64>, lsb: i128, rad: Rad) -> Approx {
    let err = rad.exp2_ceil().max(lsb - 128);
    if m.is_empty() || err >= lsb + i128::from(limbs::bit_len(&m)) {
        super::ball::undecided()
    } else {
        Approx {
            negative,
            m,
            lsb,
            err,
        }
    }
}

/// The logarithm in the base `base` of the finite `x > 0` at the working precision `w` by the
/// tables of [`log_small`], when it takes at most [`LN_LIMBS`] limbs after the point.
fn log_short(x: &BigFloat, w: u64, base: Base) -> Option<Approx> {
    let (_, m, lsb) = x.parts()?;
    log_short_of(m, lsb, x.exponent()?, w, base)
}

/// [`log_short`] of `M·2^lsb` for the normalized mantissa `m` (its top bit set) and the exponent
/// `ex` of the number.
fn log_short_of(m: &[u64], lsb: i128, ex: i64, w: u64, base: Base) -> Option<Approx> {
    let (e, extra) = reduce(m, ex);
    let n = (w + extra + 16).div_ceil(64) as usize;
    if n > LN_LIMBS {
        return None;
    }
    let (neg, v, err) = log_tab(m, lsb, e, n, base)?;
    Some(fixed_approx(neg, v, -64 * n as i128, err))
}

/// [`log_small`] in arrays of seven limbs up to four limbs after the point and of eleven beyond,
/// the magnitude as a vector.
fn log_tab(m: &[u64], lsb: i128, e: i64, n: usize, base: Base) -> Option<(bool, Vec<u64>, Rad)> {
    let (neg, mut v, err) = if n <= 4 {
        let (neg, y, err) = log_small::<7>(m, lsb, e, n, base)?;
        (neg, y.to_vec(), err)
    } else {
        let (neg, y, err) = log_small::<11>(m, lsb, e, n, base)?;
        (neg, y.to_vec(), err)
    };
    limbs::trim(&mut v);
    Some((neg, v, err))
}

/// The logarithm takes the AGM from this many bits of working precision on: measured on an Apple
/// M4 Max (a loaded machine), a logarithm of 3,000 bits took 27.2 µs by Newton's iteration and
/// 28.1 µs by the AGM, of 4,000 bits 45.8 µs against 43.2 µs, of 6,000 bits 99.6 against 79.0.
pub(crate) const AGM_BITS: u64 = 3_400;

/// A positive number of the AGM: `m·2^e`, `m` of `n` limbs with the top bit set.
struct Fl {
    m: Vec<u64>,
    e: i64,
}

impl Fl {
    /// `v·2^e` (`v` not zero) cut to `n` limbs, from below: within a relative `2^(1 - 64 n)`.
    fn of(v: &[u64], e: i64, n: usize) -> Fl {
        let bits = limbs::bit_len(v) as i64;
        let sh = 64 * n as i64 - bits;
        let mut m = if sh >= 0 {
            limbs::shl(v, sh as u64)
        } else {
            limbs::shr(v, sh.unsigned_abs())
        };
        m.resize(n, 0);
        Fl { m, e: e - sh }
    }

    /// The exponent of the top bit, plus one: `2^(top - 1) <= self < 2^top`.
    fn top(&self) -> i64 {
        self.e + 64 * self.m.len() as i64
    }
}

/// `(a + b)/2` for `a >= b`, from below within a relative `2^(2 - 64 n)`: `b` cut at `a`'s last
/// place (below one unit of it, at most `2^(1 - 64 n) a`) and the sum cut to `n` limbs.
fn agm_mean(a: &Fl, b: &Fl, n: usize) -> Fl {
    let d = a.e - b.e;
    if d >= 64 * n as i64 {
        return Fl::of(&a.m, a.e - 1, n);
    }
    let bs = if d >= 0 {
        limbs::shr(&b.m, d as u64)
    } else {
        limbs::shl(&b.m, d.unsigned_abs())
    };
    Fl::of(&limbs::add(&a.m, &bs), a.e - 1, n)
}

/// `sqrt(a b)` from below within a relative `2^(2 - 64 n)`: the integer square root of the exact
/// product (one bit cut first when its exponent is odd, `2^(2 - 128 n)` at most), and one more cut
/// when the root falls a bit short of `n` limbs.
fn agm_root(
    a: &Fl,
    b: Option<&Fl>,
    n: usize,
    prod: &mut Vec<u64>,
    root: &mut Vec<u64>,
    scratch: &mut Vec<u64>,
) -> Fl {
    let mut e = match b {
        Some(b) => {
            if core::ptr::eq(a, b) {
                nat::sqr_into(prod, &a.m, scratch);
            } else {
                nat::mul_into(prod, &a.m, &b.m, scratch);
            }
            a.e + b.e
        }
        None => {
            // `a` alone: `a·2^(64 n)` below it.
            prod.clear();
            prod.resize(n, 0);
            prod.extend_from_slice(&a.m);
            a.e - 64 * n as i64
        }
    };
    if e % 2 != 0 {
        let v = limbs::shr(prod, 1);
        prod.clear();
        prod.extend_from_slice(&v);
        e += 1;
    }
    limbs::trim(prod);
    nat::sqrt_rem(root, None, prod, scratch);
    Fl::of(root, e / 2, n)
}

/// `ln x` for `x = M·2^lsb` (exact) by the arithmetic-geometric mean, as MPFR's `mpfr_log` but on
/// numbers of `n` limbs cut from below at each step: for `s = x 2^k >= 2^(T/2 + 8)`, `ln x = π / (2
/// AGM(1, 4/s)) - k ln 2 = π a / (2 AGM(a, 1)) - k ln 2` with `a = s/4`, the AGM being homogeneous.
/// The sign, the magnitude, its last place and the bound of the error, which is about `2^-w`
/// (`extra` more bits for a result near zero); `None` when the iteration does not settle.
///
/// The formula: `AGM(1, k') = π / (2 K(k))` for `k'² + k² = 1`, and by DLMF 19.12.1 `K(k) =
/// Σ_(j >= 0) ((1/2)_j / j!)² k'^(2j) (ln(1/k') + d(j))` with `0 < d(j) <= ln 4`, so for `k' = 4/s`
/// the first term is `ln s` and every other one lies in `(0, k'^(2j) ln s]`: `0 <= π / (2 AGM(1,
/// 4/s)) - ln s <= ln(s) k'² / (1 - k'²)`.
///
/// The steps: the AGM is homogeneous and grows with each argument, so a pair cut from below by
/// relative errors at most `η` has an AGM within `(1 - η)` of the pair's, and the computed pairs'
/// AGM is within `(1 - η)^K` of `AGM(a, 1)` after `K` steps (`η = 2^(2 - 64 n)`, [`agm_mean`] and
/// [`agm_root`]). The iteration stops at `δ = (a - b)/(a + b) <= 2^-(T/4 + 3)` and takes
/// `AGM(a, b) = m AGM(1 + δ, 1 - δ) = m / (1 + S)`, `m = (a + b)/2`, `S = Σ_(j >= 1) c_j² δ^(2j)`
/// (`K(δ) = (π/2)(1 + S)`, `c_j = (1/2)_j / j! <= 1/2`): `S = δ²/4 + R`, `0 <= R <= δ⁴/(4(1 -
/// δ²))`, and `|1/(1 + S) - (1 - δ²/4)| <= R + S² <= δ⁴/3`, so `AGM(a, b) = m - (a - b)²/(16 m)`
/// within `δ⁴/3 <= 2^-(T + 12)` relatively, the correction computed to `T/2 + 60` bits of its own.
/// Then `π` within its bound, the product exact and the quotient cut (`2^(2 - 64 n)`): in all a
/// relative error `ρ <= (K + 8) η` of `ln s < 2^lb`, and `k ln 2` within `|k| δ_L`; `M` cut to `n`
/// limbs moves `ln x` by at most `2^(1 - 64 n)`.
fn ln_agm(m: &[u64], lsb: i128, w: u64, extra: u64) -> Option<(bool, Vec<u64>, i128, Rad)> {
    let mut mt = m.to_vec();
    limbs::trim(&mut mt);
    let ex = (lsb + i128::from(limbs::bit_len(&mt))) as i64;
    // `ln s` below `2^lb`: `s < 2^(T/2 + 10)`, `T` below `2^40`.
    let lw = 64 - w.leading_zeros() as u64;
    let t = w + extra + 4;
    let k = (t / 2 + 8) as i64 - ex + 1;
    let lb = 64 - (t / 2 + 16).leading_zeros() as u64;
    // The limbs: `T` bits, the `lb` of `ln s` and about the `log2 4(K + 8)` of the steps.
    let n = (t + lb + 10).div_ceil(64) as usize;
    let nw = 64 * n as i128;
    // `a = s/4 = x 2^(k - 2)`.
    let a0 = Fl::of(&mt, (lsb + i128::from(k - 2)) as i64, n);
    let one = Fl::of(&[1], 0, n);
    let (mut prod, mut root, mut scratch) = (Vec::new(), Vec::new(), Vec::new());
    let mut b = agm_root(&a0, None, n, &mut prod, &mut root, &mut scratch);
    let mut a = agm_mean(&a0, &one, n);
    let mut steps = 1u64;
    let stop = (t / 4 + 3) as i64;
    let g = loop {
        if steps > 4 * lw + 64 {
            return None;
        }
        // `|a - b|`, when their exponents are near.
        if (a.e - b.e).abs() <= 1 {
            let lo = a.e.min(b.e);
            let x = limbs::shl(&a.m, (a.e - lo) as u64);
            let y = limbs::shl(&b.m, (b.e - lo) as u64);
            let d = match limbs::cmp(&x, &y) {
                Ordering::Less => limbs::sub(&y, &x),
                _ => limbs::sub(&x, &y),
            };
            // `δ <= |a - b| / max(a, b) < 2^(top(d) - (top(max) - 1))`.
            let hi = a.top().max(b.top());
            let dt = lo + limbs::bit_len(&d) as i64;
            if d.is_empty() || dt - (hi - 1) <= -stop {
                let (big, small) = if a.top() >= b.top() {
                    (&a, &b)
                } else {
                    (&b, &a)
                };
                let mean = agm_mean(big, small, n);
                if d.is_empty() {
                    break mean;
                }
                // `(a - b)²/(16 m)` to `T/2 + 64` bits about: `d` and `m` cut to `h` bits.
                let h = t / 2 + 64;
                let dh = Fl::of(&d, lo, h.div_ceil(64) as usize);
                let mh = Fl::of(&mean.m, mean.e, h.div_ceil(64) as usize);
                let mut num = limbs::mul(&dh.m, &dh.m);
                let hl = mh.m.len();
                // A quotient of about `hl` limbs: the numerator `2 hl` limbs long at least.
                let pad = (2 * hl).saturating_sub(num.len());
                let mut v = vec![0u64; pad];
                v.extend_from_slice(&num);
                num = v;
                let mut q = vec![0u64; num.len() - hl];
                let qh = nat::div_qr(&mut q, &mut num, &mh.m, &mut scratch);
                if qh {
                    q.push(1);
                }
                limbs::trim(&mut q);
                let ec = 2 * dh.e - 64 * pad as i64 - mh.e - 4;
                // `m - c`, `c` cut at `m`'s last place.
                let sh = mean.e - ec;
                let c = if sh >= 0 {
                    limbs::shr(&q, sh as u64)
                } else {
                    limbs::shl(&q, sh.unsigned_abs())
                };
                let r = limbs::sub(&mean.m, &c);
                break Fl::of(&r, mean.e, n);
            }
        }
        let a2 = agm_mean(&a, &b, n);
        b = agm_root(&a, Some(&b), n, &mut prod, &mut root, &mut scratch);
        a = a2;
        steps += 1;
    };
    // `Q = π a / (2 g)`.
    let pi = super::consts::pi_approx(64 * n as u64 + 16);
    let pf = Fl::of(&pi.m, pi.lsb as i64, n);
    let rel_pi = Rad::pow2(pi.err - pi.lsb - limbs::bit_len(&pi.m) as i128 + 2);
    nat::mul_into(&mut prod, &pf.m, &a0.m, &mut scratch);
    let mut num = core::mem::take(&mut prod);
    let mut q = vec![0u64; num.len() - n];
    let qh = nat::div_qr(&mut q, &mut num, &g.m, &mut scratch);
    if qh {
        q.push(1);
    }
    limbs::trim(&mut q);
    let lq = i128::from(pf.e + a0.e - g.e - 1);
    // `k ln 2`.
    let kb = 64 - k.unsigned_abs().leading_zeros() as u64;
    let l2 = ln2_approx(64 * n as u64 + kb + 8);
    let kl = fix::mul1(&l2.m, k.unsigned_abs());
    let low = lq.min(l2.lsb);
    let qa = limbs::shl(&q, (lq - low) as u64);
    let ka = limbs::shl(&kl, (l2.lsb - low) as u64);
    // `ln x = Q - k ln 2`.
    let (neg, mut v) = match (k < 0, limbs::cmp(&qa, &ka)) {
        (true, _) => (false, limbs::add(&qa, &ka)),
        (false, Ordering::Less) => (true, limbs::sub(&ka, &qa)),
        (false, _) => (false, limbs::sub(&qa, &ka)),
    };
    limbs::trim(&mut v);
    let eta = Rad::pow2(2 - nw);
    let rho = eta.mul(Rad::int(steps + 8)).add(rel_pi);
    let k2 = Rad::pow2(-2 * i128::from(t / 2 + 6));
    let err = rho
        .add(k2)
        .shl(i128::from(lb))
        .mul(Rad::up(257, -8))
        .add(Rad::pow2(1 - nw))
        .add(Rad::pow2(l2.err).mul(Rad::int(k.unsigned_abs())));
    Some((neg, v, low, err))
}

/// The length of the run of bits equal to `ones` below the top bit of the mantissa `m` (the top
/// bit set): up to its length, which means all of them.
fn leading_run(m: &[u64], ones: bool) -> u64 {
    let mut count = 0u64;
    let mut first = true;
    for &limb in m.iter().rev() {
        let mut x = if ones { !limb } else { limb };
        let mut avail = 64;
        if first {
            x &= !(1u64 << 63);
            // The top bit itself is not counted.
            x <<= 1;
            avail = 63;
            first = false;
            let lz = (x.leading_zeros() as u64).min(avail);
            count += lz;
            if lz < avail {
                return count;
            }
            continue;
        }
        let lz = (x.leading_zeros() as u64).min(avail);
        count += lz;
        if lz < avail {
            return count;
        }
    }
    count
}

/// The ball of `ln 10` to `w` bits, from the cached constant (within `2^(1 - (w + 8))`).
pub(crate) fn ln10_ball(w: u64) -> Ball {
    let a = super::consts::ln10_approx(w + 8);
    Ball::of_limbs(false, &a.m, a.lsb, Rad::pow2(a.err))
}

/// The ball of `ln 2` to `w` bits.
pub(crate) fn ln2_ball(w: u64) -> Ball {
    let a = ln2_approx(w + 8);
    Ball::of_limbs(false, &a.m, a.lsb, Rad::pow2(a.err))
}

/// A result that is exact.
#[inline]
fn exact(x: BigFloat) -> (BigFloat, Ordering) {
    (x, Ordering::Equal)
}

/// `x - 1` exactly, for a finite `x` in `[1/2, 2]`.
fn minus_one(x: &BigFloat) -> BigFloat {
    x.sub_round(&BigFloat::from(1), x.prec + 2, Round::Nearest)
        .0
}

/// The logarithm in the base `base` of a positive finite `x` that is not 1: by the tables at short
/// working precisions, otherwise `ln x` or `ln x / ln base` (the ball of the constant), near 1 by
/// its rounding when `x - 1` is too small for the loop.
fn log_of(x: &BigFloat, prec: u64, round: Round, base: Base) -> (BigFloat, Ordering) {
    let e = x.exponent().expect("finite");
    // `|x - 1| < 2^-run` for the run of zeros (ones) below the top bit of a mantissa in `[1, 2)`
    // (`[1/2, 1)`): only a long one can be near enough for the rounding near `x - 1`.
    let near = base == Base::E
        && (0..=1).contains(&e)
        && x.parts()
            .is_some_and(|(_, m, _)| leading_run(m, e == 0) > prec);
    if near {
        let d = minus_one(x);
        let ed = i128::from(d.exponent().expect("x is not 1"));
        // `ln(1 + d) = d - d²/2 + …`, below `d`, within `d²` of it for `|d| < 1/2`.
        if ed < -(prec as i128) - 2
            && let Some(r) = round_near(&d, Ordering::Less, 2 * ed, prec, round)
        {
            return r;
        }
    }
    let mut xb: Option<Ball> = None;
    ziv_fast(prec, round, |w| {
        if let Some(a) = log_short(x, w + 4, base) {
            return a;
        }
        let xb = xb.get_or_insert_with(|| Ball::exact(x.clone()));
        let b = match base {
            Base::E => {
                return match ln_raw(xb, w + 4) {
                    Some((negative, m, lsb, rad)) => fixed_approx(negative, m, lsb, rad),
                    None => super::ball::undecided(),
                };
            }
            Base::Two => ln2_ball(w + 8),
            Base::Ten => ln10_ball(w + 8),
        };
        let l = ln_ball(xb, w + 4).expect("x > 0");
        match l.div(&b, w + 4) {
            Some(q) => q.approx(),
            None => super::ball::undecided(),
        }
    })
}

/// The integer `k` rounded, as a result.
fn integer(k: i64, prec: u64, round: Round) -> (BigFloat, Ordering) {
    BigFloat::from_parts(&Number::from(k), 0, prec, round)
}

impl BigFloat {
    /// The special values of the logarithms: NaN for NaN, `-∞` and numbers below zero; `-∞` for
    /// `±0`; `+∞` for `+∞`; `+0` for 1.
    fn log_special(&self, prec: u64) -> Option<(BigFloat, Ordering)> {
        match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } | Kind::Finite { negative: true, .. } => {
                Some(exact(BigFloat::nan(prec)))
            }
            Kind::Inf { negative: false } => Some(exact(BigFloat::infinity(prec))),
            Kind::Zero { .. } => Some(exact(BigFloat::neg_infinity(prec))),
            Kind::Finite { exp, mant, .. } => {
                if *exp == 1
                    && mant[mant.len() - 1] == 1 << 63
                    && mant[..mant.len() - 1].iter().all(|&l| l == 0)
                {
                    Some(exact(BigFloat::zero(prec)))
                } else {
                    None
                }
            }
        }
    }

    /// The natural logarithm rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding.
    ///
    /// `ln 1 = +0` exactly; `ln ±0 = -∞`; `ln +∞ = +∞`; NaN below zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::from(2).ln_round(53, Round::Nearest);
    /// assert_eq!(l, BigFloat::from(core::f64::consts::LN_2));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    ///
    /// # Panics
    ///
    /// When `prec` is not in [`MIN_PREC`](BigFloat::MIN_PREC)..=[`MAX_PREC`](BigFloat::MAX_PREC).
    #[track_caller]
    pub fn ln_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = self.log_special(prec) {
            return r;
        }
        log_of(self, prec, round, Base::E)
    }

    /// The natural logarithm to nearest at the precision of `self`.
    pub fn ln(&self) -> BigFloat {
        self.ln_round(self.prec, Round::Nearest).0
    }

    /// The logarithm in base 2 rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: exact for a power of two, as are its special values, those of
    /// [`ln_round`](Self::ln_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::from(0.125).log2_round(10, Round::Up);
    /// assert_eq!(l, BigFloat::from(-3));
    /// assert_eq!(o, Ordering::Equal);
    /// ```
    #[track_caller]
    pub fn log2_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = self.log_special(prec) {
            return r;
        }
        let (_, m, _) = self.parts().expect("finite");
        let power = m[m.len() - 1] == 1 << 63 && m[..m.len() - 1].iter().all(|&l| l == 0);
        if power {
            return integer(self.exponent().expect("finite") - 1, prec, round);
        }
        log_of(self, prec, round, Base::Two)
    }

    /// The logarithm in base 2 to nearest at the precision of `self`.
    pub fn log2(&self) -> BigFloat {
        self.log2_round(self.prec, Round::Nearest).0
    }

    /// The logarithm in base 10 rounded to `prec` bits in the mode `round`, and the direction of
    /// the rounding: exact for a power of ten, as are its special values, those of
    /// [`ln_round`](Self::ln_round).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::from(1000).log10_round(8, Round::Down);
    /// assert_eq!(l, BigFloat::from(3));
    /// assert_eq!(o, Ordering::Equal);
    /// ```
    #[track_caller]
    pub fn log10_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        if let Some(r) = self.log_special(prec) {
            return r;
        }
        if let Some(k) = power_of_ten(self) {
            return integer(k, prec, round);
        }
        log_of(self, prec, round, Base::Ten)
    }

    /// The logarithm in base 10 to nearest at the precision of `self`.
    pub fn log10(&self) -> BigFloat {
        self.log10_round(self.prec, Round::Nearest).0
    }

    /// `ln(1 + self)` rounded to `prec` bits in the mode `round`, and the direction of the
    /// rounding, accurate for `self` near zero, as MPFR's `mpfr_log1p`.
    ///
    /// `±0` for `±0`; `-∞` for `-1`; `+∞` for `+∞`; NaN below `-1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use a::math::numbers::Number;
    /// use core::cmp::Ordering;
    ///
    /// // ln(1 + 2^-100) is just below 2^-100.
    /// let (x, _) = BigFloat::from_parts(&Number::from(1), -100, 53, Round::Nearest);
    /// let (l, o) = x.ln_1p_round(53, Round::Down);
    /// assert!(l < x);
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn ln_1p_round(&self, prec: u64, round: Round) -> (BigFloat, Ordering) {
        let prec = BigFloat::check_prec(prec);
        match &self.kind {
            Kind::Nan | Kind::Inf { negative: true } => exact(BigFloat::nan(prec)),
            Kind::Inf { negative: false } => exact(BigFloat::infinity(prec)),
            Kind::Zero { negative } => exact(BigFloat::special(
                prec,
                Kind::Zero {
                    negative: *negative,
                },
            )),
            Kind::Finite { .. } => {
                let minus_one = BigFloat::from(-1);
                match self.partial_cmp(&minus_one).expect("not NaN") {
                    Ordering::Less => return exact(BigFloat::nan(prec)),
                    Ordering::Equal => return exact(BigFloat::neg_infinity(prec)),
                    Ordering::Greater => {}
                }
                let ex = i128::from(self.exponent().expect("finite"));
                // `ln(1 + x) = x - x²/2 + …`, below `x`, within `x²` of it for `|x| < 1/2`.
                if ex < -(prec as i128) - 2
                    && let Some(r) = round_near(self, Ordering::Less, 2 * ex, prec, round)
                {
                    return r;
                }
                // `1 + x` exactly, when it is short: its normalized mantissa and exponent.
                let short = one_plus(self);
                let x = Ball::exact(self.clone());
                let one = Ball::exact(BigFloat::from(1));
                ziv_fast(prec, round, |w| {
                    if let Some((m, lsb, e)) = &short
                        && let Some(a) = log_short_of(m, *lsb, *e, w + 4, Base::E)
                    {
                        return a;
                    }
                    // `1 + x` to `w + max(0, -e) + 8` bits: within a relative `2^-(w - e + 8)`,
                    // which changes the logarithm by about `|x| 2^-(w + 8)`.
                    let wu = w + (-ex).max(0) as u64 + 8;
                    let u = one.add(&x, wu);
                    match ln_ball(&u, w + 4) {
                        Some(l) => l.approx(),
                        None => Ball::new(BigFloat::zero(1), Rad::int(1)).approx(),
                    }
                })
            }
        }
    }

    /// `ln(1 + self)` to nearest at the precision of `self`.
    pub fn ln_1p(&self) -> BigFloat {
        self.ln_1p_round(self.prec, Round::Nearest).0
    }
}

/// `1 + x` exactly for a finite `x > -1`, when it takes at most eight limbs: the normalized
/// mantissa, the exponent of its last bit and the exponent of the number.
fn one_plus(x: &BigFloat) -> Option<(Vec<u64>, i128, i64)> {
    let (negative, m, lsb) = x.parts()?;
    let ex = i128::from(x.exponent()?);
    let low = lsb.min(0);
    if ex.max(1) - low > 64 * 8 {
        return None;
    }
    let xm = limbs::shl(m, (lsb - low) as u64);
    let one = limbs::shl(&[1], low.unsigned_abs() as u64);
    let mut v = if negative {
        limbs::sub(&one, &xm)
    } else {
        limbs::add(&one, &xm)
    };
    limbs::trim(&mut v);
    let top = *v.last()?;
    let z = u64::from(top.leading_zeros());
    let mut v = limbs::shl(&v, z);
    limbs::trim(&mut v);
    let l = low - i128::from(z);
    let e = (l + 64 * v.len() as i128) as i64;
    Some((v, l, e))
}

/// `k` when `x = 10^k` for an integer `k >= 1`.
fn power_of_ten(x: &BigFloat) -> Option<i64> {
    let (m, e) = x.mantissa_exponent()?;
    if m.is_negative() {
        return None;
    }
    let mut l = limbs::of_number(&m);
    let z = limbs::trailing_zeros(&l);
    l = limbs::shr(&l, z);
    let k = e + z as i64;
    if k < 1 {
        return None;
    }
    // `5^k` has `floor(k log2 5) + 1` bits, `log2 5 = 2.3219…`.
    let bits = limbs::bit_len(&l);
    let lo = (k as u128 * 23219 / 10000) as u64;
    if bits < lo || bits > lo + 2 {
        return None;
    }
    let five = super::exp::pow_u64(&Number::from(5), k as u64);
    (limbs::of_number(&five) == l).then_some(k)
}

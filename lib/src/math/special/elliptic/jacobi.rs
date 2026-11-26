//! The Jacobi elliptic functions `sn`, `cn` and `dn`.
//!
//! The argument is reduced modulo `2K` with `K` a double-double (the AGM in double-double), then
//! to `[0, K/2]` by the quarter-period identities, so that `cn` stays away from its zero; there
//! the descending Landen transformation (A&S 16.12) is applied to the values themselves, which
//! are products and quotients of positive quantities and keep their relative accuracy, down to a
//! parameter below 10⁻⁹ where the first-order expansion in it is exact to rounding, and the
//! ascent carries `sn` and `cn` as double-doubles (the steps of small `√μ'` by their increments,
//! the last ones as quotients in double-double), each rounded once; within `2^−40`
//! of 1, where those steps lose up to about `8K` ulps, the first-order expansion about `m = 1`
//! (A&S 16.15) at the reduced argument as a double-double instead. The parameters `m < 0` and
//! `m > 1` go to `(0, 1)` by the imaginary- and reciprocal-modulus transformations (A&S 16.10–11)
//! with the scaled argument a double-double, and `K` of the new parameter from its complement.

use super::super::dd::Dd;
use super::eft::{fast_two_sum, fma, mla, two_prod, two_sum};
use super::{k_dd, k_dd_at, k_log, ln_split};

const PI_DD: Dd = Dd {
    hi: core::f64::consts::PI,
    lo: 1.224_646_799_147_353_2e-16,
};

/// `K(m)` as a double-double, `K = π / (2 AGM(1, √m₁))`, from `m₁ = 1 − m` in `(0, 1)`.
fn k_agm_dd(m1: Dd) -> Dd {
    let mut a = Dd::new(1.0);
    let mut b = m1.sqrt();
    for _ in 0..64 {
        let d = a.sub(b);
        let an = a.add(b).mul_f64(0.5);
        if d.hi.abs() <= 2e-17 * a.hi {
            a = an;
            break;
        }
        b = a.mul(b).sqrt();
        a = an;
    }
    PI_DD.div(a.mul_f64(2.0))
}

/// The largest number of Landen steps (`m₁` down to the least subnormal takes about 12).
const STEPS: usize = 16;

/// `(sn, cn, dn)(v | m)` for a double-double `0 ≤ v ≤ K(m)/2`, `0 < m < 1`, `m₁ = 1 − m`.
///
/// The ascent carries `sn` and `cn` as double-doubles and `dn` as one (as `1 − dn` through the
/// steps of small `s`), each rounded once at the end. Rounded step by step, `sn` of the step
/// below the last gathered about an ulp per step (3.0 ulps at `(1.21 | 0.889)`), which the last
/// step's `dn` takes about one for one, and `dn` past `K/2` (`√m₁/dn`) and `1/dn` for `m < 0` up
/// to twice that: up to 3.2 and 4.9 ulps on 60,000 points of each with `μ < 0.9` (1.6 and 2.1
/// now), and 22.6 in `0.9 ≤ μ < 1 − 10⁻⁶` on 15,000 (10.8 now, on 60,000).
#[inline(always)]
fn landen(v: Dd, m: f64, m1: f64) -> (f64, f64, f64) {
    let mut sm = [0.0; STEPS];
    let mut oms = [0.0; STEPS];
    let mut n = 0;
    let mut mu = m;
    let mut kp = m1.sqrt();
    // ∏(1 + √μ'), exact to the double-double precision: the argument of the last step is
    // v/∏(1 + √μ'), whose roundings step by step (two per step) moved dn by up to 2 ulps at
    // (1.004 | 0.886).
    let mut pr = Dd::new(1.0);
    while mu >= 1e-9 && n < STEPS {
        // √μ' = (1 − k')/(1 + k') = μ/(1 + k')², 1 − √μ' = 2k'/(1 + k'), k'' = 2√k'/(1 + k').
        let q = 1.0 + kp;
        let s = mu / (q * q);
        sm[n] = s;
        oms[n] = 2.0 * kp / q;
        kp = 2.0 * kp.sqrt() / q;
        mu = s * s;
        let (h, l) = fast_two_sum(1.0, s);
        pr = pr.mul(Dd { hi: h, lo: l });
        n += 1;
    }
    let vn = if n == 0 { v } else { quot(v, pr) };
    let v = vn.hi;
    let (s0, c0) = v.sin_cos();
    if n == 0 {
        // sin and cos of v + vn.lo, to first order in vn.lo (half an ulp of v).
        let (s0, c0) = (mla(vn.lo, c0, s0), mla(-vn.lo, s0, c0));
        let t = 0.25 * mu * (v - s0 * c0);
        return (
            mla(-t, c0, s0),
            mla(t, s0, c0),
            mla(-0.5 * mu * s0, s0, 1.0),
        );
    }
    // The values at μ < 10⁻⁹ to first order in μ (A&S 16.13), with vn.lo to first order:
    // sn and cn as double-doubles, e = 1 − dn.
    let t = 0.25 * mu * (v - s0 * c0);
    let (h, l) = fast_two_sum(s0, (vn.lo - t) * c0);
    let mut sn = Dd { hi: h, lo: l };
    let (h, l) = fast_two_sum(c0, (t - vn.lo) * s0);
    let mut cn = Dd { hi: h, lo: l };
    let mut e = 0.5 * mu * s0 * s0;
    // The steps of s ≤ 1/4 (every one below the last for m < 0.987) by their increments, which are
    // below s of the values, so that their own roundings are a fraction of an ulp; with x = sn²
    // the steps (A&S 16.12.2–4) are sn' − sn = s sn cn²/(1 + s x), cn' − cn = −cn (e + s x)/(1 + s x)
    // and 1 − dn' = 2 s x/(1 + s x).
    while n > 1 && sm[n - 1] <= 0.25 {
        n -= 1;
        let s = sm[n];
        let x = sn.hi * sn.hi;
        let id = 1.0 / fma(s, x, 1.0);
        let ds = s * sn.hi * (cn.hi * cn.hi) * id;
        let g = mla(s, x, e) * id;
        e = 2.0 * s * x * id;
        // sn ≥ 0 and 0 ≤ ds ≤ sn, 0 ≤ g < 1: the sums are fast ones.
        let (h, l) = fast_two_sum(sn.hi, ds);
        let (h, l) = fast_two_sum(h, l + sn.lo);
        sn = Dd { hi: h, lo: l };
        let (p, pe) = two_prod(cn.hi, g);
        let (h, l) = fast_two_sum(cn.hi, -p);
        let (h, l) = fast_two_sum(h, (l - pe) + mla(-cn.lo, g, cn.lo));
        cn = Dd { hi: h, lo: l };
    }
    let (h, l) = fast_two_sum(1.0, -e);
    let mut dn = Dd { hi: h, lo: l };
    // The rest (next to m = 1) and the last step as quotients by 1 + s x in double-double, one
    // reciprocal for the three. dn's numerator 1 − s x moves by 2s sn δsn under errors of sn, cn;
    // (1 − s) x + cn² by 2(1 − s) sn δsn + 2 cn δcn, the smaller one where (2s − 1) sn > cn, next to
    // m = 1 (s → 1) with sn → 1, where 1 − s x cancels.
    loop {
        n -= 1;
        let s = sm[n];
        let x = sn.mul(sn);
        let t = x.mul_f64(s);
        let den = t.add_f64(1.0);
        let inv = 1.0 / den.hi;
        let num = if (2.0 * s - 1.0) * sn.hi <= cn.hi {
            Dd::new(1.0).sub(t)
        } else {
            x.mul_f64(oms[n]).add(cn.mul(cn))
        };
        let nsn = quot_by(times_one_plus(sn, s), den, inv);
        cn = quot_by(cn.mul(dn), den, inv);
        dn = quot_by(num, den, inv);
        sn = nsn;
        if n == 0 {
            return (sn.value(), cn.value(), dn.value());
        }
    }
}

/// `x (1 + s)` for `x ≥ 0` and `0 ≤ s < 1`, without rounding `1 + s`: as `fl(1 + s) x`, against
/// the argument `v/∏(1 + s)` formed exactly, each step's `sn` carried the rounding of `1 + s` (the
/// same one at every `u`), up to 3 ulps of the last step's `dn`.
#[inline(always)]
fn times_one_plus(x: Dd, s: f64) -> Dd {
    let (p, e) = two_prod(s, x.hi);
    let (h, l) = fast_two_sum(x.hi, p);
    let (hi, lo) = fast_two_sum(h, l + mla(s, x.lo, e));
    Dd { hi, lo }
}

/// `n/d` of double-doubles to about the double-double precision with one division: the quotient
/// by the reciprocal, then its residual (`Dd::div` divides twice, one after the other).
#[inline(always)]
fn quot(n: Dd, d: Dd) -> Dd {
    quot_by(n, d, 1.0 / d.hi)
}

/// `quot(n, d)` with `inv = 1/d.hi` given, for several quotients by one `d`.
#[inline(always)]
fn quot_by(n: Dd, d: Dd, inv: f64) -> Dd {
    let q = n.hi * inv;
    let r = fma(-q, d.hi, n.hi) + (n.lo - q * d.lo);
    Dd { hi: q, lo: r * inv }
}

/// The `m₁` below which `near_one` replaces the Landen steps: `2^−40` (`m` within `2^−40` of 1,
/// and of the transformed parameters, `m < 1 − 2^40` and `m` just above 1).
const M1_NEAR_ONE: f64 = 9.094_947_017_729_282e-13;

/// `(sn, cn, dn)(v | m)` for `0 ≤ v ≤ K/2` and `m₁ < 2^−40`, by the expansion about `m = 1` to the
/// first order in `m₁` (A&S 16.15.1–3), whose remainder there is below `0.005` ulp (it falls as
/// `m₁^(3/2)`; measured with mpmath): the Landen steps from such an `m` lose up to about `8K`
/// ulps of `dn`, and `e^(−v)` takes the absolute error of `v` (up to `K/2 ≈ ln(4/√m₁)/2`) as its
/// relative one, so `v` comes as a double-double. Out of line: a rare path, kept from the code of
/// the common one.
#[cold]
#[inline(never)]
fn near_one(v: Dd, m1: f64) -> (f64, f64, f64) {
    // x = e^(−v): e^(−(h + l)) = e^(−h)(1 − l) to O(l²).
    let x = (-v.hi).exp() * (1.0 - v.lo);
    let x2 = x * x;
    // tanh v = (1 − x²)/(1 + x²), 1 − x² from expm1 where it would cancel.
    let t = if v.hi < 0.5 {
        let e = (-2.0 * v.hi).exp_m1();
        let om = mla(2.0 * v.lo, 1.0 + e, -e);
        om / (2.0 - om)
    } else {
        (1.0 - x2) / (1.0 + x2)
    };
    let sech = 2.0 * x / (1.0 + x2);
    // sinh v cosh v = (1/x² − x²)/4, at most about 1/√m₁.
    let sc = 0.25 * (1.0 / x2 - x2);
    let (g, w) = (0.25 * m1, v.value());
    let sn = mla(g * (sc - w), sech * sech, t);
    let cn = mla(-g * (sc - w), t * sech, sech);
    let dn = mla(g * (sc + w), t * sech, sech);
    (sn, cn, dn)
}

/// `K(m)` for `m ≥ 0.8` from `m₁ = 1 − m` (positive, perhaps subnormal) by the logarithmic form:
/// for a transformed parameter `m` itself is rounded (to 1 for `m < −2^53` of the caller), `m₁`
/// is the accurate one.
fn k_of_m1(m1: Dd) -> Dd {
    let l = if m1.hi < f64::MIN_POSITIVE {
        // ln m₁ = ln(2^64 m₁) − 64 ln 2.
        ln_split(m1.hi * 18_446_744_073_709_551_616.0)
            .add_f64(-44.361_419_555_836_5)
            .add_f64(-1.484_189_960_861_631_7e-15)
    } else if m1.lo == 0.0 {
        ln_split(m1.hi)
    } else {
        ln_split(m1.hi).add_f64(m1.lo / m1.hi)
    };
    k_log(m1.hi, l.neg())
}

/// How the argument and the parameter of `core` come from the caller's `(u, m)`, for the
/// reduction of a huge argument, which computes them again to more bits.
#[derive(Clone, Copy)]
enum Form {
    /// `w = u`, `μ₁ = 1 − m` (`0 < m < 1`).
    Direct,
    /// `w = u √(1 − m)`, `μ₁ = 1/(1 − m)` (`m < 0`).
    Imaginary,
    /// `w = u √m`, `μ₁ = (m − 1)/m` (`m > 1`).
    Reciprocal,
}

/// The largest `|w|/K` reduced with the double-double `K`: `j·2K` is then exact to about
/// `2^37 · 2^−100 K`, below `2^−60` of the functions' scale.
const DD_REDUCTION: f64 = 68_719_476_736.0; // 2^36

/// `w − 2Kj` with `|r| ≤ K`, and whether `j` is odd, for `|w| ≤ 2^36 K`. The quotient `w/(2K)` in
/// `f64` gives `j` off by one next to an odd multiple of `K`, where `|r|` comes out just above
/// `K`: one step back corrects it.
fn reduce_dd(w: Dd, k: Dd) -> (Dd, bool) {
    let k2 = k.mul_f64(2.0);
    let mut j = (w.hi / k2.hi).round();
    let mut r = w.sub(k2.mul_f64(j));
    let a = if r.hi < 0.0 { r.neg() } else { r };
    if k.sub(a).hi < 0.0 {
        j += r.hi.signum();
        r = w.sub(k2.mul_f64(j));
    }
    (r, j % 2.0 != 0.0)
}

/// The reduction `w − 2Kj` of `|w| > 2^36 K` and whether `j` is odd: `w` and `K` of the transformed
/// parameter again from the caller's `(u, m)`, in `BigFloat` to the bits of `w/K` and 192 more,
/// `K = π/(2 AGM(1, √μ₁))` (A&S 17.6.1); `u` up to `f64::MAX` takes about 1,220 bits, and with a
/// transformed parameter `|m|` up to `f64::MAX` (whose `√|m|` scales `w`) about 1,730.
fn reduce_big(u: f64, m: f64, form: Form) -> (Dd, bool) {
    use crate::math::float::{BigFloat, Round};
    let n = Round::Nearest;
    let e2 = |x: f64| (x.abs().log2().max(0.0)) as u64;
    let bits = e2(u)
        + if let Form::Direct = form {
            0
        } else {
            e2(m) / 2 + 1
        };
    let p = bits + 192;
    let f = |x: f64| BigFloat::from_f64_round(x, p, n).0;
    let (bu, bm, one) = (f(u), f(m), f(1.0));
    let (w, mu1) = match form {
        Form::Direct => (bu, one.sub_round(&bm, p, n).0),
        Form::Imaginary => {
            let d = one.sub_round(&bm, p, n).0;
            let s = d.sqrt_round(p, n).0;
            (bu.mul_round(&s, p, n).0, one.div_round(&d, p, n).0)
        }
        Form::Reciprocal => {
            let s = bm.sqrt_round(p, n).0;
            let d = bm.sub_round(&one, p, n).0;
            (bu.mul_round(&s, p, n).0, d.div_round(&bm, p, n).0)
        }
    };
    // The AGM converges quadratically: stop when a and b agree to half the bits, then one mean.
    let half = f(0.5);
    let (mut a, mut b) = (one, mu1.sqrt_round(p, n).0);
    for _ in 0..64 {
        let d = a.sub_round(&b, p, n).0;
        let close = match (d.exponent(), a.exponent()) {
            (Some(ed), Some(ea)) => ed < ea - (p as i64) / 2 - 4,
            _ => true,
        };
        let an = a.add_round(&b, p, n).0.mul_round(&half, p, n).0;
        if close {
            a = an;
            break;
        }
        b = a.mul_round(&b, p, n).0.sqrt_round(p, n).0;
        a = an;
    }
    // 2K = π / AGM.
    let k2 = BigFloat::pi_round(p, n).0.div_round(&a, p, n).0;
    let j = w.div_round(&k2, p, n).0.round();
    let r = w.sub_round(&j.mul_round(&k2, p, n).0, p, n).0;
    let hi = r.to_f64();
    let lo = r.sub_round(&f(hi), p, n).0.to_f64();
    let odd = !j.mul_round(&half, p, n).0.is_integer();
    (Dd { hi, lo }, odd)
}

/// The values `core` finds at the reduced argument `v` (`0 ≤ v ≤ K/2`) and how they make those at
/// `w`: `swap` for `|r| = K − v` past `K/2` (the quarter-period identities), the signs, and
/// `r = √m₁` as a double-double (where it is used: past `K/2`, and by `quotients`, whose caller
/// sets it).
struct Raw {
    sv: f64,
    cv: f64,
    dv: f64,
    swap: bool,
    /// `sn` changes sign.
    sneg: bool,
    /// `cn` changes sign (and `sn` once more).
    flip: bool,
    r: Dd,
}

impl Raw {
    /// `(sn, cn, dn)(w | m)`.
    #[inline(always)]
    fn values(&self) -> (f64, f64, f64) {
        let (mut sn, mut cn, dn) = if self.swap {
            // sn(K − v) = cd v, cn(K − v) = √m₁ sd v, dn(K − v) = √m₁ nd v, the last (relative
            // where `cn` is not) with √m₁ a double-double and rounded once: it carries the
            // relative error of dn(v) into a value up to a binade lower, up to twice as many of its
            // ulps, and √m₁ and the quotient rounded apart took it from 2.6 to 3.6 ulps at
            // (69.3 | 0.890) on x86_64.
            (
                self.cv / self.dv,
                self.r.hi * self.sv / self.dv,
                quot(self.r, Dd::new(self.dv)).value(),
            )
        } else {
            (self.sv, self.cv, self.dv)
        };
        if self.sneg {
            sn = -sn;
        }
        if self.flip {
            sn = -sn;
            cn = -cn;
        }
        (sn, cn, dn)
    }

    /// `(sd, cd, nd)(w | m)` (times `√m₁` for `sd`), the values of the imaginary-modulus
    /// transformation: past `K/2` they are `(cn, sn, dn/√m₁)` at `v` themselves, which the
    /// quotients of `values` would round three or four times over.
    #[inline(always)]
    fn quotients(&self) -> (f64, f64, f64) {
        let (mut sd, mut cd, nd) = if self.swap {
            (self.cv, self.sv, quot(Dd::new(self.dv), self.r).value())
        } else {
            (
                self.r.hi * self.sv / self.dv,
                self.cv / self.dv,
                1.0 / self.dv,
            )
        };
        if self.sneg {
            sd = -sd;
        }
        if self.flip {
            sd = -sd;
            cd = -cd;
        }
        (sd, cd, nd)
    }
}

/// `(sn, cn, dn)(w | m)` for a double-double `w`, `0 < m < 1` (`m + dm` exactly, `dm` the low part
/// of a transformed parameter, else 0) and `m₁ = 1 − m` a double-double; `(u, m₀, form)` is how
/// the caller made them, for the reduction of a huge `w`.
fn core(w: Dd, m: f64, dm: f64, m1: Dd, (u, m0, form): (f64, f64, Form)) -> Raw {
    let uv = w.value();
    let neg = uv < 0.0;
    let au = uv.abs();
    let m1f = m1.value();
    // K from the complement where it is the accurate one (the same for m = 1 − m₁ exact), and at
    // m + dm: the reduction by up to four half-periods with K of the rounded m would add
    // w (K′/K) dm, up to 8 ulps of cn at m = −3.8 (μ = 0.79).
    let kd = if m >= 0.8 {
        k_of_m1(m1)
    } else if dm == 0.0 {
        k_dd(m)
    } else {
        k_dd_at(m, dm)
    };
    let kf = kd.value();
    let (v, swap, flip, sneg) = if au <= 0.4999 * kf {
        (if neg { w.neg() } else { w }, false, false, neg)
    } else {
        // The fitted K of m < 0.8 is good to 2^−57 relative: enough for the reduction by up to
        // four half-periods; otherwise (and in the logarithmic region, where ln m₁ limits it to
        // about 2^−54) K to the double-double precision by the AGM.
        let k = if m < 0.8 && au <= 8.0 * kf {
            kd
        } else {
            k_agm_dd(m1)
        };
        let (r, flip) = if au <= DD_REDUCTION * kf {
            reduce_dd(w, k)
        } else {
            reduce_big(u, m0, form)
        };
        let rneg = r.hi < 0.0;
        let a = if rneg { r.neg() } else { r };
        if a.hi > 0.5 * k.hi {
            // |r| ≤ K: K − |r| is below 0 by a rounding at most.
            let v = k.sub(a);
            (if v.hi < 0.0 { Dd::new(0.0) } else { v }, true, flip, rneg)
        } else {
            (a, false, flip, rneg)
        }
    };
    let (sv, cv, dv) = if m1f < M1_NEAR_ONE {
        near_one(v, m1f)
    } else {
        landen(v, m, m1f)
    };
    Raw {
        sv,
        cv,
        dv,
        swap,
        sneg,
        flip,
        r: if swap { m1.sqrt() } else { Dd::new(0.0) },
    }
}

/// `(sn, cn, dn)(u | m)` of `f64`s.
pub(crate) fn jacobi_f64(u: f64, m: f64) -> (f64, f64, f64) {
    const NAN3: (f64, f64, f64) = (f64::NAN, f64::NAN, f64::NAN);
    if u.is_nan() || m.is_nan() {
        return NAN3;
    }
    if m == 0.0 {
        if u.is_infinite() {
            return NAN3;
        }
        let (s, c) = u.sin_cos();
        return (s, c, 1.0);
    }
    if m == 1.0 {
        let c = 1.0 / u.cosh();
        return (u.tanh(), c, c);
    }
    if u == 0.0 {
        return (u, 1.0, 1.0);
    }
    if u.is_infinite() || m.is_infinite() {
        return NAN3;
    }
    if m > 0.0 && m < 1.0 {
        let (h, l) = two_sum(1.0, -m);
        return core(
            Dd::new(u),
            m,
            0.0,
            Dd { hi: h, lo: l },
            (u, m, Form::Direct),
        )
        .values();
    }
    if m < 0.0 {
        // sn(u|m) = √μ₁ sd(w|μ), cn = cd(w|μ), dn = nd(w|μ), μ = −m/(1 − m), μ₁ = 1/(1 − m),
        // w = u √(1 − m).
        let (h, l) = two_sum(1.0, -m);
        let d = Dd { hi: h, lo: l };
        let mu1 = Dd::new(1.0).div(d);
        let mu = Dd::new(-m).div(d);
        let w = d.sqrt().mul_f64(u);
        let mut raw = core(w, mu.hi, mu.lo, mu1, (u, m, Form::Imaginary));
        raw.r = mu1.sqrt();
        return raw.quotients();
    }
    // m > 1: sn(u|m) = sn(w|μ)/√m, cn = dn(w|μ), dn = cn(w|μ), μ = 1/m, w = u √m.
    let (h, l) = two_sum(m, -1.0);
    let mu1 = Dd { hi: h, lo: l }.div(Dd::new(m));
    let mu = Dd::new(1.0).div(Dd::new(m));
    let r = Dd::new(m).sqrt();
    let w = r.mul_f64(u);
    let (s, c, dn) = core(w, mu.hi, mu.lo, mu1, (u, m, Form::Reciprocal)).values();
    // sn/√m with √m a double-double, rounded once (divided by r.hi, up to 2.67 ulps of 1/√m
    // rather than 2.49 on 60,000 points with μ < 0.9).
    (quot(Dd::new(s), r).value(), dn, c)
}

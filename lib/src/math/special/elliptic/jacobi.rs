//! The Jacobi elliptic functions `sn`, `cn` and `dn`.
//!
//! The argument is reduced modulo `2K` with `K` a double-double (the AGM in double-double), then
//! to `[0, K/2]` by the quarter-period identities, so that `cn` stays away from its zero; there
//! the descending Landen transformation (A&S 16.12) is applied to the values themselves, which
//! are products and quotients of positive quantities and keep their relative accuracy, down to a
//! parameter below 10⁻⁹ where the first-order expansion in it is exact to rounding. The
//! parameters `m < 0` and `m > 1` go to `(0, 1)` by the imaginary- and reciprocal-modulus
//! transformations (A&S 16.10–11) with the scaled argument a double-double.

use super::super::dd::Dd;
use super::eft::{FMA, fma, mla, two_prod, two_sum};
use super::k_dd;

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

/// `(sn, cn, dn)(v | m)` for `0 ≤ v ≤ K(m)/2`, `0 < m < 1`, `m₁ = 1 − m`.
#[inline(always)]
fn landen(mut v: f64, m: f64, m1: f64) -> (f64, f64, f64) {
    let mut sm = [0.0; STEPS];
    let mut oms = [0.0; STEPS];
    let mut n = 0;
    let mut mu = m;
    let mut kp = m1.sqrt();
    while mu >= 1e-9 && n < STEPS {
        // √μ' = (1 − k')/(1 + k') = μ/(1 + k')², 1 − √μ' = 2k'/(1 + k'), k'' = 2√k'/(1 + k').
        let q = 1.0 + kp;
        let s = mu / (q * q);
        sm[n] = s;
        oms[n] = 2.0 * kp / q;
        kp = 2.0 * kp.sqrt() / q;
        mu = s * s;
        v /= 1.0 + s;
        n += 1;
    }
    let (s0, c0) = v.sin_cos();
    let t = 0.25 * mu * (v - s0 * c0);
    let mut sn = mla(-t, c0, s0);
    let mut cn = mla(t, s0, c0);
    let mut dn = mla(-0.5 * mu * s0, s0, 1.0);
    while n > 0 {
        n -= 1;
        let s = sm[n];
        let s2 = sn * sn;
        let id = 1.0 / fma(s, s2, 1.0);
        let nsn = (1.0 + s) * sn * id;
        let ncn = cn * dn * id;
        let ndn = fma(oms[n], s2, cn * cn) * id;
        // Back onto sn² + cn² = 1: the transformation amplifies errors off it.
        // sn² + cn² − 1, a cancellation: by fused operations, or with the products' errors.
        let d = if FMA {
            0.5 * nsn.mul_add(nsn, ncn.mul_add(ncn, -1.0))
        } else {
            let (a2, ea) = two_prod(nsn, nsn);
            let (b2, eb) = two_prod(ncn, ncn);
            let (t2, et) = two_sum(a2, b2);
            0.5 * ((t2 - 1.0) + (et + ea + eb))
        };
        sn = nsn - nsn * d;
        cn = ncn - ncn * d;
        dn = ndn;
    }
    (sn, cn, dn)
}

/// `(sn, cn, dn)(u | m)` for a double-double `u`, `0 < m < 1` and `m₁ = 1 − m` a double-double.
fn core(u: Dd, m: f64, m1: Dd) -> (f64, f64, f64) {
    let uv = u.value();
    let neg = uv < 0.0;
    let au = uv.abs();
    let m1f = m1.value();
    let kd = k_dd(m);
    let kf = kd.value();
    let (v, swap, flip, sneg) = if au <= 0.4999 * kf {
        (au, false, false, neg)
    } else {
        // The fitted K of m < 0.8 is good to 2^−57 relative: enough for the reduction by up to
        // four half-periods; otherwise (and in the logarithmic region, where ln m₁ limits it to
        // about 2^−54) K to the double-double precision by the AGM.
        let k = if m < 0.8 && au <= 8.0 * kf {
            kd
        } else {
            k_agm_dd(m1)
        };
        let k2 = k.mul_f64(2.0);
        let j = (uv / k2.hi).round();
        let r = u.sub(k2.mul_f64(j));
        let flip = j % 2.0 != 0.0;
        let rneg = r.hi < 0.0;
        let a = if rneg { r.neg() } else { r };
        if a.hi > 0.5 * k.hi {
            let w = k.sub(a).value();
            (w.max(0.0), true, flip, rneg)
        } else {
            (a.value(), false, flip, rneg)
        }
    };
    let (sv, cv, dv) = landen(v, m, m1f);
    let (mut sn, mut cn, dn) = if swap {
        // sn(K − v) = cd v, cn(K − v) = √m₁ sd v, dn(K − v) = √m₁ nd v.
        let r = m1f.sqrt();
        (cv / dv, r * sv / dv, r / dv)
    } else {
        (sv, cv, dv)
    };
    if sneg {
        sn = -sn;
    }
    if flip {
        sn = -sn;
        cn = -cn;
    }
    (sn, cn, dn)
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
        return core(Dd::new(u), m, Dd { hi: h, lo: l });
    }
    if m < 0.0 {
        // sn(u|m) = √μ₁ sd(w|μ), cn = cd(w|μ), dn = nd(w|μ), μ = −m/(1 − m), μ₁ = 1/(1 − m),
        // w = u √(1 − m).
        let (h, l) = two_sum(1.0, -m);
        let d = Dd { hi: h, lo: l };
        let mu1 = Dd::new(1.0).div(d);
        let mu = -m / h;
        let w = d.sqrt().mul_f64(u);
        let (s, c, dn) = core(w, mu, mu1);
        return (mu1.value().sqrt() * s / dn, c / dn, 1.0 / dn);
    }
    // m > 1: sn(u|m) = sn(w|μ)/√m, cn = dn(w|μ), dn = cn(w|μ), μ = 1/m, w = u √m.
    let (h, l) = two_sum(m, -1.0);
    let mu1 = Dd { hi: h, lo: l }.div(Dd::new(m));
    let mu = 1.0 / m;
    let r = Dd::new(m).sqrt();
    let w = r.mul_f64(u);
    let (s, c, dn) = core(w, mu, mu1);
    (s / r.hi, dn, c)
}

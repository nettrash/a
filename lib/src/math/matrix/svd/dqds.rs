//! The singular values of a real bidiagonal matrix by the differential quotient-difference
//! algorithm with shifts, dqds, as LAPACK's `lasq1`–`lasq6`: to high relative accuracy, and
//! with no square roots in its inner loop. The qd array `z` is indexed from 1 as in LAPACK, its
//! element 0 unused.

use crate::math::complex::Float;

/// The singular values of the upper bidiagonal matrix with the diagonal `d` and the
/// superdiagonal `e`, into `d`, descending; `false`, `d` untouched, when dqds did not converge.
pub(super) fn singular_values<R: Float>(d: &mut [R], e: &[R]) -> bool {
    let n = d.len();
    if n == 0 {
        return true;
    }
    if n == 1 {
        d[0] = d[0].abs();
        return true;
    }
    if n == 2 {
        let (min, max) = las2(d[0], e[0], d[1]);
        d[0] = max;
        d[1] = min;
        return true;
    }
    let mut sigmx = e.iter().fold(R::ZERO, |s, &x| s.max(x.abs()));
    if sigmx == R::ZERO {
        for d in d.iter_mut() {
            *d = d.abs();
        }
        d.sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        return true;
    }
    for x in d.iter() {
        sigmx = sigmx.max(x.abs());
    }
    // `q₁, e₁, q₂, …, q_n`, scaled so that the largest square is near the top of the range, then
    // squared.
    let eps = R::EPSILON;
    let safmin = R::MIN_POSITIVE;
    let scale = (eps / safmin).sqrt();
    let mut z = vec![R::ZERO; 4 * n + 1];
    for i in 0..n {
        z[2 * i + 1] = d[i].abs();
        if i + 1 < n {
            z[2 * i + 2] = e[i].abs();
        }
    }
    rescale(&mut z[1..2 * n], sigmx, scale);
    for x in &mut z[1..2 * n] {
        *x = *x * *x;
    }
    z[2 * n] = R::ZERO;
    if !lasq2(n, &mut z) {
        return false;
    }
    for i in 0..n {
        d[i] = z[i + 1].sqrt();
    }
    rescale(d, scale, sigmx);
    true
}

/// `x · (to / from)` without overflow or underflow on the way, as LAPACK's `lascl`.
fn rescale<R: Float>(x: &mut [R], from: R, to: R) {
    let small = R::MIN_POSITIVE;
    let big = R::ONE / small;
    let (mut from, mut to) = (from, to);
    loop {
        let from1 = from * small;
        let (mul, done);
        if from1 == from {
            // `from` is infinite.
            mul = to / from;
            done = true;
        } else {
            let to1 = to / big;
            if to1 == to {
                // `to` is zero or infinite.
                mul = to;
                done = true;
                from = R::ONE;
            } else if from1.abs() > to.abs() && to != R::ZERO {
                mul = small;
                done = false;
                from = from1;
            } else if to1.abs() > from.abs() {
                mul = big;
                done = false;
                to = to1;
            } else {
                mul = to / from;
                done = true;
            }
        }
        for x in x.iter_mut() {
            *x *= mul;
        }
        if done {
            return;
        }
    }
}

/// The singular values `(min, max)` of the upper triangular `[[f, g], [0, h]]`, as LAPACK's
/// `las2`.
fn las2<R: Float>(f: R, g: R, h: R) -> (R, R) {
    let (one, two) = (R::ONE, R::TWO);
    let (fa, ga, ha) = (f.abs(), g.abs(), h.abs());
    let (fhmn, fhmx) = (fa.min(ha), fa.max(ha));
    if fhmn == R::ZERO {
        let max = if fhmx == R::ZERO {
            ga
        } else {
            let (lo, hi) = (fhmx.min(ga), fhmx.max(ga));
            hi * (one + (lo / hi) * (lo / hi)).sqrt()
        };
        return (R::ZERO, max);
    }
    if ga < fhmx {
        let as_ = one + fhmn / fhmx;
        let at = (fhmx - fhmn) / fhmx;
        let au = (ga / fhmx) * (ga / fhmx);
        let c = two / ((as_ * as_ + au).sqrt() + (at * at + au).sqrt());
        (fhmn * c, fhmx / c)
    } else {
        let au = fhmx / ga;
        if au == R::ZERO {
            ((fhmn * fhmx) / ga, ga)
        } else {
            let as_ = one + fhmn / fhmx;
            let at = (fhmx - fhmn) / fhmx;
            let c = one
                / ((one + (as_ * au) * (as_ * au)).sqrt() + (one + (at * au) * (at * au)).sqrt());
            let min = (fhmn * c) * au;
            (min + min, ga / (c + c))
        }
    }
}

/// The state of dqds that LAPACK passes between `lasq2`, `lasq3` and `lasq4`.
struct State<R> {
    /// The first row of the unreduced block.
    i0: usize,
    /// Its last row.
    n0: usize,
    /// Ping (0) or pong (1) half of the array; 2 after a flip, which skips the tests of
    /// deflation once.
    pp: usize,
    /// The smallest `d` of the last transform, or minus the shift to take.
    dmin: R,
    /// The accumulated shift.
    sigma: R,
    /// The lower part of `sigma`.
    desig: R,
    /// The largest `q`.
    qmax: R,
    /// The type of the last shift.
    ttype: i32,
    /// `dmin` without the last element.
    dmin1: R,
    /// `dmin` without the last two.
    dmin2: R,
    /// The last `d`.
    dn: R,
    /// The `d` before.
    dn1: R,
    /// The `d` before that.
    dn2: R,
    /// The factor of the shift of type 6.
    g: R,
    /// The shift.
    tau: R,
    /// The transforms done.
    iter: usize,
}

/// The eigenvalues of the symmetric positive definite tridiagonal matrix whose `L·D·Lᵀ`
/// factorization `z[1..2n]` holds as `q₁, e₁, q₂, …`, as LAPACK's `lasq2`: into `z[1..=n]`,
/// descending. `z` has `4n + 1` elements.
fn lasq2<R: Float>(n: usize, z: &mut [R]) -> bool {
    let (zero, half) = (R::ZERO, R::HALF);
    let four = R::TWO + R::TWO;
    let eps = R::EPSILON * half;
    let safmin = R::MIN_POSITIVE;
    let tol = eps * R::from_f64(100.0);
    let tol2 = tol * tol;
    let cbias = R::from_f64(1.5);
    // Sums of the q's and e's, and diagonality.
    let mut d = zero;
    let mut e = zero;
    let mut k = 1;
    while k <= 2 * (n - 1) {
        if !(z[k] >= zero && z[k + 1] >= zero) {
            return false;
        }
        d += z[k];
        e += z[k + 1];
        k += 2;
    }
    if z[2 * n - 1] < zero || z[2 * n - 1].is_nan() {
        return false;
    }
    d += z[2 * n - 1];
    if e == zero {
        for k in 2..=n {
            z[k] = z[2 * k - 1];
        }
        z[1..=n].sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
        return true;
    }
    if d + e == zero {
        z[1..=n].fill(zero);
        return true;
    }
    // `z = (q₁, qq₁, e₁, ee₁, q₂, …)`.
    let mut k = 2 * n;
    while k >= 2 {
        z[2 * k] = zero;
        z[2 * k - 1] = z[k];
        z[2 * k - 2] = zero;
        z[2 * k - 3] = z[k - 1];
        k -= 2;
    }
    let mut s = State {
        i0: 1,
        n0: n,
        pp: 0,
        dmin: zero,
        sigma: zero,
        desig: zero,
        qmax: zero,
        ttype: 0,
        dmin1: zero,
        dmin2: zero,
        dn: zero,
        dn1: zero,
        dn2: zero,
        g: zero,
        tau: zero,
        iter: 2,
    };
    // Reversed when the end is much larger, so that the small values come out at the bottom.
    if cbias * z[4 * s.i0 - 3] < z[4 * s.n0 - 3] {
        let ipn4 = 4 * (s.i0 + s.n0);
        let mut i4 = 4 * s.i0;
        while i4 <= 2 * (s.i0 + s.n0 - 1) {
            z.swap(i4 - 3, ipn4 - i4 - 3);
            z.swap(i4 - 1, ipn4 - i4 - 5);
            i4 += 4;
        }
    }
    // Two dqd transforms with Li's test of splitting.
    let mut pp = 0;
    for _ in 0..2 {
        let mut d = z[4 * s.n0 + pp - 3];
        let mut i4 = 4 * (s.n0 - 1) + pp;
        while i4 >= 4 * s.i0 + pp {
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -zero;
                d = z[i4 - 3];
            } else {
                d = z[i4 - 3] * (d / (d + z[i4 - 1]));
            }
            i4 -= 4;
        }
        let mut d = z[4 * s.i0 + pp - 3];
        let mut i4 = 4 * s.i0 + pp;
        while i4 <= 4 * (s.n0 - 1) + pp {
            let j = i4 - 2 * pp;
            z[j - 2] = d + z[i4 - 1];
            if z[i4 - 1] <= tol2 * d {
                z[i4 - 1] = -zero;
                z[j - 2] = d;
                z[j] = zero;
                d = z[i4 + 1];
            } else if safmin * z[i4 + 1] < z[j - 2] && safmin * z[j - 2] < z[i4 + 1] {
                let t = z[i4 + 1] / z[j - 2];
                z[j] = z[i4 - 1] * t;
                d *= t;
            } else {
                z[j] = z[i4 + 1] * (z[i4 - 1] / z[j - 2]);
                d = z[i4 + 1] * (d / z[j - 2]);
            }
            i4 += 4;
        }
        z[4 * s.n0 - pp - 2] = d;
        let mut qmax = z[4 * s.i0 - pp - 2];
        let mut i4 = 4 * s.i0 - pp + 2;
        while i4 <= 4 * s.n0 - pp - 2 {
            qmax = qmax.max(z[i4]);
            i4 += 4;
        }
        s.qmax = qmax;
        pp = 1 - pp;
    }
    s.iter = 2;
    for _ in 0..=n {
        if s.n0 < 1 {
            break;
        }
        // `e(n0)` holds minus the shift of the block that split off below.
        s.desig = zero;
        s.sigma = if s.n0 == n { zero } else { -z[4 * s.n0 - 1] };
        if s.sigma < zero {
            return false;
        }
        // The last unreduced block, its largest q and a Gershgorin bound.
        let mut emax = zero;
        let mut qmin = z[4 * s.n0 - 3];
        s.qmax = qmin;
        let mut i4 = 4 * s.n0;
        let mut found = false;
        while i4 >= 8 {
            if z[i4 - 5] <= zero {
                found = true;
                break;
            }
            if qmin >= four * emax {
                qmin = qmin.min(z[i4 - 3]);
                emax = emax.max(z[i4 - 5]);
            }
            s.qmax = s.qmax.max(z[i4 - 7] + z[i4 - 5]);
            i4 -= 4;
        }
        if !found {
            i4 = 4;
        }
        s.i0 = i4 / 4;
        s.pp = 0;
        if s.n0 - s.i0 > 1 {
            let mut dee = z[4 * s.i0 - 3];
            let mut deemin = dee;
            let mut kmin = s.i0;
            let mut i4 = 4 * s.i0 + 1;
            while i4 <= 4 * s.n0 - 3 {
                dee = z[i4] * (dee / (dee + z[i4 - 2]));
                if dee <= deemin {
                    deemin = dee;
                    kmin = i4.div_ceil(4);
                }
                i4 += 4;
            }
            if (kmin - s.i0) * 2 < s.n0 - kmin && deemin <= half * z[4 * s.n0 - 3] {
                let ipn4 = 4 * (s.i0 + s.n0);
                s.pp = 2;
                let mut i4 = 4 * s.i0;
                while i4 <= 2 * (s.i0 + s.n0 - 1) {
                    z.swap(i4 - 3, ipn4 - i4 - 3);
                    z.swap(i4 - 2, ipn4 - i4 - 2);
                    z.swap(i4 - 1, ipn4 - i4 - 5);
                    z.swap(i4, ipn4 - i4 - 4);
                    i4 += 4;
                }
            }
        }
        // Minus the initial shift.
        s.dmin = -zero.max(qmin - R::TWO * qmin.sqrt() * emax.sqrt());
        let nbig = 100 * (s.n0 - s.i0 + 1);
        let mut done = false;
        for _ in 0..nbig {
            if s.i0 > s.n0 {
                done = true;
                break;
            }
            lasq3(&mut s, z, eps, tol2);
            s.pp = 1 - s.pp;
            // Splits, when an e is very small.
            if s.pp == 0
                && s.n0 >= s.i0 + 3
                && (z[4 * s.n0] <= tol2 * s.qmax || z[4 * s.n0 - 1] <= tol2 * s.sigma)
            {
                let mut splt = s.i0 - 1;
                s.qmax = z[4 * s.i0 - 3];
                let mut emin = z[4 * s.i0 - 1];
                let mut oldemn = z[4 * s.i0];
                let mut i4 = 4 * s.i0;
                while i4 <= 4 * (s.n0 - 3) {
                    if z[i4] <= tol2 * z[i4 - 3] || z[i4 - 1] <= tol2 * s.sigma {
                        z[i4 - 1] = -s.sigma;
                        splt = i4 / 4;
                        s.qmax = zero;
                        emin = z[i4 + 3];
                        oldemn = z[i4 + 4];
                    } else {
                        s.qmax = s.qmax.max(z[i4 + 1]);
                        emin = emin.min(z[i4 - 1]);
                        oldemn = oldemn.min(z[i4]);
                    }
                    i4 += 4;
                }
                z[4 * s.n0 - 1] = emin;
                z[4 * s.n0] = oldemn;
                s.i0 = splt + 1;
            }
        }
        if !done && s.i0 <= s.n0 {
            return false;
        }
    }
    if s.n0 >= 1 {
        return false;
    }
    for k in 2..=n {
        z[k] = z[4 * k - 3];
    }
    z[1..=n].sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
    true
}

/// Deflates converged values from the bottom of the block, chooses a shift and takes a dqds
/// transform, as LAPACK's `lasq3`.
fn lasq3<R: Float>(s: &mut State<R>, z: &mut [R], eps: R, tol2: R) {
    let (zero, half, one, two) = (R::ZERO, R::HALF, R::ONE, R::TWO);
    let cbias = R::from_f64(1.5);
    let tol = eps * R::from_f64(100.0);
    let n0in = s.n0;
    // Deflation.
    loop {
        if s.n0 < s.i0 {
            return;
        }
        let nn = 4 * s.n0 + s.pp;
        let mut one_value = s.n0 == s.i0;
        let mut two_values = s.n0 == s.i0 + 1;
        if !one_value && !two_values {
            // `e(n0 - 1)` negligible: one value.
            if !(z[nn - 5] > tol2 * (s.sigma + z[nn - 3])
                && z[nn - 2 * s.pp - 4] > tol2 * z[nn - 7])
            {
                one_value = true;
            } else if !(z[nn - 9] > tol2 * s.sigma && z[nn - 2 * s.pp - 8] > tol2 * z[nn - 11]) {
                // `e(n0 - 2)` negligible: two.
                two_values = true;
            }
        }
        if one_value {
            z[4 * s.n0 - 3] = z[4 * s.n0 + s.pp - 3] + s.sigma;
            s.n0 -= 1;
            continue;
        }
        if two_values {
            if z[nn - 3] > z[nn - 7] {
                z.swap(nn - 3, nn - 7);
            }
            let t = half * ((z[nn - 7] - z[nn - 3]) + z[nn - 5]);
            if z[nn - 5] > z[nn - 3] * tol2 && t != zero {
                let mut u = z[nn - 3] * (z[nn - 5] / t);
                if u <= t {
                    u = z[nn - 3] * (z[nn - 5] / (t * (one + (one + u / t).sqrt())));
                } else {
                    u = z[nn - 3] * (z[nn - 5] / (t + t.sqrt() * (t + u).sqrt()));
                }
                let t = z[nn - 7] + (u + z[nn - 5]);
                z[nn - 3] *= z[nn - 7] / t;
                z[nn - 7] = t;
            }
            z[4 * s.n0 - 7] = z[nn - 7] + s.sigma;
            z[4 * s.n0 - 3] = z[nn - 3] + s.sigma;
            s.n0 -= 2;
            continue;
        }
        break;
    }
    if s.pp == 2 {
        s.pp = 0;
    }
    // Reversed when warranted.
    if (s.dmin <= zero || s.n0 < n0in) && cbias * z[4 * s.i0 + s.pp - 3] < z[4 * s.n0 + s.pp - 3] {
        let ipn4 = 4 * (s.i0 + s.n0);
        let mut j4 = 4 * s.i0;
        while j4 <= 2 * (s.i0 + s.n0 - 1) {
            z.swap(j4 - 3, ipn4 - j4 - 3);
            z.swap(j4 - 2, ipn4 - j4 - 2);
            z.swap(j4 - 1, ipn4 - j4 - 5);
            z.swap(j4, ipn4 - j4 - 4);
            j4 += 4;
        }
        let (i0, n0, pp) = (s.i0, s.n0, s.pp);
        if n0 - i0 <= 4 {
            z[4 * n0 + pp - 1] = z[4 * i0 + pp - 1];
            z[4 * n0 - pp] = z[4 * i0 - pp];
        }
        s.dmin2 = s.dmin2.min(z[4 * n0 + pp - 1]);
        z[4 * n0 + pp - 1] = z[4 * n0 + pp - 1]
            .min(z[4 * i0 + pp - 1])
            .min(z[4 * i0 + pp + 3]);
        z[4 * n0 - pp] = z[4 * n0 - pp].min(z[4 * i0 - pp]).min(z[4 * i0 - pp + 4]);
        s.qmax = s.qmax.max(z[4 * i0 + pp - 3]).max(z[4 * i0 + pp + 1]);
        s.dmin = -zero;
    }
    lasq4(s, z, n0in);
    // dqds until `dmin` is not negative.
    let mut safe = false;
    let mut tries = 0;
    loop {
        lasq5(s, z, eps);
        s.iter += 1;
        tries += 1;
        if s.dmin >= zero && s.dmin1 >= zero {
            break;
        } else if s.dmin < zero
            && s.dmin1 > zero
            && z[4 * (s.n0 - 1) - s.pp] < tol * (s.sigma + s.dn1)
            && s.dn.abs() < tol * s.sigma
        {
            // Convergence hidden by a negative `dn`.
            z[4 * (s.n0 - 1) - s.pp + 2] = zero;
            s.dmin = zero;
            break;
        } else if s.dmin < zero && tries < 30 {
            // The shift too large: another.
            if s.ttype < -22 {
                s.tau = zero;
            } else if s.dmin1 > zero {
                s.tau = (s.tau + s.dmin) * (one - two * eps);
                s.ttype -= 11;
            } else {
                s.tau *= R::from_f64(0.25);
                s.ttype -= 12;
            }
        } else if s.dmin.is_nan() {
            if s.tau == zero {
                safe = true;
                break;
            }
            s.tau = zero;
        } else {
            // Possible underflow.
            safe = true;
            break;
        }
    }
    if safe {
        lasq6(s, z);
        s.iter += 1;
        s.tau = zero;
    }
    if s.tau < s.sigma {
        s.desig += s.tau;
        let t = s.sigma + s.desig;
        s.desig -= t - s.sigma;
        s.sigma = t;
    } else {
        let t = s.sigma + s.tau;
        s.desig = s.sigma + (s.desig - (t - s.tau));
        s.sigma = t;
    }
}

/// Chooses the shift, as LAPACK's `lasq4`.
fn lasq4<R: Float>(s: &mut State<R>, z: &[R], n0in: usize) {
    let (zero, half, one, two) = (R::ZERO, R::HALF, R::ONE, R::TWO);
    let cnst1 = R::from_f64(0.563);
    let cnst2 = R::from_f64(1.01);
    let cnst3 = R::from_f64(1.05);
    let qurtr = R::from_f64(0.25);
    let third = R::from_f64(0.333);
    let hundrd = R::from_f64(100.0);
    if s.dmin <= zero {
        s.tau = -s.dmin;
        s.ttype = -1;
        return;
    }
    let (i0, n0, pp) = (s.i0, s.n0, s.pp);
    let nn = 4 * n0 + pp;
    let (dmin, dmin1, dmin2, dn, dn1, dn2) = (s.dmin, s.dmin1, s.dmin2, s.dn, s.dn1, s.dn2);
    // The shift, set as the cases go so that an early return keeps the safe value.
    let mut t;
    if n0in == n0 {
        if dmin == dn || dmin == dn1 {
            let b1 = z[nn - 3].sqrt() * z[nn - 5].sqrt();
            let b2 = z[nn - 7].sqrt() * z[nn - 9].sqrt();
            let a2 = z[nn - 7] + z[nn - 5];
            if dmin == dn && dmin1 == dn1 {
                // Cases 2 and 3.
                let gap2 = dmin2 - a2 - dmin2 * qurtr;
                let gap1 = if gap2 > zero && gap2 > b2 {
                    a2 - dn - (b2 / gap2) * b2
                } else {
                    a2 - dn - (b1 + b2)
                };
                if gap1 > zero && gap1 > b1 {
                    t = (dn - (b1 / gap1) * b1).max(half * dmin);
                    s.ttype = -2;
                } else {
                    t = zero;
                    if dn > b1 {
                        t = dn - b1;
                    }
                    if a2 > b1 + b2 {
                        t = t.min(a2 - (b1 + b2));
                    }
                    t = t.max(third * dmin);
                    s.ttype = -3;
                }
            } else {
                // Case 4.
                s.ttype = -4;
                t = qurtr * dmin;
                let (gam, mut a2, b2, np);
                if dmin == dn {
                    gam = dn;
                    a2 = zero;
                    if z[nn - 5] > z[nn - 7] {
                        s.tau = t;
                        return;
                    }
                    b2 = z[nn - 5] / z[nn - 7];
                    np = nn as isize - 9;
                } else {
                    let np0 = nn - 2 * pp;
                    gam = dn1;
                    if z[np0 - 4] > z[np0 - 2] {
                        s.tau = t;
                        return;
                    }
                    a2 = z[np0 - 4] / z[np0 - 2];
                    if z[nn - 9] > z[nn - 11] {
                        s.tau = t;
                        return;
                    }
                    b2 = z[nn - 9] / z[nn - 11];
                    np = nn as isize - 13;
                }
                // The contribution to the norm squared from the rows above.
                a2 += b2;
                let Some(a2) = norm_above(z, np, i0, pp, a2, b2) else {
                    s.tau = t;
                    return;
                };
                let a2 = cnst3 * a2;
                // The bound of the residual of the Rayleigh quotient.
                if a2 < cnst1 {
                    t = gam * (one - a2.sqrt()) / (one + a2);
                }
            }
        } else if dmin == dn2 {
            // Case 5.
            s.ttype = -5;
            t = qurtr * dmin;
            let np = nn - 2 * pp;
            let b1 = z[np - 2];
            let mut b2 = z[np - 6];
            let gam = dn2;
            if z[np - 8] > b2 || z[np - 4] > b1 {
                s.tau = t;
                return;
            }
            let mut a2 = (z[np - 8] / b2) * (one + z[np - 4] / b1);
            if n0 - i0 > 2 {
                b2 = z[nn - 13] / z[nn - 15];
                a2 += b2;
                let Some(sum) = norm_above(z, nn as isize - 17, i0, pp, a2, b2) else {
                    s.tau = t;
                    return;
                };
                a2 = cnst3 * sum;
            }
            if a2 < cnst1 {
                t = gam * (one - a2.sqrt()) / (one + a2);
            }
        } else {
            // Case 6: nothing to go by.
            if s.ttype == -6 {
                s.g += third * (one - s.g);
            } else if s.ttype == -18 {
                s.g = qurtr * third;
            } else {
                s.g = qurtr;
            }
            t = s.g * dmin;
            s.ttype = -6;
        }
    } else if n0in == n0 + 1 {
        // One value deflated: `dmin1` and `dn1` for `dmin` and `dn`.
        if dmin1 == dn1 && dmin2 == dn2 {
            // Cases 7 and 8.
            s.ttype = -7;
            t = third * dmin1;
            if z[nn - 5] > z[nn - 7] {
                s.tau = t;
                return;
            }
            let mut b1 = z[nn - 5] / z[nn - 7];
            let mut b2 = b1;
            if b2 != zero {
                let mut i4 = 4 * n0 - 9 + pp;
                while i4 >= 4 * i0 - 1 + pp {
                    let a2 = b1;
                    if z[i4] > z[i4 - 2] {
                        s.tau = t;
                        return;
                    }
                    b1 *= z[i4] / z[i4 - 2];
                    b2 += b1;
                    if hundrd * b1.max(a2) < b2 {
                        break;
                    }
                    if i4 < 4 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            let b2 = (cnst3 * b2).sqrt();
            let a2 = dmin1 / (one + b2 * b2);
            let gap2 = half * dmin2 - a2;
            if gap2 > zero && gap2 > b2 * a2 {
                t = t.max(a2 * (one - cnst2 * a2 * (b2 / gap2) * b2));
            } else {
                t = t.max(a2 * (one - cnst2 * b2));
                s.ttype = -8;
            }
        } else {
            // Case 9.
            t = qurtr * dmin1;
            if dmin1 == dn1 {
                t = half * dmin1;
            }
            s.ttype = -9;
        }
    } else if n0in == n0 + 2 {
        // Two values deflated: `dmin2` and `dn2` for `dmin` and `dn`.
        if dmin2 == dn2 && two * z[nn - 5] < z[nn - 7] {
            // Cases 10 and 11.
            s.ttype = -10;
            t = third * dmin2;
            if z[nn - 5] > z[nn - 7] {
                s.tau = t;
                return;
            }
            let mut b1 = z[nn - 5] / z[nn - 7];
            let mut b2 = b1;
            if b2 != zero {
                let mut i4 = 4 * n0 - 9 + pp;
                while i4 >= 4 * i0 - 1 + pp {
                    if z[i4] > z[i4 - 2] {
                        s.tau = t;
                        return;
                    }
                    b1 *= z[i4] / z[i4 - 2];
                    b2 += b1;
                    if hundrd * b1 < b2 {
                        break;
                    }
                    if i4 < 4 {
                        break;
                    }
                    i4 -= 4;
                }
            }
            let b2 = (cnst3 * b2).sqrt();
            let a2 = dmin2 / (one + b2 * b2);
            let gap2 = z[nn - 7] + z[nn - 9] - z[nn - 11].sqrt() * z[nn - 9].sqrt() - a2;
            if gap2 > zero && gap2 > b2 * a2 {
                t = t.max(a2 * (one - cnst2 * a2 * (b2 / gap2) * b2));
            } else {
                t = t.max(a2 * (one - cnst2 * b2));
            }
        } else {
            t = qurtr * dmin2;
            s.ttype = -11;
        }
    } else {
        // Case 12: more than two deflated, nothing to go by.
        t = zero;
        s.ttype = -12;
    }
    s.tau = t;
}

/// The sum `a2` of the ratios `b2·Π z(i4)/z(i4 - 2)` from `i4 = np` up the block by fours, as
/// the loops of LAPACK's `lasq4` take it, until the terms are small; `None` when a ratio is
/// above one.
fn norm_above<R: Float>(z: &[R], np: isize, i0: usize, pp: usize, a2: R, b2: R) -> Option<R> {
    let (mut a2, mut b2) = (a2, b2);
    let lo = (4 * i0 + pp) as isize - 1;
    let mut i4 = np;
    while i4 >= lo && b2 != R::ZERO {
        let i = i4 as usize;
        let b1 = b2;
        if z[i] > z[i - 2] {
            return None;
        }
        b2 *= z[i] / z[i - 2];
        a2 += b2;
        if R::from_f64(100.0) * b2.max(b1) < a2 || R::from_f64(0.563) < a2 {
            break;
        }
        i4 -= 4;
    }
    Some(a2)
}

/// One dqds transform with the shift `tau`, from the ping half to the pong or back, as LAPACK's
/// `lasq5`.
fn lasq5<R: Float>(s: &mut State<R>, z: &mut [R], eps: R) {
    let (i0, n0, pp) = (s.i0, s.n0, s.pp);
    if n0 < i0 + 2 {
        return;
    }
    let dthresh = eps * (s.sigma + s.tau);
    if s.tau < dthresh * R::HALF {
        s.tau = R::ZERO;
    }
    let tau = s.tau;
    // With no shift, tiny `d`s are set to zero.
    let flush = tau == R::ZERO;
    let j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4] - tau;
    let mut dmin = d;
    s.dmin1 = -z[j4];
    let mut j4 = 4 * i0;
    if pp == 0 {
        while j4 <= 4 * (n0 - 3) {
            z[j4 - 2] = d + z[j4 - 1];
            let t = z[j4 + 1] / z[j4 - 2];
            d = d * t - tau;
            if flush && d < dthresh {
                d = R::ZERO;
            }
            dmin = dmin.min(d);
            z[j4] = z[j4 - 1] * t;
            emin = emin.min(z[j4]);
            j4 += 4;
        }
    } else {
        while j4 <= 4 * (n0 - 3) {
            z[j4 - 3] = d + z[j4];
            let t = z[j4 + 2] / z[j4 - 3];
            d = d * t - tau;
            if flush && d < dthresh {
                d = R::ZERO;
            }
            dmin = dmin.min(d);
            z[j4 - 1] = z[j4] * t;
            emin = emin.min(z[j4 - 1]);
            j4 += 4;
        }
    }
    // The last two steps.
    let dnm2 = d;
    s.dmin2 = dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm2 + z[j4p2];
    z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
    let dnm1 = z[j4p2 + 2] * (dnm2 / z[j4 - 2]) - tau;
    dmin = dmin.min(dnm1);
    s.dmin1 = dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    z[j4 - 2] = dnm1 + z[j4p2];
    z[j4] = z[j4p2 + 2] * (z[j4p2] / z[j4 - 2]);
    let dn = z[j4p2 + 2] * (dnm1 / z[j4 - 2]) - tau;
    dmin = dmin.min(dn);
    z[j4 + 2] = dn;
    z[4 * n0 - pp] = emin;
    s.dmin = dmin;
    s.dn = dn;
    s.dn1 = dnm1;
    s.dn2 = dnm2;
}

/// One dqd transform without a shift, guarded against underflow, as LAPACK's `lasq6`.
fn lasq6<R: Float>(s: &mut State<R>, z: &mut [R]) {
    let (i0, n0, pp) = (s.i0, s.n0, s.pp);
    if n0 < i0 + 2 {
        return;
    }
    let zero = R::ZERO;
    let safmin = R::MIN_POSITIVE;
    let j4 = 4 * i0 + pp - 3;
    let mut emin = z[j4 + 4];
    let mut d = z[j4];
    let mut dmin = d;
    // One step: `q̂ = d + e`, `ê = e·q_next/q̂`, `d = d·q_next/q̂`, at the offsets of the half.
    let step = |z: &mut [R], d: R, qh: usize, e: usize, qn: usize, eh: usize| -> (R, bool) {
        z[qh] = d + z[e];
        if z[qh] == zero {
            z[eh] = zero;
            (z[qn], true)
        } else if safmin * z[qn] < z[qh] && safmin * z[qh] < z[qn] {
            let t = z[qn] / z[qh];
            z[eh] = z[e] * t;
            (d * t, false)
        } else {
            z[eh] = z[qn] * (z[e] / z[qh]);
            (z[qn] * (d / z[qh]), false)
        }
    };
    let mut j4 = 4 * i0;
    while j4 <= 4 * (n0 - 3) {
        let (nd, reset) = if pp == 0 {
            step(z, d, j4 - 2, j4 - 1, j4 + 1, j4)
        } else {
            step(z, d, j4 - 3, j4, j4 + 2, j4 - 1)
        };
        d = nd;
        if reset {
            dmin = d;
            emin = zero;
        }
        dmin = dmin.min(d);
        emin = emin.min(if pp == 0 { z[j4] } else { z[j4 - 1] });
        j4 += 4;
    }
    let dnm2 = d;
    s.dmin2 = dmin;
    let mut j4 = 4 * (n0 - 2) - pp;
    let mut j4p2 = j4 + 2 * pp - 1;
    let (dnm1, reset) = step(z, dnm2, j4 - 2, j4p2, j4p2 + 2, j4);
    if reset {
        dmin = dnm1;
        emin = zero;
    }
    dmin = dmin.min(dnm1);
    s.dmin1 = dmin;
    j4 += 4;
    j4p2 = j4 + 2 * pp - 1;
    let (dn, reset) = step(z, dnm1, j4 - 2, j4p2, j4p2 + 2, j4);
    if reset {
        dmin = dn;
        emin = zero;
    }
    dmin = dmin.min(dn);
    z[j4 + 2] = dn;
    z[4 * n0 - pp] = emin;
    s.dmin = dmin;
    s.dn = dn;
    s.dn1 = dnm1;
    s.dn2 = dnm2;
}

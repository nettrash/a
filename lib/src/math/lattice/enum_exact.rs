//! The enumeration of Schnorr and Euchner in exact integer arithmetic: the fallback of the float
//! search where floats cannot follow the data (Gram–Schmidt norms that span more than `2⁴⁰`
//! within the radius, or coefficients beyond `2⁵⁰`), and the exact value of every leaf the float
//! search finds.
//!
//! On the integral Gram–Schmidt data of [`Gso`](super::enum_gso::Gso), the centre of level `l`
//! is `c_l = C_l / d_{l+1}` with the integer `C_l = λ_{t,l} − Σ_{j>l} x_j·λ_{j,l}`, and for
//! `w = Σ x_j·b_j − t` the partial norm `L_l = d_l·|π_l(w)|²` is an integer that follows from
//! the one above it by
//!
//! ```text
//! L_l = (d_l·L_{l+1} + (x_l·d_{l+1} − C_l)²) / d_{l+1},
//! ```
//!
//! an exact division, since `|π_l(w)|² = |π_{l+1}(w)|² + (x_l − c_l)²·d_{l+1}/d_l`.

use super::enum_search::Levels;
use crate::math::numbers::Number;

/// `L_lo` of the coefficients `x` (one for each level `lo..hi`), exactly.
pub(super) fn value(levels: &Levels, x: &[i64]) -> Number {
    let g = levels.gso;
    let (lo, hi) = (levels.lo, levels.hi);
    let mut v = levels.top();
    for l in (lo..hi).rev() {
        let c = centre(levels, x, l);
        let dev = &(&g.d[l + 1] * x[l - lo]) - &c;
        v = &(&(&g.d[l] * &v) + &(&dev * &dev)) / &g.d[l + 1];
    }
    v
}

/// `C_l` for the coefficients `x` above level `l` (global index).
fn centre(levels: &Levels, x: &[i64], l: usize) -> Number {
    let g = levels.gso;
    let lo = levels.lo;
    let mut c = match &levels.target {
        Some(t) => t.lam[l].clone(),
        None => Number::default(),
    };
    for j in l + 1..levels.hi {
        let xj = x[j - lo];
        if xj != 0 {
            c -= &(&g.lam[j][l] * xj);
        }
    }
    c
}

/// A coefficient as an `i64`.
#[track_caller]
fn small(n: &Number) -> i64 {
    i64::try_from(n).expect("a coefficient of the enumeration is beyond 64 bits")
}

/// Every coefficient vector with `L_lo ≤ bound` (without a target: nonzero, one of each `±x`, the
/// highest nonzero coefficient positive), with its `L_lo`; with `least`, each one found lowers
/// the bound below itself, so the last is the least.
pub(super) fn search(levels: &Levels, bound: &Number, least: bool) -> Vec<(Vec<i64>, Number)> {
    let g = levels.gso;
    let (lo, hi) = (levels.lo, levels.hi);
    let n = hi - lo;
    let mut found = Vec::new();
    if n == 0 {
        return found;
    }
    let symmetric = levels.target.is_none();
    let dlo = &g.d[lo];
    let mut bound = bound.clone();
    // L_l·d_lo ≤ bound·d_l, since |π_l(w)|² ≤ |π_lo(w)|².
    let limits = |bound: &Number| -> Vec<Number> { (lo..hi).map(|l| bound * &g.d[l]).collect() };
    let mut lim = limits(&bound);
    let mut x = vec![0i64; n];
    let mut dx = vec![0i64; n];
    let mut c: Vec<Number> = vec![Number::default(); n];
    let mut part: Vec<Number> = vec![Number::default(); n + 1];
    part[n] = levels.top();
    let mut zero = vec![false; n];

    let enter = |l: usize, x: &mut [i64], dx: &mut [i64], c: &mut [Number], zero: &mut [bool]| {
        c[l] = centre(levels, x, lo + l);
        zero[l] = symmetric && (l + 1 == n || (zero[l + 1] && x[l + 1] == 0));
        if zero[l] {
            x[l] = i64::from(l == 0);
            dx[l] = 1;
        } else {
            let den = &g.d[lo + l + 1];
            let twice = den * 2u8;
            // The nearest integer to C/d, and the side of it the centre is on.
            let r = (&(&c[l] * 2u8) + den).div_floor(&twice);
            x[l] = small(&r);
            dx[l] = if c[l] >= &r * den { 1 } else { -1 };
        }
    };
    let advance = |l: usize, x: &mut [i64], dx: &mut [i64], zero: &[bool]| {
        let next = if zero[l] {
            x[l].checked_add(1)
        } else {
            let step = dx[l];
            dx[l] = -step - step.signum();
            x[l].checked_add(step)
        };
        x[l] = next.expect("a coefficient of the enumeration is beyond 64 bits");
    };

    let mut l = n - 1;
    enter(l, &mut x, &mut dx, &mut c, &mut zero);
    loop {
        let (dl, dl1) = (&g.d[lo + l], &g.d[lo + l + 1]);
        let dev = &(dl1 * x[l]) - &c[l];
        let v = &(&(dl * &part[l + 1]) + &(&dev * &dev)) / dl1;
        if &v * dlo <= lim[l] {
            if l == 0 {
                if least {
                    bound = &v - 1u8;
                    lim = limits(&bound);
                }
                found.push((x.clone(), v));
                advance(0, &mut x, &mut dx, &zero);
                continue;
            }
            part[l] = v;
            l -= 1;
            enter(l, &mut x, &mut dx, &mut c, &mut zero);
            continue;
        }
        l += 1;
        if l == n {
            return found;
        }
        advance(l, &mut x, &mut dx, &zero);
    }
}

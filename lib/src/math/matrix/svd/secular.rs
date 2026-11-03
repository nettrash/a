//! The singular value decomposition of a broken arrow matrix, the merge step of divide and
//! conquer: the roots of the secular equation, as LAPACK's `lasd4`, and the singular vectors
//! from Gu and Eisenstat's recomputed `z`, as `lasd3`.

use crate::math::complex::Float;

/// The most iterations of one root before it is taken as it is.
const MAXIT: usize = 64;

/// The singular values and vectors of the `k × k` matrix `M` with `z` in its first row and
/// `d₁, …, d_{k-1}` on the rest of its diagonal, `0 = d₀ < d₁ < … < d_{k-1}`, the elements of `z`
/// not zero: `σ` ascending into `sigma`, and the left and right singular vectors as the columns
/// of the `k × k` blocks `u` and `v`, element `j` of each in row `pos[j]`.
pub(super) fn arrow<R: Float>(
    d: &[R],
    z: &[R],
    pos: &[usize],
    sigma: &mut [R],
    u: &mut [R],
    v: &mut [R],
) {
    let k = d.len();
    if k == 1 {
        sigma[0] = z[0].abs();
        u[pos[0]] = if z[0] < R::ZERO { -R::ONE } else { R::ONE };
        v[pos[0]] = R::ONE;
        return;
    }
    // The roots for `z` of norm 1 and `ρ = ‖z‖²`.
    let rho = z.iter().map(|&x| x * x).sum::<R>().sqrt();
    let zn: Vec<R> = z.iter().map(|&x| x / rho).collect();
    let rhoinv = R::ONE / (rho * rho);
    // `prod[j + i·k] = d_j² - σ_i²`, as `(d_j - σ_i)·(d_j + σ_i)`, each part to high relative
    // accuracy.
    let mut prod = vec![R::ZERO; k * k];
    for (i, (sigma, prod)) in sigma.iter_mut().zip(prod.chunks_exact_mut(k)).enumerate() {
        *sigma = root(i, d, &zn, rhoinv, prod);
    }
    // `ẑ`, which makes the computed values the exact ones of `D² + ẑ·ẑᵀ`, by Löwner's formula:
    // `ẑ_j² = (d_j² - σ_{k-1}²)·Π_{i<j} (d_j² - σ_i²)/(d_j² - d_i²)·Π_{j≤i<k-1} (d_j² - σ_i²)/(d_j² -
    // d_{i+1}²)`, a column of `prod` at a time.
    let mut zhat = prod[(k - 1) * k..].to_vec();
    for i in 0..k - 1 {
        let col = &prod[i * k..(i + 1) * k];
        let (lo, hi) = (d[i + 1], d[i]);
        for ((x, &p), &dj) in zhat[..=i].iter_mut().zip(&col[..=i]).zip(&d[..=i]) {
            *x *= p / ((dj - lo) * (dj + lo));
        }
        for ((x, &p), &dj) in zhat[i + 1..].iter_mut().zip(&col[i + 1..]).zip(&d[i + 1..]) {
            *x *= p / ((dj - hi) * (dj + hi));
        }
    }
    for (x, &z) in zhat.iter_mut().zip(z) {
        *x = x.abs().sqrt().copysign(z);
    }
    // `v_i = (D² - σ_i²)⁻¹·ẑ` and `u_i = (-1, d_j·v_i(j))`, normalized.
    let mut vc = vec![R::ZERO; k];
    for i in 0..k {
        let col = &prod[i * k..(i + 1) * k];
        for ((v, &z), &p) in vc.iter_mut().zip(&zhat).zip(col) {
            *v = z / p;
        }
        let (dc, dr) = d[1..].as_chunks::<4>();
        let (vcc, vcr) = vc[1..].as_chunks::<4>();
        let (mut su, mut sv) = ([R::ZERO; 4], [R::ZERO; 4]);
        for (v, d) in vcc.iter().zip(dc) {
            for l in 0..4 {
                let u = d[l] * v[l];
                su[l] += u * u;
                sv[l] += v[l] * v[l];
            }
        }
        let mut nu = R::ONE + (su[0] + su[1]) + (su[2] + su[3]);
        let mut nv = vc[0] * vc[0] + (sv[0] + sv[1]) + (sv[2] + sv[3]);
        for (v, &d) in vcr.iter().zip(dr) {
            nu += (d * *v) * (d * *v);
            nv += *v * *v;
        }
        let (nu, nv) = (R::ONE / nu.sqrt(), R::ONE / nv.sqrt());
        let (ucol, vcol) = (&mut u[i * k..(i + 1) * k], &mut v[i * k..(i + 1) * k]);
        ucol[pos[0]] = -nu;
        vcol[pos[0]] = vc[0] * nv;
        for j in 1..k {
            ucol[pos[j]] = d[j] * vc[j] * nu;
            vcol[pos[j]] = vc[j] * nv;
        }
    }
}

/// Root `i` of `f(σ) = 1/ρ + Σ z_j²/((d_j - σ)(d_j + σ))`, `‖z‖ = 1`, in `(d_i, d_{i+1})`, or
/// above `d_{k-1}` for the last, by the middle way of Li's rational interpolation of the two
/// nearest poles, safeguarded by bisection; `prod[j]` set to `(d_j - σ)(d_j + σ)`. The root is
/// held as `σ = d_K + μ` from the nearer pole `d_K`, so that the differences from the poles keep
/// their relative accuracy.
fn root<R: Float>(i: usize, d: &[R], z: &[R], rhoinv: R, prod: &mut [R]) -> R {
    let k = d.len();
    let (two, four) = (R::TWO, R::TWO + R::TWO);
    let eps = R::EPSILON * R::HALF;
    let last = i + 1 == k;
    // The pole `K` the root is held from, and `τ = σ² - d_K²` with a bracket `(lo, hi]`.
    let (origin, mut tau, mut lo, mut hi);
    if !last {
        let (di, dn) = (d[i], d[i + 1]);
        let delsq = (dn - di) * (dn + di);
        let half = delsq * R::HALF;
        // `f` halfway between the poles in `σ²`, with the terms of the other poles in `c`.
        let mut c = rhoinv;
        for j in (0..k).filter(|&j| j != i && j != i + 1) {
            c += z[j] * z[j] / ((d[j] - di) * (d[j] + di) - half);
        }
        let (zi, zn) = (z[i] * z[i], z[i + 1] * z[i + 1]);
        let w = c - zi / half + zn / half;
        if w >= R::ZERO {
            // In the lower half: from `d_i`.
            origin = i;
            (lo, hi) = (R::ZERO, half);
            let a = c * delsq + zi + zn;
            let b = zi * delsq;
            let disc = (a * a - four * b * c).abs().sqrt();
            tau = if a > R::ZERO {
                two * b / (a + disc)
            } else {
                (a - disc) / (two * c)
            };
        } else {
            // In the upper half: from `d_{i+1}`, `τ` negative.
            origin = i + 1;
            (lo, hi) = (-half, R::ZERO);
            let a = c * delsq - zi - zn;
            let b = zn * delsq;
            let disc = (a * a + four * b * c).abs().sqrt();
            tau = if a < R::ZERO {
                two * b / (a - disc)
            } else {
                -(a + disc) / (two * c)
            };
        }
    } else {
        // Above the last pole, at most `ρ` above it in `σ²`.
        let (dm, dn) = (d[k - 2], d[k - 1]);
        let rho = R::ONE / rhoinv;
        let half = rho * R::HALF;
        let mut c = rhoinv;
        for j in 0..k - 2 {
            c += z[j] * z[j] / ((d[j] - dn) * (d[j] + dn) - half);
        }
        let (zm, zn) = (z[k - 2] * z[k - 2], z[k - 1] * z[k - 1]);
        let delsq = (dn - dm) * (dn + dm);
        let w = c + zm / (-delsq - half) - zn / half;
        let a = -c * delsq + zm + zn;
        let b = zn * delsq;
        let disc = (a * a + four * b * c).abs().sqrt();
        let guess = if a < R::ZERO {
            two * b / (disc - a)
        } else {
            (a + disc) / (two * c)
        };
        origin = k - 1;
        if w <= R::ZERO {
            (lo, hi) = (half, rho);
        } else {
            (lo, hi) = (R::ZERO, half);
        }
        tau = guess;
    }
    if !(tau > lo && tau <= hi) {
        tau = (lo + hi) * R::HALF;
    }
    let dk = d[origin];
    let mut mu = tau / (dk + (dk * dk + tau).sqrt());
    // The poles below and above the root that model the two sums.
    let (pa, pb) = if last { (k - 2, k - 1) } else { (i, i + 1) };
    let split = if last { k - 1 } else { i + 1 };
    let mut iter = 0;
    loop {
        // `f` and its derivative, split into the terms of the poles up to `pa` and the rest.
        let (psi, dpsi, ea) = terms(&d[..split], &z[..split], dk, mu, &mut prod[..split]);
        let (phi, dphi, eb) = terms(&d[split..], &z[split..], dk, mu, &mut prod[split..]);
        let w = rhoinv + psi + phi;
        let dw = dpsi + dphi;
        let erretm = R::from_f64(8.0) * (ea + eb) + two * rhoinv + tau.abs() * dw;
        if w.abs() <= eps * erretm || iter >= MAXIT {
            break;
        }
        if w < R::ZERO {
            lo = lo.max(tau);
        } else {
            hi = hi.min(tau);
        }
        // The root of `c + s/(a - η) + S/(b - η)`, which matches `ψ` and `φ` and their
        // derivatives with the poles `a = d_pa² - σ²` and `b = d_pb² - σ²`.
        let (a, b) = (prod[pa], prod[pb]);
        let cc = w - a * dpsi - b * dphi;
        let aa = (a + b) * w - a * b * dw;
        let bb = a * b * w;
        let disc = (aa * aa - four * bb * cc).abs().sqrt();
        let mut eta = if last {
            if aa >= R::ZERO {
                (aa + disc) / (two * cc)
            } else {
                two * bb / (aa - disc)
            }
        } else if cc == R::ZERO {
            bb / aa
        } else if aa <= R::ZERO {
            (aa - disc) / (two * cc)
        } else {
            two * bb / (aa + disc)
        };
        // A step against the slope: Newton's instead.
        if w * eta >= R::ZERO || eta.is_nan() {
            eta = -w / dw;
        }
        let mut next = tau + eta;
        if !(next > lo && next < hi) {
            next = (lo + hi) * R::HALF;
        }
        if next == tau {
            break;
        }
        tau = next;
        mu = tau / (dk + (dk * dk + tau).sqrt());
        iter += 1;
    }
    dk + mu
}

/// `Σ z_j²/p_j`, `Σ (z_j/p_j)²` and `Σ |z_j²/p_j|` for `p_j = (d_j - σ)(d_j + σ)`, `σ = d_K + μ`,
/// the `p_j` into `prod`; in four sums each, which the compiler keeps in vector registers.
#[inline]
fn terms<R: Float>(d: &[R], z: &[R], dk: R, mu: R, prod: &mut [R]) -> (R, R, R) {
    let (dc, dr) = d.as_chunks::<4>();
    let (zc, zr) = z.as_chunks::<4>();
    let (pc, pr) = prod.as_chunks_mut::<4>();
    let (mut f, mut df, mut ab) = ([R::ZERO; 4], [R::ZERO; 4], [R::ZERO; 4]);
    for ((d, z), p) in dc.iter().zip(zc).zip(pc) {
        for l in 0..4 {
            p[l] = ((d[l] - dk) - mu) * ((d[l] + dk) + mu);
            let t = z[l] / p[l];
            f[l] += z[l] * t;
            df[l] += t * t;
            ab[l] += (z[l] * t).abs();
        }
    }
    let sum = |x: [R; 4]| (x[0] + x[1]) + (x[2] + x[3]);
    let (mut f, mut df, mut ab) = (sum(f), sum(df), sum(ab));
    for ((&d, &z), p) in dr.iter().zip(zr).zip(pr) {
        *p = ((d - dk) - mu) * ((d + dk) + mu);
        let t = z / *p;
        f += z * t;
        df += t * t;
        ab += (z * t).abs();
    }
    (f, df, ab)
}

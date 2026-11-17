//! The roots of polynomials of `f64` and `Complex<f64>` coefficients, by the Aberth–Ehrlich
//! iteration in double precision, as Bini's `polzeros` and MPSolve's floating-point stage.
//!
//! The starting points lie on circles whose radii come from the Newton polygon of the moduli of
//! the coefficients (the upper convex hull of the points `(i, log|aᵢ|)`), each circle with as many
//! points as its edge is long. All the approximations are refined at once, each by
//! `zᵢ ← zᵢ − N/(1 − N·Σⱼ 1/(zᵢ − zⱼ))` with `N = p(zᵢ)/p'(zᵢ)`, the new values used at once (the
//! Gauss–Seidel form); an approximation stops when `|p(zᵢ)|` is within the running error bound of
//! Horner's rule, as no step can then make it better. A point outside the unit circle is
//! evaluated through the reversed polynomial at `1/z`, so that neither overflows.
//!
//! An approximation on another is moved off it (Newton's step alone would keep the two together),
//! and one at its rounding with another in its error ball takes a few more steps, so that the
//! repulsion of the iteration separates two at one root. Then the approximations whose inclusion
//! discs meet (near a multiple root, or in an ill-conditioned region where several may have
//! stopped at the rounding of double precision while fewer roots are there) are polished by the
//! iteration in double-double arithmetic (`dd`); a cluster of them within the radius that the
//! rounding of that precision scatters an `m`-fold root's approximations to is taken for one root
//! of the multiplicity `m`, refined as a simple root of the `(m − 1)`-th derivative. Last, the
//! roots of a real polynomial are made pairs of exact conjugates and real numbers, every value
//! still a root to its rounding.

use super::dd;
use crate::math::complex::Complex;

/// The unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;

/// `2^k` for `−1022 ≤ k ≤ 1023`, exactly (`powi` need not be exact: a scaling that is not a
/// power of two would change the polynomial, as its rounding moves a multiple root's roots).
#[inline]
fn pow2(k: i32) -> f64 {
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The most sweeps of the iteration over all the approximations.
const MAX_SWEEPS: usize = 300;

/// The most steps an approximation within the rounding takes while another is in its error ball.
const KEEP: usize = 20;

/// A coefficient: `f64` or `Complex<f64>`.
pub(crate) trait Coef: Copy {
    /// Whether the type is real, so that the roots come in pairs of conjugates.
    const REAL: bool;
    /// The real and the imaginary part.
    fn parts(self) -> (f64, f64);
}

impl Coef for f64 {
    const REAL: bool = true;
    #[inline(always)]
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
    }
}

impl Coef for Complex<f64> {
    const REAL: bool = false;
    #[inline(always)]
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
    }
}

/// A complex number as a pair, for the inner loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct C {
    pub(crate) re: f64,
    pub(crate) im: f64,
}

impl C {
    #[inline(always)]
    pub(crate) const fn new(re: f64, im: f64) -> C {
        C { re, im }
    }
    #[inline(always)]
    pub(crate) fn add(self, o: C) -> C {
        C::new(self.re + o.re, self.im + o.im)
    }
    #[inline(always)]
    pub(crate) fn sub(self, o: C) -> C {
        C::new(self.re - o.re, self.im - o.im)
    }
    #[inline(always)]
    pub(crate) fn mul(self, o: C) -> C {
        C::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
    #[inline(always)]
    fn scale(self, s: f64) -> C {
        C::new(self.re * s, self.im * s)
    }
    #[inline(always)]
    pub(crate) fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
    /// `self / o`, by Smith's algorithm, which neither overflows nor underflows needlessly.
    #[inline]
    pub(crate) fn div(self, o: C) -> C {
        if o.re.abs() >= o.im.abs() {
            let r = o.im / o.re;
            let d = o.re + o.im * r;
            C::new((self.re + self.im * r) / d, (self.im - self.re * r) / d)
        } else {
            let r = o.re / o.im;
            let d = o.re * r + o.im;
            C::new((self.re * r + self.im) / d, (self.im * r - self.re) / d)
        }
    }
    #[inline(always)]
    pub(crate) fn recip(self) -> C {
        C::new(1.0, 0.0).div(self)
    }
}

/// What Horner's rule gives at a point: Newton's correction `p(z)/p'(z)`, whether `|p(z)|` is
/// within the rounding error of its evaluation, and the radius `n·(|p(z)| + e)/|p'(z)|` of a disc
/// about `z` that holds a root (up to the error of `p'`), `e` the bound of that error.
#[derive(Clone, Copy, Debug)]
struct Newton {
    corr: C,
    small: bool,
    rad: f64,
}

/// Newton's correction at `z` for the coefficients `a` (the constant first, `a[n]` not zero),
/// through the reversed polynomial when `|z| > 1`.
#[inline]
fn newton(a: &[C], z: C) -> Newton {
    let n = a.len() - 1;
    let az = z.abs();
    let nf = n as f64;
    if az <= 1.0 {
        let mut p = a[n];
        let mut d = C::new(0.0, 0.0);
        let mut e = p.abs() * 0.5;
        for &c in a[..n].iter().rev() {
            d = d.mul(z).add(p);
            p = p.mul(z).add(c);
            e = e * az + p.abs();
        }
        // The running bound of the error of Horner's rule in complex arithmetic, after Higham's
        // for real arithmetic, with room for the products' two roundings.
        let err = 8.0 * U * e;
        let ap = p.abs();
        let ad = d.abs();
        Newton {
            corr: p.div(d),
            small: ap <= err,
            rad: nf * (ap + err) / ad,
        }
    } else {
        let w = z.recip();
        let aw = 1.0 / az;
        let mut q = a[0];
        let mut d = C::new(0.0, 0.0);
        let mut e = q.abs() * 0.5;
        for &c in &a[1..] {
            d = d.mul(w).add(q);
            q = q.mul(w).add(c);
            e = e * aw + q.abs();
        }
        let err = 8.0 * U * e;
        let aq = q.abs();
        // p/p' = z·q / (n·q − w·q').
        let den = q.scale(nf).sub(w.mul(d));
        let ad = den.abs();
        Newton {
            corr: z.mul(q).div(den),
            small: aq <= err,
            rad: nf * az * (aq + err) / ad,
        }
    }
}

/// The starting points: on the circles of the radii the Newton polygon of `|aᵢ|` gives, as many
/// on each as its edge is long, turned by `2π·i/n + 0.7` as in Bini's `polzeros`, so that none is
/// on the real axis.
fn start(a: &[C]) -> Vec<C> {
    let n = a.len() - 1;
    // The points (i, log2|aᵢ|) of the coefficients that are not zero.
    let pts: Vec<(usize, f64)> = a
        .iter()
        .enumerate()
        .filter(|(_, c)| c.re != 0.0 || c.im != 0.0)
        .map(|(i, c)| (i, c.abs().log2()))
        .collect();
    // The upper convex hull, by Andrew's monotone chain.
    let mut hull: Vec<(usize, f64)> = Vec::with_capacity(pts.len());
    for &p in &pts {
        while hull.len() >= 2 {
            let (i1, y1) = hull[hull.len() - 2];
            let (i2, y2) = hull[hull.len() - 1];
            // Drop the middle point when it is on or below the line from the first to `p`.
            let cross = (i2 - i1) as f64 * (p.1 - y1) - (y2 - y1) * (p.0 - i1) as f64;
            if cross >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(p);
    }
    let mut z = Vec::with_capacity(n);
    let tau = core::f64::consts::TAU;
    for e in hull.windows(2) {
        let (i, yi) = e[0];
        let (j, yj) = e[1];
        let k = j - i;
        let r = ((yi - yj) / k as f64).exp2();
        for t in 0..k {
            let th = tau * t as f64 / k as f64 + tau * i as f64 / n as f64 + 0.7;
            z.push(C::new(r * th.cos(), r * th.sin()));
        }
    }
    z
}

/// The Aberth–Ehrlich iteration on `z` for the coefficients `a`; returns the radii of the discs
/// about the approximations that hold roots, as far as rounding lets them be known.
fn iterate(a: &[C], z: &mut [C]) -> Vec<f64> {
    let frozen = vec![false; z.len()];
    iterate_from(a, z, &frozen)
}

/// The iteration with the approximations of `frozen` held where they are (they still repel the
/// others through the sums).
fn iterate_from(a: &[C], z: &mut [C], frozen: &[bool]) -> Vec<f64> {
    let n = z.len();
    // Approximations beyond 2^±480 in size: the sum by divisions, whose squares would overflow.
    let wide = z.iter().any(|z| !(z.abs() < WIDE && z.abs() > 1.0 / WIDE));
    let mut zr: Vec<f64> = z.iter().map(|z| z.re).collect();
    let mut zi: Vec<f64> = z.iter().map(|z| z.im).collect();
    let mut done = frozen.to_vec();
    let mut kept = vec![0usize; n];
    let mut rad = vec![f64::INFINITY; n];
    let mut wide = wide;
    for _ in 0..MAX_SWEEPS {
        let mut moving = false;
        if !wide {
            wide = (0..n).any(|i| {
                let r = C::new(zr[i], zi[i]).abs();
                !(r < WIDE && r > 1.0 / WIDE)
            });
        }
        for i in 0..n {
            if done[i] {
                continue;
            }
            let zc = C::new(zr[i], zi[i]);
            let mut nw = newton(a, zc);
            rad[i] = nw.rad;
            // Within the rounding error: one more step at most (below), and the approximation is
            // done — unless another approximation is within its error ball, `(|p| + e)/|p'|`:
            // two at one simple root, which only the iteration's repulsion moves apart (a few
            // more steps; an ill-conditioned cluster keeps them close, and they stop).
            if nw.small && kept[i] < KEEP {
                let ball = nw.rad / n as f64;
                let crowded = (0..n).any(|j| j != i && C::new(zr[j], zi[j]).sub(zc).abs() <= ball);
                if crowded {
                    kept[i] += 1;
                    nw.small = false;
                }
            }
            if nw.small {
                done[i] = true;
            }
            moving = true;
            let corr = if nw.corr.is_finite() {
                nw.corr
            } else {
                // p'(z) = 0: a nudge off the critical point.
                C::new(zc.abs().max(1.0) * 1e-7, zc.abs().max(1.0) * 1e-7)
            };
            let s = aberth_sum(&zr, &zi, i, wide);
            if !s.is_finite() {
                // On another approximation (or too near it for the sum): moved off it by a few
                // units of its size, turned by the index, never Newton's step alone, which would
                // keep the two together for good.
                let t = zc.abs().max(f64::MIN_POSITIVE.sqrt()) * 1e-6;
                let th = 0.7 + i as f64;
                zr[i] += t * th.cos();
                zi[i] += t * th.sin();
                done[i] = false;
                continue;
            }
            let den = C::new(1.0, 0.0).sub(corr.mul(s));
            let mut w = corr.div(den);
            if !w.is_finite() {
                w = corr;
            }
            let next = zc.sub(w);
            if nw.small {
                // The last step, kept only if it does not leave the rounding error: near other
                // roots `p'` is small and a step of the noise over it can jump far.
                let after = newton(a, next);
                if !(next.is_finite() && after.small) {
                    continue;
                }
                rad[i] = after.rad;
            }
            zr[i] = next.re;
            zi[i] = next.im;
            // Newton's correction below the rounding of `z`: no step can make it better. (Not
            // Aberth's, which another approximation near it makes small anyway.)
            if corr.abs() <= U * C::new(zr[i], zi[i]).abs() {
                done[i] = true;
            }
        }
        if !moving {
            break;
        }
    }
    for i in 0..n {
        z[i] = C::new(zr[i], zi[i]);
        if !done[i] || !rad[i].is_finite() {
            rad[i] = newton(a, z[i]).rad;
        }
    }
    rad
}

/// The size of approximations beyond which (or below its inverse) the Aberth sum takes the
/// divisions that cannot overflow: `2^480`.
const WIDE: f64 = 3.121_748_550_315_992_2e144;

/// `Σ_{j≠i} 1/(zᵢ − zⱼ)`, in two loops the compiler vectorises, or (`wide`) by Smith's division
/// for each term, for approximations whose differences' squares would overflow or underflow.
#[inline]
fn aberth_sum(zr: &[f64], zi: &[f64], i: usize, wide: bool) -> C {
    let (xr, xi) = (zr[i], zi[i]);
    if wide {
        let mut s = C::new(0.0, 0.0);
        for j in (0..zr.len()).filter(|&j| j != i) {
            let d = C::new(xr - zr[j], xi - zi[j]);
            if d.re == 0.0 && d.im == 0.0 {
                return C::new(f64::INFINITY, 0.0);
            }
            s = s.add(d.recip());
        }
        return s;
    }
    let part = |r: &[f64], m: &[f64]| {
        let mut sr = 0.0;
        let mut si = 0.0;
        for (&yr, &yi) in r.iter().zip(m) {
            let dr = xr - yr;
            let di = xi - yi;
            let q = 1.0 / (dr * dr + di * di);
            sr += dr * q;
            si -= di * q;
        }
        (sr, si)
    };
    let (a, b) = part(&zr[..i], &zi[..i]);
    let (c, d) = part(&zr[i + 1..], &zi[i + 1..]);
    C::new(a + c, b + d)
}

/// Makes the roots of a real polynomial pairs of exact conjugates and real numbers, keeping
/// every value a root to its rounding (for a real polynomial `|p(z̄)| = |p(z)|`): those whose
/// discs reach the real axis and whose real parts are still roots to their rounding become
/// real; each of the others, the farthest from the axis first, stays as it is (its upper one)
/// with its exact conjugate, in the place of the unused approximation nearest that conjugate
/// (its mirror image, when the polynomial is well conditioned: for an ill-conditioned one the
/// approximations need not be mirror images, and averaging far partners would spoil both). One
/// left over is real: Newton's method on the real axis from its real part.
fn conjugates(a: &[C], z: &mut [C], rad: &[f64]) {
    let n = z.len();
    let mut used = vec![false; n];
    let mut out: Vec<C> = Vec::with_capacity(n);
    let conj = |x: C| C::new(x.re, -x.im);
    // Mirror images first: an upper one and the lower one nearest its conjugate, their discs
    // meeting and the mirror nearer than the axis (so that two values of one real root a little
    // above and below the axis are not a pair; a pair from the polishing in double-double whose
    // discs, of double precision, reach the axis is).
    let mut upper: Vec<usize> = (0..n).filter(|&i| z[i].im > 0.0).collect();
    upper.sort_by(|&i, &j| z[j].im.total_cmp(&z[i].im));
    for &i in &upper {
        let t = conj(z[i]);
        let mirror = (0..n)
            .filter(|&j| !used[j] && j != i && z[j].im < 0.0)
            .filter(|&j| {
                let d = z[j].sub(t).abs();
                d <= rad[i] + rad[j] && d < z[i].im
            })
            .min_by(|&j, &k| z[j].sub(t).abs().total_cmp(&z[k].sub(t).abs()));
        if let Some(j) = mirror {
            used[i] = true;
            used[j] = true;
            out.push(z[i]);
            out.push(t);
        }
    }
    // Real ones: the disc reaches the axis and the real part is still a root to its rounding.
    for i in 0..n {
        if !used[i]
            && (z[i].im == 0.0
                || (z[i].im.abs() <= rad[i] && newton(a, C::new(z[i].re, 0.0)).small))
        {
            out.push(C::new(z[i].re, 0.0));
            used[i] = true;
        }
    }
    // The rest, the farthest from the axis first: each as it is with its conjugate, in the place
    // of the unused one nearest that conjugate.
    let mut rest: Vec<usize> = (0..n).filter(|&i| !used[i]).collect();
    rest.sort_by(|&i, &j| z[j].im.abs().total_cmp(&z[i].im.abs()));
    if rest.len() % 2 == 1 {
        // One must be real: the one whose real part is the best root.
        let k = (0..rest.len())
            .min_by(|&x, &y| {
                let v = |i: usize| newton(a, C::new(z[rest[i]].re, 0.0)).rad;
                v(x).total_cmp(&v(y))
            })
            .expect("one at least");
        let i = rest.remove(k);
        used[i] = true;
        out.push(real_root_near(a, z[i].re));
    }
    for &i in &rest {
        if used[i] {
            continue;
        }
        used[i] = true;
        let t = conj(z[i]);
        let partner = rest
            .iter()
            .copied()
            .filter(|&j| !used[j])
            .min_by(|&j, &k| z[j].sub(t).abs().total_cmp(&z[k].sub(t).abs()))
            .expect("an even number left");
        used[partner] = true;
        let u = C::new(z[i].re, z[i].im.abs());
        out.push(u);
        out.push(conj(u));
    }
    debug_assert_eq!(out.len(), n);
    z.copy_from_slice(&out);
}

/// Newton's method on the real axis from `x`, a few steps at most, until the value is within
/// the rounding error of its evaluation.
fn real_root_near(a: &[C], mut x: f64) -> C {
    for _ in 0..64 {
        let nw = newton(a, C::new(x, 0.0));
        if nw.small || !nw.corr.re.is_finite() {
            break;
        }
        let y = x - nw.corr.re;
        if y == x {
            break;
        }
        x = y;
    }
    C::new(x, 0.0)
}

/// The value of the Taylor coefficients `p^(k)(c)/k!` for `k < m` and bounds of their rounding
/// errors.
fn taylor(a: &[C], c: C, m: usize) -> Vec<(C, f64)> {
    let mut b: Vec<C> = a.to_vec();
    let mut s: Vec<f64> = a.iter().map(|x| x.abs()).collect();
    let ac = c.abs();
    let n = a.len() - 1;
    let mut out = Vec::with_capacity(m);
    for k in 0..m.min(n + 1) {
        // Synthetic division by x − c: b becomes the quotient and its value at c the remainder.
        let len = a.len() - k;
        for i in (0..len - 1).rev() {
            let t = b[i + 1].mul(c);
            b[i] = b[i].add(t);
            s[i] += s[i + 1] * ac;
        }
        out.push((b[0], 8.0 * (n + 1) as f64 * U * s[0]));
        b.remove(0);
        s.remove(0);
    }
    out
}

/// Treats the clusters among the approximations `mov`, polished in double-double, whose discs
/// overlap: one whose approximations all lie within the radius the rounding of double-double
/// arithmetic scatters an `m`-fold root's approximations to, `(e₀/|t_m|)^(1/m)` for the bound
/// `e₀` of the rounding of `p` at that precision and the `m`-th Taylor coefficient `t_m` (not
/// vanishing to its rounding), is taken for a root of the multiplicity `m`, at their centroid.
/// (In double precision that radius is so large for high multiplicities that distinct roots far
/// apart would pass; at double-double precision only a multiple root's approximations are so
/// close.)
fn clusters(a: &[C], z: &mut [C], rad: &[f64], mov: &[usize]) {
    let k = mov.len();
    let mut parent: Vec<usize> = (0..k).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    for x in 0..k {
        for y in x + 1..k {
            let (i, j) = (mov[x], mov[y]);
            if z[i].sub(z[j]).abs() <= rad[i] + rad[j] {
                let (u, v) = (find(&mut parent, x), find(&mut parent, y));
                if u != v {
                    parent[u] = v;
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (x, &i) in mov.iter().enumerate() {
        let r = find(&mut parent, x);
        groups[r].push(i);
    }
    for g in groups.iter().filter(|g| g.len() >= 2) {
        let m = g.len();
        let mut best = C::new(0.0, 0.0);
        for &i in g {
            best = best.add(z[i]);
        }
        best = best.scale(1.0 / m as f64);
        let t = taylor(a, best, m + 1);
        if t.len() < m + 1 {
            continue;
        }
        let (tm, em) = t[m];
        // The bound of p's rounding at double-double precision: the double one's times 2^−50.
        let e0 = t[0].1 * DD_OVER_D;
        let spread = (4.0 * e0 / tm.abs()).powf(1.0 / m as f64);
        if !(tm.abs() > 4.0 * em && g.iter().all(|&i| z[i].sub(best).abs() <= 4.0 * spread)) {
            continue;
        }
        // The centre, by Newton's method on p^(m−1)/(m−1)! from the centroid (whose Taylor
        // coefficients at a point are those of p from m − 1 on), as long as it improves.
        for _ in 0..8 {
            let tt = taylor(a, best, m + 1);
            let (v, e) = tt[m - 1];
            if v.abs() <= e {
                break;
            }
            let d = tt[m].0.scale(m as f64);
            let step = v.div(d);
            let next = best.sub(step);
            if !next.is_finite() || next.sub(best).abs() > spread {
                break;
            }
            best = next;
            if step.abs() <= 4.0 * U * best.abs() {
                break;
            }
        }
        for &i in g {
            z[i] = best;
        }
    }
}

/// The unit roundoff of double-double over that of double precision, with room: `2^−50`.
const DD_OVER_D: f64 = 1.0 / (1u64 << 50) as f64;

/// The indices of the approximations whose inclusion discs meet another's.
fn meeting(z: &[C], rad: &[f64]) -> Vec<usize> {
    let n = z.len();
    let mut hit = vec![false; n];
    for i in 0..n {
        for j in i + 1..n {
            if z[i].sub(z[j]).abs() <= rad[i] + rad[j] {
                hit[i] = true;
                hit[j] = true;
            }
        }
    }
    (0..n).filter(|&i| hit[i]).collect()
}

/// The roots of the polynomial of the coefficients `a` (the constant first, the last not zero, at
/// least two), each as often as its multiplicity, ordered by their real and then their imaginary
/// parts.
pub(crate) fn roots<T: Coef>(a: &[T]) -> Vec<Complex<f64>> {
    let zeros = a
        .iter()
        .position(|c| {
            let (r, i) = c.parts();
            r != 0.0 || i != 0.0
        })
        .unwrap_or(0);
    let mut out = vec![Complex::new(0.0, 0.0); zeros];
    let mut c: Vec<C> = a[zeros..]
        .iter()
        .map(|x| {
            let (r, i) = x.parts();
            C::new(r, i)
        })
        .collect();
    let n = c.len() - 1;
    if n == 0 {
        return out;
    }
    // Scale by a power of two, exactly, so that the largest modulus is about 1.
    let big = c.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let e = big.log2().round() as i32;
    if e != 0 {
        let s = pow2(-e.clamp(-1000, 1000));
        for x in &mut c {
            *x = x.scale(s);
        }
    }
    let mut z = if n == 1 {
        vec![C::new(0.0, 0.0).sub(c[0]).div(c[1])]
    } else {
        start(&c)
    };
    if n >= 2 {
        let mut rad = iterate(&c, &mut z);
        // Approximations whose inclusion discs meet: near a multiple root, or in an
        // ill-conditioned region where several may have stopped at the rounding of double
        // precision while fewer roots are there. Past that noise, in double-double, the
        // iteration spreads them over the roots, or brings a multiple root's close together.
        let mov = meeting(&z, &rad);
        if !mov.is_empty() {
            let before: Vec<C> = mov.iter().map(|&i| z[i]).collect();
            let fine = dd::polish(&c, &mut z, &mov);
            // The discs of double-double precision for the clusters (those of double precision
            // join the clusters of nearby multiple roots).
            let mut rdd = rad.clone();
            for (k, &i) in mov.iter().enumerate() {
                let nw = newton(&c, z[i]);
                // A polished value no longer a root to the rounding of double precision (the
                // polishing stopped short): the one before.
                if !(z[i].is_finite() && nw.small) && newton(&c, before[k]).small {
                    z[i] = before[k];
                    rad[i] = newton(&c, z[i]).rad;
                    rdd[i] = rad[i];
                } else {
                    rad[i] = nw.rad;
                    rdd[i] = fine[k];
                }
            }
            clusters(&c, &mut z, &rdd, &mov);
        }
        if T::REAL {
            conjugates(&c, &mut z, &rad);
        }
        if T::REAL {
            for x in &mut z {
                if x.im.abs() <= 0.0 {
                    x.im = 0.0;
                }
            }
        }
    } else if T::REAL {
        z[0].im = 0.0;
    }
    out.extend(z.iter().map(|z| Complex::new(z.re, z.im)));
    out.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
    out
}

/// Approximations of the roots of the polynomial of the coefficients `c` (the constant and the
/// leading coefficient not zero, a degree at least 1), one for each root, without the treatment
/// of conjugates and clusters: the starting points of a refinement at a higher precision. `None`
/// when one is not finite.
pub(crate) fn approximations(c: &[C]) -> Option<Vec<C>> {
    let n = c.len() - 1;
    let big = c.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let e = big.log2().round() as i32;
    let s = pow2(-e.clamp(-1000, 1000));
    let c: Vec<C> = c.iter().map(|x| x.scale(s)).collect();
    let mut z = if n == 1 {
        vec![C::new(0.0, 0.0).sub(c[0]).div(c[1])]
    } else {
        start(&c)
    };
    if n >= 2 {
        iterate(&c, &mut z);
    }
    z.iter().all(|z| z.is_finite()).then_some(z)
}

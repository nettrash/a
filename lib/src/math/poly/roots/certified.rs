//! Certified complex roots of polynomials over the integers to any precision, as MPSolve and
//! Arb's `arb_fmpz_poly_complex_roots` find them.
//!
//! Each square-free factor (Yun's algorithm) is solved on its own. Approximations come from the
//! Aberth–Ehrlich iteration in double precision when the coefficients fit, otherwise from the
//! Newton polygon in `BigComplex`, and are refined by the iteration in `BigComplex` at doubling
//! precisions until they are isolated: about each approximation `z` of a factor `f` of the degree
//! `d`, the disc of the radius `d·|f(z)|/|f'(z)|` holds a root, since `f'/f = Σ 1/(z − rₖ)`; with
//! the values of `f` and `f'` bounded by the errors of their evaluation, and the radius rounded
//! up, `d` such discs that are pairwise disjoint hold one root each. The evaluation is Horner's
//! rule in fixed point on limbs (`fixed`: each truncation of a part counted, carried through
//! `|Re z| + |Im z|`), or, at a point whose parts are of very different sizes, in `BigComplex`
//! with each part of each operation rounded once, so `|fl(f(z)) − f(z)| ≤ γ_{2d+1}·Σ|aᵢ||z|ⁱ`
//! and the derivative's error twice that over `Σ i|aᵢ||z|^(i−1)`. Each root is then refined by
//! Newton's method at doubling precisions, a new disc accepted only inside the old one, until its
//! radius is below `2^−(prec+2)·|z|`. A disc of a real polynomial whose centre moved to the real
//! axis (its radius grown by the distance) still meets no other disc holds a real root, and the
//! conjugate of a disc that meets only one other disc is that disc's root's: those roots are made
//! real and conjugate exactly, and only one of each pair is refined.

use super::aberth::{self, C};
use super::fixed::{Fixed, Pt};
use super::yun;
use crate::math::complex::Complex;
use crate::math::float::{BigComplex, BigFloat, Round};
use crate::math::numbers::Number;
use crate::math::poly::Polynomial;
use core::cmp::Ordering;

/// The precision of the bounds.
const B: u64 = 32;

/// A factor's coefficients as exact floats, with what the error bounds need.
struct Poly {
    /// The coefficients as integers.
    f: Vec<Number>,
    /// The coefficients, exactly.
    a: Vec<BigFloat>,
    /// Their magnitudes, rounded up to `B` bits.
    abs: Vec<BigFloat>,
    /// The factor prepared for evaluations in fixed point.
    fx: Fixed,
    /// The degree.
    d: usize,
}

/// `2^e` at `B` bits.
fn pow2(e: i64) -> BigFloat {
    BigFloat::from_parts(&Number::from(1u8), e, B, Round::Nearest).0
}

/// `x` rounded up (`Round::Up`) to `B` bits.
fn up(x: &BigFloat) -> BigFloat {
    x.prec_round(B, Round::Up).0
}

fn add_up(x: &BigFloat, y: &BigFloat) -> BigFloat {
    x.add_round(y, B, Round::Up).0
}

fn mul_up(x: &BigFloat, y: &BigFloat) -> BigFloat {
    x.mul_round(y, B, Round::Up).0
}

/// The bits of `n`: `⌈log2(n + 1)⌉`.
fn bits(n: usize) -> u64 {
    64 - (n as u64).leading_zeros() as u64
}

impl Poly {
    fn new(f: &[Number]) -> Poly {
        let a: Vec<BigFloat> = f.iter().map(BigFloat::from).collect();
        let abs = a.iter().map(|x| up(&x.abs())).collect();
        Poly {
            f: f.to_vec(),
            d: f.len() - 1,
            a,
            abs,
            fx: Fixed::new(f),
        }
    }

    /// The limbs of an evaluation in fixed point to about `w` bits, and the point's parts, when
    /// they are not too far apart in size for it (a part's limbs, aligned to the other's
    /// fraction, at most a few more than the width).
    fn fixed_point(&self, z: &BigComplex, w: u64) -> Option<(Pt, Pt, usize)> {
        let wl = (w + 72).div_ceil(64) as usize;
        let re = Pt::of_float(z.re());
        let im = Pt::of_float(z.im());
        let f = re.f.max(im.f);
        let len = |p: &Pt| {
            if p.mag.is_empty() {
                0
            } else {
                p.mag.len() + f - p.f
            }
        };
        (len(&re).max(len(&im)) <= wl + 4).then_some((re, im, wl))
    }

    /// `f(z)` and `f'(z)` to about `w` bits: in fixed point (`fixed`) where the parts of `z`
    /// allow, else by Horner's rule in `BigComplex` at the precision `w`.
    fn eval(&self, z: &BigComplex, w: u64) -> (BigComplex, BigComplex) {
        if let Some((re, im, wl)) = self.fixed_point(z, w) {
            let (v, d) = self.fx.ceval(&re, &im, wl);
            return (v.float(), d.float());
        }
        self.eval_big(z, w)
    }

    /// `f(z)` and `f'(z)` by Horner's rule at the precision `w`, each operation rounded to nearest.
    fn eval_big(&self, z: &BigComplex, w: u64) -> (BigComplex, BigComplex) {
        let d = self.d;
        let mut v = BigComplex::from(self.a[d].prec_round(w, Round::Nearest).0);
        let mut dv = BigComplex::zero(w);
        let mut t = BigComplex::zero(w);
        let real = z.im().is_zero();
        for c in self.a[..d].iter().rev() {
            if real {
                t.set_mul_real(&dv, z.re(), Round::Nearest);
            } else {
                t.set_mul(&dv, z, Round::Nearest);
            }
            dv.set_add(&t, &v, Round::Nearest);
            if real {
                t.set_mul_real(&v, z.re(), Round::Nearest);
            } else {
                t.set_mul(&v, z, Round::Nearest);
            }
            v.set_add_real(&t, c, Round::Nearest);
        }
        (v, dv)
    }

    /// Upper bounds of `Σ|aᵢ|tⁱ` and `Σ i|aᵢ|t^(i−1)` for `t ≥ 0`.
    fn sums(&self, t: &BigFloat) -> (BigFloat, BigFloat) {
        let d = self.d;
        let mut s = self.abs[d].clone();
        let mut ds = BigFloat::zero(B);
        for c in self.abs[..d].iter().rev() {
            ds = add_up(&mul_up(&ds, t), &s);
            s = add_up(&mul_up(&s, t), c);
        }
        (s, ds)
    }

    /// The radius, rounded up, of a disc about `z` that holds a root: `d·(|f| + E)/(|f'| − E')`
    /// with `f` and `f'` computed to about `w` bits and `E`, `E'` the bounds of their errors (the
    /// fixed point's own, or `(2d + 2)·2^−w·Σ|aᵢ||z|ⁱ` and `(4d + 4)·2^−w·Σ i|aᵢ||z|^(i−1)` for
    /// `BigComplex`, each part of each operation rounded once); `None` when `|f'|` cannot be told
    /// from 0. Also whether `|f(z)|` is within its error.
    fn radius(&self, z: &BigComplex, w: u64) -> (Option<BigFloat>, bool) {
        let (v, dv, e, de) = self.eval_err(z, w);
        let fv = v.abs_round(B, Round::Up).0;
        let small = fv <= e;
        let num = add_up(&fv, &e);
        let fd = dv.abs_round(B, Round::Down).0;
        let den = fd.sub_round(&de, B, Round::Down).0;
        if den.partial_cmp(&0) != Some(Ordering::Greater) {
            return (None, small);
        }
        let q = num.div_round(&den, B, Round::Up).0;
        (Some(mul_up(&q, &BigFloat::from(self.d as u64))), small)
    }

    /// `f(z)`, `f'(z)` and the bounds `E`, `E'` of their errors (see `radius`), to about
    /// `max(w, 64) + 2·log2 d + 8` bits.
    fn eval_err(&self, z: &BigComplex, w: u64) -> (BigComplex, BigComplex, BigFloat, BigFloat) {
        let w = w.max(64) + 2 * bits(self.d) + 8;
        if let Some((re, im, wl)) = self.fixed_point(z, w) {
            let (v, dv) = self.fx.ceval(&re, &im, wl);
            return (v.float(), dv.float(), v.error(B), dv.error(B));
        }
        {
            let (v, dv) = self.eval_big(z, w);
            let az = up(&z.abs_round(B, Round::Up).0);
            let (s, ds) = self.sums(&az);
            let u = pow2(-(w as i64));
            let n2 = BigFloat::from(2 * self.d as u64 + 2);
            let n4 = BigFloat::from(4 * self.d as u64 + 4);
            let e = mul_up(&mul_up(&s, &n2), &u);
            let de = mul_up(&mul_up(&ds, &n4), &u);
            (v, dv, e, de)
        }
    }
}

/// An upper bound of `|x − y|` for real `x` and `y`.
fn diff_up(x: &BigFloat, y: &BigFloat) -> BigFloat {
    let a = x.sub_round(y, B, Round::Up).0.abs();
    let b = x.sub_round(y, B, Round::Down).0.abs();
    if a >= b { a } else { b }
}

/// A lower bound of `|x − y|` for real `x` and `y`.
fn diff_down(x: &BigFloat, y: &BigFloat) -> BigFloat {
    let a = x.sub_round(y, B, Round::Up).0;
    let b = x.sub_round(y, B, Round::Down).0;
    if a.is_sign_negative() != b.is_sign_negative() || a.is_zero() || b.is_zero() {
        return BigFloat::zero(B);
    }
    let (a, b) = (a.abs(), b.abs());
    if a <= b { a } else { b }
}

/// An upper bound of `|z − w|`.
fn dist_up(z: &BigComplex, w: &BigComplex) -> BigFloat {
    let re = diff_up(z.re(), w.re());
    let im = diff_up(z.im(), w.im());
    BigComplex::new(re, im).abs_round(B, Round::Up).0
}

/// A lower bound of `|z − w|`.
fn dist_down(z: &BigComplex, w: &BigComplex) -> BigFloat {
    let re = diff_down(z.re(), w.re());
    let im = diff_down(z.im(), w.im());
    BigComplex::new(re, im).abs_round(B, Round::Down).0
}

/// A disc: a centre and a radius (an upper bound).
#[derive(Clone, Debug)]
pub(crate) struct Disc {
    pub(crate) z: BigComplex,
    pub(crate) r: BigFloat,
}

impl Disc {
    /// Whether the discs are disjoint, rigorously.
    fn disjoint(&self, o: &Disc) -> bool {
        // A quick test in f64 first, with room for its errors.
        if let (Some(a), Some(b)) = (shadow(&self.z), shadow(&o.z)) {
            let ra = self.r.to_f64_round(Round::Up).0;
            let rb = o.r.to_f64_round(Round::Up).0;
            let d = a.sub(b).abs();
            let slack = 2f64.powi(-48) * (a.abs() + b.abs());
            let rr = (ra + rb) * (1.0 + 2f64.powi(-48));
            if d.is_finite() && rr.is_finite() {
                if d - slack > rr {
                    return true;
                }
                if d + slack < rr {
                    return false;
                }
            }
        }
        dist_down(&self.z, &o.z) > add_up(&self.r, &o.r)
    }
}

/// `z` in double precision when its parts are in the normal range far from its ends.
fn shadow(z: &BigComplex) -> Option<C> {
    let ok = |x: &BigFloat| x.is_zero() || x.exponent().is_some_and(|e| e.abs() < 900);
    if ok(z.re()) && ok(z.im()) {
        Some(C::new(z.re().to_f64(), z.im().to_f64()))
    } else {
        None
    }
}

/// `log2|x|` for a float that is not zero.
fn log2_abs(x: &BigFloat) -> f64 {
    let e = x.exponent().expect("not zero");
    let y = x.mul_round(&pow2(-e), 53, Round::Nearest).0;
    e as f64 + y.to_f64().abs().log2()
}

/// The starting points of the Newton polygon in `BigComplex`, at the precision `w`, for a factor
/// whose coefficients do not fit `f64`.
fn polygon_start(p: &Poly, w: u64) -> Vec<BigComplex> {
    let d = p.d;
    let pts: Vec<(usize, f64)> =
        p.a.iter()
            .enumerate()
            .filter(|(_, c)| !c.is_zero())
            .map(|(i, c)| (i, log2_abs(c)))
            .collect();
    let mut hull: Vec<(usize, f64)> = Vec::new();
    for &q in &pts {
        while hull.len() >= 2 {
            let (i1, y1) = hull[hull.len() - 2];
            let (i2, y2) = hull[hull.len() - 1];
            let cross = (i2 - i1) as f64 * (q.1 - y1) - (y2 - y1) * (q.0 - i1) as f64;
            if cross >= 0.0 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(q);
    }
    let tau = core::f64::consts::TAU;
    let mut z = Vec::with_capacity(d);
    for e in hull.windows(2) {
        let (i, yi) = e[0];
        let (j, yj) = e[1];
        let k = j - i;
        let y = (yi - yj) / k as f64;
        let fl = y.floor();
        let r = BigFloat::from((y - fl).exp2())
            .mul_round(&pow2(fl as i64), w, Round::Nearest)
            .0;
        for t in 0..k {
            let th = tau * t as f64 / k as f64 + tau * i as f64 / d as f64 + 0.7;
            let re = r.mul_round(&BigFloat::from(th.cos()), w, Round::Nearest).0;
            let im = r.mul_round(&BigFloat::from(th.sin()), w, Round::Nearest).0;
            z.push(BigComplex::new(re, im));
        }
    }
    z
}

/// Starting points for the factor `f`: from the iteration in double precision when the
/// coefficients, scaled by a power of two, fit `f64`; else from the Newton polygon.
fn initial(f: &[Number], p: &Poly) -> Vec<BigComplex> {
    let top = p.a.iter().filter_map(|c| c.exponent()).max().unwrap_or(0);
    let scale = pow2(500 - top);
    let mut c = Vec::with_capacity(f.len());
    let mut fits = true;
    for x in &p.a {
        let y = x.mul_round(&scale, 53, Round::Nearest).0.to_f64();
        // The iteration scales the largest to about 1: every other within 2^−1000 of it, so
        // that none underflows there (a coefficient lost would lose roots).
        if !x.is_zero() && x.exponent().is_none_or(|e| e < top - 1000) {
            fits = false;
            break;
        }
        c.push(C::new(y, 0.0));
    }
    if fits
        && let Some(z) = aberth::approximations(&c)
        && z.len() == p.d
        && z.iter().all(|z| z.abs().is_normal())
    {
        return z
            .iter()
            .map(|z| BigComplex::from(Complex::new(z.re, z.im)))
            .collect();
    }
    polygon_start(p, 64)
}

/// `Σ_{j≠i} 1/(zᵢ − zⱼ)`, in double precision from the shadows when they are all in range (a
/// difference of nearby points taken from the floats themselves), at 64 bits otherwise.
fn aberth_sum(z: &[BigComplex], sh: &[Option<C>], i: usize) -> Option<BigComplex> {
    let mut s = C::new(0.0, 0.0);
    let Some(zi) = sh[i] else {
        return slow_sum(z, i);
    };
    for (j, zj) in sh.iter().enumerate() {
        if j == i {
            continue;
        }
        let Some(zj) = zj else {
            return slow_sum(z, i);
        };
        let mut dd = C::new(zi.re - zj.re, zi.im - zj.im);
        if dd.abs() <= 2f64.powi(-30) * zi.abs().max(zj.abs()) {
            let e = z[i].sub_round(&z[j], 64, Round::Nearest).0;
            if e.is_zero() {
                return None;
            }
            // A difference out of the range of f64: the whole sum at 64 bits.
            if shadow(&e).is_none_or(|e| e.abs() < 2f64.powi(-900)) {
                return slow_sum(z, i);
            }
            dd = C::new(e.re().to_f64(), e.im().to_f64());
        }
        // By Smith's division, which cannot overflow where the squares of large differences
        // would (and drop the term).
        let q = dd.recip();
        if q.re.is_finite() && q.im.is_finite() {
            s = C::new(s.re + q.re, s.im + q.im);
        } else {
            return slow_sum(z, i);
        }
    }
    if s.re.is_finite() && s.im.is_finite() {
        Some(BigComplex::from(Complex::new(s.re, s.im)))
    } else {
        slow_sum(z, i)
    }
}

/// The sum at 64 bits, for points outside the range of `f64`; `None` when another point is the
/// same (the iteration must then move it off: Newton's step alone would keep them together).
fn slow_sum(z: &[BigComplex], i: usize) -> Option<BigComplex> {
    let mut s = BigComplex::zero(64);
    let one = BigComplex::from(BigFloat::from(1u8));
    for (j, zj) in z.iter().enumerate() {
        if j == i {
            continue;
        }
        let d = z[i].sub_round(zj, 64, Round::Nearest).0;
        if d.is_zero() {
            return None;
        }
        let q = one.div_round(&d, 64, Round::Nearest).0;
        s = s.add_round(&q, 64, Round::Nearest).0;
    }
    Some(s)
}

/// Moves apart approximations that are equal, which the iteration could not separate.
fn separate(z: &mut [BigComplex]) {
    let mut idx: Vec<usize> = (0..z.len()).collect();
    idx.sort_by(|&a, &b| {
        z[a].re()
            .total_cmp(z[b].re())
            .then(z[a].im().total_cmp(z[b].im()))
    });
    for k in 1..idx.len() {
        let (a, b) = (idx[k - 1], idx[k]);
        if z[a] == z[b] {
            let p = z[b].prec().0.max(z[b].prec().1).max(64);
            let m = z[b].abs_round(B, Round::Nearest).0;
            let m = if m.is_zero() { BigFloat::from(1u8) } else { m };
            let t = m.mul_round(&pow2(-40 - k as i64 % 7), p, Round::Nearest).0;
            let th = 0.3 + k as f64;
            let re = t.mul_round(&BigFloat::from(th.cos()), p, Round::Nearest).0;
            let im = t.mul_round(&BigFloat::from(th.sin()), p, Round::Nearest).0;
            z[b] = z[b]
                .add_round(&BigComplex::new(re, im), p, Round::Nearest)
                .0;
        }
    }
}

/// Sweeps of the Aberth–Ehrlich iteration at the precision `w`, at most `max`, until every
/// approximation is still to its rounding.
fn sweeps(p: &Poly, z: &mut [BigComplex], w: u64, max: usize) {
    let d = z.len();
    let mut sh: Vec<Option<C>> = z.iter().map(shadow).collect();
    let mut done = vec![false; d];
    let ew = w + 2 * bits(d) + 8;
    for _ in 0..max {
        let mut moving = false;
        for i in 0..d {
            if done[i] {
                continue;
            }
            let zi = z[i].prec_round(ew, Round::Nearest).0;
            // One evaluation for the step and the test (`radius`'s precision is ew too).
            let (v, dv, e, _) = p.eval_err(&zi, w);
            if v.is_zero() {
                done[i] = true;
                continue;
            }
            let small = v.abs_round(B, Round::Up).0 <= e;
            if small {
                done[i] = true;
                continue;
            }
            let n = if dv.is_zero() {
                // A critical point: a nudge.
                let t = zi.abs_round(B, Round::Nearest).0;
                let t = t.mul_round(&pow2(-(w as i64) / 2), w, Round::Nearest).0;
                BigComplex::new(t.clone(), t)
            } else {
                v.div_round(&dv, w, Round::Nearest).0
            };
            let Some(sb) = aberth_sum(z, &sh, i) else {
                // On another approximation: moved off it by 2^(−w/2) of its size.
                let t = zi.abs_round(B, Round::Nearest).0;
                let t = if t.is_zero() { BigFloat::from(1u8) } else { t };
                let t = t.mul_round(&pow2(-(w as i64) / 2), w, Round::Nearest).0;
                let th = 0.7 + i as f64;
                let re = t.mul_round(&BigFloat::from(th.cos()), w, Round::Nearest).0;
                let im = t.mul_round(&BigFloat::from(th.sin()), w, Round::Nearest).0;
                z[i] = zi.add_round(&BigComplex::new(re, im), w, Round::Nearest).0;
                sh[i] = shadow(&z[i]);
                done[i] = false;
                moving = true;
                continue;
            };
            let ns = n.mul_round(&sb, w, Round::Nearest).0;
            let one = BigComplex::from(BigFloat::from(1u8));
            let den = one.sub_round(&ns, w, Round::Nearest).0;
            let corr = if den.is_zero() || !den.is_finite() {
                n
            } else {
                n.div_round(&den, w, Round::Nearest).0
            };
            let next = zi.sub_round(&corr, w, Round::Nearest).0;
            if !next.is_finite() {
                done[i] = true;
                continue;
            }
            moving = true;
            // A correction below the precision ends the work on this one.
            let cz = corr.abs_round(B, Round::Up).0;
            let nz = next.abs_round(B, Round::Down).0;
            if cz <= nz.mul_round(&pow2(4 - w as i64), B, Round::Up).0 {
                done[i] = true;
            }
            sh[i] = shadow(&next);
            z[i] = next;
        }
        if !moving {
            break;
        }
    }
}

/// Isolating discs for the roots of the square-free `f` (degree at least 1, `f(0) ≠ 0`): one
/// disc for each root, pairwise disjoint, and the working precision they took.
fn isolate(f: &[Number], p: &Poly) -> (Vec<Disc>, u64) {
    let d = p.d;
    if d == 1 {
        // −a₀/a₁, exactly when it fits, else rounded with a disc of its error.
        let q = p.a[0].div_round(&p.a[1], 128, Round::Nearest).0;
        let z = BigComplex::from(-q);
        let (r, _) = p.radius(&z, 128);
        let r = r.unwrap_or_else(|| BigFloat::infinity(B));
        return (vec![Disc { z, r }], 128);
    }
    let mut z = initial(f, p);
    separate(&mut z);
    let from_f64 = z.iter().all(|z| z.prec() == (53, 53));
    let w: u64 = 64;
    if !from_f64 {
        sweeps(p, &mut z, w, 500);
    }
    isolate_from(p, z, w)
}

/// The isolation from the approximations `z` at the precision `w` on.
fn isolate_from(p: &Poly, mut z: Vec<BigComplex>, mut w: u64) -> (Vec<Disc>, u64) {
    let d = p.d;
    let f = &p.f;
    let mut last: Vec<Vec<usize>> = Vec::new();
    let mut stuck = 0;
    let mut polygon = false;
    loop {
        let mut discs = Vec::with_capacity(d);
        let mut ok = true;
        for zi in &z {
            match p.radius(zi, w) {
                (Some(r), _) => discs.push(Disc { z: zi.clone(), r }),
                (None, _) => {
                    ok = false;
                    break;
                }
            }
        }
        let groups = if ok { overlaps(&discs) } else { Vec::new() };
        if ok && groups.is_empty() {
            return (discs, w);
        }
        w *= 2;
        // A cluster that stayed one through a doubling of the precision converges linearly, as
        // to a multiple root: restart it about the root of the (m−1)-th derivative.
        for g in &groups {
            if g.len() < d && last.contains(g) {
                restart(f, &mut z, g, w);
            }
        }
        // The same groups through several doublings (or every disc in one group): the
        // approximations are not converging, as from a start in double precision that its
        // range spoilt; once, the start again from the Newton polygon at this precision.
        stuck = if ok && groups == last { stuck + 1 } else { 0 };
        if stuck >= 3 && !polygon {
            polygon = true;
            stuck = 0;
            z = polygon_start(p, w);
        }
        last = groups;
        separate(&mut z);
        sweeps(p, &mut z, w, if w <= 256 { 200 } else { 50 });
    }
}

/// The groups (of at least two, sorted) of discs that overlap, joined through their overlaps.
fn overlaps(discs: &[Disc]) -> Vec<Vec<usize>> {
    let n = discs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let mut any = false;
    for i in 0..n {
        for j in i + 1..n {
            if !discs[i].disjoint(&discs[j]) {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
                any = true;
            }
        }
    }
    if !any {
        return Vec::new();
    }
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let r = find(&mut parent, i);
        groups[r].push(i);
    }
    groups.into_iter().filter(|g| g.len() >= 2).collect()
}

/// The coefficients of `f^(k)/k!`.
fn taylor_poly(f: &[Number], k: usize) -> Vec<Number> {
    // C(i, k)·aᵢ for i ≥ k, the binomial carried along i: C(i+1, k) = C(i, k)·(i+1)/(i+1−k).
    let mut out = Vec::with_capacity(f.len() - k);
    let mut c = Number::from(1u8);
    for (i, a) in f.iter().enumerate().skip(k) {
        if i > k {
            c = &(&c * &Number::from(i as u64)) / &Number::from((i - k) as u64);
        }
        out.push(&c * a);
    }
    out
}

/// Restarts the approximations `g` of a cluster of `m` roots of `f`: on a circle about the root
/// `ρ` of `f^(m−1)` near their centroid (Newton's method at the precision `w`), of the radius
/// `|f(ρ)·m!/f^(m)(ρ)|^(1/m)`, the distance of the roots of `f` from `ρ` were they a cluster of
/// the multiplicity `m` about it.
fn restart(f: &[Number], z: &mut [BigComplex], g: &[usize], w: u64) {
    let m = g.len();
    let mut c = BigComplex::zero(w);
    for &i in g {
        c = c.add_round(&z[i], w, Round::Nearest).0;
    }
    let c = c
        .div_real_round(&BigFloat::from(m as u64), w, Round::Nearest)
        .0;
    let gp = Poly::new(&taylor_poly(f, m - 1));
    let mut rho = c;
    for _ in 0..200 {
        let (v, dv) = gp.eval(&rho, w);
        if dv.is_zero() || v.is_zero() {
            break;
        }
        let step = v.div_round(&dv, w, Round::Nearest).0;
        rho = rho.sub_round(&step, w, Round::Nearest).0;
        let a = step.abs_round(B, Round::Up).0;
        let b = rho.abs_round(B, Round::Down).0;
        if a <= b.mul_round(&pow2(8 - w as i64), B, Round::Up).0 {
            break;
        }
    }
    let fp = Poly::new(f);
    let tm = Poly::new(&taylor_poly(f, m));
    let fv = fp.eval(&rho, w).0.abs_round(53, Round::Nearest).0;
    let tv = tm.eval(&rho, w).0.abs_round(53, Round::Nearest).0;
    if fv.is_zero() || tv.is_zero() || !fv.is_finite() || !tv.is_finite() {
        return;
    }
    let l = (log2_abs(&fv) - log2_abs(&tv)) / m as f64;
    if !l.is_finite() {
        return;
    }
    let fl = l.floor();
    let r = BigFloat::from((l - fl).exp2())
        .mul_round(&pow2(fl as i64), w, Round::Nearest)
        .0;
    let tau = core::f64::consts::TAU;
    for (k, &i) in g.iter().enumerate() {
        let th = tau * k as f64 / m as f64 + 0.7;
        let re = r.mul_round(&BigFloat::from(th.cos()), w, Round::Nearest).0;
        let im = r.mul_round(&BigFloat::from(th.sin()), w, Round::Nearest).0;
        z[i] = rho.add_round(&BigComplex::new(re, im), w, Round::Nearest).0;
    }
}

/// Whether the disc `x` meets none of `discs` but those at the indices `skip`.
fn alone(x: &Disc, discs: &[Disc], skip: &[usize]) -> bool {
    discs
        .iter()
        .enumerate()
        .all(|(k, o)| skip.contains(&k) || x.disjoint(o))
}

/// Refines the isolated disc `disc` of a root of `p` by Newton's method until its radius is at
/// most `2^−bits·|z|`, each new disc inside the old one; `w` is the precision the disc was found
/// at. Returns `false` if Newton's method would not make it smaller.
fn refine(p: &Poly, disc: &mut Disc, bits_wanted: u64, mut w: u64) -> bool {
    let extra = 2 * bits(p.d) + 16;
    // The precision Newton's method needs, raised when the error of the evaluation stops it.
    let mut cap = bits_wanted + extra + 8;
    let mut fails = 0;
    loop {
        let az = disc.z.abs_round(B, Round::Down).0;
        let goal = az.mul_round(&pow2(-(bits_wanted as i64)), B, Round::Down).0;
        if disc.r <= goal {
            return true;
        }
        // The bits it has: −log2(r/|z|), and Newton's method doubles them.
        let have = if disc.r.is_zero() || az.is_zero() {
            0
        } else {
            let q = disc.r.div_round(&az, B, Round::Up).0;
            (-q.exponent().unwrap_or(0)).max(0) as u64
        };
        w = w.max((2 * have + extra).min(cap));
        let zi = disc.z.prec_round(w, Round::Nearest).0;
        let (v, dv) = p.eval(&zi, w + extra);
        if dv.is_zero() {
            return false;
        }
        let corr = v.div_round(&dv, w, Round::Nearest).0;
        let mut next = zi.sub_round(&corr, w, Round::Nearest).0;
        if disc.z.im().is_zero() {
            *next.im_mut() = BigFloat::zero(w);
        }
        if let (Some(r), _) = p.radius(&next, w) {
            // Inside the old disc: |next − z| + r ≤ r_old.
            let reach = add_up(&dist_up(&next, &disc.z), &r);
            if reach <= disc.r {
                let halved = r <= disc.r.mul_round(&pow2(-1), B, Round::Down).0;
                disc.z = next;
                disc.r = r;
                fails = 0;
                if !halved {
                    // The rounding error of the evaluation holds the radius: more precision.
                    w *= 2;
                    cap = cap.max(w);
                }
                continue;
            }
        }
        fails += 1;
        if fails >= 6 {
            return false;
        }
        w *= 2;
        cap = cap.max(w);
    }
}

/// A root found: a disc, its multiplicity, and whether it is real or the conjugate of another.
struct Found {
    disc: Disc,
    mult: u32,
    /// The index in its factor's list of the root it is the conjugate of.
    mirror: Option<usize>,
    real: bool,
    /// The poly index and the precision of the disc.
    factor: usize,
    w: u64,
}

/// The complex roots of `p` (not zero) at `prec` bits, certified.
pub(crate) fn complex_roots(p: &Polynomial<Number>, prec: u64) -> Vec<super::ComplexRoot> {
    let (g, k) = yun::strip_zeros(p);
    let mut polys: Vec<Poly> = Vec::new();
    let mut found: Vec<Found> = Vec::new();
    let mut out = Vec::new();
    if g.degree().unwrap_or(0) > 0 {
        for (h, m) in yun::squarefree(&g) {
            let poly = Poly::new(&h.c);
            let (discs, w) = isolate(&h.c, &poly);
            let fi = polys.len();
            let base = found.len();
            let mut discs = discs;
            let marks = classify(&mut discs);
            for (disc, (real, mirror)) in discs.into_iter().zip(marks) {
                found.push(Found {
                    disc,
                    mult: m,
                    mirror: mirror.map(|j| base + j),
                    real,
                    factor: fi,
                    w,
                });
            }
            polys.push(poly);
        }
    }
    // Refine each root that is not a mirror, then the mirrors from them; and refine further any
    // two whose discs, rounded to the output, meet.
    let mut want = vec![prec + 2; found.len()];
    loop {
        let mut stalled = None;
        for i in 0..found.len() {
            if found[i].mirror.is_some() {
                continue;
            }
            let f = &mut found[i];
            let poly = &polys[f.factor];
            if !refine(poly, &mut f.disc, want[i], f.w) {
                stalled = Some(f.factor);
                break;
            }
            f.w = f.w.max(want[i]);
        }
        if let Some(fi) = stalled {
            // Newton's method would not shrink a disc: the factor's isolation again, from its
            // discs' centres, at four times the precision.
            let idx: Vec<usize> = (0..found.len())
                .filter(|&i| found[i].factor == fi)
                .collect();
            let w = 4 * idx.iter().map(|&i| found[i].w).max().unwrap_or(64);
            let z: Vec<BigComplex> = idx.iter().map(|&i| found[i].disc.z.clone()).collect();
            let (mut discs, w) = isolate_from(&polys[fi], z, w);
            let marks = classify(&mut discs);
            let base = idx[0];
            for ((&i, disc), (real, mirror)) in idx.iter().zip(discs).zip(marks) {
                found[i].disc = disc;
                found[i].real = real;
                found[i].mirror = mirror.map(|j| base + j);
                found[i].w = w;
            }
            continue;
        }
        for i in 0..found.len() {
            if let Some(j) = found[i].mirror {
                let d = found[j].disc.clone();
                found[i].disc = Disc {
                    z: d.z.conj(),
                    r: d.r,
                };
            }
        }
        // The values at the output precision and their discs.
        let rounded: Vec<Disc> = found
            .iter()
            .zip(&want)
            .map(|(f, &bw)| {
                let pr = bw.saturating_sub(2).max(prec);
                let v = f.disc.z.prec_round(pr, Round::Nearest).0;
                let r = add_up(&f.disc.r, &dist_up(&v, &f.disc.z));
                Disc { z: v, r }
            })
            .collect();
        // For each root whose rounded disc meets another (or 0): the bits its value needs, from
        // the distance to the nearest it meets (`log2(|z|/distance) + 6`).
        let mut need: Vec<Option<u64>> = vec![None; rounded.len()];
        let mut raise = |i: usize, bits: u64| {
            need[i] = Some(need[i].map_or(bits, |b: u64| b.max(bits)));
        };
        // The root 0, exactly, outside every other disc.
        if k > 0 {
            let zero = Disc {
                z: BigComplex::zero(B),
                r: BigFloat::zero(B),
            };
            for (i, d) in rounded.iter().enumerate() {
                if !d.disjoint(&zero) {
                    raise(i, want[i] + 8);
                }
            }
        }
        let bits_apart = |i: usize, j: usize| -> u64 {
            let (zi, zj) = (&found[i].disc.z, &found[j].disc.z);
            let d = dist_down(zi, zj);
            let (ai, aj) = (zi.abs_round(B, Round::Up).0, zj.abs_round(B, Round::Up).0);
            let m = if ai >= aj { ai } else { aj };
            match (d.exponent(), m.exponent()) {
                (Some(de), Some(me)) if !d.is_zero() => (me - de).max(0) as u64 + 6,
                _ => 0,
            }
        };
        for i in 0..rounded.len() {
            for j in i + 1..rounded.len() {
                if !rounded[i].disjoint(&rounded[j]) {
                    let b = bits_apart(i, j);
                    raise(i, b);
                    raise(j, b);
                }
            }
        }
        let bad: Vec<(usize, u64)> = need
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.map(|b| (i, b)))
            .collect();
        if bad.is_empty() {
            for (f, d) in found.iter().zip(rounded) {
                let mut z = d.z;
                if f.real {
                    *z.im_mut() = BigFloat::zero(z.re().prec());
                }
                out.push(super::ComplexRoot {
                    value: z,
                    radius: d.r,
                    multiplicity: f.mult,
                });
            }
            break;
        }
        // Once a round each: to the bits the nearest neighbour needs, and at least 8 more; a
        // conjugate and the root it mirrors at the same precision, so that their values stay
        // exact conjugates.
        for (i, b) in bad {
            let j = found[i].mirror.unwrap_or(i);
            want[i] = (want[i] + 8).max(b).max(want[j]);
            want[j] = want[i];
        }
        for i in 0..found.len() {
            if let Some(j) = found[i].mirror {
                let w = want[i].max(want[j]);
                want[i] = w;
                want[j] = w;
            }
        }
    }
    if k > 0 {
        out.push(super::ComplexRoot {
            value: BigComplex::zero(prec),
            radius: BigFloat::zero(B),
            multiplicity: k as u32,
        });
    }
    sort(&mut out);
    out
}

/// For each disc of a real polynomial's factor: whether it is a real root (the disc then made
/// symmetric about the axis) and the index of the disc whose conjugate it is.
fn classify(discs: &mut [Disc]) -> Vec<(bool, Option<usize>)> {
    let d = discs.len();
    let mut out = vec![(false, None); d];
    for i in 0..d {
        let z = &discs[i].z;
        // Real: the disc about the real part, grown by |Im z|, meets no other disc.
        let im = z.im().abs();
        if im <= discs[i].r.mul_round(&BigFloat::from(4u8), B, Round::Up).0 || im.is_zero() {
            let grown = Disc {
                z: BigComplex::new(z.re().clone(), BigFloat::zero(z.re().prec())),
                r: add_up(&discs[i].r, &up(&im)),
            };
            if alone(&grown, discs, &[i]) {
                out[i].0 = true;
                discs[i] = grown;
            }
        }
    }
    for i in 0..d {
        if out[i].0 || out[i].1.is_some() || !discs[i].z.im().is_sign_positive() {
            continue;
        }
        let c = Disc {
            z: discs[i].z.conj(),
            r: discs[i].r.clone(),
        };
        // The disc nearest the conjugate.
        let mut best: Option<(usize, f64)> = None;
        for (j, o) in discs.iter().enumerate() {
            if j == i || out[j].0 || out[j].1.is_some() {
                continue;
            }
            let dd = dist_down(&c.z, &o.z).to_f64();
            if best.is_none_or(|(_, b)| dd < b) {
                best = Some((j, dd));
            }
        }
        if let Some((j, _)) = best
            && !c.disjoint(&discs[j])
            && alone(&c, discs, &[j])
        {
            out[j].1 = Some(i);
        }
    }
    for x in &mut out {
        if x.0 {
            x.1 = None;
        }
    }
    out
}

/// Orders the roots: the real ones increasing, then the others by their real and imaginary
/// parts, each with the positive imaginary part before its conjugate.
fn sort(out: &mut [super::ComplexRoot]) {
    let key = |r: &super::ComplexRoot| {
        let im = r.value.im();
        (
            !im.is_zero(),
            r.value.re().clone(),
            im.abs(),
            im.is_sign_negative(),
        )
    };
    out.sort_by(|x, y| {
        let (a, b) = (key(x), key(y));
        a.0.cmp(&b.0)
            .then(a.1.total_cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
            .then(a.3.cmp(&b.3))
            .then(Ordering::Equal)
    });
}

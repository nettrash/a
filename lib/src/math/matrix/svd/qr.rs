//! Golub and Kahan's implicit QR algorithm on a real upper bidiagonal matrix, as LINPACK's
//! `svdc`: the small problems of divide and conquer, and the singular values alone.

use crate::math::complex::Float;

/// Columns of a column-major real block that the rotations act on.
pub(super) struct Columns<'a, R> {
    /// The elements, column `j` from `j · ld`.
    pub data: &'a mut [R],
    /// The rows rotated.
    pub rows: usize,
    /// The leading dimension.
    pub ld: usize,
}

/// The singular vectors [`bidiagonal_qr`] turns: what it does to their columns.
pub(super) trait Vectors<R> {
    /// `(x, y) = (c·x + s·y, c·y - s·x)` on columns `i` and `j`.
    fn rotate(&mut self, i: usize, j: usize, c: R, s: R);
    /// Negates column `j`.
    fn negate(&mut self, j: usize);
    /// Swaps columns `i < j`.
    fn swap(&mut self, i: usize, j: usize);
}

impl<R: Float> Columns<'_, R> {
    /// `(x, y) = (c·x + s·y, c·y - s·x)` on columns `i` and `j`.
    #[inline]
    pub fn rotate(&mut self, i: usize, j: usize, c: R, s: R) {
        let (rows, ld) = (self.rows, self.ld);
        let (lo, hi) = (i.min(j), i.max(j));
        let (left, right) = self.data.split_at_mut(hi * ld);
        let (p, q) = (&mut left[lo * ld..lo * ld + rows], &mut right[..rows]);
        let (x, y) = if i < j { (p, q) } else { (q, p) };
        rotate_real(x, y, c, s);
    }
}

impl<R: Float> Vectors<R> for Columns<'_, R> {
    fn rotate(&mut self, i: usize, j: usize, c: R, s: R) {
        Columns::rotate(self, i, j, c, s);
    }

    fn negate(&mut self, j: usize) {
        for x in &mut self.data[j * self.ld..j * self.ld + self.rows] {
            *x = -*x;
        }
    }

    fn swap(&mut self, i: usize, j: usize) {
        let (rows, ld) = (self.rows, self.ld);
        let (left, right) = self.data.split_at_mut(j * ld);
        left[i * ld..i * ld + rows].swap_with_slice(&mut right[..rows]);
    }
}

/// `(x, y) = (c·x + s·y, c·y - s·x)`. Kept out of line: inlined into the QR sweep, the compiler
/// moves the sweep's next divisions below this loop, and each step of the sweep then waits for
/// the rotation of the vectors instead of running beside it.
#[inline(never)]
fn rotate_real<R: Float>(x: &mut [R], y: &mut [R], c: R, s: R) {
    for (x, y) in x.iter_mut().zip(y.iter_mut()) {
        let (a, b) = (*x, *y);
        *x = fma(c, a, s * b);
        *y = fma(c, b, -(s * a));
    }
}

/// Whether the target fuses multiply-adds in hardware; elsewhere `mul_add` is a library call.
const FMA: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// `a·b + c`, rounded once where the target fuses multiply-adds.
#[inline(always)]
pub(super) fn fma<R: Float>(a: R, b: R, c: R) -> R {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// The plane rotation `(c, s)` with `c·a + s·b = r` and `c·b - s·a = 0`, as BLAS's `rotg`, and
/// `r`: its length from the sum of the squares where neither can have overflowed or
/// underflowed to matter, as it nearly always is, and by `hypot` elsewhere.
#[inline]
pub(super) fn rotg<R: Float>(a: R, b: R) -> (R, R, R) {
    let roe = if a.abs() > b.abs() { a } else { b };
    let ss = a * a + b * b;
    let r = if ss >= R::MIN_POSITIVE / R::EPSILON && ss <= R::MAX {
        ss.sqrt()
    } else {
        a.hypot(b)
    };
    if r == R::ZERO {
        return (R::ZERO, R::ONE, R::ZERO);
    }
    let r = r.copysign(roe);
    (r, a / r, b / r)
}

/// A rotation of the bulge chase and what it makes of the elements it combines.
struct Chase<R> {
    /// The length `r` of `(f, g)`, signed as `rotg`'s.
    r: R,
    /// The cosine.
    cs: R,
    /// The sine.
    sn: R,
    /// `cs·a + sn·b`, the next `f`.
    f: R,
    /// `cs·b - sn·a`.
    b: R,
    /// `sn·x`, the next `g`.
    g: R,
    /// `cs·x`.
    x: R,
}

/// The rotation `rotg(f, g)` applied to `(a, b)` and to `(0, x)`. Each step of a sweep waits on
/// the one before through this, so its products are formed as quotients by the length instead:
/// `(f·a + g·b)/r` needs only `r` from the square root, where `cs·a + sn·b` needs `cs` from a
/// division after it; `rotg` and the products as they are when the length is near an end of
/// the range.
#[inline(always)]
fn chase<R: Float>(f: R, g: R, a: R, b: R, x: R) -> Chase<R> {
    let ss = fma(f, f, g * g);
    if ss >= R::MIN_POSITIVE / R::EPSILON && ss <= R::MAX {
        let roe = if f.abs() > g.abs() { f } else { g };
        let r = ss.sqrt().copysign(roe);
        Chase {
            r,
            f: fma(f, a, g * b) / r,
            g: g * x / r,
            cs: f / r,
            sn: g / r,
            b: fma(f, b, -(g * a)) / r,
            x: f * x / r,
        }
    } else {
        let (r, cs, sn) = rotg(f, g);
        Chase {
            r,
            cs,
            sn,
            f: cs * a + sn * b,
            b: cs * b - sn * a,
            g: sn * x,
            x: cs * x,
        }
    }
}

/// The singular values of the real upper bidiagonal matrix with the diagonal `s` and the
/// superdiagonal `e` (with a zero appended), by Golub and Kahan's implicit QR algorithm as
/// LINPACK's `svdc`: `s` turned into the singular values, descending, and the rotations applied
/// to the columns of `u` and `v` when given. Values it doesn't converge to are NaNs, and then it
/// returns `false`.
pub(super) fn bidiagonal_qr<R: Float, U: Vectors<R>, V: Vectors<R>>(
    s: &mut [R],
    e: &mut [R],
    mut u: Option<&mut U>,
    mut v: Option<&mut V>,
) -> bool {
    let p = s.len();
    let zero = R::ZERO;
    // An element is negligible within a few units in the last place of its neighbours. LINPACK
    // tests `test + |e| == test`, which a sweep's rounding can keep failing: for `f32` values
    // near ±1 without fused multiply-adds, every sweep left `e = 2⁻²⁴`, half a unit of `test`,
    // which rounded up, until the iterations ran out.
    let tol = R::from_f64(4.0) * R::EPSILON;
    // 1-based indices below, as in LINPACK: `s(k)` is `s[k - 1]`.
    let mut m = p;
    let mut iter = 0;
    while m > 0 {
        if iter >= 75 {
            for x in &mut s[..m] {
                *x = R::NAN;
            }
            return false;
        }
        // `l`: the last negligible superdiagonal element above `m`, 0 for none.
        let mut l = m - 1;
        while l > 0 {
            let test = s[l - 1].abs() + s[l].abs();
            if e[l - 1].abs() <= tol * test {
                e[l - 1] = zero;
                break;
            }
            l -= 1;
        }
        let kase;
        if l == m - 1 {
            kase = 4;
        } else {
            let mut ls = m;
            while ls > l {
                let mut test = zero;
                if ls != m {
                    test += e[ls - 1].abs();
                }
                if ls != l + 1 {
                    test += e[ls - 2].abs();
                }
                if s[ls - 1].abs() <= tol * test {
                    s[ls - 1] = zero;
                    break;
                }
                ls -= 1;
            }
            if ls == l {
                kase = 3;
            } else if ls == m {
                kase = 1;
            } else {
                kase = 2;
                l = ls;
            }
        }
        l += 1;
        match kase {
            1 => {
                // `s(m)` is negligible: chase `e(m - 1)` up and out.
                let mut f = e[m - 2];
                e[m - 2] = zero;
                for k in (l..m).rev() {
                    let (r, cs, sn) = rotg(s[k - 1], f);
                    s[k - 1] = r;
                    if k != l {
                        f = -sn * e[k - 2];
                        e[k - 2] = cs * e[k - 2];
                    }
                    if let Some(v) = v.as_deref_mut() {
                        v.rotate(k - 1, m - 1, cs, sn);
                    }
                }
            }
            2 => {
                // `s(l - 1)` is negligible: split there, chasing `e(l - 1)` down.
                let mut f = e[l - 2];
                e[l - 2] = zero;
                for k in l..=m {
                    let (r, cs, sn) = rotg(s[k - 1], f);
                    s[k - 1] = r;
                    f = -sn * e[k - 1];
                    e[k - 1] = cs * e[k - 1];
                    if let Some(u) = u.as_deref_mut() {
                        u.rotate(k - 1, l - 2, cs, sn);
                    }
                }
            }
            3 => {
                // A QR step with the shift from the trailing 2 × 2 block of `Bᵀ·B`.
                let scale = s[m - 1]
                    .abs()
                    .max(s[m - 2].abs())
                    .max(e[m - 2].abs())
                    .max(s[l - 1].abs())
                    .max(e[l - 1].abs());
                let sm = s[m - 1] / scale;
                let smm1 = s[m - 2] / scale;
                let emm1 = e[m - 2] / scale;
                let sl = s[l - 1] / scale;
                let el = e[l - 1] / scale;
                let b = ((smm1 + sm) * (smm1 - sm) + emm1 * emm1) * R::HALF;
                let c = (sm * emm1) * (sm * emm1);
                let mut shift = zero;
                if b != zero || c != zero {
                    shift = (b * b + c).sqrt();
                    if b < zero {
                        shift = -shift;
                    }
                    shift = c / (b + shift);
                }
                let mut f = (sl + sm) * (sl - sm) + shift;
                let mut g = sl * el;
                for k in l..m {
                    let c = chase(f, g, s[k - 1], e[k - 1], s[k]);
                    if k != l {
                        e[k - 2] = c.r;
                    }
                    (f, e[k - 1], g, s[k]) = (c.f, c.b, c.g, c.x);
                    if let Some(v) = v.as_deref_mut() {
                        v.rotate(k - 1, k, c.cs, c.sn);
                    }
                    let c = chase(f, g, e[k - 1], s[k], e[k]);
                    s[k - 1] = c.r;
                    (f, s[k], g, e[k]) = (c.f, c.b, c.g, c.x);
                    if let Some(u) = u.as_deref_mut() {
                        u.rotate(k - 1, k, c.cs, c.sn);
                    }
                }
                e[m - 2] = f;
                iter += 1;
            }
            _ => {
                // `s(l)` has converged: made not negative, and moved down into order.
                if s[l - 1] < zero {
                    s[l - 1] = -s[l - 1];
                    if let Some(v) = v.as_deref_mut() {
                        v.negate(l - 1);
                    }
                }
                let mut l = l;
                while l < p && s[l - 1] < s[l] {
                    s.swap(l - 1, l);
                    if let Some(v) = v.as_deref_mut() {
                        v.swap(l - 1, l);
                    }
                    if let Some(u) = u.as_deref_mut() {
                        u.swap(l - 1, l);
                    }
                    l += 1;
                }
                iter = 0;
                m -= 1;
            }
        }
    }
    true
}

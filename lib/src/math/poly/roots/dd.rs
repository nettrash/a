//! The polishing of double-precision approximations of roots in double-double arithmetic (about
//! 106 bits), for the approximations whose inclusion discs meet: where the polynomial is so ill
//! conditioned that several approximations stop, at the rounding of double precision, in one
//! region of the plane in which the polynomial is within its rounding error of zero, while a
//! root next to it has none. Past that noise the Aberth–Ehrlich iteration spreads them over the
//! roots again. The coefficients are doubles, so exact in double-double; the sums of the Aberth
//! correction are in double precision from the double-double differences.

use super::aberth::C;

/// `a + b` exactly as `(s, e)`.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `a·b` exactly as `(p, e)`: by a fused multiply-add where the hardware has one.
#[cfg(all(any(target_arch = "aarch64", target_feature = "fma"), not(miri)))]
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

/// `a·b` exactly as `(p, e)`, by Dekker's product of Veltkamp's halves (exact in IEEE double
/// arithmetic without a fused multiply-add, which elsewhere is a call into the library and
/// under Miri need not be exact).
#[cfg(not(all(any(target_arch = "aarch64", target_feature = "fma"), not(miri))))]
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let split = |x: f64| {
        let t = 134_217_729.0 * x; // 2^27 + 1
        let hi = t - (t - x);
        (hi, x - hi)
    };
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    let e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    (p, e)
}

/// A double-double: `hi + lo`, `|lo| ≤ ulp(hi)/2`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Dd {
    hi: f64,
    lo: f64,
}

impl Dd {
    const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };

    #[inline(always)]
    fn of(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    #[inline(always)]
    fn add(self, o: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, o.hi);
        let (t, f) = two_sum(self.lo, o.lo);
        let e = e + t;
        let (s, e) = two_sum(s, e);
        let e = e + f;
        let (hi, lo) = two_sum(s, e);
        Dd { hi, lo }
    }

    #[inline(always)]
    fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    #[inline(always)]
    fn sub(self, o: Dd) -> Dd {
        self.add(o.neg())
    }

    #[inline(always)]
    fn mul(self, o: Dd) -> Dd {
        let (p, e) = two_prod(self.hi, o.hi);
        let e = e + (self.hi * o.lo + self.lo * o.hi);
        let (hi, lo) = two_sum(p, e);
        Dd { hi, lo }
    }

    #[inline(always)]
    fn val(self) -> f64 {
        self.hi + self.lo
    }
}

/// A complex double-double.
#[derive(Clone, Copy, Debug)]
struct Cd {
    re: Dd,
    im: Dd,
}

impl Cd {
    #[inline(always)]
    fn of(z: C) -> Cd {
        Cd {
            re: Dd::of(z.re),
            im: Dd::of(z.im),
        }
    }
    #[inline(always)]
    fn add(self, o: Cd) -> Cd {
        Cd {
            re: self.re.add(o.re),
            im: self.im.add(o.im),
        }
    }
    #[inline(always)]
    fn sub(self, o: Cd) -> Cd {
        Cd {
            re: self.re.sub(o.re),
            im: self.im.sub(o.im),
        }
    }
    #[inline(always)]
    fn mul(self, o: Cd) -> Cd {
        Cd {
            re: self.re.mul(o.re).sub(self.im.mul(o.im)),
            im: self.re.mul(o.im).add(self.im.mul(o.re)),
        }
    }
    /// The value in double precision.
    #[inline(always)]
    fn c(self) -> C {
        C::new(self.re.val(), self.im.val())
    }
    /// `self/o` to about double-double precision: one Newton correction of the quotient in
    /// double precision.
    fn div(self, o: Cd) -> Cd {
        let q = Cd::of(self.c().div(o.c()));
        // q + (self − q·o)/o
        let r = self.sub(q.mul(o));
        q.add(Cd::of(r.c().div(o.c())))
    }
}

/// The unit roundoff of double-double arithmetic, with room.
const UDD: f64 = 1.0 / (1u128 << 104) as f64;

/// Newton's correction `p(z)/p'(z)` in double-double, whether `|p(z)|` is within the bound of
/// its rounding, and the radius `n·(|p| + e)/|p'|` of a disc about `z` that holds a root, through
/// the reversed polynomial outside the unit circle (as `aberth::newton`).
fn newton(a: &[C], z: Cd) -> (Cd, bool, f64) {
    let n = a.len() - 1;
    let az = z.c().abs();
    if az <= 1.0 {
        let mut p = Cd::of(a[n]);
        let mut d = Cd {
            re: Dd::ZERO,
            im: Dd::ZERO,
        };
        let mut e = a[n].abs() * 0.5;
        for &c in a[..n].iter().rev() {
            d = d.mul(z).add(p);
            p = p.mul(z).add(Cd::of(c));
            e = e * az + p.c().abs();
        }
        let err = 16.0 * UDD * e;
        let ap = p.c().abs();
        (p.div(d), ap <= err, n as f64 * (ap + err) / d.c().abs())
    } else {
        let w = Cd::of(C::new(1.0, 0.0)).div(z);
        let aw = 1.0 / az;
        let mut q = Cd::of(a[0]);
        let mut d = Cd {
            re: Dd::ZERO,
            im: Dd::ZERO,
        };
        let mut e = a[0].abs() * 0.5;
        for &c in &a[1..] {
            d = d.mul(w).add(q);
            q = q.mul(w).add(Cd::of(c));
            e = e * aw + q.c().abs();
        }
        // p/p' = z·q / (n·q − w·q').
        let nq = Cd {
            re: q.re.mul(Dd::of(n as f64)),
            im: q.im.mul(Dd::of(n as f64)),
        };
        let den = nq.sub(w.mul(d));
        let err = 16.0 * UDD * e;
        let aq = q.c().abs();
        (
            z.mul(q).div(den),
            aq <= err,
            n as f64 * az * (aq + err) / den.c().abs(),
        )
    }
}

/// The most sweeps of the polishing.
const SWEEPS: usize = 200;

/// Polishes the approximations `z` of the roots of `a` whose indices are in `mov` by the
/// Aberth–Ehrlich iteration in double-double, the others held (they still repel); the values
/// are then rounded back to double precision. Returns the radii of the discs about them that
/// hold roots, at double-double precision.
pub(crate) fn polish(a: &[C], z: &mut [C], mov: &[usize]) -> Vec<f64> {
    let n = z.len();
    let mut w: Vec<Cd> = z.iter().map(|&x| Cd::of(x)).collect();
    let mut done = vec![true; n];
    for &i in mov {
        done[i] = false;
    }
    for _ in 0..SWEEPS {
        let mut moving = false;
        for &i in mov {
            if done[i] {
                continue;
            }
            moving = true;
            let (corr, small, _) = newton(a, w[i]);
            if small {
                done[i] = true;
            }
            let cc = corr.c();
            if !(cc.re.is_finite() && cc.im.is_finite()) {
                done[i] = true;
                continue;
            }
            // Σ 1/(zᵢ − zⱼ) in double precision from the double-double differences.
            let mut s = C::new(0.0, 0.0);
            let mut on = false;
            for j in 0..n {
                if j == i {
                    continue;
                }
                let d = w[i].sub(w[j]).c();
                if d.re == 0.0 && d.im == 0.0 {
                    on = true;
                    break;
                }
                let q = d.recip();
                s = C::new(s.re + q.re, s.im + q.im);
            }
            if on || !(s.re.is_finite() && s.im.is_finite()) {
                // On another one: moved off by a few units of double precision.
                let t = w[i].c().abs().max(f64::MIN_POSITIVE) * 1e-12;
                let th = 0.7 + i as f64;
                w[i] = w[i].add(Cd::of(C::new(t * th.cos(), t * th.sin())));
                done[i] = false;
                continue;
            }
            let den = C::new(1.0, 0.0).sub(cc.mul(s));
            let step = if den.re == 0.0 && den.im == 0.0 {
                corr
            } else {
                corr.div(Cd::of(den))
            };
            let next = w[i].sub(step);
            if !(next.c().re.is_finite() && next.c().im.is_finite()) {
                done[i] = true;
                continue;
            }
            w[i] = next;
            if step.c().abs() <= UDD * w[i].c().abs() {
                done[i] = true;
            }
        }
        if !moving {
            break;
        }
    }
    for &i in mov {
        z[i] = w[i].c();
    }
    mov.iter().map(|&i| newton(a, w[i]).2).collect()
}

//! The exponential, the logarithm, the square root and the powers of quaternions.
//!
//! A quaternion `q = w + v` with the vector part `v = r·u`, `r = |v|` and `u` a unit vector,
//! multiplies like the complex number `w + r·i` with `u` in the place of `i`, as `u² = −1`; each
//! function is that of the complex number, with `u` for `i`: `e^q = e^w·(cos r + u·sin r)` and
//! `ln q = ln|q| + u·atan2(r, w)`. Where `v = 0` there is no `u`, and the functions are those of
//! the complex number `w + x·i`, which keeps the signs of zeros as
//! [`Complex`](crate::math::complex::Complex) does.

use super::Quaternion;
use super::scale::{direction, exponent, ldexp};
use crate::math::complex::Float;

impl<T: Float> Quaternion<T> {
    /// `e^self = e^w·(cos r + u·sin r)`, for the vector part `v = r·u`; the vector part is `v`
    /// times `sin(r)/r`, which is 1 to the last bit for a vector part too small for the sine to
    /// differ from its argument, so that nothing cancels near the identity. A quaternion with a
    /// zero vector part keeps it, with the signs of its zeros: `e^w + v`. Where `e^w` overflows
    /// but the parts may not, `e^w·t` is `e^(w/2)·t·e^(w/2)`, and a zero part stays zero. A
    /// vector part that is infinite, or whose length overflows, gives NaN; one longer than about
    /// `1/ε` gives a result whose direction the precision does not determine, as `r` is then
    /// rounded by more than `2π`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    /// use std::f64::consts::PI;
    ///
    /// // e^(π·u) = −1 for every unit vector u.
    /// let q = Quaternion::new(0.0, 0.0, PI, 0.0).exp();
    /// assert!((q - -Quaternion::ONE).norm() < 1e-15);
    /// assert_eq!(Quaternion::new(1.0, 0.0, -0.0, 0.0).exp(), Quaternion::new(1f64.exp(), 0.0, -0.0, 0.0));
    /// ```
    pub fn exp(self) -> Self {
        let e = self.w.exp();
        let Some((_, r)) = direction(self.vector()) else {
            return Quaternion::new(e, self.x, self.y, self.z);
        };
        let (sin, cos) = r.sin_cos();
        let f = sin / r;
        let p = Quaternion::new(cos, self.x * f, self.y * f, self.z * f);
        if e.is_infinite() {
            return p.times_exp_cold(self.w);
        }
        p * e
    }

    /// The parts times `e^w`, which overflows: by `e^(w/2)` twice, a zero part kept as it is
    /// rather than made NaN by `0·∞`.
    #[cold]
    #[inline(never)]
    fn times_exp_cold(self, w: T) -> Self {
        let h = (w * T::HALF).exp();
        let g = |t: T| if t == T::ZERO { t } else { t * h * h };
        Quaternion::new(g(self.w), g(self.x), g(self.y), g(self.z))
    }

    /// The principal natural logarithm, `ln|self| + u·θ` with `θ = atan2(r, w)` in `[0, π]`, for
    /// the vector part `v = r·u`: the one whose vector part has length at most `π`.
    ///
    /// The real part is `ln|self|` as [`Complex::ln`](crate::math::complex::Complex::ln) takes
    /// it: `ln(1 + (|q|² − 1))/2` from `(big − 1)(big + 1)` and the squares of the other parts
    /// where the largest part `big` is in `(1/2, 2)`, so that it keeps its digits near the unit
    /// sphere; `ln(big) + ln(1 + Σ(other/big)²)/2` where `|q|` overflows or is subnormal; `ln|q|`
    /// elsewhere. The vector part is `v·(θ/r)` from `v` and `w` scaled by the power of two that
    /// brings the largest part of `v` into `[1, 2)`, and `v/w` when `r/w < √ε/2`, where
    /// `atan(r/w)·w/r` is 1 to the last bit; so it keeps its digits for vector parts of any size.
    ///
    /// Where the vector part is zero the logarithm is that of the complex number `w + x·i`, with
    /// the zeros of `j` and `k` kept: `ln(−1) = π·i` and `ln(−1 − 0·i) = −π·i`. The logarithm of
    /// zero has the real part `−∞`. An infinite part gives the real part `+∞`; for `w = −∞` beside
    /// a finite nonzero vector part `v` the vector part is `π·v/|v|`, the limit, as
    /// `ln(−MAX + j) = 709.78… + π·j` is beside it (`ln(−∞ + j) = ∞ + π·j`); otherwise it is the
    /// vector part of the logarithm of the direction of the infinite parts (their signs as `±1`,
    /// the finite parts `±0`), so that `ln(−∞ + 0·i) = ∞ + π·i` and `ln(+∞ + v) = ∞ + 0·v`. A NaN
    /// part gives NaN in every part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    /// use std::f64::consts::PI;
    ///
    /// let q = Quaternion::new(1.0, 2.0, -3.0, 0.5);
    /// assert!((q.ln().exp() - q).norm() < 1e-15 * q.norm());
    /// assert_eq!(Quaternion::new(-1.0, 0.0, 0.0, 0.0).ln(), Quaternion::new(0.0, PI, 0.0, 0.0));
    /// // A tiny vector part keeps its digits.
    /// let ln = Quaternion::new(1.0, 1e-300, 0.0, 0.0).ln();
    /// assert_eq!(ln.x, 1e-300);
    /// ```
    pub fn ln(self) -> Self {
        if self.is_nan() {
            return Self::NAN;
        }
        let [x, y, z] = self.ln_vector();
        Quaternion::new(self.ln_norm(), x, y, z)
    }

    /// `ln|self|`, without overflow or underflow, and with its digits near the unit sphere.
    fn ln_norm(self) -> T {
        let a = [self.w.abs(), self.x.abs(), self.y.abs(), self.z.abs()];
        let mut i = 0;
        for j in 1..4 {
            if a[j] > a[i] {
                i = j;
            }
        }
        let big = a[i];
        let mut small = [T::ZERO; 3];
        for (s, j) in small.iter_mut().zip((0..4).filter(|&j| j != i)) {
            *s = a[j];
        }
        if big > T::HALF && big < T::TWO {
            let s: T = small.iter().map(|&t| t * t).sum();
            return ((big - T::ONE) * (big + T::ONE) + s).ln_1p() * T::HALF;
        }
        let m = self.norm();
        if (m.is_finite() && m >= T::MIN_POSITIVE) || big == T::ZERO || !big.is_finite() {
            return m.ln();
        }
        let s: T = small.iter().map(|&t| (t / big) * (t / big)).sum();
        big.ln() + s.ln_1p() * T::HALF
    }

    /// The vector part of the logarithm, `u·atan2(r, w)`: of the direction of the infinite parts
    /// for an infinite quaternion, and the complex logarithm's argument for a zero vector part.
    pub(super) fn ln_vector(self) -> [T; 3] {
        let v = self.vector();
        let finite_v = v.iter().all(|t| t.is_finite());
        if self.w == -T::INFINITY && finite_v {
            // The limit along w → −∞: the angle π about the direction of v, which is kept.
            if let Some(([x, y, z], _)) = direction(v) {
                return [x * T::PI, y * T::PI, z * T::PI];
            }
        }
        let q = if self.is_finite() {
            self
        } else {
            self.infinite_direction()
        };
        let v = q.vector();
        let [vx, vy, vz] = v;
        if vx == T::ZERO && vy == T::ZERO && vz == T::ZERO {
            return [vx.atan2(q.w), vy, vz];
        }
        let k = exponent(vx.abs().max(vy.abs()).max(vz.abs()));
        let (sx, sy, sz) = (ldexp(vx, -k), ldexp(vy, -k), ldexp(vz, -k));
        let w = ldexp(q.w, -k);
        let r = (sx * sx + sy * sy + sz * sz).sqrt();
        if w > T::ZERO && r < w * (T::EPSILON.sqrt() * T::HALF) {
            // atan(t)/t = 1 − t²/3 + …, which rounds to 1 for t below √ε/2.
            return [vx / q.w, vy / q.w, vz / q.w];
        }
        let f = r.atan2(w) / r;
        [sx * f, sy * f, sz * f]
    }

    /// The principal square root, the one whose real part is not negative: for `w ≥ 0`,
    /// `a = √((|q| + w)/2)` and the root `a + v/(2a)`; for `w < 0`, `b = √((|q| − w)/2)` and the
    /// root `r/(2b) + u·b`, for the vector part `v = r·u`, as Kahan takes the complex square root,
    /// so that nothing cancels. The quaternion is first scaled by a power of four where its parts
    /// are near the subnormals or near overflow. Where the vector part is zero the root is that
    /// of the complex number `w + x·i`: `√(−4) = 2i` and `√(−4 − 0·i) = −2i`. `√(±0) = +0`, with
    /// the zeros of the vector part kept. With an infinite part (and no NaN) the root has the real
    /// part `+∞` and the vector part's infinities, its finite parts made zeros of their signs;
    /// but for `w = −∞` and a finite vector part, the real part `+0` and `±∞` for each nonzero
    /// part of the vector, `±∞·i` for a zero one. A NaN part gives NaN in every part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(-3.0, 0.0, 4.0, 0.0);
    /// assert_eq!(q.sqrt(), Quaternion::new(1.0, 0.0, 2.0, 0.0));
    /// assert_eq!(Quaternion::new(-4.0, 0.0, 0.0, 0.0).sqrt(), Quaternion::new(0.0, 2.0, 0.0, 0.0));
    /// let p = Quaternion::new(1.0, 2.0, -3.0, 0.5);
    /// assert!((p.sqrt() * p.sqrt() - p).norm() < 1e-15 * p.norm());
    /// ```
    pub fn sqrt(self) -> Self {
        if self.is_nan() {
            return Self::NAN;
        }
        if !self.is_finite() {
            return self.sqrt_infinite();
        }
        if self.is_zero() {
            return Quaternion::new(T::ZERO, self.x, self.y, self.z);
        }
        let big = self.max_abs();
        if big > T::MAX / T::from_f64(4.0) {
            return (self * T::from_f64(0.25)).sqrt() * T::TWO;
        }
        if big < T::from_f64(4.0) * T::MIN_POSITIVE {
            // Near the subnormals: scaled up by 2^64, exactly, and the root back by 2^32.
            let up = T::from_f64(18_446_744_073_709_551_616.0);
            return (self * up).sqrt() / T::from_f64(4_294_967_296.0);
        }
        let n = self.norm();
        if self.w >= T::ZERO {
            let a = ((n + self.w) * T::HALF).sqrt();
            let d = T::TWO * a;
            return Quaternion::new(a, self.x / d, self.y / d, self.z / d);
        }
        let b = ((n - self.w) * T::HALF).sqrt();
        match direction(self.vector()) {
            None => Quaternion::new(T::ZERO, b.copysign(self.x), self.y, self.z),
            Some(([ux, uy, uz], r)) => Quaternion::new(r / (T::TWO * b), ux * b, uy * b, uz * b),
        }
    }

    #[cold]
    #[inline(never)]
    fn sqrt_infinite(self) -> Self {
        let inf = T::INFINITY;
        if self.x.is_infinite() || self.y.is_infinite() || self.z.is_infinite() || self.w > T::ZERO
        {
            let f = |t: T| {
                if t.is_infinite() {
                    t
                } else {
                    T::ZERO.copysign(t)
                }
            };
            return Quaternion::new(inf, f(self.x), f(self.y), f(self.z));
        }
        if self.x == T::ZERO && self.y == T::ZERO && self.z == T::ZERO {
            return Quaternion::new(T::ZERO, inf.copysign(self.x), self.y, self.z);
        }
        let g = |t: T| if t == T::ZERO { t } else { inf.copysign(t) };
        Quaternion::new(T::ZERO, g(self.x), g(self.y), g(self.z))
    }

    /// `self` to the integer power `n`, by repeated squaring: exact as long as the products are,
    /// and otherwise accurate normwise, to about `n·ε·|qⁿ|`, as repeated products are; a part much
    /// smaller than that can lose its digits to cancellation, as in any product. A negative power
    /// is the power of the reciprocal, and `q⁰ = 1` for every `q`. The power starts from the
    /// base, not from 1, so that `q¹` is `q` and `q²` is `q·q` for every `q`, infinities included.
    /// Where a product of the powers overflows and a later product would meet its infinities
    /// (and make NaNs of them), the power of a finite quaternion is formed again from the base
    /// scaled by a power of two, the scaled powers multiplied with an exponent of their own and
    /// scaled back once: no part is then NaN, and the parts larger than the error, the largest
    /// among them, are `±∞` with their signs. `2^1100` is `∞ + 0i + 0j + 0k`, as `powf` gives it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 1.0, 1.0, 1.0);
    /// assert_eq!(q.powi(3), Quaternion::new(-8.0, 0.0, 0.0, 0.0));
    /// assert_eq!(q.powi(-1), q.recip());
    /// assert_eq!(q.powi(0), Quaternion::ONE);
    /// // A power that overflows keeps its zero parts.
    /// let two = Quaternion::new(2.0, 0.0, 0.0, 0.0);
    /// assert_eq!(two.powi(1100), Quaternion::new(f64::INFINITY, 0.0, 0.0, 0.0));
    /// ```
    pub fn powi(self, n: i32) -> Self {
        if n == 0 {
            return Self::ONE;
        }
        let mut base = if n < 0 { self.recip() } else { self };
        let mut k = n.unsigned_abs();
        // Whether every product had finite operands; the last one may overflow on its own.
        let mut clean = true;
        while k & 1 == 0 {
            clean &= base.is_finite();
            base *= base;
            k >>= 1;
        }
        let mut r = base;
        k >>= 1;
        while k > 0 {
            clean &= base.is_finite();
            base *= base;
            if k & 1 == 1 {
                clean &= r.is_finite() && base.is_finite();
                r *= base;
            }
            k >>= 1;
        }
        if clean || !self.is_finite() || self.is_zero() {
            r
        } else {
            self.powi_scaled(n)
        }
    }

    /// The power of a finite nonzero quaternion that overflows: the same products, of `m·2^e`
    /// with the mantissa `m` kept with its largest part in `[1, 2)` and the exponent `e` in an
    /// `i64`, so that nothing overflows until the result is scaled back, once. What the scaling
    /// loses of the small parts of a mantissa is below `2⁻¹⁰⁷⁴` of its largest, so that the power
    /// is accurate normwise as the plain one is.
    #[cold]
    #[inline(never)]
    fn powi_scaled(self, n: i32) -> Self {
        let split = |q: Self| {
            let k = exponent(q.max_abs());
            (q.ldexp(-k), i64::from(k))
        };
        let (m, e) = split(self);
        let (mut b, mut eb) = if n < 0 {
            let (r, k) = split(m.recip());
            (r, k - e)
        } else {
            (m, e)
        };
        let square = |b: Self, eb: i64| {
            let (p, k) = split(Self::hamilton(b, b));
            (p, 2 * eb + k)
        };
        let mut k = n.unsigned_abs();
        while k & 1 == 0 {
            (b, eb) = square(b, eb);
            k >>= 1;
        }
        let (mut r, mut er) = (b, eb);
        k >>= 1;
        while k > 0 {
            (b, eb) = square(b, eb);
            if k & 1 == 1 {
                let (p, j) = split(Self::hamilton(r, b));
                (r, er) = (p, er + eb + j);
            }
            k >>= 1;
        }
        r.ldexp(er.clamp(-4000, 4000) as i32)
    }

    /// The principal value of `self` to the real power `t`: `e^(t·ln self)`, which is
    /// `|q|^t·(cos(tθ) + u·sin(tθ))` for `q = |q|·(cos θ + u·sin θ)`. `q⁰ = 1` for every `q`, NaN
    /// included, as for floats; otherwise a NaN gives NaN in every part. `0^t` is 0 for `t > 0`
    /// and `+∞` (in the real part, the others zero) for `t < 0`. A negative real `w` with a zero
    /// vector part takes the complex value, `(−1)^½ = i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(0.0, 0.0, 0.0, 4.0);
    /// let r = q.powf(0.5);
    /// assert!((r * r - q).norm() < 1e-15 * 4.0);
    /// assert!((q.powf(2.0) - q * q).norm() < 1e-14);
    /// ```
    pub fn powf(self, t: T) -> Self {
        if t == T::ZERO {
            return Self::ONE;
        }
        if self.is_nan() || t.is_nan() {
            return Self::NAN;
        }
        if self.is_zero() {
            return if t > T::ZERO {
                Self::ZERO
            } else {
                Quaternion::from(T::INFINITY)
            };
        }
        (self.ln() * t).exp()
    }

    /// The principal value of `self` to the quaternion power `p`: `e^(ln(self)·p)`, the logarithm
    /// multiplied by `p` on the right; `e^(p·ln(self))` differs where `p` and `ln(self)` do not
    /// commute, and the two agree for real `p` and for `p` along the axis of `self`. `q⁰ = 1` for
    /// every `q`; otherwise a NaN gives NaN in every part. `0^p` is 0 for a `p` with a positive
    /// real part, `+∞` for a negative real `p`, and NaN for the others.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 0.0, 0.0);
    /// let p = Quaternion::new(0.5, 0.25, 0.0, 0.0);
    /// let want = (q.ln() * p).exp();
    /// assert_eq!(q.powq(p), want);
    /// // Real powers are powf.
    /// assert!((q.powq(Quaternion::from(1.5)) - q.powf(1.5)).norm() < 1e-15);
    /// ```
    pub fn powq(self, p: Self) -> Self {
        if p.is_zero() {
            return Self::ONE;
        }
        if self.is_nan() || p.is_nan() {
            return Self::NAN;
        }
        if self.is_zero() {
            return if p.w > T::ZERO {
                Self::ZERO
            } else if p.x == T::ZERO && p.y == T::ZERO && p.z == T::ZERO {
                Quaternion::from(T::INFINITY)
            } else {
                Self::NAN
            };
        }
        (self.ln() * p).exp()
    }
}

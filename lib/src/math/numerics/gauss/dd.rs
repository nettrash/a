//! Double-double numbers for the Gaussian rules: a value `hi + lo` with `|lo| ≤ ulp(hi)/2`, about
//! 106 bits, by the error-free transformations of Knuth (the sum) and of Dekker (the product, or
//! one fused multiply-add where the hardware has it). The rules need them for one reason: a node
//! has to be known past the last bit of an `f64` for its weight to be accurate to the last bit,
//! because the weight of a node next to an end of the interval changes like `1/(1 − x)` with the
//! node.

/// `hi + lo`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(super) struct Dd {
    pub(super) hi: f64,
    pub(super) lo: f64,
}

/// Whether `mul_add` is a single fused instruction here. Elsewhere it is a call into the
/// library, which is exact but slow, and Dekker's product is used in its place.
const FMA: bool = cfg!(all(
    any(target_arch = "aarch64", target_feature = "fma"),
    not(miri)
));

/// `a + b = s + e` exactly.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let v = s - a;
    (s, (a - (s - v)) + (b - v))
}

/// `a + b = s + e` exactly, when `|a| ≥ |b|` or `a` is zero.
#[inline(always)]
fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// `a · b = p + e` exactly, barring overflow and underflow.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if FMA {
        return (p, a.mul_add(b, -p));
    }
    // Veltkamp's split of each factor into halves of 26 bits, whose products are exact. The
    // factors here are scaled to stay far below the split's overflow at 2^996.
    let split = |x: f64| {
        let t = 134_217_729.0 * x; // 2^27 + 1
        let h = t - (t - x);
        (h, x - h)
    };
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

impl Dd {
    /// 0.
    pub(super) const ZERO: Dd = Dd { hi: 0.0, lo: 0.0 };
    /// 1.
    pub(super) const ONE: Dd = Dd { hi: 1.0, lo: 0.0 };
    // The constants below are the leading double and the rounded remainder of the values to 60
    // digits from mpmath (`hi = float(x)`, `lo = float(x − hi)`).
    /// ln 2, to 107 bits.
    pub(super) const LN_2: Dd = Dd {
        hi: core::f64::consts::LN_2,
        lo: 2.319_046_813_846_299_6e-17,
    };
    /// √π, to 107 bits.
    pub(super) const SQRT_PI: Dd = Dd {
        hi: 1.772_453_850_905_516,
        lo: -7.666_586_499_825_799e-17,
    };
    /// ln √π, to 107 bits.
    pub(super) const LN_SQRT_PI: Dd = Dd {
        hi: 0.572_364_942_924_700_1,
        lo: 5.132_975_581_353_913e-18,
    };

    /// The exact value `x`.
    #[inline(always)]
    pub(super) const fn new(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    /// The nearest `f64`.
    #[inline(always)]
    pub(super) fn value(self) -> f64 {
        self.hi + self.lo
    }

    /// `self + b`. The error is at most about `2^−105 (|self| + |b|)`, which is all a sum that
    /// cancels can be asked for.
    #[inline(always)]
    pub(super) fn add(self, b: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, b.hi);
        let (hi, lo) = quick_two_sum(s, e + (self.lo + b.lo));
        Dd { hi, lo }
    }

    /// `self − b`.
    #[inline(always)]
    pub(super) fn sub(self, b: Dd) -> Dd {
        self.add(b.neg())
    }

    /// `−self`.
    #[inline(always)]
    pub(super) fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// `self · b`.
    #[inline(always)]
    pub(super) fn mul(self, b: Dd) -> Dd {
        let (p, e) = two_prod(self.hi, b.hi);
        let (hi, lo) = quick_two_sum(p, e + (self.hi * b.lo + self.lo * b.hi));
        Dd { hi, lo }
    }

    /// `self · b` for an `f64` `b`.
    #[inline(always)]
    pub(super) fn mul_f64(self, b: f64) -> Dd {
        let (p, e) = two_prod(self.hi, b);
        let (hi, lo) = quick_two_sum(p, e + self.lo * b);
        Dd { hi, lo }
    }

    /// `self / b`: the quotient of the leading parts, corrected twice by the remainder.
    #[inline(always)]
    pub(super) fn div(self, b: Dd) -> Dd {
        let q1 = self.hi / b.hi;
        let r = self.sub(b.mul_f64(q1));
        let q2 = r.hi / b.hi;
        let r = r.sub(b.mul_f64(q2));
        let q3 = r.hi / b.hi;
        let (hi, lo) = quick_two_sum(q1, q2);
        Dd { hi, lo }.add(Dd::new(q3))
    }

    /// `√self` for `self ≥ 0`: the root of the leading part and one Newton correction.
    pub(super) fn sqrt(self) -> Dd {
        if self.hi <= 0.0 {
            return Dd::new(self.hi.sqrt());
        }
        let s = self.hi.sqrt();
        let (p, e) = two_prod(s, s);
        let r = ((self.hi - p) - e + self.lo) / (2.0 * s);
        let (hi, lo) = quick_two_sum(s, r);
        Dd { hi, lo }
    }

    /// `self · 2^k`, exact while the result is a normal number.
    #[inline(always)]
    pub(super) fn scale(self, k: i32) -> Dd {
        let f = pow2(k);
        Dd {
            hi: self.hi * f,
            lo: self.lo * f,
        }
    }

    /// `ln self` for a `self` in `[1, 2)`, within about one unit of the last place of an `f64`
    /// absolutely, which is all a weight formed as the exponential of a sum of logarithms
    /// needs: the logarithm is below 0.7, and `ln(hi + lo) = ln hi + lo/hi` to that precision.
    pub(super) fn ln_near_one(self) -> f64 {
        self.hi.ln() + self.lo / self.hi
    }
}

/// `2^k` for `|k| ≤ 1022`, exactly.
#[inline(always)]
pub(super) fn pow2(k: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

/// The binary exponent `e` and the significand `m` of a finite non-zero `x`: `x = m · 2^e` with
/// `1 ≤ |m| < 2`, `m` of the sign of `x`.
pub(super) fn frexp(x: f64) -> (f64, i64) {
    let bits = x.to_bits();
    let raw = ((bits >> 52) & 0x7ff) as i64;
    if raw == 0 {
        // A subnormal: scaled into the normal range first, which is exact.
        let (m, e) = frexp(x * pow2(64));
        return (m, e - 64);
    }
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52));
    (m, raw - 1023)
}

/// `x · 2^k` for any `k`, correctly rounded: exact while the result is a normal number, and
/// rounded once where it falls into the subnormal range, so that a weight too small for an
/// `f64` becomes a subnormal number or zero smoothly.
pub(super) fn ldexp(x: f64, k: i64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let (m, e) = frexp(x);
    let t = e + k;
    if t > 1023 {
        return m * f64::INFINITY;
    }
    if t >= -1022 {
        return m * pow2(t as i32);
    }
    if t >= -1074 {
        // A subnormal power of two, exact, times m: one rounding.
        return m * f64::from_bits(1u64 << (t + 1074));
    }
    if t == -1075 {
        // Between half the smallest subnormal and the smallest: m/2 is exact, and its product
        // with the smallest subnormal rounds once.
        return (m * 0.5) * f64::from_bits(1);
    }
    // Below half the smallest subnormal, which rounds to zero.
    m * 0.0
}

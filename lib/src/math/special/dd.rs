//! Double-double arithmetic for the special functions: values `hi + lo` with `|lo| ≤ ulp(hi)/2`,
//! about 106 bits, by error-free transformations (a fused multiply-add for the products where
//! the hardware has one, Dekker's product elsewhere).

#![allow(dead_code)]

/// `hi + lo`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Dd {
    pub hi: f64,
    pub lo: f64,
}

/// `a + b = s + e` exactly.
#[inline(always)]
pub(crate) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a + b = s + e` exactly, for `|a| ≥ |b|` (or `a` zero).
#[inline(always)]
pub(crate) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Whether `mul_add` is one fused instruction (elsewhere it is a call into the library, and under
/// Miri it need not be exact).
const FMA: bool = cfg!(all(
    any(target_arch = "aarch64", target_feature = "fma"),
    not(miri)
));

/// `a·b + c`, fused where the hardware has it (for terms whose rounding does not matter).
#[inline(always)]
pub(crate) fn mla(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `a·b = p + e` exactly (barring overflow and underflow): by a fused multiply-add where the
/// hardware has one, else by Dekker's product of Veltkamp's halves.
#[inline(always)]
pub(crate) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if FMA {
        return (p, a.mul_add(b, -p));
    }
    // Veltkamp's split overflows from about 2^996: a large factor is scaled by 2^−64 first (exact
    // but for a subnormal error term), and the error scaled back.
    const BIG: f64 = 6.696_928_794_914_171e299; // 2^996
    if a.abs() > BIG || b.abs() > BIG {
        if !p.is_finite() {
            return (p, 0.0);
        }
        let (a, b) = if a.abs() > BIG {
            (a * 5.421_010_862_427_522e-20, b)
        } else {
            (a, b * 5.421_010_862_427_522e-20)
        };
        let (q, e) = dekker(a, b);
        debug_assert!(q * 18_446_744_073_709_551_616.0 == p);
        return (p, e * 18_446_744_073_709_551_616.0);
    }
    dekker(a, b)
}

/// Dekker's exact product, for factors below `2^996`.
#[inline(always)]
fn dekker(a: f64, b: f64) -> (f64, f64) {
    let split = |x: f64| {
        let t = 134_217_729.0 * x; // 2^27 + 1
        let hi = t - (t - x);
        (hi, x - hi)
    };
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

impl Dd {
    /// The value `x`.
    #[inline(always)]
    pub(crate) const fn new(x: f64) -> Dd {
        Dd { hi: x, lo: 0.0 }
    }

    /// The nearest `f64`.
    #[inline(always)]
    pub(crate) fn value(self) -> f64 {
        self.hi + self.lo
    }

    /// `self + b`.
    #[inline(always)]
    pub(crate) fn add(self, b: Dd) -> Dd {
        let (s, e) = two_sum(self.hi, b.hi);
        let (t, f) = two_sum(self.lo, b.lo);
        let (s, e) = fast_two_sum(s, e + t);
        let (hi, lo) = fast_two_sum(s, e + f);
        Dd { hi, lo }
    }

    /// `self + b`.
    #[inline(always)]
    pub(crate) fn add_f64(self, b: f64) -> Dd {
        let (s, e) = two_sum(self.hi, b);
        let (hi, lo) = fast_two_sum(s, e + self.lo);
        Dd { hi, lo }
    }

    /// `−self`.
    #[inline(always)]
    pub(crate) fn neg(self) -> Dd {
        Dd {
            hi: -self.hi,
            lo: -self.lo,
        }
    }

    /// `self − b`.
    #[inline(always)]
    pub(crate) fn sub(self, b: Dd) -> Dd {
        self.add(b.neg())
    }

    /// `self · b`.
    #[inline(always)]
    pub(crate) fn mul(self, b: Dd) -> Dd {
        let (p, e) = two_prod(self.hi, b.hi);
        let e = mla(self.hi, b.lo, mla(self.lo, b.hi, e));
        let (hi, lo) = fast_two_sum(p, e);
        Dd { hi, lo }
    }

    /// `self · b`.
    #[inline(always)]
    pub(crate) fn mul_f64(self, b: f64) -> Dd {
        let (p, e) = two_prod(self.hi, b);
        let (hi, lo) = fast_two_sum(p, mla(self.lo, b, e));
        Dd { hi, lo }
    }

    /// `self / b`.
    #[inline(always)]
    pub(crate) fn div(self, b: Dd) -> Dd {
        let q = self.hi / b.hi;
        // r = self − q·b, then one correction.
        let r = self.sub(b.mul_f64(q));
        let q2 = r.hi / b.hi;
        let (hi, lo) = fast_two_sum(q, q2);
        Dd { hi, lo }
    }

    /// `√self`, for `self ≥ 0`.
    #[inline(always)]
    pub(crate) fn sqrt(self) -> Dd {
        if self.hi <= 0.0 {
            return Dd::new(self.hi.sqrt());
        }
        let s = self.hi.sqrt();
        let (p, e) = two_prod(s, s);
        let r = ((self.hi - p) - e + self.lo) / (2.0 * s);
        let (hi, lo) = fast_two_sum(s, r);
        Dd { hi, lo }
    }
}

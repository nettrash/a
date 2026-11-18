//! The error-free transformations of the elliptic area, with the crate's switch on a fused
//! multiply-add: where the hardware has one (aarch64, or x86_64 built with `fma`) the products'
//! errors and the residuals come from `mul_add`; elsewhere `mul_add` is a call into the library
//! (a slow software fma, and under Miri not exact), so they come from Dekker's product and
//! `mla` is the plain `a·b + c`.

/// Whether `f64::mul_add` is a hardware instruction here.
pub(in crate::math::special) const FMA: bool = cfg!(all(
    any(target_arch = "aarch64", target_feature = "fma"),
    not(miri)
));

/// `a + b = s + e` exactly.
#[inline(always)]
pub(in crate::math::special) fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a + b = s + e` exactly, for `|a| ≥ |b|` (or `a` zero).
#[inline(always)]
pub(in crate::math::special) fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// Veltkamp's halves of `x`, `|x| < 2^996`.
#[inline(always)]
fn split(x: f64) -> (f64, f64) {
    let t = 134_217_729.0 * x; // 2^27 + 1
    let hi = t - (t - x);
    (hi, x - hi)
}

/// `a·b = p + e` exactly (barring overflow and underflow): by `mul_add`, or Dekker's product.
#[inline(always)]
pub(in crate::math::special) fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    if FMA {
        return (p, a.mul_add(b, -p));
    }
    if !p.is_finite() {
        return (p, 0.0);
    }
    const LIM: f64 = 6.696_928_794_914_171e299; // 2^996: the halves would overflow above it
    if a.abs() > LIM || b.abs() > LIM {
        // Scale the large factor by 2^−64 and the error back (exact but where it underflows).
        let s = 5.421_010_862_427_522e-20; // 2^−64
        let (a2, b2) = if a.abs() > b.abs() {
            (a * s, b)
        } else {
            (a, b * s)
        };
        let (_, e) = two_prod(a2, b2);
        return (p, e * 18_446_744_073_709_551_616.0);
    }
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
}

/// `a·b + c` with one rounding where the hardware fuses it, two elsewhere: for the terms whose
/// rounding is of second order.
#[inline(always)]
pub(in crate::math::special) fn mla(a: f64, b: f64, c: f64) -> f64 {
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

/// `c − a·b` exactly, for a residual `c ≈ a·b` (a quotient's or a square root's).
#[inline(always)]
pub(in crate::math::special) fn res(a: f64, b: f64, c: f64) -> f64 {
    if FMA {
        return (-a).mul_add(b, c);
    }
    let (p, e) = two_prod(a, b);
    (c - p) - e
}

/// `a·b + c` rounded about once: `mul_add`, or the exact product and a compensated sum.
#[inline(always)]
pub(in crate::math::special) fn fma(a: f64, b: f64, c: f64) -> f64 {
    if FMA {
        return a.mul_add(b, c);
    }
    let (p, e) = two_prod(a, b);
    let (s, f) = two_sum(p, c);
    if !s.is_finite() {
        return p + c;
    }
    s + (e + f)
}

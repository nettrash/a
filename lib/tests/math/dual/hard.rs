//! Hard cases, each against mpmath at 60 digits: exponents whose `e − 1` is not a float, the mixed
//! partial of `atan2` next to the diagonals, `hypot` at extreme ratios, `pow` where `xʸ`
//! underflows, intermediates that would leave the range before their result; and the documented
//! limit of the rule that a zero tangent part stays zero, pinned at the values it gives.

use a::math::dual::{Dual, DualNum, HyperDual, derivative, gradient, hessian, second_derivative};
use a::math::matrix::Matrix;
use core::hint::black_box;

/// The error of `got` in units of `ε` relative to `want`.
fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (want.abs() * f64::EPSILON)
}

#[test]
fn powers_keep_the_digits_of_an_inexact_exponent() {
    // 0.1·x^(−0.9) at 10³⁰⁰: x^(e − 1) with e − 1 rounded was 145 ulps off, as ln x = 691
    // multiplies the rounding of −0.9.
    let want = 1.000_000_000_000_003_9e-271;
    let d = Dual::new(1e300, 1.0).powf(0.1).eps;
    let h = HyperDual::new(1e300, 1.0, 1.0, 0.0).powf(0.1).eps1;
    let q = Dual::new(1e300, 1.0).pow(Dual::new(0.1, 0.0)).eps;
    for v in [d, h, q] {
        assert!(rel(v, want) <= 2.0, "{v:e}");
    }
    // At an ordinary point, an exponent of full precision: −7.5 + 2⁻⁵⁰ at 1000 (27 ulps before).
    let e = -7.5 + 2f64.powi(-50);
    let want = -2.371_708_245_126_298_6e-25;
    assert!(rel(Dual::new(1000.0, 1.0).powf(e).eps, want) <= 2.0);
    assert!(rel(Dual::new(1000.0, 1.0).pow(Dual::new(e, 0.0)).eps, want) <= 2.0);
}

#[test]
fn pow_has_its_mixed_partial_where_the_power_underflows() {
    // ∂²xʸ/∂x∂y = xʸ⁻¹·(1 + y ln x) is normal at (1e−200, 2), where xʸ = 1e−400 underflows; only at
    // x = 0 is it taken as its limit 0.
    let want = -9.200_340_371_976_183e-198;
    let p = HyperDual::new(1e-200, 1.0, 0.0, 0.0).pow(HyperDual::new(2.0, 0.0, 1.0, 0.0));
    let h = hessian(
        |v: &[HyperDual<f64>]| v[0].pow(v[1]),
        &Matrix::from_column_major(2, 1, vec![1e-200, 2.0]),
    );
    let n = Dual::new(Dual::new(1e-200, 1.0), Dual::new(0.0, 0.0))
        .pow(Dual::new(Dual::new(2.0, 0.0), Dual::new(1.0, 0.0)));
    for v in [p.eps1eps2, h[(0, 1)], h[(1, 0)]] {
        assert!(rel(v, want) <= 4.0, "{v:e}");
    }
    // The nested types differentiate xʸ·ln x with xʸ carried as a value, which underflows to 0,
    // and so lose its term xʸ·(1/x) = xʸ⁻¹: 0.1 % here (a documented limit of nesting).
    assert!(
        (n.eps.eps - want).abs() <= 1.1e-3 * want.abs(),
        "{:e}",
        n.eps.eps
    );
    assert!(rel(n.eps.eps, -9.210_340_371_976_183e-198) <= 4.0); // y·xʸ⁻¹·ln x alone
    let q = HyperDual::new(1e-100, 1.0, 0.0, 0.0).pow(HyperDual::new(3.5, 0.0, 1.0, 0.0));
    assert!(rel(q.eps1eps2, -8.049_047_825_479_16e-248) <= 4.0);
    let z = HyperDual::new(0.0, 1.0, 0.0, 0.0).pow(HyperDual::new(2.0, 0.0, 1.0, 0.0));
    assert_eq!((z.re, z.eps2, z.eps1eps2), (0.0, 0.0, 0.0));
}

#[test]
fn atan2_mixed_partial_keeps_its_digits_next_to_the_diagonals() {
    // (y² − x²)/r⁴ at y = 1 + 2⁻⁵², x = 1: the difference of the rounded y/r and x/r was 41 % off.
    let mixed = |y: f64, x: f64| {
        HyperDual::new(y, 1.0, 0.0, 0.0)
            .atan2(HyperDual::new(x, 0.0, 1.0, 0.0))
            .eps1eps2
    };
    let e = f64::EPSILON;
    assert!(rel(mixed(1.0 + e, 1.0), 1.110_223_024_625_156_2e-16) <= 4.0);
    assert!(rel(mixed(3.0 * (1.0 + 4.0 * e), 3.0), 4.934_324_553_889_578e-17) <= 4.0);
    // Across the other diagonal the sign turns: ∂²/∂y∂x at (1 + 2⁻⁵², −1).
    assert!(rel(mixed(1.0 + e, -1.0), 1.110_223_024_625_156_2e-16) <= 4.0);
    assert_eq!(mixed(1.0, 1.0), 0.0);
}

#[test]
fn hypot_second_partials_at_extreme_ratios() {
    // ∂²/∂x² hypot(x, y) = y²/h³: the square of y/h underflowed where the result does not.
    let (x, y) = (-3.075_235_583_731_793_4e-81, -1.684_760_308_251_667_4e-255);
    let h = HyperDual::new(x, 1.0, 1.0, 0.0).hypot(HyperDual::constant(y));
    assert!(
        rel(h.eps1eps2, 9.759_803_431_523_313e-269) <= 4.0,
        "{:e}",
        h.eps1eps2
    );
    let k = HyperDual::new(1e-100, 1.0, 1.0, 0.0).hypot(HyperDual::constant(1e-260));
    assert!(
        rel(k.eps1eps2, 9.999_999_999_999_998e-221) <= 4.0,
        "{:e}",
        k.eps1eps2
    );
    let m = HyperDual::new(1e-100, 1.0, 0.0, 0.0).hypot(HyperDual::new(1e-260, 0.0, 1.0, 0.0));
    assert!(rel(m.eps1eps2, -1e-60) <= 4.0);
}

#[test]
fn derivatives_at_the_ends_of_the_range_are_formed_in_range() {
    // cbrt″ = −2/(9c⁵) = −1.47e308 at 5.1e−186, where f′/x overflowed first.
    let c = HyperDual::new(5.102_799_809_411_062e-186, 1.0, 1.0, 0.0).cbrt();
    assert!(
        rel(c.eps1eps2, -1.469_286_884_242_159_4e308) <= 4.0,
        "{:e}",
        c.eps1eps2
    );
    // log₍ᵦ₎″ = −1/(x² ln b) = 1.4e308 for b = 0.027 at 4.4e−155, where 1/x² alone overflows; by
    // the hyper-dual number and through the nested types, which differentiate f′.
    let (x, b) = (4.437_317_020_667_586e-155, 0.026_980_563_021_900_154);
    let want = 1.405_835_712_537_533_4e308;
    let l = HyperDual::new(x, 1.0, 1.0, 0.0).log(b).eps1eps2;
    let n = Dual::new(Dual::new(x, 1.0), Dual::new(1.0, 0.0))
        .log(b)
        .eps
        .eps;
    assert!(rel(l, want) <= 4.0 && rel(n, want) <= 4.0, "{l:e} {n:e}");
    // f32 log10″ at 4.4e−20: −1/(x² ln 10) = −2.2e38, where x⁻² overflows f32.
    let x32 = 4.408_230_6e-20_f32;
    let g = HyperDual::new(x32, 1.0, 1.0, 0.0).log10().eps1eps2;
    let gn = Dual::new(Dual::new(x32, 1.0), Dual::new(1.0, 0.0))
        .log10()
        .eps
        .eps;
    for v in [g, gn] {
        let e = (f64::from(v) + 2.234_887_649_120_335_8e38).abs() / 2.234_887_649_120_335_8e38;
        assert!(e <= 4.0 * f64::from(f32::EPSILON), "{v:e}");
    }
    // Negative powers, where the standard library's powi, as 1/x^|m|, gives 0: (x⁻¹²)′ at 978.7 in
    // f32, and (x⁻⁴¹)″ at −1.5e7, both normal numbers.
    let p32 = Dual::new(black_box(978.719_5_f32), 1.0)
        .powi(black_box(-12))
        .eps;
    let e = (f64::from(p32) + 1.587_172_794_675_040_7e-38).abs() / 1.587_172_794_675_040_7e-38;
    assert!(e <= 4.0 * f64::from(f32::EPSILON), "{p32:e}");
    let p64 = HyperDual::new(black_box(-1.517_677_033_606_912_2e7), 1.0, 1.0, 0.0)
        .powi(black_box(-41))
        .eps1eps2;
    assert!(rel(p64, -2.788_178_497_513_836e-306) <= 16.0, "{p64:e}");
    // And a coefficient below 1 times a power that overflows: −0.6·x^(−1.6) at 8.5e−194.
    let (x, y): (f64, f64) = (8.470_260_661_517_206e-194, -0.597_637_718_725_217_5);
    let d = Dual::new(x, 1.0).pow(Dual::new(y, 0.0)).eps;
    let f = Dual::new(x, 1.0).powf(y).eps;
    assert!(
        d.is_finite() && f.is_finite() && rel(d, f) <= 4.0,
        "{d:e} {f:e}"
    );
}

#[test]
fn a_zero_tangent_at_an_infinite_derivative_gives_zero_where_the_limit_is_not() {
    // A documented limit. The rule that a zero tangent part contributes nothing cannot tell a
    // variable that does not move from an intermediate whose derivative is 0 (or underflows):
    // where such an intermediate meets an infinite derivative, the limit 0·∞ is taken as 0. Here
    // the composite is smooth and its derivative is 1 (or 2), and 0 is what comes back; num-dual
    // gives NaN at the same points. These assertions pin the documented behaviour.
    assert_eq!(derivative(|x: Dual<f64>| (x * x * x).cbrt(), 0.0), 0.0);
    assert_eq!(
        derivative(|x: Dual<f64>| x.powi(3).powf(1.0 / 3.0), 0.0),
        0.0
    );
    assert_eq!(derivative(|x: Dual<f64>| (x * x * x).cbrt(), 1e-300), 0.0);
    assert_eq!(derivative(|x: Dual<f64>| x.sin().powi(3).cbrt(), 0.0), 0.0);
    assert_eq!(
        second_derivative(|x: HyperDual<f64>| x.powi(6).cbrt(), 0.0),
        0.0
    );
    let origin = Matrix::from_column_major(2, 1, vec![0.0, 0.0]);
    let g = gradient(
        |v: &[Dual<f64>]| (v[0].powi(3) + v[1].powi(3)).cbrt(),
        &origin,
    );
    assert_eq!(g.as_slice(), &[0.0, 0.0]);
    // Away from the hidden zero the derivative is right, and so is the case the rule is for: a
    // variable that does not move, at a point where a derivative is infinite.
    let d = derivative(|x: Dual<f64>| (x * x * x).cbrt(), 1e-3);
    assert!((d - 1.0).abs() <= 4.0 * f64::EPSILON);
    fn f<D: DualNum<Float = f64>>(v: &[D]) -> D {
        v[0] + v[1].sqrt()
    }
    let g = gradient(f, &Matrix::from_column_major(2, 1, vec![1.0, 0.0]));
    assert_eq!((g[(0, 0)], g[(1, 0)]), (1.0, f64::INFINITY));
}

#[test]
fn atan2_mixed_partial_at_the_top_of_the_range() {
    // Where y ± x would overflow, the halves are taken: the mixed partial (y² − x²)/r⁴ is 0 (or
    // underflows to 0), not NaN or ∞ from an overflowed sum against a zero factor.
    let mixed = |y: f64, x: f64| {
        HyperDual::new(y, 1.0, 0.0, 0.0)
            .atan2(HyperDual::new(x, 0.0, 1.0, 0.0))
            .eps1eps2
    };
    for (y, x) in [
        (1e308, 1e308),
        (1e308, -1e308),
        (-1e308, -1e308),
        (1.7e308, 1.7e308),
        (1.7e308, 1e308),
        (9e307, 9e307),
        (1e308, 9e307),
    ] {
        assert_eq!(mixed(y, x), 0.0, "({y:e}, {x:e})");
    }
    // The other second partials there, 2xy/r⁴, underflow to 0 as well, and nothing is NaN.
    let h = hessian(
        |v: &[HyperDual<f64>]| v[0].atan2(v[1]),
        &Matrix::from_column_major(2, 1, vec![1e308, -1e308]),
    );
    assert!(h.as_slice().iter().all(|v| *v == 0.0), "{:?}", h.as_slice());
    // And below half the largest float the factored form is kept.
    assert!(rel(mixed(1.0 + f64::EPSILON, 1.0), 1.110_223_024_625_156_2e-16) <= 4.0);
}

#[test]
fn powers_beyond_the_floats_integers_have_the_right_sign() {
    // |e| ≥ 2⁵³ (2²⁴ in f32): every such float is an even integer and e − 1 is not a float; the
    // powers of the derivatives are taken from the value, with the parity of e − 1.
    let e = 9_007_199_254_740_994.0; // 2⁵³ + 2
    assert_eq!(Dual::new(black_box(-1.0), 1.0).powf(black_box(e)).eps, -e);
    let d = Dual::new(black_box(-1.000_000_000_000_000_2), 1.0)
        .powf(black_box(e))
        .eps;
    assert!(rel(d, -6.655_470_058_752_755e16) <= 4.0, "{d:e}");
    let h = HyperDual::new(black_box(-1.0), 1.0, 1.0, 0.0).powf(black_box(e));
    assert_eq!(h.eps1, -e);
    assert!(rel(h.eps1eps2, 8.112_963_841_460_672e31) <= 4.0);
    let p = Dual::new(black_box(-1.0), 1.0).pow(Dual::new(e, 0.0)).eps;
    assert_eq!(p, -e);
    let f = Dual::new(black_box(-1.0_f32), 1.0)
        .powf(black_box(16_777_218.0_f32))
        .eps;
    assert_eq!(f, -16_777_218.0);
    // Where the power overflows the derivative is +∞, not −∞ from a correction 1 + δ·ln x < 0.
    let big = 2f64.powi(60);
    assert_eq!(
        Dual::new(black_box(3.0), 1.0).powf(black_box(big)).eps,
        f64::INFINITY
    );
    assert_eq!(
        Dual::new(black_box(1e10), 1.0).powf(black_box(1e17)).eps,
        f64::INFINITY
    );
    let h2 = HyperDual::new(black_box(2.0), 1.0, 1.0, 0.0).powf(black_box(1e300));
    assert_eq!((h2.eps1, h2.eps1eps2), (f64::INFINITY, f64::INFINITY));
    assert_eq!(
        Dual::new(3.0, 1.0).pow(Dual::new(big, 0.0)).eps,
        f64::INFINITY
    );
    let g = Dual::new(black_box(3.0_f32), 1.0)
        .powf(black_box(2f32.powi(30)))
        .eps;
    assert_eq!(g, f32::INFINITY);
    // Both sides of 2⁵³ − 2, where e − 2 stops being a float for a negative e: f′ = e·(−1)^(e−1)
    // and f″ = e(e − 1)·(−1)^(e−2) at −1, with the parity of e − 1 and e − 2 (f″ of
    // x^(−(2⁵³ − 1)) is −8.11e31, not +8.11e31).
    let p53 = 2f64.powi(53);
    for e in [
        -(p53 - 3.0),
        -(p53 - 2.0),
        -(p53 - 1.0),
        -p53,
        -(p53 + 2.0),
        p53 - 3.0,
        p53 - 2.0,
        p53 - 1.0,
        p53,
        p53 + 2.0,
    ] {
        let h = HyperDual::new(black_box(-1.0), 1.0, 1.0, 0.0).powf(black_box(e));
        // The parity from e itself, as e − 1 and e − 2 may not be floats: (−1)^(e−1) is −1 for
        // an even e, (−1)^(e−2) is the parity of e.
        let even = e.rem_euclid(2.0) == 0.0;
        let s1 = if even { -1.0 } else { 1.0 };
        assert_eq!(h.eps1, e * s1, "f′ at e = {e}");
        let want2 = e * (e - 1.0) * -s1;
        assert!(
            h.eps1eps2.signum() == want2.signum(),
            "f″ at e = {e}: {:e}",
            h.eps1eps2
        );
        assert!(
            rel(h.eps1eps2, want2) <= 2.0,
            "f″ at e = {e}: {:e}",
            h.eps1eps2
        );
        let n = HyperDual::new(
            Dual::new(black_box(-1.0), 1.0),
            Dual::new(1.0, 0.0),
            Dual::new(1.0, 0.0),
            Dual::new(0.0, 0.0),
        )
        .powf(black_box(e));
        assert!(
            n.eps1eps2.re.signum() == want2.signum(),
            "nested f″ at e = {e}"
        );
    }
    let e32 = -(2f32.powi(24) - 1.0);
    let h32 = HyperDual::new(black_box(-1.0_f32), 1.0, 1.0, 0.0).powf(black_box(e32));
    assert!(h32.eps1eps2 < 0.0 && (h32.eps1eps2 + 2.814_749_5e14).abs() <= 2.814_749_5e14 * 4e-7);
    // At ±0 and ±∞, where xᵉ/x would be 0/0 or ∞/∞: 0 at either zero (as for every e > 1), and at
    // ±∞ the infinity of the sign of the derivative, +∞ at +∞ and −∞ at −∞ for an even power.
    for e in [p53, p53 + 2.0, 2f64.powi(60), 1e300] {
        for z in [0.0, -0.0] {
            let h = HyperDual::new(black_box(z), 1.0, 1.0, 0.0).powf(black_box(e));
            assert!(h.eps1 == 0.0 && h.eps1eps2 == 0.0, "e = {e} at {z}");
            let p = Dual::new(black_box(z), 1.0).pow(Dual::new(e, 0.0)).eps;
            assert_eq!(p, 0.0);
        }
        let up = HyperDual::new(black_box(f64::INFINITY), 1.0, 1.0, 0.0).powf(black_box(e));
        assert_eq!((up.eps1, up.eps1eps2), (f64::INFINITY, f64::INFINITY));
        let down = HyperDual::new(black_box(f64::NEG_INFINITY), 1.0, 1.0, 0.0).powf(black_box(e));
        assert_eq!(
            (down.eps1, down.eps1eps2),
            (f64::NEG_INFINITY, f64::INFINITY)
        );
    }
    let z32 = Dual::new(black_box(0.0_f32), 1.0)
        .powf(black_box(2f32.powi(24)))
        .eps;
    assert_eq!(z32, 0.0);
    // A documented limit: where xᵉ itself underflows to 0, the derivatives taken from it are 0,
    // within a factor |e| (for f′) or |e(e − 1)| (for f″) of the normal numbers. f″ of x^(−2⁶⁰) at
    // 1 + 3·2⁻⁵² is 3.8e−298 (xᵉ = 2.9e−334); f32 f″ of x^(2³⁰) at 0.99999988 is 3.0e−38.
    let u = HyperDual::new(1.000_000_000_000_000_7, 1.0, 1.0, 0.0).powf(-(2f64.powi(60)));
    assert_eq!((u.re, u.eps1eps2), (0.0, 0.0));
    let u32 = HyperDual::new(0.999_999_9_f32, 1.0, 1.0, 0.0).powf(2f32.powi(30));
    assert_eq!((u32.re, u32.eps1eps2), (0.0, 0.0));
    // An infinite exponent at x = 1, where xᵉ = 1: the derivatives are e·1 and e·(e − 1)·1, ±∞.
    let i = HyperDual::new(1.0, 1.0, 1.0, 0.0).powf(f64::INFINITY);
    assert_eq!(
        (i.re, i.eps1, i.eps1eps2),
        (1.0, f64::INFINITY, f64::INFINITY)
    );
    let j = HyperDual::new(1.0, 1.0, 1.0, 0.0).powf(f64::NEG_INFINITY);
    assert_eq!(
        (j.re, j.eps1, j.eps1eps2),
        (1.0, f64::NEG_INFINITY, f64::INFINITY)
    );
}

#[test]
fn powi_derivatives_where_the_power_is_subnormal() {
    // n·xⁿ⁻¹ is normal where xⁿ⁻¹ is subnormal: the power is taken in two halves with the
    // coefficient between them. The bound is the standard library's: its powi of such powers is
    // itself off by tens of ulps (x⁻⁵³⁷ and x⁻⁵³⁸ at 1.935 by 41 each), and the halves carry it.
    let tol = 128.0;
    let d = Dual::new(black_box(2.032_357_010_936_222), 1.0).powi(black_box(-1000));
    assert!(
        rel(d.eps, -4.920_395_356_814_053_6e-306) <= tol,
        "{:e}",
        d.eps
    );
    let h = HyperDual::new(black_box(2.032_357_010_936_222), 1.0, 1.0, 0.0).powi(black_box(-1000));
    assert!(rel(h.eps1, -4.920_395_356_814_053_6e-306) <= tol);
    assert!(rel(h.eps1eps2, 2.423_450_075_782_689_6e-303) <= tol);
    let e = Dual::new(black_box(1.935_434_144_507_380_6), 1.0)
        .powi(black_box(-1074))
        .eps;
    assert!(rel(e, -5.549_142_568_596_241_4e-306) <= tol, "{e:e}");
    let f = Dual::new(black_box(2.47_f32), 1.0)
        .powi(black_box(-100))
        .eps;
    let w = -2.175_741_084_999_921_7e-38;
    assert!(
        (f64::from(f) - w).abs() <= 16.0 * f64::from(f32::EPSILON) * w.abs(),
        "{f:e}"
    );
}

#[test]
fn nested_types_overflow_inside_a_derivative_the_coefficient_would_bring_back() {
    // A documented limit. x^(1e−300) at 1e−300: f″ = e(e − 1)·x^(e−2) = −1e300. The hyper-dual
    // number forms it from e(e − 1) and x^(e − 2); the nested Dual<Dual<_>> differentiates
    // e·x^(e−1), whose derivative part (e − 1)·x^(e−2) = −1e600 overflows before the coefficient
    // e = 1e−300 would bring it back, and gives −∞.
    let h = HyperDual::new(1e-300, 1.0, 1.0, 0.0).powf(1e-300).eps1eps2;
    assert!(rel(h, -9.999_999_999_999_999e299) <= 4.0, "{h:e}");
    let n = Dual::new(Dual::new(1e-300, 1.0), Dual::new(1.0, 0.0)).powf(1e-300);
    assert_eq!(n.eps.eps, f64::NEG_INFINITY);
    // The first derivative by the same nested number, e·x^(e−1) = 1, is right: only its own
    // derivative part overflows.
    assert!(rel(n.eps.re, 1.0) <= 4.0, "{:e}", n.eps.re);
    // The third derivative, 3.95e142 at 3.7e−148, has only nested forms, and both overflow so.
    let x = 3.698_354_355_896_144e-148;
    let dh = Dual::new(
        HyperDual::new(x, 1.0, 1.0, 0.0),
        HyperDual::new(1.0, 0.0, 0.0, 0.0),
    )
    .powf(1e-300)
    .eps
    .eps1eps2;
    let hd = HyperDual::new(
        Dual::new(x, 1.0),
        Dual::new(1.0, 0.0),
        Dual::new(1.0, 0.0),
        Dual::new(0.0, 0.0),
    )
    .powf(1e-300)
    .eps1eps2
    .eps;
    assert_eq!((dh, hd), (f64::INFINITY, f64::INFINITY));
}

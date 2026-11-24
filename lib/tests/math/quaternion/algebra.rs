//! The algebra: Hamilton's rules, the operators, the ring identities, the norm, the inverse, the
//! quotient, conversions and text.

use super::common::{Rng, identical, identical32, pow2, pow2f, rel, to32, to64};
use super::reference;
use a::math::complex::Complex;
use a::math::quaternion::Quaternion;

type Q = Quaternion<f64>;
type Q32 = Quaternion<f32>;

fn q(w: f64, x: f64, y: f64, z: f64) -> Q {
    Quaternion::new(w, x, y, z)
}

const EPS: f64 = f64::EPSILON;
const INF: f64 = f64::INFINITY;
const NAN: f64 = f64::NAN;

#[test]
fn hamilton_rules_hold_for_the_units() {
    let (i, j, k) = (Quaternion::<f64>::I, Quaternion::J, Quaternion::K);
    let m1 = -Quaternion::<f64>::ONE;
    assert_eq!(i * i, m1);
    assert_eq!(j * j, m1);
    assert_eq!(k * k, m1);
    assert_eq!(i * j * k, m1);
    assert_eq!(j * k, i);
    assert_eq!(k * i, j);
}

#[test]
fn the_units_multiply_by_the_whole_table_in_both_types() {
    let units = [Q::ONE, Q::I, Q::J, Q::K];
    // table[a][b] = (sign, index) of units[a]·units[b].
    let table = [
        [(1.0, 0), (1.0, 1), (1.0, 2), (1.0, 3)],
        [(1.0, 1), (-1.0, 0), (1.0, 3), (-1.0, 2)],
        [(1.0, 2), (-1.0, 3), (-1.0, 0), (1.0, 1)],
        [(1.0, 3), (1.0, 2), (-1.0, 1), (-1.0, 0)],
    ];
    for (a, row) in table.iter().enumerate() {
        for (b, &(s, c)) in row.iter().enumerate() {
            assert_eq!(units[a] * units[b], units[c] * s, "{a}·{b}");
            assert_eq!(to32(units[a]) * to32(units[b]), to32(units[c] * s));
        }
    }
    // The product of two pure vectors is −u·v + u × v.
    let (u, v) = (q(0.0, 1.0, 2.0, 3.0), q(0.0, -4.0, 5.0, 0.5));
    assert_eq!(
        u * v,
        q(
            -(-4.0 + 10.0 + 1.5),
            2.0 * 0.5 - 3.0 * 5.0,
            3.0 * -4.0 - 0.5,
            5.0 + 8.0
        )
    );
}

#[test]
fn the_operators_take_owned_and_borrowed_operands_and_floats_on_either_side() {
    let (a, b) = (q(1.0, 2.0, 3.0, 4.0), q(0.5, -1.0, 2.0, -0.25));
    assert_eq!(a + b, q(1.5, 1.0, 5.0, 3.75));
    assert_eq!(a - b, q(0.5, 3.0, 1.0, 4.25));
    let ab = a * b;
    #[allow(clippy::op_ref)]
    {
        assert_eq!(&a + &b, a + b);
        assert_eq!(&a + b, a + b);
        assert_eq!(a + &b, a + b);
        assert_eq!(&a - &b, a - b);
        assert_eq!(&a * &b, ab);
        assert_eq!(&a * b, ab);
        assert_eq!(a * &b, ab);
        assert_eq!(&a / &b, a / b);
        assert_eq!(&a / b, a / b);
        assert_eq!(a / &b, a / b);
        assert_eq!(a * &2.0, a * 2.0);
        assert_eq!(&a * 2.0, a * 2.0);
        assert_eq!(&a * &2.0, a * 2.0);
        assert_eq!(&2.0 * &a, a * 2.0);
        assert_eq!(&2.0 * a, a * 2.0);
        assert_eq!(2.0 * &a, a * 2.0);
        assert_eq!(&a + &1.0, a + 1.0);
        assert_eq!(&1.0 - &a, 1.0 - a);
        assert_eq!(&a / &2.0, a / 2.0);
        assert_eq!(&4.0 / &a, 4.0 / a);
        assert_eq!(-&a, -a);
    }
    // A float multiplies every part, and adds to the real part.
    assert_eq!(a * 2.0, q(2.0, 4.0, 6.0, 8.0));
    assert_eq!(2.0 * a, q(2.0, 4.0, 6.0, 8.0));
    assert_eq!(a + 1.0, q(2.0, 2.0, 3.0, 4.0));
    assert_eq!(1.0 + a, q(2.0, 2.0, 3.0, 4.0));
    assert_eq!(a - 1.0, q(0.0, 2.0, 3.0, 4.0));
    assert_eq!(1.0 - a, q(0.0, -2.0, -3.0, -4.0));
    assert_eq!(a / 2.0, q(0.5, 1.0, 1.5, 2.0));
    // 2/q = 2·q⁻¹, and a real factor stays out of the product's 0·∞.
    assert_eq!(30.0 / a, a.recip() * 30.0);
    assert_eq!(2.0 * q(INF, 0.0, 0.0, 0.0), q(INF, 0.0, 0.0, 0.0));
    // The assignments.
    let mut c = a;
    c += b;
    c -= &b;
    assert_eq!(c, a);
    c *= b;
    assert_eq!(c, ab);
    c /= &b;
    assert!(rel(c, a) < 4.0 * EPS);
    c = a;
    c += 1.0;
    c -= &1.0;
    c *= 2.0;
    c /= &2.0;
    c *= &b;
    c /= b;
    assert!(rel(c, a) < 4.0 * EPS);
    // The same operators for f32, with an f32 on the left.
    let (a32, b32) = (to32(a), to32(b));
    assert_eq!(a32 + b32, to32(a + b));
    assert_eq!(2.0f32 * a32, to32(2.0 * a));
    assert_eq!(1.0f32 - a32, to32(1.0 - a));
    assert_eq!(a32 * b32, to32(ab));
    assert_eq!(
        4.0f32 / Q32::new(0.0, 2.0, 0.0, 0.0),
        Q32::new(0.0, -2.0, 0.0, 0.0)
    );
    let mut d32 = a32;
    d32 *= 0.5f32;
    d32 /= &0.5f32;
    assert_eq!(d32, a32);
}

#[test]
fn sum_and_product_take_owned_and_borrowed_quaternions_in_order() {
    let v = [
        q(1.0, 2.0, 3.0, 4.0),
        q(0.0, 1.0, 0.0, 0.0),
        q(0.5, 0.0, -1.0, 2.0),
    ];
    assert_eq!(v.iter().sum::<Q>(), q(1.5, 3.0, 2.0, 6.0));
    assert_eq!(v.into_iter().sum::<Q>(), q(1.5, 3.0, 2.0, 6.0));
    assert_eq!(v.iter().product::<Q>(), v[0] * v[1] * v[2]);
    assert_eq!(v.into_iter().product::<Q>(), v[0] * v[1] * v[2]);
    // Not commutative: the order is the iterator's.
    assert_ne!(v.iter().rev().product::<Q>(), v.iter().product::<Q>());
    assert_eq!(core::iter::empty::<Q>().product::<Q>(), Q::ONE);
    assert_eq!(core::iter::empty::<Q>().sum::<Q>(), Q::ZERO);
}

#[test]
fn display_writes_each_part_with_its_sign_and_the_precision_given() {
    assert_eq!(format!("{}", q(1.0, 2.0, 3.0, 4.0)), "1+2i+3j+4k");
    assert_eq!(format!("{}", q(-1.5, -2.0, 0.0, -0.0)), "-1.5-2i+0j-0k");
    assert_eq!(
        format!("{:.1}", q(1.0, -0.25, 3.0, 1e-3)),
        "1.0-0.2i+3.0j+0.0k"
    );
    assert_eq!(format!("{}", q(INF, -INF, NAN, 0.0)), "inf-infi+NaNj+0k");
    assert_eq!(format!("{}", Q32::new(0.5, 1.0, -2.0, 4.0)), "0.5+1i-2j+4k");
}

#[test]
fn conversions_give_the_parts_in_order_and_the_complex_subalgebra() {
    assert_eq!(Q::from(2.5), q(2.5, 0.0, 0.0, 0.0));
    assert_eq!(Q::from([1.0, 2.0, 3.0, 4.0]), q(1.0, 2.0, 3.0, 4.0));
    assert_eq!(
        <[f64; 4]>::from(q(1.0, 2.0, 3.0, 4.0)),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(Q::from(Complex::new(1.0, -2.0)), q(1.0, -2.0, 0.0, 0.0));
    assert_eq!(
        q(1.0, -2.0, 0.0, -0.0).to_complex(),
        Some(Complex::new(1.0, -2.0))
    );
    assert_eq!(q(1.0, -2.0, 1e-300, 0.0).to_complex(), None);
    assert_eq!(q(1.0, -2.0, 0.0, NAN).to_complex(), None);
    let p = Q::from_parts(1.0, [2.0, 3.0, 4.0]);
    assert_eq!((p.scalar(), p.vector()), (1.0, [2.0, 3.0, 4.0]));
    assert_eq!(Q::default(), Q::ZERO);
    assert_eq!(q(1.0, 2.0, 3.0, 4.0).conj(), q(1.0, -2.0, -3.0, -4.0));
    assert_eq!(q(1.0, 2.0, 3.0, 4.0).dot(q(-1.0, 0.5, 2.0, 0.25)), 7.0);
    // The complex numbers multiply and divide in the subalgebra as Complex does.
    let mut rng = Rng(0x5eed);
    for _ in 0..if cfg!(miri) { 10 } else { 200 } {
        let (a, b) = (
            Complex::new(rng.signed(), rng.signed()),
            Complex::new(rng.signed(), rng.signed()),
        );
        let (qa, qb) = (Q::from(a), Q::from(b));
        assert_eq!((qa * qb).to_complex(), Some(a * b));
        assert_eq!((qa + qb).to_complex(), Some(a + b));
        let d = (qa / qb).to_complex().unwrap();
        assert!((d - a / b).abs() <= 4.0 * EPS * (a / b).abs());
    }
}

/// `Rng(seed)` reads better in the tests than `Rng::new(seed)`.
#[allow(non_snake_case)]
fn Rng(seed: u64) -> Rng {
    Rng::new(seed)
}

#[test]
fn the_ring_identities_hold_on_random_quaternions_to_rounding() {
    let mut rng = Rng(25);
    let mut worst = [0.0f64; 6];
    for _ in 0..if cfg!(miri) { 20 } else { 2000 } {
        let (a, b, c) = (rng.quaternion(), rng.quaternion(), rng.quaternion());
        let scale = a.norm() * b.norm() * c.norm();
        let e = [
            ((a * b) * c - a * (b * c)).norm() / scale,
            (a * (b + c) - (a * b + a * c)).norm() / (a.norm() * (b.norm() + c.norm())),
            ((a * b).norm() - a.norm() * b.norm()).abs() / (a.norm() * b.norm()),
            (a * a.recip() - Q::ONE)
                .norm()
                .max((a.recip() * a - Q::ONE).norm()),
            ((a * b).conj() - b.conj() * a.conj()).norm() / (a.norm() * b.norm()),
            rel((a / b) * b, a),
        ];
        for (w, e) in worst.iter_mut().zip(e) {
            *w = w.max(e / EPS);
        }
        // Division is on the right: a/b = a·b⁻¹, which is not b⁻¹·a.
        assert!(rel(a / b, reference::product(a, reference::recip(b))) < 6.0 * EPS);
    }
    eprintln!("ring identities, largest error in ε: {worst:?}");
    for (w, bound) in worst.iter().zip([8.0, 8.0, 4.0, 6.0, 4.0, 8.0]) {
        assert!(*w <= bound, "{worst:?}");
    }
    // The same in f32.
    let mut worst32 = 0.0f64;
    for _ in 0..if cfg!(miri) { 10 } else { 500 } {
        let (a, b) = (to32(rng.quaternion()), to32(rng.quaternion()));
        let e = (to64(a * b) - reference::product(to64(a), to64(b))).norm()
            / (to64(a).norm() * to64(b).norm());
        worst32 = worst32.max(e / f32::EPSILON as f64);
        let one = a * a.recip();
        assert!((to64(one) - Q::ONE).norm() < 6.0 * f32::EPSILON as f64);
    }
    eprintln!("f32 product, largest error in ε: {worst32}");
    assert!(worst32 <= 3.0);
}

#[test]
fn the_norm_neither_overflows_nor_underflows_and_is_accurate() {
    assert_eq!(q(1.0, 2.0, 2.0, 4.0).norm(), 5.0);
    assert_eq!(q(-0.0, 0.0, -0.0, 0.0).norm(), 0.0);
    let (big, tiny) = (pow2(1000), pow2(-1060));
    assert_eq!(q(3.0 * big, 0.0, 4.0 * big, 0.0).norm(), 5.0 * big);
    assert_eq!(q(0.0, 3.0 * tiny, 0.0, 4.0 * tiny).norm(), 5.0 * tiny);
    assert_eq!(q(f64::MAX, 0.0, 0.0, 0.0).norm(), f64::MAX);
    assert_eq!(q(f64::MAX, f64::MAX, 0.0, 0.0).norm(), INF);
    let m = f64::MAX / 2.0;
    assert_eq!(q(m, m, m, m).norm(), f64::MAX);
    let u = f64::from_bits(1);
    assert_eq!(q(u, 0.0, 0.0, 0.0).norm(), u);
    assert_eq!(q(u, u, u, u).norm(), 2.0 * u);
    // An infinity beats a NaN, as hypot has it.
    assert_eq!(q(NAN, INF, 0.0, 0.0).norm(), INF);
    assert_eq!(q(1.0, 2.0, -INF, 0.0).norm(), INF);
    assert!(q(NAN, 1.0, 0.0, 0.0).norm().is_nan());
    assert_eq!(Q32::new(3e30, 4e30, 0.0, 0.0).norm(), 5e30);
    assert_eq!(Q32::new(f32::MAX, 0.0, 0.0, 0.0).norm(), f32::MAX);
    // Against 160-bit arithmetic, on parts of random magnitudes from 2⁻¹⁰⁷⁴ to 2¹⁰²³.
    let mut rng = Rng(7);
    let mut worst = 0.0f64;
    let n = if cfg!(miri) { 20 } else { 3000 };
    for i in 0..n {
        let (lo, hi) = match i % 3 {
            0 => (-4.0, 4.0),
            1 => (-1074.0, -900.0),
            _ => (-1074.0, 1023.0),
        };
        let p = q(
            rng.log_uniform(lo, hi),
            rng.log_uniform(lo, hi),
            rng.log_uniform(lo, hi),
            rng.log_uniform(lo, hi),
        );
        let (got, want) = (p.norm(), reference::norm(p));
        if want >= f64::MIN_POSITIVE && want.is_finite() {
            worst = worst.max((got - want).abs() / (want * EPS));
        } else {
            assert!((got - want).abs() <= 2.0 * u, "{p}: {got} {want}");
        }
    }
    eprintln!("norm, largest error over {n} in ε of the norm: {worst}");
    assert!(worst <= 1.5);
}

#[test]
fn normalize_gives_unit_quaternions_at_every_magnitude_and_nan_otherwise() {
    let p = q(0.0, 3.0, 0.0, 4.0);
    let unit = q(0.0, 0.6, 0.0, 0.8);
    assert_eq!(p.normalize(), unit);
    for s in [pow2(1000), pow2(-1060), pow2(-1070), pow2(600)] {
        assert_eq!((p * s).normalize(), unit, "{s}");
    }
    let h = 0.5f64.sqrt();
    assert!(
        rel(
            q(f64::MAX, 0.0, f64::MAX, 0.0).normalize(),
            q(h, 0.0, h, 0.0)
        ) <= EPS
    );
    for bad in [
        Q::ZERO,
        q(INF, 0.0, 0.0, 0.0),
        q(1.0, NAN, 0.0, 0.0),
        -Q::ZERO,
    ] {
        let n = bad.normalize();
        assert!(
            n.w.is_nan() && n.x.is_nan() && n.y.is_nan() && n.z.is_nan(),
            "{bad}"
        );
    }
    let mut rng = Rng(8);
    let mut worst = 0.0f64;
    for _ in 0..if cfg!(miri) { 20 } else { 2000 } {
        let s = rng.log_uniform(-1074.0, 1000.0).abs();
        let p = rng.quaternion() * s;
        if p == Q::ZERO {
            continue;
        }
        let n = p.normalize();
        worst = worst.max((n.norm() - 1.0).abs() / EPS);
    }
    eprintln!("normalize, largest | |q/|q|| − 1 | in ε: {worst}");
    assert!(worst <= 2.0);
    let t = pow2f(-140);
    assert_eq!(
        Q32::new(0.0, 3.0 * t, 4.0 * t, 0.0).normalize(),
        Q32::new(0.0, 0.6, 0.8, 0.0)
    );
}

#[test]
fn recip_and_the_quotient_keep_their_digits_at_the_ends_of_the_range() {
    let p = q(1.0, 1.0, 1.0, 1.0);
    assert_eq!(p.recip(), q(0.25, -0.25, -0.25, -0.25));
    for e in [600, -600, 1000, -1000, -1070, 1020] {
        let s = pow2(e);
        let want = p.recip() / s;
        assert_eq!((p * s).recip(), want, "2^{e}");
    }
    // The squared norm 4·MAX² overflows, its reciprocal is subnormal.
    let m = q(f64::MAX, f64::MAX, f64::MAX, f64::MAX);
    let r = m.recip();
    assert!(rel(r, reference::recip(m)) < 1e-13, "{r}");
    assert_eq!(r.w, (0.25 / f64::MAX));
    // The reciprocal of the smallest subnormal overflows, as for floats.
    assert_eq!(q(f64::from_bits(1), 0.0, 0.0, 0.0).recip().w, INF);
    // Quotients whose products would overflow, or vanish.
    let a = q(1e300, -2e300, 3e300, 1e299);
    let b = q(2e300, 1e300, -1e300, 5e299);
    assert!(rel(a / b, reference::quotient(a, b)) < 4.0 * EPS);
    let (sa, sb) = (a * 1e-300 * 1e-300, b * 1e-300 * 1e-300);
    assert!(rel(sa / sb, reference::quotient(sa, sb)) < 1e-12);
    assert!(!(a / sb).is_finite());
    // The left and the right quotients differ.
    let (u, v) = (q(1.0, 2.0, 0.0, 0.0), q(0.0, 0.0, 1.0, 1.0));
    assert!(rel(u / v, u * v.recip()) < 2.0 * EPS);
    assert!(rel(u / v, v.recip() * u) > 0.1);
    let c = q(1e-300, 0.0, 0.0, 0.0) / q(1e300, 0.0, 0.0, 0.0);
    assert_eq!(c, Q::ZERO);
    let d = q(3.0, 0.0, 0.0, 0.0) / q(pow2(-1074), 0.0, 0.0, 0.0);
    assert_eq!(d.w, INF);
    let t = pow2(-1040);
    assert_eq!(
        q(t, 2.0 * t, 0.0, -3.0 * t) / q(t, 0.0, 0.0, 0.0),
        q(1.0, 2.0, 0.0, -3.0)
    );
    assert_eq!(
        q(t, 2.0 * t, 0.0, -3.0 * t) / q(0.0, 2.0 * t, 0.0, 0.0),
        q(1.0, -0.5, 1.5, 0.0)
    );
    // Random operands of every magnitude, against 160-bit arithmetic.
    let mut rng = Rng(9);
    let mut worst = [0.0f64; 2];
    let n = if cfg!(miri) { 20 } else { 2000 };
    for _ in 0..n {
        let sa = rng.log_uniform(-1000.0, 1000.0).abs();
        let sb = rng.log_uniform(-1000.0, 1000.0).abs();
        let (a, b) = (rng.quaternion() * sa, rng.quaternion() * sb);
        let want = reference::recip(b);
        if want.norm() > 1e-300 && want.norm() < 1e300 {
            worst[0] = worst[0].max(rel(b.recip(), want) / EPS);
        }
        let want = reference::quotient(a, b);
        if want.norm() > 1e-300 && want.norm() < 1e300 {
            worst[1] = worst[1].max(rel(a / b, want) / EPS);
        }
    }
    eprintln!("recip and quotient over {n}, largest normwise error in ε: {worst:?}");
    assert!(worst[0] <= 3.0 && worst[1] <= 6.0, "{worst:?}");
}

#[test]
fn the_quotient_takes_its_special_values_from_the_limits() {
    let one = q(1.0, 2.0, -3.0, 0.5);
    // A zero divisor divides each part by its real part, a signed zero.
    assert!(identical(one / Q::ZERO, q(INF, INF, -INF, INF)));
    assert!(identical(
        one / q(-0.0, 0.0, 0.0, 0.0),
        q(-INF, -INF, INF, -INF)
    ));
    assert!(identical(Q::ZERO.recip(), q(INF, NAN, NAN, NAN)));
    assert!(identical(
        q(-0.0, 0.0, 0.0, 0.0).recip(),
        q(-INF, NAN, NAN, NAN)
    ));
    assert!(identical(Q::ZERO / Q::ZERO, Q::new(NAN, NAN, NAN, NAN)));
    // A finite dividend over an infinite divisor: zeros.
    let z = one / q(INF, 0.0, 0.0, 0.0);
    assert_eq!(z, Q::ZERO);
    assert!(identical(z, q(0.0, 0.0, -0.0, 0.0)));
    assert_eq!(q(0.0, INF, 1.0, 0.0).recip(), Q::ZERO);
    // An infinite dividend over a finite divisor: infinities along the quotient's direction.
    assert!(identical(
        q(INF, 0.0, 0.0, 0.0) / q(1.0, 1.0, 0.0, 0.0),
        q(INF, -INF, 0.0, 0.0)
    ));
    assert!(identical(
        q(INF, 0.0, 0.0, 0.0) / 2.0,
        q(INF, 0.0, 0.0, 0.0)
    ));
    // Both infinite, or a NaN anywhere: NaN.
    assert!((q(INF, 0.0, 0.0, 0.0) / q(0.0, INF, 0.0, 0.0)).w.is_nan());
    assert!((one / q(1.0, NAN, 0.0, 0.0)).z.is_nan());
    assert!((q(NAN, 0.0, 0.0, 0.0) / one).x.is_nan());
    // A zero dividend gives zeros.
    assert_eq!(Q::ZERO / one, Q::ZERO);
    assert_eq!(Q::ZERO / (one * 1e300), Q::ZERO);
    // f32 too.
    assert!(identical32(
        Q32::ZERO.recip(),
        Q32::new(f32::INFINITY, f32::NAN, f32::NAN, f32::NAN)
    ));
    assert_eq!(Q32::new(f32::INFINITY, 1.0, 0.0, 0.0).recip(), Q32::ZERO);
    let big = Q32::new(3e38, 3e38, 3e38, 3e38);
    assert_eq!(big / big, Q32::ONE);
}

#[test]
fn f32_quotients_and_norms_at_the_ends_of_their_range() {
    let p = Q32::new(1.0, 2.0, -3.0, 0.5);
    for e in [-140, -100, 60, 100, 120] {
        let s = pow2f(e);
        assert_eq!((p * s).norm(), p.norm() * s, "2^{e}");
    }
    // The reciprocal of a quaternion of norm 2⁻¹⁴⁰ overflows.
    assert_eq!((p * pow2f(-140)).recip().w, f32::INFINITY);
    for e in [-120, -100, 60, 100, 120] {
        let s = pow2f(e);
        let r = (p * s).recip();
        let want = to32(reference::recip(to64(p * s)));
        assert!(
            (to64(r) - to64(want)).norm() <= 2.0 * f32::EPSILON as f64 * to64(want).norm(),
            "{e}"
        );
    }
    let m = Q32::new(f32::MAX, -f32::MAX, f32::MAX, f32::MAX);
    assert_eq!(m / m, Q32::ONE);
    assert!(to64(m.recip()).norm() > 0.0);
}

#[test]
fn overflowing_products_have_infinities_where_they_overflow_and_no_nans() {
    // (1 + i + j + k)² = −2 + 2i + 2j + 2k, times 1e400.
    let c = q(1e200, 1e200, 1e200, 1e200);
    assert!(identical(c * c, q(-INF, INF, INF, INF)));
    // Finite parts beside an overflowing one keep their values, however small.
    let z = q(1.0, 1e300, 0.0, 0.0);
    assert!(identical(z * z, q(-INF, 2e300, 0.0, 0.0)));
    let m = q(f64::MAX, 0.0, 0.0, 0.0);
    assert!(identical(
        m * q(2.0, 0.0, 1.0, 0.0),
        q(INF, 0.0, f64::MAX, 0.0)
    ));
    let big = q(pow2(1023), pow2(-60), 0.0, 0.0);
    assert!(identical(
        big * q(pow2(1023), 0.0, 0.0, 0.0),
        q(INF, pow2(963), 0.0, 0.0)
    ));
    assert!(identical(big * big, q(INF, pow2(964), 0.0, 0.0)));
    assert!(identical(
        q(1e300, 1e-300, 0.0, 0.0) * q(1e10, 0.0, 0.0, 0.0),
        q(INF, 1e-300 * 1e10, 0.0, 0.0)
    ));
    assert_eq!(
        (q(1e308, 1e-10, 0.0, 0.0) * q(10.0, 0.0, 0.0, 0.0)).x,
        1e-10 * 10.0
    );
    assert_eq!(
        (q(1e308, 1.0 / 3.0, 0.0, 0.0) * q(10.0, 0.0, 0.0, 0.0)).x,
        (1.0 / 3.0) * 10.0
    );
    // A part with a product past the largest float, its exact value finite: from the scaled
    // product, to its error.
    assert_eq!(
        q(f64::MAX, 1.0, 0.0, 0.0) * q(2.0, 0.0, 0.0, 0.0),
        q(INF, 2.0, 0.0, 0.0)
    );
    let h = f64::MAX / 2.0;
    assert_eq!(
        q(h, h, 0.0, 0.0) * q(1.0, -1.0, 0.0, 0.0),
        q(f64::MAX, 0.0, 0.0, 0.0)
    );
    let (a, b) = (
        q(
            3.656769420054951e-71,
            -5.1294321423529415e134,
            -2.4764550376300285e253,
            -7.625381478028052e189,
        ),
        q(
            -6.227441784825319e-61,
            6.947827117506493e261,
            2.985944345951855e-63,
            1.5720952286709074e-183,
        ),
    );
    assert!(
        reference::check_product_parts(a, b, a * b, 4.0).2,
        "{}",
        a * b
    );
    let x = (a * b).x;
    assert!((x - 2.5407e191).abs() < 1e-4 * 2.5407e191, "{x}");
    // Infinite operands keep the plain formula, and its NaNs of 0·∞.
    assert!((q(INF, 0.0, 0.0, 0.0) * q(1.0, 1.0, 0.0, 0.0)).y.is_nan());
    // Random pairs against 160-bit products, part by part: each part within 4 units of ε times
    // the sum of the absolute values of its products, the infinity of its sign where it
    // overflows. Parts of very different sizes: each its own magnitude from 2⁻¹⁰⁷⁴ to 2¹⁰²³;
    // one part of each operand large and the others moderate; a large real part times small
    // ones; and parts within 2²⁰⁰ of one another.
    let mut rng = Rng(70);
    let n = if cfg!(miri) { 20 } else { 6000 };
    let (mut overflowing, mut finite_parts, mut worst) = (0, 0, 0.0f64);
    let sign = |rng: &mut super::common::Rng| if rng.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
    for i in 0..n {
        let quat = |rng: &mut super::common::Rng| -> Q {
            let mut p = [0.0; 4];
            match i % 4 {
                0 => {
                    for t in &mut p {
                        *t = rng.log_uniform(-1074.0, 1023.0);
                    }
                }
                1 => {
                    for t in &mut p {
                        *t = rng.log_uniform(-60.0, 60.0);
                    }
                    p[(rng.next_u64() % 4) as usize] = rng.log_uniform(500.0, 1023.0);
                }
                2 => {
                    for t in &mut p[1..] {
                        *t = rng.log_uniform(-300.0, 300.0);
                    }
                    p[0] = rng.log_uniform(900.0, 1023.0);
                }
                _ => {
                    let top = 300.0 + 700.0 * rng.uniform();
                    for t in &mut p {
                        *t = if rng.uniform() < 0.1 {
                            0.0
                        } else {
                            rng.log_uniform(top - 200.0, top)
                        };
                    }
                }
            }
            q(p[0] * sign(rng), p[1], p[2], p[3])
        };
        let (a, b) = (quat(&mut rng), quat(&mut rng));
        let (over, finite, ok, e) = reference::check_product_parts(a, b, a * b, 4.0);
        assert!(ok, "{a} · {b} = {}", a * b);
        overflowing += usize::from(over);
        finite_parts += finite;
        if over {
            worst = worst.max(e);
        }
    }
    eprintln!(
        "overflowing products: {overflowing} of {n}, {finite_parts} finite parts among them, every part within 4ε of its products' sum; largest finite-part error {worst} ε of that sum"
    );
    assert!(cfg!(miri) || (overflowing > n / 4 && finite_parts > n / 4));
    // f32 too.
    let c32 = Q32::new(1e20, 1e20, 1e20, 1e20);
    let inf32 = f32::INFINITY;
    assert!(identical32(
        c32 * c32,
        Q32::new(-inf32, inf32, inf32, inf32)
    ));
    assert!(identical32(c32.powi(2), c32 * c32));
    let z32 = Q32::new(1e30, 1e-30, 0.0, 0.0) * Q32::new(1e10, 0.0, 0.0, 0.0);
    assert!(identical32(z32, Q32::new(inf32, 1e-30 * 1e10, 0.0, 0.0)));
}

#[test]
fn quotients_round_to_the_smallest_subnormals() {
    // The scaled quotient is scaled back by 2^−1075, below the smallest subnormal power, in two
    // halves: 1.75·2⁻¹⁰⁷⁵ rounds to 2⁻¹⁰⁷⁴, and 0.375·2⁻¹⁰⁷⁴ to 0.
    let (a, b) = (
        q(1.75 * pow2(-560), 0.0, 0.0, 0.0),
        q(pow2(515), 0.0, 0.0, 0.0),
    );
    assert_eq!(a / b, q(pow2(-1074), 0.0, 0.0, 0.0));
    let a = q(1.5 * pow2(-560), -1.5 * pow2(-560), 0.0, 0.0);
    assert_eq!(a / q(pow2(516), 0.0, 0.0, 0.0), Q::ZERO);
    let c = q(1.25 * pow2(-600), 0.0, 0.0, 0.0) / q(0.0, pow2(474), 0.0, 0.0);
    assert!(identical(c, q(0.0, -pow2(-1074), 0.0, 0.0)));
    // Far below: zero.
    assert_eq!(
        q(pow2(-1000), 0.0, 0.0, 0.0) / q(pow2(1000), 0.0, 0.0, 0.0),
        Q::ZERO
    );
}

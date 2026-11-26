//! Euler angles in all 24 conventions: against SciPy's `from_euler` and `as_euler`
//! (`data/scipy_rotation.txt`), at and near gimbal lock, round trips, ranges, and the product of
//! the elementary rotations.

use super::common::{Rng, angle_dist, lines, pow2, quat, rot_dist, to32, to64};
use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
use core::f64::consts::{FRAC_PI_2, PI};

type Q = Quaternion<f64>;

const EPS: f64 = f64::EPSILON;
const SCIPY: &str = include_str!("data/scipy_rotation.txt");
const FRAMES: [EulerFrame; 2] = [EulerFrame::Intrinsic, EulerFrame::Extrinsic];

/// The largest difference of two triples of angles, modulo `2π`.
fn angles_dist(a: [f64; 3], b: &[f64]) -> f64 {
    (0..3).map(|i| angle_dist(a[i], b[i])).fold(0.0, f64::max)
}

/// The range of the middle angle.
fn middle_range(s: EulerSequence) -> (f64, f64) {
    if s.is_proper() {
        (0.0, PI)
    } else {
        (-FRAC_PI_2, FRAC_PI_2)
    }
}

#[test]
fn from_euler_agrees_with_scipy_in_all_24_conventions() {
    let mut rng = Rng::new(1);
    let mut data = lines(SCIPY, "euler_from").into_iter();
    let mut worst = 0.0f64;
    for s in EulerSequence::ALL {
        for f in FRAMES {
            for _ in 0..3 {
                let a = [0, 1, 2].map(|_| 4.0 * rng.signed());
                let got = Q::from_euler(s, f, a);
                let want = quat(&data.next().unwrap());
                let d = rot_dist(got, want);
                assert!(d <= 4.0 * EPS, "{s:?} {f:?} {a:?}: {got} is not {want}");
                worst = worst.max(d);
                assert!((got.norm() - 1.0).abs() <= 4.0 * EPS);
            }
        }
    }
    eprintln!("from_euler: largest difference from SciPy: {worst}");
}

#[test]
fn to_euler_agrees_with_scipy_in_all_24_conventions() {
    let mut rng = Rng::new(2);
    let mut data = lines(SCIPY, "euler_to").into_iter();
    let mut worst = 0.0f64;
    for s in EulerSequence::ALL {
        for f in FRAMES {
            for _ in 0..3 {
                let x = rng.quaternion();
                let got = x.to_euler(s, f);
                let want = data.next().unwrap();
                let d = angles_dist(got, &want);
                assert!(d <= 8.0 * EPS, "{s:?} {f:?} {x}: {got:?} is not {want:?}");
                worst = worst.max(d);
            }
        }
    }
    eprintln!("to_euler: largest difference from SciPy: {worst}");
}

#[test]
fn at_gimbal_lock_the_third_angle_is_zero_as_scipy_sets_it() {
    let mut rng = Rng::new(3);
    let mut data = lines(SCIPY, "gimbal").into_iter();
    let mut worst = [0.0f64; 2];
    for s in EulerSequence::ALL {
        let (lo, hi) = middle_range(s);
        for f in FRAMES {
            for (c, mid) in [lo, lo + 1e-9, hi, hi - 1e-9, lo + 1e-6, hi - 1e-6]
                .into_iter()
                .enumerate()
            {
                let first = PI * rng.signed();
                let third = PI * rng.signed();
                let x = Q::from_euler(s, f, [first, mid, third]);
                let got = x.to_euler(s, f);
                let want = data.next().unwrap();
                let d = angles_dist(got, &want);
                if c < 4 {
                    // Locked: the third angle 0, the first the whole turn about the one axis.
                    assert_eq!(got[2], 0.0, "{s:?} {f:?} {c}");
                    assert!(
                        d <= 16.0 * EPS,
                        "{s:?} {f:?} case {c}: {got:?} is not {want:?}"
                    );
                    worst[0] = worst[0].max(d);
                } else {
                    // Outside the tolerance the first and third angles are each ill-conditioned,
                    // by about ε/1e-6, but represent the rotation to rounding.
                    assert!(got[2] != 0.0);
                    assert!(d <= 1e-9, "{s:?} {f:?} case {c}: {got:?} is not {want:?}");
                    worst[1] = worst[1].max(d);
                }
                assert!(rot_dist(Q::from_euler(s, f, got), x) <= 1e-7 + 8.0 * EPS);
                if c < 4 && (mid == lo || mid == hi) {
                    assert!(rot_dist(Q::from_euler(s, f, got), x) <= 8.0 * EPS);
                }
            }
        }
    }
    eprintln!("gimbal lock: largest differences from SciPy, locked and near: {worst:?}");
}

#[test]
fn the_angles_round_trip_and_stay_in_their_ranges() {
    let mut rng = Rng::new(40);
    for s in EulerSequence::ALL {
        let (lo, hi) = middle_range(s);
        for f in FRAMES {
            for _ in 0..if cfg!(miri) { 3 } else { 200 } {
                // Angles inside the ranges, away from gimbal lock: the same angles back.
                let a = [
                    PI * rng.signed(),
                    lo + 1e-3 + (hi - lo - 2e-3) * rng.uniform(),
                    PI * rng.signed(),
                ];
                let x = Q::from_euler(s, f, a);
                let b = x.to_euler(s, f);
                assert!(angles_dist(b, &a) <= 1e-12, "{s:?} {f:?} {a:?} {b:?}");
                // Any quaternion: angles in range, and the rotation back.
                let y = rng.quaternion();
                let b = y.to_euler(s, f);
                assert!(b[0].abs() <= PI && b[2].abs() <= PI);
                assert!(b[1] >= lo && b[1] <= hi, "{s:?} {b:?}");
                assert!(rot_dist(Q::from_euler(s, f, b), y) <= 16.0 * EPS);
                // The norm and the sign do not matter.
                assert!(angles_dist((-y).to_euler(s, f), &b) <= 16.0 * EPS);
                assert!(angles_dist((y * pow2(-600)).to_euler(s, f), &b) <= 16.0 * EPS);
                assert!(angles_dist((y * pow2(900)).to_euler(s, f), &b) <= 16.0 * EPS);
            }
        }
    }
}

#[test]
fn from_euler_is_the_product_of_the_elementary_rotations() {
    let mut rng = Rng::new(41);
    let e = |axis: usize, t: f64| {
        let mut v = [0.0; 3];
        v[axis] = 1.0;
        Q::from_axis_angle(v, t)
    };
    for s in EulerSequence::ALL {
        let ax = s.axes();
        assert!(ax[0] != ax[1] && ax[1] != ax[2]);
        assert_eq!(s.is_proper(), ax[0] == ax[2]);
        for _ in 0..if cfg!(miri) { 2 } else { 50 } {
            let a = [0, 1, 2].map(|_| 4.0 * rng.signed());
            let (q1, q2, q3) = (e(ax[0], a[0]), e(ax[1], a[1]), e(ax[2], a[2]));
            let intrinsic = Q::from_euler(s, EulerFrame::Intrinsic, a);
            let extrinsic = Q::from_euler(s, EulerFrame::Extrinsic, a);
            assert!((intrinsic - q1 * q2 * q3).norm() <= 4.0 * EPS);
            assert!((extrinsic - q3 * q2 * q1).norm() <= 4.0 * EPS);
            // Intrinsic rotations turn about the axes as the body carries them: the second axis
            // is the first rotation's image of the fixed one.
            let mut v = [0.0; 3];
            v[ax[1]] = 1.0;
            let moved = q1.rotate(v);
            let via = Q::from_axis_angle(moved, a[1]) * q1;
            assert!((via - q1 * q2).norm() <= 8.0 * EPS);
        }
    }
    // The sequences: all twelve, distinct.
    let mut all: Vec<[usize; 3]> = EulerSequence::ALL.iter().map(|s| s.axes()).collect();
    all.sort();
    all.dedup();
    assert_eq!(all.len(), 12);
    assert_eq!(
        EulerSequence::ALL.iter().filter(|s| s.is_proper()).count(),
        6
    );
}

/// The product of the three elementary rotations `cos(θᵢ/2) + eᵢ·sin(θᵢ/2)`, formed with `*`,
/// in the order of `frame`: what `from_euler` documents, to the bit.
fn elementary_product<T: a::math::complex::Float>(
    s: EulerSequence,
    frame: EulerFrame,
    angles: [T; 3],
) -> Quaternion<T> {
    let ax = s.axes();
    let e = |i: usize| {
        let (sin, cos) = (angles[i] * T::HALF).sin_cos();
        let mut v = [T::ZERO; 3];
        v[ax[i]] = sin;
        Quaternion::from_parts(cos, v)
    };
    match frame {
        EulerFrame::Intrinsic => e(0) * e(1) * e(2),
        EulerFrame::Extrinsic => e(2) * e(1) * e(0),
    }
}

#[test]
fn from_euler_is_that_product_bit_for_bit_with_the_signs_of_its_zeros() {
    // Signed zeros, subnormals and the ends of the range among the angles, where parts of the
    // product are zeros whose signs come from the zero parts of the factors.
    let specials = [
        0.0,
        -0.0,
        5e-324,
        -1e-310,
        PI,
        -PI,
        FRAC_PI_2,
        2.0 * PI,
        1e300,
        -1e-20,
    ];
    let mut rng = Rng::new(43);
    let angle = |rng: &mut Rng| -> f64 {
        let k = rng.next_u64();
        if k.is_multiple_of(3) {
            specials[(k >> 8) as usize % specials.len()]
        } else {
            4.0 * rng.signed()
        }
    };
    let same = |x: f64, y: f64| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
    let same32 = |x: f32, y: f32| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
    let mut zeros = 0;
    for s in EulerSequence::ALL {
        for frame in [EulerFrame::Intrinsic, EulerFrame::Extrinsic] {
            for _ in 0..if cfg!(miri) { 4 } else { 2000 } {
                let a = [0, 1, 2].map(|_| angle(&mut rng));
                let (got, want) = (Q::from_euler(s, frame, a), elementary_product(s, frame, a));
                assert!(
                    same(got.w, want.w)
                        && same(got.x, want.x)
                        && same(got.y, want.y)
                        && same(got.z, want.z),
                    "{s:?} {frame:?} {a:?}: {got:?}, not {want:?}"
                );
                zeros += [got.w, got.x, got.y, got.z]
                    .iter()
                    .filter(|&&x| x == 0.0)
                    .count();
                let a32 = a.map(|x| x as f32);
                let (got, want) = (
                    Quaternion::<f32>::from_euler(s, frame, a32),
                    elementary_product(s, frame, a32),
                );
                assert!(
                    same32(got.w, want.w)
                        && same32(got.x, want.x)
                        && same32(got.y, want.y)
                        && same32(got.z, want.z),
                    "{s:?} {frame:?} {a32:?}: {got:?}, not {want:?}"
                );
            }
        }
    }
    assert!(zeros > if cfg!(miri) { 0 } else { 1000 }, "{zeros}");
}

#[test]
fn what_is_no_rotation_has_no_angles() {
    for x in [
        Q::ZERO,
        Quaternion::new(f64::NAN, 0.0, 0.0, 0.0),
        Quaternion::new(1.0, f64::INFINITY, 0.0, 0.0),
    ] {
        for s in EulerSequence::ALL {
            for f in FRAMES {
                assert!(x.to_euler(s, f).iter().all(|t| t.is_nan()), "{x} {s:?}");
            }
        }
    }
    assert!(
        Q::from_euler(
            EulerSequence::XYZ,
            EulerFrame::Intrinsic,
            [f64::NAN, 0.0, 0.0]
        )
        .w
        .is_nan()
    );
}

#[test]
fn euler_angles_of_f32_agree_with_those_of_f64() {
    let mut rng = Rng::new(42);
    for s in EulerSequence::ALL {
        for f in FRAMES {
            for _ in 0..if cfg!(miri) { 2 } else { 50 } {
                let x = to32(rng.quaternion());
                let a = x.to_euler(s, f).map(|t| t as f64);
                let b = to64(x).to_euler(s, f);
                // Away from gimbal lock, where the first and third angles are well-conditioned.
                let (lo, hi) = middle_range(s);
                if b[1] - lo > 1e-2 && hi - b[1] > 1e-2 {
                    assert!(angles_dist(a, &b) <= 1e-5, "{s:?} {f:?} {a:?} {b:?}");
                }
                let back = Quaternion::<f32>::from_euler(s, f, x.to_euler(s, f));
                assert!(rot_dist(to64(back), to64(x)) <= 8.0 * f32::EPSILON as f64);
            }
        }
    }
}

#[test]
fn the_gimbal_tolerance_is_scipys_ten_to_the_minus_seven() {
    // The middle angle 0.8·10⁻⁷ from its singular value is locked (the third angle 0), and
    // 1.25·10⁻⁷ from it is not, in every convention and at both singular values.
    let mut rng = Rng::new(43);
    for s in EulerSequence::ALL {
        let (lo, hi) = middle_range(s);
        for f in FRAMES {
            for (mid, locked) in [
                (lo + 0.8e-7, true),
                (lo + 1.25e-7, false),
                (hi - 0.8e-7, true),
                (hi - 1.25e-7, false),
            ] {
                let a = [PI * rng.signed(), mid, PI * rng.signed()];
                let x = Q::from_euler(s, f, a);
                let b = x.to_euler(s, f);
                assert_eq!(b[2] == 0.0, locked, "{s:?} {f:?} {mid}: {b:?}");
                assert!(rot_dist(Q::from_euler(s, f, b), x) <= 2e-7);
            }
        }
    }
    // And as SciPy decides it, 0.9·10⁻⁷ and 1.1·10⁻⁷ from the lock.
    let mut rng = Rng::new(12);
    let mut data = lines(SCIPY, "gimbal_edge").into_iter();
    for (s, f) in [
        (EulerSequence::XYZ, EulerFrame::Intrinsic),
        (EulerSequence::ZXZ, EulerFrame::Extrinsic),
    ] {
        let (lo, hi) = middle_range(s);
        for mid in [lo + 0.9e-7, lo + 1.1e-7, hi - 0.9e-7, hi - 1.1e-7] {
            let a = [PI * rng.signed(), mid, PI * rng.signed()];
            let got = Q::from_euler(s, f, a).to_euler(s, f);
            let want = data.next().unwrap();
            assert_eq!(got[2] == 0.0, want[2] == 0.0, "{s:?} {f:?} {mid}");
            assert!(
                angles_dist(got, &want) <= 1e-8,
                "{s:?} {f:?} {mid}: {got:?} {want:?}"
            );
        }
    }
}

#[test]
fn quaternions_near_the_largest_float_and_subnormal_ones_have_the_angles_of_their_direction() {
    let mut rng = Rng::new(44);
    for s in EulerSequence::ALL {
        for f in FRAMES {
            for _ in 0..if cfg!(miri) { 1 } else { 20 } {
                let y = rng.quaternion();
                let b = y.to_euler(s, f);
                // Parts up to 2¹⁰²³, whose sums w ± qⱼ overflow unscaled.
                let big = y * pow2(1023);
                assert!(
                    angles_dist(big.to_euler(s, f), &b) <= 16.0 * EPS,
                    "{s:?} {f:?} {y}"
                );
            }
            // Subnormal parts, exactly a power of two below small integers.
            let y = Q::new(3.0, -4.0, 1.0, 12.0);
            let b = y.to_euler(s, f);
            for e in [-1060, -1066, -1070] {
                let t = y * pow2(e);
                assert!(
                    angles_dist(t.to_euler(s, f), &b) <= 16.0 * EPS,
                    "{s:?} {f:?} {e}"
                );
            }
            let m = Q::new(f64::MAX, -f64::MAX, f64::MAX, -f64::MAX);
            let b = Q::new(1.0, -1.0, 1.0, -1.0).to_euler(s, f);
            assert!(
                angles_dist(m.to_euler(s, f), &b) <= 16.0 * EPS,
                "{s:?} {f:?}"
            );
        }
    }
}

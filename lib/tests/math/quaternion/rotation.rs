//! Rotations: against SciPy's `Rotation` and `Slerp` (`data/scipy_rotation.txt`), against
//! `q·v·q⁻¹` in 160-bit arithmetic, and their invariants.

use super::common::{Rng, angle_dist, lines, pow2, quat, rot_dist, to32, to64, vec_dist};
use super::reference;
use a::math::matrix::Matrix;
use a::math::quaternion::Quaternion;
use core::f64::consts::{FRAC_PI_2, PI};

type Q = Quaternion<f64>;
type Q32 = Quaternion<f32>;

fn q(w: f64, x: f64, y: f64, z: f64) -> Q {
    Quaternion::new(w, x, y, z)
}

const EPS: f64 = f64::EPSILON;
const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

const SCIPY: &str = include_str!("data/scipy_rotation.txt");

fn mat_dist(a: [[f64; 3]; 3], b: &[f64]) -> f64 {
    (0..9)
        .map(|k| (a[k / 3][k % 3] - b[k]).abs())
        .fold(0.0, f64::max)
}

fn rows(m: &[f64]) -> [[f64; 3]; 3] {
    [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]]
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

fn det(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn apply(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn is_nan3(v: [f64; 3]) -> bool {
    v.iter().all(|t| t.is_nan())
}

#[test]
fn rotation_matrices_agree_with_scipy_and_are_orthogonal() {
    let mut rng = Rng::new(4);
    let mut worst = [0.0f64; 3];
    for l in lines(SCIPY, "matrix") {
        let x = rng.quaternion();
        let m = x.to_rotation_matrix();
        worst[0] = worst[0].max(mat_dist(m, &l[0..9]));
        // From SciPy's matrix back to SciPy's quaternion, and to ours.
        let p = Q::from_rotation_matrix(rows(&l[0..9]));
        worst[1] = worst[1].max(rot_dist(p, quat(&l[9..13])));
        worst[2] = worst[2].max(rot_dist(p, x));
        assert!(p.w >= 0.0);
        let r = mul(m, transpose(m));
        for (i, row) in r.iter().enumerate() {
            for (j, &t) in row.iter().enumerate() {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((t - e).abs() <= 4.0 * EPS);
            }
        }
        assert!((det(m) - 1.0).abs() <= 8.0 * EPS);
    }
    eprintln!(
        "matrix: largest differences from SciPy (as_matrix, from_matrix) and of the round trip: {worst:?}"
    );
    assert!(
        worst[0] <= 4.0 * EPS && worst[1] <= 4.0 * EPS && worst[2] <= 4.0 * EPS,
        "{worst:?}"
    );
}

#[test]
fn from_rotation_matrix_agrees_with_scipy_near_its_switches_and_for_matrices_not_quite_orthogonal()
{
    let specials = lines(SCIPY, "from_matrix");
    let mut report = Vec::new();
    for (c, l) in specials.iter().enumerate() {
        let m = rows(&l[0..9]);
        let p = Q::from_rotation_matrix(m);
        let d = rot_dist(p, quat(&l[9..13]));
        report.push(d);
        // Half turns about six axes, a little short of them, the identity and nearly it: as
        // SciPy, to rounding. Then a random rotation with noise of 1e-9, 1e-6 and 1e-3 in each
        // element, and one scaled by 1.001, which SciPy orthogonalises by an SVD and this method
        // does not: the quaternions differ by about the size of the noise.
        let bound = match c {
            0..=13 => 4.0 * EPS,
            14 => 3e-9,
            15 => 3e-6,
            16 => 3e-3,
            _ => 1e-3,
        };
        assert!(d <= bound, "case {c}: {p} is not {:?} ({d})", &l[9..13]);
        assert!((p.norm() - 1.0).abs() <= 2.0 * EPS && p.w >= 0.0);
    }
    eprintln!("from_matrix: differences from SciPy, by case: {report:?}");
}

#[test]
fn the_matrix_conversions_take_3_by_3_matrices() {
    let x = Q::from_axis_angle([1.0, 2.0, 3.0], 0.4);
    let m = x.to_matrix();
    assert_eq!(m.shape(), (3, 3));
    assert_eq!(m, Matrix::from_rows(&x.to_rotation_matrix()));
    assert_eq!(
        Q::from_matrix(&m),
        Q::from_rotation_matrix(x.to_rotation_matrix())
    );
    let m32 = to32(x).to_matrix();
    assert_eq!(
        Quaternion::<f32>::from_matrix(&m32),
        Q32::from_rotation_matrix(to32(x).to_rotation_matrix())
    );
}

#[test]
#[should_panic(expected = "a rotation matrix is 3 × 3")]
fn from_matrix_panics_on_a_matrix_that_is_not_3_by_3() {
    let _ = Q::from_matrix(&Matrix::<f64>::identity(4));
}

#[test]
#[should_panic(expected = "a rotation matrix is 3 × 3")]
fn from_matrix_panics_on_a_3_by_2_matrix() {
    let _ = Q::from_matrix(&Matrix::from_rows(&[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]));
}

#[test]
fn rotation_vectors_agree_with_scipy() {
    let mut rng = Rng::new(6);
    let scales = [1e-300, 1e-12, 1e-6, 1.0, 3.0, 10.0];
    let from = lines(SCIPY, "rotvec_from");
    let mut worst = [0.0f64; 3];
    for (c, l) in from.iter().enumerate() {
        let v = [0, 1, 2].map(|_| scales[c % 6] * rng.signed());
        let p = Q::from_rotation_vector(v);
        let want = quat(l);
        // Tiny rotations keep their digits: relative to the vector part.
        let e = if c % 6 < 3 {
            (p.vector()
                .iter()
                .zip(want.vector())
                .map(|(a, b)| (a - b).abs() / b.abs().max(1e-310)))
            .fold(0.0, f64::max)
        } else {
            rot_dist(p, want)
        };
        worst[0] = worst[0].max(e);
        assert!((p.norm() - 1.0).abs() <= 2.0 * EPS);
        // And back: the rotation vector of the quaternion, for angles below π.
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n < PI && c % 6 < 3 {
            let back = p.to_rotation_vector();
            let e = (0..3)
                .map(|i| (back[i] - v[i]).abs() / v[i].abs())
                .fold(0.0, f64::max);
            assert!(e <= 4.0 * EPS, "{v:?} {back:?}");
        }
    }
    for l in lines(SCIPY, "rotvec_to") {
        let x = rng.quaternion();
        worst[1] = worst[1].max(vec_dist(x.to_rotation_vector(), [l[0], l[1], l[2]]));
    }
    for l in lines(SCIPY, "rotvec_special") {
        let x = quat(&l[0..4]);
        let (got, want) = (x.to_rotation_vector(), [l[4], l[5], l[6]]);
        let e = (0..3)
            .map(|i| (got[i] - want[i]).abs() / want[i].abs().max(1.0))
            .fold(0.0, f64::max);
        worst[2] = worst[2].max(e);
        // The relative digits of tiny vectors too.
        for i in 0..3 {
            if want[i] != 0.0 && want[i].abs() < 1e-6 {
                assert!(
                    (got[i] - want[i]).abs() <= 4.0 * EPS * want[i].abs(),
                    "{x}: {got:?} {want:?}"
                );
            }
        }
    }
    eprintln!("rotation vectors: largest differences from SciPy (from, to, special): {worst:?}");
    assert!(
        worst[0] <= 4.0 * EPS && worst[1] <= 8.0 * EPS && worst[2] <= 8.0 * EPS,
        "{worst:?}"
    );
}

/// The largest errors of `rotate` and of the round trip `rotate_inverse(rotate(v))` against
/// `q·v·q⁻¹` in 160 bits, in units of `ε·|v|`, over `count` quaternions of norms `2^±500` (seeded).
fn rotate_errors(seed: u64, count: usize) -> [f64; 2] {
    let mut rng = Rng::new(seed);
    let mut worst = [0.0f64; 2];
    for _ in 0..count {
        let x = rng.quaternion() * rng.log_uniform(-500.0, 500.0).abs();
        let v = [0, 1, 2].map(|_| rng.signed());
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let want = reference::rotate(x, v);
        worst[0] = worst[0].max(vec_dist(x.rotate(v), want) / (n * EPS));
        let back = x.rotate_inverse(x.rotate(v));
        worst[1] = worst[1].max(vec_dist(back, v) / (n * EPS));
    }
    worst
}

#[test]
#[ignore = "a heavier sweep: run it in release"]
fn a_heavier_sweep_of_rotate_against_160_bits() {
    // 200,000 quaternions more than the test below: the largest errors seen, which the page
    // quotes as observed maxima, not as bounds.
    let worst = rotate_errors(31, 200_000);
    eprintln!(
        "rotate over 200,000 quaternions of norms 2^±500, largest errors in ε·|v| (rotate, round trip): {worst:?}"
    );
    assert!(worst[0] <= 6.0 && worst[1] <= 10.0, "{worst:?}");
}

#[test]
fn rotating_vectors_agrees_with_scipy_and_with_q_v_q_inverse_in_160_bits() {
    let mut rng = Rng::new(7);
    let mut worst = [0.0f64; 3];
    for l in lines(SCIPY, "apply") {
        let x = rng.quaternion();
        let v = [0, 1, 2].map(|_| 10.0 * rng.signed());
        worst[0] = worst[0].max(vec_dist(x.rotate(v), [l[0], l[1], l[2]]) / 10.0);
        worst[0] = worst[0].max(vec_dist(x.rotate_inverse(v), [l[3], l[4], l[5]]) / 10.0);
    }
    let mut rng = Rng::new(30);
    for _ in 0..if cfg!(miri) { 20 } else { 2000 } {
        let x = rng.quaternion() * rng.log_uniform(-500.0, 500.0).abs();
        let v = [0, 1, 2].map(|_| rng.signed());
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let want = reference::rotate(x, v);
        worst[1] = worst[1].max(vec_dist(x.rotate(v), want) / (n * EPS));
        let back = x.rotate_inverse(x.rotate(v));
        worst[2] = worst[2].max(vec_dist(back, v) / (n * EPS));
        // The matrix rotates as the quaternion does.
        let m = apply(x.to_rotation_matrix(), v);
        assert!(vec_dist(m, want) <= 8.0 * EPS * n);
    }
    eprintln!(
        "rotate: largest differences from SciPy (relative), from 160 bits and of the round trip in ε: {worst:?}"
    );
    assert!(
        worst[0] <= 4.0 * EPS && worst[1] <= 4.0 && worst[2] <= 8.0,
        "{worst:?}"
    );
}

#[test]
fn the_product_composes_rotations_the_right_factor_first() {
    let mut rng = Rng::new(31);
    for _ in 0..if cfg!(miri) { 10 } else { 500 } {
        let (a, b) = (rng.quaternion(), rng.quaternion());
        let v = [rng.signed(), rng.signed(), rng.signed()];
        let (p, r) = ((a * b).rotate(v), a.rotate(b.rotate(v)));
        assert!(vec_dist(p, r) <= 16.0 * EPS);
        let m = mul(a.to_rotation_matrix(), b.to_rotation_matrix());
        assert!(mat_dist((a * b).to_rotation_matrix(), &m.concat()) <= 16.0 * EPS);
    }
    // A quarter turn about z, then about x: (Rx·Rz) takes x to z.
    let rz = Q::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let rx = Q::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_2);
    let v = (rx * rz).rotate([1.0, 0.0, 0.0]);
    assert!(vec_dist(v, [0.0, 0.0, 1.0]) <= 2.0 * EPS);
}

#[test]
fn q_and_minus_q_are_the_same_rotation_and_the_norm_does_not_matter() {
    let mut rng = Rng::new(32);
    for _ in 0..if cfg!(miri) { 5 } else { 500 } {
        let x = rng.quaternion();
        let v = [rng.signed(), rng.signed(), rng.signed()];
        // Every element of the matrix is a product of two parts: bit for bit the same.
        assert_eq!(x.to_rotation_matrix(), (-x).to_rotation_matrix());
        assert_eq!(x.rotate(v), (-x).rotate(v));
        for s in [3.0, 1e-3, pow2(600), pow2(-600), pow2(-1000), pow2(1000)] {
            let y = x * s;
            assert!(vec_dist(y.rotate(v), x.rotate(v)) <= 8.0 * EPS, "{s}");
            assert!(
                mat_dist(y.to_rotation_matrix(), &x.to_rotation_matrix().concat()) <= 8.0 * EPS
            );
            let ((a1, t1), (a2, t2)) = (y.to_axis_angle(), x.to_axis_angle());
            assert!(vec_dist(a1, a2) <= 8.0 * EPS && (t1 - t2).abs() <= 8.0 * EPS);
            assert!(vec_dist(y.to_rotation_vector(), x.to_rotation_vector()) <= 8.0 * EPS);
            assert!((y.angle() - x.angle()).abs() <= 8.0 * EPS);
            assert!(y.angle_to(x) <= 1e-7);
        }
        // Subnormal parts, exactly a power of two below those of another quaternion.
        let y = q(3.0, -4.0, 0.0, 12.0);
        let z = y * pow2(-1070);
        assert!(vec_dist(z.rotate(v), y.rotate(v)) <= 4.0 * EPS);
        assert_eq!(z.to_rotation_matrix(), y.to_rotation_matrix());
        assert_eq!(z.to_axis_angle(), y.to_axis_angle());
        let ((a1, t1), (a2, t2)) = ((-x).to_axis_angle(), x.to_axis_angle());
        assert!(vec_dist(a1, a2) <= 4.0 * EPS && (t1 - t2).abs() <= 4.0 * EPS);
        assert!(x.angle_to(-x) <= 1e-7);
    }
}

#[test]
fn half_turns_the_identity_and_axis_angle_round_trips() {
    // Half turns about the axes, and about a diagonal.
    let r = Q::K.to_rotation_matrix();
    assert_eq!(r, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]);
    assert_eq!(Q::I.rotate([1.0, 2.0, 3.0]), [1.0, -2.0, -3.0]);
    let (axis, angle) = Q::J.to_axis_angle();
    assert_eq!((axis, angle), ([0.0, 1.0, 0.0], PI));
    assert_eq!(Q::J.to_rotation_vector(), [0.0, PI, 0.0]);
    let d = Q::from_axis_angle([1.0, 1.0, 1.0], PI);
    assert!(
        vec_dist(
            d.rotate([1.0, 0.0, 0.0]),
            [-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0]
        ) <= 4.0 * EPS
    );
    for m in [Q::I, Q::J, Q::K, d] {
        let p = Q::from_rotation_matrix(m.to_rotation_matrix());
        assert!(rot_dist(p, m) <= 4.0 * EPS, "{m}");
    }
    // The identity, whatever its sign or norm.
    for x in [Q::ONE, -Q::ONE, Q::ONE * 1e-300, Q::ONE * 7.0] {
        assert_eq!(x.to_axis_angle(), ([1.0, 0.0, 0.0], 0.0));
        assert_eq!(x.to_rotation_vector(), [0.0, 0.0, 0.0]);
        assert_eq!(x.angle(), 0.0);
        assert_eq!(x.rotate([1.0, -2.0, 3.0]), [1.0, -2.0, 3.0]);
        assert_eq!(
            x.to_rotation_matrix(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
    }
    assert_eq!(
        Q::from_rotation_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        Q::ONE
    );
    assert_eq!(Q::from_rotation_vector([0.0; 3]), Q::ONE);
    assert_eq!(Q::from_axis_angle([0.0; 3], 2.0), Q::ONE);
    // Round trips of axis and angle, for angles in [0, π] and beyond.
    let mut rng = Rng::new(33);
    for _ in 0..if cfg!(miri) { 10 } else { 1000 } {
        let axis = [rng.signed(), rng.signed(), rng.signed()];
        let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        let u = axis.map(|t| t / n);
        let t = PI * rng.uniform();
        let x = Q::from_axis_angle(axis, t);
        assert!((x.norm() - 1.0).abs() <= 2.0 * EPS);
        let (a, s) = x.to_axis_angle();
        assert!((s - t).abs() <= 8.0 * EPS, "{t} {s}");
        if t > 1e-6 && t < PI - 1e-6 {
            assert!(vec_dist(a, u) <= 1e-9);
        }
        // An angle of t + 2π is the same rotation, the quaternion negated.
        assert!((Q::from_axis_angle(axis, t + 2.0 * PI) + x).norm() <= 8.0 * EPS);
        // A negative angle is the other axis.
        let (b, s) = Q::from_axis_angle(axis, -t).to_axis_angle();
        assert!((s - t).abs() <= 8.0 * EPS);
        if t > 1e-6 && t < PI - 1e-6 {
            assert!(vec_dist(b, u.map(|c| -c)) <= 1e-9);
        }
        // Rotation vectors, both ways.
        let rv = Q::from_rotation_vector(u.map(|c| c * t));
        assert!(rot_dist(rv, x) <= 4.0 * EPS);
        assert!(vec_dist(rv.to_rotation_vector(), u.map(|c| c * t)) <= 8.0 * EPS);
        // And the rotation matrix.
        assert!(rot_dist(Q::from_rotation_matrix(x.to_rotation_matrix()), x) <= 4.0 * EPS);
    }
}

#[test]
fn what_is_no_rotation_gives_nan() {
    for x in [
        Q::ZERO,
        -Q::ZERO,
        q(NAN, 1.0, 0.0, 0.0),
        q(1.0, INF, 0.0, 0.0),
        q(INF, 0.0, 0.0, 0.0),
    ] {
        assert!(is_nan3(x.rotate([1.0, 2.0, 3.0])), "{x}");
        assert!(is_nan3(x.rotate_inverse([1.0, 2.0, 3.0])));
        assert!(x.to_rotation_matrix().iter().all(|r| is_nan3(*r)));
        let (a, t) = x.to_axis_angle();
        assert!(is_nan3(a) && t.is_nan());
        assert!(is_nan3(x.to_rotation_vector()));
        assert!(x.angle().is_nan());
        assert!(x.angle_to(Q::ONE).is_nan() && Q::ONE.angle_to(x).is_nan());
        assert!(x.slerp(Q::ONE, 0.5).w.is_nan() && Q::ONE.slerp(x, 0.5).w.is_nan());
        assert!(x.nlerp(Q::ONE, 0.5).w.is_nan());
    }
    assert!(Q::from_axis_angle([1.0, NAN, 0.0], 1.0).w.is_nan());
    assert!(Q::from_axis_angle([INF, 0.0, 0.0], 1.0).x.is_nan());
    assert!(Q::from_axis_angle([1.0, 0.0, 0.0], INF).w.is_nan());
    assert!(Q::from_rotation_vector([INF, 0.0, 0.0]).w.is_nan());
    assert!(
        Q::from_rotation_matrix([[NAN, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .w
            .is_nan()
    );
    assert!(
        Q::from_rotation_arc([NAN, 0.0, 0.0], [1.0, 0.0, 0.0])
            .w
            .is_nan()
    );
    assert_eq!(Q::from_rotation_arc([0.0; 3], [1.0, 0.0, 0.0]), Q::ONE);
    // Axes of any length, from the subnormals to near overflow.
    for s in [pow2(-1074), 1e-300, 1e300, f64::MAX] {
        let x = Q::from_axis_angle([s, s, 0.0], FRAC_PI_2);
        let h = 0.5f64.sqrt();
        let want = Q::from_axis_angle([h, h, 0.0], FRAC_PI_2);
        assert!((x - want).norm() <= 2.0 * EPS, "{s}");
    }
}

#[test]
fn slerp_agrees_with_scipy_and_follows_the_shorter_arc() {
    let mut rng = Rng::new(8);
    let mut worst = [0.0f64; 2];
    for l in lines(SCIPY, "slerp") {
        let (a, b) = (rng.quaternion(), rng.quaternion());
        let t = rng.uniform();
        worst[0] = worst[0].max(rot_dist(a.slerp(b, t), quat(&l[0..4])));
    }
    for l in lines(SCIPY, "slerp_special") {
        let (t, a, b) = (l[0], quat(&l[1..5]), quat(&l[5..9]));
        let got = a.slerp(b, t);
        worst[1] = worst[1].max(rot_dist(got, quat(&l[9..13])));
        assert!((got.norm() - 1.0).abs() <= 4.0 * EPS);
    }
    eprintln!(
        "slerp: largest differences from SciPy (random, nearly equal or opposite): {worst:?}"
    );
    assert!(worst[0] <= 8.0 * EPS && worst[1] <= 8.0 * EPS, "{worst:?}");
}

#[test]
fn slerp_turns_at_a_constant_speed_and_nlerp_along_the_same_arc() {
    let mut rng = Rng::new(34);
    for _ in 0..if cfg!(miri) { 5 } else { 500 } {
        let (a, b) = (rng.quaternion().normalize(), rng.quaternion().normalize());
        let total = a.angle_to(b);
        // The ends, the shorter arc, and the angle at t a fraction t of the whole.
        assert!((a.slerp(b, 0.0) - a).norm() <= 2.0 * EPS);
        assert!(rot_dist(a.slerp(b, 1.0), b) <= 4.0 * EPS);
        assert!(a.slerp(b, 1.0).dot(a) >= 0.0);
        for t in [0.1, 0.25, 0.5, 0.9] {
            let m = a.slerp(b, t);
            assert!((a.angle_to(m) - t * total).abs() <= 1e-13, "{t}");
            assert!((m.angle_to(b) - (1.0 - t) * total).abs() <= 1e-13);
            // slerp(t) = a·(a⁻¹b)^t, with the sign of b that makes a·b ≥ 0.
            let bb = if a.dot(b) < 0.0 { -b } else { b };
            let want = a * (a.conj() * bb).powf(t);
            assert!((m - want).norm() <= 1e-14);
            // nlerp is on the same arc: between a and b, with the angles in the same order.
            let n = a.nlerp(b, t);
            assert!((n.norm() - 1.0).abs() <= 2.0 * EPS);
            assert!((a.angle_to(n) + n.angle_to(b) - total).abs() <= 1e-12);
        }
        assert!((a.nlerp(b, 0.5) - a.slerp(b, 0.5)).norm() <= 4.0 * EPS);
        // Extrapolation.
        let e = a.slerp(b, 2.0);
        assert!((a.angle_to(e) - (2.0 * total).min(2.0 * PI - 2.0 * total)).abs() <= 1e-12);
    }
    // Nearly equal and equal rotations: no NaN, and the interpolation where it should be.
    let a = Q::from_axis_angle([1.0, 2.0, 3.0], 0.7);
    let b = a * Q::from_rotation_vector([1e-12, 0.0, 0.0]);
    let m = a.slerp(b, 0.5);
    assert!((m - a * Q::from_rotation_vector([5e-13, 0.0, 0.0])).norm() <= 2.0 * EPS);
    assert_eq!(a.slerp(a, 0.3), a.normalize());
    assert_eq!(a.slerp(a * 2.0, 7.5), a.normalize());
    assert!((a.slerp(-a, 0.3) - a).norm() <= 2.0 * EPS);
    // Rotations half a turn apart (a·b = 0): defined, and on the arc through other's own sign.
    let c = a * Q::from_axis_angle([0.0, 1.0, 0.0], PI);
    assert!(a.dot(c).abs() <= 2.0 * EPS);
    let m = a.slerp(c, 0.5);
    assert!((a.angle_to(m) - FRAC_PI_2).abs() <= 1e-15);
    // Inputs of any norm are normalised first.
    let (p, r) = (
        Q::from_axis_angle([0.0, 0.0, 1.0], 0.2),
        Q::from_axis_angle([0.0, 0.0, 1.0], 1.0),
    );
    assert!((p.slerp(r, 0.5) - (p * 1e-200).slerp(r * 1e200, 0.5)).norm() <= 2.0 * EPS);
    assert!((p.nlerp(r, 0.5) - (p * 1e-200).nlerp(r * 1e200, 0.5)).norm() <= 2.0 * EPS);
}

#[test]
fn angles_agree_with_scipy_and_keep_their_digits_for_nearly_equal_rotations() {
    let mut rng = Rng::new(10);
    let mut worst = 0.0f64;
    for l in lines(SCIPY, "magnitude") {
        let (x, p) = (rng.quaternion(), rng.quaternion());
        worst = worst
            .max((x.angle() - l[0]).abs())
            .max((x.angle_to(p) - l[1]).abs());
        assert!((x.angle_to(p) - p.angle_to(x)).abs() <= 2.0 * EPS);
    }
    eprintln!("angle, angle_to: largest differences from SciPy: {worst}");
    assert!(worst <= 8.0 * EPS);
    let a = Q::from_axis_angle([0.3, -0.2, 0.9], 2.0);
    for d in [1e-5, 1e-10, 1e-14] {
        let b = a * Q::from_axis_angle([1.0, 0.0, 0.0], d);
        assert!(
            (a.angle_to(b) - d).abs() <= 4.0 * EPS,
            "{d}: {}",
            a.angle_to(b)
        );
    }
    assert_eq!(a.angle_to(a), 0.0);
    assert!((Q::ONE.angle_to(Q::K) - PI).abs() <= 2.0 * EPS);
}

#[test]
fn the_shortest_arc_takes_one_direction_to_the_other() {
    let mut rng = Rng::new(11);
    let mut worst = 0.0f64;
    for l in lines(SCIPY, "align") {
        let a = [rng.signed(), rng.signed(), rng.signed()];
        let b = [rng.signed(), rng.signed(), rng.signed()];
        // SciPy's align_vectors([a], [b]) takes b to a.
        worst = worst.max(rot_dist(Q::from_rotation_arc(b, a), quat(&l)));
    }
    eprintln!("from_rotation_arc: largest difference from SciPy's align_vectors: {worst}");
    assert!(worst <= 8.0 * EPS);
    let unit = |v: [f64; 3]| {
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        v.map(|t| t / n)
    };
    let mut rng = Rng::new(35);
    for _ in 0..if cfg!(miri) { 10 } else { 500 } {
        let a = [rng.signed(), rng.signed(), rng.signed()];
        let b = [rng.signed(), rng.signed(), rng.signed()];
        let x = Q::from_rotation_arc(a, b);
        assert!(vec_dist(x.rotate(unit(a)), unit(b)) <= 16.0 * EPS);
        // The angle is the angle between them, and the axis is perpendicular to both.
        let (axis, t) = x.to_axis_angle();
        let cos = unit(a).iter().zip(unit(b)).map(|(p, q)| p * q).sum::<f64>();
        assert!(angle_dist(t, cos.clamp(-1.0, 1.0).acos()) <= 1e-7);
        assert!(axis.iter().zip(a).map(|(p, q)| p * q).sum::<f64>().abs() <= 1e-12);
        // Opposite and nearly opposite directions: a half turn that takes a to −a.
        for d in [0.0, 1e-12, 1e-9] {
            let c = [-a[0] + d, -a[1], -a[2]];
            let y = Q::from_rotation_arc(a, c);
            assert!(vec_dist(y.rotate(unit(a)), unit(c)) <= 1e-7, "{d}");
        }
        // Parallel directions: the identity.
        assert!(rot_dist(Q::from_rotation_arc(a, a.map(|t| 3.0 * t)), Q::ONE) <= 4.0 * EPS);
    }
    for (a, b) in [
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([0.0, 0.0, 2.0], [0.0, 0.0, -1.0]),
        ([1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]),
    ] {
        let x = Q::from_rotation_arc(a, b);
        assert!(vec_dist(x.rotate(unit(a)), unit(b)) <= 4.0 * EPS, "{a:?}");
        assert!((x.angle() - PI).abs() <= 2.0 * EPS);
    }
}

#[test]
fn the_rotations_of_f32_agree_with_those_of_f64() {
    let mut rng = Rng::new(36);
    let e32 = f32::EPSILON as f64;
    for _ in 0..if cfg!(miri) { 10 } else { 500 } {
        let x = to32(rng.quaternion());
        let y = to32(rng.quaternion());
        let x64 = to64(x);
        let v = [
            rng.signed() as f32,
            rng.signed() as f32,
            rng.signed() as f32,
        ];
        let v64 = v.map(|t| t as f64);
        let r = x.rotate(v).map(|t| t as f64);
        assert!(vec_dist(r, x64.rotate(v64)) <= 4.0 * e32);
        let m = x
            .to_rotation_matrix()
            .concat()
            .iter()
            .map(|&t| t as f64)
            .collect::<Vec<_>>();
        assert!(mat_dist(x64.to_rotation_matrix(), &m) <= 4.0 * e32);
        let p = Q32::from_rotation_matrix(x.to_rotation_matrix());
        assert!(rot_dist(to64(p), x64) <= 4.0 * e32);
        assert!(rot_dist(to64(x.slerp(y, 0.3)), x64.slerp(to64(y), 0.3_f32 as f64)) <= 4.0 * e32);
        assert!((x.angle_to(y) as f64 - x64.angle_to(to64(y))).abs() <= 1e-6);
        let rv = x.to_rotation_vector().map(|t| t as f64);
        assert!(vec_dist(rv, x64.to_rotation_vector()) <= 8.0 * e32);
    }
}

#[test]
fn tiny_rotation_vectors_give_half_the_vector_exactly() {
    // Below √ε the vector part is v/2, to the last bit, in every part.
    for s in [1e-9, 1e-12, 1e-100, 1e-300, pow2(-1060)] {
        let v = [s, -2.0 * s, 3.0 * s];
        let x = Q::from_rotation_vector(v);
        assert_eq!(x.vector(), v.map(|t| t * 0.5), "{s}");
        assert!((x.w - 1.0).abs() <= EPS);
        assert_eq!(x.to_rotation_vector(), v, "{s}");
    }
    // Just above √ε, the general formula: still within a unit or two of v/2.
    let v = [1e-8, -2e-8, 3e-8];
    let x = Q::from_rotation_vector(v);
    assert!(vec_dist(x.vector(), v.map(|t| t * 0.5)) <= 1.5e-8 * 2.0 * EPS);
}

#[test]
fn the_shortest_arc_switches_to_a_half_turn_within_the_root_of_epsilon_of_opposite() {
    let s = EPS.sqrt();
    let a = [0.6, 0.0, 0.8];
    // The axis of the arc, perpendicular to a and to the half turn's a × e (e = y here, the
    // axis least aligned with a, so that a × e = (−0.8, 0, 0.6)).
    let n = [0.0, 1.0, 0.0];
    for (delta, switched) in [
        (0.5 * s, true),
        (0.7 * s, true),
        (1.4 * s, false),
        (3.0 * s, false),
        (1e-7, false),
        (1e-5, false),
        (1e-3, false),
    ] {
        let exact = Q::from_axis_angle(n, PI - delta);
        let b = exact.rotate(a);
        let x = Q::from_rotation_arc(a, b);
        // It takes a to b within the conditioning, about 3ε/δ, or δ for the half turn.
        let image = vec_dist(x.rotate(a), b);
        if switched {
            assert_eq!(x.w, 0.0, "{delta}");
            assert!(image <= 1.01 * delta, "{delta}: {image}");
            // The half turn about the unit vector of a × e, e the axis least aligned with a (y).
            let (axis, angle) = x.to_axis_angle();
            assert!(
                vec_dist(axis, [0.8, 0.0, -0.6]) <= 2.0 * EPS
                    || vec_dist(axis, [-0.8, 0.0, 0.6]) <= 2.0 * EPS,
                "{axis:?}"
            );
            assert!((angle - PI).abs() <= 2.0 * EPS);
        } else {
            // The arc about n, not the half turn about a × e, which is √2 away.
            assert!(
                rot_dist(x, exact) <= 4.0 * EPS / delta,
                "{delta}: {}",
                rot_dist(x, exact)
            );
            assert!(image <= 8.0 * EPS / delta, "{delta}: {image}");
        }
    }
}

#[test]
fn a_scaled_rotation_matrix_moves_the_quaternion_by_the_first_order_amount() {
    // (1 + η)·R moves the quaternion by about η·√(1 − p²)/(4p), p its largest part: up to 0.43·η.
    let mut rng = Rng::new(73);
    let eta = 1e-3;
    let (mut largest, mut sum, n) = (0.0f64, 0.0, if cfg!(miri) { 10 } else { 2000 });
    for _ in 0..n {
        let x = rng.quaternion().normalize();
        let m = x.to_rotation_matrix().map(|r| r.map(|t| t * (1.0 + eta)));
        let d = rot_dist(Q::from_rotation_matrix(m), x);
        let p = x.w.abs().max(x.x.abs()).max(x.y.abs()).max(x.z.abs());
        let first = eta * (1.0 - p * p).sqrt() / (4.0 * p);
        assert!((d - first).abs() <= 0.01 * first + 1e-6, "{x}: {d} {first}");
        largest = largest.max(d);
        sum += d;
    }
    eprintln!(
        "1.001·R: largest distance {largest:e}, mean {:e}, over {n}",
        sum / n as f64
    );
    assert!(largest <= 0.44 * eta);
}

#[test]
fn quaternions_far_from_norm_1_rotate_tiny_and_huge_vectors_as_unit_ones() {
    // Read unscaled, a quaternion of norm 2⁻⁵⁰⁰ would make the products of a vector of 10⁻²⁰
    // subnormal, and one of norm 2⁵⁰⁰ those of a vector of 10²⁹⁰ overflow.
    let mut rng = Rng::new(74);
    for _ in 0..if cfg!(miri) { 5 } else { 200 } {
        let x = rng.quaternion().normalize();
        for v in [
            [1e-20, 2e-20, -3e-20],
            [1e290, -2e290, 3e290],
            [rng.signed(), rng.signed(), rng.signed()],
        ] {
            let n = v[0].hypot(v[1]).hypot(v[2]);
            let want = x.rotate(v);
            for s in [pow2(-500), pow2(-300), pow2(300), pow2(500)] {
                let got = (x * s).rotate(v);
                assert!(
                    vec_dist(got, want) <= 8.0 * EPS * n,
                    "{s}: {got:?} {want:?}"
                );
                let m = (x * s).to_rotation_matrix();
                assert!(mat_dist(m, &x.to_rotation_matrix().concat()) <= 8.0 * EPS);
            }
        }
    }
}

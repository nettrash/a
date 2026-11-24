//! Quaternions as rotations: axis and angle, rotation vectors, rotating vectors, rotation
//! matrices (Shepperd's method), slerp (Shoemake), nlerp, the angle between two rotations and the
//! shortest arc from one direction to another.
//!
//! The rotation of a quaternion `q` is the active rotation `v ↦ q·v·q⁻¹`, which for
//! `q = cos(θ/2) + u·sin(θ/2)` turns by `θ` about the unit axis `u`, counterclockwise when `u`
//! points at the viewer. A quaternion is read as the rotation of `q/|q|`, whatever its norm; the
//! zero quaternion and one with a part that is not finite are no rotation, and give NaN.

use super::Quaternion;
use super::scale::{direction, exponent};
use crate::math::complex::Float;
use crate::math::matrix::Matrix;

#[inline]
fn cross<T: Float>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot<T: Float>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

impl<T: Float> Quaternion<T> {
    /// The quaternion as a rotation: itself when its squared norm is in `[1/16, 16]`, else scaled
    /// by a power of two, exactly, so that its largest part is in `[1, 2)`, which keeps the
    /// products of the rotation formulas in range; `None` for zero and for a part that is not
    /// finite.
    #[inline]
    fn as_rotation(self) -> Option<Self> {
        let s = self.norm_sqr();
        if s >= T::from_f64(0.0625) && s <= T::from_f64(16.0) {
            Some(self)
        } else {
            self.as_rotation_cold()
        }
    }

    #[cold]
    #[inline(never)]
    fn as_rotation_cold(self) -> Option<Self> {
        if self.is_finite() && !self.is_zero() {
            Some(self.ldexp(-exponent(self.max_abs())))
        } else {
            None
        }
    }

    /// The rotation by `angle` (in radians) about `axis`: `cos(angle/2) + u·sin(angle/2)` for the
    /// unit vector `u = axis/|axis|`, a unit quaternion. The axis is normalised without overflow
    /// or underflow; a zero axis gives the identity, whatever the angle, and an axis or an angle
    /// that is not finite gives NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// // A quarter turn about z takes x to y.
    /// let q = Quaternion::<f64>::from_axis_angle([0.0, 0.0, 2.0], FRAC_PI_2);
    /// let [x, y, z] = q.rotate([1.0, 0.0, 0.0]);
    /// assert!(x.abs() < 1e-15 && (y - 1.0).abs() < 1e-15 && z == 0.0);
    /// assert_eq!(Quaternion::from_axis_angle([0.0; 3], 1.0), Quaternion::ONE);
    /// ```
    pub fn from_axis_angle(axis: [T; 3], angle: T) -> Self {
        match direction(axis) {
            None => Self::ONE,
            Some(([x, _, _], _)) if x.is_nan() => Self::NAN,
            Some(([x, y, z], _)) => {
                let (sin, cos) = (angle * T::HALF).sin_cos();
                Quaternion::new(cos, x * sin, y * sin, z * sin)
            }
        }
    }

    /// The axis and the angle of the rotation: the angle `2·atan2(|v|, |w|)` in `[0, π]`, and the
    /// unit axis `v/|v|`, turned round where `w < 0` so that the angle is not above `π`. The
    /// identity (a zero vector part) gives the axis `[1, 0, 0]` and the angle 0; zero, and a
    /// part that is not finite, give NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::<f64>::from_axis_angle([0.0, 3.0, 4.0], 2.5);
    /// let (axis, angle) = q.to_axis_angle();
    /// assert!((axis[1] - 0.6).abs() < 1e-15 && (axis[2] - 0.8).abs() < 1e-15);
    /// assert!((angle - 2.5).abs() < 1e-15);
    /// // −q is the same rotation.
    /// let (axis, angle) = (-q).to_axis_angle();
    /// assert!((axis[2] - 0.8).abs() < 1e-15 && (angle - 2.5).abs() < 1e-15);
    /// ```
    pub fn to_axis_angle(self) -> ([T; 3], T) {
        let Some(q) = self.as_rotation() else {
            return ([T::NAN; 3], T::NAN);
        };
        match direction(q.vector()) {
            None => ([T::ONE, T::ZERO, T::ZERO], T::ZERO),
            Some(([x, y, z], r)) => {
                let angle = T::TWO * r.atan2(q.w.abs());
                if q.w < T::ZERO {
                    ([-x, -y, -z], angle)
                } else {
                    ([x, y, z], angle)
                }
            }
        }
    }

    /// The angle of the rotation, `2·atan2(|v|, |w|)`, in `[0, π]`; NaN for zero and for a part
    /// that is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::<f64>::from_axis_angle([1.0, 1.0, 0.0], -2.0);
    /// assert!((q.angle() - 2.0).abs() < 1e-15);
    /// assert_eq!(Quaternion::<f64>::ONE.angle(), 0.0);
    /// ```
    pub fn angle(self) -> T {
        match self.as_rotation() {
            None => T::NAN,
            Some(q) => match direction(q.vector()) {
                None => T::ZERO,
                Some((_, r)) => T::TWO * r.atan2(q.w.abs()),
            },
        }
    }

    /// The rotation of the rotation vector `v`: by the angle `|v|` about `v/|v|`, a unit
    /// quaternion. For `|v|` below `√ε`, where `sin(|v|/2)/|v|` is `1/2` to the last bit, the
    /// vector part is `v/2`, so that tiny rotations keep their digits. The zero vector gives the
    /// identity; a part that is not finite gives NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    /// use std::f64::consts::PI;
    ///
    /// let q = Quaternion::from_rotation_vector([0.0, PI, 0.0]);
    /// assert!((q - Quaternion::J).norm() < 1e-16);
    /// assert_eq!(Quaternion::from_rotation_vector([1e-20, 0.0, 0.0]).x, 5e-21);
    /// ```
    pub fn from_rotation_vector(v: [T; 3]) -> Self {
        match direction(v) {
            None => Quaternion::new(T::ONE, v[0] * T::HALF, v[1] * T::HALF, v[2] * T::HALF),
            Some(([x, y, z], angle)) => {
                let (sin, cos) = (angle * T::HALF).sin_cos();
                if angle < T::EPSILON.sqrt() {
                    Quaternion::new(cos, v[0] * T::HALF, v[1] * T::HALF, v[2] * T::HALF)
                } else {
                    Quaternion::new(cos, x * sin, y * sin, z * sin)
                }
            }
        }
    }

    /// The rotation vector: the axis times the angle, the angle in `[0, π]`, as twice the vector
    /// part of the logarithm of `±q/|q|`, the sign the one that makes `w ≥ 0`; it keeps its
    /// digits for tiny rotations, as [`ln`](Self::ln) does. The identity gives the zero vector;
    /// zero, and a part that is not finite, give NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::<f64>::from_axis_angle([0.0, 0.0, -1.0], 1.25);
    /// let v = q.to_rotation_vector();
    /// assert!(v[0] == 0.0 && v[1] == 0.0 && (v[2] + 1.25).abs() < 1e-15);
    /// assert_eq!(Quaternion::from_rotation_vector([1e-20, 0.0, 0.0]).to_rotation_vector(), [1e-20, 0.0, 0.0]);
    /// ```
    pub fn to_rotation_vector(self) -> [T; 3] {
        let Some(q) = self.as_rotation() else {
            return [T::NAN; 3];
        };
        let q = if q.w < T::ZERO { -q } else { q };
        let [x, y, z] = q.ln_vector();
        [x * T::TWO, y * T::TWO, z * T::TWO]
    }

    /// The vector `v` rotated, `q·v·q⁻¹`, as `v + (w·t + u × t)/|q|²` for `t = 2·u × v` and the
    /// vector part `u`, which is the rotation of `q/|q|` for any norm, and does not form the
    /// products of two quaternions. A zero quaternion, or one with a part that is not finite,
    /// gives NaN. For a vector longer than about a sixty-fourth of the largest float the
    /// intermediate products may overflow.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// let a = Quaternion::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_2);
    /// let b = Quaternion::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    /// let v = [1.0, 2.0, 3.0];
    /// // The product composes, the right factor first.
    /// let (p, q) = ((a * b).rotate(v), a.rotate(b.rotate(v)));
    /// assert!((0..3).all(|i| (p[i] - q[i]).abs() < 1e-15));
    /// // The norm does not matter.
    /// let r = (a * 3.0).rotate(v);
    /// assert!((0..3).all(|i| (r[i] - a.rotate(v)[i]).abs() < 1e-15));
    /// ```
    pub fn rotate(&self, v: [T; 3]) -> [T; 3] {
        let Some(q) = self.as_rotation() else {
            return [T::NAN; 3];
        };
        let n2 = q.norm_sqr();
        let u = q.vector();
        let t = cross(u, v);
        let t = [t[0] * T::TWO, t[1] * T::TWO, t[2] * T::TWO];
        let c = cross(u, t);
        [
            v[0] + (q.w * t[0] + c[0]) / n2,
            v[1] + (q.w * t[1] + c[1]) / n2,
            v[2] + (q.w * t[2] + c[2]) / n2,
        ]
    }

    /// The vector `v` rotated by the inverse rotation, `q⁻¹·v·q`: the rotation of the
    /// conjugate, which undoes [`rotate`](Self::rotate).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::<f64>::from_axis_angle([1.0, 2.0, 3.0], 0.7);
    /// let v = q.rotate_inverse(q.rotate([4.0, 5.0, 6.0]));
    /// assert!((v[0] - 4.0).abs() < 1e-14 && (v[1] - 5.0).abs() < 1e-14 && (v[2] - 6.0).abs() < 1e-14);
    /// ```
    pub fn rotate_inverse(&self, v: [T; 3]) -> [T; 3] {
        self.conj().rotate(v)
    }

    /// The rotation matrix, its rows in the array: `R·v` is [`rotate`](Self::rotate)`(v)`, with
    /// `R = I + 2/|q|²·(w·[u]ₓ + [u]ₓ²)` for the vector part `u`, so that a quaternion of any
    /// norm gives an orthogonal matrix of determinant 1, to rounding, and `q` and `−q` the same
    /// one. A zero quaternion, or one with a part that is not finite, gives NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// // Half a turn about z.
    /// let r = Quaternion::new(0.0, 0.0, 0.0, 1.0).to_rotation_matrix();
    /// assert_eq!(r, [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]);
    /// ```
    pub fn to_rotation_matrix(self) -> [[T; 3]; 3] {
        let Some(q) = self.as_rotation() else {
            return [[T::NAN; 3]; 3];
        };
        let s = T::TWO / q.norm_sqr();
        let (x2, y2, z2) = (q.x * s, q.y * s, q.z * s);
        let (xx, yy, zz) = (q.x * x2, q.y * y2, q.z * z2);
        let (xy, xz, yz) = (q.x * y2, q.x * z2, q.y * z2);
        let (wx, wy, wz) = (q.w * x2, q.w * y2, q.w * z2);
        let one = T::ONE;
        [
            [one - (yy + zz), xy - wz, xz + wy],
            [xy + wz, one - (xx + zz), yz - wx],
            [xz - wy, yz + wx, one - (xx + yy)],
        ]
    }

    /// The unit quaternion of a rotation matrix (its rows in the array), with `w ≥ 0`, by
    /// Shepperd's method: of `4w² = 1 + tr R`, `4x² = 1 + 2R₀₀ − tr R`, `4y²` and `4z²`, the
    /// largest picks the column of `4·q·qᵀ` that is taken from `R`, sums and differences of
    /// its elements, and normalised (S. W. Shepperd, *Quaternion from rotation matrix*,
    /// J. Guidance and Control 1, 1978). The largest of the four is at least 1, so the column
    /// that is normalised is never small.
    ///
    /// A matrix that is not orthogonal is not made so: for one within `δ` of a rotation, the
    /// quaternion is within about `δ` of that rotation's, but it is not, in general, the
    /// rotation nearest in the Frobenius norm, which SciPy's `from_matrix` finds by a singular
    /// value decomposition for a matrix it does not take to be orthogonal; nor is the method
    /// blind to scale: to first order, `(1 + η)·R` moves the quaternion by `η·√(1 − p²)/(4p)` for
    /// its largest part `p ≥ ½`, up to about `0.43·η`, so that `1.001·R` gives a quaternion up to
    /// about `4·10⁻⁴` from that of `R`. A
    /// matrix far from a rotation (a reflection, a large scaling) gives a unit quaternion with
    /// no particular relation to it. A part that is not finite gives NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::from_axis_angle([1.0, -2.0, 0.5], 2.9);
    /// let p = Quaternion::from_rotation_matrix(q.to_rotation_matrix());
    /// assert!((p - q).norm() < 1e-15 || (p + q).norm() < 1e-15);
    /// assert!(p.w >= 0.0);
    /// ```
    pub fn from_rotation_matrix(m: [[T; 3]; 3]) -> Self {
        let tr = m[0][0] + m[1][1] + m[2][2];
        let one = T::ONE;
        let q = if tr >= m[0][0] && tr >= m[1][1] && tr >= m[2][2] {
            Quaternion::new(
                one + tr,
                m[2][1] - m[1][2],
                m[0][2] - m[2][0],
                m[1][0] - m[0][1],
            )
        } else if m[0][0] >= m[1][1] && m[0][0] >= m[2][2] {
            Quaternion::new(
                m[2][1] - m[1][2],
                one + m[0][0] - m[1][1] - m[2][2],
                m[0][1] + m[1][0],
                m[0][2] + m[2][0],
            )
        } else if m[1][1] >= m[2][2] {
            Quaternion::new(
                m[0][2] - m[2][0],
                m[0][1] + m[1][0],
                one - m[0][0] + m[1][1] - m[2][2],
                m[1][2] + m[2][1],
            )
        } else {
            Quaternion::new(
                m[1][0] - m[0][1],
                m[0][2] + m[2][0],
                m[1][2] + m[2][1],
                one - m[0][0] - m[1][1] + m[2][2],
            )
        };
        let q = q.normalize();
        if q.w < T::ZERO { -q } else { q }
    }

    /// The rotation matrix as a 3 × 3 [`Matrix`], as
    /// [`to_rotation_matrix`](Self::to_rotation_matrix) gives it.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::<f64>::from_axis_angle([0.0, 1.0, 0.0], 0.3);
    /// let m = q.to_matrix();
    /// assert_eq!(m, Matrix::from_rows(&q.to_rotation_matrix()));
    /// // Orthogonal, to rounding.
    /// let e = &(&m * &m.transpose()) - &Matrix::identity(3);
    /// assert!(e.as_slice().iter().all(|t| t.abs() < 1e-15));
    /// ```
    pub fn to_matrix(self) -> Matrix<T> {
        Matrix::from_rows(&self.to_rotation_matrix())
    }

    /// The unit quaternion of a 3 × 3 rotation [`Matrix`], as
    /// [`from_rotation_matrix`](Self::from_rotation_matrix) takes it.
    ///
    /// # Panics
    ///
    /// Panics if the matrix is not 3 × 3.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::matrix::Matrix;
    /// use a::math::quaternion::Quaternion;
    ///
    /// let m = Matrix::from_rows(&[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]);
    /// let q = Quaternion::from_matrix(&m);
    /// let h = 0.5f64.sqrt();
    /// assert!((q - Quaternion::new(h, 0.0, 0.0, h)).norm() < 1e-15);
    /// ```
    #[track_caller]
    pub fn from_matrix(m: &Matrix<T>) -> Self {
        assert_eq!(m.shape(), (3, 3), "a rotation matrix is 3 × 3");
        let e = |i: usize, j: usize| m[(i, j)];
        Self::from_rotation_matrix([
            [e(0, 0), e(0, 1), e(0, 2)],
            [e(1, 0), e(1, 1), e(1, 2)],
            [e(2, 0), e(2, 1), e(2, 2)],
        ])
    }

    /// The spherical linear interpolation from `self` (at `t = 0`) to `other` (at `t = 1`) along
    /// the shorter arc, at a constant angular speed (K. Shoemake, *Animating rotation with
    /// quaternion curves*, SIGGRAPH 1985): `sin((1 − t)Ω)/sin Ω·a + sin(tΩ)/sin Ω·b` for the unit
    /// quaternions `a = self/|self|` and `b = ±other/|other|`, the sign that makes `a·b ≥ 0` (for
    /// `a·b = 0`, two arcs of the same length, `other`'s own). The angle `Ω = 2·atan2(|a − b|,
    /// |a + b|)` keeps its digits for nearly equal rotations, where `arccos(a·b)` would not, and
    /// the ratios of sines have no cancellation; equal rotations give `a`. With the sign chosen,
    /// `Ω ≤ π/2`, so that `sin Ω` is never small but for small `Ω`. A `t` outside `[0, 1]`
    /// extrapolates. Zero, and a part that is not finite, give NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let a = Quaternion::from_axis_angle([0.0, 0.0, 1.0], 0.0);
    /// let b = Quaternion::from_axis_angle([0.0, 0.0, 1.0], 2.0);
    /// let m = a.slerp(b, 0.25);
    /// assert!((m - Quaternion::from_axis_angle([0.0, 0.0, 1.0], 0.5)).norm() < 1e-15);
    /// // −b is the same rotation, and the arc the same.
    /// assert!((a.slerp(-b, 0.25) - m).norm() < 1e-15);
    /// ```
    pub fn slerp(self, other: Self, t: T) -> Self {
        let a = self.normalize();
        let mut b = other.normalize();
        if a.dot(b) < T::ZERO {
            b = -b;
        }
        let omega = T::TWO * (a - b).norm().atan2((a + b).norm());
        if omega == T::ZERO {
            // |a − b| = 0: the two are equal, part for part.
            return a;
        }
        let s = omega.sin_cos().0;
        let c0 = ((T::ONE - t) * omega).sin_cos().0 / s;
        let c1 = (t * omega).sin_cos().0 / s;
        a * c0 + b * c1
    }

    /// The normalised linear interpolation from `self` (at `t = 0`) to `other` (at `t = 1`),
    /// `((1 − t)·a + t·b)/|(1 − t)·a + t·b|` for `a = self/|self|` and `b = ±other/|other|`, the
    /// sign that makes `a·b ≥ 0`, as [`slerp`](Self::slerp) takes them: the same path, at a speed
    /// that is not constant. Zero, and a part that is not finite, give NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let a = Quaternion::<f64>::from_axis_angle([0.0, 1.0, 0.0], 0.0);
    /// let b = Quaternion::from_axis_angle([0.0, 1.0, 0.0], 1.0);
    /// // Halfway, nlerp and slerp agree.
    /// assert!((a.nlerp(b, 0.5) - a.slerp(b, 0.5)).norm() < 1e-15);
    /// assert!((a.nlerp(b, 0.25).norm() - 1.0).abs() < 1e-15);
    /// ```
    pub fn nlerp(self, other: Self, t: T) -> Self {
        let a = self.normalize();
        let mut b = other.normalize();
        if a.dot(b) < T::ZERO {
            b = -b;
        }
        (a * (T::ONE - t) + b * t).normalize()
    }

    /// The angle of the rotation that takes the rotation `self` to `other`, of `self⁻¹·other`, in
    /// `[0, π]`: `2·atan2(|v|, |w|)` of `conj(self)·other`, whose norm does not matter. NaN for a
    /// zero quaternion and for a part that is not finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let a = Quaternion::<f64>::from_axis_angle([1.0, 0.0, 0.0], 0.5);
    /// let b = Quaternion::from_axis_angle([1.0, 0.0, 0.0], 2.0);
    /// assert!((a.angle_to(b) - 1.5).abs() < 1e-15);
    /// assert!((a.angle_to(-b * 7.0) - 1.5).abs() < 1e-15);
    /// ```
    pub fn angle_to(self, other: Self) -> T {
        let (Some(p), Some(q)) = (self.as_rotation(), other.as_rotation()) else {
            return T::NAN;
        };
        let d = p.conj() * q;
        match direction(d.vector()) {
            None => T::ZERO,
            Some((_, r)) => T::TWO * r.atan2(d.w.abs()),
        }
    }

    /// The rotation by the smallest angle that takes the direction of `from` to that of `to`: by
    /// the angle between them about `from × to`, as the unit quaternion of `1 + a·b + a × b` for
    /// the unit vectors `a` and `b` of the two directions. At the angle `π − δ`, `1 + a·b` is
    /// about `δ²/2` and carries an error of about `ε` from the rounding of `a·b`, so that the
    /// quaternion's error grows like `ε/δ`, the conditioning of the problem; within `|a × b| < √ε`
    /// of opposite directions, where `δ²/2` is below that error and `1 + a·b` is rounding alone,
    /// the result is the half turn about the unit vector of `a × e`, `e` the axis of the
    /// coordinate system least aligned with `a`, which takes `a` to within `δ` of `b`. A zero
    /// vector gives the identity; a part that is not finite gives NaN.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::Quaternion;
    ///
    /// let q = Quaternion::<f64>::from_rotation_arc([1.0, 0.0, 0.0], [0.0, 2.0, 2.0]);
    /// let [x, y, z] = q.rotate([1.0, 0.0, 0.0]);
    /// let h = 0.5f64.sqrt();
    /// assert!(x.abs() < 1e-15 && (y - h).abs() < 1e-15 && (z - h).abs() < 1e-15);
    /// // Opposite directions: a half turn.
    /// let r = Quaternion::from_rotation_arc([0.0, 0.0, 1.0], [0.0, 0.0, -3.0]);
    /// assert!((r.angle() - std::f64::consts::PI).abs() < 1e-15);
    /// ```
    pub fn from_rotation_arc(from: [T; 3], to: [T; 3]) -> Self {
        let (Some((a, _)), Some((b, _))) = (direction(from), direction(to)) else {
            return Self::ONE;
        };
        let d = dot(a, b);
        let c = cross(a, b);
        if d < T::ZERO && c[0].hypot(c[1]).hypot(c[2]) < T::EPSILON.sqrt() {
            let m = [a[0].abs(), a[1].abs(), a[2].abs()];
            let e = if m[0] <= m[1] && m[0] <= m[2] {
                [T::ONE, T::ZERO, T::ZERO]
            } else if m[1] <= m[2] {
                [T::ZERO, T::ONE, T::ZERO]
            } else {
                [T::ZERO, T::ZERO, T::ONE]
            };
            let Some((n, _)) = direction(cross(a, e)) else {
                return Self::NAN;
            };
            return Quaternion::from_parts(T::ZERO, n);
        }
        Quaternion::from_parts(T::ONE + d, c).normalize()
    }
}

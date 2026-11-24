//! Euler angles in the twelve sequences of axes, intrinsic or extrinsic, with the conventions of
//! SciPy's `Rotation.from_euler` and `as_euler`; the angles of a quaternion by the method of
//! E. Bernardes and S. Viollet, *Quaternion to Euler angles conversion: a direct, general and
//! computationally efficient method*, PLOS ONE 17(11), 2022, which SciPy uses too.

use super::Quaternion;
use crate::math::complex::Float;

/// A sequence of three axes about which three rotations turn, in the order they are applied:
/// the six Tait–Bryan sequences of three different axes and the six proper Euler sequences
/// whose first and last axes are the same. With an [`EulerFrame`] it is one of SciPy's 24
/// strings: [`XYZ`](Self::XYZ) intrinsic is SciPy's `"XYZ"`, extrinsic its `"xyz"`.
///
/// # Examples
///
/// ```
/// use a::math::quaternion::EulerSequence;
///
/// assert_eq!(EulerSequence::ZYX.axes(), [2, 1, 0]);
/// assert!(EulerSequence::ZXZ.is_proper());
/// assert_eq!(EulerSequence::ALL.len(), 12);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EulerSequence {
    /// About x, then y, then z.
    XYZ,
    /// About x, then z, then y.
    XZY,
    /// About y, then x, then z.
    YXZ,
    /// About y, then z, then x.
    YZX,
    /// About z, then x, then y.
    ZXY,
    /// About z, then y, then x: yaw, pitch and roll, intrinsic.
    ZYX,
    /// About x, then y, then x.
    XYX,
    /// About x, then z, then x.
    XZX,
    /// About y, then x, then y.
    YXY,
    /// About y, then z, then y.
    YZY,
    /// About z, then x, then z: the classical Euler angles.
    ZXZ,
    /// About z, then y, then z.
    ZYZ,
}

impl EulerSequence {
    /// The twelve sequences: the six Tait–Bryan ones, then the six proper ones.
    pub const ALL: [EulerSequence; 12] = [
        EulerSequence::XYZ,
        EulerSequence::XZY,
        EulerSequence::YXZ,
        EulerSequence::YZX,
        EulerSequence::ZXY,
        EulerSequence::ZYX,
        EulerSequence::XYX,
        EulerSequence::XZX,
        EulerSequence::YXY,
        EulerSequence::YZY,
        EulerSequence::ZXZ,
        EulerSequence::ZYZ,
    ];

    /// The three axes, in order: 0 for x, 1 for y and 2 for z.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::EulerSequence;
    ///
    /// assert_eq!(EulerSequence::YZY.axes(), [1, 2, 1]);
    /// ```
    pub const fn axes(self) -> [usize; 3] {
        match self {
            EulerSequence::XYZ => [0, 1, 2],
            EulerSequence::XZY => [0, 2, 1],
            EulerSequence::YXZ => [1, 0, 2],
            EulerSequence::YZX => [1, 2, 0],
            EulerSequence::ZXY => [2, 0, 1],
            EulerSequence::ZYX => [2, 1, 0],
            EulerSequence::XYX => [0, 1, 0],
            EulerSequence::XZX => [0, 2, 0],
            EulerSequence::YXY => [1, 0, 1],
            EulerSequence::YZY => [1, 2, 1],
            EulerSequence::ZXZ => [2, 0, 2],
            EulerSequence::ZYZ => [2, 1, 2],
        }
    }

    /// Whether it is a proper Euler sequence, its first and last axes the same, rather than a
    /// Tait–Bryan one.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::EulerSequence;
    ///
    /// assert!(EulerSequence::XYX.is_proper());
    /// assert!(!EulerSequence::XYZ.is_proper());
    /// ```
    pub const fn is_proper(self) -> bool {
        let a = self.axes();
        a[0] == a[2]
    }
}

/// Whether the axes of an [`EulerSequence`] turn with the body or stay fixed.
///
/// # Examples
///
/// ```
/// use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
///
/// // Intrinsic x-y-z is extrinsic z-y-x with the angles reversed.
/// let a = Quaternion::from_euler(EulerSequence::XYZ, EulerFrame::Intrinsic, [0.1, 0.2, 0.3]);
/// let b = Quaternion::from_euler(EulerSequence::ZYX, EulerFrame::Extrinsic, [0.3, 0.2, 0.1]);
/// assert!((a - b).norm() < 1e-16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EulerFrame {
    /// Each rotation is about an axis of the body as the rotations before it have left it:
    /// `q = q₁·q₂·q₃`. SciPy's upper-case sequences.
    Intrinsic,
    /// Each rotation is about an axis of the fixed frame: `q = q₃·q₂·q₁`. SciPy's lower-case
    /// sequences.
    Extrinsic,
}

/// SciPy's tolerance for gimbal lock: the middle angle within `10⁻⁷` of a value where the first
/// and the third angles turn about one axis.
const GIMBAL: f64 = 1e-7;

/// An angle in `[−2π, 2π]` brought into `[−π, π]`.
#[inline]
fn wrap<T: Float>(a: T) -> T {
    let tau = T::PI * T::TWO;
    if a > T::PI {
        a - tau
    } else if a < -T::PI {
        a + tau
    } else {
        a
    }
}

impl<T: Float> Quaternion<T> {
    /// The rotation of the Euler angles `angles` (in radians), the first about the first axis
    /// of `sequence`: the product of the three rotations `qᵢ = cos(θᵢ/2) + eᵢ·sin(θᵢ/2)` about
    /// the axes `eᵢ`, `q₁·q₂·q₃` for an intrinsic sequence and `q₃·q₂·q₁` for an extrinsic one,
    /// as SciPy's `Rotation.from_euler`. A unit quaternion, to rounding; any angles are taken.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
    /// use std::f64::consts::FRAC_PI_2;
    ///
    /// // Yaw by a quarter turn about z, then pitch by a quarter turn about the new y.
    /// let q = Quaternion::from_euler(EulerSequence::ZYX, EulerFrame::Intrinsic, [FRAC_PI_2, FRAC_PI_2, 0.0]);
    /// let [x, y, z] = q.rotate([0.0, 0.0, 1.0]);
    /// assert!((x - 0.0).abs() < 1e-15 && (y - 1.0).abs() < 1e-15 && z.abs() < 1e-15);
    /// ```
    pub fn from_euler(sequence: EulerSequence, frame: EulerFrame, angles: [T; 3]) -> Self {
        let axes = sequence.axes();
        let e = |i: usize| {
            let (sin, cos) = (angles[i] * T::HALF).sin_cos();
            let mut v = [T::ZERO; 3];
            v[axes[i]] = sin;
            Quaternion::from_parts(cos, v)
        };
        match frame {
            EulerFrame::Intrinsic => e(0) * e(1) * e(2),
            EulerFrame::Extrinsic => e(2) * e(1) * e(0),
        }
    }

    /// The Euler angles of the rotation in `sequence` and `frame`, as SciPy's
    /// `Rotation.as_euler`, by the method of Bernardes and Viollet: for the axes `(i, j, k)` of an
    /// extrinsic sequence (an intrinsic one is the extrinsic one of its axes reversed, with its
    /// angles reversed), the parts `a, b, c, d` of the quaternion (`w, qᵢ, qⱼ, ±qₖ` for a proper
    /// sequence, and for a Tait–Bryan one those of the quaternion turned by a quarter turn about
    /// `j`, `w − qⱼ, qᵢ ± qₖ, qⱼ + w, ±qₖ − qᵢ`) give the middle angle
    /// `2·atan2(√(c² + d²), √(a² + b²))`, less `π/2` for a Tait–Bryan sequence, and the others
    /// from the half sum and the half difference `atan2(b, a) ± atan2(d, c)`: no inverse cosine
    /// and no matrix, and accurate everywhere but next to gimbal lock, where the first and the
    /// third angles are themselves ill-conditioned.
    ///
    /// The first and the third angles are in `[−π, π]`; the second in `[−π/2, π/2]` for a
    /// Tait–Bryan sequence, in `[0, π]` for a proper one. At gimbal lock, where the middle angle
    /// is within `10⁻⁷` of `0` or `π` before the shift (SciPy's tolerance), the first and the
    /// third rotations are about one axis and only their sum or difference is determined: the
    /// third angle is then 0, and the first carries the rotation, as SciPy sets them (SciPy also
    /// warns). The angles are those of `q/|q|`; zero, and a part that is not finite, give NaN.
    ///
    /// SciPy's angles differ in two ways that are the same rotation: SciPy brings every angle
    /// into `[−π, π)` with a remainder, so that it gives `−π` where these are `π`, and the
    /// remainder rounds them, by up to about `4·10⁻¹⁶` for `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::quaternion::{EulerFrame, EulerSequence, Quaternion};
    ///
    /// let (s, f) = (EulerSequence::ZXZ, EulerFrame::Extrinsic);
    /// let q = Quaternion::<f64>::from_euler(s, f, [0.5, 1.0, -2.0]);
    /// let [a, b, c] = q.to_euler(s, f);
    /// assert!((a - 0.5).abs() < 1e-15 && (b - 1.0).abs() < 1e-15 && (c + 2.0).abs() < 1e-15);
    /// // Gimbal lock: the middle angle 0, and the third angle set to 0.
    /// let q = Quaternion::<f64>::from_euler(s, f, [0.5, 0.0, 0.25]);
    /// let [a, b, c] = q.to_euler(s, f);
    /// assert!((a - 0.75).abs() < 1e-15 && b == 0.0 && c == 0.0);
    /// ```
    pub fn to_euler(self, sequence: EulerSequence, frame: EulerFrame) -> [T; 3] {
        let Some(q) = self.as_euler_rotation() else {
            return [T::NAN; 3];
        };
        let s = sequence.axes();
        let extrinsic = frame == EulerFrame::Extrinsic;
        let (i, j, k) = if extrinsic {
            (s[0], s[1], s[2])
        } else {
            (s[2], s[1], s[0])
        };
        let proper = i == k;
        let k = if proper { 3 - i - j } else { k };
        // The sign of the permutation (i, j, k) of (0, 1, 2).
        let odd = (i + 1) % 3 != j;
        let sign = if odd { -T::ONE } else { T::ONE };
        let p = q.vector();
        let (a, b, c, d) = if proper {
            (q.w, p[i], p[j], p[k] * sign)
        } else {
            (
                q.w - p[j],
                p[i] + p[k] * sign,
                p[j] + q.w,
                p[k] * sign - p[i],
            )
        };
        let middle = T::TWO * c.hypot(d).atan2(a.hypot(b));
        let half_sum = b.atan2(a);
        let half_diff = d.atan2(c);
        let lock = T::from_f64(GIMBAL);
        // The first and the third angle of the extrinsic sequence (i, j, k).
        let (first, third) = if middle <= lock {
            if extrinsic {
                (T::TWO * half_sum, T::ZERO)
            } else {
                (T::ZERO, T::TWO * half_sum)
            }
        } else if (middle - T::PI).abs() <= lock {
            if extrinsic {
                (-T::TWO * half_diff, T::ZERO)
            } else {
                (T::ZERO, T::TWO * half_diff)
            }
        } else {
            (half_sum - half_diff, half_sum + half_diff)
        };
        let (middle, third) = if proper {
            (middle, third)
        } else {
            (middle - T::PI * T::HALF, third * sign)
        };
        let (first, third) = (wrap(first), wrap(third));
        if extrinsic {
            [first, middle, third]
        } else {
            [third, middle, first]
        }
    }

    /// The quaternion scaled for the sums and differences of its parts, or `None` for no
    /// rotation.
    #[inline]
    fn as_euler_rotation(self) -> Option<Self> {
        if self.is_finite() && !self.is_zero() {
            Some(self.balanced())
        } else {
            None
        }
    }
}

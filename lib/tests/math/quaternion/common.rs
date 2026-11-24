//! What the quaternion tests share: the generator of their inputs, which the data generators
//! repeat (`data/gen_common.py`), the reading of the data files, and measures of error.

use a::math::quaternion::Quaternion;

/// SplitMix64, as `data/gen_common.py` has it, so that the tests make the inputs the reference
/// outputs were computed from.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`: the top 53 bits times `2⁻⁵³`.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }

    /// Uniform in `[−1, 1)`, exactly `2u − 1`.
    pub fn signed(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }

    /// A quaternion of parts in `[−1, 1)`.
    pub fn quaternion(&mut self) -> Quaternion<f64> {
        Quaternion::new(self.signed(), self.signed(), self.signed(), self.signed())
    }

    /// A float of a random sign whose magnitude has a base-2 logarithm uniform in `[lo, hi)`.
    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let e = lo + (hi - lo) * self.uniform();
        let s = if self.next_u64() & 1 == 0 { 1.0 } else { -1.0 };
        s * e.exp2()
    }
}

/// `2^e` for `e` from −1074 to 1023, exactly: `powi` gives 0 for the subnormal powers in a debug
/// build, where it divides by a power that overflows.
pub fn pow2(e: i32) -> f64 {
    if e >= -1022 {
        2f64.powi(e)
    } else {
        f64::MIN_POSITIVE * 2f64.powi(e + 1022)
    }
}

/// `2^e` for `e` from −149 to 127, exactly.
pub fn pow2f(e: i32) -> f32 {
    if e >= -126 {
        2f32.powi(e)
    } else {
        f32::MIN_POSITIVE * 2f32.powi(e + 126)
    }
}

/// The lines of a data file with the tag, each as its numbers, in order.
pub fn lines(text: &str, tag: &str) -> Vec<Vec<f64>> {
    text.lines()
        .filter(|l| l.split_whitespace().next() == Some(tag))
        .map(|l| {
            l.split_whitespace()
                .skip(1)
                .map(|t| t.parse::<f64>().expect("a number"))
                .collect()
        })
        .collect()
}

pub fn quat(p: &[f64]) -> Quaternion<f64> {
    Quaternion::new(p[0], p[1], p[2], p[3])
}

/// `|a − b| / |b|`, or `|a − b|` where `b` is smaller than 1: the normwise error, relative where
/// it can be.
pub fn rel(a: Quaternion<f64>, b: Quaternion<f64>) -> f64 {
    (a - b).norm() / b.norm().max(f64::MIN_POSITIVE)
}

/// The distance between two rotations' quaternions, which `q` and `−q` both are.
pub fn rot_dist(a: Quaternion<f64>, b: Quaternion<f64>) -> f64 {
    let (a, b) = (a.normalize(), b.normalize());
    (a - b).norm().min((a + b).norm())
}

/// The largest difference of two vectors' parts.
pub fn vec_dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f64::max)
}

/// The difference of two angles modulo `2π`, in `[0, π]`.
pub fn angle_dist(a: f64, b: f64) -> f64 {
    let tau = 2.0 * core::f64::consts::PI;
    let d = (a - b).rem_euclid(tau);
    d.min(tau - d)
}

/// Whether `a` and `b` are equal part by part, the signs of zeros and NaNs included.
pub fn identical(a: Quaternion<f64>, b: Quaternion<f64>) -> bool {
    let same = |x: f64, y: f64| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
    same(a.w, b.w) && same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z)
}

/// The same for `f32`.
pub fn identical32(a: Quaternion<f32>, b: Quaternion<f32>) -> bool {
    let same = |x: f32, y: f32| (x.is_nan() && y.is_nan()) || x.to_bits() == y.to_bits();
    same(a.w, b.w) && same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z)
}

/// The quaternion of `f32` parts nearest to an `f64` one.
pub fn to32(q: Quaternion<f64>) -> Quaternion<f32> {
    Quaternion::new(q.w as f32, q.x as f32, q.y as f32, q.z as f32)
}

/// An `f32` quaternion's parts, exactly, as `f64`.
pub fn to64(q: Quaternion<f32>) -> Quaternion<f64> {
    Quaternion::new(q.w as f64, q.x as f64, q.y as f64, q.z as f64)
}

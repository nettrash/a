//! Twiddle factors: the roots of unity `e^(−2πi·k/n)`, from an angle reduced exactly to at most
//! π/4 by the symmetries of the circle, carried as a double-double, so that each factor is as
//! accurate as the system's `sin` and `cos` of an `f64` (within about half a unit in the last
//! place) and the symmetric factors are exact images of each other.

use crate::math::complex::{Complex, Float};

/// `2π` as a double-double.
const TAU_HI: f64 = core::f64::consts::TAU;
const TAU_LO: f64 = 2.449_293_598_294_706_4e-16;

/// The nearest quarter turn `q` (0..=4, 4 a full turn) to `k/n` and the offset `num/(4n)` of
/// a turn from it, `|num| ≤ n/2`, for `k < n`.
#[inline]
fn reduce(k: usize, n: usize) -> (u32, i64) {
    // q = ⌈octant/2⌉ for the octant ⌊8k/n⌋, by comparisons rather than a division; in u128
    // only for lengths of 2^60 and more.
    if (n as u64) < 1 << 60 {
        let (k8, n) = (8 * k as u64, n as u64);
        let q =
            (k8 >= n) as u32 + (k8 >= 3 * n) as u32 + (k8 >= 5 * n) as u32 + (k8 >= 7 * n) as u32;
        (q, (k8 / 2) as i64 - q as i64 * n as i64)
    } else {
        let (k8, n) = (8 * k as u128, n as u128);
        let q =
            (k8 >= n) as u32 + (k8 >= 3 * n) as u32 + (k8 >= 5 * n) as u32 + (k8 >= 7 * n) as u32;
        (q, ((k8 / 2) as i128 - q as i128 * n as i128) as i64)
    }
}

/// `(cos, sin)` of `2π·num/(4n)` for `|num| ≤ n/2` (an angle of at most π/4), the angle
/// carried as a double-double and the result corrected by its low part.
#[inline]
fn reduced(num: i64, n: usize) -> (f64, f64) {
    // num and 4n are exact in f64 for n < 2^51.
    let (num, den) = (num as f64, 4.0 * n as f64);
    // t = 2π·num/den as hi + lo.
    let f = num / den;
    let f_lo = (-f).mul_add(den, num) / den;
    let hi = TAU_HI * f;
    let lo = TAU_HI.mul_add(f, -hi) + TAU_HI * f_lo + TAU_LO * f;
    let (sh, ch) = hi.sin_cos();
    // sin(hi + lo) = sin hi + lo·cos hi and cos(hi + lo) = cos hi − lo·sin hi, to within lo².
    ((-lo).mul_add(sh, ch), lo.mul_add(ch, sh))
}

/// `e^(−2πi·k/n)` from `(cos, sin)` of the offset from the quarter turn `q`.
#[inline]
fn rotate(q: u32, (ct, st): (f64, f64)) -> (f64, f64) {
    // (cos, sin) of 2π·k/n = the rotation of (ct, st) by q quarter turns.
    let (c, s) = match q % 4 {
        0 => (ct, st),
        1 => (-st, ct),
        2 => (-ct, -st),
        _ => (st, -ct),
    };
    (c, -s)
}

/// `e^(−2πi·k/n)` for `n ≥ 1`, in `f64`.
pub(crate) fn root(k: usize, n: usize) -> (f64, f64) {
    debug_assert!(n >= 1);
    let (q, num) = reduce(k % n, n);
    rotate(q, reduced(num, n))
}

/// The roots `e^(−2πi·k/n)` of one `n`, any `k`, from a table of the angles of at most π/4:
/// `(cos, sin)` of `2π·u/(4n)` for `0 ≤ u ≤ n/2`, `u` a multiple of `gcd(4, n)` (the offsets
/// from the quarter turns are), so `n/8 + 1` entries for a multiple of 4, `⌊n/4⌋ + 1` for twice
/// an odd number and `⌊n/2⌋ + 1` for an odd `n`; every root is an entry with its signs and parts
/// exchanged, exactly. The values are those of [`root`].
pub(crate) struct Roots {
    n: usize,
    /// `n`, `3n`, `5n` and `7n`, the bounds of the octants of `8k` (lengths below 2^60).
    bounds: [u64; 4],
    /// `log2 gcd(4, n)`: the offsets `u` are multiples of `1 << shift`.
    shift: u32,
    tab: Vec<(f64, f64)>,
}

impl Roots {
    /// The table for `n ≥ 1`.
    pub(crate) fn new(n: usize) -> Roots {
        let shift = n.trailing_zeros().min(2);
        let len = ((n / 2) >> shift) + 1;
        let tab = (0..len).map(|u| reduced((u << shift) as i64, n)).collect();
        let m = (n as u64).min(1 << 60);
        Roots {
            n,
            bounds: [m, 3 * m, 5 * m, 7 * m],
            shift,
            tab,
        }
    }

    /// `e^(−2πi·k/n)`, as [`root`]`(k, n)`; without branches on the octant (the signs and the
    /// exchange of the parts by bit operations), as the octants of successive `k` are not
    /// predictable.
    #[inline]
    pub(crate) fn get(&self, k: usize) -> (f64, f64) {
        let k = if k < self.n { k } else { k % self.n };
        let (q, num) = if (self.n as u64) < 1 << 60 {
            let (k8, b) = (8 * k as u64, &self.bounds);
            let q = (k8 >= b[0]) as u32
                + (k8 >= b[1]) as u32
                + (k8 >= b[2]) as u32
                + (k8 >= b[3]) as u32;
            (q, (k8 / 2) as i64 - q as i64 * self.n as i64)
        } else {
            reduce(k, self.n)
        };
        let (ct, st) = self.tab[(num.unsigned_abs() >> self.shift) as usize];
        let sign = |x: f64, neg: u64| f64::from_bits(x.to_bits() ^ (neg << 63));
        // st with the sign of num; then the rotation by q quarter turns, as `rotate`:
        // q = 1 or 3 exchanges the parts, the cosine is negative for q = 1, 2, the sine for
        // q = 2, 3, and the result is (c, −s).
        let st = sign(st, (num < 0) as u64);
        let odd = q & 1 == 1;
        let (c, s) = (if odd { st } else { ct }, if odd { ct } else { st });
        let q = q as u64;
        (sign(c, ((q + 1) >> 1) & 1), sign(s, !(q >> 1) & 1))
    }

    /// `e^(−2πi·k/n)` in the float type `T`.
    #[inline]
    pub(crate) fn get_t<T: Float>(&self, k: usize) -> Complex<T> {
        let (c, s) = self.get(k);
        Complex::new(T::from_f64(c), T::from_f64(s))
    }
}

/// `e^(−2πi·k/n)` in the float type `T`.
#[inline]
pub(crate) fn root_t<T: Float>(k: usize, n: usize) -> Complex<T> {
    let (c, s) = root(k, n);
    Complex::new(T::from_f64(c), T::from_f64(s))
}

/// The table `e^(−2πi·k/n)` for `k` in `0..len`.
pub(crate) fn table<T: Float>(n: usize, len: usize) -> Vec<Complex<T>> {
    (0..len).map(|k| root_t(k, n)).collect()
}

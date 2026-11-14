//! The constants π, `e`, `ln 2`, Euler's `γ`, Catalan's `G` and `ln 10`, correctly rounded to any
//! precision.
//!
//! Each constant `C` is computed as a fixed-point number `X ≈ C·2^w`, within less than 2 units
//! (`|C·2^w - X| < 2`), and rounded straight from the bits of `X` when they decide the rounding
//! ([`decide`]), by Ziv's strategy otherwise ([`ziv`]). Up to [`TABLE_BITS`] bits after the
//! point `X` is `floor(C·2^w)` itself, cut from a table; beyond, it comes from a series summed by
//! binary splitting ([`series`]):
//!
//! - π by the Chudnovskys' series (1988), 47.11 bits a term,
//!   `π = 426880 √10005 / Σ_(k ≥ 0) (-1)^k (6k)! (13591409 + 545140134 k) / ((3k)! (k!)³ 640320^(3k))`;
//! - `e = Σ_(k ≥ 0) 1/k!`;
//! - `ln 2` by Zuniga's Ramanujan-type series (2025, Eq. 18), 11.92 bits a term,
//!   `ln 2 = (1/2160) Σ_(k ≥ 0) (1497 + 1794 k) Π_(j ≤ k) j (2j - 1) / (216 (6j + 1)(6j + 5))`;
//! - `G` by the Hessami Pilehroods' short series (2010), 7.51 bits a term,
//!   `G = (1/2) Σ_(k ≥ 1) (580k² - 184k + 15) / (9 (6k - 1)²(6k - 5)²) Π_(j < k) 32 j³ (2j - 1) / (9 (6j - 1)²(6j - 5)²)`;
//! - `γ` by the Brent–McMillan algorithm ([`euler`]);
//! - `ln 10 = 3 ln 2 + ln(5/4)`, `ln(5/4)` by Zuniga's series for `log(u/v)`.
//!
//! The last fixed-point value computed of each constant is kept, up to [`CACHE_BITS`] bits after
//! the point, in a cache shared by the threads ([`Cache`]): a shorter one is cut from it, by
//! `floor(X 2^-d)`, which stays within 2 units (`|C 2^(w-d) - floor(X 2^-d)| < 2·2^-d + 1 ≤ 2` for
//! `d ≥ 1`), so that repeated calls, and the calls of the functions' Ziv loops, cost a copy.

mod euler;
mod num;
mod series;
mod tables;

use super::limbs;
use super::ziv::{Approx, guard, ziv};
use super::{BigFloat, Kind, Round};
use crate::math::numbers::{ByteOrder, Number};
use core::cmp::Ordering;
use num::{Num, quotient, sqrt_fixed};
use series::{Poly, Series};
use std::sync::{Arc, Mutex};

/// The bits after the point of the tables of the constants: `floor(C·2^16384)`.
pub(crate) const TABLE_BITS: u64 = 16_384;

/// The most bits after the point of a value the [`Cache`] keeps: `2^27`, 16 MiB a constant, 96 MiB
/// for the six at most. A longer value is computed at each call (its computation needs several
/// times as much memory for a while).
pub(crate) const CACHE_BITS: u64 = 1 << 27;

/// The constants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Constant {
    Pi,
    E,
    Ln2,
    Euler,
    Catalan,
    Ln10,
}

impl Constant {
    /// The table, `floor(C·2^TABLE_BITS)·2^z`, and `z`: shifted so that its top bit is the top bit
    /// of its top limb.
    fn table(self) -> (&'static [u64], u64) {
        match self {
            Constant::Pi => (&PI_TABLE, 62),
            Constant::E => (&E_TABLE, 62),
            Constant::Ln2 => (crate::math::numbers::LN2_TABLE, 0),
            Constant::Euler => (&tables::EULER, 0),
            Constant::Catalan => (&tables::CATALAN, 0),
            Constant::Ln10 => (&LN10_TABLE, 62),
        }
    }

    /// `X` with `|C·2^w - X| < 2`, from the series.
    fn compute(self, w: u64) -> Vec<u64> {
        match self {
            Constant::Pi => pi_series(w),
            Constant::E => e_series(w),
            Constant::Ln2 => ln2_series(w),
            Constant::Euler => euler::gamma(w, |w| fixed(Constant::Ln2, w).0),
            Constant::Catalan => catalan_series(w),
            Constant::Ln10 => ln10_series(w),
        }
    }

    /// The constant's slot in the [`Cache`].
    fn slot(self) -> &'static Cache {
        static CACHES: [Cache; 6] = [
            Mutex::new(None),
            Mutex::new(None),
            Mutex::new(None),
            Mutex::new(None),
            Mutex::new(None),
            Mutex::new(None),
        ];
        &CACHES[self as usize]
    }
}

/// `t·2^z` in `N` limbs, for `0 < z < 64`, at compile time.
const fn shifted<const N: usize>(t: &[u64], z: u32) -> [u64; N] {
    let mut r = [0u64; N];
    let mut i = 0;
    while i < N {
        let lo = if i > 0 { t[i - 1] >> (64 - z) } else { 0 };
        let hi = if i < t.len() { t[i] << z } else { 0 };
        r[i] = hi | lo;
        i += 1;
    }
    r
}

/// `floor(π 2^16384)·2^62`: the top bit of `π`'s integer part 3 at the top.
static PI_TABLE: [u64; 257] = shifted(crate::math::numbers::PI_FULL, 62);

/// `floor(e 2^16384)·2^62`.
static E_TABLE: [u64; 257] = shifted(&tables::E, 62);

/// `floor(ln 10 · 2^16384)·2^62`.
static LN10_TABLE: [u64; 257] = shifted(&tables::LN10, 62);

/// The value of a constant kept: `X` to `w` bits after the point, `|C·2^w - X| < 2`, held as
/// `X·2^z` with its top bit at the top of its top limb, so that a rounding of it copies limbs
/// without shifting them.
struct Kept {
    w: u64,
    z: u64,
    x: Vec<u64>,
}

/// The [`Kept`] value of a constant, if any.
type Cache = Mutex<Option<Arc<Kept>>>;

/// A fixed-point value of a constant at `w` bits after the point, `|C·2^w - X| < 2^k`: the table,
/// or a value of the [`Cache`], or one computed.
struct Value {
    x: Shared,
    w: u64,
    k: i128,
}

/// The limbs of a [`Value`].
enum Shared {
    Table(&'static [u64]),
    Owned(Vec<u64>),
    Cached(Arc<Kept>),
}

impl Value {
    fn limbs(&self) -> &[u64] {
        match &self.x {
            Shared::Table(x) => x,
            Shared::Owned(x) => x,
            Shared::Cached(v) => &v.x,
        }
    }
}

/// A value of `c` to at least `w` bits after the point: the table up to [`TABLE_BITS`] (`k = 0`),
/// the cached value if it is as long, otherwise one computed (and kept when it is the longest so
/// far and not past [`CACHE_BITS`]), within 2 units (`k = 1`).
fn value(c: Constant, w: u64) -> Value {
    if w <= TABLE_BITS {
        // `|C 2^(t + z) - floor(C 2^t) 2^z| < 2^z`.
        let (x, z) = c.table();
        return Value {
            x: Shared::Table(x),
            w: TABLE_BITS + z,
            k: i128::from(z),
        };
    }
    let slot = c.slot();
    let kept = slot.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(v) = kept
        && v.w >= w
    {
        // `|C 2^(w + z) - X 2^z| < 2^(1 + z)`.
        return Value {
            w: v.w + v.z,
            k: 1 + i128::from(v.z),
            x: Shared::Cached(v),
        };
    }
    let x = c.compute(w);
    if w <= CACHE_BITS {
        let mut kept = slot.lock().unwrap_or_else(|e| e.into_inner());
        if kept.as_ref().is_none_or(|v| v.w < w) {
            let z = u64::from(x.last().map_or(0, |t| t.leading_zeros()));
            let v = Arc::new(Kept {
                w,
                z,
                x: limbs::shl(&x, z),
            });
            *kept = Some(v.clone());
            return Value {
                w: w + z,
                k: 1 + i128::from(z),
                x: Shared::Cached(v),
            };
        }
    }
    Value {
        x: Shared::Owned(x),
        w,
        k: 1,
    }
}

/// `X ≈ C·2^w` and the exponent `k` of the bound of its error, `|C·2^w - X| < 2^k`: `floor(C·2^w)`
/// with `k = 0` up to [`TABLE_BITS`] (`floor(floor(C 2^t) 2^-(t - w)) = floor(C 2^w)`), within 2
/// units beyond (`k = 1`), cut from a longer value (`floor(X' 2^-d)` of `X'` within 2 units is
/// within `2·2^-d + 1 ≤ 2` for `d ≥ 1`).
pub(crate) fn fixed(c: Constant, w: u64) -> (Vec<u64>, i128) {
    let v = value(c, w);
    let x = v.limbs();
    // Cut, a value within 2 units stays within 2 (`k = 1`), and the table, a floor shifted by `z`,
    // gives a floor (`k = 0`) once cut by at least `z`, as `w ≤ TABLE_BITS`.
    let table = matches!(v.x, Shared::Table(_));
    let k = if v.w == w {
        v.k
    } else if table {
        0
    } else {
        1
    };
    if v.w == w {
        match v.x {
            Shared::Owned(x) => (x, k),
            _ => (x.to_vec(), k),
        }
    } else {
        (limbs::shr(x, v.w - w), k)
    }
}

/// Whether the bits `lo..hi` of `x` (`hi - lo ≥ 1`, `hi ≤ bits(x)`) are not all equal.
fn mixed(x: &[u64], lo: u64, hi: u64) -> bool {
    let first = limbs::bit(x, hi - 1);
    let fill = if first { u64::MAX } else { 0 };
    let mut top = hi;
    while top > lo {
        // The bits `max(lo, 64 i)..top` of the limb `i` holding bit `top - 1`.
        let i = (top - 1) / 64;
        let base = 64 * i;
        let from = lo.max(base);
        let width = top - from;
        let mask = if width == 64 {
            u64::MAX
        } else {
            ((1u64 << width) - 1) << (from - base)
        };
        if (x[i as usize] ^ fill) & mask != 0 {
            return true;
        }
        top = from;
    }
    false
}

/// The correct rounding of a positive constant `v` known within `|v - X 2^-w| < 2^(k-w)`, when the
/// bits of `X` decide it: with `d` bits of `X` below its top `prec`, the bits `k + 1..d - 1` not
/// all equal keep the interval at least `2^(k+1)` units from the multiples of `2^d` (the numbers of
/// the precision) and from the midpoints between them (with bit `d - 1` set, `X mod 2^d - 2^(d-1)`
/// has a bit at or above `k + 1`; clear, `2^(d-1) - X mod 2^d` likewise). No number of the
/// precision and no midpoint lies between `X` and `v`, so that `v` rounds as `X` does, which bit
/// `d - 1` decides to nearest, and never exactly: away from zero when rounded up.
fn decide(x: &[u64], w: u64, k: i128, prec: u64, round: Round) -> Option<(BigFloat, Ordering)> {
    let bits = limbs::bit_len(x);
    let d = bits.checked_sub(prec)?;
    let lo = u64::try_from(k + 1).ok()?;
    if d < lo + 2 || !mixed(x, lo, d - 1) {
        return None;
    }
    let up = match round {
        Round::Nearest => limbs::bit(x, d - 1),
        Round::Up => true,
        Round::TowardZero | Round::Down => false,
    };
    Some(rounded(x, bits, w, prec, up))
}

/// The number `floor(X 2^-d)`, `d = bits - prec`, plus one when `up`, times `2^(d - w)`, with its
/// direction (`Greater` when `up`): the top `prec` bits of `X` moved into the `n = ceil(prec/64)`
/// limbs of the mantissa by one shift of `s = bits - 64n` bits, those below the precision cleared.
fn rounded(x: &[u64], bits: u64, w: u64, prec: u64, up: bool) -> (BigFloat, Ordering) {
    let n = prec.div_ceil(64) as usize;
    let s = bits as i128 - 64 * n as i128;
    let mant = if s >= 0 {
        let (k, b) = ((s / 64) as usize, (s % 64) as u32);
        let src = &x[k..];
        // A copy, then the shift in place, from the bottom: each limb reads the next one before
        // that is shifted.
        let mut mant = src[..n].to_vec();
        if b != 0 {
            let last = src.get(n).copied().unwrap_or(0);
            for i in 0..n - 1 {
                mant[i] = mant[i] >> b | mant[i + 1] << (64 - b);
            }
            mant[n - 1] = mant[n - 1] >> b | last << (64 - b);
        }
        mant
    } else {
        let shifted = limbs::shl(x, (-s) as u64);
        let mut mant = vec![0; n - shifted.len()];
        mant.extend_from_slice(&shifted);
        mant
    };
    let mut mant = mant;
    let unused = 64 * n as u64 - prec;
    mant[0] &= !((1u64 << unused) - 1);
    let mut exp = bits as i64 - w as i64;
    if up {
        let mut carry = true;
        let mut add = 1u64 << unused;
        for m in mant.iter_mut() {
            let (v, c) = m.overflowing_add(add);
            *m = v;
            carry = c;
            add = 1;
            if !carry {
                break;
            }
        }
        if carry {
            mant[n - 1] = 1 << 63;
            exp += 1;
        }
    }
    let r = BigFloat {
        prec,
        kind: Kind::Finite {
            negative: false,
            exp,
            mant,
        },
    };
    (
        r,
        if up {
            Ordering::Greater
        } else {
            Ordering::Less
        },
    )
}

/// The [`Approx`] of a constant to `w` bits after the point.
fn approx(c: Constant, w: u64) -> Approx {
    let (m, k) = fixed(c, w);
    Approx {
        negative: false,
        m,
        lsb: -i128::from(w),
        err: k - i128::from(w),
    }
}

/// π to `w` bits after the point, within `2^(1 - w)`: for the functions that need π inside their
/// working precision.
pub(crate) fn pi_approx(w: u64) -> Approx {
    approx(Constant::Pi, w)
}

/// `ln 2` to `w` bits after the point, within `2^(1 - w)`.
pub(crate) fn ln2_approx(w: u64) -> Approx {
    approx(Constant::Ln2, w)
}

/// `e` to `w` bits after the point, within `2^(1 - w)`.
pub(crate) fn e_approx(w: u64) -> Approx {
    approx(Constant::E, w)
}

/// `γ` to `w` bits after the point, within `2^(1 - w)`.
pub(crate) fn euler_approx(w: u64) -> Approx {
    approx(Constant::Euler, w)
}

/// `G` to `w` bits after the point, within `2^(1 - w)`.
pub(crate) fn catalan_approx(w: u64) -> Approx {
    approx(Constant::Catalan, w)
}

/// `ln 10` to `w` bits after the point, within `2^(1 - w)` (`2^-w` up to 16,384 bits): for `exp10`
/// and `log10`.
pub(crate) fn ln10_approx(w: u64) -> Approx {
    approx(Constant::Ln10, w)
}

/// `floor(π 2^f)` less at most 1, for the partition numbers and the Bernoulli numbers: from `X`
/// at `f + 2` bits, `|π 2^(f+2) - X| < 2`, `(X - 2)/4` lies in `(π 2^f - 1, π 2^f)`, so its floor in
/// `(π 2^f - 2, π 2^f)`.
pub(crate) fn pi_below(f: u64) -> Number {
    let (x, _) = fixed(Constant::Pi, f + 2);
    let x = limbs::sub(&x, &[2]);
    limbs::to_number(&limbs::shr(&x, 2), false, ByteOrder::LittleEndian)
}

/// π by the Chudnovskys' series, `|π 2^w - X| < 2`.
///
/// In the form of [`series`], `P(k) = (13591409 + 545140134 k) R(k)`, `R(k) = (6k - 5)(2k - 1)(6k -
/// 1)`, `Q(k) = -10939058860032000 k³`, `c_Q = 13591409`, `c_P = 1`, and `π = 426880 √10005 / S`
/// for `S = c_Q + Σ t_k = D/Q`, `D = 13591409 Q + T`. The terms' ratio is `(A + B(k+1))/(A + Bk)
/// R(k+1)/Q(k+1)`, below `2·72/10939058860032000 < 2^-46`, as `(6k-5)(2k-1)(6k-1) < 72 k³`.
///
/// With `g = w + 9`, `Y = floor(426880 Q 2^g / D)` within 1.25 of `y_N 2^g`, `y_N = 426880 Q/D`
/// ([`quotient`]), and `ρ = floor(√10005 2^g)`: `Y ρ` is within
/// `2^g (100.03 (δ 2^g + 1.25) + 0.04)` of `π 4^g`, where `δ = |y_N - π/√10005|`. As
/// `π/√10005 = 426880/S` and `S, S_N > 1.3·10^7`, `δ < 2.6·10^-9` times the tail, at most `2^-g`
/// with a tail below `2^-g`; so `Y ρ 2^-(2g - w)` is within `0.44` of `π 2^w`, and its floor
/// within `1.44`.
fn pi_series(w: u64) -> Vec<u64> {
    let g = w + 9;
    let series = Series {
        p: Poly(vec![
            -67_957_045,
            -2_100_495_856,
            23_608_573_992,
            -57_896_553_024,
            39_250_089_648,
        ]),
        q: Poly(vec![0, 0, 0, -10_939_058_860_032_000]),
        r: Poly(vec![-5, 46, -108, 72]),
        ratio: 2f64.powi(-46),
        positive: false,
    };
    let n = series.terms(g as f64);
    let (t, q) = series.sum(n, None);
    // `D = 13591409 Q + T`; `D/Q > 0`.
    let mut cut = num::Cut::new(None);
    let d = {
        let a = cut.mul(&Num::of_i128(13_591_409), &q);
        cut.add(a, t)
    };
    let qm = limbs::mul(&q.to_limbs(), &[426_880]);
    let y = quotient(&qm, i128::from(g), &d.to_limbs());
    drop(qm);
    let rho = sqrt_fixed(10_005, g);
    limbs::shr(&limbs::mul(&y, &rho), 2 * g - w)
}

/// `e = 1 + Σ_(k ≥ 1) 1/k!`, `|e 2^w - X| < 2`: `P = R = 1`, `Q(k) = k`, `c_Q = c_P = c_D = 1`; the
/// ratio of terms `1/(k+1) ≤ 1/2`. The tail below `2^-(w + 2)` and [`euler::fixed_ratio`]'s
/// `1.25 + 1/4` give `1.75`.
fn e_series(w: u64) -> Vec<u64> {
    let series = Series {
        p: Poly(vec![1]),
        q: Poly(vec![0, 1]),
        r: Poly(vec![1]),
        ratio: 0.5,
        positive: true,
    };
    let n = series.terms(w as f64 + 2.0);
    euler::fixed_ratio(&series, n, w, 1, 1, 1, 2)
}

/// `ln 2` by Zuniga's series (FLINT's form): `P(k) = (1497 + 1794 k) k (2k - 1) = 3588k³ + 1200k² -
/// 1497k`, `Q(k) = 216 (6k + 1)(6k + 5)`, `R(k) = k (2k - 1)`, `c_Q = 1497`, `c_P = 1`, `c_D = 2160`;
/// all positive for `k ≥ 1`. The ratio of terms `(A(k+1)/A(k)) R(k+1)/Q(k+1)` is at most
/// `(5085/3291)/3888 < 4·10^-4` (`A(k+1)/A(k)` falls from `A(2)/A(1)`, and
/// `k(2k-1)/(216 (6k+1)(6k+5)) < 1/3888`). The tail below `2^-(w + 2)` (times `c_P/c_D < 1`) and
/// `1.25 + 1/4` give `1.75`.
fn ln2_series(w: u64) -> Vec<u64> {
    let series = Series {
        p: Poly(vec![0, -1497, 1200, 3588]),
        q: Poly(vec![1080, 7776, 7776]),
        r: Poly(vec![0, -1, 2]),
        ratio: 4e-4,
        positive: true,
    };
    let n = series.terms(w as f64 + 2.0);
    euler::fixed_ratio(&series, n, w, 1, 2160, 1497, 1)
}

/// `G` by the Pilehroods' short series in y-cruncher's form: `P(k) = 580k² - 184k + 15`,
/// `Q(k) = 9 (6k - 1)²(6k - 5)²`, `R(k) = 32 k³ (2k - 1)`, `c_Q = 0`, `c_P = 1`, `c_D = 2`; all
/// positive for `k ≥ 1`. The ratio of terms is `(P(k+1)/P(k)) R(k)/Q(k+1)`: `P(k+1) - P(k) =
/// 1160k + 396` and `P(k) ≥ 396 k²` give `P(k+1)/P(k) ≤ 1 + 2.93/k + 1/k² ≤ 4.93`, and
/// `R(k)/Q(k+1) ≤ 64 k⁴ / (11664 k⁴)`: below `1/36`. The tail below `2^-(w + 2)` and `1.25 + 1/4`
/// give `1.75`.
fn catalan_series(w: u64) -> Vec<u64> {
    let series = Series {
        p: Poly(vec![15, -184, 580]),
        q: Poly(vec![225, -3240, 14904, -23328, 11664]),
        r: Poly(vec![0, 0, 0, -32, 64]),
        ratio: 1.0 / 36.0,
        positive: true,
    };
    let n = series.terms(w as f64 + 2.0);
    euler::fixed_ratio(&series, n, w, 1, 2, 0, 1)
}

/// `ln 10 = 3 ln 2 + ln(5/4)`, `|ln 10 · 2^w - X| < 2`: `ln 2` (from its cache) and `ln(5/4)` by
/// Zuniga's series for `log(u/v)` ([`euler::log_ratio`], 21.7 bits a term), each to `w + 3` bits
/// within 2 units, so that `3 L2 + L` is within 8 units of `2^-(w+3)`, one of `2^-w`, and its floor
/// at `w` within 2.
fn ln10_series(w: u64) -> Vec<u64> {
    let l2 = fixed(Constant::Ln2, w + 3).0;
    let l = euler::log_ratio(5, 4, w + 3);
    limbs::shr(&limbs::add(&limbs::mul(&l2, &[3]), &l), 3)
}

/// The correct rounding of a constant.
#[track_caller]
fn round(c: Constant, prec: u64, round: Round) -> (BigFloat, Ordering) {
    let prec = BigFloat::check_prec(prec);
    // The value at `prec + guard` bits, or a longer one kept, nearly always decides; otherwise
    // Ziv's loop from there.
    let v = value(c, prec + guard(prec));
    if let Some(r) = decide(v.limbs(), v.w, v.k, prec, round) {
        return r;
    }
    drop(v);
    match c {
        Constant::Pi => ziv(prec, round, pi_approx),
        Constant::E => ziv(prec, round, e_approx),
        Constant::Ln2 => ziv(prec, round, ln2_approx),
        Constant::Euler => ziv(prec, round, euler_approx),
        Constant::Catalan => ziv(prec, round, catalan_approx),
        Constant::Ln10 => ziv(prec, round, ln10_approx),
    }
}

impl BigFloat {
    /// π rounded to `prec` bits in the mode `round`, and the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (p, o) = BigFloat::pi_round(53, Round::Nearest);
    /// assert_eq!(p, BigFloat::from(core::f64::consts::PI));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn pi_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        self::round(Constant::Pi, prec, round)
    }

    /// π to nearest at `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::pi(24), BigFloat::from(core::f32::consts::PI));
    /// ```
    #[track_caller]
    pub fn pi(prec: u64) -> BigFloat {
        Self::pi_round(prec, Round::Nearest).0
    }

    /// `e`, the base of the natural logarithms, rounded to `prec` bits in the mode `round`, and
    /// the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (e, o) = BigFloat::e_round(53, Round::Down);
    /// assert_eq!(e, BigFloat::from(core::f64::consts::E));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn e_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        self::round(Constant::E, prec, round)
    }

    /// `e` to nearest at `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::e(53), BigFloat::from(core::f64::consts::E));
    /// ```
    #[track_caller]
    pub fn e(prec: u64) -> BigFloat {
        Self::e_round(prec, Round::Nearest).0
    }

    /// `ln 2` rounded to `prec` bits in the mode `round`, and the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::ln2_round(53, Round::Nearest);
    /// assert_eq!(l, BigFloat::from(core::f64::consts::LN_2));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn ln2_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        self::round(Constant::Ln2, prec, round)
    }

    /// `ln 2` to nearest at `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::ln2(24), BigFloat::from(core::f32::consts::LN_2));
    /// ```
    #[track_caller]
    pub fn ln2(prec: u64) -> BigFloat {
        Self::ln2_round(prec, Round::Nearest).0
    }

    /// Euler's constant `γ = lim (H_n - ln n) = 0.5772…` rounded to `prec` bits in the mode
    /// `round`, and the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// // γ = 0.57721566490153286…, 0x1.2788cfc6fb618f4…p-1.
    /// let (g, o) = BigFloat::euler_gamma_round(53, Round::Nearest);
    /// assert_eq!(g, BigFloat::from(0.5772156649015329));
    /// assert_eq!(o, Ordering::Greater);
    /// ```
    #[track_caller]
    pub fn euler_gamma_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        self::round(Constant::Euler, prec, round)
    }

    /// Euler's constant `γ` to nearest at `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::euler_gamma(24), BigFloat::from(0.5772157f32));
    /// ```
    #[track_caller]
    pub fn euler_gamma(prec: u64) -> BigFloat {
        Self::euler_gamma_round(prec, Round::Nearest).0
    }

    /// Catalan's constant `G = Σ_(k ≥ 0) (-1)^k / (2k + 1)² = 0.9159…` rounded to `prec` bits in
    /// the mode `round`, and the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// // G = 0.91596559417721901…
    /// let (g, o) = BigFloat::catalan_round(53, Round::Down);
    /// assert_eq!(g, BigFloat::from(0.915965594177219));
    /// assert_eq!(o, Ordering::Less);
    /// ```
    #[track_caller]
    pub fn catalan_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        self::round(Constant::Catalan, prec, round)
    }

    /// Catalan's constant `G` to nearest at `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::catalan(24), BigFloat::from(0.91596559f32));
    /// ```
    #[track_caller]
    pub fn catalan(prec: u64) -> BigFloat {
        Self::catalan_round(prec, Round::Nearest).0
    }

    /// `ln 10` rounded to `prec` bits in the mode `round`, and the direction of the rounding.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::{BigFloat, Round};
    /// use core::cmp::Ordering;
    ///
    /// let (l, o) = BigFloat::ln10_round(53, Round::Nearest);
    /// assert_eq!(l, BigFloat::from(core::f64::consts::LN_10));
    /// assert_ne!(o, Ordering::Equal);
    /// ```
    #[track_caller]
    pub fn ln10_round(prec: u64, round: Round) -> (BigFloat, Ordering) {
        self::round(Constant::Ln10, prec, round)
    }

    /// `ln 10` to nearest at `prec` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::float::BigFloat;
    ///
    /// assert_eq!(BigFloat::ln10(24), BigFloat::from(core::f32::consts::LN_10));
    /// ```
    #[track_caller]
    pub fn ln10(prec: u64) -> BigFloat {
        Self::ln10_round(prec, Round::Nearest).0
    }
}

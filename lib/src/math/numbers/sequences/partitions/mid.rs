//! The terms of Rademacher's series to a few hundred bits, in fixed point on arrays of limbs on the
//! stack: the terms that double precision does not reach, too short to pay for the allocations of
//! [`Number`]s.
//!
//! A value has `l` limbs, the least significant first: the top one its integer part, the others
//! `f = 64 (l - 1)` bits of fraction. Each operation truncates, and each value carries a bound of
//! its absolute error in units of `2^-f`. `e^x = 2^q e^r` with `r = x - q ln 2` in `[0, ln 2)`,
//! `e^r - 1` from `r 2^-s` by its Taylor series and `s` doublings; the cosines from their versines
//! the same way; each series by rectangular splitting ([`Mid::series`]).

use super::super::super::div::limbs::{div_2by1, invert_limb};
use super::super::super::mul::limbs::{add_n, mul, sub_n};
use super::super::super::root::mul_1;
use super::super::super::scratch::Scratch;
use super::super::super::{ByteOrder, Number};
use super::expsum::Sum;
use super::real::pow2;

/// The most limbs of a value.
pub(super) const MAX: usize = 23;

/// `floor(ln 2 2^1536)`, the least significant limb first, from `Σ 1 / (k 2^k)` with 128 guard
/// bits: the fraction of every value of [`MAX`] limbs.
const LN2: [u64; 24] = [
    0x4C1A_1E0B_D1D6_095D,
    0xCCCC_4E65_9393_514C,
    0xC943_E732_B479_CD33,
    0x1746_0775_DB89_90E5,
    0x7D2E_23DE_1400_B396,
    0xEE56_9D6D_FC1E_FA15,
    0x610D_30F8_8FE5_51A2,
    0x07F4_CA11_FB5B_FB90,
    0xDA2D_97C5_0F3F_D5C6,
    0x655F_A187_2F20_E3A2,
    0xF5DF_A6BD_3830_3248,
    0x72CE_87B1_9D65_48CA,
    0x256F_A0EC_7657_F74B,
    0xB9EA_9BC3_B136_603B,
    0x1ACB_DA11_317C_387E,
    0x3E96_CA16_224A_E8C5,
    0x2757_3B29_1169_B825,
    0xED2E_AE35_C138_2144,
    0x5595_52FB_4AFA_1B10,
    0xE7B8_7620_6DEB_AC98,
    0x8A0D_175B_8BAA_FA2B,
    0x40F3_4326_7298_B62D,
    0xC9E3_B398_03F2_F6AF,
    0xB172_17F7_D1CF_79AB,
];

/// A value: `l` limbs used.
pub(super) type Fx = [u64; MAX];

/// The arithmetic of values of `l` limbs.
#[derive(Clone, Copy)]
pub(super) struct Mid {
    /// The limbs of a value.
    pub(super) l: usize,
}

impl Mid {
    /// The bits of fraction.
    pub(super) fn f(self) -> u64 {
        64 * (self.l as u64 - 1)
    }

    /// The integer `c`.
    fn int(self, c: u64) -> Fx {
        let mut r = [0; MAX];
        r[self.l - 1] = c;
        r
    }

    /// The value from the fixed-point `X 2^-fx`, rounded down, `X >= 0` below `2^(64 + fx)`.
    pub(super) fn of_number(self, x: &Number, fx: u64) -> Fx {
        let y = super::real::shift(x, self.f() as i64 - fx as i64);
        let mut r = [0; MAX];
        for (i, b) in y.value.iter().enumerate().take(8 * self.l) {
            r[i / 8] |= u64::from(*b) << (8 * (i % 8));
        }
        r
    }

    /// The value from the top limbs of a table `t` of `floor(c 2^(64 (t.len() - 1)))`-like limbs
    /// whose top limb is the integer part: those below its fraction's last are dropped.
    fn of_table(self, t: &[u64], integer: u64) -> Fx {
        debug_assert!(t.len() >= self.l - 1);
        let mut r = [0; MAX];
        let take = (self.l - 1).min(t.len());
        r[self.l - 1 - take..self.l - 1].copy_from_slice(&t[t.len() - take..]);
        r[self.l - 1] = integer;
        r
    }

    /// The number of the value, with `negative`, times `2^e`, rounded down in magnitude.
    pub(super) fn to_number(self, a: &Fx, negative: bool, e: i64) -> Number {
        let bytes: Vec<u8> = a[..self.l].iter().flat_map(|x| x.to_le_bytes()).collect();
        let x = super::real::shift(
            &Number::new(bytes, ByteOrder::LittleEndian),
            e - self.f() as i64,
        );
        if negative { -x } else { x }
    }

    /// The value as a double, about.
    pub(super) fn to_f64(self, a: &Fx) -> f64 {
        let l = self.l;
        a[l - 1] as f64
            + a[l - 2] as f64 * pow2(-64)
            + if l > 2 {
                a[l - 3] as f64 * pow2(-128)
            } else {
                0.0
            }
    }

    /// `a + b`.
    fn add(self, a: &Fx, b: &Fx) -> Fx {
        let mut r = [0; MAX];
        // SAFETY: the arrays hold `l` limbs.
        unsafe { add_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), self.l) };
        r
    }

    /// `a - b`, `a >= b`.
    fn sub(self, a: &Fx, b: &Fx) -> Fx {
        let mut r = [0; MAX];
        // SAFETY: the arrays hold `l` limbs.
        unsafe { sub_n(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), self.l) };
        r
    }

    /// `a b`, rounded down: a unit.
    fn mul(self, a: &Fx, b: &Fx) -> Fx {
        let l = self.l;
        let mut p = [0u64; 2 * MAX];
        // SAFETY: the product of `l` limbs by `l` takes `2l`, without scratch below Karatsuba's
        // method's 24 limbs, which `MAX` stays below.
        unsafe {
            mul(
                p.as_mut_ptr(),
                a.as_ptr(),
                l,
                b.as_ptr(),
                l,
                Scratch::none(),
            )
        };
        let mut r = [0; MAX];
        r[..l].copy_from_slice(&p[l - 1..2 * l - 1]);
        r
    }

    /// `a c`.
    fn mul_u64(self, a: &Fx, c: u64) -> Fx {
        let mut r = [0; MAX];
        // SAFETY: the arrays hold `l` limbs; the product's integer part fits its limb.
        let carry = unsafe { mul_1(r.as_mut_ptr(), a.as_ptr(), self.l, c) };
        debug_assert!(carry == 0);
        r
    }

    /// `a / d`, rounded down: a unit.
    fn div_u64(self, a: &Fx, d: u64) -> Fx {
        let s = d.leading_zeros();
        let dn = d << s;
        let v = invert_limb(dn);
        let mut r = [0; MAX];
        // The dividend shifted left by `s`, from the top.
        let mut rem = if s == 0 { 0 } else { a[self.l - 1] >> (64 - s) };
        for i in (0..self.l).rev() {
            let lo = if i == 0 || s == 0 {
                a[i] << s
            } else {
                (a[i] << s) | (a[i - 1] >> (64 - s))
            };
            let (q, rr) = div_2by1(rem, lo, dn, v);
            r[i] = q;
            rem = rr;
        }
        r
    }

    /// `a 2^-k`, rounded down, `k < 64`.
    fn shr(self, a: &Fx, k: u32) -> Fx {
        let mut r = [0; MAX];
        for i in 0..self.l {
            let hi = if i + 1 < self.l { a[i + 1] } else { 0 };
            r[i] = if k == 0 {
                a[i]
            } else {
                (a[i] >> k) | (hi << (64 - k))
            };
        }
        r
    }

    /// Whether `a >= b`.
    fn ge(self, a: &Fx, b: &Fx) -> bool {
        for i in (0..self.l).rev() {
            if a[i] != b[i] {
                return a[i] > b[i];
            }
        }
        true
    }

    /// `a 2^-k`, rounded down, for any `k`: a unit for each 63 bits.
    fn shr_long(self, a: &Fx, mut k: u64) -> Fx {
        let mut r = *a;
        while k > 0 {
            let step = k.min(63) as u32;
            r = self.shr(&r, step);
            k -= u64::from(step);
        }
        r
    }

    /// `Σ_(j<t) c_j z^j` for `0 <= z < 2^-3` within `ez` units, `c_0 = 1`, `c_j = c_(j-1) s / d(j)`
    /// with `s = -1` when `alternate`, `d(j) >= 2` increasing, and a bound of its absolute error
    /// (the tail past `t` terms not counted): rectangular splitting as on numbers
    /// ([`super::real::series`]), the blocks cut so that the product of their `d`s fits a limb,
    /// `D G_b = T_(m-1) d_m + s^m z^m G'` with `T_i = T_(i-1) d_i + s^i z^i`, one division each.
    ///
    /// The powers are within `i (ez + 1)` units for `z^i`; a block's sum, divided by `D`, within
    /// `Σ e_i / (d_1 ... d_i) + (2 e_m + e(G') + 1) / D + 1 < 2 m (ez + 1) + 2 + e(G') / 2^m`: the
    /// whole within twice a block's bound.
    fn series(self, z: &Fx, ez: f64, t: u64, d: impl Fn(u64) -> u64, alternate: bool) -> (Fx, f64) {
        let one = self.int(1);
        // The block's length: about `√t`, and at most what keeps the product of its `d`s below
        // `2^62`, as the integer limb holds the partial sums.
        let mut m = ((t as f64).sqrt().ceil() as u64).clamp(1, 15);
        while m > 1 {
            // The last block's `d`s, the largest, up to `m` past `t` rounded up to a block.
            let end = t.div_ceil(m) * m;
            let mut p = 1u128;
            for i in end - m + 1..=end {
                p *= u128::from(d(i));
            }
            // `T_i` reaches about `D (1 + z + ...) < 1.2 D`: room in the integer limb.
            if p <= 1 << 62 {
                break;
            }
            m -= 1;
        }
        let m = m as usize;
        let mut powers = [[0u64; MAX]; 16];
        powers[0] = one;
        powers[1] = *z;
        for i in 2..=m {
            powers[i] = if i % 2 == 0 {
                self.mul(&powers[i / 2], &powers[i / 2])
            } else {
                self.mul(&powers[i - 1], z)
            };
        }
        let blocks = t.div_ceil(m as u64);
        let mut g = [0u64; MAX];
        for b in (0..blocks).rev() {
            let base = b * m as u64;
            let mut acc = one;
            let mut den = 1u64;
            for (i, power) in powers.iter().enumerate().take(m + 1).skip(1) {
                let di = d(base + i as u64);
                acc = self.mul_u64(&acc, di);
                den *= di;
                if i < m {
                    acc = if alternate && i % 2 == 1 {
                        self.sub(&acc, power)
                    } else {
                        self.add(&acc, power)
                    };
                }
            }
            let top = self.mul(&powers[m], &g);
            acc = if alternate && m % 2 == 1 {
                self.sub(&acc, &top)
            } else {
                self.add(&acc, &top)
            };
            g = self.div_u64(&acc, den);
        }
        let err = 4.0 * m as f64 * (ez + 1.0) + 6.0;
        (g, err)
    }

    /// The bits by which `r < 1` is reduced before the series at `f` bits: about `1.2 f^(1/3)`,
    /// which balances the series' products, `2 √(f / s)`, against the `s` doublings.
    fn exp_reduction(self) -> u64 {
        ((self.f() as f64).cbrt() * 1.2).clamp(4.0, 40.0) as u64
    }

    /// `e^x` as `(y, q)` with `e^x = y 2^q`, `1 <= y < 2`, for `x >= 0` below `2^60` within `ex`
    /// units, and a bound of the absolute error of `y`.
    ///
    /// `r = x - q ln 2` within `ex + q + 1` units, `r' = r 2^-s < 2^-(s + 0.5)`; `e^r' - 1 =
    /// r' Σ r'^j / (j + 1)!` by [`Mid::series`], whose sum is below 2 and within its bound; the `s`
    /// doublings `w (w + 2)` multiply an error by `2w + 2 <= 4` and add a unit.
    pub(super) fn exp(self, x: &Fx, ex: f64) -> (Fx, i64, f64) {
        let f = self.f();
        let ln2 = self.of_table(&LN2, 0);
        let mut q = (self.to_f64(x) * core::f64::consts::LOG2_E)
            .floor()
            .max(0.0) as u64;
        let mut ql = self.mul_u64(&ln2, q);
        while !self.ge(x, &ql) {
            q -= 1;
            ql = self.sub(&ql, &ln2);
        }
        let mut r = self.sub(x, &ql);
        while self.ge(&r, &ln2) {
            q += 1;
            r = self.sub(&r, &ln2);
        }
        let s = self.exp_reduction();
        let er = ex + q as f64 + 2.0;
        let r = self.shr_long(&r, s);
        let er = er * pow2(-(s as i64)) + s.div_ceil(63) as f64;
        // Terms: `t (s + 0.5) + log2((t + 1)!) >= f + 2`.
        let mut t = 1u64;
        let mut bits = s as f64 + 1.5;
        while bits < f as f64 + 2.0 {
            t += 1;
            bits += s as f64 + 0.5 + ((t + 1) as f64).log2();
        }
        let one = self.int(1);
        let (acc, ea) = self.series(&r, er, t, |j| j + 1, false);
        let mut w = self.mul(&r, &acc);
        // `acc < 2`; the tail past `t` terms adds a unit.
        let mut ew = er * 2.0 + ea * pow2(-(s as i64)) + 2.0;
        let two = self.int(2);
        // The product of two errors in units is in units of `2^-2f`: `unit` of a unit.
        let unit = pow2(-(f as i64));
        for _ in 0..s {
            let wf = self.to_f64(&w);
            w = self.mul(&w, &self.add(&w, &two));
            ew = ew * (2.0 * wf + 2.0) * 1.000_001 + ew * ew * unit + 1.0;
        }
        let y = self.add(&one, &w);
        (y, q as i64, ew)
    }

    /// `1 / y` for `1 <= y < 2` within `ey` units, and a bound of its error in units: Newton's
    /// iteration `r (2 - y r)` from a double's start, then the residual `t = 1 - y r` within three
    /// units bounds `|r - 1/y| <= (|t| + 3 + ey r) / y`, with `y >= 1`.
    fn reciprocal(self, y: &Fx, ey: f64) -> (Fx, f64) {
        let one = self.int(1);
        let two = self.int(2);
        let start = 1.0 / self.to_f64(y);
        let mut r = [0u64; MAX];
        r[self.l - 2] = (start * pow2(64)) as u64;
        let mut good = 40.0;
        while good < self.f() as f64 + 8.0 {
            let yr = self.mul(y, &r);
            r = self.mul(&r, &self.sub(&two, &yr));
            good *= 2.0;
        }
        let yr = self.mul(y, &r);
        // `t = 1 - y r`, of either sign, as a double in units.
        let t = if self.ge(&one, &yr) {
            self.to_f64(&self.sub(&one, &yr))
        } else {
            self.to_f64(&self.sub(&yr, &one))
        } * pow2(self.f() as i64);
        (r, t.abs() + 3.0 + ey + 1.0)
    }

    /// `cos(π a / b)` for `0 <= a < 2b`: its magnitude, its sign, and a bound of its absolute
    /// error; `None` for zero.
    ///
    /// The angle is brought to `y = π t`, `0 <= t <= 1/2`, as in double precision, then
    /// `cos(y) = 1 - v` with `v = 1 - cos(y)` from `y 2^-s` by its Taylor series,
    /// `1 - cos(z) = (z^2 / 2) Σ (-z^2)^j 2 / (2j + 2)!`, by [`Mid::series`], and `s` doublings
    /// `1 - cos(2z) = 2v (2 - v)`, which multiply the absolute error by at most 4 and add two units.
    /// `y` is within `a / b + 2` units, the steps of the series within two units more than the
    /// last one's error divided by their divisor.
    pub(super) fn cos_pi(self, a: u64, b: u64) -> Option<(Fx, bool, f64)> {
        let (a, b) = (u128::from(a), u128::from(b));
        let a = if a > b { 2 * b - a } else { a };
        let (a, negative) = if 2 * a > b { (b - a, true) } else { (a, false) };
        if 2 * a == b {
            return None;
        }
        if a == 0 {
            return Some((self.int(1), negative, 0.0));
        }
        let (num, den) = (a as u64, b as u64);
        let f = self.f();
        let s = ((f as f64).cbrt() * 1.0).clamp(3.0, 24.0) as u64;
        // `π` with its integer part 3, then `y 2^-s`.
        let pi = self.of_table(super::real::PI_TABLE, 3);
        let y = self.div_u64(&self.mul_u64(&pi, num), den);
        let y = self.shr_long(&y, s);
        let ey = (num as f64 / den as f64 + 2.0) * pow2(-(s as i64)) + 1.0;
        let z = self.mul(&y, &y);
        let ez = 2.0 * ey + 1.0;
        let zf = self.to_f64(&z);
        // Terms: `2 z^j / (2j + 2)!` below `2^-(f + 2)`.
        let mut j = 1u64;
        let mut lg = zf.log2() - 1.0;
        while lg > -(f as f64 + 2.0) {
            j += 1;
            lg += zf.log2() - ((2 * j + 1) as f64).log2() - ((2 * j + 2) as f64).log2();
        }
        let one = self.int(1);
        let (acc, ea) = self.series(&z, ez, j, |i| (2 * i + 1) * (2 * i + 2), true);
        // `v = z acc / 2`, then the doublings.
        let mut v = self.shr(&self.mul(&z, &acc), 1);
        let mut ev = (ez + zf * ea + 1.0) / 2.0 + 2.0;
        let two = self.int(2);
        for _ in 0..s {
            let p = self.mul(&v, &self.sub(&two, &v));
            v = self.add(&p, &p);
            ev = 4.0 * ev + 2.0;
        }
        Some((self.sub(&one, &v), negative, ev))
    }

    /// The term `T_k` to about `f - 64` bits of `f`: `(magnitude, negative, q, error)` for
    /// `T_k = magnitude 2^q`, the error in units of `2^(q - f)`, or `None` for zero.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn term(
        self,
        sum: &Sum,
        k: u64,
        n24: u64,
        c: &Fx,
        ec: f64,
        inv_c: &Fx,
        e_inv: f64,
        root: (&Fx, f64),
    ) -> Option<(Fx, bool, i64, f64)> {
        // The product of two errors in units is in units of `2^-2f`: `unit` of a unit.
        let unit = pow2(-(self.f() as i64));
        // `x = C / k`, `1 / x = k / C`.
        let x = self.div_u64(c, k);
        let ex = ec / k as f64 + 1.0;
        let inv_x = self.mul_u64(inv_c, k);
        let einv = e_inv * k as f64;
        let (y, q, ey) = self.exp(&x, ex);
        let xf = self.to_f64(&x);
        let one = self.int(1);
        let xm1 = self.sub(&x, &one);
        // `U 2^-q = y (x - 1) / (2x)`; `e^-x (x + 1) / (2x)` below `3 e^-2x` of it, added below.
        let mut a = self.mul(&y, &xm1);
        let mut ea = ey * xf + 2.0 * ex + ey * ex * unit + 1.0;
        // `e^-x (x + 1) = 2^-2q (x + 1) / y`: below a unit from `2q >= f`, and otherwise added,
        // `1 / y` within `er` relatively.
        if 2 * q < self.f() as i64 {
            let (r, er) = self.reciprocal(&y, ey);
            let xp1 = self.add(&x, &one);
            let mut b = self.mul(&r, &xp1);
            let mut left = 2 * q as u64;
            while left > 0 {
                let k = left.min(63) as u32;
                b = self.shr(&b, k);
                left -= u64::from(k);
            }
            a = self.add(&a, &b);
            // The product's unit, and one for each of the shifts.
            ea += er * (xf + 1.0) * pow2(-2 * q) + ex + 2.0 + (2 * q as u64).div_ceil(63) as f64;
        } else {
            ea += 1.0;
        }
        let af = self.to_f64(&a);
        let u = self.shr(&self.mul(&a, &inv_x), 1);
        let uf = self.to_f64(&u);
        let mut e = (ea / xf + af * einv + 1.0) / 2.0 + 1.0;
        let mut t = *root.0;
        let mut tf = self.to_f64(&t);
        let mut et = root.1;
        // Times the prefactor and 4, over `24n - 1`.
        let pre = sum.prefactor.unsigned_abs() * 4;
        t = self.div_u64(&self.mul_u64(&t, pre), n24);
        et = et * pre as f64 / n24 as f64 + 1.0;
        tf = tf * pre as f64 / n24 as f64;
        let mut negative = sum.prefactor < 0;
        for &(a, b) in &sum.cos[..sum.len] {
            let (c, neg, ecos) = self.cos_pi(a, b)?;
            let cf = self.to_f64(&c);
            et = tf * ecos + cf * et + et * ecos * unit + 1.0;
            t = self.mul(&t, &c);
            tf *= cf;
            negative ^= neg;
        }
        let v = self.mul(&t, &u);
        e = tf * e + uf * et + e * et * unit + 1.0;
        Some((v, negative, q, e * 1.000_01))
    }
}

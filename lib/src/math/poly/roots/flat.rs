//! A polynomial over the integers as one flat vector of limbs: every coefficient in two's
//! complement in the same number `w` of limbs, the least significant first, the constant
//! coefficient first. The real roots' algorithms transform such polynomials many times (Taylor
//! shifts, scalings by powers of two, reversals), and in this form each step is a run of
//! additions or shifts of fixed width, with no allocation and no signs to look after.

use crate::math::numbers::{ByteOrder, Number, from_limbs, limbs_of};
use crate::math::poly::Polynomial;

/// A polynomial of `len` coefficients of `w` limbs each, in two's complement.
#[derive(Clone, Debug)]
pub(crate) struct Flat {
    /// Limbs per coefficient.
    pub(crate) w: usize,
    /// The number of coefficients.
    pub(crate) len: usize,
    /// The limbs.
    pub(crate) d: Vec<u64>,
}

/// `r += a` on `w` limbs, two's complement, wrapping.
#[inline(always)]
fn add_to(r: &mut [u64], a: &[u64]) {
    let mut carry = false;
    for (x, &y) in r.iter_mut().zip(a) {
        let (s, c1) = x.overflowing_add(y);
        let (s, c2) = s.overflowing_add(carry as u64);
        *x = s;
        carry = c1 | c2;
    }
}

/// The negation of `x`, in place, on its limbs.
#[inline]
fn negate(x: &mut [u64]) {
    let mut carry = true;
    for l in x.iter_mut() {
        let (s, c) = (!*l).overflowing_add(carry as u64);
        *l = s;
        carry = c;
    }
}

impl Flat {
    /// The polynomial of the coefficients `c`, with room for `extra` more bits than the largest
    /// of them has.
    pub(crate) fn from_numbers(c: &[Number], extra: u64) -> Flat {
        let limbs: Vec<Vec<u64>> = c.iter().map(limbs_of).collect();
        let bits = limbs
            .iter()
            .map(|l| {
                l.last()
                    .map_or(0, |t| l.len() as u64 * 64 - t.leading_zeros() as u64)
            })
            .max()
            .unwrap_or(0);
        let w = (bits + extra + 1).div_ceil(64).max(1) as usize;
        let mut d = vec![0u64; w * c.len()];
        for (i, (l, n)) in limbs.iter().zip(c).enumerate() {
            let x = &mut d[i * w..(i + 1) * w];
            x[..l.len()].copy_from_slice(l);
            if n.is_negative() {
                negate(x);
            }
        }
        Flat { w, len: c.len(), d }
    }

    /// The limbs of the coefficient `i`.
    #[inline(always)]
    pub(crate) fn coeff(&self, i: usize) -> &[u64] {
        &self.d[i * self.w..(i + 1) * self.w]
    }

    /// The sign of the coefficient `i`: −1, 0 or 1.
    #[inline]
    pub(crate) fn sign(&self, i: usize) -> i32 {
        let x = self.coeff(i);
        if (x[self.w - 1] as i64) < 0 {
            -1
        } else if x.iter().all(|&l| l == 0) {
            0
        } else {
            1
        }
    }

    /// The number of bits of the largest magnitude of a coefficient (an upper bound, at most one
    /// more, for a negative power of two).
    pub(crate) fn bits(&self) -> u64 {
        let mut best = 0;
        for i in 0..self.len {
            let x = self.coeff(i);
            let neg = (x[self.w - 1] as i64) < 0;
            for k in (0..self.w).rev() {
                let l = if neg { !x[k] } else { x[k] };
                if l != 0 {
                    let b = k as u64 * 64 + 64 - l.leading_zeros() as u64 + neg as u64;
                    best = best.max(b);
                    break;
                }
            }
        }
        best
    }

    /// Changes the width to `w` limbs, which holds every coefficient (sign-extended or cut).
    pub(crate) fn set_width(&mut self, w: usize) {
        if w == self.w {
            return;
        }
        let mut d = vec![0u64; w * self.len];
        let m = w.min(self.w);
        for i in 0..self.len {
            let x = &self.d[i * self.w..(i + 1) * self.w];
            let y = &mut d[i * w..(i + 1) * w];
            y[..m].copy_from_slice(&x[..m]);
            if w > self.w && (x[self.w - 1] as i64) < 0 {
                y[m..].fill(u64::MAX);
            }
        }
        self.w = w;
        self.d = d;
    }

    /// Makes room for `extra` bits more than the coefficients have, and drops limbs beyond that.
    pub(crate) fn room(&mut self, extra: u64) {
        let w = (self.bits() + extra + 1).div_ceil(64).max(1) as usize;
        self.set_width(w);
    }

    /// The number of sign changes in the sequence of the coefficients, zeros skipped.
    pub(crate) fn variations(&self) -> u32 {
        let mut v = 0;
        let mut last = 0;
        for i in 0..self.len {
            let s = self.sign(i);
            if s != 0 {
                if last != 0 && s != last {
                    v += 1;
                }
                last = s;
            }
        }
        v
    }

    /// Reverses the order of the coefficients: `xⁿ·p(1/x)` for `n = len − 1`.
    pub(crate) fn reverse(&mut self) {
        let w = self.w;
        let n = self.len;
        for i in 0..n / 2 {
            let (a, b) = self.d.split_at_mut((n - 1 - i) * w);
            a[i * w..(i + 1) * w].swap_with_slice(&mut b[..w]);
        }
    }

    /// Drops the constant coefficient, which is zero: `p(x)/x`.
    pub(crate) fn drop_constant(&mut self) {
        self.d.drain(..self.w);
        self.len -= 1;
    }

    /// Whether the sum of the coefficients, `p(1)`, is zero.
    pub(crate) fn sum_is_zero(&self) -> bool {
        // On w + 1 limbs, which hold any sum of up to 2^64 coefficients.
        let w = self.w;
        let mut acc = vec![0u64; w + 1];
        let mut ext = vec![0u64; w + 1];
        for i in 0..self.len {
            let x = self.coeff(i);
            ext[..w].copy_from_slice(x);
            ext[w] = if (x[w - 1] as i64) < 0 { u64::MAX } else { 0 };
            add_to(&mut acc, &ext);
        }
        acc.iter().all(|&l| l == 0)
    }

    /// Divides every coefficient by the largest power of two that divides them all.
    pub(crate) fn remove_pow2(&mut self) {
        let mut t = u64::MAX;
        for i in 0..self.len {
            let x = self.coeff(i);
            if let Some(k) = x.iter().position(|&l| l != 0) {
                t = t.min(k as u64 * 64 + x[k].trailing_zeros() as u64);
                if t == 0 {
                    return;
                }
            }
        }
        if t == u64::MAX {
            return;
        }
        for i in 0..self.len {
            let w = self.w;
            sar(&mut self.d[i * w..(i + 1) * w], t);
        }
    }

    /// `2ⁿ·p(x/2)`, `n = len − 1`: the coefficient `i` times `2^(n−i)`. The width must hold `n`
    /// more bits.
    pub(crate) fn scale_half(&mut self) {
        let n = self.len - 1;
        let w = self.w;
        for i in 0..n {
            shl(&mut self.d[i * w..(i + 1) * w], (n - i) as u64);
        }
    }

    /// `p(2^s·x)`: the coefficient `i` times `2^(s·i)`. The width must hold `s·n` more bits.
    pub(crate) fn scale_pow2(&mut self, s: u64) {
        let w = self.w;
        for i in 1..self.len {
            shl(&mut self.d[i * w..(i + 1) * w], s * i as u64);
        }
    }

    /// `2^(s·n)·p(x/2^s)`: the coefficient `i` times `2^(s·(n−i))`. The width must hold `s·n`
    /// more bits.
    pub(crate) fn scale_inv_pow2(&mut self, s: u64) {
        let n = self.len - 1;
        let w = self.w;
        for i in 0..n {
            shl(&mut self.d[i * w..(i + 1) * w], s * (n - i) as u64);
        }
    }

    /// The Taylor shift `p(x + 1)`, in place: by the `n(n+1)/2` additions of Horner's scheme, or
    /// from `TAYLOR_DC` coefficients of `TAYLOR_DC_W` limbs by divide and conquer on the fast
    /// products. The width must hold `n + 1` more bits than the coefficients have.
    pub(crate) fn taylor1(&mut self) {
        if self.len >= DC_FROM && self.len >= taylor_dc_from(self.bits()) {
            self.taylor1_dc();
            return;
        }
        taylor_passes(self, 0, None);
    }

    /// The coefficients as numbers.
    pub(crate) fn to_numbers(&self) -> Vec<Number> {
        let mut t = vec![0u64; self.w];
        (0..self.len)
            .map(|i| {
                let x = self.coeff(i);
                if (x[self.w - 1] as i64) < 0 {
                    t.copy_from_slice(x);
                    negate(&mut t);
                    -from_limbs(&t, ByteOrder::LittleEndian)
                } else {
                    from_limbs(x, ByteOrder::LittleEndian)
                }
            })
            .collect()
    }

    /// The coefficient `i` set to `x`, which the width holds.
    fn set(&mut self, i: usize, x: &Number) {
        let w = self.w;
        let r = &mut self.d[i * w..(i + 1) * w];
        r.fill(0);
        let l = limbs_of(x);
        let k = l.iter().rposition(|&y| y != 0).map_or(0, |k| k + 1);
        debug_assert!(k <= w);
        r[..k].copy_from_slice(&l[..k]);
        if x.is_negative() {
            negate(r);
        }
    }

    /// The Taylor shift by 1 by divide and conquer: `p = lo + x^m·hi` with `m` the largest power
    /// of two below the length, `p(x + 1) = lo(x + 1) + (x + 1)^m·hi(x + 1)`, the powers
    /// `(x + 1)^(2^j)` by squarings, the products those of `Polynomial<Number>` (Kronecker
    /// substitution into the integers' products), and Horner's scheme below `TAYLOR_LEAF`.
    fn taylor1_dc(&mut self) {
        let c = self.to_numbers();
        let mut pows = vec![Polynomial::new(vec![Number::from(1u8), Number::from(1u8)])];
        while (2usize << (pows.len() - 1)) < c.len() {
            let t = pows.last().expect("one at least");
            let t = t * t;
            pows.push(t);
        }
        let r = shift_dc(&c, &pows);
        for (i, x) in r.iter().enumerate() {
            self.set(i, x);
        }
    }

    /// The number of sign variations of `p(x + 1)`, as far as 2: the coefficients of the shift
    /// are final from the constant up, one per pass, and the shift stops at the second variation.
    /// `self` is left part-shifted. The width must hold `n + 1` more bits.
    pub(crate) fn taylor1_variations(&mut self) -> u32 {
        let mut v = 0;
        let mut last = 0;
        taylor_passes(
            self,
            0,
            Some(&mut |f: &Flat, i: usize| {
                let s = f.sign(i);
                if s != 0 {
                    if last != 0 && s != last {
                        v += 1;
                    }
                    last = s;
                }
                v >= 2
            }),
        );
        v
    }

    /// The Taylor shift `p(x + 2^s)`, in place: Horner's scheme with each addition of a
    /// coefficient shifted by `s`. The width must hold `(s + 1)·n + 1` more bits.
    pub(crate) fn taylor_pow2(&mut self, s: u64) {
        if s == 0 {
            self.taylor1();
            return;
        }
        let n = self.len - 1;
        let w = self.w;
        let mut t = vec![0u64; w];
        for i in 0..n {
            for j in (i..n).rev() {
                t.copy_from_slice(&self.d[(j + 1) * w..(j + 2) * w]);
                shl(&mut t, s);
                add_to(&mut self.d[j * w..(j + 1) * w], &t);
            }
        }
    }
}

/// The least length of a Taylor shift by divide and conquer, and the length below which its
/// pieces go by Horner's scheme; by measurement (`taylor_dc_from`).
const TAYLOR_DC: usize = 256;
const TAYLOR_LEAF: usize = if cfg!(miri) { 4 } else { 64 };

/// Under Miri the divide and conquer runs from a few coefficients, so that its tests reach it.
const MIRI_DC: usize = 12;

/// The least length of a divide and conquer, under Miri or not.
const DC_FROM: usize = if cfg!(miri) { MIRI_DC } else { TAYLOR_DC };

/// The least length from which the Taylor shift by divide and conquer is faster than Horner's
/// scheme, for coefficients of `bits` bits (before the shift), measured: at most 96 bits from
/// 256 coefficients (0.76 there, 0.49 at 1024), to 1200 bits from 512 (0.94 at 1000 bits),
/// beyond from about `512 + bits/10` (Horner's additions stay in the cache longer than the
/// products' pieces do).
fn taylor_dc_from(bits: u64) -> usize {
    if cfg!(miri) {
        MIRI_DC
    } else if bits <= 96 {
        TAYLOR_DC
    } else if bits <= 1200 {
        512
    } else {
        512 + (bits / 10) as usize
    }
}

/// The coefficients of `c(x + 1)`, as many as `c` has, by divide and conquer with the powers
/// `pows[j] = (x + 1)^(2^j)`.
fn shift_dc(c: &[Number], pows: &[Polynomial<Number>]) -> Vec<Number> {
    let len = c.len();
    if len <= TAYLOR_LEAF.max(2) {
        let mut f = Flat::from_numbers(c, len as u64 + 1);
        taylor_passes(&mut f, 0, None);
        return f.to_numbers();
    }
    let j = (len - 1).ilog2() as usize;
    let m = 1usize << j;
    let lo = shift_dc(&c[..m], pows);
    let hi = shift_dc(&c[m..], pows);
    let mut r = (&Polynomial::new(hi) * &pows[j]).into_coeffs();
    r.resize(len, Number::from(0u8));
    for (x, y) in r.iter_mut().zip(&lo) {
        *x += y;
    }
    r
}

/// The passes of the Taylor shift by 1 from `from`; after pass `i` the coefficient `i` is final,
/// and `stop(f, i)` may end the shift there.
#[allow(clippy::type_complexity)]
fn taylor_passes(
    f: &mut Flat,
    from: usize,
    mut stop: Option<&mut dyn FnMut(&Flat, usize) -> bool>,
) {
    let n = f.len - 1;
    let w = f.w;
    for i in from..n {
        pass(&mut f.d, w, i, n);
        if let Some(s) = stop.as_mut()
            && s(f, i)
        {
            return;
        }
    }
    if let Some(s) = stop.as_mut() {
        s(f, n);
    }
}

/// One pass of Horner's Taylor shift: `a[j] += a[j+1]` for `j` from `n − 1` down to `i`.
#[inline]
fn pass(d: &mut [u64], w: usize, i: usize, n: usize) {
    match w {
        1 => {
            let a = &mut d[i..=n];
            let m = a.len() - 1;
            let mut acc = a[m];
            for x in a[..m].iter_mut().rev() {
                acc = x.wrapping_add(acc);
                *x = acc;
            }
        }
        2 => {
            let a = &mut d[2 * i..2 * (n + 1)];
            let m = a.len() / 2;
            let mut acc = u128::from(a[2 * m - 2]) | (u128::from(a[2 * m - 1]) << 64);
            for k in (0..m - 1).rev() {
                let x = u128::from(a[2 * k]) | (u128::from(a[2 * k + 1]) << 64);
                acc = x.wrapping_add(acc);
                a[2 * k] = acc as u64;
                a[2 * k + 1] = (acc >> 64) as u64;
            }
        }
        _ => pass_wide(d, w, i, n),
    }
}

/// A pass on coefficients of `w ≥ 3` limbs, portable: each coefficient plus the one above it,
/// with its carry.
#[cfg_attr(all(target_arch = "aarch64", not(miri)), allow(dead_code))]
fn pass_wide_portable(d: &mut [u64], w: usize, i: usize, n: usize) {
    for j in (i..n).rev() {
        let (lo, hi) = d.split_at_mut((j + 1) * w);
        let r = &mut lo[j * w..];
        let mut carry = 0u64;
        for (x, &y) in r.iter_mut().zip(&hi[..w]) {
            let t = u128::from(*x) + u128::from(y) + u128::from(carry);
            *x = t as u64;
            carry = (t >> 64) as u64;
        }
    }
}

/// A pass on coefficients of `w ≥ 3` limbs.
#[cfg(any(not(target_arch = "aarch64"), miri))]
#[inline]
fn pass_wide(d: &mut [u64], w: usize, i: usize, n: usize) {
    pass_wide_portable(d, w, i, n);
}

/// A pass on coefficients of `w ≥ 3` limbs: one chain of `adcs` for each coefficient, two limbs
/// a step, from the coefficient `n − 1` down to `i`.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline]
fn pass_wide(d: &mut [u64], w: usize, i: usize, n: usize) {
    use core::arch::asm;
    if i >= n {
        return;
    }
    assert!(w >= 3 && d.len() >= (n + 1) * w);
    let count = n - i;
    // SAFETY: the coefficients i..=n lie in `d` (asserted). Each outer step adds the w limbs at
    // `dst + 8w` to the w limbs at `dst`, reading both before writing each pair, for dst at
    // coefficient n − 1 down to i, so every access is inside coefficients i..=n; the flags are
    // clobbered (no `preserves_flags`), `sub`, `tbz` and `cbnz` leave the carry alone, and no
    // stack is used.
    unsafe {
        let dst = d.as_mut_ptr().add((n - 1) * w);
        asm!(
            "2:",
            "mov {r}, {dst}",
            "add {a}, {dst}, {w8}",
            "mov {k}, {w}",
            "cmn xzr, xzr",
            "tbz {k}, #0, 3f",
            "ldr {x0}, [{r}]",
            "ldr {y0}, [{a}], #8",
            "adcs {x0}, {x0}, {y0}",
            "str {x0}, [{r}], #8",
            "sub {k}, {k}, #1",
            "3:",
            "ldp {x0}, {x1}, [{r}]",
            "ldp {y0}, {y1}, [{a}], #16",
            "adcs {x0}, {x0}, {y0}",
            "adcs {x1}, {x1}, {y1}",
            "stp {x0}, {x1}, [{r}], #16",
            "sub {k}, {k}, #2",
            "cbnz {k}, 3b",
            "sub {dst}, {dst}, {w8}",
            "sub {cnt}, {cnt}, #1",
            "cbnz {cnt}, 2b",
            dst = inout(reg) dst => _,
            cnt = inout(reg) count => _,
            w = in(reg) w,
            w8 = in(reg) w * 8,
            r = out(reg) _,
            a = out(reg) _,
            k = out(reg) _,
            x0 = out(reg) _,
            x1 = out(reg) _,
            y0 = out(reg) _,
            y1 = out(reg) _,
            options(nostack),
        );
    }
}

/// `x <<= s` on the limbs of `x`, two's complement, wrapping.
fn shl(x: &mut [u64], s: u64) {
    if s == 0 {
        return;
    }
    let w = x.len();
    let q = (s / 64) as usize;
    let r = (s % 64) as u32;
    if q >= w {
        x.fill(0);
        return;
    }
    if r == 0 {
        x.copy_within(..w - q, q);
    } else {
        for k in (q + 1..w).rev() {
            x[k] = (x[k - q] << r) | (x[k - q - 1] >> (64 - r));
        }
        x[q] = x[0] << r;
    }
    x[..q].fill(0);
}

/// `x >>= s` on the limbs of `x`, arithmetic (sign-filling).
fn sar(x: &mut [u64], s: u64) {
    if s == 0 {
        return;
    }
    let w = x.len();
    let fill = if (x[w - 1] as i64) < 0 { u64::MAX } else { 0 };
    let q = (s / 64) as usize;
    let r = (s % 64) as u32;
    if q >= w {
        x.fill(fill);
        return;
    }
    if r == 0 {
        x.copy_within(q.., 0);
    } else {
        for k in 0..w - q - 1 {
            x[k] = (x[k + q] >> r) | (x[k + q + 1] << (64 - r));
        }
        x[w - q - 1] = ((x[w - 1] as i64) >> r) as u64;
    }
    x[w - q..].fill(fill);
}

/// The coefficients of a polynomial in double precision with bounds of their magnitudes: each
/// pair `(v, a)` with `v` the coefficient times a common power of two, rounded, and `a` at least
/// `|v|`, for the sign tests of Descartes's rule without exact arithmetic.
#[derive(Default)]
pub(crate) struct Approx {
    pub(crate) v: Vec<[f64; 2]>,
    /// Working limbs.
    tmp: Vec<u64>,
}

/// The unit roundoff of `f64`.
const U: f64 = f64::EPSILON / 2.0;

impl Flat {
    /// The polynomial in double precision, scaled so that `shifts` Taylor shifts by 1 cannot
    /// overflow; `None` when the range of `f64` is too small for them (`n·shifts` above 1700).
    pub(crate) fn approx(&self, shifts: u64, ap: &mut Approx) -> bool {
        let n = self.len as u64 - 1;
        if n * shifts > 1700 {
            return false;
        }
        let top = self.bits() as i64;
        // The largest magnitude becomes 2^(1000 − n·shifts) at most.
        let scale = 1000 - (n * shifts) as i64 - top;
        let v = &mut ap.v;
        v.clear();
        let tmp = &mut ap.tmp;
        tmp.clear();
        tmp.resize(self.w, 0);
        for i in 0..self.len {
            let x = self.coeff(i);
            let neg = (x[self.w - 1] as i64) < 0;
            let m: &[u64] = if neg {
                tmp.copy_from_slice(x);
                negate(tmp);
                tmp
            } else {
                x
            };
            let f = to_f64_scaled(m, scale);
            let f = if neg { -f } else { f };
            // The inputs' rounding (and underflow) are in the bound: |v| + 2^−1000.
            v.push([f, f.abs() + TINY]);
        }
        true
    }
}

/// The magnitude `m` (limbs) times `2^scale`, rounded to a double (to within a unit in the last
/// place: truncated to its top 64 bits first).
fn to_f64_scaled(m: &[u64], scale: i64) -> f64 {
    let Some(k) = m.iter().rposition(|&l| l != 0) else {
        return 0.0;
    };
    let lz = m[k].leading_zeros();
    let hi = if lz == 0 || k == 0 {
        m[k] << lz
    } else {
        (m[k] << lz) | (m[k - 1] >> (64 - lz))
    };
    // m ≈ hi·2^(64k − lz); times 2^scale.
    let e = 64 * k as i64 - i64::from(lz) + scale;
    let f = hi as f64;
    if e > 1023 {
        return f64::INFINITY;
    }
    if e < -1200 {
        return 0.0;
    }
    // Two steps keep the intermediate powers of two in range.
    let h = e / 2;
    f * pow2f(h) * pow2f(e - h)
}

/// `2^−1000`.
const TINY: f64 = f64::from_bits((1023 - 1000) << 52);

/// `2^k` for `−1022 ≤ k ≤ 1023`, exactly.
#[inline]
fn pow2f(k: i64) -> f64 {
    debug_assert!((-1022..=1023).contains(&k));
    f64::from_bits(((k + 1023) as u64) << 52)
}

impl Approx {
    /// The Taylor shift by 1 of the values and of the bounds, in place.
    pub(crate) fn taylor1(&mut self) {
        let n = self.v.len() - 1;
        for i in 0..n {
            let a = &mut self.v[i..];
            let m = a.len() - 1;
            let mut acc = a[m];
            for x in a[..m].iter_mut().rev() {
                acc = [x[0] + acc[0], x[1] + acc[1]];
                *x = acc;
            }
        }
    }

    /// The number of sign variations of the coefficients of the Taylor shift by 1 of the
    /// reversal, as far as 2, when every sign it needs is certain: `ops` additions the values
    /// went through before, each bounded by `ops·u` of the magnitude. `None` when a sign is not
    /// certain.
    pub(crate) fn test(&mut self, ops: u64) -> Option<u32> {
        self.v.reverse();
        let n = self.v.len() - 1;
        // The bound of the error of a value: (ops + n + 2)·u·(1 + 1/32)·a.
        let k = (ops + n as u64 + 2) as f64 * U * 1.03125;
        let mut v = 0;
        let mut last = 0;
        let mut sign = |x: [f64; 2]| -> Option<bool> {
            if !(x[0].is_finite() && x[1].is_finite()) || x[0].abs() <= k * x[1] {
                return None;
            }
            let s = if x[0] < 0.0 { -1 } else { 1 };
            if last != 0 && s != last {
                v += 1;
            }
            last = s;
            Some(v >= 2)
        };
        for i in 0..n {
            let a = &mut self.v[i..];
            let m = a.len() - 1;
            let mut acc = a[m];
            for x in a[..m].iter_mut().rev() {
                acc = [x[0] + acc[0], x[1] + acc[1]];
                *x = acc;
            }
            if sign(a[0])? {
                return Some(2);
            }
        }
        sign(self.v[n])?;
        Some(v)
    }
}

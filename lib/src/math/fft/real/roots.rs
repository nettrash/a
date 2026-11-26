//! Tables of roots of unity for the plans of the real, cosine and sine transforms, built fast and
//! accurately: `e^(−2πi·k/n) = C[k / b]·F[k mod b]` from a coarse table `C` and a fine one `F` of
//! about `√k_max` values each, both and the product carried in double-double arithmetic, so that
//! each value is the correctly rounded root but for ties within about 10⁻²⁸ of it. The two short
//! tables come from two roots summed by Taylor's series, then by products (each product's
//! relative error below 2⁻¹⁰⁰, a few thousand of them at most). A root costs one product of
//! double-doubles (about 30 operations, independent of the others) instead of a sine and a
//! cosine.

use crate::math::complex::{Complex, Float};

/// A double-double `hi + lo`, `|lo| ≤ ulp(hi)/2`.
#[derive(Clone, Copy)]
struct Dd(f64, f64);

/// A complex double-double.
#[derive(Clone, Copy)]
struct Cd {
    re: Dd,
    im: Dd,
}

/// `a + b` exactly as `s + e`.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `a + b` exactly as `s + e` for `|a| ≥ |b|`.
#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

/// `a·b` exactly as `p + e`.
#[inline(always)]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    (p, a.mul_add(b, -p))
}

impl Dd {
    #[inline(always)]
    fn add(self, o: Dd) -> Dd {
        let (s, e) = two_sum(self.0, o.0);
        let (h, l) = fast_two_sum(s, e + self.1 + o.1);
        Dd(h, l)
    }

    #[inline(always)]
    fn mul(self, o: Dd) -> Dd {
        let (p, e) = two_prod(self.0, o.0);
        let (h, l) = fast_two_sum(p, e + self.0 * o.1 + self.1 * o.0);
        Dd(h, l)
    }

    /// `self / d` for a double `d`.
    #[inline(always)]
    fn div_f(self, d: f64) -> Dd {
        let q = self.0 / d;
        let (p, e) = two_prod(q, d);
        let r = (self.0 - p - e + self.1) / d;
        let (h, l) = fast_two_sum(q, r);
        Dd(h, l)
    }

    #[inline(always)]
    fn neg(self) -> Dd {
        Dd(-self.0, -self.1)
    }
}

impl Cd {
    /// The product, its parts summed with the exact errors of the leading products.
    #[inline(always)]
    fn mul(self, o: Cd) -> Cd {
        let (a, b, c, d) = (self.re, self.im, o.re, o.im);
        // re = a·c − b·d.
        let (p1, e1) = two_prod(a.0, c.0);
        let (p2, e2) = two_prod(b.0, d.0);
        let (s, es) = two_sum(p1, -p2);
        let lo = es + (e1 - e2) + (a.0 * c.1 + a.1 * c.0) - (b.0 * d.1 + b.1 * d.0);
        let (rh, rl) = fast_two_sum(s, lo);
        // im = a·d + b·c.
        let (p3, e3) = two_prod(a.0, d.0);
        let (p4, e4) = two_prod(b.0, c.0);
        let (s, es) = two_sum(p3, p4);
        let lo = es + (e3 + e4) + (a.0 * d.1 + a.1 * d.0) + (b.0 * c.1 + b.1 * c.0);
        let (ih, il) = fast_two_sum(s, lo);
        Cd {
            re: Dd(rh, rl),
            im: Dd(ih, il),
        }
    }

    #[inline(always)]
    fn round(self) -> (f64, f64) {
        (self.re.0 + self.re.1, self.im.0 + self.im.1)
    }
}

/// `2π` as a double-double.
const TAU: Dd = Dd(core::f64::consts::TAU, 2.449_293_598_294_706_4e-16);

/// `e^(−2πi·k/n)` in double-double: the angle reduced exactly into the first octant, its sine
/// and cosine by Taylor's series.
fn cis(k: u64, n: u64) -> Cd {
    let k = k % n;
    let octant = (8 * k as u128 / n as u128) as u64;
    let q = octant.div_ceil(2);
    // The angle 2π·num/(4n), |num| ≤ n/2.
    let num = 4 * k as i128 - q as i128 * n as i128;
    let den = 4.0 * n as f64;
    let f_hi = num as f64 / den;
    let f_lo = (-f_hi).mul_add(den, num as f64) / den;
    let t = TAU.mul(Dd(f_hi, f_lo));
    let t2 = t.mul(t);
    let (mut s, mut c) = (t, Dd(1.0, 0.0));
    let (mut ts, mut tc) = (t, Dd(1.0, 0.0));
    for i in 1..14 {
        let i = i as f64;
        ts = ts.mul(t2).div_f(-(2.0 * i) * (2.0 * i + 1.0));
        tc = tc.mul(t2).div_f(-(2.0 * i - 1.0) * (2.0 * i));
        s = s.add(ts);
        c = c.add(tc);
        // The terms fall below 2⁻¹¹⁰ of the sums (at least 1/√2 here): done.
        if ts.0.abs() < 1e-33 && tc.0.abs() < 1e-33 {
            break;
        }
    }
    // (cos, sin) of the angle, rotated by q quarter turns; the root is cos − i·sin.
    let (c, s) = match q % 4 {
        0 => (c, s),
        1 => (s.neg(), c),
        2 => (c.neg(), s.neg()),
        _ => (s, c.neg()),
    };
    Cd { re: c, im: s.neg() }
}

/// The roots `e^(−2πi·(o + s·k)/n)` for `k` in `0..len`, from a coarse and a fine table:
/// `fine[j]` for the exponents `s·j`, `j < b`, and `coarse[a]` for `o + s·b·a`.
pub(in crate::math::fft) struct Roots {
    b: usize,
    len: usize,
    fine: Vec<Cd>,
    coarse: Vec<Cd>,
}

/// The powers `w^0, w^1, …` of `w`, `len` of them, from `start`.
fn powers(start: Cd, w: Cd, len: usize) -> Vec<Cd> {
    let mut v = Vec::with_capacity(len);
    let mut x = start;
    for _ in 0..len {
        v.push(x);
        x = x.mul(w);
    }
    v
}

/// `(cr + i·ci)·(fr + i·fi)` rounded, lane by lane.
#[inline]
fn products<T: Float>(cr: f64, ci: f64, fr: &[f64], fi: &[f64], out: &mut [Complex<T>]) {
    let n = out.len();
    let (fr, fi) = (&fr[..n], &fi[..n]);
    for j in 0..n {
        let re = cr * fr[j] - ci * fi[j];
        let im = cr * fi[j] + ci * fr[j];
        out[j] = Complex::new(T::from_f64(re), T::from_f64(im));
    }
}

/// The double-double products `c·f` rounded, lane by lane: [`Cd::mul`] on the parts.
#[inline]
fn products_dd<T: Float>(c: Cd, f: (&[f64], &[f64], &[f64], &[f64]), out: &mut [Complex<T>]) {
    let n = out.len();
    let (fr, fi, frl, fil) = (&f.0[..n], &f.1[..n], &f.2[..n], &f.3[..n]);
    let (a, b) = (c.re, c.im);
    for j in 0..n {
        let (p1, e1) = two_prod(a.0, fr[j]);
        let (p2, e2) = two_prod(b.0, fi[j]);
        let (s, es) = two_sum(p1, -p2);
        let lo = es + (e1 - e2) + (a.0 * frl[j] + a.1 * fr[j]) - (b.0 * fil[j] + b.1 * fi[j]);
        let re = s + lo;
        let (p3, e3) = two_prod(a.0, fi[j]);
        let (p4, e4) = two_prod(b.0, fr[j]);
        let (s, es) = two_sum(p3, p4);
        let lo = es + (e3 + e4) + (a.0 * fil[j] + a.1 * fi[j]) + (b.0 * frl[j] + b.1 * fr[j]);
        out[j] = Complex::new(T::from_f64(re), T::from_f64(s + lo));
    }
}

impl Roots {
    /// The tables for the denominator `n ≥ 1`, the offset `o`, the step `s` and `len` roots.
    pub(in crate::math::fft) fn new(n: usize, o: usize, s: usize, len: usize) -> Roots {
        let b = len.isqrt().max(1);
        let (n64, s64) = (n as u64, s as u64);
        let one = Cd {
            re: Dd(1.0, 0.0),
            im: Dd(0.0, 0.0),
        };
        let fine = powers(one, cis(s64, n64), b);
        let step = cis(((s64 as u128 * b as u128) % n64 as u128) as u64, n64);
        let coarse = powers(cis(o as u64, n64), step, len.div_ceil(b));
        Roots {
            b,
            len,
            fine,
            coarse,
        }
    }

    /// The `len` roots in `T`, the products a coarse value at a time. An `f32` table needs no
    /// double-double: the product of the two roots rounded to `f64`, within about 2⁻⁵² of the
    /// root, rounds to the nearest `f32` but for ties closer than that.
    pub(in crate::math::fft) fn table<T: Float>(&self) -> Vec<Complex<T>> {
        let mut out = vec![Complex::ZERO; self.len];
        let single = T::MANTISSA_DIGITS < 53;
        // The fine table's parts one array each, for the loops below to run in vectors.
        let fr: Vec<f64> = self.fine.iter().map(|f| f.re.0).collect();
        let fi: Vec<f64> = self.fine.iter().map(|f| f.im.0).collect();
        let frl: Vec<f64> = self.fine.iter().map(|f| f.re.1).collect();
        let fil: Vec<f64> = self.fine.iter().map(|f| f.im.1).collect();
        for (c, out) in self.coarse.iter().zip(out.chunks_mut(self.b)) {
            if single {
                let (cr, ci) = c.round();
                products(cr, ci, &fr, &fi, out);
            } else {
                products_dd(*c, (&fr, &fi, &frl, &fil), out);
            }
        }
        out
    }
}

/// The tables of fewer roots than this take them one by one from [`twiddle::root`]: the two
/// short tables would cost more.
///
/// [`twiddle::root`]: crate::math::fft::twiddle::root
const SHORT: usize = 128;

/// `e^(−2πi·(o + s·k)/n)` for `k` in `0..len`: by the two tables for long ones, one root at a
/// time for short ones.
pub(in crate::math::fft) fn affine<T: Float>(
    n: usize,
    o: usize,
    s: usize,
    len: usize,
) -> Vec<Complex<T>> {
    if len < SHORT {
        (0..len)
            .map(|k| {
                let e = (o as u64 + s as u64 * k as u64) % n as u64;
                crate::math::fft::twiddle::root_t(e as usize, n)
            })
            .collect()
    } else {
        Roots::new(n, o, s, len).table()
    }
}

/// `e^(−2πi·k/n)` for `k` in `0..len`.
pub(in crate::math::fft) fn table<T: Float>(n: usize, len: usize) -> Vec<Complex<T>> {
    affine(n, 0, 1, len)
}

/// Any root `e^(−2πi·k/n)`, `k < n`, for the `f32` tables of the complex passes: the product of
/// a coarse table (exponents `2^b` apart) and a fine one (exponents below `2^b`) of roots rounded
/// to `f64`, rounded to `f32` as [`Roots::table`] rounds an `f32` table (the nearest `f32` but
/// for ties closer than about 2⁻⁵²). A root costs two loads from tables of about `√n` values
/// and a complex product, where a table of `n/8` roots to fold into would be megabytes.
pub(in crate::math::fft) struct Grid {
    bits: u32,
    fine: Vec<(f64, f64)>,
    coarse: Vec<(f64, f64)>,
}

impl Grid {
    /// The tables for `n ≥ 1`.
    pub(in crate::math::fft) fn new(n: usize) -> Grid {
        let bits = (usize::BITS - n.saturating_sub(1).leading_zeros()).div_ceil(2);
        let b = 1usize << bits;
        let n64 = n as u64;
        let one = Cd {
            re: Dd(1.0, 0.0),
            im: Dd(0.0, 0.0),
        };
        let fine = powers(one, cis(1, n64), b);
        let coarse = powers(one, cis(b as u64 % n64, n64), n.div_ceil(b));
        Grid {
            bits,
            fine: fine.iter().map(|c| c.round()).collect(),
            coarse: coarse.iter().map(|c| c.round()).collect(),
        }
    }

    /// `e^(−2πi·k/n)` in `f32`, for `k < n`.
    #[inline(always)]
    pub(in crate::math::fft) fn get(&self, k: usize) -> Complex<f32> {
        let (cr, ci) = self.coarse[k >> self.bits];
        let (fr, fi) = self.fine[k & ((1 << self.bits) - 1)];
        Complex::new((cr * fr - ci * fi) as f32, (cr * fi + ci * fr) as f32)
    }
}

//! Shared helpers: a random generator, a reference DFT summed in double-double arithmetic, error
//! measures and the float types under test.

#![allow(dead_code)]

use a::math::complex::Complex;
use a::math::fft::FftFloat;

/// xorshift64*.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [−1, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    pub fn reals<T: FftFloat>(&mut self, n: usize) -> Vec<T> {
        (0..n).map(|_| T::from_f64(self.unit())).collect()
    }

    pub fn complexes<T: FftFloat>(&mut self, n: usize) -> Vec<Complex<T>> {
        (0..n)
            .map(|_| Complex::new(T::from_f64(self.unit()), T::from_f64(self.unit())))
            .collect()
    }
}

/// A double-double: `hi + lo` with `|lo| ≤ ulp(hi)/2`.
#[derive(Clone, Copy, Default)]
pub struct Dd(pub f64, pub f64);

impl Dd {
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }

    pub fn add(self, o: Dd) -> Dd {
        let (s, e) = Dd::two_sum(self.0, o.0);
        let e = e + self.1 + o.1;
        let (h, l) = Dd::two_sum(s, e);
        Dd(h, l)
    }

    /// The exact product of two doubles.
    pub fn prod(a: f64, b: f64) -> Dd {
        let p = a * b;
        Dd(p, a.mul_add(b, -p))
    }

    /// `self · (c + d)` for doubles `c` (a value) and `d` (its correction), to double-double.
    pub fn mul_f(self, c: f64, d: f64) -> Dd {
        let p = Dd::prod(self.0, c);
        Dd(p.0, p.1 + self.1 * c + self.0 * d).add(Dd(0.0, 0.0))
    }

    pub fn value(self) -> f64 {
        self.0 + self.1
    }
}

/// `cos(2π·k/n)` and `sin(2π·k/n)` as double-doubles, from an angle reduced to the first octant
/// and its sine and cosine to double-double by Taylor series.
pub fn cis(k: u64, n: u64) -> (Dd, Dd) {
    let k = k % n;
    // 2π as a double-double.
    const TAU_HI: f64 = core::f64::consts::TAU;
    const TAU_LO: f64 = 2.449_293_598_294_706_4e-16;
    let k8 = 8 * k as u128;
    let octant = (k8 / n as u128) as u64;
    let q = octant.div_ceil(2);
    let num = 4 * k as i128 - q as i128 * n as i128; // angle = 2π·num/(4n), |num| ≤ n/2
    // t = τ·num/(4n) to double-double: τ·(num/(4n)) with num/(4n) as a double-double.
    let f_hi = num as f64 / (4.0 * n as f64);
    let f_lo = (-f_hi).mul_add(4.0 * n as f64, num as f64) / (4.0 * n as f64);
    let t = Dd::prod(TAU_HI, f_hi).add(Dd(TAU_HI * f_lo + TAU_LO * f_hi, 0.0));
    // Taylor series of sin and cos at |t| ≤ π/4, in double-double.
    let t2 = mul(t, t);
    let (mut s, mut c) = (t, Dd(1.0, 0.0));
    let (mut ts, mut tc) = (t, Dd(1.0, 0.0));
    for i in 1..14 {
        let i = i as f64;
        ts = mul(ts, t2);
        ts = div_f(ts, -(2.0 * i) * (2.0 * i + 1.0));
        tc = mul(tc, t2);
        tc = div_f(tc, -(2.0 * i - 1.0) * (2.0 * i));
        s = s.add(ts);
        c = c.add(tc);
    }
    let neg = |d: Dd| Dd(-d.0, -d.1);
    match q % 4 {
        0 => (c, s),
        1 => (neg(s), c),
        2 => (neg(c), neg(s)),
        _ => (s, neg(c)),
    }
}

fn mul(a: Dd, b: Dd) -> Dd {
    let p = Dd::prod(a.0, b.0);
    Dd(p.0, p.1 + a.0 * b.1 + a.1 * b.0).add(Dd(0.0, 0.0))
}

fn div_f(a: Dd, d: f64) -> Dd {
    let q = a.0 / d;
    let r = Dd::prod(q, d);
    let rem = (a.0 - r.0 - r.1 + a.1) / d;
    Dd(q, rem).add(Dd(0.0, 0.0))
}

/// The DFT `Σ x[j]·e^(sign·2πi·jk/n)` of `x`, each value to about one rounding of `f64`.
pub fn dft(x: &[Complex<f64>], sign: f64) -> Vec<Complex<f64>> {
    let n = x.len() as u64;
    let table: Vec<(Dd, Dd)> = (0..n).map(|k| cis(k, n)).collect();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (Dd::default(), Dd::default());
            for (j, v) in x.iter().enumerate() {
                let (c, s) = table[((j as u128 * k as u128) % n as u128) as usize];
                let s = Dd(sign * s.0, sign * s.1);
                // (a + bi)(c + si) = (ac − bs) + (as + bc)i
                re = re.add(c.mul_f(v.re, 0.0)).add(s.mul_f(-v.im, 0.0));
                im = im.add(s.mul_f(v.re, 0.0)).add(c.mul_f(v.im, 0.0));
            }
            Complex::new(re.value(), im.value())
        })
        .collect()
}

pub fn widen<T: FftFloat>(x: &[Complex<T>]) -> Vec<Complex<f64>> {
    x.iter()
        .map(|v| Complex::new(v.re.to_f64(), v.im.to_f64()))
        .collect()
}

pub fn widen_real<T: FftFloat>(x: &[T]) -> Vec<f64> {
    x.iter().map(|v| v.to_f64()).collect()
}

/// `‖got − want‖₂ / ‖want‖₂` (the absolute norm when `want` is zero).
pub fn rel_error(got: &[Complex<f64>], want: &[Complex<f64>]) -> f64 {
    assert_eq!(got.len(), want.len());
    let (mut num, mut den) = (0.0, 0.0);
    for (g, w) in got.iter().zip(want) {
        num += (*g - *w).norm_sqr();
        den += w.norm_sqr();
    }
    if den == 0.0 {
        num.sqrt()
    } else {
        (num / den).sqrt()
    }
}

pub fn rel_error_real(got: &[f64], want: &[f64]) -> f64 {
    let g: Vec<_> = got.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let w: Vec<_> = want.iter().map(|&x| Complex::new(x, 0.0)).collect();
    rel_error(&g, &w)
}

/// The error bound for a transform of the length `n` in the type `T`: `c·ε·(log₂ n + 1)`.
pub fn bound<T: FftFloat>(n: usize, c: f64) -> f64 {
    c * T::EPSILON.to_f64() * ((n.max(1) as f64).log2() + 1.0)
}

/// The lengths whose transforms of dense random data are checked against [`dft`]: all up to
/// 64, then smooth, prime and mixed ones up to 1024.
pub fn dense_lengths() -> Vec<usize> {
    let mut v: Vec<usize> = (0..=64).collect();
    if cfg!(miri) {
        v.retain(|&n| n <= 24);
        v.extend_from_slice(&[31, 32, 37, 64]);
    } else {
        v.extend_from_slice(&[
            96, 100, 120, 127, 128, 131, 210, 243, 256, 289, 343, 360, 384, 500, 512, 625, 997,
            1000, 1009, 1024,
        ]);
    }
    v
}

/// Longer lengths, checked on sparse data against [`dft`] and by round trips.
pub fn long_lengths() -> Vec<usize> {
    if cfg!(miri) {
        return Vec::new();
    }
    vec![
        1155, 1331, 1994, 2048, 2187, 2310, 3125, 4096, 4099, 4913, 5040, 8192, 15015, 16384,
        65536, 65537, 100_000,
    ]
}

/// The DFT of a sparse `x` (most values zero), in time `O(n·nonzeros)`.
pub fn dft_sparse(x: &[Complex<f64>], sign: f64) -> Vec<Complex<f64>> {
    let n = x.len() as u64;
    let nz: Vec<(u64, Complex<f64>)> = x
        .iter()
        .enumerate()
        .filter(|(_, v)| v.re != 0.0 || v.im != 0.0)
        .map(|(j, &v)| (j as u64, v))
        .collect();
    (0..n)
        .map(|k| {
            let (mut re, mut im) = (Dd::default(), Dd::default());
            for &(j, v) in &nz {
                let (c, s) = cis(((j as u128 * k as u128) % n as u128) as u64, n);
                let s = Dd(sign * s.0, sign * s.1);
                re = re.add(c.mul_f(v.re, 0.0)).add(s.mul_f(-v.im, 0.0));
                im = im.add(s.mul_f(v.re, 0.0)).add(c.mul_f(v.im, 0.0));
            }
            Complex::new(re.value(), im.value())
        })
        .collect()
}

//! Seeded inputs: the tests' SplitMix64 and their bases, and the lattices of fplll's
//! `latticegen` (knapsack `r`, NTRU-like `n`, q-ary `q`, uniform `u`) and Goldstein–Mayer's.
use a::math::matrix::Matrix;
use a::math::numbers::Number;
use a::math::rational::Rational;

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number of `bits` random bits (nonnegative).
    pub fn unsigned(&mut self, bits: u64) -> Number {
        let mut x = Number::default();
        let mut left = bits;
        while left > 0 {
            let take = left.min(64);
            let w = if take == 64 { self.next() } else { self.next() >> (64 - take) };
            x = (x << take as usize) + Number::from(w);
            left -= take;
        }
        x
    }

    /// A number of `bits` random bits, of a random sign (the tests' `number`).
    pub fn number(&mut self, bits: u64) -> Number {
        let x = self.unsigned(bits);
        if self.next() & 1 == 1 { -x } else { x }
    }

    /// Uniform in `[0, q)`.
    pub fn below(&mut self, q: &Number) -> Number {
        let b = q.bits() + 64;
        self.unsigned(b) % q
    }

    pub fn range(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % ((hi - lo + 1) as u64)) as i64
    }
}

/// `[aᵢ | eᵢ]`, `n × (n + 1)`, `aᵢ` of `bits` bits (fplll's `latticegen r`).
pub fn knapsack(rng: &mut Rng, n: usize, bits: u64) -> Matrix<Number> {
    let a: Vec<Number> = (0..n).map(|_| rng.unsigned(bits)).collect();
    Matrix::from_fn(n, n + 1, |i, j| {
        if j == 0 { a[i].clone() } else { Number::from(u8::from(i + 1 == j)) }
    })
}

/// `n × m` uniform signed entries of `bits` bits.
pub fn uniform(rng: &mut Rng, n: usize, m: usize, bits: u64) -> Matrix<Number> {
    Matrix::from_fn(n, m, |_, _| rng.number(bits))
}

/// The NTRU-like `2d × 2d` basis `[[I, Rot(h)], [0, q·I]]`, `q` of `qbits` bits, `h` uniform
/// modulo `q` (fplll's `latticegen n d b`).
pub fn ntru(rng: &mut Rng, d: usize, qbits: u64) -> Matrix<Number> {
    let q = rng.unsigned(qbits) | (Number::from(1u8) << (qbits as usize - 1));
    let h: Vec<Number> = (0..d).map(|_| rng.below(&q)).collect();
    Matrix::from_fn(2 * d, 2 * d, |i, j| {
        if i < d {
            if j < d { Number::from(u8::from(i == j)) } else { h[(j - d + d - i) % d].clone() }
        } else if j < d {
            Number::default()
        } else {
            if i == j { q.clone() } else { Number::default() }
        }
    })
}

/// The q-ary `n × n` basis `[[I_{n−k}, H], [0, q·I_k]]`, `H` uniform modulo `q` of `qbits`
/// bits (fplll's `latticegen q n k b`).
pub fn qary(rng: &mut Rng, n: usize, k: usize, qbits: u64) -> Matrix<Number> {
    let q = rng.unsigned(qbits) | (Number::from(1u8) << (qbits as usize - 1));
    let m = n - k;
    let h: Vec<Number> = (0..m * k).map(|_| rng.below(&q)).collect();
    Matrix::from_fn(n, n, |i, j| {
        if i < m {
            if j < m { Number::from(u8::from(i == j)) } else { h[i * k + (j - m)].clone() }
        } else if i == j {
            q.clone()
        } else {
            Number::default()
        }
    })
}

/// A Goldstein–Mayer basis, `n × n`: the first row `(p, 0, …, 0)` for a prime `p` of `bits`
/// bits, row `i` `(xᵢ, eᵢ)` with `xᵢ` uniform modulo `p` (the SVP challenge's lattices).
pub fn goldstein_mayer(rng: &mut Rng, n: usize, bits: u64) -> Matrix<Number> {
    let p = (rng.unsigned(bits) | (Number::from(1u8) << (bits as usize - 1))).next_prime();
    let x: Vec<Number> = (0..n).map(|_| rng.below(&p)).collect();
    Matrix::from_fn(n, n, |i, j| {
        if i == 0 {
            if j == 0 { p.clone() } else { Number::default() }
        } else if j == 0 {
            x[i].clone()
        } else {
            Number::from(u8::from(i == j))
        }
    })
}

/// `n × m` small integers in `lo..=hi`.
pub fn small(rng: &mut Rng, n: usize, m: usize, lo: i64, hi: i64) -> Matrix<Number> {
    Matrix::from_fn(n, m, |_, _| Number::from(rng.range(lo, hi)))
}

/// An `n × m` matrix of rank `r`: the product of uniform `n × r` and `r × m` factors of `bits`.
pub fn low_rank(rng: &mut Rng, n: usize, m: usize, r: usize, bits: u64) -> Matrix<Number> {
    let x = uniform(rng, n, r, bits);
    let y = uniform(rng, r, m, bits);
    &x * &y
}

/// Rationals `p/q`, `|p| ≤ 99`, `1 ≤ q ≤ 99`.
pub fn fractions(rng: &mut Rng, n: usize, m: usize) -> Matrix<Rational> {
    Matrix::from_fn(n, m, |_, _| {
        let p = rng.range(-99, 99);
        let q = rng.range(1, 99);
        Rational::from(p) / Rational::from(q)
    })
}

/// The text of a matrix, row-major, entries separated by spaces.
pub fn text<T: std::fmt::Display>(a: &Matrix<T>) -> String {
    let mut s = String::new();
    for i in 0..a.rows() {
        for j in 0..a.cols() {
            s.push_str(&a[(i, j)].to_string());
            s.push(' ');
        }
    }
    s
}

pub fn parse_matrix(s: &str, r: usize, c: usize) -> Matrix<Number> {
    let v: Vec<Number> = s.split_whitespace().map(|t| t.parse().unwrap()).collect();
    assert_eq!(v.len(), r * c, "the reference returned {} entries, not {}", v.len(), r * c);
    Matrix::from_fn(r, c, |i, j| v[i * c + j].clone())
}

/// FNV-1a of words, as the shims compute it.
pub fn fnv(h: u64, w: &[u64]) -> u64 {
    let mut h = h;
    for &x in w {
        for b in 0..8 {
            h ^= (x >> (8 * b)) & 0xff;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    h
}
pub const FNV_INIT: u64 = 0xcbf29ce484222325;

/// The squared norm of row `i`.
pub fn norm2(a: &Matrix<Number>, i: usize) -> Number {
    (0..a.cols()).map(|j| &a[(i, j)] * &a[(i, j)]).sum()
}

/// The squared norm of the first nonzero row.
pub fn first_norm2(a: &Matrix<Number>) -> Number {
    for i in 0..a.rows() {
        let n = norm2(a, i);
        if !n.is_zero() {
            return n;
        }
    }
    Number::default()
}

/// log₂(x / y) for positive numbers.
pub fn log2_ratio(x: &Number, y: &Number) -> f64 {
    let lx = x.bits() as f64 + frac_log(x);
    let ly = y.bits() as f64 + frac_log(y);
    lx - ly
}

fn frac_log(x: &Number) -> f64 {
    // log₂ of the leading 53 bits over 2^bits.
    let b = x.bits();
    if b == 0 {
        return 0.0;
    }
    let top = if b > 60 { x >> (b - 60) as usize } else { x.clone() };
    let t: f64 = top.to_string().parse().unwrap();
    t.log2() - (b.min(60)) as f64
}

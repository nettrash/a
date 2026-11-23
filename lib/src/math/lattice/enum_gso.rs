//! The exact Gram–Schmidt data that the enumeration and BKZ work on, in the integral form of de
//! Weger and Cohen (Cohen, *A Course in Computational Algebraic Number Theory*, Algorithm 2.6.7,
//! step 2): with `dᵢ = |b*₀|²·…·|b*ᵢ₋₁|²` the Gram determinant of the first `i` rows, the
//! coefficients `λᵢⱼ = dⱼ₊₁·μᵢⱼ` are integers, and so is `dⱼ·|πⱼ(w)|²` for every integral vector
//! `w`, where `πⱼ` projects orthogonally to `b₀, …, bⱼ₋₁`. Every division below is exact.
//!
//! Also the conversion of a quotient of two numbers to a float with a bound on its error, which
//! the floating-point enumeration needs to be exact.

use crate::math::numbers::Number;

/// The dot product of two integral vectors.
pub(super) fn dot(a: &[Number], b: &[Number]) -> Number {
    let mut s = Number::default();
    for (x, y) in a.iter().zip(b) {
        if !x.is_zero() && !y.is_zero() {
            s += x * y;
        }
    }
    s
}

/// The integral Gram–Schmidt orthogonalisation of linearly independent rows `b₀, b₁, …`: the
/// Gram determinants `d₀ = 1, d₁, …` and the lower triangle of `λᵢⱼ = dⱼ₊₁·μᵢⱼ`.
///
/// Rows are orthogonalised one after another and the data can be cut back to any row, so that a
/// basis changed from row `f` on is orthogonalised again from row `f` only.
#[derive(Clone, Debug)]
pub(super) struct Gso {
    /// `d[i]`, `i ≤ len`: the Gram determinant of the first `i` rows.
    pub(super) d: Vec<Number>,
    /// `lam[i][j]`, `j < i < len`: `λᵢⱼ = dⱼ₊₁·μᵢⱼ`.
    pub(super) lam: Vec<Vec<Number>>,
}

impl Gso {
    /// No rows yet.
    pub(super) fn new() -> Self {
        Gso {
            d: vec![Number::from(1u8)],
            lam: Vec::new(),
        }
    }

    /// The data of the first `n` rows of `rows`; `None` if they are linearly dependent.
    pub(super) fn of(rows: &[Vec<Number>], n: usize) -> Option<Self> {
        let mut g = Gso::new();
        g.extend(rows, n).then_some(g)
    }

    /// How many rows are orthogonalised.
    pub(super) fn len(&self) -> usize {
        self.lam.len()
    }

    /// Forgets every row from row `n` on.
    pub(super) fn truncate(&mut self, n: usize) {
        if n < self.len() {
            self.lam.truncate(n);
            self.d.truncate(n + 1);
        }
    }

    /// Orthogonalises `rows[len..n]`; `false`, with the rows before the first dependent one
    /// orthogonalised, if a row depends on those before it.
    pub(super) fn extend(&mut self, rows: &[Vec<Number>], n: usize) -> bool {
        for k in self.len()..n {
            let mut row: Vec<Number> = Vec::with_capacity(k);
            for j in 0..=k {
                // u = d_j·⟨b_k, b*_j⟩·…, brought down one Gram determinant at a time.
                let mut u = dot(&rows[k], &rows[j]);
                for (i, (ri, di)) in row.iter().zip(&self.d).enumerate().take(j) {
                    let li = if j == k { ri } else { &self.lam[j][i] };
                    u = &(&(&self.d[i + 1] * &u) - &(ri * li)) / di;
                }
                if j < k {
                    row.push(u);
                } else {
                    if u.is_zero() {
                        return false;
                    }
                    self.d.push(u);
                }
            }
            self.lam.push(row);
        }
        true
    }

    /// The coefficients of a vector `t` on the orthogonalised rows, `λ_t,j = d_{j+1}·μ_t,j` for
    /// `j < len`, by the recurrence of a row.
    pub(super) fn coefficients(&self, rows: &[Vec<Number>], t: &[Number]) -> Vec<Number> {
        let n = self.len();
        let mut lam: Vec<Number> = Vec::with_capacity(n);
        for (row, lj) in rows.iter().zip(&self.lam).take(n) {
            let mut u = dot(t, row);
            for (i, (li, lji)) in lam.iter().zip(lj).enumerate() {
                u = &(&(&self.d[i + 1] * &u) - &(li * lji)) / &self.d[i];
            }
            lam.push(u);
        }
        lam
    }

    /// `T_j = d_j·|π_j(t)|²` for `j = 0, …, len`, from `|t|²` and the coefficients of `t`.
    pub(super) fn projected_norms(&self, norm2: &Number, lam: &[Number]) -> Vec<Number> {
        let n = self.len();
        let mut t = Vec::with_capacity(n + 1);
        t.push(norm2.clone());
        for i in 0..n {
            let next = &(&(&self.d[i + 1] * &t[i]) - &(&lam[i] * &lam[i])) / &self.d[i];
            t.push(next);
        }
        t
    }
}

/// `a / b` for `b > 0` as `(m, e)` with `a / b = m·2^e·(1 + ε)`, `|ε| < 1.01·2⁻⁵³`: `m` is an
/// integer of 63 or 64 bits rounded once to the nearest `f64`, or zero.
pub(super) fn ratio(a: &Number, b: &Number) -> (f64, i64) {
    debug_assert!(b.is_positive());
    if a.is_zero() {
        return (0.0, 0);
    }
    // The quotient of 64 to 65 bits, truncated (a relative error below 2⁻⁶³), then rounded to
    // 53 bits by `to_f64` (below 2⁻⁵³).
    let shift = 64 + b.bits() as i64 - a.bits() as i64;
    let q = if shift >= 0 {
        &(a.abs() << shift as usize) / b
    } else {
        &a.abs() / &(b << (-shift) as usize)
    };
    let m = q.to_f64();
    (if a.is_negative() { -m } else { m }, -shift)
}

/// `m·2^e` for finite `m`, without overflow or underflow on the way: `±∞` above the range of an
/// `f64`, and towards zero (possibly subnormal or zero) below it.
pub(super) fn scale(m: f64, e: i64) -> f64 {
    let mut m = m;
    let mut e = e;
    // Steps of 2^±600 keep every intermediate value finite and normal until the last.
    while e > 600 {
        m *= f64::from_bits((1023 + 600) << 52);
        e -= 600;
        if m.is_infinite() {
            return m;
        }
    }
    while e < -600 {
        m *= f64::from_bits((1023 - 600) << 52);
        e += 600;
        if m == 0.0 {
            return m;
        }
    }
    m * f64::from_bits(((1023 + e) as u64) << 52)
}

/// `⌊log₂(a / b)⌋` within one, for `a, b > 0`: the exponent a scale is chosen from.
pub(super) fn log2_ratio(a: &Number, b: &Number) -> i64 {
    a.bits() as i64 - b.bits() as i64
}

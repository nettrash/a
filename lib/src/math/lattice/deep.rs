//! LLL with deep insertions (Schnorr and Euchner, *Lattice basis reduction: improved practical
//! algorithms and solving subset sum problems*, Math. Programming 66, 1994, Section 3): a vector
//! `bₖ`, size-reduced, is inserted at the first place `i < k` where its projection `πᵢ(bₖ)`
//! orthogonal to `b₀, …, bᵢ₋₁` is shorter than `√δ·|b*ᵢ|`, not only at `k − 1` as LLL inserts.
//!
//! It runs on the machinery of the L² algorithm ([`lll_fp`](super::lll_fp)): the basis and its
//! Gram matrix exactly, the Gram–Schmidt coefficients in floating point from them, the result
//! proved reduced by intervals or exactly, with the exact integral reduction, which inserts deeply
//! too, as the last resort.

use super::Reduced;
use super::exact::{Insertion, check_delta, check_eta, reduce_exactly};
use super::l2::reduce_fp;
use crate::math::matrix::Matrix;
use crate::math::numbers::Number;

/// The parameters of [`deep_lll`].
///
/// # Examples
///
/// ```
/// use a::math::lattice::DeepLllOptions;
///
/// let o = DeepLllOptions::new().with_depth(Some(5)).with_delta(0.9);
/// assert_eq!((o.delta, o.eta, o.depth), (0.9, 0.51, Some(5)));
/// assert_eq!(DeepLllOptions::default().depth, None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepLllOptions {
    /// `δ` of the deep exchange condition; `0.99` by default.
    pub delta: f64,
    /// `η` of the size reduction; `0.51` by default.
    pub eta: f64,
    /// How deep a vector may be inserted: `Some(t)` only at the first `t` places or the last
    /// `t` before it (and always at the place just before it, LLL's exchange, so `Some(0)` is
    /// LLL); `None`, anywhere. `None` by default.
    pub depth: Option<usize>,
}

impl Default for DeepLllOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DeepLllOptions {
    /// `δ = 0.99`, `η = 0.51`, insertions anywhere.
    #[must_use]
    pub fn new() -> Self {
        DeepLllOptions {
            delta: 0.99,
            eta: 0.51,
            depth: None,
        }
    }

    /// Sets [`delta`](Self::delta).
    #[must_use]
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta;
        self
    }

    /// Sets [`eta`](Self::eta).
    #[must_use]
    pub fn with_eta(mut self, eta: f64) -> Self {
        self.eta = eta;
        self
    }

    /// Sets [`depth`](Self::depth).
    #[must_use]
    pub fn with_depth(mut self, depth: Option<usize>) -> Self {
        self.depth = depth;
        self
    }
}

/// Reduces the rows of `basis` by LLL with deep insertions: each vector is inserted at the
/// first place where that shortens the Gram–Schmidt vector there by the factor `δ`. The result
/// is also `(δ, η)`-LLL-reduced.
///
/// Precisely, the result `b₀, …, b_{n−1}` (after the zero rows, one for each linear dependency
/// among the rows given) is size-reduced, `|μᵢⱼ| ≤ η`, and meets the deep exchange condition
/// `|πᵢ(bₖ)|² ≥ δ·|b*ᵢ|²` for every `k` and every place `i < k` that
/// [`depth`](DeepLllOptions::depth) allows; at `i = k − 1` that is Lovász's condition, so the
/// basis is `(δ, η)`-LLL-reduced, as [`is_lll_reduced`](super::is_lll_reduced) checks it. These
/// conditions are proved on the result, by interval arithmetic or exactly, and the exact
/// integral reduction finishes the work if the floating point does not carry it; which of the lattice's reduced bases it is, is not promised beyond that (see [`Reduced`](super::Reduced)). With `η = 1/2`
/// exactly, which a floating-point size reduction cannot meet in general (a coefficient of exactly
/// `1/2` may round either way; L² needs `η > 1/2`), the whole reduction is the exact integral one,
/// with deep insertions, as [`lll`](super::lll) is for LLL.
///
/// Deep insertions find shorter vectors than LLL, at a cost that is not known to be polynomial:
/// an insertion can raise LLL's potential, and the reduction ends because the sequence
/// `(|b*₀|², |b*₁|², …)` falls lexicographically, each insertion lowering one of its terms by
/// the factor `δ` and leaving those before it. A depth limits the places where a vector may be
/// inserted, and so the tests made for each.
///
/// # Panics
///
/// Panics if `δ` is not in `(1/4, 1)` or `η` not in `[1/2, √δ)`.
///
/// # Examples
///
/// ```
/// use a::math::lattice::{DeepLllOptions, LllOptions, deep_lll, is_lll_reduced, lll_fp};
/// use a::math::matrix::Matrix;
/// use a::math::numbers::Number;
///
/// // A knapsack basis [aᵢ | eᵢ] of 12 numbers of about 60 bits.
/// let a: Vec<u64> = (1..=12u64).map(|i| 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i) >> 4).collect();
/// let b = Matrix::from_fn(12, 13, |i, j| {
///     Number::from(if j == 0 { a[i] } else { u64::from(i + 1 == j) })
/// });
/// let deep = deep_lll(&b, &DeepLllOptions::new());
/// assert!(is_lll_reduced(deep.basis(), 0.99, 0.51));
/// assert_eq!(&deep.transform().clone() * &b, *deep.basis());
/// // Its first vector is no longer than LLL's, here.
/// let norm2 = |m: &Matrix<Number>| (0..13).map(|j| &m[(0, j)] * &m[(0, j)]).sum::<Number>();
/// let plain = lll_fp(&b, &LllOptions::new());
/// assert!(norm2(deep.basis()) <= norm2(plain.basis()));
/// ```
#[track_caller]
pub fn deep_lll(basis: &Matrix<Number>, opts: &DeepLllOptions) -> Reduced {
    check_delta(opts.delta, false);
    check_eta(opts.eta, opts.delta, true);
    let ins = Insertion::Deep(opts.depth);
    if opts.eta == 0.5 {
        // |μ| ≤ 1/2 exactly: a floating-point size reduction cannot settle a coefficient of
        // exactly 1/2 (L² needs η > 1/2), the integral one can.
        return reduce_exactly(basis, opts.delta, ins);
    }
    reduce_fp(basis, opts.delta, opts.eta, ins)
}

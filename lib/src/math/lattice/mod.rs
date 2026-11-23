//! Lattices: the reduction of lattice bases by LLL, L², LLL with deep insertions and BKZ; the
//! shortest and closest vectors of a lattice and its short vectors by enumeration; and integer
//! relations among real numbers by PSLQ.
//!
//! A basis is a [`Matrix<Number>`] whose **rows** are the vectors `b₀, …, b_{n−1}` that generate
//! the lattice `{ Σ xᵢ·bᵢ : xᵢ ∈ ℤ }`, as fplll, NTL and FLINT have them. The rows may be linearly
//! dependent: a reduction then returns zero rows first, as many as the dependencies, and the
//! reduced basis of the lattice after them.
//!
//! A reduction returns a [`Reduced`]: the reduced basis `B'`, the unimodular transformation `U`
//! with `U·B = B'`, and the rank of the lattice.
//!
//! * [`lll`]: LLL reduction exactly, on integers (the integral LLL of de Weger and Cohen: the
//!   Gram–Schmidt coefficients held as integers `dⱼ·μᵢⱼ`, with no fraction and no rounding).
//! * [`lll_fp`]: the L² algorithm of Nguyen and Stehlé, the Gram–Schmidt orthogonalisation in
//!   floating point, of 53 bits first and of the precision of their theorem where needed, the
//!   result proved `(δ, η)`-LLL-reduced afterwards.
//! * [`deep_lll`]: LLL with deep insertions (Schnorr and Euchner), on the machinery of L², the
//!   result proved as that of [`lll_fp`].
//! * [`bkz`]: block Korkine–Zolotarev reduction (Schnorr and Euchner), by enumeration in blocks:
//!   most tours in floating point, the last ones exact, so that, unless
//!   [`max_tours`](BkzOptions::max_tours) stops it, every block's first Gram–Schmidt vector ends
//!   a shortest vector of the block's projected lattice.
//! * [`shortest_vector`], [`closest_vector`], [`short_vectors`]: enumeration of the lattice
//!   points in an ellipsoid (Fincke and Pohst; Schnorr and Euchner's order), exact, for
//!   dimensions up to about 45.
//! * [`is_lll_reduced`]: whether a basis is `(δ, η)`-LLL-reduced, exactly.
//! * [`pslq`], [`pslq_f64`]: integer relations `Σ mᵢ·xᵢ = 0` among real numbers given to some
//!   precision, by the PSLQ algorithm of Ferguson and Bailey, or a proved bound below which
//!   there is none.
//!
//! The Hermite and Smith normal forms of a basis are
//! [`Matrix::hermite_normal_form`] and [`Matrix::smith_normal_form`].
//!
//! # Examples
//!
//! ```
//! use a::math::lattice::{LllOptions, is_lll_reduced, lll};
//! use a::math::matrix::Matrix;
//! use a::math::numbers::Number;
//!
//! let b = Matrix::from_rows(&[[1, 1, 1], [-1, 0, 2], [3, 5, 6]]).map(|&x| Number::from(x));
//! let r = lll(&b, &LllOptions::new());
//! assert!(is_lll_reduced(r.basis(), 0.99, 0.51));
//! assert_eq!(&r.transform().clone() * &b, *r.basis());
//! assert_eq!(r.rank(), 3);
//! ```

mod bkz;
mod bkz_fast;
mod deep;
mod enum_exact;
mod enum_float;
mod enum_gso;
mod enum_search;
mod enumerate;
mod exact;
mod l2;
mod l2_certify;
mod l2_core;
mod l2_float;
mod pslq;
mod pslq_bound;

pub use bkz::{BkzOptions, bkz};
pub use deep::{DeepLllOptions, deep_lll};
pub use enumerate::{LatticeVector, closest_vector, short_vectors, shortest_vector};
pub use exact::{is_lll_reduced, lll};
pub use l2::lll_fp;
pub use pslq::{Pslq, PslqOptions, pslq, pslq_f64};

use crate::math::matrix::Matrix;
use crate::math::numbers::Number;

/// The parameters of an LLL reduction: Lovász's condition
/// `|b*ₖ|² ≥ (δ − μ²ₖ,ₖ₋₁)·|b*ₖ₋₁|²` and the size reduction `|μᵢⱼ| ≤ η`.
///
/// # Examples
///
/// ```
/// use a::math::lattice::LllOptions;
///
/// let o = LllOptions::new().with_delta(0.75).with_eta(0.5);
/// assert_eq!((o.delta, o.eta), (0.75, 0.5));
/// assert_eq!(LllOptions::new().delta, 0.99);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LllOptions {
    /// `δ`, in `(1/4, 1]`: `0.99` by default, as fplll's. It is taken exactly, as the dyadic
    /// rational the float is.
    pub delta: f64,
    /// `η`, in `[1/2, √δ)`: `0.51` by default, as fplll's. The exact reduction [`lll`] reduces
    /// to `1/2` whatever it is.
    pub eta: f64,
}

impl Default for LllOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl LllOptions {
    /// `δ = 0.99` and `η = 0.51`.
    #[must_use]
    pub fn new() -> Self {
        LllOptions {
            delta: 0.99,
            eta: 0.51,
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
}

/// A reduced basis: the basis `B'` (zero rows first, one for each linear dependency among the
/// rows given, then the reduced basis of the lattice), the unimodular `U` with `U·B = B'`, and
/// the rank of the lattice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reduced {
    pub(crate) basis: Matrix<Number>,
    pub(crate) transform: Matrix<Number>,
    pub(crate) rank: usize,
}

impl Reduced {
    /// The reduced basis `B'`, a row a vector, the zero rows first.
    #[must_use]
    pub fn basis(&self) -> &Matrix<Number> {
        &self.basis
    }

    /// The unimodular transformation `U`, with `U·B = B'`.
    #[must_use]
    pub fn transform(&self) -> &Matrix<Number> {
        &self.transform
    }

    /// The rank of the lattice: the number of nonzero rows of [`basis`](Self::basis), which are
    /// its last rows.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// The basis and the transformation.
    #[must_use]
    pub fn into_parts(self) -> (Matrix<Number>, Matrix<Number>) {
        (self.basis, self.transform)
    }
}

//! Rational sequences: [`Rational::bernoulli`] and [`Rational::bernoulli_numbers`], the Bernoulli
//! numbers one at a time and all up to an index, and [`Rational::harmonic`], the harmonic numbers.
//!
//! A Bernoulli number beyond the table is `|B_n| = 2 n! ζ(n) / (2π)^n`: its denominator from von
//! Staudt and Clausen's theorem, its numerator rounded from the formula computed to its bits, the
//! zeta function by its Euler product; from `n = 8000`, the low fifth of the numerator's bits
//! modulo primes by Harvey's method instead, which leaves the formula fewer bits and far fewer
//! primes ([`bernoulli`], [`modp`]). The table of them all runs the formula from the top index
//! down, each power of the zeta function's sum from the last by a product with a limb, after the
//! tangent numbers for short tables. Harmonic numbers are sums over the odd numbers by binary
//! splitting, reduced without a greatest common divisor of the long parts ([`harmonic`]).

mod bernoulli;
mod harmonic;
pub(super) mod limbs;
mod modp;

use super::Rational;
use crate::math::numbers::ByteOrder;

impl Rational {
    /// Returns the Bernoulli number `B_n`, in lowest terms, both parts in the byte order
    /// `byte_order`.
    ///
    /// `B_0 = 1`, `B_1 = -1/2`, `B_2 = 1/6`, `B_4 = -1/30`, and `B_n = 0` for every odd `n > 1`:
    /// the numbers of `x / (e^x - 1) = Σ B_n x^n / n!`, the convention of PARI, FLINT and
    /// Mathematica. The other convention, `B_1 = +1/2` (of `x / (1 - e^-x)`), differs only there:
    /// it is `-Rational::bernoulli(1, o)`, or `(-1)^n B_n` for any `n`.
    ///
    /// Up to `B_100` from tables (`B_58` the last numerator below `2^127`, then 75 limbs); those
    /// whose parts fit in 64 bits (to `B_34`, and `B_38`) are written as words in the caller,
    /// with nothing allocated, 10 cycles measured on an Apple M4 Max against 17 for FLINT 3.6.
    /// Beyond, the denominator by von Staudt and Clausen's theorem, the product of the primes `p`
    /// with `p - 1` dividing `n`, and the numerator from
    /// `|B_n| = 2 n! ζ(n) / (2π)^n`, as PARI's `bernfrac` and FLINT's `bernoulli_fmpq_ui` do: `π`,
    /// `(2π)^n` and `1/ζ(n) = Π_p (1 - p^-n)`, each to the bits of the numerator and a few more,
    /// their errors bounded, and the result rounded. From `n = 8000`, as FLINT does, a fifth of the
    /// numerator's bits (from 10 % at 8000 to 22 % from `10^5`) are its residues modulo the primes
    /// from 5 up, by Harvey's method (2010): Voronoi's congruence with `c = 2`, the binary
    /// expansions of `x / p` over the cosets of `<±2>` weighted by tables of 16 or 256 sums; and the
    /// formula then needs that many bits fewer, which cuts the primes of its Euler product most.
    /// Measured on an Apple M4 Max, `B_(10^5)` in 0.13 s against 0.26 s for FLINT 3.6, and
    /// `B_(10^6)` in 7.5 s against 12.3 s.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector: `B_n` has
    /// about `n log2(n / (2πe))` bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::ByteOrder;
    /// use a::math::rational::Rational;
    ///
    /// let o = ByteOrder::LittleEndian;
    /// assert_eq!(Rational::bernoulli(1, o).to_string(), "-1/2");
    /// assert_eq!(Rational::bernoulli(12, o).to_string(), "-691/2730");
    /// assert!(Rational::bernoulli(13, o).is_zero());
    /// // B_1 = +1/2 in the other convention.
    /// assert_eq!((-Rational::bernoulli(1, o)).to_string(), "1/2");
    /// // B_60 = -1215233140483755572040304994079820246041491/56786730.
    /// let b = Rational::bernoulli(60, o);
    /// assert_eq!(b.denom().to_string(), "56786730");
    /// assert_eq!(b.numer().to_string(), "-1215233140483755572040304994079820246041491");
    /// ```
    #[inline]
    pub fn bernoulli(n: u64, byte_order: ByteOrder) -> Rational {
        bernoulli::bernoulli(n, byte_order)
    }

    /// Returns the Bernoulli numbers `B_0, B_1, ..., B_n` (`n + 1` of them), each in lowest terms
    /// in the byte order `byte_order`, as [`bernoulli`](Self::bernoulli) gives them one by one.
    ///
    /// Up to `B_100` from the tables, the short ones as words; up to `n = 150`, from the tangent
    /// numbers by Brent and Harvey's recurrence (2011), `m^2 / 2` steps on integers for
    /// `m = n / 2`, and `|B_2k| = 2k T_k / (4^k (4^k - 1))`; beyond, by the zeta function from the
    /// top index down, as FLINT's `bernoulli_fmpq_vec_no_cache` does: `G_k = 2 k! / (2π)^k` from
    /// `G_(k+2)` by a product with `(2π)^2` and a division by `(k + 2)(k + 1)`, and `ζ(k)` from the
    /// odd powers `j^-k` in fixed point, each from `j^-(k+2)` by a product with the limb `j^2`, the
    /// precisions falling with `k` and the errors bounded as they go. Taking them one by one would
    /// cost far more: the table to `10^4` took 0.38 s, measured on an Apple M4 Max, against 0.51 s
    /// for FLINT 3.6's, while `B_(10^4)` alone takes 3.2 ms.
    ///
    /// # Panics
    ///
    /// Only when the results or their working memory cannot be allocated, as for any vector: the
    /// numbers up to `n` take about `n^2 log2(n / 17) / 32` bytes, 27 MB for `n = 10^4` and 3.7 GB for
    /// `10^5`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::ByteOrder;
    /// use a::math::rational::Rational;
    ///
    /// let o = ByteOrder::BigEndian;
    /// let b = Rational::bernoulli_numbers(10, o);
    /// assert_eq!(b.len(), 11);
    /// assert_eq!(b[10].to_string(), "5/66");
    /// assert!(b[9].is_zero());
    /// // The recurrence Σ_(k=0)^m C(m+1, k) B_k = 0 for m = 10.
    /// let s: Rational = (0..=10u64)
    ///     .map(|k| &b[k as usize] * a::math::numbers::Number::binomial(11, k, o))
    ///     .sum();
    /// assert!(s.is_zero());
    /// ```
    pub fn bernoulli_numbers(n: u64, byte_order: ByteOrder) -> Vec<Rational> {
        bernoulli::bernoulli_numbers(n, byte_order)
    }

    /// Returns the harmonic number `H_n = 1 + 1/2 + ... + 1/n` (`H_0 = 0`), in lowest terms, both
    /// parts in the byte order `byte_order`.
    ///
    /// Up to `H_88`, the last whose parts are below `2^128`, from a table made at compile time (to
    /// `H_46` written as words in the caller, with nothing allocated: 9 cycles measured on an Apple
    /// M4 Max), and up to `H_124` as `H_88` plus the rest of the terms, added as rationals.
    /// Beyond, the terms are gathered by their odd parts, `H_n = 2^-T Σ_(m odd) c_m / m` with
    /// `T = floor(log2 n)` and small integer weights `c_m`, as FLINT's `fmpq_harmonic_ui` does, and
    /// summed by binary splitting: leaves of as many terms as fit in registers, added in place into
    /// pairs of up to 16 limbs, then a balanced tree. The fraction is reduced without the greatest
    /// common divisor of its long parts: the numerator `A = H_n 2^T L`, for `L` the least common
    /// multiple of the odd numbers up to `n`, is `P L Q^-1` modulo `2^b` for its `b` bits, the top
    /// levels of the tree are taken modulo `2^b` too, and only the primes `p` that divide the
    /// numerator of `H_k`, `k = floor(n / p^e)`, rare, can divide both `A` and `L`. Measured on an
    /// Apple M4 Max, `H_(10^4)` in 0.18 ms against 0.41 ms for FLINT 3.6 and `H_(10^6)` in 79 ms
    /// against 262 ms.
    ///
    /// # Panics
    ///
    /// Only when the result or its working memory cannot be allocated, as for any vector: `H_n` has
    /// about `1.44 n` bits in each part.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::numbers::ByteOrder;
    /// use a::math::rational::Rational;
    ///
    /// let o = ByteOrder::LittleEndian;
    /// assert_eq!(Rational::harmonic(4, o).to_string(), "25/12");
    /// assert_eq!(Rational::harmonic(0, o).to_string(), "0");
    /// // H_n - H_(n-1) = 1/n.
    /// let d = Rational::harmonic(1000, o) - Rational::harmonic(999, o);
    /// assert_eq!(d.to_string(), "1/1000");
    /// ```
    #[inline]
    pub fn harmonic(n: u64, byte_order: ByteOrder) -> Rational {
        harmonic::harmonic(n, byte_order)
    }
}

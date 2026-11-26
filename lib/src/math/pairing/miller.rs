//! The optimal ate pairing: Miller's loop on the twist in homogeneous projective coordinates, and
//! the final exponentiation.

use super::curve::PairingCurve;
use super::fast::{self, Engine, with_engine};
use super::g1::G1;
use super::g2::G2;
use super::gt::Gt;
use crate::math::field::Fp12;

impl PairingCurve {
    /// The product of the Miller functions of the optimal ate pairing for the pairs, before the
    /// final exponentiation: one loop for all of them, with one square of `F_p¹²` per bit.
    ///
    /// The loop runs over the non-adjacent form of `6x + 2` for BN (65 doublings and 21
    /// additions on BN254), with the two lines through `ψ(Q)` and `−ψ²(Q)` after it, and over the
    /// binary form of `|x|` for BLS12 (63 doublings and 5 additions on BLS12-381), conjugated when
    /// `x < 0`
    /// (Vercauteren, *Optimal pairings*, IEEE Trans. Inf. Theory 56, 2010). `T` is held in
    /// homogeneous projective coordinates on the twist, and each line is evaluated at `P` as a
    /// sparse element of `F_p¹²` (see [`Fp12::mul_by_034`] and [`Fp12::mul_by_014`]). Pairs with a
    /// point at infinity contribute 1.
    ///
    /// The value is a representative: lines are scaled by factors of `F_p²`, which the final
    /// exponentiation removes, so it equals other implementations' only after that.
    ///
    /// # Panics
    ///
    /// Panics if a point is of another curve.
    #[track_caller]
    #[must_use]
    pub fn miller_loop<'c>(&'c self, pairs: &[(&G1<'c>, &G2<'c>)]) -> Fp12<'c> {
        with_engine!(self, E, N => fast::miller_loop::<E, N>(&self.fast_pairs::<E, N>(pairs)).to_public())
    }

    /// `f^(c (p¹² − 1)/r)`, the final exponentiation, as an element of `GT`.
    ///
    /// The easy part `f^((p⁶ − 1)(p² + 1))` takes a conjugation, an inversion and a Frobenius
    /// map, and leaves an element of the cyclotomic subgroup, where squares are Granger and Scott's
    /// and inverses are conjugates. The hard part raises to `c (p⁴ − p² + 1)/r` by products by
    /// `x` and Frobenius maps:
    ///
    /// * BLS12: `c = 3`, as `3 (p⁴ − p² + 1)/r = (x − 1)² (x + p) (x² + p² − 1) + 3` (Hayashida,
    ///   Hayasaka and Teruya, *Efficient final exponentiation via cyclotomic structure for pairings
    ///   over families of elliptic curves*, ePrint 2020/875): five powers by `x`.
    /// * BN: `c = 2x(6x² + 3x + 1)`, by the chain of Fuentes-Castañeda, Knapp and
    ///   Rodríguez-Henríquez (*Faster hashing to G2*, SAC 2011, §5) for
    ///   `λ₀ + λ₁p + λ₂p² + λ₃p³` with `λ₀ = 12x³ + 12x² + 6x + 1`, `λ₁ = 12x³ + 6x² + 4x`,
    ///   `λ₂ = 12x³ + 6x² + 6x`, `λ₃ = 12x³ + 6x² + 4x − 1`: three powers by `x`.
    ///
    /// The factor `c`, prime to `r`, is the one of arkworks, of the `bls12_381` crate and of blst,
    /// whose pairings this one equals. Others differ by a fixed power. The IETF draft on
    /// pairing-friendly curves raises `f_{x,Q}(P)` to `(p¹² − 1)/r` alone: on BLS12-381 its value
    /// is `e^(1/3)`. py_ecc raises to `(p¹² − 1)/r` alone too, and on BLS12-381 loops over `|x|`
    /// without conjugating: its values are `e^(1/c)` on BN254 and `e^(−1/3)` on BLS12-381. To
    /// compare a value of `GT` with theirs, raise it to `c⁻¹ mod r` (`e.pow(&c.mod_inverse(r).unwrap())`),
    /// and for py_ecc on BLS12-381 take the inverse of that. A check that a product of pairings
    /// is 1, as signatures and proofs are verified, gives the same answer as all of them, since
    /// `c` is prime to `r`.
    ///
    /// # Panics
    ///
    /// Panics if `f` is 0 (no Miller loop gives 0) or not of the curve's tower.
    #[track_caller]
    #[must_use]
    pub fn final_exponentiation<'c>(&'c self, f: &Fp12<'c>) -> Gt<'c> {
        assert!(
            f.tower().xi() == self.tower.xi(),
            "elements of different towers"
        );
        with_engine!(self, E, N => {
            let f = fast::Fp12::<E, N>::from_public(f);
            Gt::from_fast::<E, N>(self, &fast::final_exponentiation(&f))
        })
    }

    /// The optimal ate pairing `e(P, Q)`: [`miller_loop`](PairingCurve::miller_loop) then
    /// [`final_exponentiation`](PairingCurve::final_exponentiation). Bilinear and non-degenerate on
    /// `G1 × G2`; 1 when either point is at infinity.
    ///
    /// For points of the curves outside `G1` and `G2` (from `new_on_curve`) it returns an element
    /// of `GT` without panicking, but not a pairing.
    ///
    /// # Panics
    ///
    /// Panics if a point is of another curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::PairingCurve;
    /// use a::math::numbers::Number;
    ///
    /// let c = PairingCurve::bn254();
    /// let (p, q) = (c.g1_generator(), c.g2_generator());
    /// let (a, b) = (Number::from(6u8), Number::from(7u8));
    /// let e = c.pairing(&(&p * &a), &(&q * &b));
    /// assert_eq!(e, c.pairing(&p, &q).pow(&Number::from(42u8)));
    /// assert_eq!(e, c.pairing(&(&p * &b), &(&q * &a)));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn pairing<'c>(&'c self, p: &G1<'c>, q: &G2<'c>) -> Gt<'c> {
        self.multi_pairing(&[(p, q)])
    }

    /// The product `∏ e(Pᵢ, Qᵢ)` with one Miller loop and one final exponentiation, as the checks of
    /// signatures and proofs use it; 1 for no pairs.
    ///
    /// # Panics
    ///
    /// Panics if a point is of another curve.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::PairingCurve;
    /// use a::math::numbers::Number;
    ///
    /// let c = PairingCurve::bls12_381();
    /// let (p, q) = (c.g1_generator(), c.g2_generator());
    /// let five = Number::from(5u8);
    /// // e(5P, Q) · e(−P, 5Q) = 1.
    /// let product = c.multi_pairing(&[(&(&p * &five), &q), (&-&p, &(&q * &five))]);
    /// assert!(product.is_identity());
    /// assert!(c.multi_pairing(&[]).is_identity());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn multi_pairing<'c>(&'c self, pairs: &[(&G1<'c>, &G2<'c>)]) -> Gt<'c> {
        with_engine!(self, E, N => {
            let f = fast::miller_loop::<E, N>(&self.fast_pairs::<E, N>(pairs));
            Gt::from_fast::<E, N>(self, &fast::final_exponentiation(&f))
        })
    }
}

impl PairingCurve {
    /// The affine coordinates of the pairs in the engine's arithmetic, the pairs with a point at
    /// infinity left out.
    #[track_caller]
    fn fast_pairs<'c, E: Engine<N>, const N: usize>(
        &'c self,
        pairs: &[(&G1<'c>, &G2<'c>)],
    ) -> FastPairs<E, N> {
        // The points' Jacobian forms, pairs with a point at infinity left out.
        let mut jac = Vec::with_capacity(pairs.len());
        for (p, q) in pairs {
            self.check(p.curve);
            self.check(q.curve);
            let (a, b) = (p.jac::<E, N>(), q.jac::<E, N>());
            if !a.is_identity() && !b.is_identity() {
                jac.push((a, b));
            }
        }
        // Their affine forms with one inversion of F_p for all, by Montgomery's trick: Z of a point
        // of G1, and the norm Z·Z̄ of a point of G2, whose inverse gives 1/Z = Z̄/(Z·Z̄).
        let d: Vec<fast::Fp<E, N>> = jac.iter().flat_map(|(a, b)| [a.z, b.z.norm()]).collect();
        let mut prefix = Vec::with_capacity(d.len());
        let mut acc = fast::Fp::<E, N>::one();
        for x in &d {
            prefix.push(acc);
            acc = acc.mul(x);
        }
        let Some(mut inv) = acc.inverse() else {
            unreachable!("Z of a point not at infinity is not zero")
        };
        let mut inverses = vec![fast::Fp::<E, N>::zero(); d.len()];
        for i in (0..d.len()).rev() {
            inverses[i] = inv.mul(&prefix[i]);
            inv = inv.mul(&d[i]);
        }
        jac.iter()
            .enumerate()
            .map(|(i, (a, b))| {
                let zi = inverses[2 * i];
                let zi2 = zi.square();
                let wi = b.z.conjugate().mul_by_fp(&inverses[2 * i + 1]);
                let wi2 = wi.square();
                (
                    (a.x.mul(&zi2), a.y.mul(&zi2.mul(&zi))),
                    (b.x.mul(&wi2), b.y.mul(&wi2.mul(&wi))),
                )
            })
            .collect()
    }
}

/// Pairs of affine points of the curve and of the twist.
type FastPairs<E, const N: usize> = Vec<fast::pairing::Pair<E, N>>;

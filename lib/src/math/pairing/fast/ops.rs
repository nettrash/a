//! The operations of `G1`, `G2` and `GT` on the engine's arithmetic: products by scalars with the
//! curves' endomorphisms, subgroup tests, cofactor clearing and powers in `GT`.
//!
//! * `G1`, by the GLV method (Gallant, Lambert and Vanstone, CRYPTO 2001) with `φ(x, y) = (βx, y)`,
//!   `φ = [λ]` on `G1`: on BLS12, `λ = −x²`, so a scalar below `r < x⁴` is `a + b·x²` with `a`
//!   and `b` below `2^128`, and `[k]P = [a]P − [b]φ(P)`; on BN, `k = k₁ + k₂λ` with `k₁`, `k₂` of
//!   half the length, from the short basis of the lattice `{(a, b) : a + bλ ≡ 0 (mod r)}` that the
//!   extended Euclidean algorithm gives (the *Guide to Elliptic Curve Cryptography*, §3.5).
//! * `G2`, by the GLS method (Galbraith, Lin and Scott, EUROCRYPT 2009) with `ψ = [p]` on `G2`:
//!   on BLS12 `p ≡ x (mod r)`, so the scalar's four digits in base `|x|` multiply `Q`, `−ψ(Q)`,
//!   `ψ²(Q)` and `−ψ³(Q)`; on BN `p ≡ 6x² (mod r)`, and four signed digits of 64 bits or so from
//!   a reduced basis of the lattice of the relations `Σ vᵢ(6x²)ⁱ ≡ 0 (mod r)` multiply `Q`,
//!   `ψ(Q)`, `ψ²(Q)` and `ψ³(Q)` (Galbraith and Scott, Pairing 2008). The digits' products share
//!   their doublings (Straus's method).
//! * `GT`, the same decompositions with the Frobenius map `f ↦ f^p`.
//! * Products by `x`, in the tests of `G1` and `G2`, the clearing of cofactors and the final
//!   exponentiation, run over the bits of `|x|` (6 ones in 64 bits on BLS12-381).
//!
//! The endomorphisms act as these scalars only on the subgroups: a point that is not known to be
//! in its subgroup is multiplied by the whole scalar.

use super::group::{Coord, Jac, straus, wnaf_limbs};
use super::{Engine, Fp, Fp2, Fp12};
use crate::math::lattice::{LllOptions, lll};
use crate::math::matrix::Matrix;
use crate::math::numbers::{ByteOrder, Number};
use crate::math::pairing::curve::Family;

/// The limbs of `|n|`, the low limb first (none for 0).
pub(crate) fn abs_limbs(n: &Number) -> Vec<u64> {
    let mut b = n.value.clone();
    if n.byte_order == ByteOrder::BigEndian {
        b.reverse();
    }
    let mut l: Vec<u64> = b
        .chunks(8)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        })
        .collect();
    while l.last() == Some(&0) {
        l.pop();
    }
    l
}

/// The bits of `|n|`, least significant first.
fn abs_bits(n: &Number) -> Vec<bool> {
    let n = n.abs();
    (0..n.bits()).map(|i| n.bit(i)).collect()
}

/// `round(a / b)` for `b > 0`.
fn round_div(a: &Number, b: &Number) -> Number {
    (a * 2u8 + b).div_euclid(&(b * 2u8))
}

/// The constants of the groups' operations, computed on their first use.
pub(crate) struct GroupTables<E, const N: usize> {
    /// The bits of `|x|`, least significant first, and the sign of `x`.
    pub(crate) x_bits: Vec<bool>,
    pub(crate) x_negative: bool,
    /// `r`.
    pub(crate) r: Number,
    /// `β`, with `φ(x, y) = (βx, y)` acting as `[λ]` on `G1`.
    pub(crate) beta: Fp<E, N>,
    /// The decomposition of scalars on `G1`.
    glv: Glv,
    /// The base of the decompositions on `G2` and `GT`: `|x|` on BLS12, `6x²` on BN.
    pub(crate) gls_base: Number,
    /// BN's four-dimensional decomposition on `G2` and `GT`.
    gls4: Option<Gls4>,
    /// The digits of `6x²` (BN) and of `|x|` (BLS12), for `G2`'s test.
    pub(crate) g2_test: Vec<i8>,
    /// The limbs of the cofactors.
    pub(crate) h1: Vec<u64>,
    pub(crate) h2: Vec<u64>,
    /// Whether `f^p = f^(base)` decides membership of `GT` in the cyclotomic subgroup.
    pub(crate) gt_frobenius_test: bool,
}

/// How `G1`'s scalars are split.
enum Glv {
    /// `k = a + b·x²`, `[k]P = [a]P − [b]φ(P)`.
    Bls { x2: Number },
    /// `k = k₁ + k₂λ` by the reduced basis `(a₁, b₁)`, `(a₂, b₂)`.
    Bn {
        a1: Number,
        b1: Number,
        a2: Number,
        b2: Number,
    },
}

/// The scalars' decomposition `k ≡ k₀ + k₁λ + k₂λ² + k₃λ³ (mod r)` for an eigenvalue `λ` of an
/// endomorphism of order 12 (BN's `ψ` on `G2` and Frobenius on `GT`, `λ = p mod r = 6x²`),
/// with digits of about a quarter of `r`'s length (Galbraith and Scott, "Exponentiation in
/// pairing-friendly groups using homomorphisms", Pairing 2008): an LLL-reduced basis `B` of the
/// lattice `{v : Σ vᵢλⁱ ≡ 0 (mod r)}`, and Babai's rounding of `(k, 0, 0, 0)·B⁻¹`.
struct Gls4 {
    /// The rows of `B`.
    basis: [[Number; 4]; 4],
    /// The first row of the adjugate of `B`: `(k, 0, 0, 0)·B⁻¹ = k·ℓ / det B`.
    ell: [Number; 4],
    /// `det B = ±r`, made positive with `ℓ`.
    det: Number,
}

impl Gls4 {
    /// The reduced basis for `λ` modulo `r`, or `None` if it does not give short digits.
    fn build(lambda: &Number, r: &Number) -> Option<Gls4> {
        let mut powers = vec![Number::from(1u8)];
        for i in 1..4 {
            powers.push((&powers[i - 1] * lambda).rem_euclid(r));
        }
        let m = Matrix::from_fn(4, 4, |i, j| match (i, j) {
            (0, 0) => r.clone(),
            (i, 0) => -&powers[i],
            (i, j) if i == j => Number::from(1u8),
            _ => Number::from(0u8),
        });
        let red = lll(&m, &LllOptions::new());
        let b = red.basis();
        let basis: [[Number; 4]; 4] =
            core::array::from_fn(|i| core::array::from_fn(|j| b[(i, j)].clone()));
        for row in &basis {
            let s = (0..4).fold(Number::from(0u8), |s, i| s + &row[i] * &powers[i]);
            if !s.rem_euclid(r).is_zero() {
                return None;
            }
        }
        let minor = |skip: usize| {
            let rows: Vec<usize> = (0..4).filter(|&i| i != skip).collect();
            Matrix::from_fn(3, 3, |i, j| basis[rows[i]][j + 1].clone()).det()
        };
        let mut ell: [Number; 4] =
            core::array::from_fn(|j| if j % 2 == 0 { minor(j) } else { -minor(j) });
        let mut det = Matrix::from_fn(4, 4, |i, j| basis[i][j].clone()).det();
        if det.is_negative() {
            det = -det;
            for l in &mut ell {
                *l = -&*l;
            }
        }
        if &det != r {
            return None;
        }
        let g = Gls4 { basis, ell, det };
        // The digits' bound: half the sum of the basis's entries in each coordinate.
        let bound = (0..4)
            .map(|i| (0..4).fold(Number::from(0u8), |s, j| s + g.basis[j][i].abs()))
            .max()?;
        (bound.bits() <= r.bits() / 4 + 4).then_some(g)
    }

    /// The four digits of `k`, signed.
    fn digits(&self, k: &Number) -> [Number; 4] {
        let alpha: [Number; 4] =
            core::array::from_fn(|j| round_div(&(k * &self.ell[j]), &self.det));
        core::array::from_fn(|i| {
            let s = (0..4).fold(Number::from(0u8), |s, j| s + &alpha[j] * &self.basis[j][i]);
            if i == 0 { k - &s } else { -s }
        })
    }
}

impl<E: Engine<N>, const N: usize> GroupTables<E, N> {
    pub(crate) fn build() -> GroupTables<E, N> {
        let c = E::curve();
        let x = &c.x;
        let r = c.r.clone();
        let x_bits = abs_bits(x);
        let digits = |n: &Number| super::signed_digits(n);
        let p = c.tower.base().modulus();
        let (beta, glv) = match c.family {
            Family::Bls12 => (
                Fp::from_public(c.glv_beta.as_ref().expect("BLS12 has its β")),
                Glv::Bls { x2: x * x },
            ),
            Family::Bn => {
                // λ and β, cube roots of unity modulo r and p, paired by φ(G) = [λ]G.
                let three = Number::from(3u8);
                let s = (&r - &three).sqrt_mod(&r).expect("−3 is a square modulo r");
                let lambda = ((s - 1u8) * ((&r + 1u8) / 2u8)).rem_euclid(&r);
                let t = (p - &three).sqrt_mod(p).expect("−3 is a square modulo p");
                let b = ((t - 1u8) * ((p + 1u8) / 2u8)).rem_euclid(p);
                let mut beta = Fp::<E, N>::from_public(&c.tower.base().element(&b));
                let g = Jac::from_affine(Fp::from_public(&c.g1[0]), Fp::from_public(&c.g1[1]));
                let want = mul_wnaf(&g, &abs_limbs(&lambda), 5);
                let phi = Jac {
                    x: g.x.mul(&beta),
                    y: g.y,
                    z: g.z,
                };
                if !phi.equals(&want) {
                    beta = beta.square();
                    let phi = Jac {
                        x: g.x.mul(&beta),
                        y: g.y,
                        z: g.z,
                    };
                    assert!(phi.equals(&want), "no cube root of unity acts as λ");
                }
                // The extended Euclidean algorithm on (r, λ), to the remainders about √r.
                let sqrt_r = r.isqrt();
                let (mut r0, mut r1) = (r.clone(), lambda.clone());
                let (mut t0, mut t1) = (Number::from(0u8), Number::from(1u8));
                while r1 >= sqrt_r {
                    let q = &r0 / &r1;
                    let r2 = &r0 - &(&q * &r1);
                    let t2 = &t0 - &(&q * &t1);
                    (r0, r1) = (r1, r2);
                    (t0, t1) = (t1, t2);
                }
                // Now r0 ≥ √r > r1: (r1, −t1) and the shorter of (r0, −t0) and the next.
                let q = &r0 / &r1;
                let r2 = &r0 - &(&q * &r1);
                let t2 = &t0 - &(&q * &t1);
                let (a1, b1) = (r1, -&t1);
                let n0 = &r0 * &r0 + &t0 * &t0;
                let n2 = &r2 * &r2 + &t2 * &t2;
                let (a2, b2) = if n0 <= n2 { (r0, -&t0) } else { (r2, -&t2) };
                (beta, Glv::Bn { a1, b1, a2, b2 })
            }
        };
        let gls_base = match c.family {
            Family::Bls12 => x.abs(),
            Family::Bn => x * x * 6u8,
        };
        let gls4 = match c.family {
            Family::Bls12 => None,
            Family::Bn => Gls4::build(&gls_base.rem_euclid(&r), &r),
        };
        let g2_test = digits(&c.g2_test);
        // In GT: f^p = f^(p mod r) for every f of GT; the converse holds in the cyclotomic
        // subgroup exactly when gcd(Φ₁₂(p), p − base') = r for the base' ≡ p (mod r) taken.
        let gt_frobenius_test = {
            let p2 = p * p;
            let phi12 = &p2 * &p2 - &p2 + 1u8;
            let e = match c.family {
                Family::Bls12 => p - x,
                Family::Bn => p - &gls_base,
            };
            phi12.gcd(&e) == r
        };
        GroupTables {
            x_bits,
            x_negative: x.is_negative(),
            r,
            beta,
            glv,
            gls_base,
            gls4,
            g2_test,
            h1: abs_limbs(&c.h1),
            h2: abs_limbs(&c.h2),
            gt_frobenius_test,
        }
    }
}

/// `[k]P` by the width-`w` non-adjacent form, `k` nonnegative.
pub(crate) fn mul_wnaf<C: Coord>(p: &Jac<C>, k: &[u64], w: u32) -> Jac<C> {
    straus(&[(*p, k)], w)
}

/// `[k]P` for any integer `k`, without endomorphisms.
pub(crate) fn mul_plain<C: Coord>(p: &Jac<C>, k: &Number) -> Jac<C> {
    // Measured on an Apple M4 Max (widths 3 to 6 alternated, scalars of 12 to 512 bits, G1 of
    // BLS12-381 and G2 of both curves): width 4 is the fastest or within 2 % from 28 to 230
    // bits and 5 from 256 on; below 28 bits width 3 is 5–16 % faster on BLS12-381 but mixed on
    // BN254 (2 % slower at 16 bits), so 4 stays.
    let w = if k.bits() > 200 { 5 } else { 4 };
    let r = mul_wnaf(p, &abs_limbs(k), w);
    if k.is_negative() { r.neg() } else { r }
}

/// `[|x|]P`.
pub(crate) fn mul_abs_x<C: Coord, E: Engine<N>, const N: usize>(p: &Jac<C>) -> Jac<C> {
    p.mul_bits(&E::groups().x_bits)
}

/// `[x]P`.
pub(crate) fn mul_x<C: Coord, E: Engine<N>, const N: usize>(p: &Jac<C>) -> Jac<C> {
    let r = mul_abs_x::<C, E, N>(p);
    if E::groups().x_negative { r.neg() } else { r }
}

/// `φ(P) = (βX : Y : Z)`.
#[inline]
pub(crate) fn phi<E: Engine<N>, const N: usize>(p: &Jac<Fp<E, N>>) -> Jac<Fp<E, N>> {
    Jac {
        x: p.x.mul(&E::groups().beta),
        y: p.y,
        z: p.z,
    }
}

/// `[k]P` on `G1`; `in_subgroup` says whether `P` is known to be in `G1`.
pub(crate) fn g1_mul<E: Engine<N>, const N: usize>(
    p: &Jac<Fp<E, N>>,
    k: &Number,
    in_subgroup: bool,
) -> Jac<Fp<E, N>> {
    if !in_subgroup {
        return mul_plain(p, k);
    }
    let g = E::groups();
    let k = k.rem_euclid(&g.r);
    match &g.glv {
        Glv::Bls { x2 } => {
            let a = k.rem_euclid(x2);
            let b = k.div_euclid(x2);
            let (la, lb) = (abs_limbs(&a), abs_limbs(&b));
            // Width 4: measured on an Apple M4 Max (widths 3 to 6 alternated, 21 and 31 samples),
            // 0.96 times the time of width 5.
            straus(&[(*p, &la), (phi(p).neg(), &lb)], 4)
        }
        Glv::Bn { a1, b1, a2, b2 } => {
            let r = &g.r;
            let c1 = round_div(&(b2 * &k), r);
            let c2 = round_div(&(-(b1 * &k)), r);
            let k1 = &k - &(&c1 * a1) - &(&c2 * a2);
            let k2 = -(&(&c1 * b1) + &(&c2 * b2));
            let p1 = if k1.is_negative() { p.neg() } else { *p };
            let q = phi(p);
            let p2 = if k2.is_negative() { q.neg() } else { q };
            let (l1, l2) = (abs_limbs(&k1), abs_limbs(&k2));
            // Width 4: measured as on BLS12-381, 0.94 times the time of width 5.
            straus(&[(p1, &l1), (p2, &l2)], 4)
        }
    }
}

/// Whether `P`, a point of the curve, is in `G1`: always on BN; `φ(P) = [−x²]P` on BLS12.
pub(crate) fn g1_in_subgroup<E: Engine<N>, const N: usize>(p: &Jac<Fp<E, N>>) -> bool {
    match E::curve().family {
        Family::Bn => true,
        Family::Bls12 => {
            let x2p = mul_abs_x::<_, E, N>(&mul_abs_x::<_, E, N>(p));
            phi(p).equals(&x2p.neg())
        }
    }
}

/// `G1`'s cofactor cleared: `[1 − x]P` on BLS12 (RFC 9380's `h_eff`), `P` on BN.
pub(crate) fn g1_clear<E: Engine<N>, const N: usize>(p: &Jac<Fp<E, N>>) -> Jac<Fp<E, N>> {
    match E::curve().family {
        Family::Bn => *p,
        Family::Bls12 => p.add(&mul_x::<_, E, N>(p).neg()),
    }
}

/// `ψ(Q)`: `(X̄ γₓ, Ȳ γᵧ, Z̄)`.
#[inline]
pub(crate) fn psi<E: Engine<N>, const N: usize>(q: &Jac<Fp2<E, N>>) -> Jac<Fp2<E, N>> {
    let [gx, gy] = &E::tables().psi;
    Jac {
        x: q.x.conjugate().mul(gx),
        y: q.y.conjugate().mul(gy),
        z: q.z.conjugate(),
    }
}

/// `ψ²(Q)`: `(X n₁, Y n₂, Z)` with the norms of `ψ`'s constants.
#[inline]
pub(crate) fn psi2<E: Engine<N>, const N: usize>(q: &Jac<Fp2<E, N>>) -> Jac<Fp2<E, N>> {
    let [nx, ny] = &E::tables().psi2;
    Jac {
        x: q.x.mul_by_fp(nx),
        y: q.y.mul_by_fp(ny),
        z: q.z,
    }
}

/// `[k]Q` on `G2`; `in_subgroup` says whether `Q` is known to be in `G2`.
pub(crate) fn g2_mul<E: Engine<N>, const N: usize>(
    q: &Jac<Fp2<E, N>>,
    k: &Number,
    in_subgroup: bool,
) -> Jac<Fp2<E, N>> {
    if !in_subgroup {
        return mul_plain(q, k);
    }
    let g = E::groups();
    let mut k = k.rem_euclid(&g.r);
    let base = &g.gls_base;
    match E::curve().family {
        Family::Bls12 => {
            let mut d = Vec::with_capacity(4);
            for _ in 0..4 {
                d.push(abs_limbs(&k.rem_euclid(base)));
                k = k.div_euclid(base);
            }
            let q1 = psi(q);
            let q2 = psi2(q);
            let q3 = psi(&q2);
            // Width 3: measured on an Apple M4 Max (widths 3 to 6 alternated, 21 and 31 samples),
            // 0.97 times the time of width 4.
            straus(
                &[
                    (*q, &d[0]),
                    (q1.neg(), &d[1]),
                    (q2, &d[2]),
                    (q3.neg(), &d[3]),
                ],
                3,
            )
        }
        Family::Bn => match &g.gls4 {
            Some(g4) => {
                let d = g4.digits(&k);
                let q1 = psi(q);
                let q2 = psi(&q1);
                let q3 = psi(&q2);
                let l: [Vec<u64>; 4] = core::array::from_fn(|i| abs_limbs(&d[i]));
                let sign =
                    |p: Jac<Fp2<E, N>>, i: usize| if d[i].is_negative() { p.neg() } else { p };
                // Width 4: measured on an Apple M4 Max, width 3 within 1 % (0.99), 5 and 6 1.16
                // and 1.44 times its time.
                straus(
                    &[
                        (sign(*q, 0), &l[0]),
                        (sign(q1, 1), &l[1]),
                        (sign(q2, 2), &l[2]),
                        (sign(q3, 3), &l[3]),
                    ],
                    4,
                )
            }
            None => {
                let a = abs_limbs(&k.rem_euclid(base));
                let b = abs_limbs(&k.div_euclid(base));
                // Width 5, not measured: BN254 has its four digits above.
                straus(&[(*q, &a), (psi(q), &b)], 5)
            }
        },
    }
}

/// Whether `Q`, a point of the twist, is in `G2`: `ψ(Q) = [x]Q` on BLS12, `[6x²]Q` on BN.
pub(crate) fn g2_in_subgroup<E: Engine<N>, const N: usize>(q: &Jac<Fp2<E, N>>) -> bool {
    let g = E::groups();
    let want = match E::curve().family {
        Family::Bls12 => mul_x::<_, E, N>(q),
        Family::Bn => q.mul_signed(&g.g2_test),
    };
    psi(q).equals(&want)
}

/// `G2`'s cofactor cleared: Budroni and Pintore's map on BLS12, Fuentes-Castañeda, Knapp and
/// Rodríguez-Henríquez's on BN, as the public points compute them.
pub(crate) fn g2_clear<E: Engine<N>, const N: usize>(q: &Jac<Fp2<E, N>>) -> Jac<Fp2<E, N>> {
    match E::curve().family {
        Family::Bls12 => {
            let t1 = mul_x::<_, E, N>(q);
            let t2 = psi(q);
            let t3 = psi2(&q.double()).add(&t2.neg());
            let t2 = mul_x::<_, E, N>(&t1.add(&t2));
            t3.add(&t2).add(&t1.neg()).add(&q.neg())
        }
        Family::Bn => {
            let xq = mul_x::<_, E, N>(q);
            let a = psi(&xq.double().add(&xq));
            let b = psi2(&xq);
            let c = psi(&psi2(q));
            xq.add(&a).add(&b).add(&c)
        }
    }
}

/// `Σ fᵢ^kᵢ` in the cyclotomic subgroup by Straus's method, the exponents nonnegative limbs.
fn gt_straus<E: Engine<N>, const N: usize>(terms: &[(Fp12<E, N>, &[u64])], w: u32) -> Fp12<E, N> {
    let digits: Vec<Vec<i8>> = terms.iter().map(|(_, k)| wnaf_limbs(k, w)).collect();
    let len = digits.iter().map(Vec::len).max().unwrap_or(0);
    if len == 0 {
        return Fp12::one();
    }
    let size = 1usize << (w - 2);
    let tables: Vec<Vec<Fp12<E, N>>> = terms
        .iter()
        .map(|(f, _)| {
            let mut t = Vec::with_capacity(size);
            t.push(*f);
            if size > 1 {
                let f2 = f.cyclotomic_square();
                for i in 1..size {
                    let next = t[i - 1].mul(&f2);
                    t.push(next);
                }
            }
            t
        })
        .collect();
    let mut r = Fp12::one();
    let mut started = false;
    for i in (0..len).rev() {
        if started {
            r = r.cyclotomic_square();
        }
        for (t, d) in tables.iter().zip(&digits) {
            let Some(&d) = d.get(i) else { continue };
            if d != 0 {
                let e = if d > 0 {
                    t[(d as usize) / 2]
                } else {
                    t[(-d as usize) / 2].conjugate()
                };
                r = if started { r.mul(&e) } else { e };
                started = true;
            }
        }
    }
    r
}

/// `f^k` in `GT`, `k` reduced modulo `r`, by the Frobenius decomposition.
pub(crate) fn gt_pow<E: Engine<N>, const N: usize>(f: &Fp12<E, N>, k: &Number) -> Fp12<E, N> {
    let g = E::groups();
    let mut k = k.rem_euclid(&g.r);
    let base = &g.gls_base;
    match E::curve().family {
        Family::Bls12 => {
            let mut d = Vec::with_capacity(4);
            for _ in 0..4 {
                d.push(abs_limbs(&k.rem_euclid(base)));
                k = k.div_euclid(base);
            }
            // f^|x| = conj(f^p), as x < 0 and f^x = f^p on GT.
            let f1 = f.frobenius(1);
            let f2 = f.frobenius(2);
            let f3 = f.frobenius(3);
            let (f1, f3) = if g.x_negative {
                (f1.conjugate(), f3.conjugate())
            } else {
                (f1, f3)
            };
            // Width 4: measured on an Apple M4 Max (widths 3 to 6 alternated), widths 3 and 5 take
            // 1.06 and 1.10 times its time.
            gt_straus(&[(*f, &d[0]), (f1, &d[1]), (f2, &d[2]), (f3, &d[3])], 4)
        }
        Family::Bn => match &g.gls4 {
            Some(g4) => {
                let d = g4.digits(&k);
                let l: [Vec<u64>; 4] = core::array::from_fn(|i| abs_limbs(&d[i]));
                // f^(−k) = conj(f^k) in the cyclotomic subgroup.
                let term = |i: usize| {
                    let fi = f.frobenius(i);
                    if d[i].is_negative() {
                        fi.conjugate()
                    } else {
                        fi
                    }
                };
                // Width 4: measured as on BLS12-381, widths 3 and 5 take 1.08 and 1.09 times its
                // time.
                gt_straus(
                    &[
                        (term(0), &l[0]),
                        (term(1), &l[1]),
                        (term(2), &l[2]),
                        (term(3), &l[3]),
                    ],
                    4,
                )
            }
            None => {
                let a = abs_limbs(&k.rem_euclid(base));
                let b = abs_limbs(&k.div_euclid(base));
                // Width 5, not measured: BN254 has its four digits above.
                gt_straus(&[(*f, &a), (f.frobenius(1), &b)], 5)
            }
        },
    }
}

/// Whether `f`, an element of `F_p¹²`, is in `GT`: in the cyclotomic subgroup, then
/// `f^p = f^x` (BLS12, Scott's test) or `f^p = f^(6x²)` (BN), when that decides it, else
/// `f^r = 1`.
pub(crate) fn gt_in_group<E: Engine<N>, const N: usize>(f: &Fp12<E, N>) -> bool {
    if !f.is_cyclotomic() {
        return false;
    }
    let g = E::groups();
    if g.gt_frobenius_test {
        let want = match E::curve().family {
            Family::Bls12 => super::pairing::pow_x::<E, N>(f),
            Family::Bn => f.cyclotomic_pow_digits(&g.g2_test),
        };
        f.frobenius(1) == want
    } else {
        f.cyclotomic_pow_digits(&E::tables().r_digits).is_one()
    }
}

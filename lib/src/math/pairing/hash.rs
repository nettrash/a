//! Hashing to BLS12-381's groups `G1` and `G2`, as RFC 9380 specifies it (section 8.8): the
//! suites `BLS12381G1_XMD:SHA-256_SSWU_RO_` and `_NU_`, and `BLS12381G2_XMD:SHA-256_SSWU_RO_` and
//! `_NU_`; and to BN254's `G1` in the same manner by the Shallue–van de Woestijne map, as
//! gnark-crypto does it, for which RFC 9380 has no suite.
//!
//! * [`bls12_381_g1`]: [`G1Suite`], hashing to `G1` by [`hash_to_field`](crate::math::ec::hash::hash_to_field) into `F_p`, the
//!   simplified SWU map of [`a::math::ec::hash`](crate::math::ec::hash) to the curve `E'`
//!   isogenous to BLS12-381 (`Z = 11`), the 11-isogeny of appendix E.2, and the product by
//!   `h_eff = 1 − x = 0xd201000000010001`.
//! * [`bls12_381_g2`]: [`G2Suite`], hashing to `G2` by [`hash_to_field_fp2`] into `F_p²` (`m = 2`),
//!   the simplified SWU map over `F_p²` ([`SimplifiedSwuFp2`], `Z = −(2 + I)`) to the curve
//!   isogenous to the twist, the 3-isogeny of appendix E.3 ([`IsogenyMapFp2`]), and the cofactor
//!   cleared by Budroni and Pintore's method of appendix G.3, [`G2::clear_cofactor`], which is the
//!   product by RFC 9380's `h_eff`.
//! * [`bn254_g1`]: a [`G1Suite`] for BN254, `BN254G1_XMD:SHA-256_SVDW_RO_` and `_NU_` as
//!   gnark-crypto names them: [`hash_to_field`](crate::math::ec::hash::hash_to_field) with SHA-256, the Shallue–van de Woestijne map of
//!   section 6.6.1 with `Z = 1` to `y² = x³ + 3` itself, and nothing to clear, as `E(F_p)` is `G1`.
//!
//! Each suite has, as the suites of `a::math::ec::hash` do, `hash_to_curve` (the `_RO_` suite),
//! `encode_to_curve` (the `_NU_` suite), and the steps they are made of: `hash_to_field`,
//! `map_to_curve` and `clear_cofactor`. `F_p²` is the tower's, `F_p[u]/(u² + 1)`, whose `u` is the
//! RFC's `I`, and `sgn0` on it is [`Fp2::sgn0`] (section 4.1).
//!
//! The suites compute on the pairing's own arithmetic. `G1`'s simplified SWU map is the
//! straight-line form of appendix F.2, one exponentiation and no inversion, its point kept as a
//! fraction that the 11-isogeny takes homogeneously; `hash_to_curve` adds the two points on `E′`
//! and applies the isogeny once to their sum, as section 6.6.3 allows. The map over `F_p²` and the
//! Shallue–van de Woestijne map keep the forms of section 6, an inversion and one or two square
//! roots for each element. BN254's `G2` has no hashing here.
//!
//! **Nothing here is constant-time**: the maps branch on whether an element is a square. They are
//! for public messages, such as those of a BLS signature's verification, not for hashing secrets
//! an attacker could time.
//!
//! # Examples
//!
//! ```
//! use a::math::pairing::hash;
//! use a::math::pairing::pairing;
//!
//! // RFC 9380, appendix J.9.1: BLS12381G1_XMD:SHA-256_SSWU_RO_ of "abc".
//! let g1 = hash::bls12_381_g1();
//! let p = g1.hash_to_curve(b"abc", b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_");
//! let (x, _) = p.coordinates().unwrap();
//! assert!(x.value().to_str_radix(16).starts_with("3567bc5ef9c690c2ab2ecdf6a96ef1c139cc0b2f"));
//! assert!(p.is_in_subgroup());
//!
//! // A message hashed to G2, as BLS signatures on G2 hash theirs, pairs like any point of G2.
//! let g2 = hash::bls12_381_g2();
//! let q = g2.hash_to_curve(b"a message", b"MY-APP-V01-CS01-with-BLS12381G2_XMD:SHA-256_SSWU_RO_");
//! assert!(q.is_in_subgroup());
//! let c = g1.curve();
//! assert_eq!(pairing(&c.g1_generator().double(), &q), pairing(&c.g1_generator(), &q.double()));
//! ```

use super::curve::PairingCurve;
use super::fast::group::Jac;
use super::fast::h2c::{Iso, SswuFp, SswuFp2, Svdw, add_a, from_okm};
use super::fast::{self, Bls, Bn, ops};
use super::g1::G1;
use super::g2::G2;
use crate::hash::{Digest, Sha256, expand_message_xmd};
use crate::math::ec::hash::{IsogenyMap, ShallueVanDeWoestijne, SimplifiedSwu, element_bytes, hex};
use crate::math::field::{Fp, Fp2, PrimeField, Tower};
use crate::math::numbers::{ByteOrder, Number};
use std::sync::OnceLock;

/// An `expand_message` of RFC 9380, section 5.3: `(msg, DST, len_in_bytes) -> uniform_bytes`.
type Expander = fn(&[u8], &[u8], usize) -> Vec<u8>;

/// `count` elements of `F_p²`, `2·l` bytes of `expand` each.
#[track_caller]
fn to_field_fp2<'t>(
    tower: &'t Tower<'t>,
    expand: Expander,
    l: usize,
    msg: &[u8],
    dst: &[u8],
    count: usize,
) -> Vec<Fp2<'t>> {
    let len = count
        .checked_mul(2)
        .and_then(|c| c.checked_mul(l))
        .unwrap_or_else(|| panic!("{count} elements of {} bytes are too many to expand", 2 * l));
    let bytes = expand(msg, dst, len);
    let f = tower.base();
    let e = |tv: &[u8]| f.element(&Number::new(tv.to_vec(), ByteOrder::BigEndian));
    bytes
        .chunks_exact(2 * l)
        .map(|tv| Fp2::new(tower, e(&tv[..l]), e(&tv[l..])))
        .collect()
}

/// `count` elements of the tower's `F_p²` hashed from `msg` with the domain separation tag `dst`,
/// by RFC 9380's `hash_to_field` (section 5.2) for `m = 2`, with `expand_message_xmd` on the hash
/// `H`, for the security level of `k` bits.
///
/// The element `u_i = e₀ + e₁·u` takes the `L = ceil((ceil(log2 p) + k) / 8)` bytes at
/// `L·2i` and at `L·(2i + 1)` of [`expand_message_xmd`]`::<H>(msg, dst, count · 2 · L)`, each
/// reduced modulo `p`. The tower's `u`, with `u² = β`, is the `I` of the RFC's basis `(1, I)` when
/// `β = −1`, as on BN254 and BLS12-381.
///
/// # Panics
///
/// Panics as [`expand_message_xmd`] does: for an empty `dst`, and when `2 · count · L` is above
/// 65535 bytes or 255 blocks of the hash (63 elements of BLS12-381's `F_p²` with SHA-256).
///
/// # Examples
///
/// ```
/// use a::hash::Sha256;
/// use a::math::field::Tower;
/// use a::math::pairing::hash::{self, hash_to_field_fp2};
///
/// // Appendix J.10.1: the first element of the message "abc".
/// let t = Tower::bls12_381();
/// let dst = b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_RO_";
/// let u = hash_to_field_fp2::<Sha256>(t, b"abc", dst, 2, 128);
/// assert!(u[0].c0().value().to_str_radix(16).starts_with("15f7c0aa8f6b296a"));
/// assert_eq!(u, hash::bls12_381_g2().hash_to_field(b"abc", dst, 2));
/// ```
#[track_caller]
#[must_use]
pub fn hash_to_field_fp2<'t, H: Digest>(
    tower: &'t Tower<'t>,
    msg: &[u8],
    dst: &[u8],
    count: usize,
    k: u32,
) -> Vec<Fp2<'t>> {
    to_field_fp2(
        tower,
        expand_message_xmd::<H>,
        element_bytes(tower.base(), k),
        msg,
        dst,
        count,
    )
}

/// The inverse of `x`, or 0 for 0: RFC 9380's `inv0`.
fn inv0<'t>(x: &Fp2<'t>) -> Fp2<'t> {
    x.inverse().unwrap_or_else(|| Fp2::zero(x.tower()))
}

/// `x` or `−x`, whichever has the sign `negative` by [`Fp2::sgn0`]; 0 for 0.
fn with_sign<'t>(x: Fp2<'t>, negative: bool) -> Fp2<'t> {
    if x.sgn0() == negative { x } else { -x }
}

/// The element `v` of `F_p` in the tower's `F_p²`.
fn small<'t>(tower: &'t Tower<'t>, v: i64) -> Fp2<'t> {
    Fp2::from_base(
        tower,
        tower
            .base()
            .element(&Number::new_from_i64(v, ByteOrder::LittleEndian)),
    )
}

/// `x³ + a·x + b`.
fn cubic<'t>(x: &Fp2<'t>, a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
    (x.square() + a) * x + b
}

/// The simplified Shallue–van de Woestijne–Ulas map of RFC 9380, section 6.6.2, from `F_p²` to
/// the curve `y² = x³ + A·x + B` over it, `A·B ≠ 0`: the map of
/// [`SimplifiedSwu`] on the elements of a [`Tower`]'s
/// `F_p²`, with the `sgn0` of section 4.1 for `m = 2` ([`Fp2::sgn0`]) and the square roots of
/// [`Fp2::sqrt`].
///
/// For `u`, `x1 = (−B/A)·(1 + 1/(Z²u⁴ + Z·u²))`, or `B/(Z·A)` where that denominator is 0; `x` is
/// `x1` if `g(x1)` is a square and `x2 = Z·u²·x1` otherwise; `y` is the square root of `g(x)` with
/// `sgn0(y) = sgn0(u)`. That is an inversion, one or two square roots of `F_p²` (each two of
/// `F_p`) and a few products.
///
/// # Examples
///
/// ```
/// use a::math::field::{Fp2, Tower};
/// use a::math::numbers::Number;
/// use a::math::pairing::hash::SimplifiedSwuFp2;
///
/// // The curve isogenous to BLS12-381's twist: A' = 240·I, B' = 1012·(1 + I), Z = −(2 + I).
/// let t = Tower::bls12_381();
/// let n = |x: i64| Number::from(x);
/// let a = Fp2::from_numbers(t, &n(0), &n(240));
/// let b = Fp2::from_numbers(t, &n(1012), &n(1012));
/// let map = SimplifiedSwuFp2::new(&a, &b, &Fp2::from_numbers(t, &n(-2), &n(-1)));
/// let (x, y) = map.map(&Fp2::from_numbers(t, &n(5), &n(7)));
/// assert_eq!(y.square(), &(&x.square() * &x) + &(&(&a * &x) + &b));
/// ```
#[derive(Clone, Debug)]
pub struct SimplifiedSwuFp2<'t> {
    /// `A`.
    a: Fp2<'t>,
    /// `B`.
    b: Fp2<'t>,
    /// `Z`.
    z: Fp2<'t>,
    /// `−B / A`.
    minus_b_over_a: Fp2<'t>,
    /// `B / (Z·A)`, `x1` in the exceptional cases.
    exceptional: Fp2<'t>,
}

impl<'t> SimplifiedSwuFp2<'t> {
    /// The map to `y² = x³ + a·x + b` with the constant `z`.
    ///
    /// # Panics
    ///
    /// Panics if the elements are of different towers, if `a` or `b` is 0, if the curve is
    /// singular, and if `z` does not meet the four criteria of section 6.6.2: `z` is not a square
    /// of `F_p²`, `z ≠ −1`, `x³ + a·x + b − z` is irreducible over `F_p²` (it has no root there,
    /// which [`new`](SimplifiedSwuFp2::new) tests by `gcd(x^(p²) − x, g − z) = 1`), and
    /// `g(b / (z·a))` is a square.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Tower};
    /// use a::math::numbers::Number;
    /// use a::math::pairing::hash::SimplifiedSwuFp2;
    ///
    /// let t = Tower::bls12_381();
    /// let n = |x: i64| Number::from(x);
    /// let (a, b) = (Fp2::from_numbers(t, &n(0), &n(240)), Fp2::from_numbers(t, &n(1012), &n(1012)));
    /// let z = Fp2::from_numbers(t, &n(-2), &n(-1));
    /// assert_eq!(SimplifiedSwuFp2::new(&a, &b, &z).z(), &z);
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(a: &Fp2<'t>, b: &Fp2<'t>, z: &Fp2<'t>) -> SimplifiedSwuFp2<'t> {
        assert!(
            !a.is_zero() && !b.is_zero(),
            "the simplified SWU map needs A ≠ 0 and B ≠ 0; map to an isogenous curve (RFC 9380, section 6.6.3)"
        );
        check_nonsingular(a, b);
        assert!(
            sswu_z_valid(a, b, z),
            "Z does not meet the criteria of RFC 9380, section 6.6.2"
        );
        let inv_a = a.inverse().unwrap_or_else(|| unreachable!("A ≠ 0"));
        let exceptional = &(b * &inv_a) / z;
        let minus_b_over_a = -(b * &inv_a);
        SimplifiedSwuFp2 {
            a: a.clone(),
            b: b.clone(),
            z: z.clone(),
            minus_b_over_a,
            exceptional,
        }
    }

    /// The `Z` of RFC 9380, appendix H.2, for the curve `y² = x³ + a·x + b` over `F_p²`: the first
    /// that meets the four criteria of section 6.6.2 of `c`, `−c` for `c = I, 1 + I, 2 + I, …`,
    /// as the RFC's script counts from the generator `I` of `F_p²`. It is `−(2 + I)` for the curve
    /// isogenous to BLS12-381's twist.
    ///
    /// # Panics
    ///
    /// Panics as [`new`](SimplifiedSwuFp2::new) does for `a` and `b`, and if no `c` up to
    /// `p − 1 + I` gives one.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Tower};
    /// use a::math::numbers::Number;
    /// use a::math::pairing::hash::SimplifiedSwuFp2;
    ///
    /// let t = Tower::bls12_381();
    /// let n = |x: i64| Number::from(x);
    /// let (a, b) = (Fp2::from_numbers(t, &n(0), &n(240)), Fp2::from_numbers(t, &n(1012), &n(1012)));
    /// assert_eq!(SimplifiedSwuFp2::find_z(&a, &b), Fp2::from_numbers(t, &n(-2), &n(-1)));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn find_z(a: &Fp2<'t>, b: &Fp2<'t>) -> Fp2<'t> {
        let t = a.tower();
        assert!(
            !a.is_zero() && !b.is_zero(),
            "the simplified SWU map needs A ≠ 0 and B ≠ 0"
        );
        check_nonsingular(a, b);
        let f = t.base();
        let mut c0 = Number::default();
        while c0 < *f.modulus() {
            let c = Fp2::new(t, f.element(&c0), f.one());
            for z in [c.clone(), -c] {
                if sswu_z_valid(a, b, &z) {
                    return z;
                }
            }
            c0 += 1u8;
        }
        panic!("no element c + I meets the criteria of RFC 9380, section 6.6.2")
    }

    /// `A`.
    #[must_use]
    pub fn a(&self) -> &Fp2<'t> {
        &self.a
    }

    /// `B`.
    #[must_use]
    pub fn b(&self) -> &Fp2<'t> {
        &self.b
    }

    /// `Z`.
    #[must_use]
    pub fn z(&self) -> &Fp2<'t> {
        &self.z
    }

    /// The point `(x, y)` that `u` maps to, as section 6.6.2 computes it.
    ///
    /// # Panics
    ///
    /// Panics if `u` is of another tower.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Tower};
    /// use a::math::numbers::Number;
    /// use a::math::pairing::hash::SimplifiedSwuFp2;
    ///
    /// let t = Tower::bls12_381();
    /// let n = |x: i64| Number::from(x);
    /// let (a, b) = (Fp2::from_numbers(t, &n(0), &n(240)), Fp2::from_numbers(t, &n(1012), &n(1012)));
    /// let map = SimplifiedSwuFp2::new(&a, &b, &Fp2::from_numbers(t, &n(-2), &n(-1)));
    /// // u and −u give the same x, and y the sign of u.
    /// let u = Fp2::from_numbers(t, &n(3), &n(-4));
    /// let (x, y) = map.map(&u);
    /// assert_eq!(map.map(&-&u), (x, -&y));
    /// assert_eq!(y.sgn0(), u.sgn0());
    /// ```
    #[track_caller]
    #[must_use]
    pub fn map(&self, u: &Fp2<'t>) -> (Fp2<'t>, Fp2<'t>) {
        let t = self.a.tower();
        let zu2 = &self.z * &u.square();
        let tv1 = inv0(&(&zu2.square() + &zu2));
        let x1 = if tv1.is_zero() {
            self.exceptional.clone()
        } else {
            &self.minus_b_over_a * &(tv1 + Fp2::one(t))
        };
        let (x, y) = match cubic(&x1, &self.a, &self.b).sqrt() {
            Some(y) => (x1, y),
            None => {
                let x2 = zu2 * x1;
                let y = cubic(&x2, &self.a, &self.b).sqrt().unwrap_or_else(|| {
                    unreachable!("g(x2) = Z³u⁶·g(x1) is a square when g(x1) is not")
                });
                (x2, y)
            }
        };
        (x, with_sign(y, u.sgn0()))
    }
}

/// Panics if `y² = x³ + a·x + b` is singular: `4·a³ + 27·b² = 0` (and for elements of different
/// towers).
#[track_caller]
fn check_nonsingular<'t>(a: &Fp2<'t>, b: &Fp2<'t>) {
    let t = a.tower();
    let disc = &(&small(t, 4) * &(&a.square() * a)) + &(&small(t, 27) * &b.square());
    assert!(!disc.is_zero(), "the curve is singular");
}

/// Whether `z` meets the criteria of RFC 9380, section 6.6.2, for `y² = x³ + a·x + b` over `F_p²`.
fn sswu_z_valid<'t>(a: &Fp2<'t>, b: &Fp2<'t>, z: &Fp2<'t>) -> bool {
    let t = a.tower();
    if z.is_square() || *z == -Fp2::one(t) {
        return false;
    }
    if !cubic(&(b / &(z * a)), a, b).is_square() {
        return false;
    }
    // g(x) − Z, a cubic, is irreducible: it has no root in F_p², gcd(x^(p²) − x, g − Z) = 1.
    !has_root(&[b - z, a.clone(), Fp2::zero(t)])
}

/// Whether the monic cubic `x³ + c[2]·x² + c[1]·x + c[0]` has a root in `F_p²`: whether
/// `gcd(x^(p²) − x, f)` is not constant, with `x^(p²) mod f` by squarings and products modulo
/// `f`.
fn has_root<'t>(c: &[Fp2<'t>; 3]) -> bool {
    let t = c[0].tower();
    let zero = Fp2::zero(t);
    // The product of two remainders modulo f, of degree below 3.
    let mul = |x: &[Fp2<'t>; 3], y: &[Fp2<'t>; 3]| -> [Fp2<'t>; 3] {
        let mut d = vec![zero.clone(); 5];
        for i in 0..3 {
            for j in 0..3 {
                d[i + j] = &d[i + j] + &(&x[i] * &y[j]);
            }
        }
        // x³ = −(c₂x² + c₁x + c₀).
        for k in (3..5).rev() {
            let lead = d[k].clone();
            for i in 0..3 {
                d[k - 3 + i] = &d[k - 3 + i] - &(&lead * &c[i]);
            }
        }
        [d[0].clone(), d[1].clone(), d[2].clone()]
    };
    let p = t.base().modulus();
    let q = p * p;
    let x = [zero.clone(), Fp2::one(t), zero.clone()];
    let mut r = [Fp2::one(t), zero.clone(), zero.clone()];
    for i in (0..q.bits()).rev() {
        r = mul(&r, &r);
        if q.bit(i) {
            r = mul(&r, &x);
        }
    }
    r[1] = &r[1] - &Fp2::one(t);
    let f = vec![c[0].clone(), c[1].clone(), c[2].clone(), Fp2::one(t)];
    poly_gcd_degree(f, r.to_vec()) > 0
}

/// The degree of `gcd(a, b)` over `F_p²`, by Euclid's algorithm; `a` is not zero.
fn poly_gcd_degree<'t>(mut a: Vec<Fp2<'t>>, mut b: Vec<Fp2<'t>>) -> usize {
    let trim = |v: &mut Vec<Fp2<'t>>| {
        while v.last().is_some_and(Fp2::is_zero) {
            v.pop();
        }
    };
    trim(&mut a);
    trim(&mut b);
    while !b.is_empty() {
        // a mod b.
        let inv = b[b.len() - 1]
            .inverse()
            .unwrap_or_else(|| unreachable!("the leading coefficient is not 0"));
        while a.len() >= b.len() {
            let q = &a[a.len() - 1] * &inv;
            let shift = a.len() - b.len();
            for (i, bi) in b.iter().enumerate() {
                a[shift + i] = &a[shift + i] - &(&q * bi);
            }
            trim(&mut a);
        }
        core::mem::swap(&mut a, &mut b);
    }
    a.len() - 1
}

/// A rational map `(x, y) → (x_num(x) / x_den(x), y · y_num(x) / y_den(x))` between curves over
/// `F_p²`, the shape in which RFC 9380, appendix E.3, gives the 3-isogeny onto BLS12-381's twist:
/// [`IsogenyMap`] on the elements of a [`Tower`]'s `F_p²`.
///
/// # Examples
///
/// ```
/// use a::math::pairing::hash;
///
/// let iso = hash::bls12_381_g2().isogeny();
/// assert_eq!(iso.degree(), 3);
/// ```
#[derive(Clone, Debug)]
pub struct IsogenyMapFp2<'t> {
    /// The coefficients of `x_num`, the constant first.
    x_num: Vec<Fp2<'t>>,
    /// The coefficients of `x_den`, the constant first.
    x_den: Vec<Fp2<'t>>,
    /// The coefficients of `y_num`, the constant first.
    y_num: Vec<Fp2<'t>>,
    /// The coefficients of `y_den`, the constant first.
    y_den: Vec<Fp2<'t>>,
}

impl<'t> IsogenyMapFp2<'t> {
    /// The map with the four polynomials given by their coefficients, the constant first, with the
    /// leading coefficients 1 of `x_den` and `y_den` written in.
    ///
    /// # Panics
    ///
    /// Panics if a polynomial has no coefficients, if its leading one is 0, or if they are of
    /// different towers.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::field::{Fp2, Tower};
    /// use a::math::numbers::Number;
    /// use a::math::pairing::hash::IsogenyMapFp2;
    ///
    /// // The identity map, x / 1 and y · 1 / 1.
    /// let t = Tower::bls12_381();
    /// let (zero, one) = (Fp2::zero(t), Fp2::one(t));
    /// let id = IsogenyMapFp2::new(&[zero, one.clone()], &[one.clone()], &[one.clone()], &[one]);
    /// let x = Fp2::from_numbers(t, &Number::from(4u8), &Number::from(9u8));
    /// assert_eq!(id.map(&x, &x.square()), Some((x.clone(), x.square())));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn new(
        x_num: &[Fp2<'t>],
        x_den: &[Fp2<'t>],
        y_num: &[Fp2<'t>],
        y_den: &[Fp2<'t>],
    ) -> IsogenyMapFp2<'t> {
        let all = [x_num, x_den, y_num, y_den];
        let first = all
            .iter()
            .find_map(|c| c.first())
            .expect("a polynomial of the map has no coefficients");
        for c in all {
            let lead = c
                .last()
                .expect("a polynomial of the map has no coefficients");
            assert!(!lead.is_zero(), "a leading coefficient of the map is 0");
            for e in c {
                // A sum checks the towers.
                let _ = e + first;
            }
        }
        IsogenyMapFp2 {
            x_num: x_num.to_vec(),
            x_den: x_den.to_vec(),
            y_num: y_num.to_vec(),
            y_den: y_den.to_vec(),
        }
    }

    /// The degree of the map, that of `x_num`: the degree of the isogeny.
    #[must_use]
    pub fn degree(&self) -> usize {
        self.x_num.len() - 1
    }

    /// The coefficients of `x_num`, the constant first.
    #[must_use]
    pub fn x_num(&self) -> &[Fp2<'t>] {
        &self.x_num
    }

    /// The coefficients of `x_den`, the constant first.
    #[must_use]
    pub fn x_den(&self) -> &[Fp2<'t>] {
        &self.x_den
    }

    /// The coefficients of `y_num`, the constant first.
    #[must_use]
    pub fn y_num(&self) -> &[Fp2<'t>] {
        &self.y_num
    }

    /// The coefficients of `y_den`, the constant first.
    #[must_use]
    pub fn y_den(&self) -> &[Fp2<'t>] {
        &self.y_den
    }

    /// The image of the point `(x, y)`, or `None`, the point at infinity, where `x_den` or `y_den`
    /// is 0, as section 6.6.3 requires for the exceptional cases.
    ///
    /// # Panics
    ///
    /// Panics if `x` or `y` is of another tower.
    #[track_caller]
    #[must_use]
    pub fn map(&self, x: &Fp2<'t>, y: &Fp2<'t>) -> Option<(Fp2<'t>, Fp2<'t>)> {
        let horner = |c: &[Fp2<'t>]| {
            let mut acc = c[c.len() - 1].clone();
            for k in c[..c.len() - 1].iter().rev() {
                acc = &(&acc * x) + k;
            }
            acc
        };
        let (xd, yd) = (horner(&self.x_den), horner(&self.y_den));
        // One inversion for both denominators; it is None exactly when one of them is 0.
        let inv = (&xd * &yd).inverse()?;
        let xn = horner(&self.x_num);
        let yn = horner(&self.y_num);
        Some((&(&xn * &yd) * &inv, &(&(y * &yn) * &xd) * &inv))
    }
}

/// The suites of a `G1`: [`hash_to_curve`](G1Suite::hash_to_curve), the `_RO_` suite, and
/// [`encode_to_curve`](G1Suite::encode_to_curve), the `_NU_` suite, with their steps.
///
/// | Function | Suites | `k` | `L` | Map | `h_eff` |
/// | --- | --- | ---: | ---: | --- | --- |
/// | [`bls12_381_g1`] | `BLS12381G1_XMD:SHA-256_SSWU_RO_`, `_NU_` (RFC 9380, section 8.8.1) | 128 | 64 | simplified SWU to `E': y² = x³ + A'·x + B'`, `Z = 11`, then the 11-isogeny of appendix E.2 | `0xd201000000010001 = 1 − x` |
/// | [`bn254_g1`] | `BN254G1_XMD:SHA-256_SVDW_RO_`, `_NU_` (gnark-crypto's) | 128 | 48 | Shallue–van de Woestijne (section 6.6.1), `Z = 1` | 1 |
///
/// # Examples
///
/// ```
/// use a::math::numbers::Number;
/// use a::math::pairing::hash;
///
/// let s = hash::bls12_381_g1();
/// assert_eq!(s.nu_id(), "BLS12381G1_XMD:SHA-256_SSWU_NU_");
/// assert_eq!((s.k(), s.l()), (128, 64));
/// assert_eq!(s.h_eff(), &Number::from(0xd201_0000_0001_0001_u64));
/// let u = s.hash_to_field(b"msg", b"MY-DST", 2);
/// let p = s.clear_cofactor(&(s.map_to_curve(&u[0]) + s.map_to_curve(&u[1])));
/// assert_eq!(p, s.hash_to_curve(b"msg", b"MY-DST"));
/// ```
pub struct G1Suite {
    /// The IDs of the `_RO_` and `_NU_` suites.
    ids: [&'static str; 2],
    /// The curve.
    curve: &'static PairingCurve,
    /// The map.
    map: G1Map,
    /// `L`.
    l: usize,
    /// `h_eff`.
    h_eff: Number,
    /// The map on the pairing's own arithmetic.
    fast: G1Fast,
}

/// The maps of the two suites on the engines.
enum G1Fast {
    Bls(SswuFp<Bls, 6>, Iso<fast::Fp<Bls, 6>>),
    Bn(Svdw<Bn, 4>),
}

/// How a suite of `G1` maps an element to its curve.
#[derive(Debug)]
enum G1Map {
    /// The simplified SWU map to an isogenous curve, then the isogeny (BLS12-381).
    SswuIsogeny(SimplifiedSwu<'static>, IsogenyMap<'static>),
    /// The Shallue–van de Woestijne map to the curve itself (BN254).
    Svdw(ShallueVanDeWoestijne<'static>),
}

impl core::fmt::Debug for G1Suite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("G1Suite")
            .field("ro", &self.ro_id())
            .field("nu", &self.nu_id())
            .finish_non_exhaustive()
    }
}

impl G1Suite {
    /// `count` elements' bytes, `L` each, by `expand_message_xmd` with SHA-256.
    #[track_caller]
    fn expand(&self, msg: &[u8], dst: &[u8], count: usize) -> Vec<u8> {
        let len = count.checked_mul(self.l).unwrap_or_else(|| {
            panic!(
                "{count} elements of {} bytes are too many to expand",
                self.l
            )
        });
        expand_message_xmd::<Sha256>(msg, dst, len)
    }
}

impl G1Suite {
    /// The ID of the suite of [`hash_to_curve`](G1Suite::hash_to_curve):
    /// `BLS12381G1_XMD:SHA-256_SSWU_RO_`, or `BN254G1_XMD:SHA-256_SVDW_RO_`.
    #[must_use]
    pub fn ro_id(&self) -> &'static str {
        self.ids[0]
    }

    /// The ID of the suite of [`encode_to_curve`](G1Suite::encode_to_curve):
    /// `BLS12381G1_XMD:SHA-256_SSWU_NU_`, or `BN254G1_XMD:SHA-256_SVDW_NU_`.
    #[must_use]
    pub fn nu_id(&self) -> &'static str {
        self.ids[1]
    }

    /// The curve, [`PairingCurve::bls12_381`] or [`PairingCurve::bn254`].
    #[must_use]
    pub fn curve(&self) -> &'static PairingCurve {
        self.curve
    }

    /// The field `F_p` of the coordinates, in which [`hash_to_field`](G1Suite::hash_to_field)
    /// gives its elements.
    #[must_use]
    pub fn field(&self) -> &'static PrimeField {
        self.curve.base_field()
    }

    /// The security level `k`, 128 bits.
    #[must_use]
    pub fn k(&self) -> u32 {
        128
    }

    /// `L = ceil((ceil(log2 p) + 128) / 8)`, the bytes of each element: 64 on BLS12-381, 48 on
    /// BN254.
    #[must_use]
    pub fn l(&self) -> usize {
        self.l
    }

    /// `h_eff`: `0xd201000000010001 = 1 − x` on BLS12-381, 1 on BN254 (whose `E(F_p)` is `G1`).
    #[must_use]
    pub fn h_eff(&self) -> &Number {
        &self.h_eff
    }

    /// The map's `Z`: 11 on BLS12-381, 1 on BN254.
    #[must_use]
    pub fn z(&self) -> &Fp<'static> {
        match &self.map {
            G1Map::SswuIsogeny(m, _) => m.z(),
            G1Map::Svdw(m) => m.z(),
        }
    }

    /// The simplified SWU map to the curve `E'` of section 8.8.1, on BLS12-381.
    #[must_use]
    pub fn sswu(&self) -> Option<&SimplifiedSwu<'static>> {
        match &self.map {
            G1Map::SswuIsogeny(m, _) => Some(m),
            G1Map::Svdw(_) => None,
        }
    }

    /// The 11-isogeny from `E'` onto BLS12-381, of appendix E.2, on BLS12-381.
    #[must_use]
    pub fn isogeny(&self) -> Option<&IsogenyMap<'static>> {
        match &self.map {
            G1Map::SswuIsogeny(_, iso) => Some(iso),
            G1Map::Svdw(_) => None,
        }
    }

    /// The Shallue–van de Woestijne map to the curve, on BN254.
    #[must_use]
    pub fn svdw(&self) -> Option<&ShallueVanDeWoestijne<'static>> {
        match &self.map {
            G1Map::Svdw(m) => Some(m),
            G1Map::SswuIsogeny(..) => None,
        }
    }

    /// `count` elements of `F_p` from `msg` and the domain separation tag `dst`, by
    /// [`hash_to_field`](crate::math::ec::hash::hash_to_field) with SHA-256 and `k = 128`.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty or `count` is above 127 on BLS12-381, 170 on BN254 (255 blocks of
    /// SHA-256).
    #[track_caller]
    #[must_use]
    pub fn hash_to_field(&self, msg: &[u8], dst: &[u8], count: usize) -> Vec<Fp<'static>> {
        let bytes = self.expand(msg, dst, count);
        bytes
            .chunks_exact(self.l)
            .map(|tv| match self.fast {
                G1Fast::Bls(..) => from_okm::<Bls, 6>(tv).to_public(),
                G1Fast::Bn(..) => from_okm::<Bn, 4>(tv).to_public(),
            })
            .collect()
    }

    /// The point of the curve that `u` maps to, before the cofactor is cleared: on BLS12-381 by
    /// the simplified SWU map to `E'` and the 11-isogeny, the identity where the isogeny is not
    /// defined; on BN254 by the Shallue–van de Woestijne map.
    ///
    /// # Panics
    ///
    /// Panics if `u` is not an element of the curve's `F_p`.
    #[track_caller]
    #[must_use]
    pub fn map_to_curve<'c>(&self, u: &Fp<'c>) -> G1<'c> {
        let curve: &'c PairingCurve = self.curve;
        assert!(
            u.field().modulus() == curve.base_field().modulus(),
            "elements of different fields"
        );
        match &self.fast {
            G1Fast::Bls(m, iso) => {
                let (xn, xd, y) = m.map(&fast::Fp::from_public(u));
                G1::from_jac::<Bls, 6>(curve, &iso.map(&xn, &xd, &y), false)
            }
            G1Fast::Bn(m) => {
                let (x, y) = m.map(&fast::Fp::from_public(u));
                G1::from_jac::<Bn, 4>(curve, &Jac::from_affine(x, y), true)
            }
        }
    }

    /// `h_eff · p`, by [`G1::clear_cofactor`]: a point of `G1`.
    ///
    /// # Panics
    ///
    /// Panics if `p` is not a point of the suite's curve.
    #[track_caller]
    #[must_use]
    pub fn clear_cofactor<'c>(&self, p: &G1<'c>) -> G1<'c> {
        self.curve.check(p.curve());
        p.clear_cofactor()
    }

    /// `hash_to_curve`, the suite `BLS12381G1_XMD:SHA-256_SSWU_RO_`: the cofactor cleared from the
    /// sum of the points that the two elements of
    /// [`hash_to_field`](G1Suite::hash_to_field)`(msg, dst, 2)` map to.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::hash;
    ///
    /// // Appendix J.9.1, the empty message.
    /// let s = hash::bls12_381_g1();
    /// let p = s.hash_to_curve(b"", b"QUUX-V01-CS02-with-BLS12381G1_XMD:SHA-256_SSWU_RO_");
    /// let (x, _) = p.coordinates().unwrap();
    /// assert!(x.value().to_str_radix(16).starts_with("52926add2207b76ca4fa57a8734416c8dc95e2"));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn hash_to_curve(&self, msg: &[u8], dst: &[u8]) -> G1<'static> {
        let bytes = self.expand(msg, dst, 2);
        let (t0, t1) = bytes.split_at(self.l);
        let curve = self.curve;
        match &self.fast {
            G1Fast::Bls(m, iso) => {
                // Both points on E′, their sum, the isogeny once, the cofactor.
                let on_e = |t: &[u8]| {
                    let (xn, xd, y) = m.map(&from_okm::<Bls, 6>(t));
                    Jac {
                        x: xn.mul(&xd),
                        y: y.mul(&xd.square().mul(&xd)),
                        z: xd,
                    }
                };
                let sum = add_a(&on_e(t0), &on_e(t1), &m.a);
                let p = iso.map_jac(&sum);
                G1::from_jac::<Bls, 6>(curve, &ops::g1_clear(&p), true)
            }
            G1Fast::Bn(m) => {
                let pt = |t: &[u8]| {
                    let (x, y) = m.map(&from_okm::<Bn, 4>(t));
                    Jac::from_affine(x, y)
                };
                G1::from_jac::<Bn, 4>(curve, &pt(t0).add(&pt(t1)), true)
            }
        }
    }

    /// `encode_to_curve`, the suite `BLS12381G1_XMD:SHA-256_SSWU_NU_`: the cofactor cleared from
    /// the point that the one element of [`hash_to_field`](G1Suite::hash_to_field)`(msg, dst, 1)`
    /// maps to; not uniform.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty.
    #[track_caller]
    #[must_use]
    pub fn encode_to_curve(&self, msg: &[u8], dst: &[u8]) -> G1<'static> {
        let bytes = self.expand(msg, dst, 1);
        let curve = self.curve;
        match &self.fast {
            G1Fast::Bls(m, iso) => {
                let (xn, xd, y) = m.map(&from_okm::<Bls, 6>(&bytes));
                G1::from_jac::<Bls, 6>(curve, &ops::g1_clear(&iso.map(&xn, &xd, &y)), true)
            }
            G1Fast::Bn(m) => {
                let (x, y) = m.map(&from_okm::<Bn, 4>(&bytes));
                G1::from_jac::<Bn, 4>(curve, &Jac::from_affine(x, y), true)
            }
        }
    }
}

/// The suites of BLS12-381's `G2`, RFC 9380 section 8.8.2: `BLS12381G2_XMD:SHA-256_SSWU_RO_`
/// ([`hash_to_curve`](G2Suite::hash_to_curve)) and `BLS12381G2_XMD:SHA-256_SSWU_NU_`
/// ([`encode_to_curve`](G2Suite::encode_to_curve)), from [`bls12_381_g2`].
///
/// | `m` | `k` | `L` | Map | `h_eff` |
/// | ---: | ---: | ---: | --- | --- |
/// | 2 | 128 | 64 | simplified SWU over `F_p²` to `E': y² = x³ + 240·I·x + 1012·(1 + I)`, `Z = −(2 + I)`, then the 3-isogeny of appendix E.3 | `0xbc69f08f…aaa95551`, by appendix G.3 |
///
/// # Examples
///
/// ```
/// use a::math::pairing::hash;
///
/// let s = hash::bls12_381_g2();
/// assert_eq!(s.ro_id(), "BLS12381G2_XMD:SHA-256_SSWU_RO_");
/// let q = s.encode_to_curve(b"msg", b"MY-DST");
/// assert!(q.is_in_subgroup());
/// // The cofactor clearing of appendix G.3 is the product by h_eff.
/// let u = s.hash_to_field(b"msg", b"MY-DST", 1);
/// let q0 = s.map_to_curve(&u[0]);
/// assert_eq!(s.clear_cofactor(&q0), &q0 * s.h_eff());
/// assert_eq!(s.clear_cofactor(&q0), q);
/// ```
pub struct G2Suite {
    /// BLS12-381.
    curve: &'static PairingCurve,
    /// The simplified SWU map to `E'` over `F_p²`.
    map: SimplifiedSwuFp2<'static>,
    /// The 3-isogeny from `E'` onto the twist.
    iso: IsogenyMapFp2<'static>,
    /// `L`.
    l: usize,
    /// `h_eff`.
    h_eff: Number,
    /// The map and the isogeny on the pairing's own arithmetic.
    fast_map: SswuFp2<Bls, 6>,
    fast_iso: Iso<fast::Fp2<Bls, 6>>,
}

impl core::fmt::Debug for G2Suite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("G2Suite")
            .field("ro", &self.ro_id())
            .field("nu", &self.nu_id())
            .finish_non_exhaustive()
    }
}

impl G2Suite {
    /// `count` elements of `F_p²` in the engine's arithmetic, as `hash_to_field` gives them.
    #[track_caller]
    fn fast_field(&self, msg: &[u8], dst: &[u8], count: usize) -> Vec<fast::Fp2<Bls, 6>> {
        let l = self.l;
        let len = count
            .checked_mul(2)
            .and_then(|c| c.checked_mul(l))
            .unwrap_or_else(|| {
                panic!("{count} elements of {} bytes are too many to expand", 2 * l)
            });
        let bytes = expand_message_xmd::<Sha256>(msg, dst, len);
        bytes
            .chunks_exact(2 * l)
            .map(|tv| fast::Fp2::new(from_okm(&tv[..l]), from_okm(&tv[l..])))
            .collect()
    }
}

impl G2Suite {
    /// `BLS12381G2_XMD:SHA-256_SSWU_RO_`, the suite of [`hash_to_curve`](G2Suite::hash_to_curve).
    #[must_use]
    pub fn ro_id(&self) -> &'static str {
        "BLS12381G2_XMD:SHA-256_SSWU_RO_"
    }

    /// `BLS12381G2_XMD:SHA-256_SSWU_NU_`, the suite of
    /// [`encode_to_curve`](G2Suite::encode_to_curve).
    #[must_use]
    pub fn nu_id(&self) -> &'static str {
        "BLS12381G2_XMD:SHA-256_SSWU_NU_"
    }

    /// The curve, [`PairingCurve::bls12_381`].
    #[must_use]
    pub fn curve(&self) -> &'static PairingCurve {
        self.curve
    }

    /// The tower, whose `F_p²` holds the coordinates of `G2` and the elements of
    /// [`hash_to_field`](G2Suite::hash_to_field).
    #[must_use]
    pub fn tower(&self) -> &'static Tower<'static> {
        self.curve.tower()
    }

    /// The security level `k`, 128 bits.
    #[must_use]
    pub fn k(&self) -> u32 {
        128
    }

    /// `L = 64`, the bytes of each of the two coordinates of an element.
    #[must_use]
    pub fn l(&self) -> usize {
        self.l
    }

    /// RFC 9380's `h_eff` for `G2`, `0xbc69f08f…aaa95551` of 636 bits, of which
    /// [`clear_cofactor`](G2Suite::clear_cofactor) computes the product without multiplying by
    /// it.
    #[must_use]
    pub fn h_eff(&self) -> &Number {
        &self.h_eff
    }

    /// `Z = −(2 + I)`.
    #[must_use]
    pub fn z(&self) -> &Fp2<'static> {
        self.map.z()
    }

    /// The simplified SWU map to the curve `E'` of section 8.8.2.
    #[must_use]
    pub fn sswu(&self) -> &SimplifiedSwuFp2<'static> {
        &self.map
    }

    /// The 3-isogeny from `E'` onto BLS12-381's twist, of appendix E.3.
    #[must_use]
    pub fn isogeny(&self) -> &IsogenyMapFp2<'static> {
        &self.iso
    }

    /// `count` elements of `F_p²` from `msg` and the domain separation tag `dst`, by
    /// [`hash_to_field_fp2`] with SHA-256 and `k = 128`.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty or `count` is above 63 (255 blocks of SHA-256).
    #[track_caller]
    #[must_use]
    pub fn hash_to_field(&self, msg: &[u8], dst: &[u8], count: usize) -> Vec<Fp2<'static>> {
        self.fast_field(msg, dst, count)
            .iter()
            .map(|u| u.to_public())
            .collect()
    }

    /// The point of the twist that `u` maps to, by the simplified SWU map to `E'` and the
    /// 3-isogeny, before the cofactor is cleared; the identity where the isogeny is not defined.
    ///
    /// # Panics
    ///
    /// Panics if `u` is not an element of BLS12-381's `F_p²`.
    #[track_caller]
    #[must_use]
    pub fn map_to_curve<'c>(&self, u: &Fp2<'c>) -> G2<'c> {
        let curve: &'c PairingCurve = self.curve;
        let t = curve.tower();
        assert!(
            u.tower().base().modulus() == t.base().modulus() && u.tower().xi() == t.xi(),
            "elements of different towers"
        );
        let (x, y) = self.fast_map.map(&fast::Fp2::from_public(u));
        let q = self.fast_iso.map(&x, &fast::Fp2::one(), &y);
        G2::from_jac::<Bls, 6>(curve, &q, false)
    }

    /// `h_eff · q`, a point of `G2`, by [`G2::clear_cofactor`]: Budroni and Pintore's
    /// `[x² − x − 1]·q + [x − 1]·ψ(q) + ψ²(2q)`, the steps of appendix G.3.
    ///
    /// # Panics
    ///
    /// Panics if `q` is not a point of BLS12-381's twist.
    #[track_caller]
    #[must_use]
    pub fn clear_cofactor<'c>(&self, q: &G2<'c>) -> G2<'c> {
        self.curve.check(q.curve());
        q.clear_cofactor()
    }

    /// `hash_to_curve`, the suite `BLS12381G2_XMD:SHA-256_SSWU_RO_`: the cofactor cleared from the
    /// sum of the points that the two elements of
    /// [`hash_to_field`](G2Suite::hash_to_field)`(msg, dst, 2)` map to.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::pairing::hash;
    ///
    /// // Appendix J.10.1, the empty message.
    /// let s = hash::bls12_381_g2();
    /// let q = s.hash_to_curve(b"", b"QUUX-V01-CS02-with-BLS12381G2_XMD:SHA-256_SSWU_RO_");
    /// let (x, _) = q.coordinates().unwrap();
    /// assert!(x.c0().value().to_str_radix(16).starts_with("141ebfbdca40eb85b87142e130ab689c"));
    /// ```
    #[track_caller]
    #[must_use]
    pub fn hash_to_curve(&self, msg: &[u8], dst: &[u8]) -> G2<'static> {
        let u = self.fast_field(msg, dst, 2);
        let on_e = |u: &fast::Fp2<Bls, 6>| {
            let (x, y) = self.fast_map.map(u);
            Jac::from_affine(x, y)
        };
        let sum = add_a(&on_e(&u[0]), &on_e(&u[1]), &self.fast_map.a);
        let q = self.fast_iso.map_jac(&sum);
        G2::from_jac::<Bls, 6>(self.curve, &ops::g2_clear(&q), true)
    }

    /// `encode_to_curve`, the suite `BLS12381G2_XMD:SHA-256_SSWU_NU_`: the cofactor cleared from
    /// the point that the one element of [`hash_to_field`](G2Suite::hash_to_field)`(msg, dst, 1)`
    /// maps to; not uniform.
    ///
    /// # Panics
    ///
    /// Panics if `dst` is empty.
    #[track_caller]
    #[must_use]
    pub fn encode_to_curve(&self, msg: &[u8], dst: &[u8]) -> G2<'static> {
        let u = self.fast_field(msg, dst, 1);
        let (x, y) = self.fast_map.map(&u[0]);
        let q = self.fast_iso.map(&x, &fast::Fp2::one(), &y);
        G2::from_jac::<Bls, 6>(self.curve, &ops::g2_clear(&q), true)
    }
}

/// The suites of BLS12-381's `G1` (RFC 9380, section 8.8.1), built on the first call.
///
/// # Examples
///
/// ```
/// use a::math::pairing::hash;
///
/// let s = hash::bls12_381_g1();
/// assert_eq!(s.ro_id(), "BLS12381G1_XMD:SHA-256_SSWU_RO_");
/// assert_eq!(s.z(), &s.field().from_u64(11));
/// assert_eq!(s.isogeny().map(|i| i.degree()), Some(11));
/// ```
#[must_use]
pub fn bls12_381_g1() -> &'static G1Suite {
    static SUITE: OnceLock<G1Suite> = OnceLock::new();
    SUITE.get_or_init(|| {
        let curve = PairingCurve::bls12_381();
        let f = curve.base_field();
        let h = |s: &str| hex(f, s);
        let map = SimplifiedSwu::new(&h(G1_ISO_A), &h(G1_ISO_B), &f.from_u64(11));
        let all = |c: &[&str]| -> Vec<Fp<'static>> { c.iter().map(|s| hex(f, s)).collect() };
        let monic = |c: &[&str]| -> Vec<Fp<'static>> {
            c.iter().map(|s| hex(f, s)).chain([f.one()]).collect()
        };
        let iso = IsogenyMap::new(
            &all(&G1_ISO_X_NUM),
            &monic(&G1_ISO_X_DEN),
            &all(&G1_ISO_Y_NUM),
            &monic(&G1_ISO_Y_DEN),
        );
        let fp = |x: &Fp<'static>| fast::Fp::<Bls, 6>::from_public(x);
        let fv = |c: &[Fp<'static>]| c.iter().map(fp).collect::<Vec<_>>();
        let fast = G1Fast::Bls(
            SswuFp::new(fp(map.a()), fp(map.b()), fp(map.z())),
            Iso {
                x_num: fv(iso.x_num()),
                x_den: fv(iso.x_den()),
                y_num: fv(iso.y_num()),
                y_den: fv(iso.y_den()),
            },
        );
        G1Suite {
            ids: [
                "BLS12381G1_XMD:SHA-256_SSWU_RO_",
                "BLS12381G1_XMD:SHA-256_SSWU_NU_",
            ],
            curve,
            map: G1Map::SswuIsogeny(map, iso),
            l: element_bytes(f, 128),
            h_eff: Number::from(0xd201_0000_0001_0001_u64),
            fast,
        }
    })
}

/// A hashing of BN254's `G1` in the manner of RFC 9380, which defines no suite for BN254:
/// `BN254G1_XMD:SHA-256_SVDW_RO_` and `BN254G1_XMD:SHA-256_SVDW_NU_`, as gnark-crypto names and
/// defines them, built on the first call. They take [`hash_to_field`](crate::math::ec::hash::hash_to_field) with SHA-256, `k = 128` and
/// `L = 48`, the Shallue–van de Woestijne map of section 6.6.1 to `y² = x³ + 3` with the `Z = 1`
/// of appendix H.1, and `h_eff = 1`, as `E(F_p)` is `G1`.
///
/// Its outputs are those of gnark-crypto's `HashToG1` and `EncodeToG1` on its published vectors;
/// they are not interoperable with any other hashing to BN254, such as the "try and increment"
/// of some Ethereum contracts.
///
/// # Examples
///
/// ```
/// use a::math::pairing::hash;
///
/// let s = hash::bn254_g1();
/// assert_eq!(s.ro_id(), "BN254G1_XMD:SHA-256_SVDW_RO_");
/// assert_eq!((s.l(), s.z()), (48, &s.field().one()));
/// let p = s.hash_to_curve(b"abc", b"QUUX-V01-CS02-with-BN254G1_XMD:SHA-256_SVDW_RO_");
/// let (x, _) = p.coordinates().unwrap();
/// assert!(x.value().to_str_radix(16).starts_with("23f717bee89b1003957139f193e6be7d"));
/// ```
#[must_use]
pub fn bn254_g1() -> &'static G1Suite {
    static SUITE: OnceLock<G1Suite> = OnceLock::new();
    SUITE.get_or_init(|| {
        let curve = PairingCurve::bn254();
        let f = curve.base_field();
        let map = ShallueVanDeWoestijne::new(&f.zero(), &curve.b(), &f.one());
        let fp = |x: &Fp<'static>| fast::Fp::<Bn, 4>::from_public(x);
        let fast = G1Fast::Bn(Svdw::new(fp(map.a()), fp(map.b()), fp(map.z())));
        G1Suite {
            ids: [
                "BN254G1_XMD:SHA-256_SVDW_RO_",
                "BN254G1_XMD:SHA-256_SVDW_NU_",
            ],
            curve,
            map: G1Map::Svdw(map),
            l: element_bytes(f, 128),
            h_eff: Number::from(1u8),
            fast,
        }
    })
}

/// The suites of BLS12-381's `G2` (RFC 9380, section 8.8.2), built on the first call.
///
/// # Examples
///
/// ```
/// use a::math::field::Fp2;
/// use a::math::numbers::Number;
/// use a::math::pairing::hash;
///
/// let s = hash::bls12_381_g2();
/// let t = s.tower();
/// assert_eq!(s.z(), &Fp2::from_numbers(t, &Number::from(-2), &Number::from(-1)));
/// assert_eq!((s.k(), s.l()), (128, 64));
/// assert_eq!(s.h_eff().bits(), 636);
/// ```
#[must_use]
pub fn bls12_381_g2() -> &'static G2Suite {
    static SUITE: OnceLock<G2Suite> = OnceLock::new();
    SUITE.get_or_init(|| {
        let curve = PairingCurve::bls12_381();
        let t = curve.tower();
        let f = t.base();
        let h = |c: &(&str, &str)| Fp2::new(t, hex(f, c.0), hex(f, c.1));
        let n = |x: i64| Number::new_from_i64(x, ByteOrder::LittleEndian);
        let map = SimplifiedSwuFp2::new(
            &Fp2::from_numbers(t, &n(0), &n(240)),
            &Fp2::from_numbers(t, &n(1012), &n(1012)),
            &Fp2::from_numbers(t, &n(-2), &n(-1)),
        );
        let monic = |c: &[(&str, &str)]| -> Vec<Fp2<'static>> {
            c.iter().map(h).chain([Fp2::one(t)]).collect()
        };
        let iso = IsogenyMapFp2::new(
            &G2_ISO_X_NUM.iter().map(h).collect::<Vec<_>>(),
            &monic(&G2_ISO_X_DEN),
            &G2_ISO_Y_NUM.iter().map(h).collect::<Vec<_>>(),
            &monic(&G2_ISO_Y_DEN),
        );
        let h_eff = Number::from_str_radix(G2_H_EFF, 16, ByteOrder::LittleEndian)
            .unwrap_or_else(|_| unreachable!("the constant is valid hexadecimal"));
        let fp2 = |x: &Fp2<'static>| fast::Fp2::<Bls, 6>::from_public(x);
        let fv = |c: &[Fp2<'static>]| c.iter().map(fp2).collect::<Vec<_>>();
        let fast_map = SswuFp2::new(fp2(map.a()), fp2(map.b()), fp2(map.z()));
        let fast_iso = Iso {
            x_num: fv(iso.x_num()),
            x_den: fv(iso.x_den()),
            y_num: fv(iso.y_num()),
            y_den: fv(iso.y_den()),
        };
        G2Suite {
            curve,
            map,
            iso,
            l: element_bytes(f, 128),
            h_eff,
            fast_map,
            fast_iso,
        }
    })
}

/// `A'` and `B'` of the curve `E'` isogenous to BLS12-381's `E` (section 8.8.1).
const G1_ISO_A: &str = "144698a3b8e9433d693a02c96d4982b0ea985383ee66a8d8e8981aefd881ac98936f8da0e0f97f5cf428082d584c1d";
/// See [`G1_ISO_A`].
const G1_ISO_B: &str = "12e2908d11688030018b12e8753eee3b2016c1f0f24f4070a0b9c14fcef35ef55a23215a316ceaa5d1cc48e98e172be0";

/// RFC 9380's `h_eff` for BLS12-381's `G2` (section 8.8.2).
const G2_H_EFF: &str = "bc69f08f2ee75b3584c6a0ea91b352888e2a8e9145ad7689986ff031508ffe1329c2f178731db956d82bf015d1212b02ec0ec69d7477c1ae954cbc06689f6a359894c0adebbf6b4e8020005aaa95551";

/// The 11-isogeny from `E'` onto BLS12-381's `E` of RFC 9380, appendix E.2: the coefficients
/// `k_(i,j)` of `x_num`, `x_den`, `y_num` and `y_den`, the constant first, in hexadecimal; the
/// leading 1 of `x_den` and `y_den` is added where the map is built.
const G1_ISO_X_NUM: [&str; 12] = [
    "11a05f2b1e833340b809101dd99815856b303e88a2d7005ff2627b56cdb4e2c85610c2d5f2e62d6eaeac1662734649b7",
    "17294ed3e943ab2f0588bab22147a81c7c17e75b2f6a8417f565e33c70d1e86b4838f2a6f318c356e834eef1b3cb83bb",
    "d54005db97678ec1d1048c5d10a9a1bce032473295983e56878e501ec68e25c958c3e3d2a09729fe0179f9dac9edcb0",
    "1778e7166fcc6db74e0609d307e55412d7f5e4656a8dbf25f1b33289f1b330835336e25ce3107193c5b388641d9b6861",
    "e99726a3199f4436642b4b3e4118e5499db995a1257fb3f086eeb65982fac18985a286f301e77c451154ce9ac8895d9",
    "1630c3250d7313ff01d1201bf7a74ab5db3cb17dd952799b9ed3ab9097e68f90a0870d2dcae73d19cd13c1c66f652983",
    "d6ed6553fe44d296a3726c38ae652bfb11586264f0f8ce19008e218f9c86b2a8da25128c1052ecaddd7f225a139ed84",
    "17b81e7701abdbe2e8743884d1117e53356de5ab275b4db1a682c62ef0f2753339b7c8f8c8f475af9ccb5618e3f0c88e",
    "80d3cf1f9a78fc47b90b33563be990dc43b756ce79f5574a2c596c928c5d1de4fa295f296b74e956d71986a8497e317",
    "169b1f8e1bcfa7c42e0c37515d138f22dd2ecb803a0c5c99676314baf4bb1b7fa3190b2edc0327797f241067be390c9e",
    "10321da079ce07e272d8ec09d2565b0dfa7dccdde6787f96d50af36003b14866f69b771f8c285decca67df3f1605fb7b",
    "6e08c248e260e70bd1e962381edee3d31d79d7e22c837bc23c0bf1bc24c6b68c24b1b80b64d391fa9c8ba2e8ba2d229",
];
const G1_ISO_X_DEN: [&str; 10] = [
    "8ca8d548cff19ae18b2e62f4bd3fa6f01d5ef4ba35b48ba9c9588617fc8ac62b558d681be343df8993cf9fa40d21b1c",
    "12561a5deb559c4348b4711298e536367041e8ca0cf0800c0126c2588c48bf5713daa8846cb026e9e5c8276ec82b3bff",
    "b2962fe57a3225e8137e629bff2991f6f89416f5a718cd1fca64e00b11aceacd6a3d0967c94fedcfcc239ba5cb83e19",
    "3425581a58ae2fec83aafef7c40eb545b08243f16b1655154cca8abc28d6fd04976d5243eecf5c4130de8938dc62cd8",
    "13a8e162022914a80a6f1d5f43e7a07dffdfc759a12062bb8d6b44e833b306da9bd29ba81f35781d539d395b3532a21e",
    "e7355f8e4e667b955390f7f0506c6e9395735e9ce9cad4d0a43bcef24b8982f7400d24bc4228f11c02df9a29f6304a5",
    "772caacf16936190f3e0c63e0596721570f5799af53a1894e2e073062aede9cea73b3538f0de06cec2574496ee84a3a",
    "14a7ac2a9d64a8b230b3f5b074cf01996e7f63c21bca68a81996e1cdf9822c580fa5b9489d11e2d311f7d99bbdcc5a5e",
    "a10ecf6ada54f825e920b3dafc7a3cce07f8d1d7161366b74100da67f39883503826692abba43704776ec3a79a1d641",
    "95fc13ab9e92ad4476d6e3eb3a56680f682b4ee96f7d03776df533978f31c1593174e4b4b7865002d6384d168ecdd0a",
];
const G1_ISO_Y_NUM: [&str; 16] = [
    "90d97c81ba24ee0259d1f094980dcfa11ad138e48a869522b52af6c956543d3cd0c7aee9b3ba3c2be9845719707bb33",
    "134996a104ee5811d51036d776fb46831223e96c254f383d0f906343eb67ad34d6c56711962fa8bfe097e75a2e41c696",
    "cc786baa966e66f4a384c86a3b49942552e2d658a31ce2c344be4b91400da7d26d521628b00523b8dfe240c72de1f6",
    "1f86376e8981c217898751ad8746757d42aa7b90eeb791c09e4a3ec03251cf9de405aba9ec61deca6355c77b0e5f4cb",
    "8cc03fdefe0ff135caf4fe2a21529c4195536fbe3ce50b879833fd221351adc2ee7f8dc099040a841b6daecf2e8fedb",
    "16603fca40634b6a2211e11db8f0a6a074a7d0d4afadb7bd76505c3d3ad5544e203f6326c95a807299b23ab13633a5f0",
    "4ab0b9bcfac1bbcb2c977d027796b3ce75bb8ca2be184cb5231413c4d634f3747a87ac2460f415ec961f8855fe9d6f2",
    "987c8d5333ab86fde9926bd2ca6c674170a05bfe3bdd81ffd038da6c26c842642f64550fedfe935a15e4ca31870fb29",
    "9fc4018bd96684be88c9e221e4da1bb8f3abd16679dc26c1e8b6e6a1f20cabe69d65201c78607a360370e577bdba587",
    "e1bba7a1186bdb5223abde7ada14a23c42a0ca7915af6fe06985e7ed1e4d43b9b3f7055dd4eba6f2bafaaebca731c30",
    "19713e47937cd1be0dfd0b8f1d43fb93cd2fcbcb6caf493fd1183e416389e61031bf3a5cce3fbafce813711ad011c132",
    "18b46a908f36f6deb918c143fed2edcc523559b8aaf0c2462e6bfe7f911f643249d9cdf41b44d606ce07c8a4d0074d8e",
    "b182cac101b9399d155096004f53f447aa7b12a3426b08ec02710e807b4633f06c851c1919211f20d4c04f00b971ef8",
    "245a394ad1eca9b72fc00ae7be315dc757b3b080d4c158013e6632d3c40659cc6cf90ad1c232a6442d9d3f5db980133",
    "5c129645e44cf1102a159f748c4a3fc5e673d81d7e86568d9ab0f5d396a7ce46ba1049b6579afb7866b1e715475224b",
    "15e6be4e990f03ce4ea50b3b42df2eb5cb181d8f84965a3957add4fa95af01b2b665027efec01c7704b456be69c8b604",
];
const G1_ISO_Y_DEN: [&str; 15] = [
    "16112c4c3a9c98b252181140fad0eae9601a6de578980be6eec3232b5be72e7a07f3688ef60c206d01479253b03663c1",
    "1962d75c2381201e1a0cbd6c43c348b885c84ff731c4d59ca4a10356f453e01f78a4260763529e3532f6102c2e49a03d",
    "58df3306640da276faaae7d6e8eb15778c4855551ae7f310c35a5dd279cd2eca6757cd636f96f891e2538b53dbf67f2",
    "16b7d288798e5395f20d23bf89edb4d1d115c5dbddbcd30e123da489e726af41727364f2c28297ada8d26d98445f5416",
    "be0e079545f43e4b00cc912f8228ddcc6d19c9f0f69bbb0542eda0fc9dec916a20b15dc0fd2ededda39142311a5001d",
    "8d9e5297186db2d9fb266eaac783182b70152c65550d881c5ecd87b6f0f5a6449f38db9dfa9cce202c6477faaf9b7ac",
    "166007c08a99db2fc3ba8734ace9824b5eecfdfa8d0cf8ef5dd365bc400a0051d5fa9c01a58b1fb93d1a1399126a775c",
    "16a3ef08be3ea7ea03bcddfabba6ff6ee5a4375efa1f4fd7feb34fd206357132b920f5b00801dee460ee415a15812ed9",
    "1866c8ed336c61231a1be54fd1d74cc4f9fb0ce4c6af5920abc5750c4bf39b4852cfe2f7bb9248836b233d9d55535d4a",
    "167a55cda70a6e1cea820597d94a84903216f763e13d87bb5308592e7ea7d4fbc7385ea3d529b35e346ef48bb8913f55",
    "4d2f259eea405bd48f010a01ad2911d9c6dd039bb61a6290e591b36e636a5c871a5c29f4f83060400f8b49cba8f6aa8",
    "accbb67481d033ff5852c1e48c50c477f94ff8aefce42d28c0f9a88cea7913516f968986f7ebbea9684b529e2561092",
    "ad6b9514c767fe3c3613144b45f1496543346d98adf02267d5ceef9a00d9b8693000763e3b90ac11e99b138573345cc",
    "2660400eb2e4f3b628bdd0d53cd76f2bf565b94e72927c1cb748df27942480e420517bd8714cc80d1fadc1326ed06f7",
    "e0fa1d816ddc03e6b24255e0d7819c171c40f65e273b853324efcd6356caa205ca2f570f13497804415473a1d634b8f",
];

/// The 3-isogeny from `E'` onto BLS12-381's twist of RFC 9380, appendix E.3, as above, each
/// coefficient `c0 + c1·I` as the pair `(c0, c1)`.
const G2_ISO_X_NUM: [(&str, &str); 4] = [
    (
        "5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97d6",
        "5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97d6",
    ),
    (
        "0",
        "11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71a",
    ),
    (
        "11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71e",
        "8ab05f8bdd54cde190937e76bc3e447cc27c3d6fbd7063fcd104635a790520c0a395554e5c6aaaa9354ffffffffe38d",
    ),
    (
        "171d6541fa38ccfaed6dea691f5fb614cb14b4e7f4e810aa22d6108f142b85757098e38d0f671c7188e2aaaaaaaa5ed1",
        "0",
    ),
];
const G2_ISO_X_DEN: [(&str, &str); 2] = [
    (
        "0",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa63",
    ),
    (
        "c",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa9f",
    ),
];
const G2_ISO_Y_NUM: [(&str, &str); 4] = [
    (
        "1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf54439d87d27e500fc8c25ebf8c92f6812cfc71c71c6d706",
        "1530477c7ab4113b59a4c18b076d11930f7da5d4a07f649bf54439d87d27e500fc8c25ebf8c92f6812cfc71c71c6d706",
    ),
    (
        "0",
        "5c759507e8e333ebb5b7a9a47d7ed8532c52d39fd3a042a88b58423c50ae15d5c2638e343d9c71c6238aaaaaaaa97be",
    ),
    (
        "11560bf17baa99bc32126fced787c88f984f87adf7ae0c7f9a208c6b4f20a4181472aaa9cb8d555526a9ffffffffc71c",
        "8ab05f8bdd54cde190937e76bc3e447cc27c3d6fbd7063fcd104635a790520c0a395554e5c6aaaa9354ffffffffe38f",
    ),
    (
        "124c9ad43b6cf79bfbf7043de3811ad0761b0f37a1e26286b0e977c69aa274524e79097a56dc4bd9e1b371c71c718b10",
        "0",
    ),
];
const G2_ISO_Y_DEN: [(&str, &str); 3] = [
    (
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa8fb",
    ),
    (
        "0",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffa9d3",
    ),
    (
        "12",
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaa99",
    ),
];

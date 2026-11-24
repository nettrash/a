//! Montgomery multiplication at a fixed width, [`Montgomery<N>`], and the modular arithmetic of
//! [`Uint<N>`] built on it: [`Uint::pow_mod`], [`Uint::mul_mod`], [`Uint::add_mod`],
//! [`Uint::sub_mod`] and [`Uint::mod_inverse`].
//!
//! Modulo an odd `m > 1` of `N` limbs, with `R = 2^(64·N)`, a number `x` is kept as its Montgomery
//! form `x·R mod m`. The product of two forms is reduced by Montgomery's REDC (P. L. Montgomery,
//! *Modular multiplication without trial division*, Math. Comp. 44, 1985), which divides by `R`
//! instead of by `m`: for each limb a multiple of `m` that clears the low limb is added, and the
//! limb is shifted out. The product and the reduction are interleaved a limb of the multiplier at a
//! time, the method CIOS of Koç, Acar and Kaliski (*Analyzing and comparing Montgomery
//! multiplication algorithms*, IEEE Micro 16(3), 1996): `2·N² + N` products of limbs and one
//! conditional subtraction at the end. Powers are taken by the sliding-window method (Menezes, van
//! Oorschot and Vanstone, *Handbook of Applied Cryptography*, 1996, Algorithm 14.85), with a
//! window sized to the exponent; inverses by the binary algorithm (Hankerson, Menezes and
//! Vanstone, *Guide to Elliptic Curve Cryptography*, 2004, Algorithm 2.22).
//!
//! `R mod m` and `R² mod m` are remainders of Knuth's algorithm D (*The Art of Computer
//! Programming*, vol. 2, 4.3.1), taken here one quotient limb at a time from the top of the
//! dividend, which needs no room for the dividend: the same division reduces the double-width
//! products of [`Uint::mul_mod`]. An even modulus `2ᵏ·q` is split: modulo the odd `q` in
//! Montgomery form, modulo `2ᵏ` by wrapping arithmetic, and the two remainders put together by the
//! Chinese remainder theorem in Garner's form, with `q⁻¹ mod 2ᵏ` from Newton's iteration.
//!
//! Nothing here is constant-time: the time of a power depends on the bits of the exponent, the
//! final subtraction of a product is taken or not depending on the values, and the inverse takes
//! as many steps as its values need. None of it is meant for secret values.

use super::Uint;
use core::cmp::Ordering;

/// The arithmetic modulo an odd `m > 1` of `N` limbs in Montgomery form, `R = 2^(64·N)`.
///
/// A number `x` is represented by its form `x·R mod m`, a [`Uint<N>`] below `m`:
/// [`to_montgomery`](Self::to_montgomery) and [`from_montgomery`](Self::from_montgomery) convert,
/// and [`mul`](Self::mul), [`square`](Self::square), [`add`](Self::add), [`sub`](Self::sub),
/// [`neg`](Self::neg), [`pow`](Self::pow) and [`inverse`](Self::inverse) work on forms. Every
/// result is below `m`; an argument may be any `Uint<N>` and stands for its residue modulo `m`:
/// one at or above `m` is reduced first.
///
/// [`new`](Self::new) computes `−m⁻¹ mod 2⁶⁴`, `R mod m` and `R² mod m` once; then a product
/// costs `2·N² + N` products of limbs (CIOS). The modulus may be as large as `R − 1`: the sum in
/// the reduction is kept with its extra carry bit. Nothing is constant-time.
///
/// # Examples
///
/// ```
/// use a::math::uint::{Montgomery, Uint};
///
/// let m = Montgomery::new(&Uint::<4>::from_u64(1_000_003));
/// assert_eq!(m.modulus(), &Uint::from_u64(1_000_003));
/// let x = m.to_montgomery(&Uint::from_u64(2));
/// let y = m.pow(&x, &Uint::<1>::from_u64(20));
/// assert_eq!(m.from_montgomery(&y), Uint::from_u64((1 << 20) % 1_000_003));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Montgomery<const N: usize> {
    /// The modulus, odd and greater than 1.
    modulus: Uint<N>,
    /// `−1/m mod 2⁶⁴`: the factor by which a multiple of `m` clears the low limb of a sum.
    minv: u64,
    /// `R mod m`, the form of 1.
    one: Uint<N>,
    /// `R² mod m`, by which one product takes a number to its form.
    r2: Uint<N>,
}

impl<const N: usize> Montgomery<N> {
    /// The arithmetic modulo `m`.
    ///
    /// It computes `−m⁻¹ mod 2⁶⁴` by Newton's iteration and `R mod m` and `R² mod m` by long
    /// division, `O(N²)` operations on limbs.
    ///
    /// # Panics
    ///
    /// If `m` is even or 1 (also for `N = 0`, which holds no odd number).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<4>::from_u64(97));
    /// assert_eq!(m.modulus(), &Uint::from_u64(97));
    /// // R = 2^256 = 61 modulo 97.
    /// assert_eq!(m.one(), Uint::from_u64(61));
    /// ```
    #[track_caller]
    pub fn new(m: &Uint<N>) -> Self {
        assert!(
            N > 0 && m.as_limbs()[0] & 1 == 1 && m.bits() > 1,
            "the modulus must be odd and greater than 1"
        );
        let limbs = m.as_limbs();
        let divisor = Divisor::new(limbs);
        // R and R² are a one above N and 2·N zero limbs.
        let one = divisor.rem(N + 1, |i| u64::from(i == N));
        let r2 = divisor.rem(2 * N + 1, |i| u64::from(i == 2 * N));
        Montgomery {
            modulus: *m,
            minv: inverse_limb(limbs[0]).wrapping_neg(),
            one: Uint::from_limbs(one),
            r2: Uint::from_limbs(r2),
        }
    }

    /// The modulus.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// assert_eq!(Montgomery::new(&Uint::<2>::MAX).modulus(), &Uint::MAX);
    /// ```
    #[inline]
    pub fn modulus(&self) -> &Uint<N> {
        &self.modulus
    }

    /// The form of 1, `R mod m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<2>::from_u64(1_000_003));
    /// assert_eq!(m.from_montgomery(&m.one()), Uint::from_u64(1));
    /// ```
    #[inline]
    pub fn one(&self) -> Uint<N> {
        self.one
    }

    /// The form of `x`, `x·R mod m`, below `m`, for any `x`: one product by `R² mod m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<1>::from_u64(101));
    /// // 2^64 = 79 modulo 101, so the form of 3 is 3 · 79 = 35, and 104 is 3 again.
    /// assert_eq!(m.to_montgomery(&Uint::from_u64(3)), Uint::from_u64(35));
    /// assert_eq!(m.to_montgomery(&Uint::from_u64(104)), Uint::from_u64(35));
    /// ```
    pub fn to_montgomery(&self, x: &Uint<N>) -> Uint<N> {
        // x < R and R² mod m < m: the product is below R·m, as REDC needs.
        self.redc_mul(x, &self.r2)
    }

    /// The number a form stands for, `x·R⁻¹ mod m`, below `m`, for any `x`: one reduction.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<3>::from_u64(1_000_003));
    /// let x = Uint::from_u64(123_456);
    /// assert_eq!(m.from_montgomery(&m.to_montgomery(&x)), x);
    /// ```
    pub fn from_montgomery(&self, x: &Uint<N>) -> Uint<N> {
        // REDC of x < R: (x + q·m)/R < 1 + m, below m after the final subtraction.
        self.redc_mul(x, &Uint::from_u64(1))
    }

    /// The form of the product of what `a` and `b` stand for: `a·b·R⁻¹ mod m`, by CIOS.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<4>::from_u64(1_000_003));
    /// let a = m.to_montgomery(&Uint::from_u64(1234));
    /// let b = m.to_montgomery(&Uint::from_u64(5678));
    /// let p = m.from_montgomery(&m.mul(&a, &b));
    /// assert_eq!(p, Uint::from_u64(1234 * 5678 % 1_000_003));
    /// ```
    pub fn mul(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
        // REDC needs a·b < m·R: one factor below m is enough.
        if *a < self.modulus || *b < self.modulus {
            self.redc_mul(a, b)
        } else {
            self.redc_mul(&self.reduce(a), b)
        }
    }

    /// The form of the square of what `a` stands for: `a²·R⁻¹ mod m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<2>::from_u64(1_000_003));
    /// let a = m.to_montgomery(&Uint::from_u64(999_999));
    /// assert_eq!(m.from_montgomery(&m.square(&a)), Uint::from_u64(16));
    /// ```
    pub fn square(&self, a: &Uint<N>) -> Uint<N> {
        let a = self.reduce(a);
        self.redc_mul(&a, &a)
    }

    /// The form of the sum, `a + b mod m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<1>::from_u64(101));
    /// let s = m.add(&m.to_montgomery(&Uint::from_u64(60)), &m.to_montgomery(&Uint::from_u64(50)));
    /// assert_eq!(m.from_montgomery(&s), Uint::from_u64(9));
    /// ```
    pub fn add(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
        add_reduced(&self.reduce(a), &self.reduce(b), &self.modulus)
    }

    /// The form of the difference, `a − b mod m`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<1>::from_u64(101));
    /// let d = m.sub(&m.to_montgomery(&Uint::from_u64(5)), &m.to_montgomery(&Uint::from_u64(7)));
    /// assert_eq!(m.from_montgomery(&d), Uint::from_u64(99));
    /// ```
    pub fn sub(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
        sub_reduced(&self.reduce(a), &self.reduce(b), &self.modulus)
    }

    /// The form of the negation, `−a mod m`: 0 for 0, else `m − a`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<1>::from_u64(101));
    /// let n = m.neg(&m.to_montgomery(&Uint::from_u64(1)));
    /// assert_eq!(m.from_montgomery(&n), Uint::from_u64(100));
    /// assert_eq!(m.neg(&Uint::ZERO), Uint::ZERO);
    /// ```
    pub fn neg(&self, a: &Uint<N>) -> Uint<N> {
        sub_reduced(&Uint::ZERO, &self.reduce(a), &self.modulus)
    }

    /// The form of `x^e`, what the form `x` stands for raised to the power `e`, an integer of any
    /// width `M`; `x⁰` is the form of 1, also for `x = 0`.
    ///
    /// By the sliding-window method with a window of `w` = 1 to 6 bits chosen from the length `ℓ`
    /// of `e` (1 up to 12 bits, 2 up to 24, 3 up to 80, 4 up to 240, 5 up to 672, 6 beyond): about
    /// `ℓ` squarings and `ℓ/(w + 1)` products, and for `w ≥ 2` a table of `2^(w−1)` odd powers,
    /// which costs a squaring and `2^(w−1) − 1` products. The table is on the stack, so from 33
    /// limbs on the window is at most 4 bits and from 129 limbs on at most 2, which keeps the
    /// table within 8 KiB up to 512 limbs. The time depends on the exponent's bits.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// // Fermat: 3^(p−1) = 1 modulo the prime p = 2^127 − 1.
    /// let p = Uint::<2>::from_limbs([u64::MAX, u64::MAX >> 1]);
    /// let m = Montgomery::new(&p);
    /// let e = p.overflowing_sub(Uint::from_u64(1)).0;
    /// let x = m.pow(&m.to_montgomery(&Uint::from_u64(3)), &e);
    /// assert_eq!(x, m.one());
    /// ```
    pub fn pow<const M: usize>(&self, x: &Uint<N>, e: &Uint<M>) -> Uint<N> {
        self.pow_limbs(x, e.as_limbs())
    }

    /// The form of `x^e` for an exponent given as limbs, the low one first (`&[e]` for a `u64`),
    /// of any length; as [`pow`](Self::pow).
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<1>::from_u64(1_000_003));
    /// let x = m.to_montgomery(&Uint::from_u64(10));
    /// assert_eq!(m.from_montgomery(&m.pow_limbs(&x, &[6])), Uint::from_u64(1_000_000));
    /// // 10^(2^64) by an exponent of two limbs.
    /// let y = m.pow_limbs(&x, &[0, 1]);
    /// let mut z = x;
    /// for _ in 0..64 {
    ///     z = m.square(&z);
    /// }
    /// assert_eq!(y, z);
    /// ```
    pub fn pow_limbs(&self, x: &Uint<N>, e: &[u64]) -> Uint<N> {
        let x = self.reduce(x);
        power_of(&x, &self.one, e, &|a, b| self.redc_mul(a, b), &|a| {
            self.redc_mul(a, a)
        })
    }

    /// The form of the inverse of what `x` stands for, `x⁻¹·R mod m`, or `None` when
    /// `gcd(x, m) ≠ 1` (0 included). `m` need not be prime.
    ///
    /// By the binary algorithm on the residue (Hankerson, Menezes and Vanstone, Algorithm 2.22,
    /// with a stop at a common divisor), then two products by `R² mod m`: `O(N²)` operations on
    /// limbs, at most `2·64·N` steps of shifts and subtractions.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::{Montgomery, Uint};
    ///
    /// let m = Montgomery::new(&Uint::<2>::from_u64(91)); // 7 · 13
    /// let x = m.to_montgomery(&Uint::from_u64(10));
    /// let y = m.inverse(&x).unwrap();
    /// assert_eq!(m.from_montgomery(&y), Uint::from_u64(82)); // 10 · 82 = 820 = 9 · 91 + 1
    /// assert_eq!(m.mul(&x, &y), m.one());
    /// assert_eq!(m.inverse(&m.to_montgomery(&Uint::from_u64(14))), None);
    /// ```
    pub fn inverse(&self, x: &Uint<N>) -> Option<Uint<N>> {
        // y = 1/(x R) mod m, and the form of 1/x is y·R² = REDC(REDC(y·R²)·R²).
        let y = binary_inverse(&self.reduce(x), &self.modulus)?;
        Some(self.redc_mul(&self.redc_mul(&y, &self.r2), &self.r2))
    }

    /// `x mod m`: `x` itself when it is below `m`, else one reduction of its product by `R mod m`.
    #[inline]
    fn reduce(&self, x: &Uint<N>) -> Uint<N> {
        if *x < self.modulus {
            *x
        } else {
            // x·(R mod m)·R⁻¹ = x modulo m, and x·(R mod m) < R·m.
            self.redc_mul(x, &self.one)
        }
    }

    /// `a·b·R⁻¹ mod m`, below `m`, for `a·b < m·R`, by CIOS (Koç, Acar and Kaliski): for each limb
    /// of `b`, add `a·bᵢ`, then the multiple `q·m` that clears the low limb, and shift a limb out.
    ///
    /// The running sum `t` stays below `a + m < 2·R` between the rows: it is kept in `N` limbs
    /// and a carry `top`, which is what makes a modulus near `R` work. At the end `t < 2·m`, and
    /// one conditional subtraction leaves it below `m`.
    fn redc_mul(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
        let (a, b, m) = (a.as_limbs(), b.as_limbs(), self.modulus.as_limbs());
        let mut t = [0u64; N];
        let mut top = 0u64;
        for &bi in b {
            // t += a·bᵢ: N + 1 limbs and a carry bit (`over`).
            let mut c = 0u64;
            for (tj, &aj) in t.iter_mut().zip(a) {
                let p = u128::from(aj) * u128::from(bi) + u128::from(*tj) + u128::from(c);
                *tj = p as u64;
                c = (p >> 64) as u64;
            }
            let (high, over) = top.overflowing_add(c);
            // t = (t + q·m)/2⁶⁴, with q·m ≡ −t mod 2⁶⁴.
            let q = t[0].wrapping_mul(self.minv);
            let p = u128::from(q) * u128::from(m[0]) + u128::from(t[0]);
            let mut c = (p >> 64) as u64;
            for j in 1..N {
                let p = u128::from(q) * u128::from(m[j]) + u128::from(t[j]) + u128::from(c);
                t[j - 1] = p as u64;
                c = (p >> 64) as u64;
            }
            let (low, carry) = high.overflowing_add(c);
            t[N - 1] = low;
            top = u64::from(over) + u64::from(carry);
        }
        let t = Uint::from_limbs(t);
        let (d, borrow) = t.overflowing_sub(self.modulus);
        // t ≥ m when the carry is set (t ≥ R > m) or the subtraction does not borrow.
        if top != 0 || !borrow { d } else { t }
    }
}

impl<const N: usize> Uint<N> {
    /// `self^exp mod m`, below `m`, for any nonzero `m` and an exponent of any width `M`.
    ///
    /// As [`Number::pow_mod`](crate::math::numbers::Number::pow_mod): `x⁰ = 1` for every `x`, 0
    /// included, so that modulo 1, where every remainder is 0, the result is 0; a base at or above
    /// `m` is reduced. An odd `m` goes through [`Montgomery`]; an even one, `2ᵏ·q` with `q` odd, is
    /// split into the power modulo `q` in Montgomery form and the power modulo `2ᵏ` by wrapping
    /// products, put together by the Chinese remainder theorem. `O(N²·ℓ)` operations on limbs for
    /// an exponent of `ℓ` bits, and for an even `m` another `O(N²·log N)` for `q⁻¹ mod 2^(64·N)`
    /// (`⌈log₂ N⌉` Newton steps of `O(N²)`). Not constant-time: the time depends on the exponent's
    /// bits.
    ///
    /// # Panics
    ///
    /// If `m` is zero, with the message of `%`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// // 4^13 mod 497 = 445, and modulo the even 500 = 2^2 · 125 it is 364.
    /// let b = Uint::<4>::from_u64(4);
    /// let e = Uint::<1>::from_u64(13);
    /// assert_eq!(b.pow_mod(&e, &Uint::from_u64(497)), Uint::from_u64(445));
    /// assert_eq!(b.pow_mod(&e, &Uint::from_u64(500)), Uint::from_u64(364));
    /// assert_eq!(Uint::<4>::ZERO.pow_mod(&Uint::<1>::ZERO, &Uint::from_u64(7)), Uint::from_u64(1));
    /// ```
    #[track_caller]
    pub fn pow_mod<const M: usize>(&self, exp: &Uint<M>, m: &Uint<N>) -> Uint<N> {
        if m.is_zero() {
            by_zero();
        }
        let e = exp.as_limbs();
        if m.bits() == 1 {
            return Uint::ZERO;
        }
        if exp.is_zero() {
            return Uint::from_u64(1);
        }
        let k = trailing_zeros(m.as_limbs());
        if k == 0 {
            return pow_odd(self, e, m);
        }
        // m = 2ᵏ·q: modulo 2^(64·N) by wrapping products, then the low k bits.
        let one = Uint::from_u64(1);
        let r2 = low_bits(&power_of(self, &one, e, &mul_low, &|a| mul_low(a, a)), k);
        let q = shr(m, k);
        if q.bits() == 1 {
            return r2;
        }
        crt(&pow_odd(self, e, &q), &q, &r2, k)
    }

    /// `self · rhs mod m`, below `m`, for any nonzero `m`: the double-width product reduced by
    /// long division (Knuth's algorithm D), `O(N²)` operations on limbs.
    ///
    /// # Panics
    ///
    /// If `m` is zero, with the message of `%`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// // 2^128 − 1 is 1 modulo 2^128 − 2, and so is its square, though the square needs 256 bits.
    /// let x = Uint::<2>::MAX;
    /// let m = Uint::<2>::from_limbs([u64::MAX - 1, u64::MAX]);
    /// assert_eq!(x.mul_mod(&x, &m), Uint::from_u64(1));
    /// assert_eq!(Uint::<1>::from_u64(7).mul_mod(&Uint::from_u64(8), &Uint::from_u64(10)), Uint::from_u64(6));
    /// ```
    #[track_caller]
    pub fn mul_mod(&self, rhs: &Uint<N>, m: &Uint<N>) -> Uint<N> {
        if m.is_zero() {
            by_zero();
        }
        let (low, high) = self.widening_mul(*rhs);
        let (low, high) = (low.as_limbs(), high.as_limbs());
        let r = Divisor::new(m.as_limbs()).rem(2 * N, |i| if i < N { low[i] } else { high[i - N] });
        Uint::from_limbs(r)
    }

    /// `self + rhs mod m`, below `m`, for any nonzero `m`; operands at or above `m` are reduced
    /// first.
    ///
    /// # Panics
    ///
    /// If `m` is zero, with the message of `%`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let m = Uint::<2>::MAX;
    /// let x = m.overflowing_sub(Uint::from_u64(1)).0;
    /// // (m − 1) + (m − 1) = m − 2 modulo m, though the sum does not fit in 128 bits.
    /// assert_eq!(x.add_mod(&x, &m), m.overflowing_sub(Uint::from_u64(2)).0);
    /// assert_eq!(Uint::<1>::from_u64(17).add_mod(&Uint::from_u64(5), &Uint::from_u64(10)), Uint::from_u64(2));
    /// ```
    #[track_caller]
    pub fn add_mod(&self, rhs: &Uint<N>, m: &Uint<N>) -> Uint<N> {
        if m.is_zero() {
            by_zero();
        }
        add_reduced(&rem(self, m), &rem(rhs, m), m)
    }

    /// `self − rhs mod m`, below `m`, for any nonzero `m`; operands at or above `m` are reduced
    /// first.
    ///
    /// # Panics
    ///
    /// If `m` is zero, with the message of `%`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// assert_eq!(Uint::<1>::from_u64(3).sub_mod(&Uint::from_u64(5), &Uint::from_u64(10)), Uint::from_u64(8));
    /// assert_eq!(Uint::<1>::from_u64(3).sub_mod(&Uint::from_u64(25), &Uint::from_u64(10)), Uint::from_u64(8));
    /// ```
    #[track_caller]
    pub fn sub_mod(&self, rhs: &Uint<N>, m: &Uint<N>) -> Uint<N> {
        if m.is_zero() {
            by_zero();
        }
        sub_reduced(&rem(self, m), &rem(rhs, m), m)
    }

    /// The inverse of `self` modulo `m`: the `r < m` with `self · r ≡ 1 (mod m)`, or `None` when
    /// there is none, which is when `gcd(self, m) ≠ 1`. As for
    /// [`Number::mod_inverse`](crate::math::numbers::Number::mod_inverse) and GMP's `mpz_invert`,
    /// every number has the inverse 0 modulo 1.
    ///
    /// Modulo an odd `m` by the binary algorithm (Hankerson, Menezes and Vanstone, Algorithm 2.22);
    /// modulo an even `2ᵏ·q`, the inverse modulo `2ᵏ` by Newton's iteration and the one modulo `q`
    /// by the binary algorithm, put together by the Chinese remainder theorem. `O(N²)` operations
    /// on limbs for an odd `m`; `O(N²·log N)` for an even one, whose Newton iterations modulo
    /// `2^(64·N)` take `⌈log₂ N⌉` steps of `O(N²)` each.
    ///
    /// # Panics
    ///
    /// If `m` is zero, with the message of `%`.
    ///
    /// # Examples
    ///
    /// ```
    /// use a::math::uint::Uint;
    ///
    /// let m = Uint::<2>::from_u64(1000);
    /// assert_eq!(Uint::<2>::from_u64(7).mod_inverse(&m), Some(Uint::from_u64(143)));
    /// assert_eq!(Uint::<2>::from_u64(6).mod_inverse(&m), None);
    /// assert_eq!(Uint::<2>::from_u64(6).mod_inverse(&Uint::from_u64(1)), Some(Uint::ZERO));
    /// ```
    #[track_caller]
    pub fn mod_inverse(&self, m: &Uint<N>) -> Option<Uint<N>> {
        if m.is_zero() {
            by_zero();
        }
        if m.bits() == 1 {
            return Some(Uint::ZERO);
        }
        let k = trailing_zeros(m.as_limbs());
        if k == 0 {
            return binary_inverse(&rem(self, m), m);
        }
        // Modulo 2ᵏ an inverse exists exactly for odd numbers.
        if self.as_limbs()[0] & 1 == 0 {
            return None;
        }
        let r2 = low_bits(&inverse_wrapping(self), k);
        let q = shr(m, k);
        if q.bits() == 1 {
            return Some(r2);
        }
        let r1 = binary_inverse(&rem(self, &q), &q)?;
        Some(crt(&r1, &q, &r2, k))
    }
}

/// Panics as `%` does for a divisor of zero, as `Number::pow_mod` does for a modulus of zero.
#[cold]
#[inline(never)]
#[track_caller]
fn by_zero() -> ! {
    panic!("attempt to calculate the remainder with a divisor of zero")
}

/// `x^e mod m` for an odd `m > 1`, through its Montgomery form.
fn pow_odd<const N: usize>(x: &Uint<N>, e: &[u64], m: &Uint<N>) -> Uint<N> {
    let mont = Montgomery::new(m);
    mont.from_montgomery(&mont.pow_limbs(&mont.to_montgomery(x), e))
}

/// `1/x mod 2⁶⁴` for an odd `x`, by Newton's iteration `y ← y·(2 − x·y)`, which doubles the
/// number of right low bits: `y = x` is right to 3 bits (`x² ≡ 1 mod 8`), and five steps make 96.
const fn inverse_limb(x: u64) -> u64 {
    let mut y = x;
    let mut i = 0;
    while i < 5 {
        y = y.wrapping_mul(2u64.wrapping_sub(x.wrapping_mul(y)));
        i += 1;
    }
    y
}

/// `1/x mod 2^(64·N)` for an odd `x`: Newton's iteration from the inverse of the low limb, each
/// step doubling the 64, 128, … right low bits, `⌈log₂ N⌉` steps of two wrapping products.
fn inverse_wrapping<const N: usize>(x: &Uint<N>) -> Uint<N> {
    let two = Uint::from_u64(2);
    let mut y = Uint::from_u64(inverse_limb(x.as_limbs()[0]));
    let mut right = 64;
    while right < 64 * N {
        let (d, _) = two.overflowing_sub(mul_low(x, &y));
        y = mul_low(&y, &d);
        right *= 2;
    }
    y
}

/// The `y < 2ᵏ·q` with `y ≡ r1 (mod q)` and `y ≡ r2 (mod 2ᵏ)`, for an odd `q`, `r1 < q`,
/// `r2 < 2ᵏ` and `2ᵏ·q < 2^(64·N)`, in Garner's form: `y = r1 + q·((r2 − r1)·q⁻¹ mod 2ᵏ)`.
fn crt<const N: usize>(r1: &Uint<N>, q: &Uint<N>, r2: &Uint<N>, k: u32) -> Uint<N> {
    let (d, _) = r2.overflowing_sub(*r1);
    let h = low_bits(&mul_low(&d, &inverse_wrapping(q)), k);
    // q·h < q·2ᵏ fits, and r1 + q·h < q + q·(2ᵏ − 1) = 2ᵏ·q.
    let (y, _) = r1.overflowing_add(mul_low(q, &h));
    y
}

/// [`power`] with a table of odd powers on the stack of at most 8 KiB up to 512 limbs: 32 powers
/// up to 32 limbs, 8 up to 128, and 2 beyond.
fn power_of<const N: usize>(
    x: &Uint<N>,
    one: &Uint<N>,
    e: &[u64],
    mul: &impl Fn(&Uint<N>, &Uint<N>) -> Uint<N>,
    sqr: &impl Fn(&Uint<N>) -> Uint<N>,
) -> Uint<N> {
    if N <= 32 {
        power::<_, 32>(x, one, e, mul, sqr)
    } else if N <= 128 {
        power::<_, 8>(x, one, e, mul, sqr)
    } else {
        power::<_, 2>(x, one, e, mul, sqr)
    }
}

/// `x^e` in a ring with the unit `one`, the product `mul` and the square `sqr`, by the
/// sliding-window method (Handbook of Applied Cryptography, Algorithm 14.85): from the top bit
/// down, a zero bit costs a squaring, and a window of up to `w` bits that starts and ends with a
/// one costs its length in squarings and one product by a power from the table of the odd powers
/// `x, x³, …, x^(2^w − 1)`. The window grows with the exponent's length `ℓ`: it is the `w` that
/// minimises `T(w) + ℓ/(w + 1)`, about `ℓ/(w + 1)` windows and the cost `T(w)` of the table, which
/// is 0 for `w = 1` (no table) and `2^(w−1)` for `w ≥ 2` (a squaring for `x²` and `2^(w−1) − 1`
/// products), a tie going to the smaller `w`: 1 up to 12 bits, 2 up to 24, 3 up to 80, 4 up to
/// 240, 5 up to 672, 6 beyond. It is at most `1 + log₂ K` for a table of `K` powers.
fn power<T: Copy, const K: usize>(
    x: &T,
    one: &T,
    e: &[u64],
    mul: &impl Fn(&T, &T) -> T,
    sqr: &impl Fn(&T) -> T,
) -> T {
    let bits = match e.iter().rposition(|&l| l != 0) {
        None => return *one,
        Some(i) => 64 * i + 64 - e[i].leading_zeros() as usize,
    };
    let bit = |i: usize| (e[i / 64] >> (i % 64)) & 1 == 1;
    let w = match bits {
        0..=12 => 1,
        13..=24 => 2,
        25..=80 => 3,
        81..=240 => 4,
        241..=672 => 5,
        _ => 6,
    }
    .min(K.ilog2() as usize + 1);
    let mut table = [*x; K];
    let count = 1 << (w - 1);
    if count > 1 {
        let x2 = sqr(x);
        for i in 1..count {
            table[i] = mul(&table[i - 1], &x2);
        }
    }
    // The bits below `i` are still to be read; the top one is set, so the first window starts
    // the power.
    let mut acc: Option<T> = None;
    let mut i = bits;
    while i > 0 {
        if !bit(i - 1) {
            acc = acc.map(|a| sqr(&a));
            i -= 1;
            continue;
        }
        // The window from bit i − 1 down to the lowest set bit j ≥ i − w.
        let mut j = i.saturating_sub(w);
        while !bit(j) {
            j += 1;
        }
        let mut v = 0;
        for p in (j..i).rev() {
            v = v << 1 | usize::from(bit(p));
        }
        acc = Some(match acc {
            None => table[v >> 1],
            Some(mut a) => {
                for _ in j..i {
                    a = sqr(&a);
                }
                mul(&a, &table[v >> 1])
            }
        });
        i = j;
    }
    acc.unwrap_or(*one)
}

/// `1/x mod m` for an odd `m > 1` and `x < m`, or `None` when `gcd(x, m) ≠ 1`: the binary
/// algorithm of Hankerson, Menezes and Vanstone (Algorithm 2.22).
///
/// It keeps `x1·x ≡ u` and `x2·x ≡ v (mod m)` from `u = x, v = m, x1 = 1, x2 = 0`: halves the
/// even one of `u` and `v` and its coefficient modulo `m`, and subtracts the smaller of the two
/// odd ones from the larger, until one is 1 (the inverse is its coefficient) or they meet at a
/// common divisor greater than 1 (there is none). Each step shortens `u·v` by a bit: at most
/// `2·64·N` steps of `O(N)`.
fn binary_inverse<const N: usize>(x: &Uint<N>, m: &Uint<N>) -> Option<Uint<N>> {
    if x.is_zero() {
        return None;
    }
    let (mut u, mut v) = (*x, *m);
    let (mut x1, mut x2) = (Uint::from_u64(1), Uint::ZERO);
    loop {
        while u.as_limbs()[0] & 1 == 0 {
            u = shr(&u, 1);
            x1 = halve(&x1, m);
        }
        while v.as_limbs()[0] & 1 == 0 {
            v = shr(&v, 1);
            x2 = halve(&x2, m);
        }
        if u.bits() == 1 {
            return Some(x1);
        }
        if v.bits() == 1 {
            return Some(x2);
        }
        match u.cmp(&v) {
            Ordering::Equal => return None,
            Ordering::Greater => {
                u = u.overflowing_sub(v).0;
                x1 = sub_reduced(&x1, &x2, m);
            }
            Ordering::Less => {
                v = v.overflowing_sub(u).0;
                x2 = sub_reduced(&x2, &x1, m);
            }
        }
    }
}

/// `x/2 mod m` for an odd `m` and `x < m`: `x/2` or `(x + m)/2`, the carry of the sum shifted in.
fn halve<const N: usize>(x: &Uint<N>, m: &Uint<N>) -> Uint<N> {
    if x.as_limbs()[0] & 1 == 0 {
        return shr(x, 1);
    }
    let (s, carry) = x.overflowing_add(*m);
    let mut limbs = shr(&s, 1).into_limbs();
    limbs[N - 1] |= u64::from(carry) << 63;
    Uint::from_limbs(limbs)
}

/// `a + b mod m` for `a, b < m`.
fn add_reduced<const N: usize>(a: &Uint<N>, b: &Uint<N>, m: &Uint<N>) -> Uint<N> {
    let (s, carry) = a.overflowing_add(*b);
    // s < 2·m: once m when the true sum is at least m.
    if carry || s >= *m {
        s.overflowing_sub(*m).0
    } else {
        s
    }
}

/// `a − b mod m` for `a, b < m`.
fn sub_reduced<const N: usize>(a: &Uint<N>, b: &Uint<N>, m: &Uint<N>) -> Uint<N> {
    let (d, borrow) = a.overflowing_sub(*b);
    if borrow { d.overflowing_add(*m).0 } else { d }
}

/// `x mod m` for a nonzero `m`: `x` when it is below `m`, else a long division.
fn rem<const N: usize>(x: &Uint<N>, m: &Uint<N>) -> Uint<N> {
    if x < m {
        return *x;
    }
    let limbs = x.as_limbs();
    Uint::from_limbs(Divisor::new(m.as_limbs()).rem(N, |i| limbs[i]))
}

/// The low half of `a·b`: the product modulo `2^(64·N)`, `N·(N + 1)/2` products of limbs.
fn mul_low<const N: usize>(a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
    let (a, b) = (a.as_limbs(), b.as_limbs());
    let mut r = [0u64; N];
    for i in 0..N {
        let mut c = 0u64;
        for j in 0..N - i {
            let p = u128::from(a[i]) * u128::from(b[j]) + u128::from(r[i + j]) + u128::from(c);
            r[i + j] = p as u64;
            c = (p >> 64) as u64;
        }
    }
    Uint::from_limbs(r)
}

/// `x mod 2ᵏ`, for `k < 64·N`.
fn low_bits<const N: usize>(x: &Uint<N>, k: u32) -> Uint<N> {
    let mut limbs = x.into_limbs();
    let (whole, part) = ((k / 64) as usize, k % 64);
    for (i, l) in limbs.iter_mut().enumerate().skip(whole) {
        *l = if i == whole {
            *l & ((1 << part) - 1)
        } else {
            0
        };
    }
    Uint::from_limbs(limbs)
}

/// `x / 2ᵏ`, for `k < 64·N`.
fn shr<const N: usize>(x: &Uint<N>, k: u32) -> Uint<N> {
    let x = x.as_limbs();
    let (whole, part) = ((k / 64) as usize, k % 64);
    let mut r = [0u64; N];
    for i in 0..N - whole {
        let low = x[i + whole] >> part;
        let high = if part > 0 && i + whole + 1 < N {
            x[i + whole + 1] << (64 - part)
        } else {
            0
        };
        r[i] = low | high;
    }
    Uint::from_limbs(r)
}

/// The number of zero bits below the lowest set one, for a nonzero `x`.
fn trailing_zeros(x: &[u64]) -> u32 {
    let i = x.iter().position(|&l| l != 0).unwrap_or(0);
    64 * i as u32 + x[i].trailing_zeros()
}

/// Remainders modulo a nonzero `m` by Knuth's algorithm D (TAOCP vol. 2, 4.3.1), one quotient
/// limb at a time from the top of the dividend: the remainder so far, below the divisor, and the
/// next limb make the next partial dividend, whose quotient is a single limb. So no room for the
/// dividend or the quotient is needed, and a dividend of any length (`R`, `R²`, a double-width
/// product) can be given limb by limb.
struct Divisor<const N: usize> {
    /// `m·2ˢ` in the low `n` limbs, the top one with its high bit set.
    d: [u64; N],
    /// The number of limbs of `m` up to its highest nonzero one.
    n: usize,
    /// The shift that normalises `m`.
    s: u32,
}

impl<const N: usize> Divisor<N> {
    /// The divisor `m`, which must not be zero.
    fn new(m: &[u64; N]) -> Self {
        let n = m.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        debug_assert!(n > 0, "the divisor must not be zero");
        let s = m[n - 1].leading_zeros();
        let mut d = [0; N];
        for i in 0..n {
            let low = if s > 0 && i > 0 {
                m[i - 1] >> (64 - s)
            } else {
                0
            };
            d[i] = m[i] << s | low;
        }
        Divisor { d, n, s }
    }

    /// `x mod m` for the `x` of `len` limbs that `limb(i)` gives, the low one at 0.
    ///
    /// The limbs of `x·2ˢ` go in from the top: `x·2ˢ mod m·2ˢ = 2ˢ·(x mod m)`, which is shifted
    /// back at the end.
    fn rem(&self, len: usize, limb: impl Fn(usize) -> u64) -> [u64; N] {
        let s = self.s;
        let mut r = [0; N];
        for i in (0..=len).rev() {
            let high = if i < len { limb(i) } else { 0 };
            let y = if s == 0 {
                high
            } else {
                let low = if i > 0 { limb(i - 1) } else { 0 };
                high << s | low >> (64 - s)
            };
            self.step(&mut r, y);
        }
        if s > 0 {
            for i in 0..self.n {
                let high = if i + 1 < self.n {
                    r[i + 1] << (64 - s)
                } else {
                    0
                };
                r[i] = r[i] >> s | high;
            }
        }
        r
    }

    /// `r ← (r·2⁶⁴ + t) mod d` for `r < d`: one step of algorithm D. The quotient limb is
    /// estimated from the top two limbs of the partial dividend and the top limb of `d`, and
    /// corrected by the next limbs of each (step D3), after which it is right or one too large
    /// (Knuth's Theorem B); step D6 adds `d` back in the second case.
    fn step(&self, r: &mut [u64; N], t: u64) {
        let (n, d) = (self.n, &self.d);
        if n == 1 {
            let u = u128::from(r[0]) << 64 | u128::from(t);
            r[0] = (u % u128::from(d[0])) as u64;
            return;
        }
        // The partial dividend is (r[n−1], …, r[0], t), below d·2⁶⁴.
        let top = u128::from(r[n - 1]) << 64 | u128::from(r[n - 2]);
        let third = if n > 2 { r[n - 3] } else { t };
        let dt = u128::from(d[n - 1]);
        let mut q = top / dt;
        let mut rh = top % dt;
        while q >> 64 != 0 || q * u128::from(d[n - 2]) > (rh << 64 | u128::from(third)) {
            q -= 1;
            rh += dt;
            if rh >> 64 != 0 {
                break;
            }
        }
        let q = q as u64;
        // (r, t) − q·d, over n + 1 limbs; the low n are the remainder.
        let mut out = [0u64; N];
        let mut carry = 0u64;
        let mut borrow = false;
        for j in 0..n {
            let p = u128::from(q) * u128::from(d[j]) + u128::from(carry);
            carry = (p >> 64) as u64;
            let u = if j == 0 { t } else { r[j - 1] };
            let (x, b1) = u.overflowing_sub(p as u64);
            let (x, b2) = x.overflowing_sub(u64::from(borrow));
            out[j] = x;
            borrow = b1 | b2;
        }
        let (x, b1) = r[n - 1].overflowing_sub(carry);
        let (_, b2) = x.overflowing_sub(u64::from(borrow));
        if b1 | b2 {
            // The quotient was one too large: the difference is negative, at least −d.
            let mut c = false;
            for j in 0..n {
                let (x, c1) = out[j].overflowing_add(d[j]);
                let (x, c2) = x.overflowing_add(u64::from(c));
                out[j] = x;
                c = c1 | c2;
            }
        }
        *r = out;
    }
}

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
//! conditional subtraction at the end. From 11 limbs on, a product or a square is that of
//! [`Number`]'s schoolbook kernels (assembly on aarch64 and x86-64), the square with its cross
//! products once, and a separate reduction by its REDC kernel, two limbs of the quotient at a
//! time. Powers are taken by the sliding-window method (Menezes, van Oorschot and Vanstone,
//! *Handbook of Applied Cryptography*, 1996, Algorithm 14.85), with a window sized to the
//! exponent; inverses by the binary GCD in Pornin's batches of 31 steps on approximations
//! (*Optimized binary GCD for modular inversion*, IACR ePrint 2020/972). From 11 limbs on, powers
//! and inverses go through [`Number`]'s, faster there even with the conversions.
//!
//! `R mod m` and `R² mod m` are remainders of Knuth's algorithm D (*The Art of Computer
//! Programming*, vol. 2, 4.3.1), taken here one quotient limb at a time from the top of the
//! dividend, which needs no room for the dividend, each by Möller and Granlund's 3-by-2 division
//! with the divisor's reciprocal: the same division reduces the double-width products of
//! [`Uint::mul_mod`]. An even modulus `2ᵏ·q` is split: modulo the odd `q` in
//! Montgomery form, modulo `2ᵏ` by wrapping arithmetic, and the two remainders put together by the
//! Chinese remainder theorem in Garner's form, with `q⁻¹ mod 2ᵏ` from Newton's iteration.
//!
//! Nothing here is constant-time: the time of a power depends on the bits of the exponent, the
//! final subtraction of a product is taken or not depending on the values, and the inverse takes
//! as many steps as its values need. None of it is meant for secret values.

use super::Uint;
use super::limbs::{
    compare, div_2by1, div_normalized, reciprocal_3by2, reciprocal_word, shift_in, shift_out,
};
use crate::math::numbers::{Number, basecase, redc_inverse, redc_limbs};
use core::mem::MaybeUninit;

/// The arithmetic modulo an odd `m > 1` of `N` limbs in Montgomery form, `R = 2^(64·N)`.
///
/// A number `x` is represented by its form `x·R mod m`, a [`Uint<N>`] below `m`:
/// [`to_montgomery`](Self::to_montgomery) and [`from_montgomery`](Self::from_montgomery) convert,
/// and [`mul`](Self::mul), [`square`](Self::square), [`add`](Self::add), [`sub`](Self::sub),
/// [`neg`](Self::neg), [`pow`](Self::pow) and [`inverse`](Self::inverse) work on forms. Every
/// result is below `m`; an argument may be any `Uint<N>` and stands for its residue modulo `m`:
/// one at or above `m` is reduced first.
///
/// [`new`](Self::new) computes `−m⁻¹ mod 2¹²⁸`, `R mod m` and `R² mod m` once; then a product
/// costs `2·N² + N` products of limbs (CIOS, or from 11 limbs on the kernels of [`Number`]). The
/// modulus may be as large as `R − 1`: the sum in the reduction is kept with its extra carry bit.
/// Nothing is constant-time.
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
    /// `−1/m mod 2¹²⁸`, the low limb first: the low limb is the factor by which a multiple of `m`
    /// clears the low limb of a sum, and both are the factor of the kernel's REDC.
    minv: [u64; 2],
    /// `R mod m`, the form of 1.
    one: Uint<N>,
    /// `R² mod m`, by which one product takes a number to its form.
    r2: Uint<N>,
}

impl<const N: usize> Montgomery<N> {
    /// The arithmetic modulo `m`.
    ///
    /// It computes `−m⁻¹ mod 2¹²⁸` by Newton's iteration and `R mod m` and `R² mod m` by long
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
            minv: redc_inverse(limbs[0], if N > 1 { limbs[1 % N] } else { 0 }),
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
        self.redc_square(&a)
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
        if N >= BIG {
            // The number the form stands for, its power by `Number`, and back to a form.
            let y = Number::from(&self.from_montgomery(x)).pow_mod(
                &super::number::number_of_limbs(e),
                &Number::from(&self.modulus),
            );
            return self.to_montgomery(&by_number(y));
        }
        let x = self.reduce(x);
        power_of(&x, &self.one, e, &|a, b| self.redc_mul(a, b), &|a| {
            self.redc_square(a)
        })
    }

    /// The form of the inverse of what `x` stands for, `x⁻¹·R mod m`, or `None` when
    /// `gcd(x, m) ≠ 1` (0 included). `m` need not be prime.
    ///
    /// By the binary GCD on the residue in Pornin's batches of 31 steps (*Optimized binary GCD for
    /// modular inversion*, IACR ePrint 2020/972), with a stop at a common divisor, then two
    /// products by `R² mod m`: `O(N²)` operations on limbs. One limb takes the extended Euclidean
    /// algorithm with the hardware's division; from 11 limbs on, [`Number::mod_inverse`] is used.
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
        if N >= BIG {
            let y =
                Number::from(&self.from_montgomery(x)).mod_inverse(&Number::from(&self.modulus))?;
            return Some(self.to_montgomery(&by_number(y)));
        }
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

    /// `a·b·R⁻¹ mod m`, below `m`, for `a·b < m·R`: from [`KERNEL`] limbs on the product by the
    /// schoolbook kernel of [`Number`] and then REDC by its kernel
    /// ([`redc_kernel`](Self::redc_kernel)), below by CIOS ([`redc_rows`](Self::redc_rows)).
    #[inline]
    fn redc_mul(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
        if N >= KERNEL {
            let mut t = MaybeUninit::<[[u64; N]; 2]>::uninit();
            let t = t.as_mut_ptr().cast::<u64>();
            // SAFETY: `t` has room for the `2N` limbs the kernel writes, apart from `a` and `b`,
            // which hold `N ≥ 1` limbs each.
            unsafe { basecase::mul(t, a.as_limbs().as_ptr(), b.as_limbs().as_ptr(), N) };
            // SAFETY: the kernel wrote all `2N` limbs at `t`: the product, below `m·R`.
            unsafe { self.redc_kernel(t) }
        } else {
            self.redc_rows(a, b)
        }
    }

    /// The form of `a²` for `a < m`: from [`KERNEL`] limbs on the full square by the kernel of
    /// [`Number`], its `N(N − 1)/2` cross products once and doubled, then REDC by its kernel
    /// ([`redc_kernel`](Self::redc_kernel)); below by the interleaved rows of
    /// [`redc_rows`](Self::redc_rows), as fast there.
    #[inline]
    fn redc_square(&self, a: &Uint<N>) -> Uint<N> {
        if N >= KERNEL {
            let mut t = MaybeUninit::<[[u64; N]; 2]>::uninit();
            let t = t.as_mut_ptr().cast::<u64>();
            // SAFETY: `t` has room for the `2N` limbs the kernel writes, apart from `a`, which
            // holds `N ≥ 1` limbs.
            unsafe { basecase::sqr(t, a.as_limbs().as_ptr(), N) };
            // SAFETY: the kernel wrote all `2N` limbs at `t`: the square, below `m·R`.
            unsafe { self.redc_kernel(t) }
        } else {
            self.redc_rows(a, a)
        }
    }

    /// REDC of the `2N` limbs at `t`, below `m·R`, which it clobbers: `t·R⁻¹ mod m`, below `m`.
    /// The kernel adds the multiples of `m` that clear the low half, two limbs at a time with the
    /// factor `−m⁻¹ mod 2¹²⁸`, and leaves the high half, below `2m`, as `N` limbs and a carry; one
    /// conditional subtraction makes it a residue.
    ///
    /// # Safety
    ///
    /// `t` must point to `2N` initialised limbs, valid for reads and writes.
    #[inline]
    unsafe fn redc_kernel(&self, t: *mut u64) -> Uint<N> {
        let mut r = MaybeUninit::<[u64; N]>::uninit();
        // SAFETY: `r` has room for the `N ≥ 1` limbs the kernel writes, apart from `t` and from
        // the modulus, which holds `N` limbs; `minv` is `−m⁻¹ mod 2¹²⁸` and `t < m·R`, as the
        // caller guarantees.
        let (r, over) = unsafe {
            let over = redc_limbs(
                r.as_mut_ptr().cast(),
                t,
                self.modulus.as_limbs().as_ptr(),
                N,
                self.minv,
            );
            (Uint::from_limbs(r.assume_init()), over)
        };
        // r + over·R < 2m: at least m when the carry is set (R > m) or the subtraction does not
        // borrow.
        let (d, borrow) = r.overflowing_sub(self.modulus);
        if over || !borrow { d } else { r }
    }

    /// `a·b·R⁻¹ mod m`, below `m`, for `a·b < m·R`, by CIOS (Koç, Acar and Kaliski): for each limb
    /// of `b`, add `a·bᵢ`, then the multiple `q·m` that clears the low limb, and shift a limb out.
    ///
    /// The running sum `t` stays below `a + m < 2·R` between the rows: it is kept in `N` limbs
    /// and a carry `top`, which is what makes a modulus near `R` work. At the end `t < 2·m`, and
    /// one conditional subtraction leaves it below `m`.
    #[inline]
    fn redc_rows(&self, a: &Uint<N>, b: &Uint<N>) -> Uint<N> {
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
            let q = t[0].wrapping_mul(self.minv[0]);
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
        if N >= BIG {
            return by_number(Number::from(self).pow_mod(&Number::from(exp), &Number::from(m)));
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

    /// `self · rhs mod m`, below `m`, for any nonzero `m`: the double-width product (from 11
    /// limbs on by the schoolbook kernel of [`Number`]) reduced by long division (Knuth's
    /// algorithm D, with Möller and Granlund's reciprocal of the divisor's top limbs, computed
    /// once a call), `O(N²)` operations on limbs. For operands below `m` the division takes `N`
    /// quotient limbs.
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
        let p = product(self, rhs);
        let p = p.as_flattened();
        Uint::from_limbs(Divisor::new(m.as_limbs()).rem(2 * N, |i| p[i]))
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
    /// Modulo an odd `m` by the binary GCD in Pornin's batches of 31 steps (*Optimized binary GCD
    /// for modular inversion*, IACR ePrint 2020/972); modulo an even `2ᵏ·q`, the inverse modulo
    /// `2ᵏ` by Newton's iteration and the one modulo `q` by the binary GCD, put together by the
    /// Chinese remainder theorem. `O(N²)` operations on limbs for an odd `m`; `O(N²·log N)` for an
    /// even one, whose Newton iterations modulo `2^(64·N)` take `⌈log₂ N⌉` steps of `O(N²)` each.
    /// One limb, any modulus, takes the extended Euclidean algorithm with the hardware's division;
    /// from 11 limbs on, [`Number::mod_inverse`] is used.
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
        if N >= BIG {
            return Number::from(self)
                .mod_inverse(&Number::from(m))
                .map(by_number);
        }
        if N == 1 {
            // Any modulus of one limb, odd or even.
            let (x, m) = (self.as_limbs()[0], m.as_limbs()[0]);
            return euclid_inverse(x % m, m).map(Uint::from_u64);
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

/// The `2N` limbs of `a·b`, the low half first: from [`KERNEL`] limbs on by the schoolbook kernel
/// of [`Number`], below by [`Uint::widening_mul`].
#[inline]
fn product<const N: usize>(a: &Uint<N>, b: &Uint<N>) -> [[u64; N]; 2] {
    if N >= KERNEL {
        let mut t = MaybeUninit::<[[u64; N]; 2]>::uninit();
        // SAFETY: `t` has room for the `2N` limbs the kernel writes, apart from `a` and `b`, which
        // hold `N ≥ 1` limbs each; it writes all of them.
        unsafe {
            basecase::mul(
                t.as_mut_ptr().cast(),
                a.as_limbs().as_ptr(),
                b.as_limbs().as_ptr(),
                N,
            );
            t.assume_init()
        }
    } else {
        let (low, high) = a.widening_mul(*b);
        [low.into_limbs(), high.into_limbs()]
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
/// A result of `Number`'s modular arithmetic, below the modulus, so of at most `N` limbs.
fn by_number<const N: usize>(x: Number) -> Uint<N> {
    match Uint::try_from(&x) {
        Ok(v) => v,
        Err(_) => unreachable!("a residue below a modulus of N limbs has at most N limbs"),
    }
}

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

/// `1/x mod m` for an odd `m > 1` and `x < m`, or `None` when `gcd(x, m) ≠ 1`: the binary GCD in
/// the optimised form of Pornin ("Optimized binary GCD for modular inversion", IACR ePrint
/// 2020/972, Algorithm 2), not constant-time.
///
/// The binary GCD keeps `a ≡ u·x` and `b ≡ v·x (mod m)` from `a = x, b = m, u = 1, v = 0`, `b`
/// odd: an even `a` is halved, an odd one, if below `b`, swapped with it, then `a ← (a − b)/2`,
/// with `u` following modulo `m`; when `a` reaches 0, `b` is the GCD, and for `b = 1`, `v` the
/// inverse. Here 31 of those steps at a time run on 64-bit approximations of `a` and `b` — their
/// low 31 bits, which decide every parity exactly, and their top 33 bits at a common position,
/// which decide the comparisons nearly always — and record the steps as factors `f, g`,
/// `|f|, |g| ≤ 2³¹`, with `2³¹·a' = f₀·a + g₀·b`, `2³¹·b' = f₁·a + g₁·b`; those are then applied to
/// the full `a` and `b` (negated where a wrong comparison made one negative) and to `u` and `v`
/// modulo `m`, divided by `2³¹` there as in Montgomery's reduction. Below 65 bits the
/// approximations are exact. Each batch shortens `a·b` by about 31 bits, so about `2·64·N/31`
/// batches of `O(N)` products.
fn binary_inverse<const N: usize>(x: &Uint<N>, m: &Uint<N>) -> Option<Uint<N>> {
    if x.is_zero() {
        return None;
    }
    let ml = m.as_limbs();
    if N == 1 {
        return euclid_inverse(x.as_limbs()[0], ml[0]).map(Uint::from_u64);
    }
    // −1/m mod 2³¹.
    let minv = inverse_limb(ml[0]).wrapping_neg() & LOW31;
    let (mut a, mut b) = (*x.as_limbs(), *ml);
    let mut u = [0u64; N];
    u[0] = 1;
    let mut v = [0u64; N];
    // Each batch shortens a·b by about 31 bits; the cap only guards the loop.
    let mut batches = 2 * 64 * N / 31 + 8;
    while a.iter().any(|&l| l != 0) && batches > 0 {
        batches -= 1;
        let n = bit_length(&a).max(bit_length(&b)).max(64);
        let (mut f0, mut g0, mut f1, mut g1) = steps(approximation(&a, n), approximation(&b, n));
        let (na, negative_a) = combine(&a, &b, f0, g0);
        let (nb, negative_b) = combine(&a, &b, f1, g1);
        (a, b) = (na, nb);
        if negative_a {
            (f0, g0) = (-f0, -g0);
        }
        if negative_b {
            (f1, g1) = (-f1, -g1);
        }
        let nu = combine_mod(&u, &v, f0, g0, ml, minv);
        let nv = combine_mod(&u, &v, f1, g1, ml, minv);
        (u, v) = (nu, nv);
    }
    debug_assert!(batches > 0, "the binary GCD did not converge");
    let mut one = [0u64; N];
    one[0] = 1;
    (b == one).then(|| Uint::from_limbs(v))
}

/// `1/x mod m` for one limb, `m > 1` (odd or even) and `x < m`, or `None` when `gcd(x, m) ≠ 1`: the
/// extended Euclidean algorithm with the hardware's 64-bit division, on the remainders
/// `r₋₁ = m, r₀ = x, rᵢ₊₁ = rᵢ₋₁ − qᵢ·rᵢ` and the cofactors of `x`, `t₋₁ = 0, t₀ = 1,
/// tᵢ₊₁ = tᵢ₋₁ − qᵢ·tᵢ`, whose signs alternate, so that their absolute values `|tᵢ₊₁| =
/// |tᵢ₋₁| + qᵢ·|tᵢ|` grow, at most `m`, and are kept without signs. At the last nonzero
/// remainder `r_k`, the GCD, `t_k·x ≡ r_k (mod m)`, and the inverse is `|t_k|` or `m − |t_k|` by
/// the parity of `k`. On one limb its divisions are faster than the batches of
/// [`binary_inverse`] and the halvings of the binary algorithm.
fn euclid_inverse(x: u64, m: u64) -> Option<u64> {
    let (mut r0, mut r1) = (m, x);
    let (mut t0, mut t1) = (0u64, 1u64);
    // Whether the cofactor t1 is the one of an even index, positive.
    let mut even = true;
    while r1 != 0 {
        let q = r0 / r1;
        (r0, r1) = (r1, r0 - q * r1);
        (t0, t1) = (t1, t0 + q * t1);
        even = !even;
    }
    // Now t0 is the cofactor of r0, of the parity opposite to t1's.
    (r0 == 1).then_some(if even { m - t0 } else { t0 })
}

/// The low 31 bits.
const LOW31: u64 = (1 << 31) - 1;

/// The number of bits up to the highest set one.
fn bit_length<const N: usize>(x: &[u64; N]) -> u32 {
    match x.iter().rposition(|&l| l != 0) {
        Some(i) => 64 * i as u32 + 64 - x[i].leading_zeros(),
        None => 0,
    }
}

/// The approximation of `x` for a batch at the length `n ≥ 64`: the bits `n − 33` to `n − 1` of
/// `x` above its low 31 bits; `x` itself when `n = 64`.
fn approximation<const N: usize>(x: &[u64; N], n: u32) -> u64 {
    if n == 64 {
        return x[0];
    }
    let s = n - 33;
    let (i, sh) = ((s / 64) as usize, s % 64);
    let mut w = x[i] >> sh;
    if sh > 0 && i + 1 < N {
        w |= x[i + 1] << (64 - sh);
    }
    (w & ((1 << 33) - 1)) << 31 | (x[0] & LOW31)
}

/// Thirty-one steps of the binary GCD on the approximations `a` and `b` (`b` odd), and their
/// factors `(f₀, g₀, f₁, g₁)`: halve an even `a`; else, if `a < b`, swap them (and their factors),
/// then `a ← (a − b)/2`; the factors of `b` double at each step, so that both stay at the scale
/// 2³¹ at the end. Without branches on the values, which a predictor cannot learn.
#[inline]
fn steps(mut a: u64, mut b: u64) -> (i64, i64, i64, i64) {
    let (mut f0, mut g0, mut f1, mut g1) = (1i64, 0i64, 0i64, 1i64);
    for _ in 0..31 {
        let odd = (a & 1).wrapping_neg();
        let swap = u64::from(a < b).wrapping_neg() & odd;
        let d = (a ^ b) & swap;
        a ^= d;
        b ^= d;
        let s = swap as i64;
        let df = (f0 ^ f1) & s;
        f0 ^= df;
        f1 ^= df;
        let dg = (g0 ^ g1) & s;
        g0 ^= dg;
        g1 ^= dg;
        let o = odd as i64;
        a = a.wrapping_sub(b & odd);
        f0 -= f1 & o;
        g0 -= g1 & o;
        a >>= 1;
        f1 <<= 1;
        g1 <<= 1;
    }
    (f0, g0, f1, g1)
}

/// `|(a·f + b·g)/2³¹|` and whether it is negative, for factors of at most `2³¹` in size, which
/// make the sum divisible by `2³¹` and keep it within `N` limbs.
fn combine<const N: usize>(a: &[u64; N], b: &[u64; N], f: i64, g: i64) -> ([u64; N], bool) {
    let mut r = [0u64; N];
    let mut carry = 0i128;
    for j in 0..N {
        let t = i128::from(a[j]) * i128::from(f) + i128::from(b[j]) * i128::from(g) + carry;
        r[j] = t as u64;
        carry = t >> 64;
    }
    let top = carry as i64;
    shift31(&mut r, top);
    let negative = top < 0;
    if negative {
        negate(&mut r);
    }
    (r, negative)
}

/// `(u·f + v·g)/2³¹ mod m` for `u, v < m` and factors of at most `2³¹` in size: the multiple
/// `k·m` with `k = t·(−1/m) mod 2³¹` makes the sum `t` divisible by `2³¹`; the quotient is then
/// within `3m` of 0 and brought into `[0, m)`.
fn combine_mod<const N: usize>(
    u: &[u64; N],
    v: &[u64; N],
    f: i64,
    g: i64,
    m: &[u64; N],
    minv: u64,
) -> [u64; N] {
    let mut r = [0u64; N];
    let mut carry = 0i128;
    for j in 0..N {
        let t = i128::from(u[j]) * i128::from(f) + i128::from(v[j]) * i128::from(g) + carry;
        r[j] = t as u64;
        carry = t >> 64;
    }
    let k = (r[0].wrapping_mul(minv)) & LOW31;
    let mut c = 0u128;
    for j in 0..N {
        let t = u128::from(r[j]) + u128::from(k) * u128::from(m[j]) + c;
        r[j] = t as u64;
        c = t >> 64;
    }
    let mut top = (carry as i64).wrapping_add(c as i64);
    shift31(&mut r, top);
    top >>= 31;
    // Into [0, m): at most three corrections either way.
    while top < 0 {
        top += i64::from(add_into(&mut r, m));
    }
    while top > 0 || !less(&r, m) {
        top -= i64::from(sub_from(&mut r, m));
    }
    r
}

/// Shifts the `N` limbs of `r`, with the signed limb `top` above them, right by 31 bits.
fn shift31<const N: usize>(r: &mut [u64; N], top: i64) {
    for j in 0..N {
        let next = if j + 1 < N { r[j + 1] } else { top as u64 };
        r[j] = (r[j] >> 31) | (next << 33);
    }
}

/// `−r` in two's complement over the `N` limbs.
fn negate<const N: usize>(r: &mut [u64; N]) {
    let mut c = true;
    for x in r.iter_mut() {
        let (y, o) = (!*x).overflowing_add(u64::from(c));
        *x = y;
        c = o;
    }
}

/// `r += m` over the `N` limbs; the carry out.
fn add_into<const N: usize>(r: &mut [u64; N], m: &[u64; N]) -> bool {
    let mut c = false;
    for (x, &y) in r.iter_mut().zip(m) {
        let (s, c1) = x.overflowing_add(y);
        let (s, c2) = s.overflowing_add(u64::from(c));
        *x = s;
        c = c1 | c2;
    }
    c
}

/// `r −= m` over the `N` limbs; the borrow out.
fn sub_from<const N: usize>(r: &mut [u64; N], m: &[u64; N]) -> bool {
    let mut b = false;
    for (x, &y) in r.iter_mut().zip(m) {
        let (s, b1) = x.overflowing_sub(y);
        let (s, b2) = s.overflowing_sub(u64::from(b));
        *x = s;
        b = b1 | b2;
    }
    b
}

/// Whether `r < m`.
fn less<const N: usize>(r: &[u64; N], m: &[u64; N]) -> bool {
    for j in (0..N).rev() {
        if r[j] != m[j] {
            return r[j] < m[j];
        }
    }
    false
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
#[inline]
fn rem<const N: usize>(x: &Uint<N>, m: &Uint<N>) -> Uint<N> {
    if x < m { *x } else { rem_by_division(x, m) }
}

/// [`rem`] of an `x` at or above `m`, out of line so that the common case stays small.
#[inline(never)]
fn rem_by_division<const N: usize>(x: &Uint<N>, m: &Uint<N>) -> Uint<N> {
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

/// From this many limbs on, `pow_mod` and `mod_inverse` (and `Montgomery::pow` and
/// `Montgomery::inverse`) go through [`Number`]'s, whose products and reductions are on assembly
/// kernels and whose GCD is Lehmer's and subquadratic: faster there even with the conversions.
// Measured on an Apple M4 Max (sweep 1 to 32 limbs; moduli odd of the full width, odd with the top
// limb all ones and even with 1 to 32 low zero bits; exponents of `N` limbs, of one limb and 65537;
// random operands below the modulus): at 11 limbs `Number`'s take 0.43 to 0.59 of the time of these
// for powers and for inverses modulo an odd number, and 0.70 to 0.86 for inverses modulo an even
// one. At 10 limbs `Number`'s inverse modulo an even number takes 0.95 to 1.16 times as long as
// this one, which sees at once that an even number has none (up to 1.8 times below), though its
// powers by long exponents and its inverses modulo an odd number are faster from 2 limbs on.
const BIG: usize = 11;

/// From this many limbs on, Montgomery products and squares are the product or the square by the
/// schoolbook kernels of [`Number`] (assembly on aarch64 and x86-64) and a separate REDC by its
/// kernel, rather than the interleaved rows of CIOS, and the double-width product of
/// [`Uint::mul_mod`] is the schoolbook kernel's.
// Measured on an Apple M4 Max (sweep 6 to 24 limbs; `Montgomery::{mul, square, pow}`, `pow_mod`
// and `mul_mod`, odd moduli of the full width and with the top limb all ones, exponents of `N`
// limbs and of one): at 11 limbs the kernels take 0.97 to 0.99 of the time of the rows for
// products, 0.91 to 0.93 for squares and 0.87 to 0.91 for `mul_mod`, at 12 limbs 0.84 to 0.88. At
// 10 limbs they would take 0.98 to 0.99 for products and 0.91 to 0.93 for squares, but 1.02 to
// 1.035 times as long for powers by exponents of one limb (from 11 limbs on, powers go to
// `Number`'s); at 9 limbs 1.05 times as long for products and 1.11 for powers.
const KERNEL: usize = 11;

/// Remainders modulo a nonzero `m` by Knuth's algorithm D (TAOCP vol. 2, 4.3.1), with the
/// normalised divisor and its reciprocal computed once: the dividend, of up to `2N + 1` limbs
/// (`R`, `R²`, a double-width product), is shifted as the divisor into a buffer on the stack, and
/// [`div_normalized`] reduces it there, its loops unrolled for a divisor of `N` limbs.
struct Divisor<const N: usize> {
    /// `m·2ˢ` in the low `n` limbs, the top one with its high bit set.
    d: [u64; N],
    /// The number of limbs of `m` up to its highest nonzero one.
    n: usize,
    /// The shift that normalises `m`.
    s: u32,
    /// The reciprocal of the top limb (`n = 1`) or of the top two limbs (`n ≥ 2`) of `d`, by which
    /// each quotient limb is taken without a division (Möller and Granlund).
    v: u64,
}

impl<const N: usize> Divisor<N> {
    /// The divisor `m`, which must not be zero.
    #[inline]
    fn new(m: &[u64; N]) -> Self {
        let n = m.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        debug_assert!(n > 0, "the divisor must not be zero");
        let s = m[n - 1].leading_zeros();
        let mut d = [0; N];
        for (i, di) in d.iter_mut().enumerate().take(n) {
            *di = shift_in(m, i, s);
        }
        let v = if n == 1 {
            reciprocal_word(d[0])
        } else {
            reciprocal_3by2(d[n - 1], d[n - 2])
        };
        Divisor { d, n, s, v }
    }

    /// `x mod m` for the `x` of `len ≤ 2N + 1` limbs that `limb(i)` gives, the low one at 0.
    ///
    /// `x·2ˢ mod m·2ˢ = 2ˢ·(x mod m)`: the remainder of the shifted `x` is shifted back.
    #[inline]
    fn rem(&self, len: usize, limb: impl Fn(usize) -> u64) -> [u64; N] {
        let (s, n) = (self.s, self.n);
        let mut r = [0u64; N];
        if len < n {
            // Fewer limbs than m, so below it.
            for (i, ri) in r.iter_mut().enumerate().take(len) {
                *ri = limb(i);
            }
            return r;
        }
        if n == 1 {
            // From the top limb of x·2ˢ, below 2ˢ ≤ d, one 2-by-1 division a limb; where that
            // limb is 0 and the next below d, as for a product of two residues, from the next.
            let (d, v) = (self.d[0], self.v);
            let mut high = limb(len - 1);
            let mut rest = (high >> 1) >> (63 - s);
            let mut i = len;
            let low = if len > 1 { limb(len - 2) } else { 0 };
            let next = high << s | (low >> 1) >> (63 - s);
            if rest == 0 && next < d {
                (rest, high, i) = (next, low, len - 1);
            }
            while i > 0 {
                i -= 1;
                let low = if i > 0 { limb(i - 1) } else { 0 };
                rest = div_2by1(rest, high << s | (low >> 1) >> (63 - s), d, v).1;
                high = low;
            }
            r[0] = rest >> s;
            return r;
        }
        // n ≥ 2, so N ≥ 2 and the 3N limbs of the buffer hold the len + 1 ≤ 2N + 2 of x·2ˢ.
        let mut w = [[0u64; N]; 3];
        let u = w.as_flattened_mut();
        let mut low = 0u64;
        for (i, ui) in u.iter_mut().enumerate().take(len) {
            let high = limb(i);
            *ui = high << s | (low >> 1) >> (63 - s);
            low = high;
        }
        u[len] = (low >> 1) >> (63 - s);
        let d = &self.d[..n];
        // Where the top limb of x·2ˢ is 0 and the n below it are below d, as for a product of two
        // residues, the division starts a limb lower, with one quotient limb less.
        let mut len = len;
        if u[len] == 0 && compare(&u[len - n..len], d).is_lt() {
            len -= 1;
        }
        if len < n {
            // x·2ˢ < d: x is its own remainder.
        } else if n == N {
            div_normalized::<N>(u, len, d, self.v, &mut []);
        } else {
            div_normalized::<0>(u, len, d, self.v, &mut []);
        }
        u[n] = 0;
        for (i, ri) in r.iter_mut().enumerate().take(n) {
            *ri = shift_out(u, i, s);
        }
        r
    }
}

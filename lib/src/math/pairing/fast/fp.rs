//! `F_p` of a pairing-friendly curve in `N` limbs, in Montgomery's form with `R = 2^(64 N)`, the
//! prime and its constants known at compile time.
//!
//! The products are Montgomery's CIOS loop without the carry word that a full-width prime needs:
//! the primes here leave at least two bits free at the top (254 bits in 4 limbs, 381 in 6), so
//! that every intermediate sum of the loop fits `N` limbs and its result is below `2p` (Gueron and
//! Krasnov's and the gnark authors' observation on moduli with a spare bit). Sums are reduced
//! below `p` by one conditional subtraction, so every element is held in `[0, p)` and equality is
//! that of the limbs.

use super::Engine;
use core::marker::PhantomData;

#[cfg(all(target_arch = "aarch64", not(miri)))]
use super::aarch64 as k;

/// The limbs of `N` as limbs of `M`, for `N = M` (the kernels' dispatch on the engine's size).
///
/// Safe for any `N` and `M`: the conversion checks the length, which is a constant, so that
/// under `N == M` (where every call is) the check folds away, and a call with `N ≠ M` would panic.
/// The dispatch instantiates the other size in its dead branches, which rules out a compile-time
/// assertion here.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline(always)]
fn view<const N: usize, const M: usize>(x: &[u64; N]) -> &[u64; M] {
    match x.as_slice().try_into() {
        Ok(v) => v,
        Err(_) => unreachable!("a view of {N} limbs as {M}"),
    }
}

/// The first `M ≤ 12` limbs of a double-width value; as [`view`], safe for any `M` (a panic
/// beyond 12), the bounds constant and folded away.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline(always)]
fn low<const M: usize>(x: &Wide) -> &[u64; M] {
    match x[..M].try_into() {
        Ok(v) => v,
        Err(_) => unreachable!("a slice of {M} limbs has {M}"),
    }
}

/// A double-width value of `M ≤ 12` limbs, the rest zero.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline(always)]
fn widen<const M: usize>(x: [u64; M]) -> Wide {
    let mut t = [0u64; 12];
    t[..M].copy_from_slice(&x);
    t
}

/// An element of `F_p` for the engine `E`, in Montgomery's form, below `p`.
pub(crate) struct Fp<E, const N: usize>(pub(crate) [u64; N], PhantomData<fn() -> E>);

impl<E, const N: usize> Clone for Fp<E, N> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}

impl<E, const N: usize> Copy for Fp<E, N> {}

impl<E, const N: usize> PartialEq for Fp<E, N> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<E, const N: usize> Eq for Fp<E, N> {}

/// `a + b c + carry`, its low word, the high word in `carry`.
#[inline(always)]
fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let t = u128::from(a) + u128::from(b) * u128::from(c) + u128::from(*carry);
    *carry = (t >> 64) as u64;
    t as u64
}

/// `a + b + carry` with the carry out.
#[inline(always)]
fn adc(a: u64, b: u64, carry: &mut bool) -> u64 {
    let (s, c1) = a.overflowing_add(b);
    let (s, c2) = s.overflowing_add(u64::from(*carry));
    *carry = c1 | c2;
    s
}

/// `a − b − borrow` with the borrow out.
#[inline(always)]
fn sbb(a: u64, b: u64, borrow: &mut bool) -> u64 {
    let (d, b1) = a.overflowing_sub(b);
    let (d, b2) = d.overflowing_sub(u64::from(*borrow));
    *borrow = b1 | b2;
    d
}

/// A double-width value: `2N` limbs of 12, the rest zero.
pub(crate) type Wide = [u64; 12];

/// The double-width product of two values of `N` limbs.
#[inline(always)]
pub(crate) fn mul_wide<const N: usize>(a: &[u64; N], b: &[u64; N]) -> Wide {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if N == 6 {
            return k::mul_wide6(view(a), view(b));
        }
        if N == 4 {
            return widen(k::mul_wide4(view(a), view(b)));
        }
    }
    let mut t = [0u64; 12];
    for i in 0..N {
        let mut c = 0u64;
        for j in 0..N {
            t[i + j] = mac(t[i + j], a[i], b[j], &mut c);
        }
        t[i + N] = c;
    }
    t
}

/// The double-width square: the cross products once, doubled, then the squares of the limbs.
#[inline(always)]
pub(crate) fn sqr_wide<const N: usize>(a: &[u64; N]) -> Wide {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if N == 6 {
            return k::sqr_wide6(view(a));
        }
        if N == 4 {
            return widen(k::sqr_wide4(view(a)));
        }
    }
    let mut t = [0u64; 12];
    for i in 0..N {
        let mut c = 0u64;
        for j in i + 1..N {
            t[i + j] = mac(t[i + j], a[i], a[j], &mut c);
        }
        t[i + N] = c;
    }
    // Double.
    let mut top = 0u64;
    for v in t.iter_mut().take(2 * N) {
        let next = *v >> 63;
        *v = (*v << 1) | top;
        top = next;
    }
    // Add the squares of the limbs.
    let mut carry = false;
    for i in 0..N {
        let sq = u128::from(a[i]) * u128::from(a[i]);
        t[2 * i] = adc(t[2 * i], sq as u64, &mut carry);
        t[2 * i + 1] = adc(t[2 * i + 1], (sq >> 64) as u64, &mut carry);
    }
    t
}

/// Double-width `a + b`.
#[inline(always)]
pub(crate) fn wide_add<const N: usize>(a: &Wide, b: &Wide) -> Wide {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if N == 6 {
            return k::wide_add6(a, b);
        }
        if N == 4 {
            return widen(k::wide_add4(low(a), low(b)));
        }
    }
    let mut s = [0u64; 12];
    let mut carry = false;
    for i in 0..2 * N {
        s[i] = adc(a[i], b[i], &mut carry);
    }
    s
}

/// Double-width `a − b` for `a ≥ b`.
#[inline(always)]
pub(crate) fn wide_sub_plain<const N: usize>(a: &Wide, b: &Wide) -> Wide {
    #[cfg(all(target_arch = "aarch64", not(miri)))]
    {
        if N == 6 {
            return k::wide_sub_plain6(a, b);
        }
        if N == 4 {
            return widen(k::wide_sub_plain4(low(a), low(b)));
        }
    }
    let mut d = [0u64; 12];
    let mut borrow = false;
    for i in 0..2 * N {
        d[i] = sbb(a[i], b[i], &mut borrow);
    }
    d
}

impl<E: Engine<N>, const N: usize> Fp<E, N> {
    /// The element of these limbs, already in Montgomery's form and below `p`.
    #[inline(always)]
    pub(crate) const fn raw(l: [u64; N]) -> Self {
        Fp(l, PhantomData)
    }

    /// 0.
    #[inline(always)]
    pub(crate) const fn zero() -> Self {
        Fp([0; N], PhantomData)
    }

    /// 1.
    #[inline(always)]
    pub(crate) const fn one() -> Self {
        Fp(E::R, PhantomData)
    }

    /// Whether this is 0.
    #[inline(always)]
    pub(crate) fn is_zero(&self) -> bool {
        self.0.iter().fold(0, |a, &l| a | l) == 0
    }

    /// `x − p` when `x ≥ p`, else `x`, for `x < 2p`.
    #[inline(always)]
    fn reduce_once(x: [u64; N]) -> [u64; N] {
        let mut d = [0u64; N];
        let mut borrow = false;
        for i in 0..N {
            d[i] = sbb(x[i], E::P[i], &mut borrow);
        }
        // x < p exactly when the subtraction borrowed.
        let mask = 0u64.wrapping_sub(u64::from(borrow));
        let mut r = [0u64; N];
        for i in 0..N {
            r[i] = (x[i] & mask) | (d[i] & !mask);
        }
        r
    }

    /// `self + o`.
    #[inline(always)]
    pub(crate) fn add(&self, o: &Self) -> Self {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 6 {
                return Fp(
                    *view(&k::add6(view(&self.0), view(&o.0), view(&E::P))),
                    PhantomData,
                );
            }
            if N == 4 {
                return Fp(
                    *view(&k::add4(view(&self.0), view(&o.0), view(&E::P))),
                    PhantomData,
                );
            }
        }
        // Below 2p < 2^(64 N): no carry out.
        let mut s = [0u64; N];
        let mut carry = false;
        for i in 0..N {
            s[i] = adc(self.0[i], o.0[i], &mut carry);
        }
        Fp(Self::reduce_once(s), PhantomData)
    }

    /// `2 self`.
    #[inline(always)]
    pub(crate) fn double(&self) -> Self {
        self.add(self)
    }

    /// `self − o`.
    #[inline(always)]
    pub(crate) fn sub(&self, o: &Self) -> Self {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 6 {
                return Fp(
                    *view(&k::sub6(view(&self.0), view(&o.0), view(&E::P))),
                    PhantomData,
                );
            }
            if N == 4 {
                return Fp(
                    *view(&k::sub4(view(&self.0), view(&o.0), view(&E::P))),
                    PhantomData,
                );
            }
        }
        let mut d = [0u64; N];
        let mut borrow = false;
        for i in 0..N {
            d[i] = sbb(self.0[i], o.0[i], &mut borrow);
        }
        let mask = 0u64.wrapping_sub(u64::from(borrow));
        let mut carry = false;
        for (x, p) in d.iter_mut().zip(E::P.iter()) {
            *x = adc(*x, p & mask, &mut carry);
        }
        Fp(d, PhantomData)
    }

    /// `self / 2`: `self` plus `p` when odd, shifted right by one bit (in Montgomery's form
    /// too, as `(aR)/2 = (a/2)R`).
    #[inline(always)]
    pub(crate) fn half(&self) -> Self {
        let mask = 0u64.wrapping_sub(self.0[0] & 1);
        // Below 2p < 2^(64 N): no carry out.
        let mut s = [0u64; N];
        let mut carry = false;
        for i in 0..N {
            s[i] = adc(self.0[i], E::P[i] & mask, &mut carry);
        }
        let mut r = [0u64; N];
        for i in 0..N {
            let hi = if i + 1 < N { s[i + 1] << 63 } else { 0 };
            r[i] = (s[i] >> 1) | hi;
        }
        Fp(r, PhantomData)
    }

    /// `−self`.
    #[inline(always)]
    pub(crate) fn neg(&self) -> Self {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 6 {
                return Fp(*view(&k::neg6(view(&self.0), view(&E::P))), PhantomData);
            }
            if N == 4 {
                return Fp(*view(&k::neg4(view(&self.0), view(&E::P))), PhantomData);
            }
        }
        Self::zero().sub(self)
    }

    /// `self · o`, by Montgomery's CIOS loop without the extra carry word.
    #[inline(always)]
    pub(crate) fn mul(&self, o: &Self) -> Self {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 6 {
                let r = k::mont_mul6(view(&self.0), view(&o.0), view(&E::P), E::INV);
                return Fp(*view(&r), PhantomData);
            }
            if N == 4 {
                let r = k::mont_mul4(view(&self.0), view(&o.0), view(&E::P), E::INV);
                return Fp(*view(&r), PhantomData);
            }
        }
        let (a, b) = (&self.0, &o.0);
        let mut r = [0u64; N];
        for &bi in b.iter() {
            let mut c1 = 0u64;
            r[0] = mac(r[0], a[0], bi, &mut c1);
            let k = r[0].wrapping_mul(E::INV);
            let mut c2 = 0u64;
            // r[0] + k p[0] is 0 modulo 2^64: only its carry is kept.
            let _ = mac(r[0], k, E::P[0], &mut c2);
            for j in 1..N {
                r[j] = mac(r[j], a[j], bi, &mut c1);
                r[j - 1] = mac(r[j], k, E::P[j], &mut c2);
            }
            r[N - 1] = c1.wrapping_add(c2);
        }
        Fp(Self::reduce_once(r), PhantomData)
    }

    /// `self²`: the double-width square, its cross products once, then Montgomery's reduction.
    #[inline(always)]
    pub(crate) fn square(&self) -> Self {
        // The kernels' product of `self` by itself is the shorter chain: one block against the
        // square's two (measured 8–10 % faster in a chain of dependent squares, 4 and 6 limbs).
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 4 || N == 6 {
                return self.mul(self);
            }
        }
        Self::redc(&sqr_wide::<N>(&self.0))
    }

    /// The sum of two elements, not reduced: below `2p`, which the products below take.
    #[inline(always)]
    pub(crate) fn add_lazy(&self, o: &Self) -> [u64; N] {
        let mut s = [0u64; N];
        let mut carry = false;
        for i in 0..N {
            s[i] = adc(self.0[i], o.0[i], &mut carry);
        }
        s
    }

    /// Montgomery's reduction `t / R mod p` of a double-width `t < pR`, below `p`.
    #[inline(always)]
    pub(crate) fn redc(t: &Wide) -> Self {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 6 {
                return Fp(*view(&k::redc6(t, view(&E::P), E::INV)), PhantomData);
            }
            if N == 4 {
                return Fp(*view(&k::redc4(low(t), view(&E::P), E::INV)), PhantomData);
            }
        }
        let mut t = *t;
        let mut top = false;
        for i in 0..N {
            let k = t[i].wrapping_mul(E::INV);
            let mut c = 0u64;
            let _ = mac(t[i], k, E::P[0], &mut c);
            for j in 1..N {
                t[i + j] = mac(t[i + j], k, E::P[j], &mut c);
            }
            // c + top into the upper half, from position i + N on (its carry is at most one).
            let mut carry = top;
            t[i + N] = adc(t[i + N], c, &mut carry);
            top = carry;
        }
        // (t + mp)/R < 2p < R: the top carry is 0.
        let mut r = [0u64; N];
        r.copy_from_slice(&t[N..2 * N]);
        Fp(Self::reduce_once(r), PhantomData)
    }

    /// `t₀ − t₁` for double-width `t₀, t₁ < p²`, plus `pR` when negative: below `pR`.
    #[inline(always)]
    pub(crate) fn wide_sub(t0: &Wide, t1: &Wide) -> Wide {
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            if N == 6 {
                return k::wide_sub6(t0, t1, view(&E::P));
            }
            if N == 4 {
                return widen(k::wide_sub4(low(t0), low(t1), view(&E::P)));
            }
        }
        let mut d = [0u64; 12];
        let mut borrow = false;
        for i in 0..2 * N {
            d[i] = sbb(t0[i], t1[i], &mut borrow);
        }
        let mask = 0u64.wrapping_sub(u64::from(borrow));
        let mut carry = false;
        for i in 0..N {
            d[N + i] = adc(d[N + i], E::P[i] & mask, &mut carry);
        }
        d
    }

    /// `self^e` for the exponent's limbs, least significant first, by a window of 4 bits.
    pub(crate) fn pow(&self, e: &[u64]) -> Self {
        let mut table = [Self::one(); 16];
        for i in 1..16 {
            table[i] = table[i - 1].mul(self);
        }
        let mut r = Self::one();
        let mut started = false;
        for limb in e.iter().rev() {
            for k in (0..16).rev() {
                let d = ((limb >> (4 * k)) & 15) as usize;
                if started {
                    r = r.square().square().square().square();
                }
                if d != 0 {
                    r = if started { r.mul(&table[d]) } else { table[d] };
                    started = true;
                }
            }
        }
        r
    }

    /// The canonical limbs, out of Montgomery's form.
    #[inline]
    pub(crate) fn canonical(&self) -> [u64; N] {
        let mut one = [0u64; N];
        one[0] = 1;
        self.mul(&Fp(one, PhantomData)).0
    }

    /// The element of canonical limbs `x < p`.
    #[inline]
    pub(crate) fn from_canonical(x: [u64; N]) -> Self {
        Fp(x, PhantomData).mul(&Fp(E::R2, PhantomData))
    }

    /// The element of `x < p` given as `8 N` little-endian bytes or fewer.
    pub(crate) fn from_le_bytes(b: &[u8]) -> Self {
        let mut l = [0u64; N];
        for (i, chunk) in b.chunks(8).enumerate() {
            let mut w = [0u8; 8];
            w[..chunk.len()].copy_from_slice(chunk);
            l[i] = u64::from_le_bytes(w);
        }
        Self::from_canonical(l)
    }

    /// Whether the canonical value is odd.
    #[inline]
    pub(crate) fn is_odd(&self) -> bool {
        self.canonical()[0] & 1 == 1
    }

    /// The element of the small integer `k`.
    pub(crate) fn from_u64(k: u64) -> Self {
        let mut l = [0u64; N];
        l[0] = k;
        Self::from_canonical(l)
    }

    /// `1 / self`, `None` for 0: Lehmer's algorithm on the limbs of `aR`, which gives
    /// `(aR)⁻¹`, then a product by `R³` back into Montgomery's form, `a⁻¹R`.
    pub(crate) fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let raw = Fp(super::inv::invert(&self.0, &E::P), PhantomData);
        Some(raw.mul(&E::tables().r3))
    }

    /// A square root, if there is one: `self^((p + 1)/4)`, as `p ≡ 3 (mod 4)` for both curves.
    pub(crate) fn sqrt(&self) -> Option<Self> {
        let r = self.pow(&E::tables().sqrt_exp);
        (r.square() == *self).then_some(r)
    }
}

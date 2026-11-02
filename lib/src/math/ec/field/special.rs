//! The primes of P-256 and P-384, whose Montgomery reductions need few or no products of limbs.

use super::{Field, Mont};
use crate::math::numbers::Number;

/// The field of a NIST prime: the arithmetic of [`Mont`], whose elements are in the same form, with
/// the products of the kernels where a target has them.
pub(in super::super) struct Nist<const N: usize> {
    /// The arithmetic.
    mont: Mont<N>,
}

impl<const N: usize> Nist<N> {
    /// The field modulo `mont`'s prime.
    pub(in super::super) fn new(mont: Mont<N>) -> Self {
        Nist { mont }
    }
}

/// Generates the field's products: the kernels' where they are, Montgomery's otherwise.
macro_rules! products {
    ($n:literal, $mul:expr, $sqr:expr) => {
        impl Products<$n> for Nist<$n> {
            #[inline(always)]
            fn product(&self, a: &[u64; $n], b: &[u64; $n]) -> [u64; $n] {
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                return $mul(a, b);
                #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
                self.mont.mul(a, b)
            }

            #[inline(always)]
            fn square(&self, a: &[u64; $n]) -> [u64; $n] {
                #[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
                return $sqr(a);
                #[cfg(not(all(target_arch = "aarch64", target_endian = "little", not(miri))))]
                self.mont.sqr(a)
            }
        }
    };
}

/// The products of a field of `N` limbs.
pub(in super::super) trait Products<const N: usize> {
    /// `a b`.
    fn product(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N];
    /// `a^2`.
    fn square(&self, a: &[u64; N]) -> [u64; N];
}

#[cfg(all(target_arch = "aarch64", target_endian = "little", not(miri)))]
use super::aarch64::{p256_mul, p256_sqr, p384_mul, p384_sqr};

products!(4, p256_mul, p256_sqr);
products!(6, p384_mul, p384_sqr);

impl<const N: usize> Field for Nist<N>
where
    Nist<N>: Products<N>,
{
    type E = [u64; N];

    #[inline(always)]
    fn limbs(&self) -> usize {
        N
    }

    #[inline(always)]
    fn load(&self, x: &[u64]) -> [u64; N] {
        self.mont.load(x)
    }

    #[inline(always)]
    fn store(&self, x: &[u64; N], r: &mut [u64]) {
        self.mont.store(x, r);
    }

    #[inline(always)]
    fn zero(&self) -> [u64; N] {
        self.mont.zero()
    }

    #[inline(always)]
    fn one(&self) -> [u64; N] {
        self.mont.one()
    }

    #[inline(always)]
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.mont.add(a, b)
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.mont.sub(a, b)
    }

    #[inline(always)]
    fn neg(&self, a: &[u64; N]) -> [u64; N] {
        self.mont.neg(a)
    }

    #[inline(always)]
    fn half(&self, a: &[u64; N]) -> [u64; N] {
        self.mont.half(a)
    }

    #[inline(always)]
    fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.product(a, b)
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; N]) -> [u64; N] {
        self.square(a)
    }

    #[inline(always)]
    fn is_zero(&self, a: &[u64; N]) -> bool {
        self.mont.is_zero(a)
    }

    #[inline(always)]
    fn equal(&self, a: &[u64; N], b: &[u64; N]) -> bool {
        self.mont.equal(a, b)
    }

    fn enter(&self, x: &Number) -> [u64; N] {
        self.mont.enter(x)
    }

    fn leave(&self, x: &[u64; N]) -> Number {
        self.mont.leave(x)
    }

    #[inline(always)]
    fn modulus(&self) -> &Number {
        self.mont.modulus()
    }
}

//! A field of any length, on [`Number`].

use super::{Field, limbs_of, number_of};
use crate::math::numbers::{ByteOrder, Divisor, Number};

/// The field modulo a prime longer than the arrays of [`Mont`](super::Mont) take, whose elements
/// are numbers below it, in little-endian order, where `Number`'s paths are the fastest; its
/// reductions go through a [`Divisor`], which pays for itself from a few divisions, with the
/// multiplication's and the division's own methods for long primes.
pub(in super::super) struct Big {
    /// The prime.
    p: Number,
    /// The prime prepared for division.
    d: Divisor,
    /// The limbs of an element.
    n: usize,
}

impl Big {
    /// The field modulo the prime `p`.
    pub(in super::super) fn new(p: &Number) -> Big {
        Big {
            p: p.clone(),
            d: Divisor::new(p),
            n: limbs_of(p).len(),
        }
    }
}

impl Field for Big {
    type E = Number;

    fn limbs(&self) -> usize {
        self.n
    }

    fn load(&self, x: &[u64]) -> Number {
        number_of(&x[..self.n], ByteOrder::LittleEndian)
    }

    fn store(&self, x: &Number, r: &mut [u64]) {
        let l = limbs_of(x);
        r[..l.len()].copy_from_slice(&l);
        r[l.len()..self.n].fill(0);
    }

    fn zero(&self) -> Number {
        Number::new_from_u8(0, ByteOrder::LittleEndian)
    }

    fn one(&self) -> Number {
        Number::new_from_u8(1, ByteOrder::LittleEndian)
    }

    fn add(&self, a: &Number, b: &Number) -> Number {
        let s = a + b;
        if s >= self.p { s - &self.p } else { s }
    }

    fn sub(&self, a: &Number, b: &Number) -> Number {
        let d = a - b;
        if d.is_negative() { d + &self.p } else { d }
    }

    fn neg(&self, a: &Number) -> Number {
        if a.is_zero() { a.clone() } else { &self.p - a }
    }

    fn mul(&self, a: &Number, b: &Number) -> Number {
        self.d.rem_euclid(&(a * b))
    }

    fn sqr(&self, a: &Number) -> Number {
        self.d.rem_euclid(&(a * a))
    }

    fn is_zero(&self, a: &Number) -> bool {
        a.is_zero()
    }

    fn enter(&self, x: &Number) -> Number {
        let r = x.rem_euclid(&self.p);
        number_of(&limbs_of(&r), ByteOrder::LittleEndian)
    }

    fn leave(&self, x: &Number) -> Number {
        number_of(&limbs_of(x), self.p.byte_order)
    }

    fn modulus(&self) -> &Number {
        &self.p
    }
}

//! Inversion in a prime field: Lehmer's extended Euclid of
//! [`Number::mod_inverse`](crate::math::numbers::Number::mod_inverse) on the value.
//!
//! The value leaves the field's form for one short number and comes back from the inverse's
//! bytes, about a tenth of the inversion's time for a prime of four limbs. Bernstein and Yang's
//! division steps on limbs of 62 bits ("Fast constant-time gcd computation and modular
//! inversion", 2019, in their variable-time form, with the steps that only add a multiple of `f`
//! taken several at a time) were tried in their place and took about 1.5 times as long on an
//! Apple M4 Max from four limbs to nine: their 62 steps a batch are a chain of dependent
//! instructions, where Lehmer's steps find some 64 bits of quotients from the top two limbs at
//! a time (see [`Number::gcd_ext`](crate::math::numbers::Number::gcd_ext)).

use super::prime_form::{Arith, Canon, canonical, read_limbs};
use crate::math::ec::field::number_of;
use crate::math::numbers::ByteOrder;

/// `1 / e`, or `None` for 0.
pub(super) fn inverse<F: Arith>(f: &F, e: &F::E) -> Option<F::E> {
    if f.is_zero(e) {
        return None;
    }
    let c = canonical(f, e);
    let r = number_of(c.limbs(), ByteOrder::LittleEndian).mod_inverse(f.modulus())?;
    let mut out = Canon::zeros(f.limbs());
    read_limbs(&r, out.limbs_mut());
    Some(f.of_canon(out.limbs()))
}

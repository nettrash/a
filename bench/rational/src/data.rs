// Inputs: random rationals as sign and little-endian parts, and their forms in every library.
#![allow(dead_code)]
use a::math::numbers::{ByteOrder, Number};
use a::math::rational::Rational;
use malachite_base::num::conversion::traits::ToStringBase;
use malachite_nz::integer::Integer as MInt;
use malachite_nz::natural::Natural;
use malachite_q::Rational as MQ;
use num_bigint::{BigInt, Sign};
use num_rational::BigRational;

pub const LE: ByteOrder = ByteOrder::LittleEndian;

/// A rational as its sign and the little-endian bytes of its parts (trimmed).
#[derive(Clone, Debug)]
pub struct Q {
    pub neg: bool,
    pub num: Vec<u8>,
    pub den: Vec<u8>,
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next() as u128 * n as u128) >> 64) as u64
    }
    pub fn bool(&mut self) -> bool {
        self.next() & 1 == 1
    }
    /// A random number of exactly `bits` bits (top bit set), little-endian bytes.
    pub fn exact(&mut self, bits: u64) -> Vec<u8> {
        if bits == 0 {
            return vec![0];
        }
        let nb = bits.div_ceil(8) as usize;
        let mut v = Vec::with_capacity(nb);
        while v.len() < nb {
            v.extend_from_slice(&self.next().to_le_bytes());
        }
        v.truncate(nb);
        let top = ((bits - 1) % 8) as u32;
        let last = v.last_mut().unwrap();
        *last &= ((1u16 << (top + 1)) - 1) as u8;
        *last |= 1 << top;
        v
    }
    /// A random number in [1, 2^bits), little-endian bytes.
    pub fn upto(&mut self, bits: u64) -> Vec<u8> {
        loop {
            let nb = bits.div_ceil(8) as usize;
            let mut v = Vec::with_capacity(nb);
            while v.len() < nb {
                v.extend_from_slice(&self.next().to_le_bytes());
            }
            v.truncate(nb);
            let extra = (nb as u64) * 8 - bits;
            if extra > 0 {
                *v.last_mut().unwrap() &= 0xFFu8 >> extra;
            }
            let v = trim(v);
            if v != [0] {
                return v;
            }
        }
    }
}

pub fn trim(mut v: Vec<u8>) -> Vec<u8> {
    while v.len() > 1 && *v.last().unwrap() == 0 {
        v.pop();
    }
    if v.is_empty() {
        v.push(0);
    }
    v
}

pub fn le_bytes(n: &Number) -> Vec<u8> {
    let mut v = n.value.clone();
    if n.byte_order == ByteOrder::BigEndian {
        v.reverse();
    }
    trim(v)
}

pub fn number(bytes: &[u8], neg: bool) -> Number {
    Number::new_signed(bytes.to_vec(), neg, LE)
}

/// Reduces a raw fraction to lowest terms (by the crate) and returns its parts.
pub fn reduce(neg: bool, num: Vec<u8>, den: Vec<u8>) -> Q {
    let r = Rational::new(number(&num, neg), number(&den, false));
    of_ours(&r)
}

pub fn of_ours(r: &Rational) -> Q {
    Q { neg: r.is_negative(), num: le_bytes(r.numer()), den: le_bytes(r.denom()) }
}

/// A random rational with a numerator of up to `nb` bits and a denominator of up to `db`
/// bits (`exact`: the top bits set before the reduction), the sign random when `signed`.
pub fn rand_q(rng: &mut Rng, nb: u64, db: u64, exact: bool, signed: bool) -> Q {
    let num = if exact { rng.exact(nb) } else { rng.upto(nb) };
    let den = if exact { rng.exact(db) } else { rng.upto(db) };
    let neg = signed && rng.bool();
    reduce(neg, num, den)
}

pub fn ours(q: &Q) -> Rational {
    Rational::new(number(&q.num, q.neg), number(&q.den, false))
}

pub fn limbs(b: &[u8]) -> Vec<u64> {
    b.chunks(8)
        .map(|c| {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(w)
        })
        .collect()
}

pub fn nat(b: &[u8]) -> Natural {
    Natural::from_owned_limbs_asc(limbs(b))
}

pub fn mint(b: &[u8], neg: bool) -> MInt {
    MInt::from_sign_and_abs(!neg, nat(b))
}

pub fn mal(q: &Q) -> MQ {
    MQ::from_sign_and_naturals(!q.neg, nat(&q.num), nat(&q.den))
}

pub fn bigint(b: &[u8], neg: bool) -> BigInt {
    let s = if neg { Sign::Minus } else { Sign::Plus };
    BigInt::from_bytes_le(s, b)
}

pub fn num(q: &Q) -> BigRational {
    BigRational::new_raw(bigint(&q.num, q.neg), bigint(&q.den, false))
}

#[cfg(feature = "rug")]
pub fn rint(b: &[u8], neg: bool) -> rug::Integer {
    let z = rug::Integer::from_digits(b, rug::integer::Order::Lsf);
    if neg { -z } else { z }
}

#[cfg(feature = "rug")]
pub fn rugq(q: &Q) -> rug::Rational {
    rug::Rational::from((rint(&q.num, q.neg), rint(&q.den, false)))
}

// Canonical text in base 16: "-p/q", or "p" for an integer.
pub fn hx_ours(r: &Rational) -> String {
    r.to_str_radix(16)
}

pub fn hx_mal(q: &MQ) -> String {
    q.to_string_base(16)
}

pub fn hx_num(q: &BigRational) -> String {
    if q.denom() == &BigInt::from(1) {
        q.numer().to_str_radix(16)
    } else {
        format!("{}/{}", q.numer().to_str_radix(16), q.denom().to_str_radix(16))
    }
}

#[cfg(feature = "rug")]
pub fn hx_rug(q: &rug::Rational) -> String {
    q.to_string_radix(16)
}

pub fn hx_int_ours(n: &Number) -> String {
    n.to_str_radix(16)
}

/// A pool size for parts of `limbs` limbs: 16,384 limbs' worth, at least `min`.
pub fn pool_for(limbs: u64, min: usize) -> usize {
    ((16384 / limbs.max(1)) as usize).clamp(min, 16384)
}

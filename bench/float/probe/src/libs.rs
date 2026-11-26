//! Exact conversions of a `Raw` into each library's float, and of each result into a `Canon`.
use crate::ffi::*;
use crate::h::{Canon, Raw};
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::numbers::{ByteOrder, Number};

pub const N: Round = Round::Nearest;

pub fn number(neg: bool, limbs: &[u64]) -> Number {
    let mut value: Vec<u8> = limbs.iter().flat_map(|w| w.to_le_bytes()).collect();
    while value.last() == Some(&0) {
        value.pop();
    }
    if value.is_empty() {
        value.push(0);
    }
    Number { value, negative: neg && limbs.iter().any(|&w| w != 0), byte_order: ByteOrder::LittleEndian }
}

pub fn number_limbs(n: &Number) -> (bool, Vec<u64>) {
    let mut bytes = n.value.clone();
    if n.byte_order == ByteOrder::BigEndian {
        bytes.reverse();
    }
    while bytes.len() % 8 != 0 {
        bytes.push(0);
    }
    let limbs = bytes.chunks(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect();
    (n.negative, limbs)
}

// ---- ours ----

pub fn ours(r: &Raw, p: u64) -> BigFloat {
    assert!(p >= r.bits);
    let m = number(r.neg, &r.limbs);
    let (x, o) = BigFloat::from_parts(&m, r.exp, p, N);
    assert_eq!(o, core::cmp::Ordering::Equal);
    x
}

pub fn c_ours(x: &BigFloat) -> Canon {
    if x.is_nan() {
        return Canon::Nan;
    }
    if x.is_infinite() {
        return Canon::Inf(x.is_sign_negative());
    }
    if x.is_zero() {
        return Canon::Zero(x.is_sign_negative());
    }
    let (m, e) = x.mantissa_exponent().unwrap();
    let (neg, l) = number_limbs(&m);
    Canon::fin(neg, l, e)
}

pub fn c_ours2(z: &BigComplex) -> Canon {
    Canon::Many(vec![c_ours(z.re()), c_ours(z.im())])
}

pub fn ours_c(re: &Raw, im: &Raw, p: u64) -> BigComplex {
    BigComplex::new(ours(re, p), ours(im, p))
}

// ---- MPFR / MPC ----

pub fn mpfr(r: &Raw, p: u64) -> MF {
    let mut x = MF::new(p);
    set_mpfr(x.p(), r);
    x
}

pub fn set_mpfr(x: P, r: &Raw) {
    let z = MZ::from_limbs(r.neg, &r.limbs);
    unsafe {
        let o = mpfr_set_z_2exp(x, &z.0, r.exp as _, RN);
        assert_eq!(o, 0);
    }
}

const EXP_ZERO: i64 = i64::MIN + 1;
const EXP_NAN: i64 = i64::MIN + 2;
const EXP_INF: i64 = i64::MIN + 3;

pub fn c_mpfr(x: C) -> Canon {
    let x = unsafe { &*x };
    let neg = x.sign < 0;
    match x.exp as i64 {
        EXP_NAN => Canon::Nan,
        EXP_INF => Canon::Inf(neg),
        EXP_ZERO => Canon::Zero(neg),
        e => {
            let n = (x.prec as u64).div_ceil(64) as usize;
            let l = unsafe { std::slice::from_raw_parts(x.d, n).to_vec() };
            Canon::fin(neg, l, e - 64 * n as i64)
        }
    }
}

pub fn mpc(re: &Raw, im: &Raw, p: u64) -> MC {
    let mut z = MC::new(p);
    set_mpfr(&mut z.0.re, re);
    set_mpfr(&mut z.0.im, im);
    z
}

pub fn c_mpc(z: *const Mpc) -> Canon {
    let z = unsafe { &*z };
    Canon::Many(vec![c_mpfr(&z.re), c_mpfr(&z.im)])
}

// ---- rug ----
pub mod rg {
    use super::*;
    pub use rug::{Assign, Complex, Float};
    pub fn f(r: &Raw, p: u64) -> Float {
        let mut x = Float::new(p as u32);
        set_mpfr(x.as_raw_mut() as *mut Mpfr, r);
        x
    }
    pub fn c(x: &Float) -> Canon {
        c_mpfr(x.as_raw() as *const Mpfr)
    }
    pub fn cx(re: &Raw, im: &Raw, p: u64) -> Complex {
        Complex::with_val(p as u32, (f(re, p), f(im, p)))
    }
    pub fn cc(z: &Complex) -> Canon {
        Canon::Many(vec![c(z.real()), c(z.imag())])
    }
}

// ---- astro-float ----
pub mod ast {
    use super::*;
    pub use astro_float::{BigFloat as AF, Consts, RoundingMode as ARM, Sign as AS};
    pub const RM: ARM = ARM::ToEven;
    /// astro-float's precision is a whole number of words.
    pub fn wp(p: u64) -> u64 {
        p.div_ceil(64) * 64
    }
    pub fn f(r: &Raw, p: u64) -> AF {
        let len = (p.div_ceil(64) as usize).max(r.limbs.len());
        let mut m = r.limbs.clone();
        m.resize(len, 0);
        // shift left so that the top bit of M is the top bit of the words
        let sh = 64 * len as u64 - r.bits;
        let m = shl_limbs(&m, sh);
        let s = if r.neg { AS::Neg } else { AS::Pos };
        let x = AF::from_raw_parts(&m, 64 * len, s, r.e() as i32, false);
        assert!(!x.is_nan(), "astro from_raw_parts");
        x
    }
    pub fn c(x: &AF) -> Canon {
        if x.is_nan() {
            return Canon::Nan;
        }
        if x.is_inf() {
            return Canon::Inf(x.is_inf_neg());
        }
        if x.is_zero() {
            return Canon::Zero(x.is_negative());
        }
        let (m, _n, s, e, _) = x.as_raw_parts().unwrap();
        Canon::fin(s == AS::Neg, m.to_vec(), e as i64 - 64 * m.len() as i64)
    }
}

pub fn shl_limbs(m: &[u64], sh: u64) -> Vec<u64> {
    let ws = (sh / 64) as usize;
    let bs = (sh % 64) as u32;
    let n = m.len();
    let mut r = vec![0u64; n];
    for i in (0..n).rev() {
        if i < ws {
            break;
        }
        let src = i - ws;
        let mut v = m[src] << bs;
        if bs > 0 && src > 0 {
            v |= m[src - 1] >> (64 - bs);
        }
        r[i] = v;
    }
    r
}

// ---- dashu-float ----
pub mod dsh {
    use super::*;
    pub use dashu_float::round::mode::HalfEven;
    pub use dashu_float::{Context, FBig};
    pub use dashu_int::{IBig, UBig};
    pub type DF = FBig<HalfEven, 2>;
    pub fn f(r: &Raw, p: u64) -> DF {
        let u = UBig::from_words(&r.limbs);
        let i = if r.neg { -IBig::from(u) } else { IBig::from(u) };
        DF::from_parts(i, r.exp as isize).with_precision(p as usize).value()
    }
    pub fn c(x: &DF) -> Canon {
        let r = x.repr();
        if r.is_infinite() {
            return Canon::Inf(r.sign() == dashu_int::Sign::Negative);
        }
        if r.significand().is_zero() {
            return Canon::Zero(false);
        }
        let (s, w) = r.significand().as_sign_words();
        Canon::fin(s == dashu_int::Sign::Negative, w.to_vec(), r.exponent() as i64)
    }
    pub fn ctx(p: u64) -> Context<HalfEven> {
        Context::<HalfEven>::new(p as usize)
    }
}

// ---- malachite-float ----
pub mod mal {
    use super::*;
    pub use malachite_base::num::conversion::traits::IntegerMantissaAndExponent;
    pub use malachite_base::rounding_modes::RoundingMode as MRM;
    pub use malachite_float::Float as MF2;
    pub use malachite_nz::natural::Natural;
    pub const RM: MRM = MRM::Nearest;
    pub fn f(r: &Raw, p: u64) -> MF2 {
        let n = Natural::from_limbs_asc(&r.limbs);
        let (x, o) = MF2::from_natural_prec_round(n, p, RM);
        assert_eq!(o, core::cmp::Ordering::Equal);
        let x = x << r.exp;
        if r.neg { -x } else { x }
    }
    pub fn c(x: &MF2) -> Canon {
        if x.is_nan() {
            return Canon::Nan;
        }
        if x.is_infinite() {
            return Canon::Inf(x.is_sign_negative());
        }
        if x.is_zero() {
            return Canon::Zero(x.is_sign_negative());
        }
        let (m, e) = x.integer_mantissa_and_exponent();
        Canon::fin(x.is_sign_negative(), m.to_limbs_asc(), e)
    }
}

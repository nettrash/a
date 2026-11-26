//! Montgomery's form for primes of more than nine limbs, the length chosen at run time: an element
//! is `x R mod p` for `R = 2^(64 n)`, fully reduced, in a vector of `n` limbs. A product is the
//! numbers' schoolbook kernel (`basecase`, in assembly where the crate has it) into `2 n` limbs on
//! the stack and the kernel's REDC; sums and differences are carry chains with one correction. The
//! assignments (`+=`, `-=`, `*=`) write into the element's own vector, so that they allocate nothing.
//!
//! An element made from its value (bytes, a number, a word) keeps the value
//! ([`Value::WideValue`]) until an operation needs the residue, which it makes once: decoding and
//! encoding again, the value and the hash then take no conversion.
//!
//! This replaces, for prime fields, the `Number` form of the curves (`ec::field::Big`), whose
//! products are a multiplication of numbers and a division by a prepared divisor.

use super::prime_form::{Arith, Value, WideValue};
use crate::math::ec::field::{Field, limbs_of, number_of};
use crate::math::numbers::{ByteOrder, Number, basecase, redc_inverse, redc_limbs};
use std::sync::OnceLock;

/// The most limbs whose double-length products go on the stack (4096-bit primes).
const STACK: usize = 64;

/// The field modulo an odd prime of `n > 9` limbs, in Montgomery's form.
#[derive(Clone)]
pub(super) struct Wide {
    /// The prime's limbs, the low one first.
    m: Vec<u64>,
    /// `-1 / m mod 2^128`, for the kernel's REDC.
    minv2: [u64; 2],
    /// `R mod m`, one.
    one: Vec<u64>,
    /// `R² mod m`.
    r2: Vec<u64>,
    /// `R³ mod m`: the product by it takes the inverse of a residue `xR`, `x⁻¹R⁻¹`, to `x⁻¹R`.
    r3: Vec<u64>,
    /// The prime.
    p: Number,
}

impl Wide {
    /// The field of the odd prime `p`, of more than one limb.
    pub(super) fn new(p: &Number) -> Wide {
        let m = limbs_of(p);
        let n = m.len();
        let r = Number::new_from_u8(1, p.byte_order) << (64 * n);
        let one = r.rem_euclid(p);
        let r2 = (&one * &one).rem_euclid(p);
        let r3 = (&r2 * &one).rem_euclid(p);
        let pad = |x: &Number| {
            let mut l = limbs_of(x);
            l.resize(n, 0);
            l
        };
        Wide {
            minv2: redc_inverse(m[0], m[1]),
            one: pad(&one),
            r2: pad(&r2),
            r3: pad(&r3),
            m,
            p: p.clone(),
        }
    }

    #[inline(always)]
    fn n(&self) -> usize {
        self.m.len()
    }

    /// Runs `f` on a buffer of `2 n` limbs: on the stack up to [`STACK`] limbs.
    #[inline(always)]
    fn with_double<R>(&self, f: impl FnOnce(&mut [u64]) -> R) -> R {
        let n = self.n();
        if n <= STACK {
            let mut t = [0u64; 2 * STACK];
            f(&mut t[..2 * n])
        } else {
            f(&mut vec![0u64; 2 * n])
        }
    }

    /// REDC of the `2 n` limbs `t`, below `R m`, which it clobbers, into `r`.
    #[inline(always)]
    fn reduce_into(&self, t: &mut [u64], r: &mut [u64]) {
        let n = self.n();
        debug_assert!(t.len() == 2 * n && r.len() == n);
        // SAFETY: `t` has `2 n` limbs and `r` `n`, apart from it; `m` has `n` limbs.
        let over = unsafe {
            redc_limbs(
                r.as_mut_ptr(),
                t.as_mut_ptr(),
                self.m.as_ptr(),
                n,
                self.minv2,
            )
        };
        // The kernel's result is below `R + m`: one subtraction makes it a residue below `m`.
        if over || !lt(r, &self.m) {
            sub_in_place(r, &self.m);
        }
    }

    /// The residue `v` holds, for an update in place: an element held as its value becomes its
    /// residue first.
    #[inline(always)]
    fn residue_mut<'a>(&self, v: &'a mut Value) -> &'a mut Vec<u64> {
        if let Value::WideValue(x) = v {
            let r = match x.residue.take() {
                Some(r) => r,
                None => self.of_canon(&x.value),
            };
            *v = Value::Wide(r);
        }
        match v {
            Value::Wide(x) => x,
            _ => unreachable!("an element of another form"),
        }
    }

    /// `a b / R mod m` into `r`.
    #[inline(always)]
    fn mul_limbs(&self, a: &[u64], b: &[u64], r: &mut [u64]) {
        let n = self.n();
        self.with_double(|t| {
            // SAFETY: `t` holds `2 n` limbs, apart from `a` and `b` of `n`.
            unsafe { basecase::mul(t.as_mut_ptr(), a.as_ptr(), b.as_ptr(), n) };
            self.reduce_into(t, r);
        })
    }

    /// `a² / R mod m` into `r`.
    #[inline(always)]
    fn sqr_into(&self, a: &[u64], r: &mut [u64]) {
        let n = self.n();
        self.with_double(|t| {
            // SAFETY: as in `mul_into`.
            unsafe { basecase::sqr(t.as_mut_ptr(), a.as_ptr(), n) };
            self.reduce_into(t, r);
        })
    }

    /// `a + b mod m` into `a`.
    #[inline(always)]
    fn add_in_place(&self, a: &mut [u64], b: &[u64]) {
        let carry = add_to(a, b);
        if carry || !lt(a, &self.m) {
            sub_in_place(a, &self.m);
        }
    }

    /// `a - b mod m` into `a`.
    #[inline(always)]
    fn sub_in_place_mod(&self, a: &mut [u64], b: &[u64]) {
        if sub_in_place(a, b) {
            add_to(a, &self.m);
        }
    }
}

/// `a < b`, for limbs of the same length, the low one first.
#[inline(always)]
fn lt(a: &[u64], b: &[u64]) -> bool {
    for i in (0..a.len()).rev() {
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// `a += b`, and the carry out.
#[inline(always)]
fn add_to(a: &mut [u64], b: &[u64]) -> bool {
    let mut c = false;
    for (x, &y) in a.iter_mut().zip(b) {
        let (s, c1) = x.overflowing_add(y);
        let (s, c2) = s.overflowing_add(u64::from(c));
        *x = s;
        c = c1 | c2;
    }
    c
}

/// `a -= b`, and the borrow out.
#[inline(always)]
fn sub_in_place(a: &mut [u64], b: &[u64]) -> bool {
    let mut c = false;
    for (x, &y) in a.iter_mut().zip(b) {
        let (s, b1) = x.overflowing_sub(y);
        let (s, b2) = s.overflowing_sub(u64::from(c));
        *x = s;
        c = b1 | b2;
    }
    c
}

impl Field for Wide {
    type E = Vec<u64>;

    fn limbs(&self) -> usize {
        self.n()
    }

    fn load(&self, x: &[u64]) -> Vec<u64> {
        x[..self.n()].to_vec()
    }

    fn store(&self, x: &Vec<u64>, r: &mut [u64]) {
        r[..self.n()].copy_from_slice(x);
    }

    fn zero(&self) -> Vec<u64> {
        vec![0; self.n()]
    }

    fn one(&self) -> Vec<u64> {
        self.one.clone()
    }

    fn add(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let mut r = a.clone();
        self.add_in_place(&mut r, b);
        r
    }

    fn sub(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let mut r = a.clone();
        self.sub_in_place_mod(&mut r, b);
        r
    }

    fn neg(&self, a: &Vec<u64>) -> Vec<u64> {
        if self.is_zero(a) {
            return a.clone();
        }
        let mut r = self.m.clone();
        sub_in_place(&mut r, a);
        r
    }

    fn mul(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let mut r = vec![0; self.n()];
        self.mul_limbs(a, b, &mut r);
        r
    }

    fn sqr(&self, a: &Vec<u64>) -> Vec<u64> {
        let mut r = vec![0; self.n()];
        self.sqr_into(a, &mut r);
        r
    }

    fn is_zero(&self, a: &Vec<u64>) -> bool {
        a.iter().all(|&x| x == 0)
    }

    fn equal(&self, a: &Vec<u64>, b: &Vec<u64>) -> bool {
        a == b
    }

    fn enter(&self, x: &Number) -> Vec<u64> {
        let mut c = limbs_of(&x.rem_euclid(&self.p));
        c.resize(self.n(), 0);
        self.of_canon(&c)
    }

    fn leave(&self, x: &Vec<u64>) -> Number {
        let mut c = vec![0; self.n()];
        self.canon(x, &mut c);
        number_of(&c, self.p.byte_order)
    }

    fn modulus(&self) -> &Number {
        &self.p
    }
}

impl Arith for Wide {
    #[inline(always)]
    fn get<'a>(&self, v: &'a Value) -> &'a Vec<u64> {
        match v {
            Value::Wide(x) => x,
            Value::WideValue(x) => x.residue.get_or_init(|| self.of_canon(&x.value)),
            _ => unreachable!("an element of another form"),
        }
    }

    #[inline(always)]
    fn put(&self, e: Vec<u64>) -> Value {
        Value::Wide(e)
    }

    // Kept as its value until an operation needs the residue: an element that is decoded and
    // encoded again, or compared by its hash, is not converted at all.
    fn put_canon(&self, x: &[u64]) -> Value {
        Value::WideValue(WideValue {
            value: x.to_vec(),
            residue: OnceLock::new(),
        })
    }

    fn put_number(&self, x: &Number) -> Value {
        let mut l = limbs_of(&x.rem_euclid(&self.p));
        l.resize(self.n(), 0);
        self.put_canon(&l)
    }

    fn canon_value(&self, v: &Value, out: &mut [u64]) {
        match v {
            Value::WideValue(x) => out.copy_from_slice(&x.value),
            _ => self.canon(self.get(v), out),
        }
    }

    fn update(&self, v: &mut Value, g: impl FnOnce(&Vec<u64>) -> Vec<u64>) {
        let r = g(self.get(v));
        *v = Value::Wide(r);
    }

    fn mul_into(&self, a: &mut Value, b: &Value) {
        let b = self.get(b);
        let x = self.residue_mut(a);
        let n = self.n();
        self.with_double(|t| {
            // SAFETY: `t` holds `2 n` limbs, apart from `x` and `b` of `n`.
            unsafe { basecase::mul(t.as_mut_ptr(), x.as_ptr(), b.as_ptr(), n) };
            self.reduce_into(t, x);
        })
    }

    fn add_into(&self, a: &mut Value, b: &Value) {
        let b = self.get(b);
        self.add_in_place(self.residue_mut(a), b);
    }

    fn sub_into(&self, a: &mut Value, b: &Value) {
        let b = self.get(b);
        self.sub_in_place_mod(self.residue_mut(a), b);
    }

    fn canon(&self, e: &Vec<u64>, out: &mut [u64]) {
        // x / R: REDC of x itself, as the low half of 2 n limbs.
        let n = self.n();
        self.with_double(|t| {
            t[..n].copy_from_slice(e);
            t[n..].fill(0);
            self.reduce_into(t, out);
        })
    }

    fn of_canon(&self, x: &[u64]) -> Vec<u64> {
        let mut r = vec![0; self.n()];
        self.mul_limbs(x, &self.r2, &mut r);
        r
    }

    #[inline(always)]
    fn same(&self, a: &Vec<u64>, b: &Vec<u64>) -> bool {
        a == b
    }

    // The inverse of the residue `xR` itself, `x⁻¹R⁻¹`, times `R³` in Montgomery's product: `x⁻¹R`,
    // with no conversion out of the form first.
    fn invert_value(&self, x: &Vec<u64>) -> Option<Vec<u64>> {
        if self.is_zero(x) {
            return None;
        }
        let inv = number_of(x, ByteOrder::LittleEndian).mod_inverse(&self.p)?;
        let mut l = limbs_of(&inv);
        l.resize(self.n(), 0);
        let mut r = vec![0; self.n()];
        self.mul_limbs(&l, &self.r3, &mut r);
        Some(r)
    }

    // `R = 2^(64 n)` is a square, so the symbol of the residue `xR` is that of `x`.
    fn legendre_value(&self, x: &Vec<u64>) -> i32 {
        number_of(x, ByteOrder::LittleEndian).jacobi(&self.p)
    }

    // `Number`'s power, by Montgomery's products on its limbs, as the `Number` form took it.
    fn power(&self, x: &Vec<u64>, e: &[u64]) -> Vec<u64> {
        let mut c = vec![0; self.n()];
        self.canon(x, &mut c);
        let r = number_of(&c, ByteOrder::LittleEndian)
            .pow_mod(&number_of(e, ByteOrder::LittleEndian), &self.p);
        let mut l = limbs_of(&r);
        l.resize(self.n(), 0);
        self.of_canon(&l)
    }
}

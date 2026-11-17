//! The engine of the greatest common divisors over a field `T`: the polynomials' vectors with the
//! core's division and the `poly_mul` kernel of the coefficients, and Euclid's algorithm for the
//! base case.

use super::hgcd::Engine;
use crate::math::poly::basic::trim;
use crate::math::poly::coeff::Field;
use crate::math::poly::div::schoolbook_div_rem;
use core::marker::PhantomData;

/// The engine over the field `T`.
pub(crate) struct FieldEngine<T> {
    /// [`Engine::hgcd_cutoff`].
    pub(crate) hgcd: usize,
    /// [`Engine::gcd_cutoff`].
    pub(crate) gcd: usize,
    /// [`Engine::xgcd_cutoff`].
    pub(crate) xgcd: usize,
    _t: PhantomData<T>,
}

impl<T: Field> FieldEngine<T> {
    /// The engine with Euclid's algorithm alone.
    pub(crate) fn euclid() -> FieldEngine<T> {
        FieldEngine {
            hgcd: usize::MAX,
            gcd: usize::MAX,
            xgcd: usize::MAX,
            _t: PhantomData,
        }
    }
}

impl<T: Field> Engine for FieldEngine<T> {
    type P = Vec<T>;
    type C = T;

    #[inline]
    fn len(&self, a: &Vec<T>) -> usize {
        a.len()
    }
    fn zero(&self) -> Vec<T> {
        Vec::new()
    }
    fn one(&self) -> Vec<T> {
        vec![T::one()]
    }
    fn lead(&self, a: &Vec<T>) -> T {
        a[a.len() - 1].clone()
    }
    fn add(&self, a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
        let (l, s) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let mut r = l.clone();
        for (x, y) in r.iter_mut().zip(s) {
            T::add_assign(x, y);
        }
        trim(&mut r);
        r
    }
    fn sub(&self, a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
        let mut r = a.clone();
        if r.len() < b.len() {
            r.resize(b.len(), T::zero());
        }
        for (x, y) in r.iter_mut().zip(b) {
            T::sub_assign(x, y);
        }
        trim(&mut r);
        r
    }
    fn neg(&self, a: &Vec<T>) -> Vec<T> {
        a.iter().map(T::neg).collect()
    }
    fn mul(&self, a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut r = T::poly_mul(a, b);
        trim(&mut r);
        r
    }
    fn divrem(&self, a: &Vec<T>, b: &Vec<T>) -> (Vec<T>, Vec<T>) {
        let inv = T::inv(&b[b.len() - 1]);
        let mut r = a.clone();
        let mut q = schoolbook_div_rem(&mut r, b, &inv);
        trim(&mut q);
        (q, r)
    }
    fn shr(&self, a: &Vec<T>, k: usize) -> Vec<T> {
        a.get(k..).map_or_else(Vec::new, <[T]>::to_vec)
    }
    fn low(&self, a: &Vec<T>, k: usize) -> Vec<T> {
        let mut r = a[..k.min(a.len())].to_vec();
        trim(&mut r);
        r
    }
    fn shl_add(&self, hi: &Vec<T>, k: usize, lo: &Vec<T>) -> Vec<T> {
        if hi.is_empty() {
            return lo.clone();
        }
        let mut r = vec![T::zero(); (k + hi.len()).max(lo.len())];
        r[..lo.len()].clone_from_slice(lo);
        for (x, y) in r[k..].iter_mut().zip(hi) {
            T::add_assign(x, y);
        }
        trim(&mut r);
        r
    }
    fn cmul(&self, x: &T, y: &T) -> T {
        T::mul(x, y)
    }
    fn cneg(&self, x: &T) -> T {
        T::neg(x)
    }
    fn cpow(&self, x: &T, mut e: u64) -> T {
        let mut r = T::one();
        let mut b = x.clone();
        while e > 0 {
            if e & 1 == 1 {
                T::mul_assign(&mut r, &b);
            }
            e >>= 1;
            if e > 0 {
                b = T::mul(&b, &b);
            }
        }
        r
    }
    fn cone(&self) -> T {
        T::one()
    }
    fn czero(&self) -> T {
        T::zero()
    }
    fn euclid(
        &self,
        mut a: Vec<T>,
        mut b: Vec<T>,
        mut rec: Option<&mut Vec<(usize, T)>>,
        shift: usize,
    ) -> Vec<T> {
        while !b.is_empty() {
            let l = &b[b.len() - 1];
            if let Some(rec) = rec.as_deref_mut() {
                rec.push((b.len() - 1 + shift, l.clone()));
                if b.len() == 1 {
                    return b;
                }
            }
            let inv = T::inv(l);
            schoolbook_div_rem(&mut a, &b, &inv);
            core::mem::swap(&mut a, &mut b);
        }
        a
    }
    fn hgcd_cutoff(&self) -> usize {
        self.hgcd
    }
    fn gcd_cutoff(&self) -> usize {
        self.gcd
    }
    fn xgcd_cutoff(&self) -> usize {
        self.xgcd
    }
}

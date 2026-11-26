// Helpers that turn an operation on each library's values into a timed contender and the
// texts of its first results for the check.
#![allow(dead_code)]
use std::cmp::Ordering;
use std::hint::black_box;

use a::math::numbers::Number;
use a::math::rational::Rational;
use malachite_q::Rational as MQ;
use num_rational::{BigRational, Ratio};

use crate::cyc;
use crate::data::*;
use crate::harness::Ent;

pub trait Show {
    fn show(&self) -> String;
}

impl Show for Rational {
    fn show(&self) -> String {
        hx_ours(self)
    }
}
impl Show for MQ {
    fn show(&self) -> String {
        hx_mal(self)
    }
}
impl Show for BigRational {
    fn show(&self) -> String {
        hx_num(self)
    }
}
#[cfg(feature = "rug")]
impl Show for rug::Rational {
    fn show(&self) -> String {
        hx_rug(self)
    }
}
#[cfg(feature = "rug")]
impl Show for rug::Integer {
    fn show(&self) -> String {
        self.to_string_radix(16)
    }
}
impl Show for Ratio<i64> {
    fn show(&self) -> String {
        let (n, d) = (*self.numer(), *self.denom());
        let s = if n < 0 { format!("-{:x}", n.unsigned_abs()) } else { format!("{n:x}") };
        if d == 1 { s } else { format!("{s}/{d:x}") }
    }
}
impl Show for Number {
    fn show(&self) -> String {
        self.to_str_radix(16)
    }
}
impl Show for malachite_nz::integer::Integer {
    fn show(&self) -> String {
        use malachite_base::num::conversion::traits::ToStringBase;
        self.to_string_base(16)
    }
}
impl Show for num_bigint::BigInt {
    fn show(&self) -> String {
        self.to_str_radix(16)
    }
}
impl Show for Ordering {
    fn show(&self) -> String {
        format!("{self:?}")
    }
}
impl Show for bool {
    fn show(&self) -> String {
        format!("{self}")
    }
}
impl Show for f64 {
    fn show(&self) -> String {
        format!("{:016x}", self.to_bits())
    }
}
impl Show for f32 {
    fn show(&self) -> String {
        format!("{:08x}", self.to_bits())
    }
}
impl Show for String {
    fn show(&self) -> String {
        self.clone()
    }
}
impl Show for i128 {
    fn show(&self) -> String {
        format!("{self:x}")
    }
}
impl<T: Show> Show for Option<T> {
    fn show(&self) -> String {
        match self {
            Some(x) => x.show(),
            None => "None".into(),
        }
    }
}
impl<T: Show, E> Show for Result<T, E> {
    fn show(&self) -> String {
        match self {
            Ok(x) => x.show(),
            Err(_) => "Err".into(),
        }
    }
}
impl<T: Show> Show for Vec<T> {
    fn show(&self) -> String {
        let v: Vec<String> = self.iter().map(|x| x.show()).collect();
        v.join(",")
    }
}
impl Show for u64 {
    fn show(&self) -> String {
        format!("{self}")
    }
}
impl Show for i64 {
    fn show(&self) -> String {
        format!("{self}")
    }
}
impl Show for usize {
    fn show(&self) -> String {
        format!("{self}")
    }
}

/// A binary operation over pools `a` and `b` (cycled together).
pub fn mk2<'a, A, B, R: Show>(
    name: &'static str,
    a: &'a [A],
    b: &'a [B],
    k: usize,
    f: impl Fn(&'a A, &'a B) -> R + 'a,
) -> (Ent<'a>, Vec<String>) {
    let checks = (0..k.min(a.len())).map(|i| f(&a[i], &b[i]).show()).collect();
    let n = a.len();
    let mut j = 0usize;
    (Ent::new(name, move |c| cyc!(j, n, c, black_box(f(black_box(&a[j]), black_box(&b[j]))))), checks)
}

/// A unary operation over pool `a`.
pub fn mk1<'a, A, R: Show>(name: &'static str, a: &'a [A], k: usize, f: impl Fn(&'a A) -> R + 'a) -> (Ent<'a>, Vec<String>) {
    let checks = (0..k.min(a.len())).map(|i| f(&a[i]).show()).collect();
    let n = a.len();
    let mut j = 0usize;
    (Ent::new(name, move |c| cyc!(j, n, c, black_box(f(black_box(&a[j]))))), checks)
}

/// The same without results to show (the check is done elsewhere).
pub fn run1<'a, A, R>(name: &'static str, a: &'a [A], f: impl Fn(&'a A) -> R + 'a) -> Ent<'a> {
    let n = a.len();
    let mut j = 0usize;
    Ent::new(name, move |c| cyc!(j, n, c, black_box(f(black_box(&a[j])))))
}

/// Collects the contenders: the first is ours, whose checks the others must match.
pub struct Set<'a> {
    pub ents: Vec<Ent<'a>>,
    ours: Vec<String>,
    what: String,
}

impl<'a> Set<'a> {
    pub fn new(what: &str, ours: (Ent<'a>, Vec<String>)) -> Set<'a> {
        Set { ents: vec![ours.0], ours: ours.1, what: what.to_string() }
    }
    pub fn add(&mut self, e: (Ent<'a>, Vec<String>)) {
        self.add_note(e, "");
    }
    /// Adds a contender whose results are expected to differ (`differs` says why), the share
    /// that differs going into its note.
    pub fn add_note(&mut self, e: (Ent<'a>, Vec<String>), differs: &str) {
        let (mut ent, checks) = e;
        let bad = self.ours.iter().zip(&checks).filter(|(a, b)| a != b).count();
        if differs.is_empty() {
            let m = crate::harness::agree(&self.ours, &checks, &format!("{} {}", self.what, ent.name));
            ent = ent.note(&m);
        } else {
            ent = ent.note(&format!("{differs}: {bad}/{} results differ from ours", checks.len()));
        }
        self.ents.push(ent);
    }
    pub fn add_unchecked(&mut self, e: Ent<'a>, why: &str) {
        self.ents.push(e.note(why));
    }
    pub fn checks_len(&self) -> usize {
        self.ours.len()
    }
    pub fn ours_checks(&self) -> &[String] {
        &self.ours
    }
}

/// What a C kernel's one-shot call leaves for the check.
#[derive(Clone, Copy, PartialEq)]
pub enum Res {
    /// The rational result variable.
    Q,
    /// The integer result variable.
    Z,
    /// The return value as an ordering.
    Sign,
    /// The return value as a boolean.
    Bool,
    /// The return value as the bits of an f64.
    F64,
    /// The return value as is.
    Int,
}

fn res_show(res: Res, v: i64, s: impl FnOnce(i32) -> String) -> String {
    match res {
        Res::Q => s(1),
        Res::Z => s(2),
        Res::Sign => v.cmp(&0).show(),
        Res::Bool => (v != 0).show(),
        Res::F64 => f64::from_bits(v as u64).show(),
        Res::Int => format!("{v}"),
    }
}

use crate::ffi::*;

pub fn gmp2<'a>(name: &'static str, a: &'a Gmp, b: &'a Gmp, op: i32, k: usize, res: Res) -> (Ent<'a>, Vec<String>) {
    let checks = (0..k.min(a.1))
        .map(|i| {
            let v = a.bin_one(op, b, i);
            res_show(res, v, |w| a.str(0, w, 16))
        })
        .collect();
    (Ent::new(name, move |c| {
        a.bin(op, b, c);
    }), checks)
}

pub fn gmp1<'a>(name: &'static str, a: &'a Gmp, op: i32, e: i64, k: usize, res: Res) -> (Ent<'a>, Vec<String>) {
    let checks = (0..k.min(a.1))
        .map(|i| {
            let v = a.un_one(op, i, e);
            res_show(res, v, |w| a.str(0, w, 16))
        })
        .collect();
    (Ent::new(name, move |c| {
        a.un(op, c, e);
    }), checks)
}

#[cfg(has_flint)]
pub fn flint2<'a>(name: &'static str, a: &'a Flint, b: &'a Flint, op: i32, k: usize, res: Res) -> (Ent<'a>, Vec<String>) {
    let checks = (0..k.min(a.1))
        .map(|i| {
            let v = a.bin_one(op, b, i);
            res_show(res, v, |w| a.str(0, w, 16))
        })
        .collect();
    (Ent::new(name, move |c| {
        a.bin(op, b, c);
    }), checks)
}

#[cfg(has_flint)]
pub fn flint1<'a>(name: &'static str, a: &'a Flint, op: i32, e: i64, k: usize, res: Res) -> (Ent<'a>, Vec<String>) {
    let checks = (0..k.min(a.1))
        .map(|i| {
            let v = a.un_one(op, i, e);
            res_show(res, v, |w| a.str(0, w, 16))
        })
        .collect();
    (Ent::new(name, move |c| {
        a.un(op, c, e);
    }), checks)
}

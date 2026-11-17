//! The eval area against vectors from PARI/GP (`data/eval.txt`, made by the area's `gen/eval.gp`):
//! evaluation at many points and products of linear factors over Z and Z/nZ, interpolation over Q
//! and modulo primes.

use super::common::size;
use a::math::numbers::{ByteOrder, Number};
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

const DATA: &str = include_str!("data/eval.txt");

fn int(s: &str) -> Number {
    Number::from_str_radix(s, 10, ByteOrder::LittleEndian).unwrap()
}

fn ints(s: &str) -> Vec<Number> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(int).collect()
    }
}

fn rat(s: &str) -> Rational {
    match s.split_once('/') {
        Some((n, d)) => Rational::new(int(n), int(d)),
        None => Rational::from(int(s)),
    }
}

fn rats(s: &str) -> Vec<Rational> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(',').map(rat).collect()
    }
}

/// The lines of the kind `k`, split at `|`; under Miri the short ones only.
fn lines(k: char) -> impl Iterator<Item = Vec<&'static str>> {
    DATA.lines()
        .filter(move |l| l.starts_with(k) && l.len() <= size(usize::MAX, 400))
        .map(|l| l[2..].split('|').collect())
}

#[test]
fn integers() {
    let mut count = 0;
    for f in lines('Z') {
        let p = Polynomial::new(ints(f[0]));
        assert_eq!(p.eval_many(&ints(f[1])), ints(f[2]));
        count += 1;
    }
    for f in lines('R') {
        let p = Polynomial::from_roots(&ints(f[0]));
        assert_eq!(p.coeffs(), &ints(f[1])[..]);
        count += 1;
    }
    assert!(count >= size(10, 1));
}

#[test]
fn rationals() {
    let mut count = 0;
    for f in lines('Q') {
        let (xs, ys) = (rats(f[0]), rats(f[1]));
        let want = Polynomial::new(rats(f[2]));
        assert_eq!(Polynomial::interpolate(&xs, &ys), want);
        if xs.len() <= 20 {
            assert_eq!(Polynomial::interpolate_newton(&xs, &ys), want);
            assert_eq!(Polynomial::interpolate_lagrange(&xs, &ys), want);
        }
        count += 1;
    }
    assert!(count >= size(5, 1));
}

#[test]
fn modular() {
    let mut count = 0;
    for f in lines('M') {
        let n = int(f[0]);
        let p = ModPolynomial::new(&ints(f[1]), &n);
        assert_eq!(p.eval_many(&ints(f[2])), ints(f[3]), "n = {n}");
        count += 1;
    }
    for f in lines('P') {
        let n = int(f[0]);
        let p = ModPolynomial::from_roots(&ints(f[1]), &n);
        assert_eq!(p, ModPolynomial::new(&ints(f[2]), &n), "n = {n}");
        count += 1;
    }
    for f in lines('I') {
        let n = int(f[0]);
        let (xs, ys) = (ints(f[1]), ints(f[2]));
        let want = ModPolynomial::new(&ints(f[3]), &n);
        assert_eq!(ModPolynomial::interpolate(&xs, &ys, &n), want, "n = {n}");
        assert_eq!(ModPolynomial::interpolate_newton(&xs, &ys, &n), want);
        assert_eq!(ModPolynomial::interpolate_lagrange(&xs, &ys, &n), want);
        count += 1;
    }
    assert!(count >= size(60, 1));
}

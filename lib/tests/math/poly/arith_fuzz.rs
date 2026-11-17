//! Regressions from the differential fuzz of the merged tree: the exact division by a divisor far
//! wider than the dividend (it must say `None`, and a true division by a wide divisor must still
//! be found), the pseudo-division by a constant, the evaluation of integer polynomials at large
//! points, and the inverse of power series over the rationals through the integers.

use super::arith_z::exact_poly;
use super::common::*;
use a::math::numbers::Number;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

/// `2^bits`.
fn pow2(bits: u32) -> Number {
    Number::from(2).pow(bits)
}

#[test]
fn exact_division_by_a_much_wider_divisor() {
    // 1 + x + ... + x^(n-1) by 2^bits x: not divisible, past the Kronecker crossover.
    let n = size(61, 14);
    let a = Polynomial::new(vec![Number::from(1); n]);
    for bits in [64u32, 100, 300, 520, 998] {
        let d = Polynomial::new(vec![Number::default(), pow2(bits)]);
        assert_eq!(a.div_exact(&d), None, "2^{bits} x");
        let d = Polynomial::new(vec![pow2(bits), Number::from(3)]);
        assert_eq!(a.div_exact(&d), None, "2^{bits} + 3x");
    }
    // A divisor longer than the quotient, one coefficient wider than the field the quotient's
    // bound alone would give (within Mignotte's bound on a divisor, so the division is tried).
    let n = size(100, 16);
    let lq = size(13, 3);
    let mut dc = vec![Number::default(); n];
    dc[0] = pow2((n as u32) * 9 / 10);
    dc[n - 1] = Number::from(1);
    let d = Polynomial::new(dc);
    let a = Polynomial::new(vec![Number::from(1); n + lq - 1]);
    assert_eq!(a.div_exact(&d), None);
    let q = Polynomial::new(vec![Number::from(1); lq]);
    assert_eq!((&d * &q).div_exact(&d), Some(q));
    // True divisions by wide divisors, of quotients long and short: (2^w + x) q / (2^w + x),
    // (2^w x - 1) q, and a divisor whose coefficients are wider than the quotient's by far.
    let mut s = 0x0F0E_0D0C_0B0A_0908;
    for (bits, lq) in [
        (600u32, size(60, 13)),
        (300, size(20, 13)),
        (1000, size(50, 13)),
    ] {
        let q = Polynomial::new(vec![Number::from(1); lq]);
        for d in [
            Polynomial::new(vec![pow2(bits), Number::from(1)]),
            Polynomial::new(vec![Number::from(-1), pow2(bits)]),
            Polynomial::new(vec![pow2(bits), Number::default(), -pow2(bits / 2)]),
        ] {
            let a = &d * &q;
            assert_eq!(a.div_exact(&d), Some(q.clone()), "2^{bits}, {lq}");
            let r = exact_poly(&mut s, lq, 3);
            let b = &d * &r;
            assert_eq!(b.div_exact(&d), Some(r.clone()));
            // One coefficient off: not divisible.
            let mut c = b.coeffs().to_vec();
            c[0] = &c[0] + &Number::from(1);
            assert_eq!(Polynomial::new(c).div_exact(&d), None);
        }
    }
}

#[test]
fn pseudo_division_by_a_constant() {
    let mut s = 0xDEAD_BEEF_0123_4567;
    let lens = [1, 2, 3, size(40, 6), size(300, 9)];
    let consts = [
        Number::from(1),
        Number::from(-1),
        Number::from(3),
        -(Number::from(3).pow(126) + Number::from(1)),
        pow2(200),
    ];
    for &m in &lens {
        let a = exact_poly(&mut s, m, 9);
        for l in &consts {
            let d = Polynomial::constant(l.clone());
            let (q, r) = a.pseudo_div_rem(&d);
            assert!(r.is_zero());
            // l^m a = q l.
            let lm = Polynomial::constant(l.pow(m as u32));
            assert_eq!(&q * &d, &lm * &a, "m = {m}, l = {l}");
        }
    }
    let (q, r) = Polynomial::<Number>::zero().pseudo_div_rem(&ints(&[5]));
    assert!(q.is_zero() && r.is_zero());
}

#[test]
fn evaluation_at_large_points() {
    let mut s = 0x1357_9BDF_2468_ACE0;
    for len in [0usize, 1, 2, 4, 5, 6, 9, 17, size(64, 20), size(300, 24)] {
        let p = poly(&mut s, len, 5);
        let c = p.coeffs().to_vec();
        let mut pts = vec![
            Number::from(0),
            Number::from(1),
            Number::from(-1),
            Number::from(7),
            -pow2(65),
        ];
        pts.push(number(&mut s, size(1500, 40)));
        pts.push(-(Number::from(3).pow(size(2000, 60) as u32) + Number::from(1)));
        for x in &pts {
            let want = oracle_eval(&c, x);
            assert_eq!(p.eval(x), want, "len {len}");
            assert_eq!(p.eval_many(core::slice::from_ref(x)), vec![want]);
        }
    }
}

/// The inverse modulo `x^n` by the recurrence on rationals, the oracle.
fn inv_oracle(a: &Polynomial<Rational>, n: usize) -> Polynomial<Rational> {
    let c = a.coeffs();
    let inv = &Rational::from(1) / &c[0];
    let mut g: Vec<Rational> = vec![inv.clone()];
    for j in 1..n {
        let mut acc = Rational::default();
        for i in 1..=j.min(c.len() - 1) {
            acc = &acc + &(&c[i] * &g[j - i]);
        }
        g.push(-&(&acc * &inv));
    }
    g.truncate(n);
    Polynomial::new(g)
}

#[test]
fn rational_power_series_inverses_through_the_integers() {
    let mut s = 0x2222_4444_6666_8888;
    for (len, n) in [
        (1, 5),
        (2, 9),
        (3, 8),
        (5, 17),
        (17, size(40, 12)),
        (size(30, 6), size(70, 10)),
    ] {
        for round in 0..3 {
            let mut c: Vec<Rational> = (0..len).map(|_| rational(&mut s, 6)).collect();
            while c[0].is_zero() {
                c[0] = rational(&mut s, 6);
            }
            if round == 1 {
                // A constant sharing factors with the common denominator, negative.
                c[0] = Rational::new(Number::from(-12), Number::from(35));
            }
            if round == 2 {
                // Integers only.
                for x in c.iter_mut() {
                    *x = Rational::from(Number::from(3).pow(40) + x.numer().clone());
                }
            }
            let a = Polynomial::new(c);
            let g = a.inv_series(n);
            assert_eq!(g, inv_oracle(&a, n), "len {len}, n {n}, round {round}");
            let one = (&a * &g).truncate(n);
            assert_eq!(one, Polynomial::one());
        }
    }
    // The fuzz case (q_series 3003927): an integer series of 0 and ±1 whose Newton steps
    // multiply operands with zeros at the top.
    let c = [
        -1i64, 0, -1, 1, -1, 1, -1, 1, 1, 0, 0, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1,
        -1, 1, -1, 0, -1, -1, -1, 0, -1, -1, 1, 1, 1, -1, 1, 1, -1, 0, 1, -1, -1, -1, -1, -1, 0, 1,
        -1, -1, -1, -1, -1, 1, 1, 1, 0, 1, 1, 1,
    ];
    let a = rats(&c);
    let n = size(600, 40);
    assert_eq!(a.inv_series(n), inv_oracle(&a, n));
    // 1 / (1 - x/2) = Σ x^i / 2^i.
    let a = Polynomial::new(vec![
        Rational::from(1),
        Rational::new((-1).into(), 2.into()),
    ]);
    let g = a.inv_series(size(20, 9));
    for (i, c) in g.coeffs().iter().enumerate() {
        assert_eq!(*c, Rational::new(1.into(), Number::from(2).pow(i as u32)));
    }
}

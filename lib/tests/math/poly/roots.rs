//! The roots in double precision: `Polynomial<f64>::roots`, `Polynomial<Complex<f64>>::roots`
//! and the companion matrices, against known roots, backward errors evaluated at 256 bits, and
//! the eigenvalues of the companion matrices.

use super::common::*;
use a::math::complex::Complex;
use a::math::float::{BigComplex, BigFloat, Round};
use a::math::matrix::Matrix;
use a::math::poly::Polynomial;
use a::math::rational::Rational;

type C = Complex<f64>;

/// The backward error of `z` as a root of the coefficients `c`: `|p(z)|/Σ|cᵢ||z|ⁱ`, at 256 bits.
fn backward(c: &[C], z: C) -> f64 {
    let w = 256;
    let zb = BigComplex::from(z);
    let az = BigFloat::from(z.abs());
    let mut v = BigComplex::zero(w);
    let mut s = BigFloat::zero(w);
    for a in c.iter().rev() {
        v = v.mul_round(&zb, w, Round::Nearest).0;
        v = v.add_round(&BigComplex::from(*a), w, Round::Nearest).0;
        s = s.mul_round(&az, w, Round::Nearest).0;
        s = s.add_round(&BigFloat::from(a.abs()), w, Round::Nearest).0;
    }
    if s.is_zero() {
        return 0.0;
    }
    v.abs_round(64, Round::Nearest)
        .0
        .div_round(&s, 64, Round::Nearest)
        .0
        .to_f64()
}

fn real(c: &[f64]) -> Vec<C> {
    c.iter().map(|&x| C::new(x, 0.0)).collect()
}

/// The coefficients of `Π (x − rᵢ)`.
fn from_roots(r: &[C]) -> Vec<C> {
    let mut c = vec![C::new(1.0, 0.0)];
    for &z in r {
        let mut d = vec![C::new(0.0, 0.0); c.len() + 1];
        for (i, &x) in c.iter().enumerate() {
            d[i + 1] += x;
            d[i] -= z * x;
        }
        c = d;
    }
    c
}

/// Whether `r` is ordered by the real and then the imaginary parts.
fn ordered(r: &[C]) -> bool {
    r.windows(2)
        .all(|w| w[0].re < w[1].re || (w[0].re == w[1].re && w[0].im <= w[1].im))
}

/// Matches each of `a` with a distinct nearest of `b`, the largest distance.
fn matching(a: &[C], b: &[C]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut used = vec![false; b.len()];
    let mut worst = 0f64;
    for x in a {
        let (j, d) = b
            .iter()
            .enumerate()
            .filter(|(j, _)| !used[*j])
            .map(|(j, y)| (j, (*x - *y).abs()))
            .min_by(|p, q| p.1.total_cmp(&q.1))
            .unwrap();
        used[j] = true;
        worst = worst.max(d);
    }
    worst
}

fn rnd(s: &mut u64) -> f64 {
    (xorshift(s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

#[test]
fn known_roots() {
    let r = Polynomial::new(vec![2.0, -3.0, 1.0]).roots();
    assert_eq!(r.len(), 2);
    assert!((r[0] - C::new(1.0, 0.0)).abs() < 1e-15 && r[0].im == 0.0);
    assert!((r[1] - C::new(2.0, 0.0)).abs() < 1e-15 && r[1].im == 0.0);
    let r = Polynomial::new(vec![1.0, 0.0, 1.0]).roots();
    assert_eq!(r[1], r[0].conj());
    assert!((r[1] - C::I).abs() < 1e-15);
    // Degree 1, a constant, roots at 0.
    assert_eq!(Polynomial::new(vec![3.0, 2.0]).roots(), [C::new(-1.5, 0.0)]);
    assert!(Polynomial::new(vec![5.0]).roots().is_empty());
    let r = Polynomial::new(vec![0.0, 0.0, -1.0, 1.0]).roots();
    assert_eq!(r[..2], [C::new(0.0, 0.0); 2]);
    assert!((r[2] - C::new(1.0, 0.0)).abs() < 1e-15);
    let r = Polynomial::new(vec![C::new(0.0, -1.0), C::new(1.0, 0.0)]).roots();
    assert_eq!(r, [C::I]);
}

#[test]
fn roots_of_unity() {
    let ns: &[usize] = if cfg!(miri) {
        &[2, 3, 5]
    } else {
        &[2, 3, 5, 8, 17, 64, 255]
    };
    for &n in ns {
        let mut c = vec![0.0; n + 1];
        c[0] = -1.0;
        c[n] = 1.0;
        let r = Polynomial::new(c).roots();
        assert_eq!(r.len(), n);
        assert!(ordered(&r));
        let mut exact: Vec<C> = (0..n)
            .map(|k| C::from_polar(1.0, core::f64::consts::TAU * k as f64 / n as f64))
            .collect();
        exact.sort_by(|a, b| a.re.total_cmp(&b.re));
        assert!(matching(&r, &exact) < 1e-13 * n as f64, "n = {n}");
        // Real coefficients: exact conjugate pairs, one real root 1 (and −1 for an even n).
        let reals = r.iter().filter(|z| z.im == 0.0).count();
        assert_eq!(reals, if n % 2 == 0 { 2 } else { 1 });
        for z in &r {
            assert!(r.contains(&z.conj()));
        }
    }
}

#[test]
fn multiple_roots_are_found_as_one() {
    // (x − 1)⁵, (x + 2)³(x − 0.5)², (x² + 1)²: the clusters are taken for their multiple roots.
    let p = Polynomial::new(vec![-1.0, 5.0, -10.0, 10.0, -5.0, 1.0]);
    for z in p.roots() {
        assert!((z - C::new(1.0, 0.0)).abs() < 1e-12, "{z:?}");
    }
    let c = from_roots(&[
        C::new(-2.0, 0.0),
        C::new(-2.0, 0.0),
        C::new(-2.0, 0.0),
        C::new(0.5, 0.0),
        C::new(0.5, 0.0),
    ]);
    let r = Polynomial::new(c.iter().map(|z| z.re).collect()).roots();
    for (z, e) in r.iter().zip([-2.0, -2.0, -2.0, 0.5, 0.5]) {
        assert!((z.re - e).abs() < 1e-10 && z.im == 0.0, "{z:?}");
    }
    let r = Polynomial::new(vec![1.0, 0.0, 2.0, 0.0, 1.0]).roots();
    for z in &r {
        assert!((z.abs() - 1.0).abs() < 1e-12 && z.re.abs() < 1e-12, "{z:?}");
    }
    assert_eq!(r[0], r[1]);
    // Complex multiple roots: (x − i)³.
    let c = from_roots(&[C::I, C::I, C::I]);
    for z in Polynomial::new(c).roots() {
        assert!((z - C::I).abs() < 1e-12, "{z:?}");
    }
}

#[test]
fn wilkinson() {
    // ∏ (x − i) for i = 1..=n: the large roots are ill conditioned, the small ones not; the
    // backward error is small for all.
    let ns: &[usize] = if cfg!(miri) { &[5] } else { &[5, 10, 15, 20] };
    for &n in ns {
        let r: Vec<C> = (1..=n).map(|i| C::new(i as f64, 0.0)).collect();
        let c = from_roots(&r);
        let got = Polynomial::new(c.iter().map(|z| z.re).collect()).roots();
        assert_eq!(got.len(), n);
        for z in &got {
            assert!(
                backward(&c, *z) < 1e-14,
                "n {n}: {z:?} {}",
                backward(&c, *z)
            );
        }
        assert!((got[0] - C::new(1.0, 0.0)).abs() < 1e-12);
    }
}

#[test]
fn random_real_polynomials_have_small_backward_errors() {
    let mut s = 0x1234_5678;
    for t in 0..size(200, 4) {
        let n = 2 + below(
            &mut s,
            if t % 10 == 0 {
                size(300, 8)
            } else {
                size(40, 8)
            } as u64,
        ) as usize;
        let mut c: Vec<f64> = (0..=n).map(|_| rnd(&mut s)).collect();
        // Coefficients of very different sizes now and then.
        if t % 3 == 1 {
            for (i, x) in c.iter_mut().enumerate() {
                *x *= 2f64.powi(((i * 37 + t) % 41) as i32 - 20);
            }
        }
        let p = Polynomial::new(c.clone());
        let r = p.roots();
        assert_eq!(r.len(), p.degree().unwrap());
        assert!(ordered(&r));
        let cc = real(&c);
        for z in &r {
            assert!(z.re.is_finite() && z.im.is_finite());
            assert!(
                backward(&cc, *z) < 64.0 * f64::EPSILON,
                "t {t} n {n}: {z:?}"
            );
            // Pairs of exact conjugates.
            if z.im != 0.0 {
                assert!(r.contains(&z.conj()));
            }
        }
    }
}

#[test]
fn random_complex_polynomials_have_small_backward_errors() {
    let mut s = 0x8765_4321;
    for _ in 0..size(100, 3) {
        let n = 1 + below(&mut s, size(50, 6) as u64) as usize;
        let c: Vec<C> = (0..=n).map(|_| C::new(rnd(&mut s), rnd(&mut s))).collect();
        let r = Polynomial::new(c.clone()).roots();
        assert_eq!(r.len(), n);
        assert!(ordered(&r));
        for z in &r {
            assert!(backward(&c, *z) < 64.0 * f64::EPSILON, "{z:?}");
        }
    }
}

#[test]
fn roots_agree_with_the_eigenvalues_of_the_companion_matrix() {
    let mut s = 0xC0FF_EE00;
    for _ in 0..size(40, 2) {
        // Well-conditioned: roots chosen on and inside the unit disc, apart.
        let n = 2 + below(&mut s, 12) as usize;
        let roots: Vec<C> = (0..n)
            .map(|k| C::from_polar(0.5 + 0.5 * (k as f64 / n as f64), 0.3 + 2.0 * k as f64))
            .collect();
        let c = from_roots(&roots);
        let p = Polynomial::new(c.clone());
        let mine = p.roots();
        let eig = p.companion().eigenvalues();
        assert!(matching(&mine, &eig) < 1e-10, "{mine:?} {eig:?}");
        assert!(matching(&mine, &roots) < 1e-10);
        // The real part of the same polynomial, real coefficients.
        let pr = Polynomial::new(c.iter().map(|z| z.re).collect());
        let mine = pr.roots();
        let eig = pr.companion().eigenvalues();
        assert!(matching(&mine, &eig) < 1e-8, "{mine:?} {eig:?}");
    }
}

#[test]
fn companion_matrices() {
    let p = Polynomial::new(vec![2.0, -3.0, 1.0]);
    assert_eq!(p.companion(), Matrix::from_rows(&[[0.0, -2.0], [1.0, 3.0]]));
    assert_eq!(Polynomial::new(vec![4.0]).companion().rows(), 0);
    // 2x³ − x + 6 over the rationals: last column −aᵢ/aₙ, ones below the diagonal.
    let q = rats(&[6, -1, 0, 2]);
    let c = q.companion();
    let half = |n: i64| Rational::new(n.into(), 2.into());
    assert_eq!(c[(0, 2)], Rational::from(-3));
    assert_eq!(c[(1, 2)], half(1));
    assert_eq!(c[(2, 2)], Rational::from(0));
    assert_eq!(c[(1, 0)], Rational::from(1));
    assert_eq!(c[(2, 1)], Rational::from(1));
    assert_eq!(c[(0, 0)], Rational::from(0));
    // det(C) = (−1)ⁿ·a₀/aₙ.
    assert_eq!(c.det(), Rational::from(-3));
    let z = Polynomial::new(vec![C::new(0.0, 2.0), C::new(0.0, 1.0)]).companion();
    assert_eq!(z[(0, 0)], C::new(-2.0, 0.0));
}

#[test]
#[should_panic(expected = "the zero polynomial has every number as a root")]
fn roots_of_zero_panic() {
    Polynomial::<f64>::zero().roots();
}

#[test]
#[should_panic(expected = "a coefficient is not finite")]
fn roots_of_nan_panic() {
    Polynomial::new(vec![1.0, f64::NAN, 1.0]).roots();
}

#[test]
#[should_panic(expected = "a coefficient is not finite")]
fn complex_roots_of_infinity_panic() {
    Polynomial::new(vec![C::new(1.0, f64::INFINITY), C::ONE]).roots();
}

#[test]
#[should_panic(expected = "the zero polynomial has no companion matrix")]
fn companion_of_zero_panics() {
    Polynomial::<Rational>::zero().companion();
}

/// The generator of the fuzz findings F1 and F2 (a linear congruential one).
struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Real roots in [−2, 2] and conjugate pairs in the unit square, multiplied in `f64`.
fn product_real(d: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg(seed * 1000 + d as u64);
    let mut p = vec![1.0f64];
    while p.len() <= d {
        if r.f() < 0.5 || p.len() == d {
            let z = 4.0 * r.f() - 2.0;
            let mut q = vec![0.0; p.len() + 1];
            for (j, &x) in p.iter().enumerate() {
                q[j + 1] += x;
                q[j] -= x * z;
            }
            p = q;
        } else {
            let (a, b) = (2.0 * r.f() - 1.0, 2.0 * r.f() - 1.0);
            let (s1, s0) = (-2.0 * a, a * a + b * b);
            let mut q = vec![0.0; p.len() + 2];
            for (j, &x) in p.iter().enumerate() {
                q[j + 2] += x;
                q[j + 1] += x * s1;
                q[j] += x * s0;
            }
            p = q;
        }
    }
    p
}

/// `d` roots in the square [−1, 1]², multiplied in `Complex<f64>`.
fn product_complex(d: usize, seed: u64) -> Vec<C> {
    let mut r = Lcg(seed * 1000 + d as u64);
    let z: Vec<C> = (0..d)
        .map(|_| C::new(2.0 * r.f() - 1.0, 2.0 * r.f() - 1.0))
        .collect();
    from_roots(&z)
}

/// The checks of the findings: as many roots as the degree, each a root of a polynomial within
/// 2^−40 of the coefficients, no value repeated (the roots are distinct to far more than the
/// rounding), and for a real polynomial the non-real ones in exact conjugate pairs.
fn check_products(c: &[C], r: &[C], real_poly: bool) {
    assert_eq!(r.len(), c.len() - 1);
    for z in r {
        let b = backward(c, *z);
        assert!(
            b <= 2f64.powi(-40),
            "degree {}: {z:?} backward error {b:e}",
            c.len() - 1
        );
    }
    for (i, z) in r.iter().enumerate() {
        assert!(
            !r[i + 1..].contains(z),
            "degree {}: {z:?} twice",
            c.len() - 1
        );
        if real_poly && z.im != 0.0 {
            assert!(r.contains(&z.conj()), "{z:?} without its conjugate");
        }
    }
}

#[test]
fn products_of_random_roots() {
    // Fuzz findings F1 (unconverged values: the last step after |p| reached its rounding jumped
    // far near close roots; far partners averaged into conjugates) and F2 (one value 33 times:
    // approximations on one another kept together by Newton's step, and tens of distinct roots
    // taken for one multiple root).
    let reals: &[(usize, u64)] = if cfg!(miri) {
        &[(12, 1)]
    } else {
        &[
            (101, 1),
            (113, 2),
            (200, 5),
            (250, 4),
            (300, 2),
            (300, 5),
            (370, 1),
        ]
    };
    for &(d, s) in reals {
        let c = product_real(d, s);
        let r = Polynomial::new(c.clone()).roots();
        check_products(&real(&c), &r, true);
    }
    // The root 0.23820553332920967 of `product_real(113, 2)` was missing.
    if !cfg!(miri) {
        let r = Polynomial::new(product_real(113, 2)).roots();
        assert!(
            r.iter()
                .any(|z| z.im == 0.0 && (z.re - 0.238_205_533_329_209_67).abs() < 1e-9)
        );
    }
    let cs: &[(usize, u64)] = if cfg!(miri) {
        &[(10, 0)]
    } else {
        &[(114, 0), (150, 2), (161, 3), (250, 3), (300, 2)]
    };
    for &(d, s) in cs {
        let c = product_complex(d, s);
        let r = Polynomial::new(c.clone()).roots();
        check_products(&c, &r, false);
    }
}

#[test]
fn roots_beyond_the_square_root_of_the_range() {
    // Fuzz finding F3: the squares in the Aberth sum overflowed past 1e154 and the iteration
    // stopped at −4.2e157 for −1e200.
    let r = Polynomial::new(vec![1.0, 0.0, 1e200, 1.0]).roots();
    assert!((r[0].re / -1e200 - 1.0).abs() < 1e-12, "{r:?}");
    let r = Polynomial::new(vec![1.0, 2.0, 3.0, -1e100, -1e-100]).roots();
    assert!((r[0].re / -1e200 - 1.0).abs() < 1e-12, "{r:?}");
    let c = [1.0, 2.0, 3.0, -1e100, -1e-100];
    for z in &r {
        assert!(backward(&real(&c), *z) <= 2f64.powi(-40), "{z:?}");
    }
}

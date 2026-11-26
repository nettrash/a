//! The fast paths of the speed pass against plain oracles: polynomials modulo 2 (packed
//! products, divisions, gcds, powers and factorisations), the word kernels of the linear helpers
//! (derivatives, scaling, sums and differences, evaluation), the roots modulo small primes, the
//! content over the integers, the compositions over the rationals and the products of floats.

use super::common::{below, number, size, xorshift};
use a::math::complex::Complex;
use a::math::numbers::Number;
use a::math::poly::{ModPolynomial, Polynomial};
use a::math::rational::Rational;

/// splitmix64: a generator that is not linear over GF(2). (A bit of xorshift is an LFSR sequence
/// of order at most 64, a very structured class of polynomials over GF(2), whose Euclidean runs
/// are short.)
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `len` random coefficients modulo 2 (the top one 1), 64 from each draw of [`splitmix`].
fn bits(state: &mut u64, len: usize) -> Vec<u64> {
    let mut v: Vec<u64> = Vec::with_capacity(len);
    while v.len() < len {
        let w = splitmix(state);
        v.extend((0..64.min(len - v.len())).map(|j| (w >> j) & 1));
    }
    if let Some(t) = v.last_mut() {
        *t = 1;
    }
    v
}

/// The product over GF(2) by the schoolbook method.
fn gf2_school(a: &[u64], b: &[u64]) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![0u64; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x == 1 {
            for (j, &y) in b.iter().enumerate() {
                c[i + j] ^= y;
            }
        }
    }
    while c.last() == Some(&0) {
        c.pop();
    }
    c
}

fn m2(c: &[u64]) -> ModPolynomial {
    ModPolynomial::from_u64(c, 2)
}

#[test]
fn products_modulo_two_agree_with_the_schoolbook_product() {
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    // Around the switch to Karatsuba's method (28 words: 1,792 coefficients), one and two
    // steps of it.
    let lens = [1usize, 2, 63, 64, 65, 127, 700, 1728, 1729, 3000, 3600];
    for &la in &lens[..size(lens.len(), 5)] {
        for &lb in &lens[..size(lens.len(), 5)] {
            let a = bits(&mut s, la);
            let b = bits(&mut s, lb);
            let p = &m2(&a) * &m2(&b);
            assert_eq!(p.word_coeffs().unwrap(), gf2_school(&a, &b), "{la} x {lb}");
            let q = &m2(&a) * &m2(&a);
            assert_eq!(
                q.word_coeffs().unwrap(),
                gf2_school(&a, &a),
                "square of {la}"
            );
        }
    }
}

#[test]
fn divisions_gcds_and_powers_modulo_two() {
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    for _ in 0..size(60, 6) {
        let la = 1 + below(&mut s, size(2000, 200) as u64) as usize;
        let lb = 1 + below(&mut s, size(1200, 150) as u64) as usize;
        let (a, d) = (m2(&bits(&mut s, la)), m2(&bits(&mut s, lb)));
        let (q, r) = a.div_rem(&d);
        assert_eq!(&(&q * &d) + &r, a);
        assert!(r.length() < d.length());
        assert_eq!(&a % &d, r);
        // gcd: a common divisor of both, and the cofactors coprime.
        let f = m2(&bits(&mut s, 1 + la / 3));
        let (x, y) = (&a * &f, &d * &f);
        let g = x.gcd(&y);
        assert!((&x % &g).is_zero() && (&y % &g).is_zero());
        assert!((&x / &g).gcd(&(&y / &g)).is_one());
        // x^e and a^e modulo d by repeated products.
        if d.length() >= 2 {
            let e = below(&mut s, 40);
            let xp = ModPolynomial::x(&Number::from(2u8));
            let mut want = ModPolynomial::one(&Number::from(2u8)) % &d;
            let mut wa = want.clone();
            for _ in 0..e {
                want = want.mul_mod(&xp, &d);
                wa = wa.mul_mod(&a, &d);
            }
            assert_eq!(xp.pow_mod(&Number::from(e), &d), want);
            assert_eq!(a.pow_mod(&Number::from(e), &d), wa);
        }
    }
}

/// `q·d + r == a` with `deg r < deg d`: the quotient and the remainder, whatever computed them.
fn check_div_rem(a: &ModPolynomial, d: &ModPolynomial) -> (ModPolynomial, ModPolynomial) {
    let (q, r) = a.div_rem(d);
    assert_eq!(&(&q * d) + &r, *a, "{} by {}", a.length(), d.length());
    assert!(r.length() < d.length());
    assert_eq!(&(a % d), &r);
    (q, r)
}

#[test]
fn long_divisions_gcds_and_powers_modulo_two() {
    // Past the crossovers to Barrett's method on polynomials (a divisor of degree 65,536 for one
    // division, 14,336 for the many of pow_mod, and a quotient of 4,096 coefficients; 200, 130
    // and 70 under Miri) and to the half gcd (degree 6,144 in the shorter operand, its base
    // degree 768; 300 and 100 under Miri), with random dense operands, so that the Euclidean
    // runs are long.
    let mut s = 0x0F1E_2D3C_4B5A_6978u64;
    let (dn, qn) = (size(65_536, 200), size(4096, 70));
    for &(ld, lq) in &[
        (dn + 1, qn),
        (dn + 1, qn - 1),
        (dn + 40, dn + 40),
        (dn + 3, dn + dn / 2 + 11),
    ] {
        let d = m2(&bits(&mut s, ld));
        let a = m2(&bits(&mut s, ld + lq - 1));
        check_div_rem(&a, &d);
        // A multiple plus a short remainder.
        let r0 = m2(&bits(&mut s, ld / 2));
        let (q, r) = check_div_rem(&(&(&a * &d) + &r0), &d);
        assert_eq!((q, r), (a.clone(), r0));
    }
    // pow_mod modulo a long f against the exact power reduced once.
    let dr = size(14_336, 130);
    let f = m2(&bits(&mut s, dr + 9));
    let b = m2(&bits(&mut s, dr - 5));
    for e in [0u64, 1, 2, 3, 5] {
        let mut p = ModPolynomial::one(&Number::from(2u8));
        for _ in 0..e {
            p = &p * &b;
        }
        let (_, want) = check_div_rem(&p, &f);
        assert_eq!(b.pow_mod(&Number::from(e), &f), want, "b^{e}");
    }
    let e = 5 * dr as u64 + 17;
    let mut xe = vec![0u64; e as usize + 1];
    xe[e as usize] = 1;
    let (_, want) = check_div_rem(&m2(&xe), &f);
    assert_eq!(m2(&[0, 1]).pow_mod(&Number::from(e), &f), want, "x^{e}");
    // gcds by the half gcd: a planted common factor, against the extended gcd (the half gcd of
    // the word engine, with its own divisions) and its cofactors.
    let gn = size(6144, 300);
    for &(la, lb, lg) in &[
        (gn + 5, gn + 1, 1),
        (gn + size(900, 30), gn + 20, size(700, 40)),
        (2 * gn, gn + 3, 3),
        (size(40_000, 900), size(33_000, 700), size(500, 20)),
    ] {
        let g0 = m2(&bits(&mut s, lg));
        let (x, y) = (&m2(&bits(&mut s, la)) * &g0, &m2(&bits(&mut s, lb)) * &g0);
        let g = x.gcd(&y);
        assert!((&x % &g).is_zero() && (&y % &g).is_zero());
        assert!((&g % &g0).is_zero());
        assert!((&x / &g).gcd(&(&y / &g)).is_one());
        assert_eq!(y.gcd(&x), g);
        if la == gn + 5 {
            let (h, u, v) = x.xgcd(&y);
            assert_eq!(h, g);
            assert_eq!(&(&u * &x) + &(&v * &y), g);
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn products_modulo_two_by_the_word_transforms() {
    // Past the switch to the transforms on words (from 4,096 words in the shorter operand when
    // la^0.585·lb ≥ 12.25·(la + lb)·log₂(64(la + lb)), in words: from 29,537 words against
    // twice as many; here 2.9% past it, never under Miri), both the product of words and the
    // packed product's: a sparse shorter operand of 1,945,600 coefficients (30,400 words)
    // against a dense one twice as long, against the sum of the dense one's shifted copies, one
    // for each term of the sparse one, on packed words.
    let mut s = 0x7A5E_0D15_C0FF_EE01u64;
    let (la, lb) = (1_945_600usize, 3_891_200usize);
    let mut a = vec![0u64; la];
    a[la - 1] = 1;
    let mut terms = vec![la - 1];
    for _ in 0..40 {
        let t = below(&mut s, la as u64) as usize;
        if a[t] == 0 {
            a[t] = 1;
            terms.push(t);
        }
    }
    let b = bits(&mut s, lb);
    let mut bw = vec![0u64; lb.div_ceil(64)];
    for (i, &c) in b.iter().enumerate() {
        bw[i / 64] |= c << (i % 64);
    }
    let mut acc = vec![0u64; (la + lb).div_ceil(64) + 1];
    for &t in &terms {
        let (q, r) = (t / 64, t % 64);
        for (i, &x) in bw.iter().enumerate() {
            acc[q + i] ^= x << r;
            if r > 0 {
                acc[q + i + 1] ^= x >> (64 - r);
            }
        }
    }
    let want: Vec<u64> = (0..la + lb - 1)
        .map(|i| (acc[i / 64] >> (i % 64)) & 1)
        .collect();
    // On the words of both, and on the packed forms of both (made packed by a product by 1),
    // which take the packed product's own switch.
    let one = m2(&[1]);
    let (pa, pb) = (&m2(&a) * &one, &m2(&b) * &one);
    for p in [&m2(&a) * &m2(&b), &pa * &pb] {
        assert_eq!(p.word_coeffs().unwrap(), &want[..]);
    }
}

#[test]
fn factorisations_modulo_two_multiply_back() {
    let mut s = 0xDEAD_BEEF_F00D_1234u64;
    for _ in 0..size(30, 3) {
        let d = 1 + below(&mut s, size(400, 40) as u64) as usize;
        let mut c = bits(&mut s, d + 1);
        if below(&mut s, 2) == 0 {
            // a repeated factor
            let lg = 2 + below(&mut s, 20) as usize;
            let g = m2(&bits(&mut s, lg));
            c = (&(&m2(&c) * &g) * &g).word_coeffs().unwrap().to_vec();
        }
        let f = m2(&c);
        let (l, fs) = f.factor();
        let mut p = ModPolynomial::with_coeffs(&f, &[l]);
        for (g, e) in &fs {
            assert!(g.is_irreducible() && g.is_monic());
            p = &p * &g.pow(u64::from(*e));
        }
        assert_eq!(p, f);
        assert_eq!(f.factor_berlekamp().1, fs);
        let sq = f.squarefree().1;
        let mut q = ModPolynomial::one(&Number::from(2u8));
        for (g, e) in &sq {
            q = &q * &g.pow(u64::from(*e));
        }
        assert_eq!(q, f);
    }
}

#[test]
fn linear_helpers_modulo_words_agree_with_the_integers() {
    let mut s = 0x0F0F_F0F0_1234_4321u64;
    let moduli = [
        2u64,
        3,
        65521,
        (1 << 50) - 27,
        (1 << 63) - 25,
        (1 << 63) + 1,
        u64::MAX - 1,
        u64::MAX - 58,
    ];
    for &n in &moduli {
        let nn = Number::from(n);
        for _ in 0..size(8, 2) {
            let la = 1 + below(&mut s, size(3000, 80) as u64) as usize;
            let lb = 1 + below(&mut s, size(3000, 80) as u64) as usize;
            let a: Vec<u64> = (0..la).map(|_| below(&mut s, n)).collect();
            let b: Vec<u64> = (0..lb).map(|_| below(&mut s, n)).collect();
            let (pa, pb) = (
                ModPolynomial::from_u64(&a, n),
                ModPolynomial::from_u64(&b, n),
            );
            let (za, zb) = (pa.to_polynomial(), pb.to_polynomial());
            let back = |p: &Polynomial<Number>| ModPolynomial::from_polynomial(p, &nn);
            assert_eq!(
                pa.derivative(),
                back(&za.derivative()),
                "derivative modulo {n}"
            );
            assert_eq!(&pa + &pb, back(&(&za + &zb)), "sum modulo {n}");
            assert_eq!(&pa - &pb, back(&(&za - &zb)), "difference modulo {n}");
            let mut t = pa.clone();
            t -= &pb;
            assert_eq!(t, &pa - &pb);
            let c = Number::from(below(&mut s, n));
            assert_eq!(pa.scale(&c), back(&za.scale(&c)), "scale modulo {n}");
            let x = number(&mut s, 12);
            assert_eq!(pa.eval(&x), za.eval(&x).rem_euclid(&nn), "value modulo {n}");
        }
    }
}

#[test]
fn roots_modulo_small_primes() {
    let mut s = 0x5555_AAAA_3333_CCCCu64;
    for &p in &[2u64, 3, 5, 7, 31, 61] {
        for _ in 0..size(20, 3) {
            // Known roots with multiplicities times a factor without roots or with some.
            let mut f = ModPolynomial::one(&Number::from(p));
            let mut want = std::collections::BTreeMap::new();
            for _ in 0..1 + below(&mut s, 4) {
                let r = below(&mut s, p);
                let e = 1 + below(&mut s, 3) as u32;
                f = &f * &ModPolynomial::from_u64(&[(p - r) % p, 1], p).pow(u64::from(e));
                *want.entry(r).or_insert(0u32) += e;
            }
            let g: Vec<u64> = (0..1 + below(&mut s, size(200, 20) as u64))
                .map(|_| below(&mut s, p))
                .chain([1])
                .collect();
            let g = ModPolynomial::from_u64(&g, p);
            for r in 0..p {
                let mut m = 0u32;
                let mut h = g.clone();
                let lin = ModPolynomial::from_u64(&[(p - r) % p, 1], p);
                while h.length() > 1 && h.eval(&Number::from(r)).is_zero() {
                    h = &h / &lin;
                    m += 1;
                }
                if m > 0 {
                    *want.entry(r).or_insert(0) += m;
                }
            }
            let f = &f * &g;
            let got: Vec<(Number, u32)> = f.roots();
            let want: Vec<(Number, u32)> = want
                .into_iter()
                .map(|(r, e)| (Number::from(r), e))
                .collect();
            assert_eq!(got, want, "roots modulo {p}");
        }
    }
}

#[test]
fn contents_of_integer_polynomials() {
    let mut s = 0x7777_1111_2222_9999u64;
    for _ in 0..size(200, 10) {
        let lg = 1 + below(&mut s, 12) as usize;
        let g = number(&mut s, lg);
        let len = 1 + below(&mut s, 40) as usize;
        let c: Vec<Number> = (0..len)
            .map(|_| {
                let lc = 1 + below(&mut s, 20) as usize;
                &number(&mut s, lc) * &g
            })
            .collect();
        let p = Polynomial::new(c.clone());
        let want = c.iter().fold(Number::default(), |acc, x| acc.gcd(x));
        assert_eq!(p.content(), want);
    }
}

#[test]
fn compositions_over_the_rationals_agree_with_horner() {
    let mut s = 0x4242_4242_1717_1717u64;
    let q = |s: &mut u64| {
        let mut d = number(s, 4);
        if d.is_zero() {
            d = Number::from(1u8);
        }
        Rational::new(number(s, 4), d)
    };
    for _ in 0..size(40, 4) {
        let p = Polynomial::new((0..1 + below(&mut s, 8)).map(|_| q(&mut s)).collect());
        let g = Polynomial::new((0..1 + below(&mut s, 6)).map(|_| q(&mut s)).collect());
        let mut want = Polynomial::zero();
        for c in p.coeffs().iter().rev() {
            want = &(&want * &g) + &Polynomial::constant(c.clone());
        }
        assert_eq!(p.compose(&g), want);
    }
}

#[test]
fn float_products_agree_with_the_schoolbook_product() {
    let mut s = 0x0123_4567_89AB_CDEFu64;
    let unit = |s: &mut u64| (xorshift(s) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0;
    for &(la, lb) in &[
        (1usize, 1usize),
        (3, 5),
        (4, 4),
        (5, 300),
        (64, 64),
        (300, 700),
        (2000, 1500),
    ][..size(7, 4)]
    {
        let a: Vec<f64> = (0..la).map(|_| unit(&mut s)).collect();
        let b: Vec<f64> = (0..lb).map(|_| unit(&mut s)).collect();
        let p = &Polynomial::new(a.clone()) * &Polynomial::new(b.clone());
        let mut want = vec![0.0; la + lb - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                want[i + j] += x * y;
            }
        }
        let norm =
            (a.iter().map(|x| x * x).sum::<f64>() * b.iter().map(|x| x * x).sum::<f64>()).sqrt();
        for (x, y) in p.coeffs().iter().zip(&want) {
            assert!((x - y).abs() <= 1e-13 * norm.max(1.0), "{la} x {lb}");
        }
        let ca: Vec<Complex<f64>> = a
            .iter()
            .map(|&x| Complex {
                re: x,
                im: -x / 2.0,
            })
            .collect();
        let cb: Vec<Complex<f64>> = b.iter().map(|&x| Complex { re: x / 3.0, im: x }).collect();
        let cp = &Polynomial::new(ca.clone()) * &Polynomial::new(cb.clone());
        for (k, z) in cp.coeffs().iter().enumerate() {
            let mut w = Complex { re: 0.0, im: 0.0 };
            for i in k.saturating_sub(lb - 1)..=k.min(la - 1) {
                let (u, v) = (ca[i], cb[k - i]);
                w.re += u.re * v.re - u.im * v.im;
                w.im += u.re * v.im + u.im * v.re;
            }
            assert!((z.re - w.re).abs() + (z.im - w.im).abs() <= 1e-12 * norm.max(1.0));
        }
    }
}

/// The product of master's schoolbook loop: each `c_k` summed from `+0` over the non-zero `a_i`
/// by increasing `i`, a product then a sum.
fn school_f64(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; a.len() + b.len() - 1];
    for (i, &x) in a.iter().enumerate() {
        if x != 0.0 {
            for (j, &y) in b.iter().enumerate() {
                c[i + j] += x * y;
            }
        }
    }
    c
}

/// [`school_f64`] for complex floats: `(ar·br − ai·bi, ar·bi + ai·br)` added to each part.
fn school_complex(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut c = vec![Complex { re: 0.0, im: 0.0 }; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if x.re != 0.0 || x.im != 0.0 {
            for (j, y) in b.iter().enumerate() {
                c[i + j].re += x.re * y.re - x.im * y.im;
                c[i + j].im += x.re * y.im + x.im * y.re;
            }
        }
    }
    c
}

/// `2^e` exactly, for a normal power (`f64::powi` has no stated precision; Miri varies it).
fn pow2(e: i32) -> f64 {
    assert!((-1022..=1023).contains(&e));
    f64::from_bits(((1023 + e) as u64) << 52)
}

/// The same bits, or both NaN (whose payload and sign no operation promises).
fn same(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()
}

#[test]
fn float_products_keep_the_componentwise_bound() {
    // Coefficients far smaller than the norms, whose products are exact in the schoolbook
    // method: a_i = b_i = 2^-i gives c_k = (min(k, 2n - 2 - k) + 1)·2^-k. A transform loses
    // their relative accuracy entirely (errors of 1e78 to 1e248 relative); the schoolbook
    // method, which the products take up to the documented line, keeps the bound γ_m Σ|a_i b_j|.
    let gamma = |m: usize| m as f64 * f64::EPSILON / 2.0 / (1.0 - m as f64 * f64::EPSILON / 2.0);
    for &n in &[150usize, 160, 200, 256, 300, 400, 441][..size(7, 1)] {
        let a: Vec<f64> = (0..n).map(|i| pow2(-(i as i32))).collect();
        let p = Polynomial::new(a.clone());
        for c in [(&p * &p.clone()).into_coeffs(), p.sqr().into_coeffs()] {
            assert_eq!(c.len(), 2 * n - 1);
            for (k, &x) in c.iter().enumerate() {
                let m = k.min(2 * n - 2 - k) + 1;
                let want = m as f64 * pow2(-(k as i32));
                assert!((x - want).abs() <= gamma(m) * want, "{n} x {n}, c_{k}");
            }
        }
    }
    // Unbalanced, against 5,000 coefficients (the schoolbook method up to 404 there): all
    // terms positive, so Σ|a_i b_j| is the sum itself.
    for &la in &[150usize, 300, 404][..size(3, 1)] {
        let lb = size(5000, 600);
        let a: Vec<f64> = (0..la).map(|i| pow2(-((i % 60) as i32))).collect();
        let b: Vec<f64> = (0..lb).map(|j| pow2(-8 * ((j % 60) as i32))).collect();
        let want = school_f64(&a, &b);
        let c = (&Polynomial::new(a) * &Polynomial::new(b)).into_coeffs();
        for (k, (&x, &w)) in c.iter().zip(&want).enumerate() {
            assert!((x - w).abs() <= 2.0 * gamma(la) * w, "{la} x {lb}, c_{k}");
        }
    }
    // Complex: a_i = (1 + i)·2^-i, (1 + i)² = 2i, so c_k = 2i·m·2^-k.
    for &n in &[64usize, 80, 100][..size(3, 1)] {
        let a: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let t = pow2(-(i as i32));
                Complex { re: t, im: t }
            })
            .collect();
        let p = Polynomial::new(a);
        let c = (&p * &p.clone()).into_coeffs();
        for (k, z) in c.iter().enumerate() {
            let m = k.min(2 * n - 2 - k) + 1;
            let want = 2.0 * m as f64 * pow2(-(k as i32));
            let err = z.re.abs() + (z.im - want).abs();
            assert!(err <= 4.0 * gamma(m + 1) * want, "complex {n} x {n}, c_{k}");
        }
    }
}

#[test]
fn float_products_below_the_crossover_are_the_schoolbook_sums() {
    // Below the line of the transforms each coefficient is the schoolbook's sum bit for bit, so
    // that an infinity or a NaN reaches only the coefficients of its terms, signed zeros and
    // subnormals included; the shapes cover the generic loop and the kernel of 32 (8 complex)
    // coefficients at a time (a first operand of 16 or more, the second of 448 or more and at
    // least 4 times as long; on both sides of each bound, and with a first operand shorter than
    // the kernel's 32), up to the crossover (441 × 441; 300 × 5,000; complex 101 × 101 and 100 ×
    // 2,000).
    let mut s = 0x5EED_F10A_7000_0001u64;
    let value = |s: &mut u64| match below(s, 24) {
        0 | 1 => 0.0,
        2 => -0.0,
        3 => f64::INFINITY,
        4 => f64::NEG_INFINITY,
        5 => f64::NAN,
        6 => f64::from_bits(xorshift(s) & 0x000F_FFFF_FFFF_FFFF),
        7 => pow2(below(s, 1200) as i32 - 600),
        _ => (xorshift(s) as i64) as f64 / 9.2e18,
    };
    let shapes: &[(usize, usize)] = &[
        (1, 1),
        (3, 40),
        (16, 448),
        (16, 447),
        (24, 500),
        (40, 39),
        (15, 460),
        (300, 20),
        (64, 600),
        (112, 447),
        (112, 448),
        (200, 600),
        (300, 300),
        (441, 441),
        (300, 5000),
    ];
    for &(la, lb) in &shapes[..size(shapes.len(), 7)] {
        for special in [false, true] {
            let mut mk = |n: usize| -> Vec<f64> {
                let mut v: Vec<f64> = (0..n)
                    .map(|_| {
                        if special {
                            value(&mut s)
                        } else {
                            (xorshift(&mut s) as i64) as f64 / 9.2e18
                        }
                    })
                    .collect();
                v[n - 1] = 1.5;
                v
            };
            let (a, b) = (mk(la), mk(lb));
            let c = (&Polynomial::new(a.clone()) * &Polynomial::new(b.clone())).into_coeffs();
            let want = school_f64(&a, &b);
            assert_eq!(c.len(), want.len());
            for (k, (x, w)) in c.iter().zip(&want).enumerate() {
                assert!(same(*x, *w), "{la} x {lb}, c_{k}: {x} for {w}");
            }
        }
    }
    // One infinity among 300 × 300 small integers (exact sums): infinite exactly at the
    // coefficients of its terms, every other one exact.
    let n = size(300, 40);
    let mut a: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64).collect();
    a[1] = f64::INFINITY;
    let b: Vec<f64> = (0..n).map(|j| 1.0 + (j % 5) as f64).collect();
    let c = (&Polynomial::new(a.clone()) * &Polynomial::new(b.clone())).into_coeffs();
    for (k, x) in c.iter().enumerate() {
        if (1..=n).contains(&k) {
            assert_eq!(*x, f64::INFINITY, "c_{k}");
        } else {
            let w: f64 = (k.saturating_sub(n - 1)..=k.min(n - 1))
                .map(|i| a[i] * b[k - i])
                .sum();
            assert_eq!(*x, w, "c_{k}");
        }
    }
    let cshapes: &[(usize, usize)] = &[
        (4, 4),
        (8, 64),
        (9, 63),
        (20, 200),
        (64, 64),
        (101, 101),
        (100, 2000),
    ];
    for &(la, lb) in &cshapes[..size(cshapes.len(), 5)] {
        let mut mk = |n: usize| -> Vec<Complex<f64>> {
            let mut v: Vec<Complex<f64>> = (0..n)
                .map(|_| Complex {
                    re: value(&mut s),
                    im: value(&mut s),
                })
                .collect();
            v[n - 1] = Complex { re: 1.0, im: 0.5 };
            v
        };
        let (a, b) = (mk(la), mk(lb));
        let c = (&Polynomial::new(a.clone()) * &Polynomial::new(b.clone())).into_coeffs();
        let want = school_complex(&a, &b);
        for (k, (x, w)) in c.iter().zip(&want).enumerate() {
            assert!(
                same(x.re, w.re) && same(x.im, w.im),
                "complex {la} x {lb}, c_{k}"
            );
        }
    }
}

#[test]
fn float_differences_keep_positive_zeros() {
    // 0 − (+0) = +0: the coefficients of a difference past the end of the minuend, as `-=` and
    // IEEE subtraction give them (a negation would give −0).
    let a = Polynomial::new(vec![1.0f64]);
    let b = Polynomial::new(vec![0.0f64, -0.0, 0.0, 2.0]);
    let d = (&a - &b).into_coeffs();
    let bits: Vec<u64> = d.iter().map(|x| x.to_bits()).collect();
    assert_eq!(bits, [1.0f64, 0.0, 0.0, -2.0].map(f64::to_bits));
    let mut e = a.clone();
    e -= &b;
    assert_eq!(
        e.coeffs().iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        bits
    );
    let z = |re: f64, im: f64| Complex { re, im };
    let ca = Polynomial::new(vec![z(1.0, 0.0)]);
    let cb = Polynomial::new(vec![z(0.0, 0.0), z(0.0, -0.0), z(2.0, 1.0)]);
    let cd = (&ca - &cb).into_coeffs();
    let cbits: Vec<(u64, u64)> = cd
        .iter()
        .map(|w| (w.re.to_bits(), w.im.to_bits()))
        .collect();
    assert_eq!(cbits[1], (0, 0));
    assert_eq!(cbits[2], ((-2.0f64).to_bits(), (-1.0f64).to_bits()));
    let mut ce = ca.clone();
    ce -= &cb;
    let ebits: Vec<(u64, u64)> = ce
        .coeffs()
        .iter()
        .map(|w| (w.re.to_bits(), w.im.to_bits()))
        .collect();
    assert_eq!(ebits, cbits);
}

#[test]
fn packed_results_modulo_two_read_as_words() {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let hash = |p: &ModPolynomial| {
        let mut h = DefaultHasher::new();
        p.hash(&mut h);
        h.finish()
    };
    let two = Number::from(2u8);
    let mut s = 0x2468_ACE0_1357_9BDFu64;
    for _ in 0..size(40, 4) {
        let la = 1 + below(&mut s, size(900, 90) as u64) as usize;
        let lb = 1 + below(&mut s, size(700, 70) as u64) as usize;
        let (a, b) = (bits(&mut s, la), bits(&mut s, lb));
        let (pa, pb) = (m2(&a), m2(&b));
        // Over the integers, then reduced: the oracle for every operation.
        let (za, zb) = (pa.to_polynomial(), pb.to_polynomial());
        let back = |p: &Polynomial<Number>| ModPolynomial::from_polynomial(p, &two);
        let prod = &pa * &pb;
        assert_eq!(prod.word_coeffs().unwrap(), gf2_school(&a, &b));
        assert_eq!(prod, back(&(&za * &zb)));
        assert_eq!(hash(&prod), hash(&back(&(&za * &zb))));
        assert_eq!(prod.length(), la + lb - 1);
        assert_eq!(prod.degree(), Some(la + lb - 2));
        assert_eq!(prod.coeff(0), Number::from(a[0] & b[0]));
        assert_eq!(prod.to_string(), back(&(&za * &zb)).to_string());
        let sq = &pa * &pa;
        assert_eq!(sq, back(&(&za * &za)));
        assert_eq!(&pa + &pb, back(&(&za + &zb)));
        assert_eq!(&pa - &pb, back(&(&za - &zb)));
        assert_eq!(-&prod, prod);
        assert_eq!(prod.derivative(), back(&(&za * &zb).derivative()));
        for x in [0i64, 1, 2, 3, -1, -4] {
            let xn = Number::from(x);
            assert_eq!(prod.eval(&xn), (&za * &zb).eval(&xn).rem_euclid(&two));
        }
        assert!(prod.scale(&Number::from(6u8)).is_zero());
        assert_eq!(prod.scale(&Number::from(-3i64)), prod);
        // Mutation after a packed result, and packed results as operands again.
        let mut t = prod.clone();
        t += &pa;
        assert_eq!(t, back(&(&(&za * &zb) + &za)));
        t -= &prod;
        assert_eq!(t, pa);
        let (q, r) = prod.div_rem(&pb);
        assert_eq!(q, pa);
        assert!(r.is_zero());
        let (q, r) = (&prod + &pa).div_rem(&sq);
        assert_eq!(&(&q * &sq) + &r, &prod + &pa);
        assert!(r.length() < sq.length());
        assert_eq!(&(&prod + &pa) % &sq, r);
        let g = prod.gcd(&(&pb * &pb));
        assert!((&prod % &g).is_zero() && (&(&pb * &pb) % &g).is_zero());
        assert_eq!(g.leading(), Some(Number::from(1u8)));
        if pb.length() >= 2 {
            let e = Number::from(below(&mut s, 50));
            let mut want = ModPolynomial::one(&two) % &pb;
            for _ in 0..u64::try_from(&e).unwrap() {
                want = want.mul_mod(&prod, &pb);
            }
            assert_eq!(prod.pow_mod(&e, &pb), want);
        }
        assert_eq!(prod.coeffs().len(), prod.length());
    }
}

#[test]
fn one_root_one_point_and_constant_early_outs() {
    let mut s = 0x1357_2468_ACE0_BDF1u64;
    let big = &Number::from(1u8) << 127usize;
    let p127 = &big - &Number::from(1u8);
    for _ in 0..size(30, 3) {
        let r = number(&mut s, 24);
        // Over the integers: x - r.
        let z = Polynomial::from_roots(std::slice::from_ref(&r));
        assert_eq!(z, Polynomial::new(vec![-&r, Number::from(1u8)]));
        assert_eq!(z.eval(&r), Number::default());
        for n in [Number::from(65521u32), p127.clone()] {
            let f = ModPolynomial::from_roots(std::slice::from_ref(&r), &n);
            assert_eq!(f.length(), 2);
            assert!(f.eval(&r).is_zero());
            assert_eq!(f, ModPolynomial::from_polynomial(&z, &n));
            let y = number(&mut s, 24);
            let want = ModPolynomial::from_polynomial(&Polynomial::new(vec![y.clone()]), &n);
            for p in [
                ModPolynomial::interpolate(std::slice::from_ref(&r), std::slice::from_ref(&y), &n),
                ModPolynomial::interpolate_newton(
                    std::slice::from_ref(&r),
                    std::slice::from_ref(&y),
                    &n,
                ),
                ModPolynomial::interpolate_lagrange(
                    std::slice::from_ref(&r),
                    std::slice::from_ref(&y),
                    &n,
                ),
            ] {
                assert_eq!(p, want);
            }
            let xs = [number(&mut s, 30), number(&mut s, 30)];
            assert_eq!(want.eval_many(&xs), vec![want.coeff(0); 2]);
        }
        let w = below(&mut s, u64::MAX);
        let n = 2 + below(&mut s, u64::MAX - 2);
        let f = ModPolynomial::from_roots_u64(&[w], n);
        assert_eq!(f, ModPolynomial::from_u64(&[(n - w % n) % n, 1], n));
        let y = below(&mut s, u64::MAX);
        assert_eq!(
            ModPolynomial::interpolate_u64(&[w], &[y], n),
            ModPolynomial::from_u64(&[y], n)
        );
    }
}

//! Rational approximation by the AAA algorithm, [`a::math::numerics::aaa`] and
//! [`a::math::numerics::aaa_complex`], against `scipy.interpolate.AAA` on the problems of the
//! paper and of SciPy's documentation (the data in `data/aaa_*.txt`, made by
//! `data/gen_aaa.py`), and against the functions themselves.

use a::math::complex::Complex;
use a::math::numerics::{Aaa, AaaOptions, aaa, aaa_complex, aaa_complex_with, aaa_with};

/// One problem of `data/aaa_*.txt`: the samples as SciPy keeps them, and what SciPy made of
/// them (clean-up off).
struct Reference {
    points: Vec<Complex<f64>>,
    values: Vec<Complex<f64>>,
    support: Vec<usize>,
    weights: Vec<Complex<f64>>,
    errors: Vec<f64>,
    at: Vec<Complex<f64>>,
    approximation: Vec<Complex<f64>>,
    poles: Vec<Complex<f64>>,
}

fn word(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("16 hexadecimal digits"))
}

fn reference(text: &str) -> Reference {
    let mut lines = text.lines();
    let mut section = |name: &str| -> Vec<Vec<String>> {
        let head = lines.next().expect("a section");
        let mut parts = head.split_whitespace();
        assert_eq!(parts.next(), Some(name));
        let n: usize = parts.next().unwrap().parse().unwrap();
        (0..n)
            .map(|_| {
                lines
                    .next()
                    .unwrap()
                    .split_whitespace()
                    .map(String::from)
                    .collect()
            })
            .collect()
    };
    let samples = section("samples");
    let support = section("terms");
    let weights = section("weights");
    let errors = section("errors");
    let points = section("points");
    let poles = section("poles");
    let pair = |r: &Vec<String>, k: usize| Complex::new(word(&r[k]), word(&r[k + 1]));
    Reference {
        points: samples.iter().map(|r| pair(r, 0)).collect(),
        values: samples.iter().map(|r| pair(r, 2)).collect(),
        support: support.iter().map(|r| r[0].parse().unwrap()).collect(),
        weights: weights.iter().map(|r| pair(r, 0)).collect(),
        errors: errors.iter().map(|r| word(&r[0])).collect(),
        at: points.iter().map(|r| pair(r, 0)).collect(),
        approximation: points.iter().map(|r| pair(r, 2)).collect(),
        poles: poles.iter().map(|r| pair(r, 0)).collect(),
    }
}

/// SciPy's poles within 3 of the origin, near the samples, each against the nearest of `mine`:
/// they agree to the accuracy the weights determine them, which is rounding for the problems
/// whose poles are simple and isolated and less where they cluster. (Poles far from the samples
/// depend on the last digits of the weights, which the two decompositions round differently.)
fn same_poles(name: &str, r: &Reference, mine: &[Complex<f64>]) {
    let tol = match name {
        "tan" | "runge" | "spiral" | "exp" => 1e-13,
        "gamma" => 1e-8,
        "sqrt" => 1e-5,
        "circle" => 1e-4,
        _ => return,
    };
    assert_eq!(mine.len(), r.poles.len(), "{name}: the number of poles");
    for q in r.poles.iter().filter(|q| q.abs() < 3.0) {
        let d = mine
            .iter()
            .map(|p| (p - q).abs())
            .fold(f64::INFINITY, f64::min);
        assert!(
            d < tol,
            "{name}: SciPy's pole {q} is {d:e} from the nearest"
        );
    }
}

const REAL: [(&str, &str); 6] = [
    ("exp", include_str!("data/aaa_exp.txt")),
    ("tan", include_str!("data/aaa_tan.txt")),
    ("abs", include_str!("data/aaa_abs.txt")),
    ("sqrt", include_str!("data/aaa_sqrt.txt")),
    ("runge", include_str!("data/aaa_runge.txt")),
    ("gamma", include_str!("data/aaa_gamma.txt")),
];

const COMPLEX: [(&str, &str); 2] = [
    ("spiral", include_str!("data/aaa_spiral.txt")),
    ("circle", include_str!("data/aaa_circle.txt")),
];

/// The weights divided by the one largest in size, which removes the sign or phase a singular
/// vector is free to have.
fn normalised(w: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let big = w
        .iter()
        .copied()
        .fold(Complex::ZERO, |b, v| if v.abs() > b.abs() { v } else { b });
    w.iter().map(|&v| v / big).collect()
}

/// The values against SciPy's, relative to the largest |f|, and the normalised weights.
fn deviations(
    r: &Reference,
    w: &[Complex<f64>],
    eval: impl Fn(Complex<f64>) -> Complex<f64>,
) -> (f64, f64) {
    let big = r.values.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let dw = normalised(w)
        .iter()
        .zip(normalised(&r.weights))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    let dv =
        r.at.iter()
            .zip(&r.approximation)
            .map(|(&t, &v)| (eval(t) - v).abs() / big)
            .fold(0.0, f64::max);
    (dw, dv)
}

/// The support points are SciPy's, in order, except where the two codes break a tie differently.
/// Each step takes the sample of largest error, and the codes can part where that choice rests on
/// the last bits of the arithmetic, which differ between platforms (the libm's `hypot` in
/// `|f − r|`, the order of the sums in the decompositions); from there they follow different
/// paths of the same algorithm. Two such places are allowed. The first step, when the test finds
/// the choice to be a tie: `|f − mean|`, its error, within 4 ulps at SciPy's support point and at
/// this one. And near convergence, after 90 % of the terms, where a step can be ill-conditioned —
/// its weights the singular vector of a nearly singular Loewner matrix, determined to a few digits
/// only. The number of terms must be the same, and the errors must agree to 1 % at every step
/// before the one where they part. Returns the number of support points that agree.
fn same_support(name: &str, r: &Reference, got: &[Complex<f64>], errors: &[f64]) -> usize {
    let want: Vec<Complex<f64>> = r.support.iter().map(|&i| r.points[i]).collect();
    assert_eq!(got.len(), want.len(), "{name}: the number of terms");
    let k = (0..got.len())
        .find(|&k| got[k] != want[k])
        .unwrap_or(got.len());
    if k == 0 && !got.is_empty() {
        let mean = r.values.iter().fold(Complex::ZERO, |s, &v| s + v) / r.values.len() as f64;
        let error = |p: Complex<f64>| {
            let i = r.points.iter().position(|&q| q == p).expect("a sample");
            (r.values[i] - mean).abs()
        };
        let (mine, theirs) = (error(got[0]), error(want[0]));
        assert!(
            (mine - theirs).abs() <= 4.0 * f64::EPSILON * theirs,
            "{name}: the first support point is not a tie: |f − mean| {mine} for SciPy's {theirs}"
        );
    } else {
        assert!(
            10 * k >= 9 * got.len(),
            "{name}: the support points part at step {k}"
        );
    }
    for (s, (&a, &b)) in errors
        .iter()
        .zip(&r.errors)
        .enumerate()
        .take(k.saturating_sub(1))
    {
        assert!(
            (a - b).abs() <= 1e-2 * b,
            "{name}: error {a} at step {s} for SciPy's {b}"
        );
    }
    k
}

#[test]
fn aaa_chooses_scipys_support_points_on_real_problems() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // exp, tan(πx/2) at midpoints, |x|, √x, Runge's function on 1000 points and Γ on 100, at
    // SciPy's default tolerance, clean-up off: the same terms, the same support points (|x|: the
    // first 36 of 39; its step 36 is ill-conditioned, the error jumping from 1.2e-10 to 2e-8 in
    // both codes and the two differing by a third there, and the 37th support points are
    // neighbouring samples), values within 1e-13 of SciPy's relative to max|f| (Γ's within rtol,
    // below; 2·rtol when the support points part) and weights within 1e-4 (a singular vector of a
    // nearly singular matrix: it is the values that are well determined).
    let opts = AaaOptions::new().with_clean_up(false);
    for (name, text) in REAL {
        let r = reference(text);
        let x: Vec<f64> = r.points.iter().map(|p| p.re).collect();
        let f: Vec<f64> = r.values.iter().map(|p| p.re).collect();
        let a = aaa_with(&x, &f, &opts).unwrap();
        assert!(a.converged(), "{name}");
        let got: Vec<Complex<f64>> = a
            .support_points()
            .iter()
            .map(|&p| Complex::new(p, 0.0))
            .collect();
        let agree = same_support(name, &r, &got, a.errors());
        let w: Vec<Complex<f64>> = a.weights().iter().map(|&p| Complex::new(p, 0.0)).collect();
        let (dw, dv) = deviations(&r, &w, |t| Complex::new(a.evaluate(t.re), 0.0));
        same_poles(name, &r, &a.poles());
        let tied = agree < got.len();
        assert!(tied || dw < 1e-4, "{name}: weights {dw:e} from SciPy's");
        // Γ's values between the samples, near its poles, are determined by the data only to a few
        // 1e-13 of max|f|: changing each sample by 1 to 4 ulps, with the support points the same,
        // moves them by up to 4.0e-13 (a quarter of 400 trials by more than 1e-13), and the last
        // bits of `hypot` in the decompositions do as much (1.6e-13 on x86_64 with Ubuntu's
        // x86-64-v3 glibc, as in CI). They are held to rtol, what the method promises at the
        // samples. The other problems move by 1.7e-14 at most in the same trials.
        let limit = if tied {
            2.0 * opts.rtol
        } else if name == "gamma" {
            opts.rtol
        } else {
            1e-13
        };
        assert!(dv < limit, "{name}: values {dv:e} from SciPy's");
        if !tied {
            // The errors of the early steps, with a pole among the samples, are sensitive to the
            // last bits of the weights: 1e-6 relative, against the 1e-13 of the values.
            assert_eq!(a.errors().len(), r.errors.len(), "{name}");
            let big = f.iter().map(|v| v.abs()).fold(0.0, f64::max);
            for (e, want) in a.errors().iter().zip(&r.errors) {
                assert!(
                    (e - want).abs() < 1e-6 * want + 1e-13 * big,
                    "{name}: error {e} for SciPy's {want}"
                );
            }
        }
    }
}

#[test]
fn aaa_chooses_scipys_support_points_on_complex_problems() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // tan(πz/2) on the paper's spiral and log(2 + z⁴)/(1 − 16z⁴) on the unit circle. The
    // circle's values are symmetric under conjugation and a quarter turn, and the first error,
    // |f − mean|, is the same at the conjugate samples 365 and 366 but for the last bit of
    // `hypot`, which the platforms' libms round differently: SciPy (on macOS) and the aarch64
    // build take 366, the x86_64 build finds the two equal and takes the first. From there the
    // two follow mirrored paths, and what must still agree is what the function determines: the
    // number of terms, the values between the samples — which are good only to about 3e-10 of
    // max|f| there, the test points reaching radius 1.1 and the branch points of the logarithm
    // lying at 2^(1/4) = 1.19, so that two paths differ by 1.6e-10 — and the poles of the
    // function, ±1 of the tangent and ±1/2, ±i/2 here, to rounding.
    let opts = AaaOptions::new().with_clean_up(false);
    for (name, text) in COMPLEX {
        let r = reference(text);
        let a = aaa_complex_with(&r.points, &r.values, &opts).unwrap();
        assert!(a.converged(), "{name}");
        let agree = same_support(name, &r, a.support_points(), a.errors());
        let (dw, dv) = deviations(&r, a.weights(), |t| a.evaluate(t));
        if agree == a.support_points().len() {
            same_poles(name, &r, &a.poles());
            assert!(
                dw < 1e-4 && dv < 1e-13,
                "{name}: weights {dw:e}, values {dv:e}"
            );
        } else {
            assert!(dv < 1e-9, "{name}: values {dv:e} from SciPy's");
        }
        let poles: Vec<Complex<f64>> = if name == "spiral" {
            vec![Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0)]
        } else {
            (0..4)
                .map(|k| Complex::from_polar(0.5, core::f64::consts::FRAC_PI_2 * f64::from(k)))
                .collect()
        };
        for want in poles {
            let d = nearest(&a.poles(), want);
            assert!(d < 1e-12, "{name}: the pole {want} is missed by {d:e}");
        }
    }
}

/// `n` points equally spaced on `[a, b]`.
fn grid(n: usize, a: f64, b: f64) -> Vec<f64> {
    (0..n)
        .map(|k| a + (b - a) * k as f64 / (n - 1) as f64)
        .collect()
}

/// The largest |f − r| over `n` points of `[a, b]` that are not samples.
fn off_samples(r: &Aaa<f64>, f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    (0..n)
        .map(|k| a + (b - a) * (k as f64 + 0.3) / n as f64)
        .map(|t| (r.evaluate(t) - f(t)).abs())
        .fold(0.0, f64::max)
}

#[test]
fn the_approximations_follow_the_functions_between_the_samples() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    let x = grid(1000, -1.0, 1.0);
    let exp: Vec<f64> = x.iter().map(|t| t.exp()).collect();
    let r = aaa(&x, &exp).unwrap();
    assert!(off_samples(&r, f64::exp, -1.0, 1.0, 3000) < 1e-12);
    let runge = |t: f64| 1.0 / (1.0 + 25.0 * t * t);
    let f: Vec<f64> = x.iter().map(|&t| runge(t)).collect();
    let r = aaa(&x, &f).unwrap();
    assert_eq!(r.degree(), 2);
    assert!(off_samples(&r, runge, -1.0, 1.0, 3000) < 1e-14);
    // Γ through two of its poles, between its 100 samples.
    let x = grid(100, -1.5, 1.5);
    let g: Vec<f64> = x.iter().map(|&t| a::math::special::gamma(t)).collect();
    let r = aaa(&x, &g).unwrap();
    assert!(r.converged());
    for k in 0..300 {
        let t = -1.49 + 2.98 * (k as f64 + 0.5) / 300.0;
        let want = a::math::special::gamma(t);
        assert!(
            (r.evaluate(t) - want).abs() < 1e-9 * want.abs().max(1.0),
            "Γ at {t}"
        );
    }
}

#[test]
fn tan_has_its_poles_at_plus_and_minus_one() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // tan(πx/2) at 1000 midpoints of [-1, 1]: the poles at ±1, just beyond the samples, to
    // twelve digits, with the residue −2/π of each.
    let x: Vec<f64> = (0..1000)
        .map(|k| -1.0 + (2.0 * k as f64 + 1.0) / 1000.0)
        .collect();
    let f: Vec<f64> = x
        .iter()
        .map(|t| (core::f64::consts::FRAC_PI_2 * t).tan())
        .collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged());
    let poles = r.poles();
    let residues = r.residues();
    assert_eq!(poles.len(), r.degree());
    for want in [-1.0, 1.0] {
        let k = (0..poles.len())
            .min_by(|&i, &j| {
                (poles[i].re - want)
                    .abs()
                    .total_cmp(&(poles[j].re - want).abs())
            })
            .unwrap();
        assert!(
            (poles[k] - Complex::new(want, 0.0)).abs() < 1e-12,
            "{}",
            poles[k]
        );
        let res = residues[k] - Complex::new(-2.0 / core::f64::consts::PI, 0.0);
        assert!(res.abs() < 1e-9, "residue {}", residues[k]);
    }
    // Poles of real data come in conjugate pairs or are real.
    for p in &poles {
        assert!(p.im == 0.0 || poles.iter().any(|q| (q - p.conj()).abs() < 1e-9 * p.abs()));
    }
}

#[test]
fn runges_function_is_found_exactly() {
    let x = grid(1000, -1.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| 1.0 / (1.0 + 25.0 * t * t)).collect();
    let r = aaa(&x, &f).unwrap();
    assert_eq!(r.support_points().len(), 3);
    let poles = r.poles();
    assert_eq!(poles.len(), 2);
    for (p, res) in poles.iter().zip(r.residues()) {
        assert!(
            p.re.abs() < 1e-14 && (p.im.abs() - 0.2).abs() < 1e-14,
            "{p}"
        );
        // The residue of 1/(1 + 25x²) at ±i/5 is ∓i/10.
        assert!(
            (res - Complex::new(0.0, -0.1 * p.im.signum())).abs() < 1e-13,
            "{res}"
        );
    }
    assert!(r.zeros().iter().all(|z| z.abs() > 1e3), "no zeros in reach");
}

#[test]
fn gamma_has_its_poles_at_zero_and_minus_one() {
    // The residues of Γ at 0 and −1 are 1 and −1.
    let x = grid(100, -1.5, 1.5);
    let g: Vec<f64> = x.iter().map(|&t| a::math::special::gamma(t)).collect();
    let r = aaa(&x, &g).unwrap();
    let poles = r.poles();
    let residues = r.residues();
    for (want, res) in [(0.0, 1.0), (-1.0, -1.0)] {
        let k = (0..poles.len())
            .min_by(|&i, &j| {
                (poles[i].re - want)
                    .abs()
                    .total_cmp(&(poles[j].re - want).abs())
            })
            .unwrap();
        assert!(
            (poles[k] - Complex::new(want, 0.0)).abs() < 1e-12,
            "{}",
            poles[k]
        );
        assert!(
            (residues[k] - Complex::new(res, 0.0)).abs() < 1e-9,
            "{}",
            residues[k]
        );
    }
}

#[test]
fn abs_and_sqrt_converge_root_exponentially() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // Newman's theorem: the best rational approximations of |x| converge like exp(−π√n), and AAA
    // follows at a similar rate, where a polynomial of degree n is only 1/n off. So ln(error)/√m
    // stays below a negative constant at every step, which no algebraic rate can do.
    let x = grid(1000, -1.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| t.abs()).collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged() && r.support_points().len() < 45);
    // The errors of single steps jump (6 terms are worse than 5), so it is the best error of the
    // first m steps that is measured.
    let best = |errors: &[f64], m: usize| errors[..m].iter().copied().fold(f64::INFINITY, f64::min);
    for m in (10..=r.errors().len()).step_by(5) {
        let e = best(r.errors(), m);
        assert!(e.ln() / (m as f64).sqrt() < -1.5, "|x|: {e} by {m} terms");
    }
    // A polynomial of degree 30 is at least 0.28/30 off (Bernstein); 30 terms of AAA are 3e-8.
    assert!(best(r.errors(), 30) < 1e-7);
    let x = grid(1000, 0.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| t.sqrt()).collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged() && r.support_points().len() < 20);
    for m in (5..=r.errors().len()).step_by(5) {
        let e = best(r.errors(), m);
        assert!(e.ln() / (m as f64).sqrt() < -3.0, "√x: {e} by {m} terms");
    }
}

/// A uniform number in [−1/2, 1/2) from a seeded linear congruential generator.
fn noise(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}

#[test]
fn noisy_data_stop_at_the_term_limit_and_say_so() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    let x = grid(500, -1.0, 1.0);
    let mut seed = 7;
    let f: Vec<f64> = x
        .iter()
        .map(|t| t.exp() + 1e-8 * noise(&mut seed))
        .collect();
    let opts = AaaOptions::new().with_max_terms(30);
    let r = aaa_with(&x, &f, &opts).unwrap();
    assert!(!r.converged());
    assert_eq!(r.errors().len(), 30);
    assert!(r.support_points().len() <= 30);
    // The best step is the one returned, whose error is the noise's: neither far below the noise
    // amplitude 5e-9 (overfitting it) nor much above it. The last step is 200 times worse.
    let smallest = r.errors().iter().copied().fold(f64::INFINITY, f64::min);
    assert!((r.error() - smallest).abs() < 1e-6 * smallest);
    assert!(r.error() > 1e-9 && r.error() < 2e-8, "{}", r.error());
    assert!(*r.errors().last().unwrap() > 100.0 * r.error());
}

#[test]
fn too_few_samples_are_used_as_far_as_they_determine_the_weights() {
    // One sample: the constant.
    let r = aaa(&[0.5f64], &[2.0]).unwrap();
    assert!(r.converged() && r.degree() == 0);
    assert_eq!(r.evaluate(7.0), 2.0);
    assert!(r.poles().is_empty() && r.zeros().is_empty());
    // Two: at most one support point (more would leave the weights free); the constant through
    // the first is the best that can be said, and it is not within the tolerance.
    let r = aaa(&[0.0f64, 1.0], &[1.0, 3.0]).unwrap();
    assert_eq!(r.support_points().len(), 1);
    assert!(!r.converged() && r.error() == 2.0);
    // Three: two support points, the rational interpolant of type (1, 1) through all three.
    let r = aaa(&[0.0f64, 1.0, 2.0], &[1.0, 0.5, 1.0 / 3.0]).unwrap();
    assert!(r.converged() && r.degree() == 1);
    assert!((r.evaluate(3.0) - 0.25).abs() < 1e-15, "1/(x + 1)");
    // Four samples of the same function: still two terms, and exact.
    let r = aaa(&[0.0f64, 1.0, 2.0, 5.0], &[1.0, 0.5, 1.0 / 3.0, 1.0 / 6.0]).unwrap();
    assert!(r.converged() && r.degree() == 1);
    let poles = r.poles();
    assert!(poles.len() == 1 && (poles[0] - Complex::new(-1.0, 0.0)).abs() < 1e-14);
}

#[test]
fn repeated_and_non_finite_samples_are_left_out() {
    // A repeated point keeps the first value given (as SciPy), and a sample with a point or a
    // value that is not finite is dropped.
    let x: [f64; 7] = [0.0, 1.0, 1.0, 2.0, 3.0, f64::NAN, 4.0];
    let f = [1.0, 0.5, 9.0, 1.0 / 3.0, f64::INFINITY, 1.0, 0.2];
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged());
    assert!((r.evaluate(1.0) - 0.5).abs() < 1e-15);
    assert!((r.evaluate(3.0) - 0.25).abs() < 1e-15);
    assert!(aaa(&[f64::NAN], &[1.0]).is_none());
    assert!(aaa::<f64>(&[], &[]).is_none());
}

#[test]
fn invalid_options_give_none() {
    let x = grid(20, 0.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| t.exp()).collect();
    for opts in [
        AaaOptions::new().with_rtol(-1.0),
        AaaOptions::new().with_rtol(f64::NAN),
        AaaOptions::new().with_max_terms(0),
        AaaOptions::new().with_clean_up_tol(f64::INFINITY),
    ] {
        assert!(aaa_with(&x, &f, &opts).is_none());
    }
    // rtol = 0 is allowed: it runs to the term limit.
    let r = aaa_with(&x, &f, &AaaOptions::new().with_rtol(0.0).with_max_terms(4)).unwrap();
    assert_eq!(r.errors().len(), 4);
}

#[test]
#[should_panic(expected = "differ in length")]
fn points_and_values_of_different_lengths_panic() {
    let _ = aaa(&[0.0f64, 1.0], &[1.0]);
}

#[test]
fn evaluation_at_and_next_to_support_points_and_at_infinity() {
    let x = grid(200, 0.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| 1.0 / (t + 0.5)).collect();
    let r = aaa(&x, &f).unwrap();
    for (&z, &v) in r.support_points().iter().zip(r.support_values()) {
        assert_eq!(r.evaluate(z), v);
        // So close that the term overflows: the value at the point.
        let near = if z == 0.0 { 5e-324 } else { z.next_up() };
        assert!(
            (r.evaluate(near) - v).abs() <= 4.0 * f64::EPSILON * v.abs(),
            "next to {z}"
        );
    }
    // 1/(x + 1/2) tends to 0 at infinity, which the limit of the barycentric form finds.
    assert!(r.evaluate(f64::INFINITY).abs() < 1e-12);
    assert!(r.evaluate(f64::NAN).is_nan());
    // Off the real line, through the complex evaluation: 1/(i + 1/2).
    let v = r.evaluate_complex(Complex::new(0.0, 1.0));
    assert!((v - Complex::new(0.5, 1.0).recip()).abs() < 1e-12, "{v}");
}

#[test]
fn zeros_include_support_points_of_value_zero() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // x²/(x² + 1/100) on [-1, 1]: the sample at 0, of value 0, is the first support point, as
    // the one farthest from the mean; it is a zero of the approximation, listed as it is, and the
    // numerator has the other zero of the double one there (review round 1: a test of this rule
    // that had no support point of value 0).
    let x = grid(201, -1.0, 1.0);
    let f: Vec<f64> = x.iter().map(|&t| t * t / (t * t + 0.01)).collect();
    let r = aaa(&x, &f).unwrap();
    assert_eq!((r.support_points()[0], r.support_values()[0]), (0.0, 0.0));
    let zeros = r.zeros();
    assert!(zeros.contains(&Complex::new(0.0, 0.0)), "{zeros:?}");
    assert_eq!(
        zeros.iter().filter(|z| z.abs() < 1e-12).count(),
        2,
        "{zeros:?}"
    );
    // sin on [-3.2, 3.2] has its zeros at 0 and ±π, 0 among the samples.
    let x = grid(601, -3.2, 3.2);
    let f: Vec<f64> = x
        .iter()
        .map(|&t| if t == 0.0 { 0.0 } else { t.sin() })
        .collect();
    let r = aaa(&x, &f).unwrap();
    let zeros = r.zeros();
    for want in [-core::f64::consts::PI, 0.0, core::f64::consts::PI] {
        assert!(
            zeros
                .iter()
                .any(|z| (z - Complex::new(want, 0.0)).abs() < 1e-10),
            "no zero at {want}: {zeros:?}"
        );
    }
    // The zero function has no zeros to list.
    assert!(
        aaa(&[0.0f64, 1.0, 2.0], &[0.0, 0.0, 0.0])
            .unwrap()
            .zeros()
            .is_empty()
    );
}

/// The poles of an approximation whose residue over their distance from the samples is below
/// 1e-13 times the geometric mean of the nonzero |f|: the doublets by the criterion of the
/// clean-up.
fn spurious<S: a::math::matrix::Scalar<Real = f64>>(r: &Aaa<S>, z: &[S], f: &[S]) -> usize {
    let logs: Vec<f64> = f
        .iter()
        .map(|v| v.to_complex().abs())
        .filter(|a| *a > 0.0)
        .map(f64::ln)
        .collect();
    let mean = (logs.iter().sum::<f64>() / logs.len() as f64).exp();
    r.poles()
        .iter()
        .zip(r.residues())
        .filter(|(p, res)| {
            let d = z
                .iter()
                .map(|t| (**p - t.to_complex()).abs())
                .fold(f64::INFINITY, f64::min);
            res.abs() / d < 1e-13 * mean
        })
        .count()
}

#[test]
fn the_clean_up_removes_froissart_doublets() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // |x| on 300 points, run with no tolerance to 50 terms: past the accuracy of the arithmetic
    // the steps leave poles of negligible residue, and the clean-up takes the support points
    // nearest them away and solves for the weights again. That keeps the approximation good, but
    // not to the error the iteration reached: the new weights are the plain singular vector of
    // the paper and SciPy, of a Loewner matrix past the condition 1/(3ε), and how many digits of
    // the error they keep rests on the last bits of the decompositions, which differ between
    // platforms. Here the error goes from 1.8e-15 to 1.4e-15 on aarch64 (2 doublets) and from
    // 3.9e-15 to 1.4e-13 on x86_64 (8 doublets); over 200 to 500 points and 40 to 60 terms it
    // was at most 5.8e-13 after the clean-up. Scaling the columns first, as the iteration does,
    // is no better (on aarch64, 1.1e-15 to 2.9e-13 over that family, against 1.4e-15 to
    // 1.2e-13). A clean-up that does not solve again is 4.5e-2 off, and one that takes the
    // largest singular vector 7.3.
    let x = grid(300, -1.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| t.abs()).collect();
    let base = AaaOptions::new().with_rtol(0.0).with_max_terms(50);
    let raw = aaa_with(&x, &f, &base.with_clean_up(false)).unwrap();
    let before = spurious(&raw, &x, &f);
    assert!(before > 0, "the uncleaned approximation has doublets");
    let clean = aaa_with(&x, &f, &base).unwrap();
    assert!(clean.doublets() > 0);
    assert!(clean.support_points().len() < raw.support_points().len());
    assert!(spurious(&clean, &x, &f) < before);
    assert!(clean.error() < 1e-12, "{}", clean.error());
    // The paper's example: log(2 + z⁴)/(1 − 16z⁴) on 1000 points of the unit circle, no
    // tolerance, 50 terms. How many doublets the best step has depends on the last bits of the
    // arithmetic (debug and optimised builds of the decompositions differ there); that the
    // approximation is good and has the four poles at radius 1/2 does not.
    let z: Vec<Complex<f64>> = (0..1000)
        .map(|k| Complex::from_polar(1.0, 2.0 * core::f64::consts::PI * k as f64 / 1000.0))
        .collect();
    let f: Vec<Complex<f64>> = z
        .iter()
        .map(|&w| {
            let w4 = w * w * w * w;
            (w4 + 2.0).ln() / (1.0 - 16.0 * w4)
        })
        .collect();
    let opts = AaaOptions::new().with_rtol(0.0).with_max_terms(50);
    let raw = aaa_complex_with(&z, &f, &opts.with_clean_up(false)).unwrap();
    let clean = aaa_complex_with(&z, &f, &opts).unwrap();
    assert!(clean.error() < 1e-13, "{}", clean.error());
    // One pass, as in the paper: solving for the weights again can leave a new doublet, but
    // never more than there were.
    assert!(spurious(&clean, &z, &f) <= spurious(&raw, &z, &f));
    for k in 0..4 {
        let want = Complex::from_polar(0.5, core::f64::consts::FRAC_PI_2 * k as f64);
        assert!(
            clean.poles().iter().any(|p| (p - want).abs() < 1e-10),
            "{want}"
        );
    }
}

/// Runge's function with a pole at 1.3 added, which AAA finds with four terms: the poles 1.3 and
/// ±i/5.
fn runge_and_pole<T: a::math::numerics::Real>(t: T) -> T {
    let (one, c) = (T::ONE, T::from_f64(25.0));
    one / (one + c * t * t) + T::from_f64(0.3) / (t - T::from_f64(1.3))
}

/// Whether `2^k·f` holds the values of `f`: each value scaled exactly, none of them subnormal.
fn same_data<T: a::math::numerics::Real>(f: &[T], scaled: &[T], s: T) -> bool {
    f.iter()
        .zip(scaled)
        .all(|(&a, &b)| b / s == a && (b == T::ZERO || b.abs() >= T::MIN_POSITIVE))
}

/// AAA of `2^k·f` against AAA of `f` (`base`), when `2^k·f` holds the same values: the same
/// support points, weights, poles, zeros and convergence, bit for bit, and the reported errors,
/// values and residues `2^k` times as large.
fn scaled_by<T>(base: &Aaa<T>, x: &[T], f: &[T], s: T, k: i32)
where
    T: a::math::numerics::Real + a::math::matrix::Scalar<Real = T> + core::fmt::Debug,
{
    let scaled: Vec<T> = f.iter().map(|&v| v * s).collect();
    let r = aaa(x, &scaled).unwrap();
    assert_eq!(r.support_points(), base.support_points(), "2^{k}");
    assert_eq!(r.weights(), base.weights(), "2^{k}");
    assert_eq!(r.converged(), base.converged(), "2^{k}");
    assert_eq!(r.poles(), base.poles(), "2^{k}");
    assert_eq!(r.zeros(), base.zeros(), "2^{k}");
    assert_eq!(r.errors().len(), base.errors().len(), "2^{k}");
    // Quantities in the units of the values are scaled back exactly while they stay normal.
    let normal = |v: T| {
        let a = a::math::complex::Float::abs(v);
        v == T::ZERO || (a >= T::MIN_POSITIVE && a::math::complex::Float::is_finite(v))
    };
    let same = |got: T, want: T| !normal(want) || got == want;
    assert!(same(r.error(), base.error() * s), "2^{k}: error");
    for (g, w) in r.residues().iter().zip(base.residues()) {
        assert!(
            same(g.re, w.re * s) && same(g.im, w.im * s),
            "2^{k}: residue"
        );
    }
    for t in [-0.97, -0.3, 0.123, 0.5, 0.999] {
        let t = T::from_f64(t);
        assert!(same(r.evaluate(t), base.evaluate(t) * s), "2^{k} at {t:?}");
    }
}

#[test]
fn the_size_of_the_values_changes_nothing() {
    // Review round 1, B2: the 2-norms of the Loewner columns and of the weights were sums of
    // unscaled squares, which overflowed or underflowed at ordinary f32 magnitudes. Values of
    // size 2e-22 gave the pole 1.3 at −51.4 and called that converged; values of 1e18 one support
    // point and no convergence; f64 failed the same way below 2^-519 and above 2^510. The values
    // are now divided by the power of two of the largest before anything is computed, so the
    // approximation of 2^k·f is that of f, bit for bit, whenever 2^k·f holds the same values.
    let n = if cfg!(miri) { 40 } else { 400 };
    let x: Vec<f32> = (0..n)
        .map(|k| -1.0 + 2.0 * k as f32 / (n - 1) as f32)
        .collect();
    let f: Vec<f32> = x.iter().map(|&t| runge_and_pole(t)).collect();
    let base = aaa(&x, &f).unwrap();
    assert!(base.converged() && base.degree() == 3);
    assert!(
        base.poles()
            .iter()
            .any(|p| (p.re - 1.3).abs() < 1e-4 && p.im == 0.0)
    );
    let mut compared = 0;
    let steps = if cfg!(miri) { 20 } else { 1 };
    for k in (-140..=120).step_by(steps) {
        // Two factors, since 2f32.powi(k) is 1/2^|k| and 2^|k| overflows below 2^-127.
        let s = 2f32.powi(k / 2) * 2f32.powi(k - k / 2);
        let scaled: Vec<f32> = f.iter().map(|&v| v * s).collect();
        if same_data(&f, &scaled, s) {
            compared += 1;
            scaled_by(&base, &x, &f, s, k);
        } else {
            // Below 2^-114 some of these values are subnormal floats that keep only part of their
            // bits: other data, of which nothing is claimed but that what is reported holds (to
            // the smallest subnormal, which the error in the units of the data is rounded to).
            let opts = AaaOptions::new().with_max_terms(20);
            let r = aaa_with(&x, &scaled, &opts).unwrap();
            let big = scaled.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let bound = opts.rtol * big + f32::from_bits(1);
            assert!(!r.converged() || r.error() <= bound, "2^{k}: {}", r.error());
        }
    }
    // Every value of these data is at least 1e-8 in size, so every k from -100 on holds the same
    // values (from about -114 on, for these 400 points).
    let from = (-140..=120).step_by(steps).filter(|&k| k >= -100).count();
    assert!(compared >= from, "{compared} scales compared");
    // f64 over 2^-1000 .. 2^1000, where every scaled value is normal.
    let x: Vec<f64> = (0..n)
        .map(|k| -1.0 + 2.0 * k as f64 / (n - 1) as f64)
        .collect();
    let f: Vec<f64> = x.iter().map(|&t| runge_and_pole(t)).collect();
    let base = aaa(&x, &f).unwrap();
    assert!(base.converged() && base.degree() == 3);
    for k in (-1000..=1000).step_by(if cfg!(miri) { 500 } else { 8 }) {
        let s = 2f64.powi(k);
        let scaled: Vec<f64> = f.iter().map(|&v| v * s).collect();
        assert!(same_data(&f, &scaled, s), "2^{k}");
        scaled_by(&base, &x, &f, s, k);
    }
    // Complex samples take the same path: 1/(z − 2) + 1/z² beyond and inside the unit circle.
    let z: Vec<Complex<f32>> = (0..n / 2)
        .map(|k| Complex::from_polar(1.0, 2.0 * core::f32::consts::PI * k as f32 / (n / 2) as f32))
        .collect();
    let g: Vec<Complex<f32>> = z
        .iter()
        .map(|&w| (w - 2.0).recip() + (w * w).recip())
        .collect();
    let base = aaa_complex(&z, &g).unwrap();
    assert!(base.converged());
    for k in (-100..=100).step_by(if cfg!(miri) { 50 } else { 10 }) {
        let s = 2f32.powi(k);
        let scaled: Vec<Complex<f32>> = g.iter().map(|&v| v * s).collect();
        let r = aaa_complex(&z, &scaled).unwrap();
        assert_eq!(r.support_points(), base.support_points(), "2^{k}");
        assert_eq!(r.weights(), base.weights(), "2^{k}");
        assert_eq!(r.poles(), base.poles(), "2^{k}");
        assert_eq!(r.converged(), base.converged(), "2^{k}");
    }
}

#[test]
fn the_clean_up_follows_the_papers_rules() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // The rules of the paper and of SciPy: a pole is a Froissart doublet when its residue over
    // its distance from the nearest sample is below clean_up_tol times the geometric mean of the
    // nonzero |f|, and the support point nearest each doublet is removed. Here they are checked
    // against the poles and residues of the approximation before its clean-up (review round 1:
    // removing the farthest support point instead, or leaving out the distance, passed the tests).
    let x = grid(300, -1.0, 1.0);
    let f: Vec<f64> = x.iter().map(|t| t.abs()).collect();
    let base = AaaOptions::new().with_rtol(0.0).with_max_terms(50);
    let raw = aaa_with(&x, &f, &base.with_clean_up(false)).unwrap();
    let logs: Vec<f64> = f.iter().filter(|v| **v > 0.0).map(|v| v.ln()).collect();
    let mean = (logs.iter().sum::<f64>() / logs.len() as f64).exp();
    // Each pole's residue over the geometric mean, and that over the distance to the samples too.
    let poles = raw.poles();
    let ratios: Vec<(f64, f64)> = poles
        .iter()
        .zip(raw.residues())
        .map(|(p, r)| {
            let d = x
                .iter()
                .map(|&t| (*p - Complex::new(t, 0.0)).abs())
                .fold(f64::INFINITY, f64::min);
            (r.abs() / mean, r.abs() / mean / d)
        })
        .collect();
    let nearest = |p: Complex<f64>| {
        raw.support_points()
            .iter()
            .copied()
            .min_by(|a, b| {
                (p - Complex::new(*a, 0.0))
                    .abs()
                    .total_cmp(&(p - Complex::new(*b, 0.0)).abs())
            })
            .unwrap()
    };
    // The default threshold, and one between a pole's two ratios, where the distance decides.
    let middle = ratios
        .iter()
        .filter(|(alone, over)| *over > 4.0 * alone && *over > 1e-13)
        .map(|(alone, over)| (alone * over).sqrt())
        .find(|t| ratios.iter().all(|(_, over)| (over / t).ln().abs() > 0.1))
        .expect("a pole whose distance from the samples decides");
    for tol in [1e-13, middle] {
        let clean = aaa_with(&x, &f, &base.with_clean_up_tol(tol)).unwrap();
        let doublets: Vec<usize> = (0..poles.len()).filter(|&k| ratios[k].1 < tol).collect();
        assert_eq!(clean.doublets(), doublets.len(), "clean_up_tol {tol:e}");
        assert!(doublets.len() < ratios.iter().filter(|(a, _)| *a < tol).count() || tol == 1e-13);
        let mut removed: Vec<f64> = doublets.iter().map(|&k| nearest(poles[k])).collect();
        removed.sort_by(f64::total_cmp);
        removed.dedup();
        let mut lost: Vec<f64> = raw
            .support_points()
            .iter()
            .copied()
            .filter(|p| !clean.support_points().contains(p))
            .collect();
        lost.sort_by(f64::total_cmp);
        assert_eq!(lost, removed, "clean_up_tol {tol:e}");
    }
}

#[test]
fn the_approximations_work_in_f32() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    let x: Vec<f32> = (0..500).map(|k| -1.0 + 2.0 * k as f32 / 499.0).collect();
    let f: Vec<f32> = x.iter().map(|t| t.exp()).collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged(), "{}", r.error());
    for k in 0..200 {
        let t = -0.99 + 1.98 * (k as f32 + 0.5) / 200.0;
        assert!((r.evaluate(t) - t.exp()).abs() < 1e-5, "at {t}");
    }
    let f: Vec<f32> = x.iter().map(|t| 1.0 / (1.0 + 25.0 * t * t)).collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged() && r.degree() == 2);
    for p in r.poles() {
        assert!(p.re.abs() < 1e-5 && (p.im.abs() - 0.2).abs() < 1e-5, "{p}");
    }
    // And on complex f32 samples: 1/(z − 2) on the unit circle.
    let z: Vec<Complex<f32>> = (0..200)
        .map(|k| Complex::from_polar(1.0, 2.0 * core::f32::consts::PI * k as f32 / 200.0))
        .collect();
    let f: Vec<Complex<f32>> = z.iter().map(|&w| (w - 2.0).recip()).collect();
    let r = aaa_complex(&z, &f).unwrap();
    assert!(r.converged());
    let poles = r.poles();
    assert!(
        poles
            .iter()
            .any(|p| (p - Complex::new(2.0, 0.0)).abs() < 1e-4),
        "{poles:?}"
    );
}

/// Support points, coefficients and the support points of value zero, in `f64`.
type Sums = (Vec<Complex<f64>>, Vec<Complex<f64>>, Vec<Complex<f64>>);

/// The support points and the coefficients `w_j` (`poles`) or `w_j·f_j` (`zeros`, over the
/// support points of nonzero value) of the sum whose zeros the poles or zeros are, in `f64`
/// (exactly, for `f32` data), and the support points of value zero.
fn sums<S: a::math::matrix::Scalar>(r: &Aaa<S>, zeros: bool) -> Sums {
    use a::math::complex::Float;
    let c64 = |v: S| Complex::new(v.re().to_f64(), v.im().to_f64());
    let (mut z, mut c, mut at_zero) = (Vec::new(), Vec::new(), Vec::new());
    for ((&p, &f), &w) in r
        .support_points()
        .iter()
        .zip(r.support_values())
        .zip(r.weights())
    {
        if !zeros {
            z.push(c64(p));
            c.push(c64(w));
        } else if f == S::zero() {
            at_zero.push(c64(p));
        } else {
            z.push(c64(p));
            c.push(c64(w) * c64(f));
        }
    }
    (z, c, at_zero)
}

/// The relative backward error of `x` as a zero of `Σ c_i/(x − z_i)`, as `Aaa::poles` defines
/// it: the size of the relative changes of the coefficients and of `x` itself that make it an
/// exact zero. It is written about the support point `z_j` of the largest term `|c_j/(x − z_j)|`,
/// `F = c_j + (x − z_j)·Σ_{i≠j} c_i/(x − z_i)`, so that it is finite at and next to one, as
/// `|F| / (|c_j| + |x − z_j|·Σ_{i≠j}|c_i/(x − z_i)| + |x|·|F′|)`; it is about ε at the float
/// nearest a zero, even one closer to a support point than the floats there are apart.
fn backward_error(x: Complex<f64>, z: &[Complex<f64>], c: &[Complex<f64>]) -> f64 {
    let mut j = 0;
    let mut big = -1.0;
    for (i, (&p, &q)) in z.iter().zip(c).enumerate() {
        let distance = (x - p).abs();
        if distance == 0.0 {
            j = i;
            break;
        }
        if q.abs() / distance > big {
            big = q.abs() / distance;
            j = i;
        }
    }
    let y = x - z[j];
    let (mut sum, mut size, mut bend) = (Complex::<f64>::ZERO, 0.0, Complex::<f64>::ZERO);
    for i in (0..z.len()).filter(|&i| i != j) {
        let t = c[i] / (x - z[i]);
        sum += t;
        size += t.abs();
        bend += t / (x - z[i]);
    }
    let f = c[j] + y * sum;
    let slope = sum - y * bend;
    f.abs() / (c[j].abs() + y.abs() * size + x.abs() * slope.abs())
}

/// The largest backward error of the poles and of the zeros of `r` as zeros of their sums,
/// in units of the type's ε; a zero at a support point of value zero counts 0.
fn worst_backward_error<S: a::math::matrix::Scalar>(r: &Aaa<S>, eps: f64) -> (f64, f64) {
    use a::math::complex::Float;
    let c64 = |p: Complex<S::Real>| Complex::new(p.re.to_f64(), p.im.to_f64());
    let (z, c, _) = sums(r, false);
    let poles = r
        .poles()
        .into_iter()
        .map(|p| backward_error(c64(p), &z, &c))
        .fold(0.0, f64::max);
    let (z, c, at_zero) = sums(r, true);
    let zeros = r
        .zeros()
        .into_iter()
        .map(c64)
        .filter(|p| !at_zero.contains(p))
        .map(|p| backward_error(p, &z, &c))
        .fold(0.0, f64::max);
    (poles / eps, zeros / eps)
}

/// The distance from `want` to the nearest of `roots`.
fn nearest(roots: &[Complex<f64>], want: Complex<f64>) -> f64 {
    roots
        .iter()
        .map(|p| (p - want).abs())
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn zeros_of_the_data_at_the_samples_are_found() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with a hundred rows, step after step.
        return;
    }
    // sin(5πx) at 101 equally spaced points of [-1, 1]: every zero j/5 is a sample, where the
    // computed sine is about 1e−16 (not 0), so the numerator's zero lies within a fraction of
    // a float of a support point whose coefficient is small. Newton's method on the sum itself
    // was thrown off by the pole of the sum beside it: the zero at 1 came out at 1.0278 and the
    // one at −0.2 at −0.198, where |r| is 0.42 and 0.032.
    let x = grid(101, -1.0, 1.0);
    let f: Vec<f64> = x
        .iter()
        .map(|t| (5.0 * core::f64::consts::PI * t).sin())
        .collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged());
    let zeros = r.zeros();
    for j in -5..=5 {
        let want = Complex::new(f64::from(j) / 5.0, 0.0);
        let d = nearest(&zeros, want);
        assert!(d < 1e-12, "the zero {want} is missed by {d:e}");
    }
    // |r| at the zeros in [-1, 1] is at the level of its rounding (measured: 8.2e−16 at worst).
    for z in zeros
        .iter()
        .filter(|z| z.re.abs() <= 1.0 && z.im.abs() < 1e-9)
    {
        assert!(
            r.evaluate_complex(*z).abs() < 1e-14,
            "r({z}) = {}",
            r.evaluate_complex(*z)
        );
    }
    let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
    assert!(poles <= 8.0 && zeros <= 8.0, "{poles} {zeros}");
}

#[test]
fn a_pole_one_float_from_a_sample_is_found() {
    if cfg!(miri) {
        // Singular value decompositions, step after step.
        return;
    }
    // 1/(x − c) + cos x with c one float above the 17th of 40 samples, run with no tolerance to
    // ten terms: the denominator returned vanishes at c (to rounding), and the pole there was
    // reported at 0.350.
    let x: Vec<f64> = (0..40).map(|k| -1.0 + k as f64 * (2.0 / 39.0)).collect();
    let c = x[16].next_up();
    let f: Vec<f64> = x.iter().map(|&t| 1.0 / (t - c) + t.cos()).collect();
    let o = AaaOptions::new()
        .with_rtol(0.0)
        .with_max_terms(10)
        .with_clean_up(false);
    let r = aaa_with(&x, &f, &o).unwrap();
    let d = nearest(&r.poles(), Complex::new(c, 0.0));
    assert!(
        d < 1e-12,
        "the pole {c} is missed by {d:e}: {:?}",
        r.poles()
    );
    let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
    assert!(poles <= 8.0 && zeros <= 8.0, "{poles} {zeros}");
}

#[test]
fn poles_of_the_data_at_the_samples_are_found() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // tan(5πx) at 101 equally spaced points of [0, 1], which hold its poles 0.1, 0.3, ..., 0.9,
    // where the computed tangent is about ±1e16: they were 3.4e−8 off.
    let x = grid(101, 0.0, 1.0);
    let f: Vec<f64> = x
        .iter()
        .map(|t| (5.0 * core::f64::consts::PI * t).tan())
        .collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged());
    let poles = r.poles();
    for k in 0..5 {
        let want = Complex::new(0.1 + 0.2 * f64::from(k), 0.0);
        let d = nearest(&poles, want);
        assert!(d < 1e-13, "the pole {want} is missed by {d:e}");
    }
    // tan(πz/2) on the 31 x 31 grid of [-1, 1]² in the complex plane, in f64 and f32: the poles
    // ±1 are samples. They were 2e−6 off in f64, and 7 and 15 away in f32.
    let g = grid(31, -1.0, 1.0);
    let z: Vec<Complex<f64>> = g
        .iter()
        .flat_map(|&a| g.iter().map(move |&b| Complex::new(a, b)))
        .collect();
    let tan = |w: Complex<f64>| {
        let e = (w * Complex::new(0.0, core::f64::consts::PI)).exp();
        (e - 1.0) / ((e + 1.0) * Complex::I)
    };
    let f: Vec<Complex<f64>> = z.iter().map(|&w| tan(w)).collect();
    let r = aaa_complex(&z, &f).unwrap();
    assert!(r.converged());
    for want in [-1.0, 1.0] {
        let d = nearest(&r.poles(), Complex::new(want, 0.0));
        assert!(
            d < 1e-15,
            "the pole {want} is missed by {d:e}: {:?}",
            r.poles()
        );
    }
    let z32: Vec<Complex<f32>> = z
        .iter()
        .map(|w| Complex::new(w.re as f32, w.im as f32))
        .collect();
    let f32s: Vec<Complex<f32>> = f
        .iter()
        .map(|w| Complex::new(w.re as f32, w.im as f32))
        .collect();
    let r = aaa_complex(&z32, &f32s).unwrap();
    assert!(r.converged());
    let poles: Vec<Complex<f64>> = r
        .poles()
        .iter()
        .map(|p| Complex::new(f64::from(p.re), f64::from(p.im)))
        .collect();
    for want in [-1.0, 1.0] {
        let d = nearest(&poles, Complex::new(want, 0.0));
        assert!(
            d < 1e-6,
            "f32: the pole {want} is missed by {d:e}: {poles:?}"
        );
    }
}

/// The zeros of the real function `r` on `[-1, 1]`, where it changes sign between neighbours
/// of `n` equally spaced points, each narrowed by bisection to neighbouring floats: zeros and
/// real poles alike.
fn sign_changes(r: &Aaa<f64>, n: usize) -> Vec<f64> {
    let t = grid(n, -1.0, 1.0);
    let mut out = Vec::new();
    for w in t.windows(2) {
        let (mut a, mut b) = (w[0], w[1]);
        let (fa, fb) = (r.evaluate(a), r.evaluate(b));
        if fa == 0.0 {
            out.push(a);
            continue;
        }
        if fa.signum() == fb.signum() || fb == 0.0 {
            continue;
        }
        loop {
            let c = 0.5 * (a + b);
            if c == a || c == b {
                break;
            }
            if r.evaluate(c).signum() == fa.signum() {
                a = c;
            } else {
                b = c;
            }
        }
        out.push(0.5 * (a + b));
    }
    out
}

#[test]
fn every_reported_pole_and_zero_is_a_root_to_rounding() {
    if cfg!(miri) {
        // Eighty approximations, each a sequence of singular value decompositions.
        return;
    }
    // sin(kπx), k = 1..8, at 21 to 1001 equally spaced points of [-1, 1], default options; the
    // grids hold zeros of the sine where k divides m − 1, and ±1 always. Of these 80, 35 had a
    // zero of the sine more than 1e−9 from every reported zero (one of them 0.20 away), and the
    // poles — spurious, in clusters, conditioned like 1e9, where the eigenvalue estimates are
    // poor — were up to 2e6 times their rounding off the roots of the returned denominator.
    let mut cases = 0;
    for k in 1..=8 {
        for m in [21usize, 41, 50, 81, 101, 150, 201, 301, 401, 1001] {
            let x = grid(m, -1.0, 1.0);
            let f: Vec<f64> = x
                .iter()
                .map(|t| (k as f64 * core::f64::consts::PI * t).sin())
                .collect();
            let r = aaa(&x, &f).unwrap();
            // Every pole and zero is one of its sum, to rounding.
            let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
            assert!(poles <= 8.0 && zeros <= 8.0, "k {k} m {m}: {poles} {zeros}");
            // Every zero and real pole of r in [-1, 1] is reported.
            let (zs, ps) = (r.zeros(), r.poles());
            for xi in sign_changes(&r, 4001) {
                let d =
                    nearest(&zs, Complex::new(xi, 0.0)).min(nearest(&ps, Complex::new(xi, 0.0)));
                assert!(
                    d < 1e-12,
                    "k {k} m {m}: r changes sign at {xi}, {d:e} from every root"
                );
            }
            // From 50 points on, r is close to the sine between the samples too, and has its
            // zeros: each j/k within 1e−9 of a reported zero. (On 21 and 41 points it is not:
            // sin(8πx) from 21 samples is 9e4 off between them, and its zeros are elsewhere.)
            if r.converged() {
                cases += 1;
                if m >= 50 {
                    for j in -k..=k {
                        let want = Complex::new(f64::from(j) / f64::from(k), 0.0);
                        let d = nearest(&zs, want);
                        assert!(d < 1e-9, "k {k} m {m}: the zero {want} is missed by {d:e}");
                    }
                }
            }
        }
    }
    assert_eq!(cases, 79);
    // In f32: sin(5πx) at 101 points, every j/5 within the f32 rounding of a reported zero, and
    // the sine at 201 points with k = 3 and 8, whose clusters of spurious poles include pairs
    // so close to the axis that the eigenvalues came out real.
    let x: Vec<f32> = (0..101).map(|i| -1.0 + 2.0 * i as f32 / 100.0).collect();
    let f: Vec<f32> = x
        .iter()
        .map(|t| (5.0 * core::f32::consts::PI * t).sin())
        .collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged());
    let zeros: Vec<Complex<f64>> = r
        .zeros()
        .iter()
        .map(|p| Complex::new(f64::from(p.re), f64::from(p.im)))
        .collect();
    for j in -5..=5 {
        let want = Complex::new(f64::from(j) / 5.0, 0.0);
        let d = nearest(&zeros, want);
        assert!(d < 1e-6, "f32: the zero {want} is missed by {d:e}");
    }
    for k in [3.0f32, 5.0, 8.0] {
        for m in [101usize, 201] {
            let x: Vec<f32> = (0..m)
                .map(|i| -1.0 + 2.0 * i as f32 / (m - 1) as f32)
                .collect();
            let f: Vec<f32> = x
                .iter()
                .map(|t| (k * core::f32::consts::PI * t).sin())
                .collect();
            let r = aaa(&x, &f).unwrap();
            let (poles, zeros) = worst_backward_error(&r, f64::from(f32::EPSILON));
            assert!(
                poles <= 8.0 && zeros <= 8.0,
                "f32 k {k} m {m}: {poles} {zeros}"
            );
        }
    }
}

#[test]
fn conjugate_pairs_given_as_two_real_estimates_are_found() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // cos(kπx) on equally spaced points: among the spurious poles, a conjugate pair that the
    // eigenvalue routine of the real matrix returns as two real numbers, far apart (−1.94 and
    // 1.65 for a pair at −0.143 ± 0.735i in one such case). No iteration that keeps them real can
    // reach the pair; without the restart as a pair, two poles here were no poles at all.
    for (k, m) in [(5.0, 550usize), (5.5, 200), (6.5, 350), (7.5, 150)] {
        let x = grid(m, -1.0, 1.0);
        let f: Vec<f64> = x
            .iter()
            .map(|t| (k * core::f64::consts::PI * t).cos())
            .collect();
        let r = aaa(&x, &f).unwrap();
        assert!(r.converged());
        let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
        assert!(poles <= 8.0 && zeros <= 8.0, "k {k} m {m}: {poles} {zeros}");
        for p in r.poles() {
            assert!(
                p.im == 0.0 || r.poles().iter().any(|q| *q == p.conj()),
                "k {k} m {m}: {p} without its conjugate"
            );
        }
    }
}

/// The real zeros `s` of the denominator of `r` beyond the samples — sign changes of
/// `Σ w_j/(t − z_j)` between neighbours of 2000 points of `[−8, −1]` and of `[1, 8]`, narrowed by
/// bisection to neighbouring floats — each with its distance from the nearest reported pole in
/// units of the rounding level `ε·(cond + |s|)` there. A real pole the iteration missed is
/// thousands of levels away; one it found, about one.
fn real_zeros_beyond_the_samples(r: &Aaa<f64>) -> Vec<(f64, f64)> {
    let d = |t: f64| {
        r.support_points()
            .iter()
            .zip(r.weights())
            .map(|(&z, &w)| w / (t - z))
            .sum::<f64>()
    };
    let poles = r.poles();
    let mut out = Vec::new();
    for (a, b) in [(-8.0f64, -1.0f64), (1.0, 8.0)] {
        let t = |i: usize| a + (b - a) * i as f64 / 2000.0;
        for i in 0..2000 {
            let (mut lo, mut hi) = (t(i), t(i + 1));
            if lo.abs() <= 1.0 || hi.abs() <= 1.0 {
                continue;
            }
            let sign = d(lo).signum();
            if sign == d(hi).signum() {
                continue;
            }
            loop {
                let mid = 0.5 * (lo + hi);
                if mid == lo || mid == hi {
                    break;
                }
                if d(mid).signum() == sign {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let (mut size, mut slope) = (0.0f64, 0.0f64);
            for (&z, &w) in r.support_points().iter().zip(r.weights()) {
                size += (w / (lo - z)).abs();
                slope += w / ((lo - z) * (lo - z));
            }
            let level = f64::EPSILON * (size / slope.abs() + lo.abs());
            out.push((lo, nearest(&poles, Complex::new(lo, 0.0)) / level));
        }
    }
    out
}

/// The largest distance of [`real_zeros_beyond_the_samples`] from the reported poles, in
/// rounding levels; 0 when there is none.
fn real_poles_beyond_the_samples_missed_by(r: &Aaa<f64>) -> f64 {
    real_zeros_beyond_the_samples(r)
        .iter()
        .map(|z| z.1)
        .fold(0.0, f64::max)
}

/// Chebyshev points of the second kind, `−cos(πj/(m − 1))`, on `[−1, 1]`.
fn chebyshev_points(m: usize) -> Vec<f64> {
    (0..m)
        .map(|j| -(core::f64::consts::PI * j as f64 / (m - 1) as f64).cos())
        .collect()
}

#[test]
fn a_pair_estimated_where_the_denominator_has_two_real_zeros_is_parted() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // At the default options: cos(5.181314019599456πx) at 267 Chebyshev points, 23 terms. The
    // denominator returned changes sign twice beyond the samples, real poles where r is about
    // ±1e11, and the pole estimates have a conjugate pair there instead; kept symmetric, the
    // iteration could not part it, and on aarch64 3.7656 ± 0.1136i were reported (backward error
    // 5.8e3 ε) and the two real poles, −3.893 and 3.4787, not at all. Where those two lie rests on
    // the last bits of the weights, a singular vector of a nearly singular Loewner matrix: with
    // the values from the platform's libm and decompositions that sum in different orders, the
    // weights on x86_64 are 7e-4 from aarch64's (relative to the largest) on the same support
    // points, and the real poles there are −5.569 and 4.2776 — the poles of an approximation of
    // an entire function are spurious and move with the weights. So the test asks for the sign
    // changes of the denominator returned, wherever they are: two at least beyond the samples,
    // each a reported pole (below).
    //
    // Then Re 1/(x − p − qi) + sin(28.92x) at 686 Chebyshev points, where the pair
    // −0.395 ± 0.711i was missing and a real "pole" 66.95 was reported (3.2e5 ε), and
    // e^(−1.95x)·sin(33.41x + 0.346) at 69 equally spaced points with no tolerance and 39 terms
    // at most (5.6e5 ε). These three were found on aarch64; on x86_64 the round-2 code, before
    // the restarts, gets all three right. So three more, from a screen of that code over 6000
    // approximations of the families of
    // `random_approximations_have_their_poles_and_zeros_to_rounding`: it got 43 of them wrong on
    // aarch64, and these three on x86_64 as well, each time giving a conjugate pair where the
    // denominator has real zeros and leaving those out (on aarch64 2.4875 ± 1.3567i,
    // 2.4294 ± 1.7278i and 2.0831 ± 0.0195i, at 8.7e4, 1.1e4 and 7.7e3 ε; on x86_64
    // 7.9419 ± 21.6282i, 3.0529 ± 1.3794i and 2.0071 ± 0.3293i, at 4.6e3, 1.2e4 and 1.2e5 ε).
    let pi = core::f64::consts::PI;
    let x = chebyshev_points(267);
    let k = 5.181314019599456;
    let f: Vec<f64> = x.iter().map(|t| (k * pi * t).cos()).collect();
    let r = aaa(&x, &f).unwrap();
    assert!(r.converged() && r.degree() == 22, "{}", r.degree());
    let real = real_zeros_beyond_the_samples(&r);
    assert!(real.len() >= 2, "{real:?}");
    let x686 = chebyshev_points(686);
    let (p, q, a) = (-0.520623153200292, 0.2801744654576636, 28.92484695650297);
    let rational: Vec<f64> = x686
        .iter()
        .map(|t| (t - p) / ((t - p) * (t - p) + q * q) + (a * t).sin())
        .collect();
    let x69 = grid(69, -1.0, 1.0);
    let (s, phase, e) = (33.41414807834296, 0.3464306810174358, -1.9509582032917798);
    let damped: Vec<f64> = x69
        .iter()
        .map(|t| (e * t).exp() * (s * t + phase).sin())
        .collect();
    let no_tolerance = AaaOptions::new().with_rtol(0.0).with_max_terms(39);
    let cosine = |k: f64, x: &[f64]| -> Vec<f64> { x.iter().map(|t| (k * pi * t).cos()).collect() };
    let (x379, x330, x405) = (
        chebyshev_points(379),
        grid(330, -1.0, 1.0),
        chebyshev_points(405),
    );
    let (s, phase, e) = (19.15839752735216, 0.07692011700532808, 1.9792885372465059);
    let rising: Vec<f64> = x405
        .iter()
        .map(|t| (e * t).exp() * (s * t + phase).sin())
        .collect();
    for (name, r) in [
        ("cos", r),
        ("rational", aaa(&x686, &rational).unwrap()),
        ("damped", aaa_with(&x69, &damped, &no_tolerance).unwrap()),
        (
            "cos 379",
            aaa(&x379, &cosine(4.451014462540858, &x379)).unwrap(),
        ),
        (
            "cos 330",
            aaa(&x330, &cosine(7.7971180541115865, &x330)).unwrap(),
        ),
        (
            "rising",
            aaa_with(&x405, &rising, &no_tolerance.with_max_terms(28)).unwrap(),
        ),
    ] {
        let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
        assert!(poles <= 8.0 && zeros <= 8.0, "{name}: {poles} {zeros}");
        assert_eq!(r.poles().len(), r.degree(), "{name}");
        let missed = real_poles_beyond_the_samples_missed_by(&r);
        assert!(
            missed < 16.0,
            "{name}: a real pole {missed} levels from every pole"
        );
        for p in r.poles() {
            assert!(
                p.im == 0.0 || r.poles().iter().any(|q| *q == p.conj()),
                "{name}: {p} without its conjugate"
            );
        }
    }
    // A spurious pair too, but one the data hold better: 2.0e-8 from this on aarch64 and 1.7e-5
    // on x86_64, whose weights are 4.4e-6 from aarch64's.
    let pair = Complex::new(-0.39516266922719656, 0.7108123300665021);
    let poles = aaa(&x686, &rational).unwrap().poles();
    assert!(nearest(&poles, pair) < 1e-4, "{poles:?}");
}

/// A seeded xorshift generator of uniform numbers in [0, 1).
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.next()
    }
}

#[test]
fn random_approximations_have_their_poles_and_zeros_to_rounding() {
    if cfg!(miri) {
        // A hundred and fifty approximations, each a sequence of singular value decompositions.
        return;
    }
    // 150 approximations from 40 to 250 samples, equally spaced or at Chebyshev points: in turn
    // cos(kπx) for k in [3, 13] at the default options, e^(cx)·sin(ax + b) with no tolerance and
    // a term limit from 15 to 45, and Re 1/(x − p − qi) + sin(ax) at the default options. For
    // each, every pole and zero within 8 ε (backward error), as many poles as the degree, and
    // every real zero of the denominator beyond the samples among the poles. Four of these were
    // wrong before the restarts, all with no tolerance: poles off by backward errors of 2.2e3 to
    // 3.6e7 ε, real poles beyond the samples missing.
    let pi = core::f64::consts::PI;
    let mut rng = Xorshift(79190);
    let mut worst: f64 = 0.0;
    for i in 0..150 {
        let m = 40 + (211.0 * rng.next()) as usize;
        let x = if rng.next() < 0.5 {
            grid(m, -1.0, 1.0)
        } else {
            chebyshev_points(m)
        };
        let (f, o): (Vec<f64>, AaaOptions<f64>) = match i % 3 {
            0 => {
                let k = rng.range(3.0, 13.0);
                let f = x.iter().map(|t| (k * pi * t).cos()).collect();
                (f, AaaOptions::new())
            }
            1 => {
                let (a, b, c) = (
                    rng.range(5.0, 40.0),
                    rng.range(0.0, 3.0),
                    rng.range(-2.0, 2.0),
                );
                let terms = 15 + (31.0 * rng.next()) as usize;
                let f = x
                    .iter()
                    .map(|t| (c * t).exp() * (a * t + b).sin())
                    .collect();
                (f, AaaOptions::new().with_rtol(0.0).with_max_terms(terms))
            }
            _ => {
                let (p, q, a) = (
                    rng.range(-1.5, 1.5),
                    rng.range(0.01, 0.5),
                    rng.range(1.0, 30.0),
                );
                let f = x
                    .iter()
                    .map(|t| (t - p) / ((t - p) * (t - p) + q * q) + (a * t).sin())
                    .collect();
                (f, AaaOptions::new())
            }
        };
        let r = aaa_with(&x, &f, &o).unwrap();
        let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
        assert!(poles <= 8.0 && zeros <= 8.0, "case {i}: {poles} {zeros}");
        worst = worst.max(poles).max(zeros);
        assert_eq!(r.poles().len(), r.degree(), "case {i}");
        let missed = real_poles_beyond_the_samples_missed_by(&r);
        assert!(
            missed < 16.0,
            "case {i}: a real pole {missed} levels from every pole"
        );
    }
    // Measured: 1.0 ε at worst.
    assert!(worst <= 8.0);
}

#[test]
fn a_far_pair_is_not_taken_for_two_real_poles_it_is_far_from() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with hundreds of rows, step after step.
        return;
    }
    // cos(12.770224537185758πx) at 238 Chebyshev points: on aarch64 the denominator has the real
    // zeros −5.351 and 4.846 beyond the samples, ill-conditioned (rounding moves them by 0.02). A
    // conjugate pair 12.98 ± 11.20i, two hundred times that from both, has a backward error of
    // 109 ε, which a bound of 4·(m + 1)·ε let pass for a pair of poles (with 8 ε it is started
    // again, and the two real poles are found). Those poles are spurious, and rest on the last
    // bits of the data and the weights: on x86_64 the Chebyshev points (from the libm's cosine)
    // differ in the last bit at three of the 39 support points, the same samples as on aarch64,
    // the decompositions give weights 2.4e-4 apart, and the denominator has no real zero beyond
    // the samples, the pair 0.125 ± 3.528i in their place, with no far pair to reject. There the
    // case checks only what every approximation must satisfy: so the test asks for the real zeros
    // of the denominator returned, each a reported pole, rather than for −5.351 and 4.846.
    let k = 12.770224537185758;
    let x = chebyshev_points(238);
    let f: Vec<f64> = x
        .iter()
        .map(|t| (k * core::f64::consts::PI * t).cos())
        .collect();
    let r = aaa(&x, &f).unwrap();
    let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
    assert!(poles <= 8.0 && zeros <= 8.0, "{poles} {zeros}");
    assert_eq!(r.poles().len(), r.degree());
    let missed = real_poles_beyond_the_samples_missed_by(&r);
    assert!(missed < 16.0, "a real pole {missed} levels from every pole");
}

#[test]
fn multiple_zeros_are_given_as_often_as_they_are_found() {
    if cfg!(miri) {
        // Singular value decompositions of matrices with a thousand rows, step after step.
        return;
    }
    // (x − 0.3)^p·(1 + x²) at 1000 equally spaced points: the numerator has p zeros within about
    // ε^(1/p) of 0.3, rounding moves them that far, and the test of one zero found twice may not
    // take them for one: the slope of the polynomial points opposite ways at two neighbours of a
    // double zero, and among more than two it is turned by the others, where nothing tells. Taking
    // the slope alone, two of four zeros, two of five and three of six were given; with no slope
    // test, the double zero of x²/(x² + 0.01) was given once.
    let x = grid(1000, -1.0, 1.0);
    for p in [2, 4, 6] {
        let f: Vec<f64> = x
            .iter()
            .map(|t| (t - 0.3).powi(p) * (1.0 + t * t))
            .collect();
        let r = aaa(&x, &f).unwrap();
        let zeros = r.zeros();
        let near = zeros
            .iter()
            .filter(|z| (*z - Complex::new(0.3, 0.0)).abs() < 0.05)
            .count();
        assert_eq!((zeros.len(), near), (r.degree(), p as usize), "p {p}");
        let (poles, zeros) = worst_backward_error(&r, f64::EPSILON);
        assert!(poles <= 8.0 && zeros <= 8.0, "p {p}: {poles} {zeros}");
    }
    let f: Vec<f64> = x.iter().map(|t| t * t / (t * t + 0.01)).collect();
    let r = aaa(&x, &f).unwrap();
    let near: Vec<Complex<f64>> = r.zeros().into_iter().filter(|z| z.abs() < 1e-9).collect();
    assert_eq!(near.len(), 2, "{:?}", r.zeros());
}

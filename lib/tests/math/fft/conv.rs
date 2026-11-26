//! Convolutions and correlations against direct sums in double-double.

use super::common::*;
use a::math::complex::Complex;
use a::math::fft::{
    FftFloat, circular_convolve, circular_correlate, convolve, convolve_into, correlate,
    correlate_into,
};

fn direct(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut c = vec![(Dd::default(), Dd::default()); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for (j, y) in b.iter().enumerate() {
            let (re, im) = &mut c[i + j];
            *re = re.add(Dd::prod(x.re, y.re)).add(Dd::prod(-x.im, y.im));
            *im = im.add(Dd::prod(x.re, y.im)).add(Dd::prod(x.im, y.re));
        }
    }
    c.into_iter()
        .map(|(r, i)| Complex::new(r.value(), i.value()))
        .collect()
}

fn circular(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let n = a.len();
    let lin = direct(a, b);
    let mut c = vec![Complex::ZERO; n];
    for (k, v) in lin.into_iter().enumerate() {
        c[k % n] += v;
    }
    c
}

/// The error of a convolution relative to `‖a‖·‖b‖`, the scale its rounding errors have.
fn conv_error(
    got: &[Complex<f64>],
    want: &[Complex<f64>],
    a: &[Complex<f64>],
    b: &[Complex<f64>],
) -> f64 {
    let norm = |v: &[Complex<f64>]| v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
    let diff: Vec<_> = got.iter().zip(want).map(|(g, w)| *g - *w).collect();
    norm(&diff) / (norm(a) * norm(b)).max(f64::MIN_POSITIVE)
}

fn pairs() -> Vec<(usize, usize)> {
    let mut v = vec![
        (0, 0),
        (0, 5),
        (5, 0),
        (1, 1),
        (1, 7),
        (7, 1),
        (3, 3),
        (8, 9),
        (16, 16),
        (33, 2),
        (2, 33),
    ];
    if !cfg!(miri) {
        v.extend_from_slice(&[
            (64, 64),
            (100, 10),
            (10, 100),
            (65, 70),
            (200, 300),
            (1000, 1000),
            (4000, 17),
            (5000, 3000),
        ]);
    }
    v
}

fn check_real<T: FftFloat>(rng: &mut Rng, c: f64) {
    for (na, nb) in pairs() {
        let (a, b) = (rng.reals::<T>(na), rng.reals::<T>(nb));
        let cx = |v: &[T]| {
            widen_real(v)
                .into_iter()
                .map(|x| Complex::new(x, 0.0))
                .collect::<Vec<_>>()
        };
        let (aw, bw) = (cx(&a), cx(&b));
        let got = cx(&convolve(&a, &b));
        let want = direct(&aw, &bw);
        assert_eq!(got.len(), want.len());
        let e = conv_error(&got, &want, &aw, &bw);
        assert!(
            e <= bound::<T>(na + nb, c),
            "convolve {na}x{nb} {}: {e:e}",
            core::any::type_name::<T>()
        );
        let rev: Vec<_> = bw.iter().rev().copied().collect();
        let got = cx(&correlate(&a, &b));
        let e = conv_error(&got, &direct(&aw, &rev), &aw, &bw);
        assert!(e <= bound::<T>(na + nb, c), "correlate {na}x{nb}: {e:e}");
        if na == nb {
            let got = cx(&circular_convolve(&a, &b));
            let e = conv_error(
                &got,
                &if na == 0 {
                    Vec::new()
                } else {
                    circular(&aw, &bw)
                },
                &aw,
                &bw,
            );
            assert!(e <= bound::<T>(na, c), "circular {na}: {e:e}");
        }
    }
}

fn check_complex<T: FftFloat>(rng: &mut Rng, c: f64) {
    for (na, nb) in pairs() {
        let (a, b) = (rng.complexes::<T>(na), rng.complexes::<T>(nb));
        let (aw, bw) = (widen(&a), widen(&b));
        let e = conv_error(&widen(&convolve(&a, &b)), &direct(&aw, &bw), &aw, &bw);
        assert!(
            e <= bound::<T>(na + nb, c),
            "complex convolve {na}x{nb}: {e:e}"
        );
        let rev: Vec<_> = bw.iter().rev().map(|v| v.conj()).collect();
        let e = conv_error(&widen(&correlate(&a, &b)), &direct(&aw, &rev), &aw, &bw);
        assert!(
            e <= bound::<T>(na + nb, c),
            "complex correlate {na}x{nb}: {e:e}"
        );
        if na == nb && na > 0 {
            let e = conv_error(
                &widen(&circular_convolve(&a, &b)),
                &circular(&aw, &bw),
                &aw,
                &bw,
            );
            assert!(e <= bound::<T>(na, c), "complex circular {na}: {e:e}");
            // c[k] = Σ a[(j + k) mod n]·conj(b[j]) = circular(a, b') with b'[j] = conj(b[−j]).
            let flipped: Vec<_> = (0..na).map(|j| bw[(na - j) % na].conj()).collect();
            let e = conv_error(
                &widen(&circular_correlate(&a, &b)),
                &circular(&aw, &flipped),
                &aw,
                &bw,
            );
            assert!(
                e <= bound::<T>(na, c),
                "complex circular correlate {na}: {e:e}"
            );
        }
    }
}

#[test]
fn real_convolutions_and_correlations() {
    let mut rng = Rng(42);
    check_real::<f64>(&mut rng, 10.0);
    check_real::<f32>(&mut rng, 10.0);
}

#[test]
fn complex_convolutions_and_correlations() {
    let mut rng = Rng(43);
    check_complex::<f64>(&mut rng, 10.0);
    check_complex::<f32>(&mut rng, 10.0);
}

#[test]
fn circular_lengths_across_the_crossover() {
    let mut rng = Rng(44);
    let lens: &[usize] = if cfg!(miri) {
        &[1, 2, 5, 33]
    } else {
        &[1, 2, 5, 31, 32, 33, 64, 97, 100, 257, 1000]
    };
    for &n in lens {
        let (a, b) = (rng.reals::<f64>(n), rng.reals::<f64>(n));
        let cx = |v: &[f64]| v.iter().map(|&x| Complex::new(x, 0.0)).collect::<Vec<_>>();
        let e = conv_error(
            &cx(&circular_convolve(&a, &b)),
            &circular(&cx(&a), &cx(&b)),
            &cx(&a),
            &cx(&b),
        );
        assert!(e <= bound::<f64>(n, 10.0), "n={n}: {e:e}");
        let flipped: Vec<_> = (0..n).map(|j| b[(n - j) % n]).collect();
        let e = conv_error(
            &cx(&circular_correlate(&a, &b)),
            &circular(&cx(&a), &cx(&flipped)),
            &cx(&a),
            &cx(&b),
        );
        assert!(e <= bound::<f64>(n, 10.0), "correlate n={n}: {e:e}");
    }
}

/// `convolve_into` and `correlate_into` give the bits of `convolve` and `correlate` on every
/// path (scalar, direct, padded product, overlap-add), into a slice of the exact length.
fn check_into<T: FftFloat>(rng: &mut Rng) {
    for (na, nb) in pairs() {
        let (a, b) = (rng.reals::<T>(na), rng.reals::<T>(nb));
        let want = convolve(&a, &b);
        let mut got = vec![T::from_f64(7.0); want.len()];
        convolve_into(&a, &b, &mut got);
        assert_eq!(got, want, "convolve_into {na}x{nb}");
        let want = correlate(&a, &b);
        correlate_into(&a, &b, &mut got);
        assert_eq!(got, want, "correlate_into {na}x{nb}");
        let (a, b) = (rng.complexes::<T>(na), rng.complexes::<T>(nb));
        let want = convolve(&a, &b);
        let mut got = vec![Complex::new(T::from_f64(7.0), T::ZERO); want.len()];
        convolve_into(&a, &b, &mut got);
        assert_eq!(got, want, "complex convolve_into {na}x{nb}");
        let want = correlate(&a, &b);
        correlate_into(&a, &b, &mut got);
        assert_eq!(got, want, "complex correlate_into {na}x{nb}");
    }
}

#[test]
fn into_slices_agree_with_the_vectors() {
    let mut rng = Rng(77);
    check_into::<f64>(&mut rng);
    check_into::<f32>(&mut rng);
}

#[test]
#[should_panic(expected = "len(a) + len(b) − 1 values")]
fn into_slices_must_have_the_length() {
    let mut out = [0.0; 4];
    convolve_into(&[1.0, 2.0], &[1.0, 2.0], &mut out);
}

#[test]
#[should_panic(expected = "lengths differ")]
fn circular_lengths_must_agree() {
    let _ = circular_convolve(&[1.0, 2.0], &[1.0]);
}

/// The linear convolution in double-double when `b` is mostly zeros, in time `O(len(a)·nonzeros)`.
fn direct_sparse(a: &[Complex<f64>], b: &[Complex<f64>]) -> Vec<Complex<f64>> {
    let nz: Vec<(usize, Complex<f64>)> = b
        .iter()
        .enumerate()
        .filter(|(_, v)| v.re != 0.0 || v.im != 0.0)
        .map(|(j, &v)| (j, v))
        .collect();
    let mut c = vec![(Dd::default(), Dd::default()); a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        for &(j, y) in &nz {
            let (re, im) = &mut c[i + j];
            *re = re.add(Dd::prod(x.re, y.re)).add(Dd::prod(-x.im, y.im));
            *im = im.add(Dd::prod(x.re, y.im)).add(Dd::prod(x.im, y.re));
        }
    }
    c.into_iter()
        .map(|(r, i)| Complex::new(r.value(), i.value()))
        .collect()
}

/// `n` values with `nz` of them nonzero, spread out, the first and the last among them.
fn sparse<T: FftFloat>(rng: &mut Rng, n: usize, nz: usize, complex: bool) -> Vec<Complex<T>> {
    let mut v = vec![Complex::ZERO; n];
    for q in 0..nz.min(n) {
        let j = if nz <= 1 { 0 } else { q * (n - 1) / (nz - 1) };
        let im = if complex { rng.unit() } else { 0.0 };
        v[j] = Complex::new(T::from_f64(rng.unit()), T::from_f64(im));
    }
    v
}

fn re<T: FftFloat>(v: &[Complex<T>]) -> Vec<T> {
    v.iter().map(|x| x.re).collect()
}

fn cxr<T: FftFloat>(v: &[T]) -> Vec<Complex<f64>> {
    v.iter().map(|x| Complex::new(x.to_f64(), 0.0)).collect()
}

/// Every count of taps of the direct kernels (their steps of 2·W taps, the taps left over, the
/// outputs at the ends and past the last full step), both operands longer, convolutions and
/// correlations, real and complex. The bound is tighter than the others' (c = 4 for 10): the
/// direct sums measure below 1.7 units of rounding.
fn check_tap_counts<T: FftFloat>(rng: &mut Rng) {
    let nts: Vec<usize> = if cfg!(miri) {
        vec![1, 2, 3, 5, 9, 17]
    } else {
        (1..=40).chain([47, 64, 65, 100]).collect()
    };
    for &nt in &nts {
        let nls: Vec<usize> = if cfg!(miri) {
            vec![nt + 1, 70]
        } else {
            vec![nt, nt + 1, 2 * nt + 5, 64 + nt - 1, 300]
        };
        for nl in nls {
            for (na, nb) in [(nl, nt), (nt, nl)] {
                let (a, b) = (rng.reals::<T>(na), rng.reals::<T>(nb));
                let (aw, bw) = (cxr(&a), cxr(&b));
                let e = conv_error(&cxr(&convolve(&a, &b)), &direct(&aw, &bw), &aw, &bw);
                assert!(e <= bound::<T>(na + nb, 4.0), "convolve {na}x{nb}: {e:e}");
                let rev: Vec<_> = bw.iter().rev().copied().collect();
                let e = conv_error(&cxr(&correlate(&a, &b)), &direct(&aw, &rev), &aw, &bw);
                assert!(e <= bound::<T>(na + nb, 4.0), "correlate {na}x{nb}: {e:e}");
                let (a, b) = (rng.complexes::<T>(na), rng.complexes::<T>(nb));
                let (aw, bw) = (widen(&a), widen(&b));
                let e = conv_error(&widen(&convolve(&a, &b)), &direct(&aw, &bw), &aw, &bw);
                assert!(
                    e <= bound::<T>(na + nb, 4.0),
                    "complex convolve {na}x{nb}: {e:e}"
                );
                let rev: Vec<_> = bw.iter().rev().map(|v| v.conj()).collect();
                let e = conv_error(&widen(&correlate(&a, &b)), &direct(&aw, &rev), &aw, &bw);
                assert!(
                    e <= bound::<T>(na + nb, 4.0),
                    "complex correlate {na}x{nb}: {e:e}"
                );
            }
        }
    }
}

#[test]
fn direct_kernels_every_tap_count() {
    let mut rng = Rng(45);
    check_tap_counts::<f64>(&mut rng);
    check_tap_counts::<f32>(&mut rng);
}

/// Long operands, by transforms (padded products and overlap-add of many blocks), against
/// references in double-double on a sparse shorter operand.
fn check_long<T: FftFloat>(rng: &mut Rng) {
    for &(nl, nt, nz) in &[
        (3000, 2999, 40),
        (5000, 700, 30),
        (20_000, 50, 50),
        (50_000, 300, 20),
        (100_000, 1000, 12),
        (200_000, 5000, 8),
        (65_537, 4097, 8),
    ] {
        for complex in [false, true] {
            let long = if complex {
                rng.complexes::<T>(nl)
            } else {
                rng.reals::<T>(nl)
                    .into_iter()
                    .map(|x| Complex::new(x, T::ZERO))
                    .collect()
            };
            let short = sparse::<T>(rng, nt, nz, complex);
            let (lw, sw) = (widen(&long), widen(&short));
            let want = direct_sparse(&lw, &sw);
            // Both orders, as convolutions; the correlation of the long with the short
            // reversed (and conjugated) is the same convolution.
            let srev: Vec<Complex<T>> = short.iter().rev().map(|v| v.conj()).collect();
            let gots: Vec<Vec<Complex<f64>>> = if complex {
                vec![
                    widen(&convolve(&long, &short)),
                    widen(&convolve(&short, &long)),
                    widen(&correlate(&long, &srev)),
                ]
            } else {
                let (l, s, r) = (re(&long), re(&short), re(&srev));
                vec![
                    cxr(&convolve(&l, &s)),
                    cxr(&convolve(&s, &l)),
                    cxr(&correlate(&l, &r)),
                ]
            };
            for (q, got) in gots.iter().enumerate() {
                let e = conv_error(got, &want, &lw, &sw);
                assert!(
                    e <= bound::<T>(nl + nt, 10.0),
                    "{} {nl}x{nt} complex={complex} form {q}: {e:e}",
                    core::any::type_name::<T>()
                );
            }
        }
    }
}

#[test]
#[cfg_attr(miri, ignore = "too long for Miri")]
fn long_operands_by_transforms() {
    let mut rng = Rng(46);
    check_long::<f64>(&mut rng);
    check_long::<f32>(&mut rng);
}

/// Circular convolutions and correlations of every kind of length (direct, by a transform of
/// the length, by a padded linear convolution for a large prime factor), real and complex, with
/// a sparse `b` for the reference.
fn check_circular<T: FftFloat>(rng: &mut Rng) {
    let lens: &[usize] = if cfg!(miri) {
        &[3, 40, 67]
    } else {
        &[
            3, 40, 67, 128, 200, 255, 256, 1000, 1009, 1024, 2310, 4096, 4099, 8191, 10_007, 20_000,
        ]
    };
    for &n in lens {
        for complex in [false, true] {
            let a = if complex {
                rng.complexes::<T>(n)
            } else {
                rng.reals::<T>(n)
                    .into_iter()
                    .map(|x| Complex::new(x, T::ZERO))
                    .collect()
            };
            let b = sparse::<T>(rng, n, 9, complex);
            let (aw, bw) = (widen(&a), widen(&b));
            let lin = direct_sparse(&aw, &bw);
            let mut want = vec![Complex::ZERO; n];
            for (k, v) in lin.into_iter().enumerate() {
                want[k % n] += v;
            }
            // b'[j] = conj(b[−j]): the correlation with b' is the convolution with b.
            let flip: Vec<Complex<T>> = (0..n).map(|j| b[(n - j) % n].conj()).collect();
            let gots: Vec<Vec<Complex<f64>>> = if complex {
                vec![
                    widen(&circular_convolve(&a, &b)),
                    widen(&circular_correlate(&a, &flip)),
                ]
            } else {
                let (x, y, f) = (re(&a), re(&b), re(&flip));
                vec![
                    cxr(&circular_convolve(&x, &y)),
                    cxr(&circular_correlate(&x, &f)),
                ]
            };
            for (q, got) in gots.iter().enumerate() {
                let e = conv_error(got, &want, &aw, &bw);
                assert!(
                    e <= bound::<T>(n, 10.0),
                    "{} circular {n} complex={complex} form {q}: {e:e}",
                    core::any::type_name::<T>()
                );
            }
        }
    }
}

#[test]
fn circular_every_path() {
    let mut rng = Rng(47);
    check_circular::<f64>(&mut rng);
    check_circular::<f32>(&mut rng);
}

#[test]
fn exact_small_integers() {
    // Integers small enough that every path is exact up to its rounding of the transforms:
    // the direct path exactly, the transforms to within a half.
    let pairs: &[(usize, usize)] = if cfg!(miri) {
        &[(5, 3), (100, 7)]
    } else {
        &[(5, 3), (100, 7), (1000, 37), (3000, 2000)]
    };
    for &(na, nb) in pairs {
        let a: Vec<f64> = (0..na).map(|i| ((i * 7 + 3) % 11) as f64 - 5.0).collect();
        let b: Vec<f64> = (0..nb).map(|i| ((i * 5 + 1) % 7) as f64 - 3.0).collect();
        let mut want = vec![0.0; na + nb - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                want[i + j] += x * y;
            }
        }
        let got = convolve(&a, &b);
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-6, "{na}x{nb} [{k}]: {g} vs {w}");
        }
    }
}

// The cache of plans, seen through the public API: a call whose plan is kept allocates less than
// one that plans (the allocations are counted on the calling thread only), and gives the same
// bits.

use crate::allocations::count;

/// Lengths of circular convolutions that go through transforms of the length itself, used by no
/// other test: 17 of them, one more than the cache keeps.
const CACHE_LENGTHS: [usize; 17] = [
    4608, 5120, 6144, 6912, 7168, 7680, 8064, 8640, 9216, 10240, 10752, 11520, 12288, 12800, 13824,
    14336, 15360,
];

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[cfg_attr(
    miri,
    ignore = "too long for Miri; the cache runs under Miri in the threads test"
)]
fn plan_cache_keeps_then_evicts_the_least_recently_used() {
    let mut rng = Rng(48);
    let ops: Vec<(Vec<f64>, Vec<f64>)> = CACHE_LENGTHS
        .iter()
        .map(|&n| (rng.reals::<f64>(n), rng.reals::<f64>(n)))
        .collect();
    let (a, b) = &ops[0];
    let want = bits(&circular_convolve(a, b));
    // Other tests share the cache and may evict a plan between two calls, or (rarely) plan one of
    // the same length: the sequence is tried a few times, every result must be the same bits.
    let mut kept = false;
    let mut evicted = false;
    for _ in 0..5 {
        let (first, _) = count(|| circular_convolve(a, b));
        assert_eq!(bits(&first), want);
        let (again, hit) = count(|| circular_convolve(a, b));
        assert_eq!(bits(&again), want);
        // The other 16 lengths push the first one out (it is the least recently used).
        for (x, y) in &ops[1..] {
            let _ = circular_convolve(x, y);
        }
        let (after, miss) = count(|| circular_convolve(a, b));
        assert_eq!(bits(&after), want, "the same bits from a new plan");
        let (_, hit_again) = count(|| circular_convolve(a, b));
        if miss > hit && hit_again < miss {
            evicted = true;
        }
        if hit_again <= hit {
            kept = true;
        }
        if kept && evicted {
            break;
        }
    }
    assert!(kept, "a repeated call kept its plan");
    assert!(evicted, "a plan was evicted after 16 others");
}

#[test]
#[cfg_attr(miri, ignore = "too long for Miri")]
fn plan_cache_skips_plans_past_its_bytes() {
    // 3·2^18 reals of f64: an estimated 24 MiB of plan, past the 16 MiB the cache keeps.
    let n = 3 << 18;
    let mut rng = Rng(49);
    let (a, b) = (rng.reals::<f64>(n), rng.reals::<f64>(n));
    let (x, y) = (
        rng.reals::<f64>(CACHE_LENGTHS[3]),
        rng.reals::<f64>(CACHE_LENGTHS[3]),
    );
    let small = bits(&circular_convolve(&x, &y));
    let mut kept = false;
    for _ in 0..5 {
        let (_, small_hit) = count(|| circular_convolve(&x, &y));
        let (first, c1) = count(|| circular_convolve(&a, &b));
        let (second, c2) = count(|| circular_convolve(&a, &b));
        assert_eq!(bits(&first), bits(&second));
        assert_eq!(c1, c2, "both plan: the plan is not kept");
        // The large plan evicted nothing: the small one is still kept (unless another test
        // pushed it out meanwhile: then the sequence is tried again).
        let (again, c3) = count(|| circular_convolve(&x, &y));
        assert_eq!(bits(&again), small);
        if c3 <= small_hit {
            kept = true;
            break;
        }
    }
    assert!(kept, "the large plan evicted the small one");
}

#[test]
fn plan_cache_shared_by_threads() {
    // Convolutions by transforms of several types and lengths, run on many threads at once while
    // another evicts plans: every result is the same bits as on one thread.
    let pairs: &[(usize, usize)] = if cfg!(miri) {
        &[(1000, 1000)]
    } else {
        &[(1000, 1000), (3000, 2500), (20_000, 3000), (5000, 4000)]
    };
    let mut rng = Rng(50);
    let real: Vec<(Vec<f64>, Vec<f64>)> = pairs
        .iter()
        .map(|&(m, n)| (rng.reals::<f64>(m), rng.reals::<f64>(n)))
        .collect();
    type Pair = (Vec<Complex<f32>>, Vec<Complex<f32>>);
    let cplx: Vec<Pair> = pairs
        .iter()
        .map(|&(m, n)| (rng.complexes::<f32>(m), rng.complexes::<f32>(n)))
        .collect();
    let want_r: Vec<Vec<f64>> = real.iter().map(|(a, b)| convolve(a, b)).collect();
    let want_c: Vec<Vec<Complex<f32>>> = cplx.iter().map(|(a, b)| correlate(a, b)).collect();
    let threads = if cfg!(miri) { 3 } else { 8 };
    std::thread::scope(|s| {
        for t in 0..threads {
            let (real, cplx, want_r, want_c) = (&real, &cplx, &want_r, &want_c);
            s.spawn(move || {
                for round in 0..if cfg!(miri) { 1 } else { 3 } {
                    for q in 0..real.len() {
                        let i = (q + t + round) % real.len();
                        let (a, b) = &real[i];
                        assert_eq!(bits(&convolve(a, b)), bits(&want_r[i]), "thread {t}");
                        let (a, b) = &cplx[i];
                        let got = correlate(a, b);
                        assert!(
                            got.iter()
                                .zip(&want_c[i])
                                .all(|(x, y)| x.re.to_bits() == y.re.to_bits()
                                    && x.im.to_bits() == y.im.to_bits()),
                            "thread {t}"
                        );
                    }
                }
            });
        }
        if !cfg!(miri) {
            s.spawn(|| {
                let mut rng = Rng(51);
                for &n in &CACHE_LENGTHS {
                    let (a, b) = (rng.reals::<f64>(n), rng.reals::<f64>(n));
                    let _ = circular_convolve(&a, &b);
                }
            });
        }
    });
}

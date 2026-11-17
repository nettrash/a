//! The `_with_scratch` methods of the real, cosine and sine transforms against the allocating
//! ones, short scratch, empty batches of every algorithm, and clones.

use super::common::*;
use a::math::complex::Complex;
use a::math::fft::{Dct, DctKind, Dst, DstKind, FftFloat, RealFft};

/// Lengths of every algorithm of `RealFft`: direct (odd ≤ 15, even ≤ 16), halves, odd splits by
/// 3, 5, 7, 11 and recursively, the complex transform of the length (primes, 13·13, 13·17).
fn real_lengths() -> Vec<usize> {
    if cfg!(miri) {
        return vec![0, 1, 2, 9, 16, 18, 21, 34, 45];
    }
    vec![
        0, 1, 2, 3, 4, 9, 15, 16, 17, 18, 21, 25, 32, 34, 45, 49, 64, 77, 101, 121, 143, 169, 221,
        243, 1000, 1024, 1155, 4096,
    ]
}

/// Lengths of every algorithm of the cosine and sine transforms: direct, Makhoul (odd and
/// even, the real transform direct or by halves), IV even and odd, I split (recursively) and
/// padded.
fn trig_lengths() -> Vec<usize> {
    if cfg!(miri) {
        return vec![0, 2, 3, 9, 17, 18, 33];
    }
    vec![
        0, 2, 3, 4, 8, 9, 10, 15, 16, 17, 18, 20, 31, 32, 33, 34, 63, 64, 65, 100, 127, 129, 255,
        256, 1000, 1025,
    ]
}

const DCTS: [DctKind; 4] = [DctKind::I, DctKind::II, DctKind::III, DctKind::IV];
const DSTS: [DstKind; 4] = [DstKind::I, DstKind::II, DstKind::III, DstKind::IV];

#[test]
fn empty_batches_of_every_algorithm() {
    for n in real_lengths() {
        for_both(n);
    }
    fn for_both(n: usize) {
        empty_real::<f64>(n);
        empty_real::<f32>(n);
    }
    fn empty_real<T: FftFloat>(n: usize) {
        let fft = RealFft::<T>::new(n);
        let mut spec: Vec<Complex<T>> = Vec::new();
        let mut reals: Vec<T> = Vec::new();
        fft.forward(&[], &mut spec);
        fft.inverse(&[], &mut reals);
        let mut scratch = vec![Complex::ZERO; fft.scratch_len()];
        fft.forward_with_scratch(&[], &mut spec, &mut scratch);
        fft.inverse_with_scratch(&[], &mut reals, &mut scratch);
    }
    for n in trig_lengths() {
        let mut empty: Vec<f64> = Vec::new();
        let mut empty32: Vec<f32> = Vec::new();
        for kind in DCTS {
            if kind == DctKind::I && n == 1 {
                continue;
            }
            let plan = Dct::<f64>::new(n, kind);
            plan.transform(&mut empty);
            plan.transform_to(&[], &mut empty);
            let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
            plan.transform_with_scratch(&mut empty, &mut scratch);
            Dct::<f32>::new(n, kind).transform(&mut empty32);
        }
        for kind in DSTS {
            let plan = Dst::<f64>::new(n, kind);
            plan.transform(&mut empty);
            plan.transform_to(&[], &mut empty);
            let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
            plan.transform_with_scratch(&mut empty, &mut scratch);
            Dst::<f32>::new(n, kind).transform(&mut empty32);
        }
    }
}

fn real_with_scratch<T: FftFloat>(n: usize, rng: &mut Rng) {
    let fft = RealFft::<T>::new(n);
    let h = fft.spectrum_len();
    for runs in [1usize, 3] {
        let x = rng.reals::<T>(runs * n);
        let mut want = vec![Complex::ZERO; runs * h];
        fft.forward(&x, &mut want);
        // Exactly the scratch one run needs, and more.
        for extra in [0usize, 1000] {
            let mut scratch = vec![Complex::ZERO; fft.scratch_len() + extra];
            let mut got = vec![Complex::ZERO; runs * h];
            fft.forward_with_scratch(&x, &mut got, &mut scratch);
            let e = rel_error(&widen(&got), &widen(&want));
            assert!(e <= bound::<T>(n, 2.0), "forward n={n} runs={runs}: {e:e}");
            let mut back = vec![T::ZERO; runs * n];
            fft.inverse_with_scratch(&want, &mut back, &mut scratch);
            let mut back2 = vec![T::ZERO; runs * n];
            fft.inverse(&want, &mut back2);
            let e = rel_error_real(&widen_real(&back), &widen_real(&back2));
            assert!(e <= bound::<T>(n, 2.0), "inverse n={n} runs={runs}: {e:e}");
            // And each run as on its own.
            for i in 0..runs {
                let mut one = vec![T::ZERO; n];
                fft.inverse(&want[i * h..(i + 1) * h], &mut one);
                let e = rel_error_real(&widen_real(&back[i * n..(i + 1) * n]), &widen_real(&one));
                assert!(
                    e <= bound::<T>(n, 2.0),
                    "inverse batch n={n} run {i}: {e:e}"
                );
            }
        }
    }
}

#[test]
fn real_with_scratch_as_without() {
    let mut rng = Rng(0xABCD);
    for n in real_lengths().into_iter().filter(|&n| n > 0) {
        real_with_scratch::<f64>(n, &mut rng);
        real_with_scratch::<f32>(n, &mut rng);
    }
}

#[test]
fn trig_with_scratch_as_without() {
    let mut rng = Rng(0xDC7);
    for n in trig_lengths().into_iter().filter(|&n| n > 0) {
        let x = rng.reals::<f64>(3 * n);
        for kind in DCTS {
            if kind == DctKind::I && n < 2 {
                continue;
            }
            let plan = Dct::<f64>::new(n, kind);
            let mut want = x.clone();
            plan.transform(&mut want);
            for extra in [0usize, 7] {
                let mut scratch = vec![Complex::ZERO; plan.scratch_len() + extra];
                let mut got = x.clone();
                plan.transform_with_scratch(&mut got, &mut scratch);
                assert_eq!(got, want, "{kind:?} n={n}");
            }
        }
        for kind in DSTS {
            let plan = Dst::<f64>::new(n, kind);
            let mut want = x.clone();
            plan.transform(&mut want);
            let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
            let mut got = x.clone();
            plan.transform_with_scratch(&mut got, &mut scratch);
            assert_eq!(got, want, "{kind:?} n={n}");
        }
    }
}

#[test]
#[should_panic(expected = "scratch")]
fn a_short_scratch_panics_forward() {
    let fft = RealFft::<f64>::new(45);
    assert!(fft.scratch_len() > 0);
    let mut scratch = vec![Complex::ZERO; fft.scratch_len() - 1];
    let mut y = vec![Complex::ZERO; fft.spectrum_len()];
    fft.forward_with_scratch(&[0.0; 45], &mut y, &mut scratch);
}

#[test]
#[should_panic(expected = "scratch")]
fn a_short_scratch_panics_inverse() {
    let fft = RealFft::<f32>::new(1155);
    assert!(fft.scratch_len() > 0);
    let mut scratch = vec![Complex::ZERO; fft.scratch_len() - 1];
    let x = vec![Complex::ZERO; fft.spectrum_len()];
    fft.inverse_with_scratch(&x, &mut [0.0; 1155], &mut scratch);
}

#[test]
#[should_panic(expected = "scratch")]
fn a_short_scratch_panics_dct() {
    let plan = Dct::<f64>::new(100, DctKind::II);
    assert!(plan.scratch_len() > 0);
    let mut scratch = vec![Complex::ZERO; plan.scratch_len() - 1];
    plan.transform_with_scratch(&mut [1.0; 100], &mut scratch);
}

#[test]
#[should_panic(expected = "scratch")]
fn a_short_scratch_panics_dst() {
    let plan = Dst::<f32>::new(33, DstKind::I);
    assert!(plan.scratch_len() > 0);
    let mut scratch = vec![Complex::ZERO; plan.scratch_len() - 1];
    plan.transform_with_scratch(&mut [1.0; 33], &mut scratch);
}

#[test]
fn clones_share_the_plan() {
    let mut rng = Rng(77);
    let x = rng.reals::<f64>(90);
    // Against the plan cloned (not one planned anew: under Miri the sines and cosines of
    // two plans may differ in their last bits).
    let fft = RealFft::<f64>::new(90);
    let copy = fft.clone();
    let mut b = vec![Complex::ZERO; 46];
    fft.forward(&x, &mut b);
    drop(fft);
    let mut a = vec![Complex::ZERO; 46];
    copy.forward(&x, &mut a);
    assert_eq!(a, b);
    let dct = Dct::<f64>::new(90, DctKind::IV);
    let (mut p, mut q) = (x.clone(), x.clone());
    dct.clone().transform(&mut p);
    dct.transform(&mut q);
    assert_eq!(p, q);
    let dst = Dst::<f32>::new(45, DstKind::III);
    let x32 = rng.reals::<f32>(45);
    let (mut p, mut q) = (x32.clone(), x32.clone());
    dst.clone().transform(&mut p);
    dst.transform(&mut q);
    assert_eq!(p, q);
    fn shared<P: Clone + Send + Sync>() {}
    shared::<RealFft<f32>>();
    shared::<Dct<f64>>();
    shared::<Dst<f32>>();
}

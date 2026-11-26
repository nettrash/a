//! Real transforms against the double-double DFT, round trips, batches and the ignored parts.

use super::common::*;
use a::math::complex::Complex;
use a::math::fft::{FftFloat, RealFft};

fn check<T: FftFloat>(n: usize, rng: &mut Rng, c: f64) {
    let fft = RealFft::<T>::new(n);
    let h = fft.spectrum_len();
    assert_eq!(h, if n == 0 { 0 } else { n / 2 + 1 });
    let x = rng.reals::<T>(n);
    let mut y = vec![Complex::ZERO; h];
    fft.forward(&x, &mut y);
    let full = dft(
        &widen_real(&x)
            .iter()
            .map(|&v| Complex::new(v, 0.0))
            .collect::<Vec<_>>(),
        -1.0,
    );
    let e = rel_error(&widen(&y), &full[..h]);
    assert!(
        e <= bound::<T>(n, c),
        "forward n={n} {}: {e:e}",
        core::any::type_name::<T>()
    );
    let mut back = vec![T::ZERO; n];
    fft.inverse(&y, &mut back);
    let want: Vec<f64> = widen_real(&x).iter().map(|v| v * n as f64).collect();
    let e = rel_error_real(&widen_real(&back), &want);
    assert!(e <= bound::<T>(n, 2.0 * c), "round trip n={n}: {e:e}");
}

#[test]
fn dense_lengths_against_the_dft() {
    let mut rng = Rng(0xFEED_FACE_0000_0001);
    for n in dense_lengths() {
        check::<f64>(n, &mut rng, 10.0);
        check::<f32>(n, &mut rng, 10.0);
    }
}

#[test]
fn long_lengths_round_trip() {
    let mut rng = Rng(99);
    for n in long_lengths() {
        let fft = RealFft::<f64>::new(n);
        let x = rng.reals::<f64>(n);
        let mut y = vec![Complex::ZERO; fft.spectrum_len()];
        fft.forward(&x, &mut y);
        // Against the complex transform on the same data, through its sparse-free identity:
        // X[0] = Σ x.
        let s: f64 = x.iter().sum();
        assert!(
            (y[0].re - s).abs() <= 1e-12 * (n as f64).sqrt() && y[0].im == 0.0,
            "X[0] n={n}"
        );
        let mut back = vec![0.0; n];
        fft.inverse(&y, &mut back);
        let want: Vec<f64> = x.iter().map(|v| v * n as f64).collect();
        let e = rel_error_real(&back, &want);
        assert!(e <= bound::<f64>(n, 20.0), "round trip n={n}: {e:e}");
    }
}

#[test]
fn the_inverse_ignores_the_imaginary_parts_of_the_real_bins() {
    for n in [1usize, 2, 3, 4, 7, 8, 16, 33] {
        let fft = RealFft::<f64>::new(n);
        let mut rng = Rng(n as u64 + 5);
        let x = rng.reals::<f64>(n);
        let mut y = vec![Complex::ZERO; fft.spectrum_len()];
        fft.forward(&x, &mut y);
        let mut a = vec![0.0; n];
        fft.inverse(&y, &mut a);
        y[0].im = 123.0;
        if n.is_multiple_of(2) {
            y[n / 2].im = -45.0;
        }
        let mut b = vec![0.0; n];
        fft.inverse(&y, &mut b);
        assert!(rel_error_real(&a, &b) < 1e-14, "n={n}");
    }
}

#[test]
fn batches() {
    let mut rng = Rng(3);
    for n in [1usize, 2, 3, 6, 9, 16, 31] {
        let fft = RealFft::<f32>::new(n);
        let h = fft.spectrum_len();
        let x = rng.reals::<f32>(4 * n);
        let mut all = vec![Complex::ZERO; 4 * h];
        fft.forward(&x, &mut all);
        for i in 0..4 {
            let mut one = vec![Complex::ZERO; h];
            fft.forward(&x[i * n..(i + 1) * n], &mut one);
            assert!(
                rel_error(&widen(&all[i * h..(i + 1) * h]), &widen(&one)) <= bound::<f32>(n, 4.0)
            );
        }
        let mut back = vec![0.0f32; 4 * n];
        fft.inverse(&all, &mut back);
        let want: Vec<f64> = widen_real(&x).iter().map(|v| v * n as f64).collect();
        assert!(rel_error_real(&widen_real(&back), &want) <= bound::<f32>(n, 20.0));
    }
}

#[test]
#[should_panic(expected = "n/2 + 1")]
fn a_wrong_spectrum_length_panics() {
    let fft = RealFft::<f64>::new(8);
    let mut y = [Complex::ZERO; 4];
    fft.forward(&[0.0; 8], &mut y);
}

/// The lengths of every algorithm: the halves of even lengths, odd lengths summed directly, split
/// by each radix from 3 to 19 (9 and 15 too, recursively), and through the complex transform
/// (primes, and the lengths without a factor up to 19: 23·23, 23·29).
fn algorithm_lengths() -> Vec<usize> {
    if cfg!(miri) {
        return vec![2, 4, 9, 15, 17, 21, 25, 27, 45, 49, 57, 105];
    }
    vec![
        2, 4, 6, 10, 9, 15, 17, 21, 25, 27, 33, 35, 39, 45, 49, 51, 55, 57, 63, 75, 77, 81, 95, 99,
        105, 121, 125, 135, 143, 165, 169, 189, 221, 225, 231, 247, 289, 323, 343, 361, 375, 385,
        405, 441, 525, 529, 667, 675, 729, 1155, 1331, 2048, 2187, 3000, 3375,
    ]
}

#[test]
fn every_algorithm_against_the_dft() {
    let mut rng = Rng(0x5EED);
    for n in algorithm_lengths() {
        check::<f64>(n, &mut rng, 10.0);
        check::<f32>(n, &mut rng, 10.0);
    }
}

#[test]
fn the_spectrum_against_the_complex_transform() {
    use a::math::fft::Fft;
    let lengths: &[usize] = if cfg!(miri) {
        &[16, 45]
    } else {
        &[
            4096, 6144, 10000, 15015, 16807, 19683, 65536, 99225, 100_000,
        ]
    };
    let mut rng = Rng(77);
    for &n in lengths {
        let x = rng.reals::<f64>(n);
        let fft = RealFft::<f64>::new(n);
        let mut y = vec![Complex::ZERO; fft.spectrum_len()];
        fft.forward(&x, &mut y);
        let mut z: Vec<Complex<f64>> = x.iter().map(|&v| Complex::new(v, 0.0)).collect();
        Fft::<f64>::new(n).forward(&mut z);
        let e = rel_error(&y, &z[..y.len()]);
        assert!(e <= bound::<f64>(n, 10.0), "n={n}: {e:e}");
        let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let f32p = RealFft::<f32>::new(n);
        let mut y32 = vec![Complex::ZERO; f32p.spectrum_len()];
        f32p.forward(&x32, &mut y32);
        let e = rel_error(&widen(&y32), &z[..y.len()]);
        assert!(e <= bound::<f32>(n, 10.0), "f32 n={n}: {e:e}");
        let mut back = vec![0.0f32; n];
        f32p.inverse(&y32, &mut back);
        let want: Vec<f64> = x.iter().map(|v| v * n as f64).collect();
        let e = rel_error_real(&widen_real(&back), &want);
        assert!(e <= bound::<f32>(n, 20.0), "f32 round trip n={n}: {e:e}");
    }
}

#[test]
fn residue_classes_of_very_different_sizes() {
    // The odd lengths pair the residue sequences modulo a factor: the error stays relative to
    // the whole spectrum when one class dwarfs the others.
    let mut rng = Rng(123);
    let (lengths, classes): (&[usize], &[usize]) = if cfg!(miri) {
        (&[45], &[3])
    } else {
        (&[45, 105, 135, 225, 343, 1155, 2187], &[3, 5, 7, 9, 15])
    };
    for &n in lengths {
        for &r in classes {
            let x: Vec<f64> = (0..n)
                .map(|j| rng.unit() * if j % r == 1 { 1e6 } else { 1e-6 })
                .collect();
            let fft = RealFft::<f64>::new(n);
            let mut y = vec![Complex::ZERO; fft.spectrum_len()];
            fft.forward(&x, &mut y);
            let full = dft(
                &x.iter().map(|&v| Complex::new(v, 0.0)).collect::<Vec<_>>(),
                -1.0,
            );
            let e = rel_error(&y, &full[..y.len()]);
            assert!(e <= bound::<f64>(n, 10.0), "n={n} r={r}: {e:e}");
            let mut back = vec![0.0; n];
            fft.inverse(&y, &mut back);
            let want: Vec<f64> = x.iter().map(|v| v * n as f64).collect();
            assert!(
                rel_error_real(&back, &want) <= bound::<f64>(n, 20.0),
                "n={n} r={r}"
            );
        }
    }
}

#[test]
fn batches_of_every_algorithm() {
    let mut rng = Rng(4);
    for n in algorithm_lengths().into_iter().filter(|&n| n <= 231) {
        let fft = RealFft::<f64>::new(n);
        let h = fft.spectrum_len();
        let x = rng.reals::<f64>(3 * n);
        let mut all = vec![Complex::ZERO; 3 * h];
        fft.forward(&x, &mut all);
        let mut back = vec![0.0; 3 * n];
        fft.inverse(&all, &mut back);
        for i in 0..3 {
            let mut one = vec![Complex::ZERO; h];
            fft.forward(&x[i * n..(i + 1) * n], &mut one);
            assert_eq!(&all[i * h..(i + 1) * h], &one[..], "n={n}");
            let mut b = vec![0.0; n];
            fft.inverse(&one, &mut b);
            assert_eq!(&back[i * n..(i + 1) * n], &b[..], "n={n}");
        }
    }
}

#[test]
fn every_algorithm_ignores_the_imaginary_parts_of_the_real_bins() {
    for n in algorithm_lengths().into_iter().filter(|&n| n <= 1155) {
        let fft = RealFft::<f64>::new(n);
        let mut rng = Rng(n as u64 + 50);
        let x = rng.reals::<f64>(n);
        let mut y = vec![Complex::ZERO; fft.spectrum_len()];
        fft.forward(&x, &mut y);
        assert_eq!(y[0].im, 0.0, "n={n}");
        if n.is_multiple_of(2) {
            assert_eq!(y[n / 2].im, 0.0, "n={n}");
        }
        let mut a = vec![0.0; n];
        fft.inverse(&y, &mut a);
        y[0].im = 7.0;
        if n.is_multiple_of(2) {
            y[n / 2].im = -3.0;
        }
        let mut b = vec![0.0; n];
        fft.inverse(&y, &mut b);
        assert_eq!(a, b, "n={n}");
    }
}

#[test]
fn plans_are_send_and_sync() {
    fn check<P: Send + Sync>() {}
    check::<RealFft<f32>>();
    check::<RealFft<f64>>();
    check::<a::math::fft::Dct<f64>>();
    check::<a::math::fft::Dst<f32>>();
    check::<a::math::fft::Fft2<f64>>();
    check::<a::math::fft::Fft3<f32>>();
}

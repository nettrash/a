//! Complex transforms against the double-double DFT, round trips, batches and the API's edges.

use super::common::*;
use a::math::complex::Complex;
use a::math::fft::{Fft, FftFloat};

/// One length: forward and inverse against the DFT, out of place against in place.
fn check_dense<T: FftFloat>(n: usize, rng: &mut Rng, c: f64) {
    let fft = Fft::<T>::new(n);
    assert_eq!(fft.len(), n);
    let x = rng.complexes::<T>(n);
    let xw = widen(&x);
    let mut y = x.clone();
    fft.forward(&mut y);
    let e = rel_error(&widen(&y), &dft(&xw, -1.0));
    assert!(
        e <= bound::<T>(n, c),
        "forward n={n} {}: error {e:e}",
        core::any::type_name::<T>()
    );
    let mut z = x.clone();
    fft.inverse(&mut z);
    let e = rel_error(&widen(&z), &dft(&xw, 1.0));
    assert!(e <= bound::<T>(n, c), "inverse n={n}: error {e:e}");
    // Out of place agrees with in place.
    let mut o = vec![Complex::ZERO; n];
    fft.forward_to(&x, &mut o);
    assert!(
        rel_error(&widen(&o), &widen(&y)) <= bound::<T>(n, c),
        "forward_to n={n}"
    );
    fft.inverse_to(&x, &mut o);
    assert!(
        rel_error(&widen(&o), &widen(&z)) <= bound::<T>(n, c),
        "inverse_to n={n}"
    );
}

#[test]
fn dense_lengths_against_the_dft() {
    let mut rng = Rng(0x1234_5678_9ABC_DEF1);
    for n in dense_lengths() {
        check_dense::<f64>(n, &mut rng, 3.0);
        check_dense::<f32>(n, &mut rng, 3.0);
    }
}

/// A length of each way of planning: the butterflies from tables (11 to 19, alone and mixed),
/// a prime factor above them by its own plan (Rader's and Bluestein's), Rader's algorithm with
/// such a factor in `p − 1`, Bluestein's for a prime, and the radix-2 pass with each.
#[test]
fn every_algorithm_against_the_dft() {
    let mut rng = Rng(0x0DD5_EED5_0FA1_1234);
    let lengths: &[usize] = if cfg!(miri) {
        &[22, 46, 47]
    } else {
        &[
            22, 26, 34, 38, 46, 47, 69, 94, 111, 143, 289, 323, 361, 437, 667, 1031, 1427, 2062,
        ]
    };
    for &n in lengths {
        check_dense::<f64>(n, &mut rng, 3.0);
        check_dense::<f32>(n, &mut rng, 3.0);
    }
}

/// Long batches of short transforms (run in chunks pass by pass, the lanes across the runs for
/// one-pass lengths), forward and inverse, in and out of place, against one run at a time.
#[test]
fn long_batches_of_short_transforms() {
    let mut rng = Rng(0x0BA7_C4E5);
    let runs = if cfg!(miri) { 9 } else { 301 };
    for n in [2usize, 3, 4, 5, 7, 8, 11, 12, 16, 17, 30, 64] {
        let fft = Fft::<f32>::new(n);
        let x = rng.complexes::<f32>(runs * n);
        for inv in [false, true] {
            let mut all = x.clone();
            let mut to = vec![Complex::ZERO; x.len()];
            if inv {
                fft.inverse(&mut all);
                fft.inverse_to(&x, &mut to);
            } else {
                fft.forward(&mut all);
                fft.forward_to(&x, &mut to);
            }
            for (i, run) in x.chunks(n).enumerate() {
                let mut one = run.to_vec();
                if inv {
                    fft.inverse(&mut one);
                } else {
                    fft.forward(&mut one);
                }
                let (r, one) = (i * n..(i + 1) * n, widen(&one));
                let b = bound::<f32>(n, 2.0);
                assert!(
                    rel_error(&widen(&all[r.clone()]), &one) <= b,
                    "n={n} run {i} {inv}"
                );
                assert!(
                    rel_error(&widen(&to[r]), &one) <= b,
                    "to n={n} run {i} {inv}"
                );
            }
        }
    }
}

#[test]
fn plans_clone_and_run_on_threads() {
    let fft = Fft::<f64>::new(1000);
    let x = Rng(5).complexes::<f64>(1000);
    let mut want = x.clone();
    fft.forward(&mut want);
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let (f, x) = (fft.clone(), x.clone());
            std::thread::spawn(move || {
                let mut y = x;
                f.forward(&mut y);
                y
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), want);
    }
    let d = format!("{fft:?}");
    println!("{d}");
    assert!(d.contains("n: 1000") && d.contains("isa: "));
}

#[test]
fn long_lengths_on_sparse_data_and_round_trips() {
    let mut rng = Rng(0x0BAD_CAFE_F00D_1234);
    for n in long_lengths() {
        // Eight values at random places.
        let mut x = vec![Complex::<f64>::ZERO; n];
        for _ in 0..8 {
            let j = (rng.next() % n as u64) as usize;
            x[j] = Complex::new(rng.unit(), rng.unit());
        }
        let fft = Fft::<f64>::new(n);
        let mut y = x.clone();
        fft.forward(&mut y);
        let e = rel_error(&y, &dft_sparse(&x, -1.0));
        assert!(
            e <= bound::<f64>(n, 10.0),
            "sparse forward n={n}: error {e:e}"
        );
        // Dense round trip: inverse(forward(x)) = n·x.
        let d = rng.complexes::<f64>(n);
        let mut r = d.clone();
        fft.forward(&mut r);
        fft.inverse(&mut r);
        let want: Vec<_> = d.iter().map(|v| *v * n as f64).collect();
        let e = rel_error(&r, &want);
        assert!(e <= bound::<f64>(n, 20.0), "round trip n={n}: error {e:e}");
        let fft32 = Fft::<f32>::new(n);
        let d32 = rng.complexes::<f32>(n);
        let mut r32 = d32.clone();
        fft32.forward(&mut r32);
        fft32.inverse(&mut r32);
        let want: Vec<_> = widen(&d32).iter().map(|v| *v * n as f64).collect();
        let e = rel_error(&widen(&r32), &want);
        assert!(
            e <= bound::<f32>(n, 20.0),
            "f32 round trip n={n}: error {e:e}"
        );
    }
}

#[test]
fn impulses_and_pure_tones() {
    for n in [1usize, 2, 3, 4, 5, 7, 8, 12, 16, 17, 30, 64, 97] {
        let fft = Fft::<f64>::new(n);
        // δ → all ones.
        let mut x = vec![Complex::ZERO; n];
        x[0] = Complex::ONE;
        fft.forward(&mut x);
        for v in &x {
            assert!((*v - Complex::ONE).abs() < 1e-14, "δ n={n}");
        }
        // e^(2πi·3j/n) → n·δ[k − 3 mod n].
        let k0 = 3 % n;
        let mut t: Vec<Complex<f64>> = (0..n)
            .map(|j| {
                let a = core::f64::consts::TAU * ((j * k0) % n) as f64 / n as f64;
                Complex::new(a.cos(), a.sin())
            })
            .collect();
        fft.forward(&mut t);
        for (k, v) in t.iter().enumerate() {
            let want = if k == k0 { n as f64 } else { 0.0 };
            assert!(
                (*v - Complex::new(want, 0.0)).abs() < 1e-12 * n as f64,
                "tone n={n} k={k}"
            );
        }
    }
}

#[test]
fn batches_transform_each_run() {
    let mut rng = Rng(77);
    for n in [1usize, 2, 5, 8, 12, 17, 64, 100] {
        let fft = Fft::<f64>::new(n);
        let x = rng.complexes::<f64>(3 * n);
        let mut all = x.clone();
        fft.forward(&mut all);
        for (i, run) in x.chunks(n).enumerate() {
            let mut one = run.to_vec();
            fft.forward(&mut one);
            let e = rel_error(&all[i * n..(i + 1) * n], &one);
            assert!(e <= bound::<f64>(n, 4.0), "batch n={n} run {i}: {e:e}");
        }
        let mut scratch = vec![Complex::ZERO; fft.scratch_len()];
        let mut again = x.clone();
        fft.forward_with_scratch(&mut again, &mut scratch);
        assert!(rel_error(&again, &all) <= bound::<f64>(n, 4.0));
        fft.inverse_with_scratch(&mut again, &mut scratch);
        let mut inv = all.clone();
        fft.inverse(&mut inv);
        assert!(rel_error(&again, &inv) <= bound::<f64>(n, 4.0));
    }
}

#[test]
fn empty_and_zero_lengths() {
    let fft = Fft::<f64>::new(0);
    assert!(fft.is_empty());
    let mut x: [Complex<f64>; 0] = [];
    fft.forward(&mut x);
    fft.inverse(&mut x);
    let fft = Fft::<f32>::new(5);
    let mut x: Vec<Complex<f32>> = Vec::new();
    fft.forward(&mut x);
}

#[test]
#[should_panic(expected = "not a multiple")]
fn a_length_that_is_not_a_multiple_panics() {
    let fft = Fft::<f64>::new(4);
    let mut x = [Complex::ZERO; 6];
    fft.forward(&mut x);
}

#[test]
#[should_panic(expected = "scratch")]
fn short_scratch_panics() {
    let fft = Fft::<f64>::new(1009);
    let mut x = vec![Complex::ZERO; 1009];
    let mut s = vec![Complex::ZERO; fft.scratch_len() - 1];
    fft.forward_with_scratch(&mut x, &mut s);
}

#[test]
fn plans_are_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Fft<f32>>();
    check::<Fft<f64>>();
    check::<a::math::fft::RealFft<f64>>();
    check::<a::math::fft::Fft2<f64>>();
    check::<a::math::fft::Fft3<f32>>();
    check::<a::math::fft::Dct<f64>>();
    check::<a::math::fft::Dst<f32>>();
}

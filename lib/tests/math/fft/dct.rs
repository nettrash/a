//! Cosine and sine transforms of the eight types against their definitions in double-double,
//! and the inverse pairs.

use super::common::*;
use a::math::fft::{Dct, DctKind, Dst, DstKind, FftFloat};

/// `cos(π·a/b)` and `sin(π·a/b)` in double-double.
fn trig(a: u64, b: u64) -> (Dd, Dd) {
    let (c, s) = cis(a % (2 * b), 2 * b);
    (c, s)
}

fn sum(terms: impl Iterator<Item = (f64, Dd)>) -> Dd {
    let mut acc = Dd::default();
    for (x, t) in terms {
        acc = acc.add(t.mul_f(x, 0.0));
    }
    acc
}

fn dct_ref(x: &[f64], kind: DctKind) -> Vec<f64> {
    let n = x.len() as u64;
    (0..n)
        .map(|k| {
            let v = match kind {
                DctKind::I => {
                    let s = sum((1..n - 1).map(|j| (x[j as usize], trig(j * k, n - 1).0)));
                    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                    Dd(x[0], 0.0)
                        .add(Dd(sign * x[n as usize - 1], 0.0))
                        .add(Dd(2.0 * s.0, 2.0 * s.1))
                }
                DctKind::II => {
                    let s = sum((0..n).map(|j| (x[j as usize], trig((2 * j + 1) * k, 2 * n).0)));
                    Dd(2.0 * s.0, 2.0 * s.1)
                }
                DctKind::III => {
                    let s = sum((1..n).map(|j| (x[j as usize], trig(j * (2 * k + 1), 2 * n).0)));
                    Dd(x[0], 0.0).add(Dd(2.0 * s.0, 2.0 * s.1))
                }
                DctKind::IV => {
                    let s =
                        sum((0..n)
                            .map(|j| (x[j as usize], trig((2 * j + 1) * (2 * k + 1), 4 * n).0)));
                    Dd(2.0 * s.0, 2.0 * s.1)
                }
            };
            v.value()
        })
        .collect()
}

fn dst_ref(x: &[f64], kind: DstKind) -> Vec<f64> {
    let n = x.len() as u64;
    (0..n)
        .map(|k| {
            let v = match kind {
                DstKind::I => {
                    let s = sum((0..n).map(|j| (x[j as usize], trig((j + 1) * (k + 1), n + 1).1)));
                    Dd(2.0 * s.0, 2.0 * s.1)
                }
                DstKind::II => {
                    let s =
                        sum((0..n).map(|j| (x[j as usize], trig((2 * j + 1) * (k + 1), 2 * n).1)));
                    Dd(2.0 * s.0, 2.0 * s.1)
                }
                DstKind::III => {
                    let s =
                        sum((0..n - 1)
                            .map(|j| (x[j as usize], trig((j + 1) * (2 * k + 1), 2 * n).1)));
                    let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
                    Dd(sign * x[n as usize - 1], 0.0).add(Dd(2.0 * s.0, 2.0 * s.1))
                }
                DstKind::IV => {
                    let s =
                        sum((0..n)
                            .map(|j| (x[j as usize], trig((2 * j + 1) * (2 * k + 1), 4 * n).1)));
                    Dd(2.0 * s.0, 2.0 * s.1)
                }
            };
            v.value()
        })
        .collect()
}

const DCTS: [DctKind; 4] = [DctKind::I, DctKind::II, DctKind::III, DctKind::IV];
const DSTS: [DstKind; 4] = [DstKind::I, DstKind::II, DstKind::III, DstKind::IV];

fn lengths() -> Vec<usize> {
    let mut v: Vec<usize> = (1..=40).collect();
    if !cfg!(miri) {
        v.extend_from_slice(&[63, 64, 65, 100, 127, 128, 129, 255, 256, 500, 1000, 1024]);
    } else {
        v.truncate(17);
    }
    v
}

fn check<T: FftFloat>(rng: &mut Rng) {
    for n in lengths() {
        let x = rng.reals::<T>(n);
        let xw = widen_real(&x);
        for kind in DCTS {
            if kind == DctKind::I && n < 2 {
                continue;
            }
            let plan = Dct::<T>::new(n, kind);
            assert_eq!((plan.len(), plan.kind()), (n, kind));
            let mut y = x.clone();
            plan.transform(&mut y);
            let e = rel_error_real(&widen_real(&y), &dct_ref(&xw, kind));
            assert!(
                e <= bound::<T>(n, 10.0),
                "{kind:?} n={n} {}: {e:e}",
                core::any::type_name::<T>()
            );
            let mut o = vec![T::ZERO; n];
            plan.transform_to(&x, &mut o);
            assert!(rel_error_real(&widen_real(&o), &widen_real(&y)) <= bound::<T>(n, 4.0));
        }
        for kind in DSTS {
            let plan = Dst::<T>::new(n, kind);
            let mut y = x.clone();
            plan.transform(&mut y);
            let e = rel_error_real(&widen_real(&y), &dst_ref(&xw, kind));
            assert!(
                e <= bound::<T>(n, 10.0),
                "{kind:?} n={n} {}: {e:e}",
                core::any::type_name::<T>()
            );
        }
    }
}

#[test]
fn the_eight_types_against_their_definitions() {
    let mut rng = Rng(1001);
    check::<f64>(&mut rng);
    check::<f32>(&mut rng);
}

#[test]
fn inverse_pairs() {
    let mut rng = Rng(5);
    for n in [2usize, 3, 4, 5, 8, 16, 17, 64, 100] {
        let x = rng.reals::<f64>(n);
        let scaled = |f: f64| x.iter().map(|v| v * f).collect::<Vec<f64>>();
        let run_dct = |k: DctKind, v: &mut Vec<f64>| Dct::<f64>::new(n, k).transform(v);
        let run_dst = |k: DstKind, v: &mut Vec<f64>| Dst::<f64>::new(n, k).transform(v);
        let tol = bound::<f64>(n, 20.0);
        let mut y = x.clone();
        run_dct(DctKind::I, &mut y);
        run_dct(DctKind::I, &mut y);
        assert!(
            rel_error_real(&y, &scaled(2.0 * (n as f64 - 1.0))) <= tol,
            "DCT-I n={n}"
        );
        let mut y = x.clone();
        run_dct(DctKind::II, &mut y);
        run_dct(DctKind::III, &mut y);
        assert!(
            rel_error_real(&y, &scaled(2.0 * n as f64)) <= tol,
            "DCT-II/III n={n}"
        );
        let mut y = x.clone();
        run_dct(DctKind::IV, &mut y);
        run_dct(DctKind::IV, &mut y);
        assert!(
            rel_error_real(&y, &scaled(2.0 * n as f64)) <= tol,
            "DCT-IV n={n}"
        );
        let mut y = x.clone();
        run_dst(DstKind::I, &mut y);
        run_dst(DstKind::I, &mut y);
        assert!(
            rel_error_real(&y, &scaled(2.0 * (n as f64 + 1.0))) <= tol,
            "DST-I n={n}"
        );
        let mut y = x.clone();
        run_dst(DstKind::II, &mut y);
        run_dst(DstKind::III, &mut y);
        assert!(
            rel_error_real(&y, &scaled(2.0 * n as f64)) <= tol,
            "DST-II/III n={n}"
        );
        let mut y = x.clone();
        run_dst(DstKind::IV, &mut y);
        run_dst(DstKind::IV, &mut y);
        assert!(
            rel_error_real(&y, &scaled(2.0 * n as f64)) <= tol,
            "DST-IV n={n}"
        );
    }
}

#[test]
fn batches_and_empty() {
    let mut rng = Rng(8);
    let n = 12;
    let x = rng.reals::<f64>(3 * n);
    for kind in DCTS {
        let plan = Dct::<f64>::new(n, kind);
        let mut all = x.clone();
        plan.transform(&mut all);
        for i in 0..3 {
            let mut one = x[i * n..(i + 1) * n].to_vec();
            plan.transform(&mut one);
            assert!(rel_error_real(&all[i * n..(i + 1) * n], &one) <= bound::<f64>(n, 4.0));
        }
    }
    let mut empty: Vec<f64> = Vec::new();
    Dct::<f64>::new(0, DctKind::II).transform(&mut empty);
    Dst::<f64>::new(0, DstKind::I).transform(&mut empty);
    Dct::<f64>::new(0, DctKind::I).transform(&mut empty);
}

#[test]
#[should_panic(expected = "DCT-I")]
fn dct_one_of_length_one_panics() {
    let _ = Dct::<f64>::new(1, DctKind::I);
}

#[test]
fn in_place_and_out_of_place_agree() {
    let mut rng = Rng(9);
    let lengths: &[usize] = if cfg!(miri) {
        &[1, 2, 3, 17, 18]
    } else {
        &[1, 2, 3, 16, 17, 18, 31, 33, 34, 64, 65, 100, 129, 255]
    };
    for &n in lengths {
        let x = rng.reals::<f64>(2 * n);
        for kind in DCTS {
            if kind == DctKind::I && n < 2 {
                continue;
            }
            let plan = Dct::<f64>::new(n, kind);
            let mut a = x.clone();
            plan.transform(&mut a);
            let mut b = vec![0.0; 2 * n];
            plan.transform_to(&x, &mut b);
            assert_eq!(a, b, "{kind:?} n={n}");
        }
        for kind in DSTS {
            let plan = Dst::<f64>::new(n, kind);
            let mut a = x.clone();
            plan.transform(&mut a);
            let mut b = vec![0.0; 2 * n];
            plan.transform_to(&x, &mut b);
            assert_eq!(a, b, "{kind:?} n={n}");
        }
    }
}

#[test]
fn long_lengths_round_trip() {
    let lengths: &[usize] = if cfg!(miri) {
        &[33, 34]
    } else {
        &[
            257, 512, 513, 1000, 1023, 1025, 2049, 4096, 4097, 6000, 16385,
        ]
    };
    let mut rng = Rng(10);
    for &n in lengths {
        let x = rng.reals::<f64>(n);
        let scaled = |f: f64| x.iter().map(|v| v * f).collect::<Vec<f64>>();
        let tol = bound::<f64>(n, 20.0);
        let pairs: [(DctKind, DctKind, f64); 3] = [
            (DctKind::I, DctKind::I, 2.0 * (n as f64 - 1.0)),
            (DctKind::II, DctKind::III, 2.0 * n as f64),
            (DctKind::IV, DctKind::IV, 2.0 * n as f64),
        ];
        for (a, b, f) in pairs {
            let mut y = x.clone();
            Dct::<f64>::new(n, a).transform(&mut y);
            Dct::<f64>::new(n, b).transform(&mut y);
            let e = rel_error_real(&y, &scaled(f));
            assert!(e <= tol, "DCT {a:?}/{b:?} n={n}: {e:e}");
        }
        let pairs: [(DstKind, DstKind, f64); 3] = [
            (DstKind::I, DstKind::I, 2.0 * (n as f64 + 1.0)),
            (DstKind::II, DstKind::III, 2.0 * n as f64),
            (DstKind::IV, DstKind::IV, 2.0 * n as f64),
        ];
        for (a, b, f) in pairs {
            let mut y = x.clone();
            Dst::<f64>::new(n, a).transform(&mut y);
            Dst::<f64>::new(n, b).transform(&mut y);
            let e = rel_error_real(&y, &scaled(f));
            assert!(e <= tol, "DST {a:?}/{b:?} n={n}: {e:e}");
        }
    }
}

#[test]
fn impulses_at_long_lengths() {
    // The transform of e_j is column j of the matrix, known in double-double.
    let lengths: &[usize] = if cfg!(miri) {
        &[33]
    } else {
        &[257, 1000, 1025, 4096, 4097]
    };
    for &n in lengths {
        for j in [0, 1, n / 3, n - 1] {
            let mut x = vec![0.0f64; n];
            x[j] = 1.0;
            let mut y = x.clone();
            Dct::<f64>::new(n, DctKind::I).transform(&mut y);
            let c = if j == 0 || j == n - 1 { 1.0 } else { 2.0 };
            let want: Vec<f64> = (0..n)
                .map(|k| c * trig((j * k) as u64, n as u64 - 1).0.value())
                .collect();
            let e = rel_error_real(&y, &want);
            assert!(e <= bound::<f64>(n, 10.0), "DCT-I n={n} j={j}: {e:e}");
            let mut y = x.clone();
            Dst::<f64>::new(n, DstKind::I).transform(&mut y);
            let want: Vec<f64> = (0..n)
                .map(|k| 2.0 * trig(((j + 1) * (k + 1)) as u64, n as u64 + 1).1.value())
                .collect();
            let e = rel_error_real(&y, &want);
            assert!(e <= bound::<f64>(n, 10.0), "DST-I n={n} j={j}: {e:e}");
            let mut y = x.clone();
            Dct::<f64>::new(n, DctKind::IV).transform(&mut y);
            let want: Vec<f64> = (0..n)
                .map(|k| {
                    2.0 * trig(((2 * j + 1) * (2 * k + 1)) as u64, 4 * n as u64)
                        .0
                        .value()
                })
                .collect();
            let e = rel_error_real(&y, &want);
            assert!(e <= bound::<f64>(n, 10.0), "DCT-IV n={n} j={j}: {e:e}");
        }
    }
}

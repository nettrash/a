//! Two- and three-dimensional transforms against separable double-double DFTs.

use super::common::*;
use a::math::complex::Complex;
use a::math::fft::{Fft2, Fft3};

/// The DFT along one axis of a row-major array of the shape `dims`.
fn dft_axis(x: &[Complex<f64>], dims: &[usize], axis: usize, sign: f64) -> Vec<Complex<f64>> {
    let n = dims[axis];
    let stride: usize = dims[axis + 1..].iter().product();
    let outer: usize = dims[..axis].iter().product();
    let mut y = x.to_vec();
    for o in 0..outer {
        for i in 0..stride {
            let base = o * n * stride + i;
            let line: Vec<_> = (0..n).map(|j| x[base + j * stride]).collect();
            let t = dft(&line, sign);
            for (j, v) in t.into_iter().enumerate() {
                y[base + j * stride] = v;
            }
        }
    }
    y
}

fn reference(x: &[Complex<f64>], dims: &[usize], sign: f64) -> Vec<Complex<f64>> {
    let mut y = x.to_vec();
    for axis in 0..dims.len() {
        y = dft_axis(&y, dims, axis, sign);
    }
    y
}

#[test]
fn two_dimensions() {
    let mut rng = Rng(21);
    let shapes: &[(usize, usize)] = if cfg!(miri) {
        &[(2, 3), (4, 4)]
    } else {
        &[
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 3),
            (4, 4),
            (5, 8),
            (16, 9),
            (17, 31),
            (32, 64),
            (100, 37),
        ]
    };
    for &(r, c) in shapes {
        let fft = Fft2::<f64>::new(r, c);
        assert_eq!(fft.shape(), (r, c));
        let x = rng.complexes::<f64>(r * c);
        let mut y = x.clone();
        fft.forward(&mut y);
        let e = rel_error(&y, &reference(&x, &[r, c], -1.0));
        assert!(e <= bound::<f64>(r * c, 10.0), "2D {r}x{c}: {e:e}");
        fft.inverse(&mut y);
        let want: Vec<_> = x.iter().map(|v| *v * (r * c) as f64).collect();
        assert!(
            rel_error(&y, &want) <= bound::<f64>(r * c, 20.0),
            "2D round trip {r}x{c}"
        );
        // f32, and a batch of two.
        let fft = Fft2::<f32>::new(r, c);
        let x32 = rng.complexes::<f32>(2 * r * c);
        let mut y32 = x32.clone();
        fft.forward(&mut y32);
        for i in 0..2 {
            let part = &x32[i * r * c..(i + 1) * r * c];
            let e = rel_error(
                &widen(&y32[i * r * c..(i + 1) * r * c]),
                &reference(&widen(part), &[r, c], -1.0),
            );
            assert!(
                e <= bound::<f32>(r * c, 10.0),
                "2D f32 {r}x{c} array {i}: {e:e}"
            );
        }
    }
}

#[test]
fn three_dimensions() {
    let mut rng = Rng(31);
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(2, 2, 3)]
    } else {
        &[
            (1, 1, 1),
            (2, 2, 2),
            (2, 3, 5),
            (4, 1, 6),
            (8, 8, 8),
            (5, 7, 9),
            (16, 4, 13),
        ]
    };
    for &(a, b, c) in shapes {
        let fft = Fft3::<f64>::new(a, b, c);
        assert_eq!(fft.shape(), (a, b, c));
        let x = rng.complexes::<f64>(a * b * c);
        let mut y = x.clone();
        fft.forward(&mut y);
        let e = rel_error(&y, &reference(&x, &[a, b, c], -1.0));
        assert!(e <= bound::<f64>(a * b * c, 10.0), "3D {a}x{b}x{c}: {e:e}");
        let mut z = x.clone();
        fft.inverse(&mut z);
        let e = rel_error(&z, &reference(&x, &[a, b, c], 1.0));
        assert!(
            e <= bound::<f64>(a * b * c, 10.0),
            "3D inverse {a}x{b}x{c}: {e:e}"
        );
    }
}

#[test]
fn empty_shapes() {
    let fft = Fft2::<f64>::new(0, 5);
    let mut x: Vec<Complex<f64>> = Vec::new();
    fft.forward(&mut x);
    let fft = Fft3::<f64>::new(3, 0, 2);
    fft.inverse(&mut x);
}

/// The transform of a row-major array, one axis at a time by the one-dimensional plan on
/// gathered lines.
fn by_lines(x: &[Complex<f64>], dims: &[usize], inverse: bool) -> Vec<Complex<f64>> {
    use a::math::fft::Fft;
    let mut y = x.to_vec();
    for axis in 0..dims.len() {
        let n = dims[axis];
        let stride: usize = dims[axis + 1..].iter().product();
        let outer: usize = dims[..axis].iter().product();
        let fft = Fft::<f64>::new(n);
        for o in 0..outer {
            for i in 0..stride {
                let base = o * n * stride + i;
                let mut line: Vec<_> = (0..n).map(|j| y[base + j * stride]).collect();
                if inverse {
                    fft.inverse(&mut line);
                } else {
                    fft.forward(&mut line);
                }
                for (j, v) in line.into_iter().enumerate() {
                    y[base + j * stride] = v;
                }
            }
        }
    }
    y
}

#[test]
fn tiles_of_columns_against_lines() {
    let mut rng = Rng(41);
    let shapes2: &[(usize, usize)] = if cfg!(miri) {
        &[(3, 19), (18, 2)]
    } else {
        &[
            (1, 1000),
            (1000, 1),
            (37, 1000),
            (513, 40),
            (2, 4096),
            (4096, 3),
            (300, 17),
            // Above 4 MiB: the columns by tiles rather than by the passes along the rows.
            (700, 400),
        ]
    };
    for &(r, c) in shapes2 {
        let x = rng.complexes::<f64>(2 * r * c);
        let fft = Fft2::<f64>::new(r, c);
        let mut y = x.clone();
        fft.forward(&mut y);
        let mut z = x.clone();
        fft.inverse(&mut z);
        for i in 0..2 {
            let part = &x[i * r * c..(i + 1) * r * c];
            let s = i * r * c..(i + 1) * r * c;
            let e = rel_error(&y[s.clone()], &by_lines(part, &[r, c], false));
            assert!(e <= bound::<f64>(r * c, 4.0), "2D {r}x{c}: {e:e}");
            let e = rel_error(&z[s], &by_lines(part, &[r, c], true));
            assert!(e <= bound::<f64>(r * c, 4.0), "2D inverse {r}x{c}: {e:e}");
        }
    }
    let shapes3: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(3, 2, 17)]
    } else {
        &[
            (2, 2, 3000),
            (300, 2, 7),
            (5, 300, 3),
            (64, 64, 64),
            (1, 37, 1),
            (40, 1, 33),
            // The first axis above 4 MiB by tiles, the second within its planes by passes.
            (72, 72, 72),
        ]
    };
    for &(a, b, c) in shapes3 {
        let x = rng.complexes::<f64>(a * b * c);
        let fft = Fft3::<f64>::new(a, b, c);
        let mut y = x.clone();
        fft.forward(&mut y);
        let e = rel_error(&y, &by_lines(&x, &[a, b, c], false));
        assert!(e <= bound::<f64>(a * b * c, 4.0), "3D {a}x{b}x{c}: {e:e}");
        fft.inverse(&mut y);
        let want: Vec<_> = x.iter().map(|v| *v * (a * b * c) as f64).collect();
        assert!(
            rel_error(&y, &want) <= bound::<f64>(a * b * c, 20.0),
            "3D round trip {a}x{b}x{c}"
        );
    }
}

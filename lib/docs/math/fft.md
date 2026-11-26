[A](../../../README.md) › [Math](../../../README.md#math)

# Fourier transforms

`a::math::fft` computes discrete Fourier transforms of `f32`, `f64`, `Complex<f32>` and `Complex<f64>` sequences of any length, the cosine and sine transforms of types I to IV, transforms of two and three dimensions, and convolutions and correlations. A transform is planned once for its length and applied as often as needed; a plan never changes and may be shared between threads. The float type of a plan is `f32` or `f64` (the sealed trait `FftFloat`), and the values are the crate's [`Complex<T>`](complex.md).

- `Fft::<T>::new(n)` plans the complex transform `X[k] = Σ x[j]·e^(−2πi·jk/n)`, forward, and `x[j] = Σ X[k]·e^(2πi·jk/n)`, inverse, for any `n` (0 and 1 give the identity). `forward` and `inverse` work in place, `forward_to` and `inverse_to` from one slice into another of the same length; `forward_with_scratch` and `inverse_with_scratch` work in place with the working memory from the caller, at least `scratch_len()` values, and never allocate. `len` and `is_empty` give the length, and `Debug` the plan the length got: `Fft { n: 1009, algorithm: "Rader(4·4·3·3·7)", isa: Neon, .. }`.
- `RealFft::<T>::new(n)` plans the transform of `n` reals: `forward(input, output)` takes runs of `n` reals and gives for each the `n/2 + 1` values `X[0..=n/2]` that determine the rest (`X[n − k]` is the conjugate of `X[k]`), as FFTW's `r2c` and NumPy's `rfft` do; `inverse(input, output)` takes runs of `n/2 + 1` values and gives `n` reals. `spectrum_len()` is `n/2 + 1` (0 for `n = 0`); `scratch_len`, `forward_with_scratch` and `inverse_with_scratch` are as for `Fft`.
- `Dct::<T>::new(n, kind)` and `Dst::<T>::new(n, kind)` plan the cosine and sine transforms of the types `DctKind::{I, II, III, IV}` and `DstKind::{I, II, III, IV}`, FFTW's `REDFT00`, `REDFT10`, `REDFT01`, `REDFT11` and `RODFT00`, `RODFT10`, `RODFT01`, `RODFT11`, unscaled as FFTW's. `transform` works in place, `transform_to` from one slice into another, and `transform_with_scratch` in place with at least `scratch_len()` complex values of working memory from the caller, without allocating; `len`, `is_empty` and `kind` describe the plan.
- `Fft2::<T>::new(rows, cols)` and `Fft3::<T>::new(n0, n1, n2)` plan the transforms of complex arrays of two and three dimensions stored with the last index varying fastest; `forward` and `inverse` work in place, `shape` gives the shape.
- `convolve(&a, &b)`, `correlate(&a, &b)`, `circular_convolve(&a, &b)` and `circular_correlate(&a, &b)` take two slices of the same type, `f32`, `f64`, `Complex<f32>` or `Complex<f64>` (the sealed trait `Sample`), and return a new vector; `convolve_into(&a, &b, &mut out)` and `correlate_into(&a, &b, &mut out)` write the `len(a) + len(b) − 1` values into a slice of that length instead of a new vector (and panic for another length). They choose between a direct sum and transforms by the lengths, and keep the plans of their transforms for the next calls.
- The sections below give the plans and their working memory, each kind of transform and the convolutions, then the examples; the methods are after the examples, then how the module was tested and the measurements against FFTW, rustfft, realfft and rustdct, and Apple's Accelerate.

## Plans, batches and working memory

- A plan is immutable, `Send` and `Sync`. `Fft`, `RealFft`, `Dct` and `Dst` are `Clone`: a clone shares the plan's tables, and costs one reference count. `Fft2` and `Fft3` are not `Clone`; they hold one `Fft` per dimension and, for each dimension but the last, the plan of its transforms across the rows: the Stockham passes of its length planned for interleaved sequences (for an array of at most 4 MiB whose length along that dimension has no prime factor above 19), or else a clone of its `Fft` and the width of its tiles ([below](#two-and-three-dimensions-1)).
- Every method takes a batch: any number of sequences (or arrays) of the planned length one after the other in one slice, each transformed in turn, and panics when the length of the slice is not a multiple of the planned one, or when an output's length differs from its input's (for `RealFft`, when the output does not hold `n/2 + 1` values for each sequence of the input, or `n` for each of `n/2 + 1`). Short sequences of a batch go through the passes of the complex transform together, as many as fit in 256 values a call, when the working memory has room for them; each result is that of its sequence transformed alone, to within rounding.
- Every inverse is unscaled, as FFTW's: `Fft`'s `inverse(forward(x)) = n·x`, `RealFft`'s the same, `Fft2`'s `rows·cols·x` and `Fft3`'s `n0·n1·n2·x`; for the cosine and sine transforms see [below](#cosine-and-sine-transforms).
- `scratch_len()` is the working memory one sequence needs, in complex values. It is 0 for the lengths of one butterfly of the complex transform (2, 3, 4, 5, 7, 8, 11, 13, 17 and 19, and the composites 6, 9, 10, 12, 15, 16, 20, 24 and 25); for the real transforms of 2 (summed directly), of the odd lengths 3 to 15 (the complex butterfly on the reals) and of the even lengths whose half is one butterfly (every even length from 4 to 26, then 30, 32, 34, 38, 40, 48 and 50: the whole transform in one kernel); and for the cosine and sine transforms summed directly (up to 8 values, 16 for type I). A `_with_scratch` method panics with "scratch of … values, … needed" when given less, and never allocates; given more, `Fft::inverse_with_scratch`, `Fft::forward_with_scratch` and `RealFft::inverse_with_scratch` let the short sequences of a batch run together. The other methods take their working memory themselves, on the stack up to 512 complex values and from the heap beyond, once for the whole batch: `Fft`'s `forward`, `inverse`, `forward_to` and `inverse_to` (one run's, or the room a batch of short runs takes together; one run of a length of one butterfly calls its kernel directly and takes none), and `RealFft`'s, `Dct`'s and `Dst`'s methods; `RealFft::forward` of a batch of even lengths stages its complex transforms in a buffer of up to 4096 values (one sequence's, when that is longer) from the heap. `Fft2` and `Fft3` have no `_with_scratch` form: each call takes one buffer, on the stack up to 512 values and from the heap beyond, for the scratch of the transforms of the rows and, for each other dimension, the second buffer of its passes along the rows (the size of one array) or a tile of at most 1 MiB (or one column, when a column is longer) with the scratch of its transforms.
- `forward_to`, `inverse_to` and `transform_to` have no `_with_scratch` form.

## The complex transform

- Every length is planned in time `O(n log n)`: by Stockham passes of the radices 2, 4 and 8 for the factors 2, hard-coded butterflies of 3, 5 and 7, and the odd butterfly from tables for the primes 11 to 19 (a length of 6, 9, 10, 12, 15, 16, 20, 24 or 25 is one butterfly of its own); a prime factor above 19 of a composite length by a plan of its own; and a prime length above 19 by Rader's algorithm or by Bluestein's with a smooth inner length, whichever the planner estimates faster. `Debug` shows the choice: `"8·8·4·4"` for 1024, `"8·8·5·5·5·5·5·5"` for 10⁶, `"Rader(4·4·3·3·7)"` for 1009, `"Bluestein(20480: 8·8·8·8·5)"` for 10007, `"2·[Rader(4·5·5)]"` for 202.
- The kernels are NEON on little-endian aarch64, SSE2 or, detected when the plan is made, AVX2 with FMA on x86_64, and portable Rust elsewhere and under Miri; the same algorithm in each, so a plan's results differ between instruction sets only by rounding.
- On an Apple M4 Max, one thread, a forward transform of `Complex<f64>` takes 1.26 µs at 1024 and 4.02 ms at 2²⁰, 0.592 and 0.494 of FFTW's time; of `Complex<f32>` 617 ns and 1.88 ms, 0.287 and 0.277 of FFTW's time and 2.22 and 0.987 times Accelerate's. The tables are [below](#measured).

## Real sequences

- `forward` gives `X[0]` with an imaginary part of exactly 0, and `X[n/2]` too for an even `n`; `inverse` ignores the imaginary parts of `X[0]` and, for an even `n`, of `X[n/2]`, as FFTW's `c2r` does.
- An even length takes the complex transform of `n/2` (a half of one butterfly the whole transform in one kernel); an odd one with a factor among 19, 15, 9, 17, 13, 11, 7, 5 and 3 splits by the first of them in this order that divides it; a prime above 15, or a length with no such factor, takes the complex transform of `n`; the odd lengths 3 to 15 take the complex butterfly of `n` on the reals, and `n = 2` is summed directly. Batches are as for `Fft`.

## Cosine and sine transforms

The eight transforms, of a sequence `X[0..n]` into `Y[0..n]`, unscaled as FFTW defines them:

| Kind | FFTW | `Y[k]` | Inverse, up to the factor |
| --- | --- | --- | --- |
| `DctKind::I` (`n ≥ 2`) | `REDFT00` | `X[0] + (−1)^k·X[n−1] + 2·Σ_{j=1}^{n−2} X[j]·cos(π·jk/(n−1))` | itself, `2(n − 1)` |
| `DctKind::II` | `REDFT10` | `2·Σ X[j]·cos(π·(j + 1/2)·k/n)` | DCT-III, `2n` |
| `DctKind::III` | `REDFT01` | `X[0] + 2·Σ_{j=1}^{n−1} X[j]·cos(π·j·(k + 1/2)/n)` | DCT-II, `2n` |
| `DctKind::IV` | `REDFT11` | `2·Σ X[j]·cos(π·(j + 1/2)·(k + 1/2)/n)` | itself, `2n` |
| `DstKind::I` | `RODFT00` | `2·Σ X[j]·sin(π·(j + 1)·(k + 1)/(n + 1))` | itself, `2(n + 1)` |
| `DstKind::II` | `RODFT10` | `2·Σ X[j]·sin(π·(j + 1/2)·(k + 1)/n)` | DST-III, `2n` |
| `DstKind::III` | `RODFT01` | `(−1)^k·X[n−1] + 2·Σ_{j=0}^{n−2} X[j]·sin(π·(j + 1)·(k + 1/2)/n)` | DST-II, `2n` |
| `DstKind::IV` | `RODFT11` | `2·Σ X[j]·sin(π·(j + 1/2)·(k + 1/2)/n)` | itself, `2n` |

`Dct::new(1, DctKind::I)` panics ("DCT-I needs a length of at least 2"); every other kind takes any length. All of them take `O(n log n)` time by FFTW's algorithms on the real transform, and the lengths up to 8 (16 for type I) are summed directly from the matrix.

## Two and three dimensions

`Fft2` computes `X[k, l] = Σ x[i, j]·e^(−2πi·(ik/rows + jl/cols))` for arrays of `rows × cols` values stored row after row, and `Fft3` the same over three indices: the last, contiguous dimension as one batch of rows, then each other dimension, for an array of at most 4 MiB whose length along it has no prime factor above 19, by the passes of its transform run along the rows (the columns as interleaved sequences, the vectors across them), else by tiles of adjacent columns gathered into a buffer, so that every pass over the array reads and writes whole cache lines.

## Convolution and correlation

- `convolve(&a, &b)` is the linear convolution, the `len(a) + len(b) − 1` values `c[k] = Σ a[i]·b[k − i]` (none when either is empty), as NumPy's `convolve` gives them in its full mode.
- `correlate(&a, &b)` is the cross-correlation, the `len(a) + len(b) − 1` values `c[k + len(b) − 1] = Σ a[i + k]·conj(b[i])` for the lags `k` from `−(len(b) − 1)` to `len(a) − 1`, NumPy's `correlate` in its full mode: the convolution with `b` reversed and conjugated, without a copy of `b`.
- `circular_convolve(&a, &b)` and `circular_correlate(&a, &b)` take two sequences of the same length `n` (and panic when the lengths differ): `c[k] = Σ a[j]·b[(k − j) mod n]` and `c[k] = Σ a[(j + k) mod n]·conj(b[j])`.
- The path is chosen by a model of the costs of the lengths: a direct sum by SIMD kernels while the shorter operand is short, the product of the transforms of the operands padded to a length `2^a·3^b·5^c·7^d`, or overlap-add when one operand is much longer than the other. With the constants of this release the direct path is taken against a long operand (10⁴ to 10⁶ values) up to a shorter one of 112 to 125 values for `f32`, 91 to 99 for `f64`, 43 to 46 for `Complex<f32>` and 38 to 40 for `Complex<f64>`, and against 1,000 values up to 172, 130, 54 and 47 (the model on the constants of `conv.rs`; measured [below](#convolution)). A circular convolution is direct, by three transforms of the length `n`, or the linear convolution by transforms of a smooth length of at least `2n − 1` folded modulo `n` (for a prime `n`, whose own transforms would take Bluestein's algorithm).
- **The plans are kept.** The transforms' plans are kept for the next calls of the same lengths, in one cache shared by all the threads: at most 16 plans and at most 16 MiB of them by an estimate of twice `L` complex values for a length with no prime factor above 13 (measured: 1.5 times for a real plan, twice for a complex one) and twelve times for the others (Bluestein's and Rader's plans, measured at up to about ten). The least recently used plan goes first; a plan larger than 16 MiB alone is made, used and dropped and evicts nothing. The largest smooth plans kept are of 2²⁰ values for `f32` and `Complex<f32>` and of 2¹⁹ for `f64` and `Complex<f64>`. A repeated call skips the planning and gives the same bits as the first: the choice of the path does not look at the cache. The memory stays held, up to the 16 MiB, until other plans push it out; there is no call to empty the cache. The lock is held only to look a plan up, to insert one and to unlink those evicted; plans are made, run and dropped outside it.
- The direct path's sums are in the order of the taps, in the type; the transforms' error is a few units of rounding of the type times `‖a‖·‖b‖·log₂ len` (the tests check `10·ε·(log₂ len + 1)` relative to `‖a‖·‖b‖`). With infinities or NaNs among the values the result may hold NaN where the exact sums would be infinite: the direct path reads the operands between zeros, and every value of a transform mixes all the inputs.

## Examples

```rust
use a::math::complex::Complex;
use a::math::fft::{
    Dct, DctKind, Dst, DstKind, Fft, Fft2, Fft3, RealFft, circular_convolve, circular_correlate,
    convolve, correlate,
};

let close = |a: &[Complex<f64>], b: &[Complex<f64>], tol: f64| {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (*x - *y).abs() < tol)
};

// The complex transform of 5 values and back: the inverse is unscaled.
let fft = Fft::<f64>::new(5);
let x: Vec<Complex<f64>> = (0..5).map(|j| Complex::new(j as f64, 1.0)).collect();
let mut y = x.clone();
fft.forward(&mut y);
assert!((y[0] - Complex::new(10.0, 5.0)).abs() < 1e-12);
fft.inverse(&mut y);
let x5: Vec<Complex<f64>> = x.iter().map(|&v| v * 5.0).collect();
assert!(close(&y, &x5, 1e-12));

// Out of place, into a slice of the same length.
let mut z = vec![Complex::ZERO; 5];
fft.forward_to(&x, &mut z);
let mut w = x.clone();
fft.forward(&mut w);
assert!(close(&z, &w, 1e-12));

// A batch of three sequences of 8, the working memory from the caller.
let fft8 = Fft::<f32>::new(8);
let mut scratch = vec![Complex::ZERO; fft8.scratch_len()];
let mut data = vec![Complex::ONE; 24];
fft8.forward_with_scratch(&mut data, &mut scratch);
for run in data.chunks(8) {
    assert!((run[0] - Complex::new(8.0, 0.0)).abs() < 1e-6);
    assert!(run[1..].iter().all(|v| v.abs() < 1e-6));
}

// A prime length by Rader's algorithm; a clone shares the plan with another thread.
let plan = Fft::<f64>::new(1009);
assert!(format!("{plan:?}").contains("Rader"));
let copy = plan.clone();
let sum = std::thread::spawn(move || {
    let mut x = vec![Complex::ONE; 1009];
    copy.forward(&mut x);
    x[0]
})
.join()
.unwrap();
assert!((sum - Complex::new(1009.0, 0.0)).abs() < 1e-9);

// Real sequences: the n/2 + 1 values of the spectrum, those of the complex transform.
let real = RealFft::<f64>::new(6);
assert_eq!(real.spectrum_len(), 4);
let r = [1.0, 2.0, 0.0, -1.0, 0.5, 3.0];
let mut spectrum = vec![Complex::ZERO; 4];
real.forward(&r, &mut spectrum);
let mut full: Vec<Complex<f64>> = r.iter().map(|&v| Complex::new(v, 0.0)).collect();
Fft::<f64>::new(6).forward(&mut full);
assert!(close(&spectrum, &full[..4], 1e-12));
assert_eq!((spectrum[0].im, spectrum[3].im), (0.0, 0.0));
let mut back = [0.0; 6];
real.inverse(&spectrum, &mut back);
assert!(back.iter().zip(&r).all(|(b, x)| (b - 6.0 * x).abs() < 1e-12));

// Two sequences in one call give each its own spectrum.
let mut two = vec![Complex::ZERO; 8];
real.forward(&[r, r].concat(), &mut two);
assert_eq!(two[..4], two[4..]);

// DCT-II and DCT-III are each other's inverses up to 2n.
let n = 16;
let s: Vec<f64> = (0..n).map(|j| (j as f64).sin()).collect();
let mut t = vec![0.0; n];
Dct::<f64>::new(n, DctKind::II).transform_to(&s, &mut t);
Dct::<f64>::new(n, DctKind::III).transform(&mut t);
assert!(t.iter().zip(&s).all(|(t, s)| (t - 2.0 * n as f64 * s).abs() < 1e-11));

// DST-I is its own inverse up to 2(n + 1), DCT-I up to 2(n − 1).
let dst = Dst::<f64>::new(7, DstKind::I);
let mut u = s[..7].to_vec();
dst.transform(&mut u);
dst.transform(&mut u);
assert!(u.iter().zip(&s).all(|(u, s)| (u - 16.0 * s).abs() < 1e-11));
let dct1 = Dct::<f32>::new(9, DctKind::I);
let mut scratch = vec![Complex::ZERO; dct1.scratch_len()];
let mut v: Vec<f32> = (0..9).map(|j| j as f32).collect();
dct1.transform_with_scratch(&mut v, &mut scratch);
dct1.transform_with_scratch(&mut v, &mut scratch);
assert!(v.iter().enumerate().all(|(j, v)| (v - 16.0 * j as f32).abs() < 1e-3));

// Two dimensions: X[0, 0] is the sum, X[1, 0] the first row's sum less the second's.
let fft2 = Fft2::<f64>::new(2, 3);
let a2: Vec<Complex<f64>> = (0..6).map(|v| Complex::new(v as f64, 0.0)).collect();
let mut b2 = a2.clone();
fft2.forward(&mut b2);
assert!((b2[0] - Complex::new(15.0, 0.0)).abs() < 1e-12);
assert!((b2[3] - Complex::new(-9.0, 0.0)).abs() < 1e-12);
fft2.inverse(&mut b2);
let a6: Vec<Complex<f64>> = a2.iter().map(|&v| v * 6.0).collect();
assert!(close(&b2, &a6, 1e-12));
assert_eq!(Fft3::<f32>::new(2, 3, 4).shape(), (2, 3, 4));

// Convolutions and correlations, as NumPy's full modes.
assert_eq!(convolve(&[1.0, 2.0, 3.0], &[1.0, 1.0]), vec![1.0, 3.0, 5.0, 3.0]);
assert_eq!(correlate(&[1.0, 2.0, 3.0], &[1.0, 0.0]), vec![0.0, 1.0, 2.0, 3.0]);
let i = Complex::new(0.0, 1.0);
assert_eq!(correlate(&[Complex::ONE], &[i]), vec![Complex::new(0.0, -1.0)]);
assert_eq!(circular_convolve(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0]), vec![3.0, 1.0, 2.0]);
assert_eq!(circular_correlate(&[1.0, 2.0, 3.0], &[0.0, 1.0, 0.0]), vec![2.0, 3.0, 1.0]);

// A long convolution goes through transforms; its plans are kept, so the same call again
// skips the planning and gives the same bits.
let long: Vec<f64> = (0..10_000).map(|j| ((j * 7919) % 101) as f64 / 101.0).collect();
let taps: Vec<f64> = (0..1000).map(|j| 1.0 / (1.0 + j as f64)).collect();
let c = convolve(&long, &taps);
assert_eq!(c.len(), 10_999);
let direct: f64 = (0..1000).map(|t| taps[t] * long[5000 - t]).sum();
assert!((c[5000] - direct).abs() < 1e-9);
assert_eq!(convolve(&long, &taps), c);
```

## The method

### The complex transform

**Stockham passes.** Every length is a sequence of passes, one per factor `r` of `n`, in the Stockham autosort form of decimation in frequency: a pass of radix `r` over sub-transforms of the length `L = r·m` of `s` interleaved sequences reads `x[q + s·(p + k·m)]` and writes `y[q + s·(r·p + j)] = W_L^(j·p) · Σ_k x[q + s·(p + k·m)]·W_r^(j·k)`. After the passes of all the factors (`s` growing from 1, `m` shrinking to 1) the result is in natural order: no bit reversal and no transposition. The passes alternate between the data and one buffer of `n` values; the last pass, which has no twiddles and its own kernel without the loop over `p`, runs in place, so an in-place transform ends in the data and an out-of-place one writes its output without a copy. The factors 2 take radix-8 passes, with one or two radix-4 passes for what is left (a radix-2 pass for a single factor 2); the factors 3, 5 and 7 have hard-coded butterflies, and the primes 11 to 19 the general odd butterfly with its cosines and sines from a table (the sums `x_k ± x_(r−k)` halve its products), each instantiated for its radix so that its loops unroll into registers. The odd factors come after the twos and the largest last. A length that is one of the composites 6, 9, 10, 12, 15, 16, 20, 24 and 25 is one butterfly of its own, by decimation in time `n = R₁·R₂` with every value in registers (the `R₂` butterflies of `R₁`, the twiddles `W_n^(n₂·k₁)` by rotations for the quarter and eighth turns and from a table for the rest, then the `R₁` butterflies of `R₂`), and `Debug` shows it as one radix (`"16"`). Measured on the M4 and dropped: radix-16 passes in longer transforms (slower at every length: 16 streams at power-of-two strides, and more vectors than registers), and four-step and six-step algorithms with blocks of columns in the caches, about twice as slow as the plain passes at 2²⁰ and 2²² in `f64`.

**Twiddles.** Each pass has its own table of `W_L^(j·p)`, stored expanded for the products as `[wr, wr, −wi, wi]` (or by blocks of SIMD lanes for a first pass that runs its vectors along `p`): a product is one multiplication, one exchange of the parts and one fused multiply-add. The inverse runs the same tables with the conjugate products and `+i` for `−i` in the butterflies, by kernels instantiated for each direction: no conjugation passes. All the tables of a plan come from one table of the roots of its length (each pass's `W_L^e` is `W_n^(e·n/L)`) that holds only the angles of at most π/4: `n/8 + 1` entries for a multiple of 4, `n/4 + 1` for twice an odd number and `n/2 + 1` for an odd `n`. Every root is an entry with its signs and parts exchanged, exactly, the octant found by comparisons and the signs set by bit operations, without a division or a branch. A table of 512 entries or more (`TABLE_PRODUCTS`) comes from the real transforms' generator ([below](#real-transforms)): each entry the correctly rounded `f64` root but for ties within about 10⁻²⁸ of it. A shorter one is the system's `sin` and `cos` of the angle carried as a double-double and corrected by its low part, each entry as accurate as `sin` and `cos` of an `f64` (within about half a unit in the last place). Either way the factors that the symmetries of the circle relate are exact images of each other in `f64`. The `f32` tables of a length below 512 are rounded from these. From 512 on (`GRID_F32`) every `f32` twiddle of the passes is the product, in `f64`, of a coarse and a fine table of the real transforms' generator (the exponents `2^b` apart and below `2^b`, `b = ⌈log₂ n/2⌉`, about `√n` values each) rounded to `f32` (`Grid`): the product is within about 2⁻⁵² of the root, so the value is the nearest `f32` to the root but for ties closer than that, as the real transforms' `f32` tables are, and the symmetric factors are exact images of each other only as far as their nearest `f32` values are (no longer by construction). A replica of `Grid` in Python against mpmath found no value off the correctly rounded `f32` root, both parts, at every `k` for `n` = 1024, 1025, 3072 and 65,536 and at 150,000 sampled `k` for 1,024,001 and 2²⁰. The tables are written once, in order, without zeroing their memory first.

**Primes.** A prime length up to 19 is one pass. Above 19 it takes Rader's algorithm, the cyclic convolution of length `p − 1` of the input permuted by a primitive root with the transform of the permuted roots (computed once in `f64` and divided by `p − 1`; the inverse is the forward transform written to the reversed indices), or Bluestein's, the convolution with the chirp `W_2n^(k²)` (its angle exact from `k² mod 2n`) by transforms of a smooth length `m` in `[2n − 1, 2(2n − 1))` of the form `2^a·3^b·5^c·7^d`, the cheapest by the estimate rather than a power of two. The planner estimates the time of each from the passes each inner length needs, measured per radix on the M4: Rader's for 101 and 1009 (`p − 1 = 2²·5²` and `2⁴·3²·7`), Bluestein's for 10007, 100003 and 1000003, whose `p − 1` has a large prime factor. A prime factor above 19 of a composite length is a pass of its own, taken last; its sequences are adjacent there, so Rader's or Bluestein's algorithm runs on blocks of up to 16 of them at once (a divisor of their count), with the inner transform planned for that many interleaved sequences so that the vectors run across them, and the twiddles of the pass applied on the way out; one sequence at a time when no block divides the count. Bluestein's algorithm for the whole length is taken instead when it is estimated faster. Rader's and Bluestein's kernels are computed in `f64`, through the plan's own inner transform when the type is `f64`.

**SIMD.** The butterflies and passes are written once over a block of complex values and instantiated per instruction set: NEON (`f64`: one complex value a vector; `f32`: two, with a block of one for what is left), SSE2, and AVX2 with FMA (two `f64` and four `f32` values a vector, with SSE2 blocks for what is left), chosen at run time when the plan is made; the portable block is a `Complex<T>`. The vectors run along the contiguous index `q`; the first pass (`s = 1`) of `f32` (and of AVX2's `f64`) runs them along `p` instead, with the twiddles by lanes, and stores the results by rows. AVX2's `f32` first pass takes the 128-bit block. Where the target has FEAT_FCMA, as the Apple ones do, the NEON radix-4 and radix-8 butterflies add `±i·b` by one FCADD (inline assembly); the composite and odd butterflies keep the exchange of the parts and the fused multiply-add, since the assembly keeps the compiler from unrolling their loops, and a target without FCMA takes the exchange everywhere (the same values, bit for bit).

**Batches.** A length of one butterfly (2, 3, 4, 5, 7, 8, 11, 13, 17 and 19, and the composites 6, 9, 10, 12, 15, 16, 20, 24 and 25) runs in place with no working memory, its vector lanes across the runs of a batch; one run calls the butterfly's kernel directly, without the loop of a batch. Longer short transforms of a batch go through the passes together, up to 256 values a call, each kernel called once for the chunk, as far as the working memory allows: `scratch_len` is what one run needs, the allocating methods take room for the chunk when given more than one run, and a `_with_scratch` caller that passes more gets the same.

**Planning.** `Fft::new` of `f64` takes 2.40 µs at 1024, about two transforms' time, and 1.95 ms at 2²⁰, about half of one, and 13.5 µs, 183 µs and 1.98 ms for Rader's 1009 and Bluestein's 10,007 and 100,003 ([below](#the-complex-transform-2)); the cost is the table of roots (each entry's lookup, signs and expanded store), and for Rader's and Bluestein's lengths the transform of the kernel; an `f32` plan of 512 values or more takes its twiddles from the products of two short tables instead of the lookups (above). The estimate of the time of Bluestein's candidate lengths is computed from their exponents, without factoring them.

### Real transforms

**Even lengths** `n = 2m` pack the reals in pairs, `z[j] = x[2j] + i·x[2j + 1]`. A half of one butterfly (`m` from 2 to 13, 15, 16, 17, 19, 20, 24 and 25) runs the whole transform in one kernel, the butterfly of `m` and the pass over the pairs in registers, with no working memory. Otherwise one sequence takes the complex transform of `m` out of place straight into the output, its working memory not initialised (on the stack up to 512 values), and a batch takes its complex transforms as batches, up to 4096 values at a time, into a buffer the pass then reads. The pass separates the spectra of the even and odd halves over the pairs of bins `(k, m − k)`: with `a = Z[k]`, `b = conj Z[m − k]`, `u = (a + b)/2` and `t = w_k·(a − b)` for `w_k = −i·e^(−2πi·k/n)/2`, `X[k] = u + t` and `X[m − k] = conj(u − t)`; `X[0]` and `X[m]` come from `Z[0]`, and `X[m/2]` is `conj Z[m/2]`. For `f64`, when the last pass of the transform of `m` is a butterfly of 2, 3, 4, 5 or 8, one sequence (and `forward_with_scratch`, and the real transform inside the types II by Makhoul's step) runs that last pass and the pass over the pairs as one kernel: its butterflies `q` and `s − q` together, whose outputs are each other's partners (`Z[q + s·j]` and `Z[s − q + s·(r − 1 − j)]`), so that the spectrum is written from the registers; `f32` keeps the separate pass, whose vectors hold four pairs, which measured faster than the fused kernel's blocks of one value. The inverse runs the same pass with `w_k = i·e^(2πi·k/n)` for every sequence of a batch straight into the output, then the inverse complex transforms of all of them as one batch, in place, which leaves the reals there (one sequence with its working memory not initialised). The separate pass is vectorised: NEON (two `f64` or four `f32` pairs a step, the parts split into lanes by de-interleaving loads, the top bins reversed in the registers), SSE2 and AVX2 with FMA (two or four, and four or eight pairs, chosen once when the plan is made), and portable Rust elsewhere and under Miri, the same expressions lane by lane. `forward_with_scratch` runs the sequences one by one (each copied into its output and transformed there in place), since the staging of a batch needs memory its contract does not give.

**Odd lengths** have no packing into a half-length transform. One with a factor among 19, 15, 9, 17, 13, 11, 7, 5 and 3, `r` the first of them in this order that divides it (19 first, then the composites, then the other primes from the largest; measured), `n = r·m`, splits into the `r` sequences of the residues of the index modulo `r`. Their spectra come two at a time from one complex transform of `m` of `x_{2p} + i·x_{2p+1}`, all `(r − 1)/2` of them as one batch, and the last from the real transform of `m`, recursively; one pass of radix `r` then makes only the first half of the spectrum, `X[s + q·m] = Σ_a e^(−2πi·a·s/n)·Y_a[s]·e^(−2πi·a·q/r)` for `s ≤ (m − 1)/2`, writing the bins beyond `n/2` as the conjugates they mirror. That pass is a kernel per instruction set on blocks of one complex value, the complex passes' butterfly of `r` (written out for 3, 5 and 7, one butterfly in registers for the composites 9 and 15, the odd butterfly from a table for 11 to 19) after the twiddles `e^(−2πi·a·s/n)`; the inverse is its transpose. This costs about `1 + 1/r + …` complex transforms of `m`, near half the complex transform of `n` on reals. The odd lengths 3 to 15 take the complex butterfly of `n` on the reals as one kernel, with no working memory, and `n = 2` is summed directly from the table of `e^(−2πi·jk/n)` (the even lengths 4 to 16, summed directly before, take the fused kernels above); primes above 15 and the lengths with no factor up to 19 run the complex transform of `n` on the reals (a prime then takes Rader's or Bluestein's algorithm). Combining two real sequences of a batch into one complex one was not used: it loses accuracy for a sequence much smaller than its partner.

**The tables** of the real transforms' pass, of the odd splits and of the cosine and sine transforms of types II to IV are built as `e^(−2πi·(o + s·k)/n) = C[a]·F[j]` for `k = a·b + j`, `b` about `√len`: a fine table `F` of the powers of one root and a coarse table `C` of the powers of another, both and their products carried in double-double, the two roots summed by Taylor's series in double-double. Each value is then the correctly rounded root but for ties within about 10⁻²⁸ of it (the bench's `rootcheck`, against the tests' double-double roots up to `n = 2²³`, found no value off, where the system's `sin` and `cos` are off by up to one unit in the last place) and costs one product of double-doubles instead of a sine and a cosine; an `f32` table takes the product of the two roots rounded to `f64` and then to `f32`, the nearest `f32` but for ties closer than about 2⁻⁵². Tables of fewer than 128 roots take the roots one by one from the sines and cosines of the complex transform's short tables. The complex transform's octant tables of 512 entries or more come from this generator too (below that the sines and cosines one by one are faster), and its `f32` twiddles from 512 values on from the same products of a coarse and a fine table (`Grid`, [above](#the-complex-transform-1)): one generator of correctly rounded roots, with the sines and cosines for the short tables.

### Cosine and sine transforms

The algorithms are FFTW's accurate ones (those it enables by default), on the real transform:

- Types II and III by Makhoul's reordering (FFTW's `reodft010e-r2hc`): `v` holds the even-indexed values, then the odd-indexed ones backwards; with `V` its real transform and `c = e^(−πi·k/2n)·V[k]`, DCT-II is `y[k] = 2·Re c`, `y[n − k] = −2·Im c`. DCT-III is the transpose, `V[k] = e^(πi·k/2n)·(X[k] − i·X[n − k])` through the inverse real transform and the inverse reordering. For an even `n` the reordering writes straight into the packed pairs of the real transform.
- Type IV of an even `n` by the complex transform of `n/2` of `(x[2j] + i·x[n − 1 − 2j])·e^(−πi·j/n)`; with `W = 2·e^(−πi·(4p + 1)/4n)·Z[p]`, `y[2p] = Re W` and `y[n − 1 − 2p] = −Im W`, every step in place. Of an odd `n` by the real transform of `n` of a permutation of the values with signs, and sums and differences of the spectrum's parts times `√2` (Chan and Ho, FFTW's `reodft11e-r2hc-odd`).
- Type I of an odd `n` by FFTW's split-radix step (`reodft00e-splitradix`): with `N = n − 1`, `A` the DCT-I of the `N/2 + 1` even-indexed values and `B` the DCT-II of the `N/2` odd-indexed ones, `y[k] = A[k] + B[k]` and `y[N − k] = A[k] − B[k]`, recursively, which costs about one real transform of `n` where the symmetric extension would take one of `2(n − 1)`; DST-I alike, with the DST-I of the odd-indexed values and the DST-II of the even-indexed ones. An even `n` takes the real transform of the symmetric (antisymmetric) extension, `2(n − 1)` (`2(n + 1)`) values, as FFTW does.
- The sine transforms take the same steps with the signs and orders that turn them into the cosine transforms of their types, `DST-II(x)[k] = DCT-II((−1)^j·x[j])[n − 1 − k]`, `DST-III(X)[j] = (−1)^j·DCT-III(X[n − 1 − k])[j]` and `DST-IV(x)[k] = (−1)^k·DCT-IV(x[n − 1 − j])[k]`, folded into the reordering and the twiddle passes rather than run as passes of their own.
- Lengths up to 8 (16 for type I) are summed directly from the matrix, each column padded to 8 entries (16 beyond 8) and the sums held in registers in two banks (for the even and the odd columns), one run of 8 by loops unrolled for that length; one run of such a length skips the checks and the loop of a batch, and none takes working memory. The crossovers were measured.

The passes around the transforms are loops over slices cut to the lengths they cover, read and written from both ends, which the compiler vectorises (NEON, and SSE2 at the baseline of x86_64); the permutation and the `√2` pass of the odd type IV are scalar, and the type I split de-interleaves its input in one pass and combines its two halves with reversed stores. For `f64`, a type II of an even length of 1024 or more inside the type I split (DCT-I of 2049 values and more, DST-I of 2047) is reordered for Makhoul's step straight from the input, every fourth value, and its twiddle pass and the combination of the split run as one loop, so that neither its values nor its result are copied (measured: within ±2 % of the separate passes for shorter ones down to 256, and for `f32`, which keeps them). Elsewhere DCT-II's twiddle pass runs after the real transform's pass, not fused into it.

### Two and three dimensions

The last, contiguous dimension is transformed as one batch of rows over all the arrays of the call. Each other dimension, for an array of at most 4 MiB whose length along it has no prime factor above 19, runs the Stockham passes of that length along the rows, planned for as many interleaved sequences as there are columns (the vectors running across the columns, each pass reading and writing the whole array, which stays in the caches), the second buffer of the passes the size of one array. A larger array, or a length with a larger prime factor, goes by tiles of adjacent columns: up to 256 bytes of each row (two cache lines of 128 bytes: 16 `Complex<f64>` or 32 `Complex<f32>`) and at most 1 MiB in all are gathered into a buffer, column after column, transformed there as one batch of contiguous sequences and scattered back, so that every pass over the array reads and writes whole cache lines even where the stride of the rows, a power of two, maps the values of a column to the same sets of the cache. The gather and the scatter copy value by value.

### Convolution and correlation

`convolve` takes one of four paths, chosen by a model of their costs in nanoseconds:

- **Scalar**: for at most 16 products, operands of at most 16 values, the double loop on stack copies.
- **Direct**: `c[k] = Σ_t taps[t]·x[k + t]` with `taps` the shorter operand reversed and `x` the longer between `len(short) − 1` zeros, by SIMD kernels: NEON on aarch64, SSE2 and AVX2 with FMA (chosen at run time) on x86_64, and the same steps on arrays of lanes elsewhere and under Miri, from one body of macros. A kernel keeps a block of outputs in registers (8 vectors on NEON, 4 on AVX2 and SSE2) while the taps run past it two vectors at a time: for each lane `s` of the taps the vectors of `x` from `k + t + s` are loaded once, and each serves the tap `t + s` for one block of sums and `t + W + s` for the next, kept apart so that no multiply-add waits for another; the taps left over are taken one at a time. For at most 4 taps another kernel holds the taps in registers, broadcast once, and computes in place on the longer operand every output that takes all of them (four vectors of outputs at a time, then one, then the last values one by one); the `len(short) − 1` outputs at either end are summed over the values inside, so nothing is copied. Complex values stay interleaved: the kernel sums `A = Σ x·re(tap)` and `B = Σ x·im(tap)` lane by lane and forms `A + i·B` once per block, so its steps have no shuffles. With more taps, the outputs that every tap reaches (a multiple of 64 of them) are computed in place on the longer operand; only those at the two ends read short copies between zeros (on the stack up to 512 values, the taps up to 128). A correlation is `conv(op(u), v) = op(conv(u, op(v)))` for `op` a reversal, a conjugation or both, so whichever operand is reversed nothing is copied; the result is reversed or conjugated in place.
- **Transforms of the padded product**: both operands padded to a length `L ≥ len(a) + len(b) − 1` of the form `2^a·3^b·5^c·7^d` (`b ≤ 4`, `c ≤ 2`, `d ≤ 1`, and `a ≥ 1` for real data, which go through `RealFft` and its `L/2 + 1` values), the cheapest by the model rather than the shortest; the spectra multiplied with the scale `1/L` folded in (a SIMD pass), and the product transformed back.
- **Overlap-add**, when the longer operand has at least four times the shorter's length: the longer is cut into blocks of `B = L − len(short) + 1` values, each padded to `L`, transformed, multiplied by the shorter operand's spectrum (computed once) and transformed back, and the results added where they overlap; the blocks go through the transforms in batches of up to 65,536 values a call. The padded product is the case of one block.

The model prices the direct path at `direct + copy·len + mac·len·len(short)`, a transform of the length `L = 2^a·3^b·5^c·7^d` at `call + fft·L·(a + w₃b + w₅c + w₇d)` (`fft·L·log₂ L` for a power of two; factors 11 and 13 and larger primes have weights of their own), planning at half of `plan_call + plan·L`, and each pass over the buffers at `pass·L`. It takes the cheapest of the direct path, the padded product at its cheapest length, and overlap-add at every length `L` of that form from `2·len(short)` up to the padded length or `64·max(len(short), 16)`, whichever is smaller. A lower bound of every transform path skips that search when the direct path is cheaper anyway: each plans a length `L ≥ 2·len(short) − 1`, transforms the shorter operand once and its blocks there and back, at least `len(long)` values each way, and a transform of `L` values costs at least `fft·L·log₂ L` times the least weight of a radix per bit; being below every path's estimate, the bound never changes the path chosen. Half of the planning: with the plans kept, a first call pays all of it and a repeated one none, and the choice may not depend on what is kept, or the same call would give other bits after another one; charging all of it measured up to 3.2 times slower on repeated calls where kept plans make the transforms win, charging none of it up to 2.9 times slower on first calls, and a half at most 1.6 times (2.0 for repeated `f32` calls) either way, the geometric means within 1.03. The constants are measured per type by the bench's `calibrate` (`mac` by sweeps of the direct path near the crossover, where its taps are longer than calibrate's fit sees), and were measured again on the final transforms and kernels of v0.2: the direct path's constants are the new fits, the transforms' the earlier ones scaled by one factor per type (0.85 for `f32`, 0.90 for `f64`, 0.87 for `Complex<f32>`, none for `Complex<f64>`), so that the transform lengths chosen stay and the lower bound still skips the search for short direct convolutions (the fits themselves would add that search, about 0.2 µs, to small ones); they are to be measured again whenever the transforms or the kernels change; on x86_64 without AVX2 the cost of a product of the direct path is doubled, an estimate (SSE2's kernels have a quarter of AVX2's products a cycle and its transforms about half of AVX2's speed), not a measurement.

`circular_convolve` takes the cheapest of the direct path on `a` extended periodically (`n²` products), three transforms of the length `n`, and the linear convolution by transforms of a smooth length of at least `2n − 1`, folded modulo `n`, which the model's costs favour when `n` has a large prime factor (whose transforms of `n` would take Bluestein's algorithm). `circular_correlate` is the circular convolution with `conj(b[(n − j) mod n])`, written straight into the buffer each path fills (the folded linear path makes one copy); it takes the time of `circular_convolve`.

## Testing

The tests of `lib/tests/math/fft` (54 of them) check every algorithm of every transform against references summed in double-double arithmetic: the DFT for the complex and real transforms on dense data at every length to 64 and at chosen ones to 2062, and on sparse data at long lengths to 100,000, the definitions of the eight cosine and sine transforms, and the double-double matrix columns (impulses) at long lengths, with the relative L2 error bounded by `c·ε·(log₂ n + 1)`, `c` 3 for the complex transform, 10 for the others, 20 for round trips and 4 for batches against single runs; every batch against its runs one by one; the ignored imaginary parts of the real inverse for every algorithm; the two and three dimensions against line-by-line transforms (tiles of widths not a multiple of the tile, long columns, axes of 1); in place against out of place; every `_with_scratch` method with exactly `scratch_len()` values and with more against the allocating one, and short scratch panicking; empty batches of every algorithm (a panic of `RealFft::forward` on an empty batch of an even length of 18 and more was found by the fuzzing below and fixed); clones; `Send` and `Sync`; convolutions on every path (scalar, direct with its interior and its ends, either operand longer, padded product, overlap-add with full and partial batches, the three circular paths) against direct double-double sums, the direct kernels at every count of taps, exact small integers, and the plan cache (kept, evicted least recently used, a plan past the bytes never kept, eight threads with another churning the cache, every result bit-identical to the one-thread result); the real transform at lengths of every radix of the odd split (3 to 19, the composites 9 and 15 among them) and of the complex fallback (23·23, 23·29), and with one class of residues of 10⁶ against 10⁻⁶ for the others, for the classes 3, 5, 7, 9 and 15; the allocation-free `convolve_into` and `correlate_into` against the vectors, and a slice of another length panicking; arrays of two and three dimensions by the passes along the rows and by tiles, (700, 400) and (72, 72, 72) among them.

The gates passed on the code of the release that added the module: `cargo fmt`, `clippy -D warnings`, clippy for `wasm32`, a check on the MSRV 1.88, rustdoc with `-D warnings`, the math tests in release and debug (1271) and the doctests (707). Under Miri, one test a process, on `aarch64-apple-darwin`, `s390x-unknown-linux-gnu` (big-endian) and `i686-unknown-linux-gnu` (32-bit), 49 of the 52 tests it then had ran and passed on each (three are too long for Miri: the long operands and two of the cache's; the threads test runs the cache under Miri with three threads); Miri runs the portable kernels, which follow the same steps as the vector ones lane by lane. It caught a violation of Stacked Borrows in the setup of Rader's and Bluestein's kernels, and on 32 bits a bound `1 << 60` overflowing `usize`; both were fixed, and with the second Rader's `a·g mod p`, which could overflow a 32-bit `usize` for `p > 2¹⁶`. On x86_64, in Docker's `linux/amd64` (`rust:1.88`), the 52 tests passed on the AVX2 and FMA path, and 52 again on the SSE2 path, by the same test binary under `qemu-x86_64 -cpu Nehalem`, where the plans reported `isa: Sse2`. The x86_64 kernels were checked for correctness under emulation only; their speed has never been measured.

The module was also fuzzed differentially against FFTW and the double-double reference: every API, the `_with_scratch` forms and batches among them, `f32` and `f64`, on random, sparse, constant and impulse data of ordinary, tiny and huge scales, and on empty data and lengths 0. The first run, 5,211,326 cases on the code before its last changes, found the empty-batch panic above; the second, 3,478,231 cases on the final code, found nothing, its worst error 0.14 of the tests' bound (in DCT-IV). Its only rejected cases are convolutions with near-constant operands, where an error measured relative to `‖a‖·‖b‖` does not apply; FFTW in the same precision fails them alike.

The speed pass of v0.2 (the composite butterflies, the fused real kernels and last pass, the odd splits' kernels and radices, the products of two tables for `f32` twiddles, FCADD, the few-tap convolutions and the `_into` forms, the passes along the rows of `Fft2` and `Fft3`, the uninitialised working memory) was checked on its final code by the 54 tests in release and debug and the 17 doctests of the module; under Miri on `aarch64-apple-darwin`, 51 of the 54 (the same three too long); in the CI's x86_64 container (Ubuntu 26.04, x86-64-v3, Rust 1.100), the 54 on the AVX2 and FMA path, and on the SSE2 path by the suite built with `--cfg a_baseline`, which makes every run-time detection of AVX2 report false; on the M4, the NEON kernels' fallback for a target without FEAT_FCMA by the suite built with `RUSTFLAGS="-C target-feature=-fcma"` (no FCADD left in the module's code, against 534 instructions in the default build), the 54 passing, and by the audit's hashes of every output, bit-identical to FCADD's; by clippy for `x86_64-apple-darwin` with Rust 1.100, with no warning; and by the differential fuzzing above against FFTW, 7,334,013 cases in two runs (the second, 2,979,558, on the final code) with no failure and a worst error of 0.015 of the bound. An independent audit then compared it with double-double references and with the code before the pass: 809 complex lengths (every one to 700, composites, powers of two and their neighbours to 2²², primes to 1,000,003, the edges of the tables' switches) and 1,221 real ones in seven methods and batches, the eight cosine and sine types at 283 lengths, 46 arrays of two and 20 of three dimensions, and the convolutions on every path, with a worst error of 0.097 of the tests' bound; the methods bit-identical to each other; working memory full of NaN giving the same bits as zeroed; no `_with_scratch` method allocating (a counting allocator); eight threads bit-identical to one; and Miri, with no undefined behaviour, on its tests of every new unsafe path on `aarch64-apple-darwin`, `s390x-unknown-linux-gnu` (big-endian) and `i686-unknown-linux-gnu` (32-bit). Real transforms of `f32` subnormals of a few significant bits (about 10⁻⁴¹ to 10⁻⁴⁴) are far less accurate than the bound, as they were before (up to three times more so now); nothing is promised there, and `f64` subnormals and every normal scale showed no excess.

## Measured

Measured on an Apple M4 Max (macOS 27.2.1) on 2026-11-26 between 06:51 and 07:05, in the final run of the release's speed pass, which timed every area of the crate in turn with nothing else running: one thread on every side, by the harness `bench/fft` of the repository (its method is in `bench/README.md`). `a` is built as a user builds it, by rustc 1.100 stable with the default release profile (no LTO, no `target-cpu`), and so are rustfft 6.4.1, realfft 3.5.1 and rustdct 0.7.1; FFTW 3.3.11 is Homebrew's build (which on arm64 has no SIMD codelets: FFTW runs its scalar ones here), called from a driver built with `-O3 -mcpu=native`; vDSP is Accelerate's, of macOS 27.2.1. Each figure is the median of 15 samples of at least 1 ms, `a` and each reference alternated sample by sample in one process, after each reference's output on the row's input was compared with `a`'s (no mismatch). The x86_64 paths were not timed for these tables.

The references, each in its best case: FFTW planned before the clock with `FFTW_MEASURE` (and with `FFTW_PATIENT` too for the powers of two from 64 to 2²⁰, the faster kept), in place for the complex transforms and out of place for the real and trigonometric ones; rustfft's `process_with_scratch` in place and `process_immutable_with_scratch` out of place; realfft's and rustdct's `process_with_scratch` (`rustdct` in place, against `transform_with_scratch`); vDSP on split complex data and with its real data split into even and odd values before the clock (`vDSP_fft_zip`, `vDSP_DFT_Execute` and `vDSP_DFT_Interleaved_Execute` for the complex transforms, `vDSP_fft_zrip`, `vDSP_DFT_zrop` and the interleaved real DFTs for real data, `vDSP_fftm_zip` and `vDSP_fftm_zrip` for batches, `vDSP_DCT_Execute` for the `f32` DCT-II, III and IV, `vDSP_fft2d_zip`, and the direct `vDSP_conv` and `vDSP_zconv` on operands padded and split before the clock, into a preallocated output). The convolutions of FFTW, rustfft and realfft are the library's transforms with the harness's spectral product and overlap-add, the output preallocated, at the fastest of the padded 7-smooth length, the next power of two and the overlap-add lengths `2^k` from `2·len(b)` to `64·len(b)`, chosen per row; the circular ones two forward transforms of `n` and one inverse. Planning is compared with FFTW's `FFTW_ESTIMATE`, rustfft's, realfft's and rustdct's planners and vDSP's setups, each made and destroyed in the loop.

A case is an operation at one length and type; its ratio `a/x` is `a`'s time over that of the fastest reference that does the same work (below 1 `a` is faster), and a loss is a ratio above 1.05. Counted as the same work though they favour the reference: vDSP's layouts and padding prepared before the clock, realfft and FFTW's `c2r` using their input as scratch where `a` keeps its input, and the preallocated outputs where `convolve`, `correlate` and the circular forms return a new vector. Not counted: rustfft's out-of-place form that overwrites its input (`process_outofplace_with_scratch`, a weaker contract; the form that keeps it is the reference), and `FFTW_MEASURE`'s planning, which times candidate plans. Over the 1,602 cases there are 300 losses, 145 ties within 5 % and 1,157 cases below 0.95; the median ratio is 0.762.

Each cell gives `a`'s time, then `a/x` and the reference `x`.

### The complex transform

`Fft::forward_with_scratch` in place, powers of two:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 2 | 3.04 ns, 1.31 FFTW | 3.12 ns, 1.88 rustfft |
| 4 | 3.98 ns, 0.952 FFTW | 4.14 ns, 0.966 FFTW |
| 8 | 7.10 ns, 1.11 FFTW | 6.38 ns, 0.987 FFTW |
| 16 | 10.4 ns, 0.890 FFTW | 10.7 ns, 0.961 rustfft |
| 32 | 31.5 ns, 1.15 rustfft | 18.6 ns, 1.15 rustfft |
| 64 | 53.0 ns, 0.929 vDSP | 29.5 ns, 1.07 vDSP |
| 128 | 121 ns, 0.953 vDSP | 65.1 ns, 1.30 vDSP |
| 256 | 259 ns, 0.949 vDSP | 128 ns, 1.08 vDSP |
| 512 | 510 ns, 0.798 vDSP | 274 ns, 1.64 vDSP |
| 1024 | 1.26 µs, 0.924 vDSP | 617 ns, 2.22 vDSP |
| 2048 | 2.86 µs, 0.905 vDSP | 1.31 µs, 2.22 vDSP |
| 4096 | 6.94 µs, 1.02 vDSP | 3.65 µs, 2.85 vDSP |
| 16,384 | 35.6 µs, 0.655 vDSP | 16.0 µs, 1.94 vDSP |
| 65,536 | 179 µs, 0.649 FFTW | 85.4 µs, 1.80 vDSP |
| 262,144 | 910 µs, 0.729 rustfft | 425 µs, 1.75 vDSP |
| 1,048,576 | 4.02 ms, 0.707 rustfft | 1.88 ms, 0.987 vDSP |
| 4,194,304 | 19.4 ms, 0.734 rustfft | 9.03 ms, 1.12 vDSP |

Smooth lengths `3·2^k` and `5·2^k`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 6 | 7.88 ns, 1.32 FFTW | 7.54 ns, 1.27 FFTW |
| 12 | 10.0 ns, 1.07 FFTW | 10.1 ns, 1.08 FFTW |
| 24 | 15.9 ns, 0.774 rustfft | 15.7 ns, 1.17 rustfft |
| 48 | 48.7 ns, 1.01 vDSP | 29.2 ns, 0.879 vDSP |
| 96 | 88.6 ns, 0.915 vDSP | 50.6 ns, 0.547 vDSP |
| 384 | 427 ns, 0.732 vDSP | 227 ns, 1.37 vDSP |
| 1536 | 2.00 µs, 0.653 rustfft | 1.05 µs, 2.50 vDSP |
| 6144 | 11.5 µs, 0.726 rustfft | 6.62 µs, 3.13 vDSP |
| 24,576 | 67.7 µs, 0.794 rustfft | 30.9 µs, 2.58 vDSP |
| 98,304 | 323 µs, 0.820 rustfft | 152 µs, 2.14 vDSP |
| 786,432 | 2.91 ms, 0.704 rustfft | 1.41 ms, 1.42 vDSP |
| 10 | 10.3 ns, 1.09 FFTW | 9.21 ns, 1.01 rustfft |
| 20 | 14.4 ns, 0.877 FFTW | 14.2 ns, 0.866 FFTW |
| 40 | 35.2 ns, 0.951 vDSP | 24.4 ns, 0.937 vDSP |
| 80 | 76.8 ns, 0.940 vDSP | 43.3 ns, 0.920 vDSP |
| 320 | 324 ns, 0.674 vDSP | 172 ns, 1.22 vDSP |
| 1280 | 1.65 µs, 0.597 FFTW | 853 ns, 2.04 vDSP |
| 5120 | 8.93 µs, 0.586 FFTW | 5.58 µs, 3.15 vDSP |
| 20,480 | 41.6 µs, 0.590 FFTW | 19.7 µs, 2.16 vDSP |
| 81,920 | 244 µs, 0.671 FFTW | 116 µs, 2.06 vDSP |
| 655,360 | 2.26 ms, 0.604 rustfft | 1.10 ms, 1.61 vDSP |

Short and odd smooth lengths, `3^k`, `5^k`, `7^k` and mixed ones (15,360 is `15·2^10`):

| n | `f64` | `f32` |
| ---: | --- | --- |
| 3 | 4.92 ns, 1.07 FFTW | 5.90 ns, 1.27 FFTW |
| 5 | 6.91 ns, 1.06 FFTW | 5.96 ns, 0.932 FFTW |
| 7 | 7.22 ns, 0.989 FFTW | 7.75 ns, 1.06 FFTW |
| 9 | 10.2 ns, 0.960 FFTW | 10.4 ns, 1.04 FFTW |
| 11 | 9.97 ns, 0.864 rustfft | 10.4 ns, 0.995 rustfft |
| 13 | 11.8 ns, 0.831 rustfft | 10.8 ns, 0.880 rustfft |
| 15 | 13.4 ns, 0.916 FFTW | 13.5 ns, 0.992 rustfft |
| 25 | 20.1 ns, 0.676 FFTW | 20.4 ns, 0.691 FFTW |
| 27 | 34.7 ns, 0.732 FFTW | 25.7 ns, 0.592 FFTW |
| 49 | 46.6 ns, 0.600 FFTW | 34.3 ns, 0.456 rustfft |
| 81 | 104 ns, 0.689 FFTW | 69.1 ns, 0.531 rustfft |
| 243 | 344 ns, 0.582 FFTW | 203 ns, 0.385 FFTW |
| 2187 | 4.29 µs, 0.614 FFTW | 2.43 µs, 0.369 FFTW |
| 19,683 | 65.5 µs, 0.806 FFTW | 35.2 µs, 0.459 FFTW |
| 531,441 | 2.75 ms, 0.828 FFTW | 1.41 ms, 0.480 FFTW |
| 625 | 896 ns, 0.550 FFTW | 517 ns, 0.315 FFTW |
| 15,625 | 36.8 µs, 0.535 FFTW | 20.7 µs, 0.335 FFTW |
| 390,625 | 1.25 ms, 0.465 FFTW | 680 µs, 0.274 FFTW |
| 2401 | 4.27 µs, 0.528 FFTW | 2.49 µs, 0.322 FFTW |
| 117,649 | 329 µs, 0.445 FFTW | 182 µs, 0.258 FFTW |
| 823,543 | 2.73 ms, 0.422 FFTW | 1.49 ms, 0.245 rustfft |
| 18 | 25.6 ns, 0.826 FFTW | 21.5 ns, 0.694 FFTW |
| 36 | 44.3 ns, 1.27 vDSP | 26.6 ns, 1.09 vDSP |
| 60 | 63.5 ns, 1.02 vDSP | 36.8 ns, 1.07 vDSP |
| 100 | 101 ns, 0.602 FFTW | 73.8 ns, 0.473 rustfft |
| 360 | 438 ns, 0.578 FFTW | 242 ns, 0.397 rustfft |
| 1000 | 1.42 µs, 0.567 FFTW | 766 ns, 0.310 FFTW |
| 2160 | 3.89 µs, 0.684 FFTW | 1.89 µs, 0.373 FFTW |
| 6000 | 13.0 µs, 0.730 FFTW | 6.22 µs, 0.371 FFTW |
| 10,000 | 23.4 µs, 0.709 FFTW | 11.5 µs, 0.353 FFTW |
| 15,360 | 41.0 µs, 0.828 FFTW | 17.9 µs, 2.47 vDSP |
| 48,000 | 122 µs, 0.621 FFTW | 60.6 µs, 0.351 FFTW |
| 100,000 | 275 µs, 0.584 FFTW | 135 µs, 0.295 FFTW |
| 1,000,000 | 3.30 ms, 0.429 FFTW | 1.65 ms, 0.229 FFTW |

Primes, by one pass (17), Rader's algorithm (101 and 1009) and Bluestein's (10,007 and above), and twice a prime:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 17 | 17.4 ns, 0.838 rustfft | 15.8 ns, 0.929 rustfft |
| 101 | 293 ns, 0.313 FFTW | 190 ns, 0.224 rustfft |
| 1009 | 4.05 µs, 0.337 rustfft | 2.32 µs, 0.234 rustfft |
| 10,007 | 111 µs, 0.680 FFTW | 55.1 µs, 0.542 rustfft |
| 100,003 | 1.58 ms, 0.719 FFTW | 771 µs, 0.473 rustfft |
| 1,000,003 | 19.9 ms, 0.783 rustfft | 9.76 ms, 0.747 rustfft |
| 202 | 752 ns, 0.365 FFTW | 543 ns, 0.276 rustfft |
| 2018 | 10.3 µs, 0.380 rustfft | 6.51 µs, 0.305 rustfft |
| 20,014 | 260 µs, 0.755 FFTW | 123 µs, 0.540 rustfft |
| 200,006 | 3.47 ms, 0.725 FFTW | 1.68 ms, 0.505 rustfft |
| 2,000,006 | 40.9 ms, 0.756 rustfft | 20.9 ms, 0.747 rustfft |

The inverse, `inverse_with_scratch`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 16 | 11.3 ns, 0.956 FFTW | 11.0 ns, 0.974 rustfft |
| 64 | 55.7 ns, 0.972 vDSP | 30.6 ns, 1.10 vDSP |
| 1024 | 1.42 µs, 1.02 vDSP | 672 ns, 2.40 vDSP |
| 4096 | 7.52 µs, 1.09 vDSP | 4.12 µs, 3.21 vDSP |
| 65,536 | 199 µs, 0.698 FFTW | 90.0 µs, 1.88 vDSP |
| 1,048,576 | 4.25 ms, 0.745 rustfft | 1.98 ms, 1.05 vDSP |
| 1000 | 1.56 µs, 0.640 FFTW | 832 ns, 0.342 FFTW |
| 1,000,000 | 3.62 ms, 0.478 FFTW | 1.80 ms, 0.250 FFTW |
| 1009 | 4.11 µs, 0.332 rustfft | 2.32 µs, 0.233 rustfft |
| 10,007 | 113 µs, 0.680 FFTW | 56.6 µs, 0.568 rustfft |
| 100,003 | 1.59 ms, 0.729 FFTW | 773 µs, 0.482 rustfft |

Out of place, `forward_to`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 16 | 6.36 ns, 0.761 rustfft | 6.48 ns, 1.23 rustfft |
| 64 | 64.4 ns, 1.12 vDSP | 34.5 ns, 1.23 vDSP |
| 256 | 291 ns, 1.06 vDSP | 145 ns, 1.68 vDSP |
| 1024 | 1.46 µs, 1.03 vDSP | 705 ns, 2.53 vDSP |
| 4096 | 7.94 µs, 1.07 vDSP | 3.88 µs, 3.04 vDSP |
| 65,536 | 180 µs, 0.718 FFTW | 87.4 µs, 1.81 vDSP |
| 1,048,576 | 4.13 ms, 0.750 rustfft | 1.93 ms, 0.965 vDSP |
| 1000 | 1.58 µs, 0.640 FFTW | 838 ns, 0.353 FFTW |
| 10,007 | 122 µs, 0.748 FFTW | 60.6 µs, 0.594 rustfft |

The allocating `forward` of one sequence:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 16 | 10.9 ns, 0.916 FFTW | 10.8 ns, 1.00 rustfft |
| 1024 | 1.44 µs, 0.680 rustfft | 735 ns, 0.565 rustfft |
| 65,536 | 192 µs, 0.672 FFTW | 96.6 µs, 0.602 rustfft |

Batches of 1000 transforms in one call of `forward`, in ns a transform, against FFTW's `plan_many_dft`, rustfft on the whole buffer and `vDSP_fftm_zip`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 2 | 0.337 ns, 1.23 rustfft | 0.277 ns, 1.49 rustfft |
| 3 | 0.604 ns, 0.979 rustfft | 0.500 ns, 1.15 rustfft |
| 4 | 0.590 ns, 0.831 rustfft | 0.512 ns, 0.927 rustfft |
| 5 | 1.27 ns, 0.689 rustfft | 0.985 ns, 0.840 rustfft |
| 6 | 1.74 ns, 1.15 rustfft | 1.36 ns, 1.29 rustfft |
| 8 | 1.82 ns, 0.672 rustfft | 1.36 ns, 0.823 rustfft |
| 10 | 3.94 ns, 0.897 rustfft | 2.82 ns, 1.09 rustfft |
| 12 | 4.79 ns, 0.979 rustfft | 3.28 ns, 1.23 rustfft |
| 15 | 8.07 ns, 0.916 rustfft | 5.50 ns, 1.14 rustfft |
| 16 | 6.12 ns, 0.760 rustfft | 4.20 ns, 0.914 rustfft |
| 17 | 11.5 ns, 0.800 rustfft | 7.12 ns, 0.977 rustfft |
| 32 | 23.4 ns, 0.981 rustfft | 12.1 ns, 1.05 rustfft |
| 64 | 48.7 ns, 0.751 vDSP | 28.5 ns, 0.787 vDSP |
| 101 | 299 ns, 0.322 FFTW | 183 ns, 0.226 rustfft |
| 128 | 129 ns, 0.908 vDSP | 61.4 ns, 0.941 vDSP |
| 1024 | 1.43 µs, 0.935 vDSP | 663 ns, 2.19 vDSP |

Planning, `Fft::new`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 16 | 168 ns, 1.98 vDSP | 165 ns, 1.78 vDSP |
| 64 | 307 ns, 2.10 vDSP | 329 ns, 2.24 vDSP |
| 1024 | 2.40 µs, 1.35 vDSP | 1.91 µs, 0.498 rustfft |
| 4096 | 8.49 µs, 1.20 vDSP | 5.12 µs, 0.352 rustfft |
| 65,536 | 123 µs, 0.673 vDSP | 60.9 µs, 0.925 FFTW |
| 1,048,576 | 1.95 ms, 4.50 FFTW | 1.48 ms, 3.36 FFTW |
| 1000 | 2.26 µs, 0.476 rustfft | 1.82 µs, 0.412 rustfft |
| 1009 | 13.5 µs, 1.13 rustfft | 15.1 µs, 1.38 rustfft |
| 10,007 | 183 µs, 0.760 FFTW | 192 µs, 1.01 FFTW |
| 100,003 | 1.98 ms, 0.705 FFTW | 2.06 ms, 1.06 FFTW |

### Real transforms

`RealFft::forward` and `inverse`, the allocating forms out of place: powers of two, then other even lengths, odd lengths with small factors, and primes and lengths with a factor above 19.

`f64`:

| n | `forward` | `inverse` |
| ---: | --- | --- |
| 8 | 4.03 ns, 1.09 FFTW | 4.22 ns, 1.22 FFTW |
| 16 | 4.77 ns, 0.752 FFTW | 5.52 ns, 0.929 FFTW |
| 32 | 11.8 ns, 0.809 FFTW | 11.9 ns, 0.724 FFTW |
| 64 | 30.9 ns, 0.886 FFTW | 44.5 ns, 1.16 FFTW |
| 128 | 64.5 ns, 0.753 FFTW | 77.0 ns, 0.885 FFTW |
| 256 | 157 ns, 0.906 vDSP | 178 ns, 1.03 vDSP |
| 512 | 328 ns, 0.862 vDSP | 376 ns, 0.937 vDSP |
| 1024 | 700 ns, 0.819 vDSP | 764 ns, 0.856 vDSP |
| 4096 | 3.60 µs, 0.892 vDSP | 4.17 µs, 0.997 vDSP |
| 8192 | 7.85 µs, 0.888 vDSP | 9.25 µs, 1.04 vDSP |
| 16,384 | 18.9 µs, 0.898 vDSP | 21.4 µs, 1.000 vDSP |
| 65,536 | 217 µs, 1.71 realfft | 98.2 µs, 0.778 realfft |
| 262,144 | 431 µs, 0.749 realfft | 499 µs, 0.883 realfft |
| 1,048,576 | 2.06 ms, 0.724 realfft | 2.23 ms, 0.760 realfft |
| 12 | 4.76 ns, 0.961 FFTW | 5.43 ns, 1.15 FFTW |
| 80 | 42.6 ns, 0.842 vDSP | 52.7 ns, 1.09 FFTW |
| 96 | 57.0 ns, 0.946 FFTW | 67.2 ns, 1.03 vDSP |
| 384 | 252 ns, 0.783 vDSP | 271 ns, 0.845 vDSP |
| 640 | 445 ns, 0.752 vDSP | 474 ns, 0.846 FFTW |
| 1000 | 862 ns, 0.770 FFTW | 929 ns, 0.807 FFTW |
| 1536 | 1.21 µs, 0.791 FFTW | 1.35 µs, 0.854 FFTW |
| 5120 | 4.79 µs, 0.760 FFTW | 5.42 µs, 0.879 FFTW |
| 6144 | 6.26 µs, 0.842 FFTW | 6.79 µs, 0.862 realfft |
| 6000 | 6.81 µs, 0.819 FFTW | 7.31 µs, 0.903 FFTW |
| 10,000 | 11.7 µs, 0.775 FFTW | 12.8 µs, 0.842 FFTW |
| 30,000 | 39.3 µs, 0.711 FFTW | 44.5 µs, 0.798 FFTW |
| 100,000 | 151 µs, 0.704 FFTW | 168 µs, 0.790 FFTW |
| 1,000,000 | 1.76 ms, 0.655 FFTW | 1.95 ms, 0.706 FFTW |
| 15 | 10.5 ns, 1.45 FFTW | 10.9 ns, 1.75 FFTW |
| 45 | 54.9 ns, 0.956 FFTW | 71.6 ns, 1.33 FFTW |
| 135 | 147 ns, 1.10 FFTW | 157 ns, 1.16 FFTW |
| 243 | 341 ns, 1.13 FFTW | 344 ns, 1.22 FFTW |
| 625 | 794 ns, 0.791 FFTW | 800 ns, 0.915 FFTW |
| 1155 | 1.45 µs, 0.819 FFTW | 1.66 µs, 0.880 FFTW |
| 2187 | 3.17 µs, 0.872 FFTW | 3.40 µs, 0.989 FFTW |
| 2401 | 3.21 µs, 0.795 FFTW | 3.36 µs, 0.850 FFTW |
| 3125 | 4.63 µs, 0.721 FFTW | 4.83 µs, 0.897 FFTW |
| 16,807 | 28.6 µs, 0.752 FFTW | 30.8 µs, 0.686 FFTW |
| 19,683 | 39.8 µs, 0.944 FFTW | 42.1 µs, 0.857 FFTW |
| 99,225 | 211 µs, 0.844 FFTW | 232 µs, 0.761 FFTW |
| 177,147 | 451 µs, 0.916 FFTW | 459 µs, 0.794 FFTW |
| 101 | 433 ns, 0.519 FFTW | 358 ns, 0.424 FFTW |
| 1009 | 4.92 µs, 0.504 FFTW | 4.89 µs, 0.525 FFTW |
| 10,007 | 125 µs, 0.819 FFTW | 124 µs, 0.793 FFTW |
| 100,003 | 1.62 ms, 0.613 FFTW | 1.69 ms, 0.624 FFTW |
| 15,015 | 25.4 µs, 0.719 FFTW | 27.1 µs, 0.643 FFTW |

`f32`:

| n | `forward` | `inverse` |
| ---: | --- | --- |
| 8 | 3.99 ns, 1.07 FFTW | 3.96 ns, 1.14 FFTW |
| 16 | 4.88 ns, 0.867 vDSP | 5.19 ns, 0.877 FFTW |
| 32 | 11.1 ns, 0.963 vDSP | 11.5 ns, 0.858 vDSP |
| 64 | 27.6 ns, 1.41 vDSP | 26.8 ns, 1.37 vDSP |
| 128 | 45.7 ns, 1.11 vDSP | 47.8 ns, 1.27 vDSP |
| 256 | 104 ns, 1.46 vDSP | 101 ns, 1.45 vDSP |
| 512 | 196 ns, 1.33 vDSP | 204 ns, 1.35 vDSP |
| 1024 | 400 ns, 1.81 vDSP | 422 ns, 2.19 vDSP |
| 4096 | 1.97 µs, 2.34 vDSP | 2.03 µs, 2.36 vDSP |
| 8192 | 5.45 µs, 3.37 vDSP | 5.83 µs, 3.58 vDSP |
| 16,384 | 9.83 µs, 2.45 vDSP | 9.37 µs, 2.35 vDSP |
| 65,536 | 45.5 µs, 2.04 vDSP | 46.0 µs, 2.18 vDSP |
| 262,144 | 239 µs, 1.77 vDSP | 234 µs, 1.87 vDSP |
| 1,048,576 | 1.01 ms, 1.38 vDSP | 1.05 ms, 1.52 vDSP |
| 12 | 4.64 ns, 0.961 FFTW | 5.44 ns, 1.17 FFTW |
| 80 | 34.5 ns, 1.06 vDSP | 33.9 ns, 0.991 vDSP |
| 96 | 46.8 ns, 1.10 vDSP | 42.8 ns, 0.984 vDSP |
| 384 | 155 ns, 0.894 vDSP | 156 ns, 0.894 vDSP |
| 640 | 260 ns, 1.45 vDSP | 264 ns, 1.42 vDSP |
| 1000 | 477 ns, 0.435 FFTW | 510 ns, 0.494 FFTW |
| 1536 | 689 ns, 2.00 vDSP | 740 ns, 2.25 vDSP |
| 5120 | 2.65 µs, 2.70 vDSP | 2.83 µs, 2.97 vDSP |
| 6144 | 3.75 µs, 3.15 vDSP | 4.07 µs, 3.45 vDSP |
| 6000 | 3.92 µs, 0.516 FFTW | 4.31 µs, 0.537 FFTW |
| 10,000 | 7.13 µs, 0.495 FFTW | 7.38 µs, 0.470 FFTW |
| 30,000 | 21.6 µs, 0.436 FFTW | 22.6 µs, 0.432 FFTW |
| 100,000 | 78.9 µs, 0.386 FFTW | 84.7 µs, 0.408 FFTW |
| 1,000,000 | 904 µs, 0.341 FFTW | 990 µs, 0.397 FFTW |
| 15 | 11.3 ns, 1.59 FFTW | 10.8 ns, 1.67 FFTW |
| 45 | 50.7 ns, 1.09 FFTW | 62.1 ns, 1.31 FFTW |
| 135 | 140 ns, 1.03 FFTW | 137 ns, 0.981 FFTW |
| 243 | 274 ns, 0.908 FFTW | 281 ns, 0.980 FFTW |
| 625 | 608 ns, 0.639 FFTW | 632 ns, 0.740 FFTW |
| 1155 | 1.21 µs, 0.680 FFTW | 1.33 µs, 0.747 FFTW |
| 2187 | 2.44 µs, 0.681 FFTW | 2.63 µs, 0.781 FFTW |
| 2401 | 2.50 µs, 0.614 FFTW | 2.60 µs, 0.663 FFTW |
| 3125 | 3.33 µs, 0.568 FFTW | 3.48 µs, 0.587 FFTW |
| 16,807 | 19.5 µs, 0.547 FFTW | 20.6 µs, 0.592 FFTW |
| 19,683 | 24.7 µs, 0.608 FFTW | 26.8 µs, 0.673 FFTW |
| 99,225 | 142 µs, 0.577 FFTW | 155 µs, 0.618 FFTW |
| 177,147 | 274 µs, 0.574 FFTW | 298 µs, 0.607 FFTW |
| 101 | 223 ns, 0.286 FFTW | 229 ns, 0.283 FFTW |
| 1009 | 2.79 µs, 0.291 FFTW | 2.86 µs, 0.300 FFTW |
| 10,007 | 70.5 µs, 0.656 realfft | 69.8 µs, 0.641 realfft |
| 100,003 | 919 µs, 0.581 realfft | 981 µs, 0.605 realfft |
| 15,015 | 18.2 µs, 0.523 FFTW | 19.5 µs, 0.550 FFTW |

The `_with_scratch` forms:

| n | `forward_with_scratch` `f64` | `f32` | `inverse_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 15 | 9.04 ns, 1.32 FFTW | 9.71 ns, 1.37 FFTW | 9.96 ns, 1.61 FFTW | 10.4 ns, 1.61 FFTW |
| 16 | 6.19 ns, 0.969 FFTW | 6.21 ns, 0.932 FFTW | 4.72 ns, 0.795 FFTW | 4.56 ns, 0.767 FFTW |
| 64 | 36.8 ns, 1.06 FFTW | 35.7 ns, 1.10 realfft | 45.6 ns, 1.19 FFTW | 28.2 ns, 1.00 realfft |
| 101 | 325 ns, 0.389 FFTW | 204 ns, 0.254 FFTW | 323 ns, 0.381 FFTW | 209 ns, 0.264 FFTW |
| 1000 | 933 ns, 0.833 FFTW | 518 ns, 0.470 FFTW | 926 ns, 0.810 FFTW | 509 ns, 0.495 FFTW |
| 1024 | 765 ns, 0.799 FFTW | 446 ns, 0.601 realfft | 769 ns, 0.703 FFTW | 426 ns, 0.606 realfft |
| 65,536 | 238 µs, 1.88 realfft | 50.6 µs, 0.656 realfft | 99.9 µs, 0.791 realfft | 48.7 µs, 0.654 realfft |

Batches, `n × count` sequences in one call of `forward`, in ns a transform, against FFTW's `plan_many_dft_r2c`, realfft in a loop and `vDSP_fftm_zrip`:

| n × count | `f64` | `f32` |
| ---: | --- | --- |
| 8 × 2048 | 1.59 ns, 0.701 FFTW | 1.54 ns, 0.881 FFTW |
| 15 × 1092 | 7.38 ns, 1.42 FFTW | 8.63 ns, 1.78 FFTW |
| 16 × 1024 | 4.16 ns, 0.807 FFTW | 4.08 ns, 0.936 FFTW |
| 32 × 512 | 11.5 ns, 0.929 FFTW | 11.0 ns, 0.953 FFTW |
| 60 × 272 | 44.4 ns, 1.02 FFTW | 26.3 ns, 0.608 FFTW |
| 64 × 256 | 40.4 ns, 1.37 FFTW | 22.2 ns, 0.762 FFTW |
| 100 × 160 | 77.3 ns, 0.888 FFTW | 45.3 ns, 0.605 FFTW |
| 1024 × 64 | 806 ns, 0.847 vDSP | 426 ns, 0.865 vDSP |

Planning, `RealFft::new`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 16 | 208 ns, 2.27 vDSP | 211 ns, 1.69 vDSP |
| 1024 | 2.44 µs, 1.02 vDSP | 2.02 µs, 0.703 realfft |
| 65,536 | 97.5 µs, 0.684 vDSP | 43.4 µs, 0.266 realfft |
| 1,048,576 | 1.53 ms, 0.667 vDSP | 633 µs, 0.337 FFTW |
| 1000 | 2.37 µs, 0.648 realfft | 2.13 µs, 0.591 realfft |
| 1001 | 9.85 µs, 2.27 realfft | 9.64 µs, 2.23 realfft |
| 1009 | 13.3 µs, 1.10 realfft | 15.1 µs, 1.36 realfft |

### Cosine and sine transforms

`transform_to` against FFTW's `r2r` kinds and, for the `f32` DCT-II, III and IV, `vDSP_DCT_Execute`; `transform_with_scratch` in place against rustdct.

DCT-I:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 9 | 11.1 ns, 0.522 FFTW | 10.0 ns, 0.412 FFTW | 13.9 ns, 0.472 rustdct | 11.7 ns, 0.396 rustdct |
| 17 | 31.5 ns, 0.910 FFTW | 28.1 ns, 0.701 FFTW | 29.5 ns, 0.770 rustdct | 26.9 ns, 1.02 rustdct |
| 65 | 112 ns, 0.982 FFTW | 95.5 ns, 0.875 FFTW | 103 ns, 0.502 rustdct | 94.8 ns, 0.791 rustdct |
| 257 | 417 ns, 1.10 FFTW | 326 ns, 0.910 FFTW | 383 ns, 0.382 rustdct | 311 ns, 0.535 rustdct |
| 1025 | 1.66 µs, 1.01 FFTW | 1.19 µs, 0.735 FFTW | 1.47 µs, 0.278 rustdct | 1.08 µs, 0.392 rustdct |
| 4097 | 8.48 µs, 1.13 FFTW | 4.86 µs, 0.699 FFTW | 7.54 µs, 0.290 rustdct | 4.36 µs, 0.305 rustdct |
| 16,385 | 36.5 µs, 1.01 FFTW | 23.7 µs, 0.760 FFTW | 39.0 µs, 0.311 rustdct | 22.1 µs, 0.325 rustdct |
| 65,537 | 156 µs, 0.887 FFTW | 100 µs, 0.684 FFTW | 155 µs, 0.274 rustdct | 96.7 µs, 0.320 rustdct |
| 64 | 155 ns, 0.855 FFTW | 114 ns, 0.642 FFTW | 139 ns, 0.309 rustdct | 100 ns, 0.268 rustdct |
| 1000 | 4.85 µs, 0.636 FFTW | 3.18 µs, 0.436 FFTW | 4.32 µs, 0.183 rustdct | 2.99 µs, 0.152 rustdct |
| 1024 | 5.05 µs, 0.625 FFTW | 3.44 µs, 0.446 FFTW | 4.50 µs, 0.517 rustdct | 3.24 µs, 0.596 rustdct |
| 4096 | 13.7 µs, 0.810 FFTW | 8.07 µs, 0.528 FFTW | 12.3 µs, 0.218 rustdct | 7.29 µs, 0.178 rustdct |

DCT-II:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 | 4.82 ns, 1.23 FFTW | 3.31 ns, 0.853 FFTW | 7.25 ns, 0.966 rustdct | 6.75 ns, 0.912 rustdct |
| 16 | 20.4 ns, 0.828 FFTW | 17.2 ns, 1.73 vDSP | 22.8 ns, 1.95 rustdct | 23.4 ns, 2.11 rustdct |
| 48 | 51.4 ns, 0.622 FFTW | 42.0 ns, 1.12 vDSP | 41.5 ns, 0.346 rustdct | 38.4 ns, 0.417 rustdct |
| 64 | 68.8 ns, 0.932 FFTW | 54.8 ns, 2.06 vDSP | 54.8 ns, 0.799 rustdct | 45.4 ns, 0.693 rustdct |
| 255 | 388 ns, 0.577 FFTW | 306 ns, 0.485 FFTW | 336 ns, 0.463 rustdct | 284 ns, 0.551 rustdct |
| 256 | 267 ns, 0.786 FFTW | 174 ns, 1.78 vDSP | 215 ns, 0.570 rustdct | 149 ns, 0.466 rustdct |
| 1000 | 1.35 µs, 0.879 FFTW | 764 ns, 0.472 FFTW | 1.12 µs, 0.195 rustdct | 647 ns, 0.142 rustdct |
| 1023 | 3.17 µs, 0.733 FFTW | 2.46 µs, 0.621 FFTW | 2.86 µs, 0.604 rustdct | 2.33 µs, 0.724 rustdct |
| 1024 | 1.20 µs, 0.815 FFTW | 714 ns, 1.74 vDSP | 965 ns, 0.498 rustdct | 581 ns, 0.374 rustdct |
| 3000 | 5.01 µs, 0.932 FFTW | 2.54 µs, 0.492 FFTW | 4.01 µs, 0.223 rustdct | 2.12 µs, 0.159 rustdct |
| 4096 | 6.43 µs, 0.918 FFTW | 3.54 µs, 1.95 vDSP | 5.13 µs, 0.540 rustdct | 2.70 µs, 0.369 rustdct |
| 10,000 | 18.3 µs, 0.708 FFTW | 9.63 µs, 0.502 FFTW | 16.7 µs, 0.269 rustdct | 8.57 µs, 0.207 rustdct |
| 16,384 | 29.3 µs, 0.706 FFTW | 15.5 µs, 1.83 vDSP | 27.0 µs, 0.591 rustdct | 14.1 µs, 0.415 rustdct |
| 65,536 | 256 µs, 1.15 FFTW | 72.1 µs, 0.680 vDSP | 140 µs, 0.656 rustdct | 67.7 µs, 0.442 rustdct |
| 1,048,576 | 3.20 ms, 0.596 FFTW | 1.77 ms, 0.796 vDSP | 2.72 ms, 0.618 rustdct | 1.67 ms, 0.549 rustdct |

DCT-III:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 | 4.80 ns, 1.24 FFTW | 3.27 ns, 0.841 FFTW | 7.19 ns, 0.970 rustdct | 6.70 ns, 0.682 rustdct |
| 16 | 19.1 ns, 0.790 FFTW | 15.2 ns, 1.19 vDSP | 23.2 ns, 1.89 rustdct | 23.0 ns, 1.89 rustdct |
| 48 | 49.9 ns, 0.578 FFTW | 39.1 ns, 0.864 vDSP | 42.0 ns, 0.304 rustdct | 38.1 ns, 0.359 rustdct |
| 64 | 79.2 ns, 1.03 FFTW | 53.7 ns, 1.32 vDSP | 65.7 ns, 0.927 rustdct | 46.1 ns, 0.644 rustdct |
| 255 | 400 ns, 0.595 FFTW | 314 ns, 0.496 FFTW | 350 ns, 0.427 rustdct | 288 ns, 0.479 rustdct |
| 256 | 292 ns, 0.852 FFTW | 174 ns, 1.35 vDSP | 245 ns, 0.651 rustdct | 151 ns, 0.392 rustdct |
| 1000 | 1.48 µs, 0.959 FFTW | 770 ns, 0.474 FFTW | 1.19 µs, 0.192 rustdct | 655 ns, 0.135 rustdct |
| 1023 | 3.12 µs, 0.742 FFTW | 2.44 µs, 0.617 FFTW | 2.82 µs, 0.547 rustdct | 2.30 µs, 0.644 rustdct |
| 1024 | 1.27 µs, 0.849 FFTW | 708 ns, 1.28 vDSP | 1.04 µs, 0.539 rustdct | 577 ns, 0.298 rustdct |
| 3000 | 5.07 µs, 0.966 FFTW | 2.66 µs, 0.516 FFTW | 4.45 µs, 0.233 rustdct | 2.24 µs, 0.157 rustdct |
| 4096 | 6.39 µs, 0.914 FFTW | 3.52 µs, 1.29 vDSP | 5.39 µs, 0.573 rustdct | 3.04 µs, 0.328 rustdct |
| 10,000 | 17.5 µs, 0.805 FFTW | 9.47 µs, 0.490 FFTW | 15.4 µs, 0.232 rustdct | 8.37 µs, 0.187 rustdct |
| 16,384 | 29.0 µs, 0.794 FFTW | 15.7 µs, 1.40 vDSP | 25.7 µs, 0.485 rustdct | 14.0 µs, 0.315 rustdct |
| 65,536 | 126 µs, 0.640 FFTW | 72.1 µs, 0.577 vDSP | 118 µs, 0.440 rustdct | 66.3 µs, 0.319 rustdct |
| 1,048,576 | 2.72 ms, 0.555 FFTW | 1.60 ms, 0.569 vDSP | 2.62 ms, 0.440 rustdct | 1.54 ms, 0.359 rustdct |

DCT-IV:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 | 4.79 ns, 0.262 FFTW | 3.35 ns, 0.186 FFTW | 7.26 ns, 0.441 rustdct | 6.79 ns, 0.364 rustdct |
| 16 | 18.4 ns, 0.731 FFTW | 18.5 ns, 1.39 vDSP | 25.9 ns, 1.22 rustdct | 26.5 ns, 1.22 rustdct |
| 48 | 46.1 ns, 0.486 FFTW | 42.2 ns, 0.849 vDSP | 49.1 ns, 0.531 rustdct | 43.5 ns, 0.597 rustdct |
| 64 | 79.7 ns, 1.06 FFTW | 61.4 ns, 1.52 vDSP | 74.0 ns, 0.836 rustdct | 55.6 ns, 0.624 rustdct |
| 255 | 730 ns, 1.08 FFTW | 665 ns, 0.955 FFTW | 685 ns, 0.990 rustdct | 646 ns, 1.22 rustdct |
| 256 | 299 ns, 1.02 FFTW | 204 ns, 1.57 vDSP | 282 ns, 0.652 rustdct | 192 ns, 0.453 rustdct |
| 1000 | 1.39 µs, 0.892 FFTW | 875 ns, 0.549 FFTW | 1.29 µs, 0.204 rustdct | 829 ns, 0.155 rustdct |
| 1023 | 4.46 µs, 0.990 FFTW | 3.90 µs, 0.910 FFTW | 4.18 µs, 0.904 rustdct | 3.77 µs, 1.15 rustdct |
| 1024 | 1.26 µs, 0.869 FFTW | 815 ns, 1.50 vDSP | 1.14 µs, 0.531 rustdct | 761 ns, 0.363 rustdct |
| 3000 | 4.59 µs, 0.913 FFTW | 2.76 µs, 0.557 FFTW | 4.62 µs, 0.234 rustdct | 2.65 µs, 0.167 rustdct |
| 4096 | 6.22 µs, 0.924 FFTW | 3.77 µs, 1.46 vDSP | 5.68 µs, 0.554 rustdct | 3.57 µs, 0.360 rustdct |
| 10,000 | 16.1 µs, 0.731 FFTW | 9.95 µs, 0.517 FFTW | 16.1 µs, 0.187 rustdct | 9.46 µs, 0.141 rustdct |
| 16,384 | 27.4 µs, 0.714 FFTW | 15.9 µs, 1.28 vDSP | 27.2 µs, 0.489 rustdct | 16.1 µs, 0.308 rustdct |
| 65,536 | 123 µs, 0.681 FFTW | 68.7 µs, 0.503 vDSP | 123 µs, 0.440 rustdct | 70.1 µs, 0.318 rustdct |
| 1,048,576 | 2.63 ms, 0.558 FFTW | 1.45 ms, 0.506 vDSP | 2.64 ms, 0.424 rustdct | 1.46 ms, 0.323 rustdct |

DST-I:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 7 | 8.34 ns, 0.354 FFTW | 6.37 ns, 0.227 FFTW | 10.9 ns, 0.458 rustdct | 8.98 ns, 0.387 rustdct |
| 15 | 15.8 ns, 0.403 FFTW | 14.1 ns, 0.340 FFTW | 19.5 ns, 0.196 rustdct | 14.6 ns, 0.145 rustdct |
| 63 | 100 ns, 0.868 FFTW | 85.1 ns, 0.731 FFTW | 91.9 ns, 0.428 rustdct | 80.9 ns, 0.649 rustdct |
| 255 | 398 ns, 1.02 FFTW | 324 ns, 0.864 FFTW | 362 ns, 0.359 rustdct | 307 ns, 0.523 rustdct |
| 1023 | 1.62 µs, 0.959 FFTW | 1.19 µs, 0.743 FFTW | 1.44 µs, 0.271 rustdct | 1.09 µs, 0.393 rustdct |
| 4095 | 8.20 µs, 1.09 FFTW | 4.77 µs, 0.695 FFTW | 7.30 µs, 0.277 rustdct | 4.29 µs, 0.296 rustdct |
| 16,383 | 35.4 µs, 0.975 FFTW | 23.3 µs, 0.760 FFTW | 33.1 µs, 0.265 rustdct | 21.2 µs, 0.314 rustdct |
| 65,535 | 156 µs, 0.879 FFTW | 97.5 µs, 0.658 FFTW | 152 µs, 0.268 rustdct | 91.9 µs, 0.300 rustdct |
| 64 | 146 ns, 0.641 FFTW | 95.0 ns, 0.470 FFTW | 122 ns, 0.451 rustdct | 83.3 ns, 0.413 rustdct |
| 1000 | 2.61 µs, 0.591 FFTW | 1.48 µs, 0.343 FFTW | 2.32 µs, 0.260 rustdct | 1.34 µs, 0.195 rustdct |
| 1024 | 4.73 µs, 0.563 FFTW | 3.05 µs, 0.381 FFTW | 4.17 µs, 0.184 rustdct | 2.85 µs, 0.149 rustdct |
| 4096 | 25.8 µs, 0.282 FFTW | 15.3 µs, 0.180 FFTW | 24.3 µs, 0.222 rustdct | 14.6 µs, 0.167 rustdct |

DST-II:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 | 4.77 ns, 0.293 FFTW | 3.27 ns, 0.207 FFTW | 7.28 ns, 1.00 rustdct | 6.67 ns, 0.900 rustdct |
| 16 | 20.3 ns, 0.752 FFTW | 17.1 ns, 0.754 FFTW | 23.9 ns, 2.08 rustdct | 23.5 ns, 1.97 rustdct |
| 48 | 50.6 ns, 0.600 FFTW | 41.7 ns, 0.587 FFTW | 41.8 ns, 0.373 rustdct | 39.0 ns, 0.412 rustdct |
| 64 | 68.1 ns, 0.914 FFTW | 55.0 ns, 0.833 FFTW | 55.5 ns, 0.645 rustdct | 45.5 ns, 0.571 rustdct |
| 255 | 392 ns, 0.570 FFTW | 304 ns, 0.486 FFTW | 339 ns, 0.493 rustdct | 278 ns, 0.530 rustdct |
| 256 | 271 ns, 0.792 FFTW | 174 ns, 0.509 FFTW | 218 ns, 0.521 rustdct | 155 ns, 0.443 rustdct |
| 1000 | 1.35 µs, 0.874 FFTW | 752 ns, 0.465 FFTW | 1.13 µs, 0.195 rustdct | 641 ns, 0.140 rustdct |
| 1023 | 3.17 µs, 0.761 FFTW | 2.46 µs, 0.620 FFTW | 2.87 µs, 0.623 rustdct | 2.32 µs, 0.686 rustdct |
| 1024 | 1.19 µs, 0.803 FFTW | 691 ns, 0.482 FFTW | 956 ns, 0.460 rustdct | 569 ns, 0.345 rustdct |
| 3000 | 5.17 µs, 0.948 FFTW | 2.74 µs, 0.538 FFTW | 4.32 µs, 0.238 rustdct | 2.14 µs, 0.161 rustdct |
| 4096 | 6.30 µs, 0.899 FFTW | 4.38 µs, 0.693 FFTW | 5.61 µs, 0.555 rustdct | 3.11 µs, 0.406 rustdct |
| 10,000 | 18.4 µs, 0.714 FFTW | 9.74 µs, 0.507 FFTW | 16.6 µs, 0.269 rustdct | 8.63 µs, 0.210 rustdct |
| 16,384 | 28.8 µs, 0.702 FFTW | 15.0 µs, 0.488 FFTW | 27.0 µs, 0.552 rustdct | 13.8 µs, 0.388 rustdct |
| 65,536 | 148 µs, 0.673 FFTW | 71.1 µs, 0.445 FFTW | 141 µs, 0.617 rustdct | 66.7 µs, 0.411 rustdct |
| 1,048,576 | 3.20 ms, 0.599 FFTW | 1.74 ms, 0.447 FFTW | 3.15 ms, 0.678 rustdct | 1.68 ms, 0.525 rustdct |

DST-III:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 | 4.77 ns, 0.295 FFTW | 3.27 ns, 0.210 FFTW | 7.29 ns, 0.910 rustdct | 6.67 ns, 0.831 rustdct |
| 16 | 19.6 ns, 0.838 FFTW | 15.7 ns, 0.697 FFTW | 22.9 ns, 1.99 rustdct | 24.6 ns, 2.12 rustdct |
| 48 | 49.2 ns, 0.596 FFTW | 39.1 ns, 0.527 FFTW | 41.6 ns, 0.318 rustdct | 39.3 ns, 0.347 rustdct |
| 64 | 79.6 ns, 1.03 FFTW | 54.5 ns, 0.791 FFTW | 66.4 ns, 0.773 rustdct | 46.4 ns, 0.550 rustdct |
| 255 | 406 ns, 0.586 FFTW | 315 ns, 0.503 FFTW | 354 ns, 0.440 rustdct | 288 ns, 0.462 rustdct |
| 256 | 294 ns, 0.853 FFTW | 173 ns, 0.500 FFTW | 247 ns, 0.592 rustdct | 151 ns, 0.363 rustdct |
| 1000 | 1.40 µs, 0.907 FFTW | 764 ns, 0.470 FFTW | 1.19 µs, 0.195 rustdct | 656 ns, 0.132 rustdct |
| 1023 | 3.18 µs, 0.767 FFTW | 2.46 µs, 0.607 FFTW | 2.87 µs, 0.563 rustdct | 2.31 µs, 0.606 rustdct |
| 1024 | 1.25 µs, 0.834 FFTW | 720 ns, 0.499 FFTW | 1.03 µs, 0.499 rustdct | 577 ns, 0.284 rustdct |
| 3000 | 4.95 µs, 0.947 FFTW | 2.69 µs, 0.526 FFTW | 4.45 µs, 0.237 rustdct | 2.27 µs, 0.157 rustdct |
| 4096 | 6.44 µs, 0.916 FFTW | 3.48 µs, 0.557 FFTW | 5.38 µs, 0.534 rustdct | 3.01 µs, 0.312 rustdct |
| 10,000 | 17.6 µs, 0.806 FFTW | 9.15 µs, 0.481 FFTW | 15.4 µs, 0.236 rustdct | 8.23 µs, 0.187 rustdct |
| 16,384 | 29.1 µs, 0.797 FFTW | 15.8 µs, 0.508 FFTW | 25.7 µs, 0.459 rustdct | 14.2 µs, 0.307 rustdct |
| 65,536 | 126 µs, 0.485 FFTW | 71.8 µs, 0.450 FFTW | 118 µs, 0.419 rustdct | 66.8 µs, 0.309 rustdct |
| 1,048,576 | 2.71 ms, 0.557 FFTW | 1.60 ms, 0.408 FFTW | 2.58 ms, 0.417 rustdct | 1.54 ms, 0.346 rustdct |

DST-IV:

| n | `transform_to` `f64` | `f32` | `transform_with_scratch` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 | 4.81 ns, 0.268 FFTW | 3.32 ns, 0.185 FFTW | 7.32 ns, 0.446 rustdct | 6.74 ns, 0.358 rustdct |
| 16 | 18.5 ns, 0.732 FFTW | 18.6 ns, 0.770 FFTW | 25.4 ns, 1.20 rustdct | 26.7 ns, 1.15 rustdct |
| 48 | 45.8 ns, 0.490 FFTW | 40.5 ns, 0.500 FFTW | 48.3 ns, 0.518 rustdct | 43.6 ns, 0.583 rustdct |
| 64 | 72.5 ns, 0.974 FFTW | 60.5 ns, 0.906 FFTW | 70.1 ns, 0.774 rustdct | 54.6 ns, 0.606 rustdct |
| 255 | 729 ns, 0.964 FFTW | 662 ns, 0.935 FFTW | 680 ns, 0.994 rustdct | 648 ns, 1.23 rustdct |
| 256 | 283 ns, 0.966 FFTW | 187 ns, 0.642 FFTW | 268 ns, 0.617 rustdct | 178 ns, 0.414 rustdct |
| 1000 | 1.32 µs, 0.838 FFTW | 811 ns, 0.509 FFTW | 1.23 µs, 0.192 rustdct | 765 ns, 0.149 rustdct |
| 1023 | 4.47 µs, 0.987 FFTW | 3.93 µs, 0.920 FFTW | 4.22 µs, 0.912 rustdct | 3.80 µs, 1.11 rustdct |
| 1024 | 1.18 µs, 0.815 FFTW | 742 ns, 0.553 FFTW | 1.07 µs, 0.499 rustdct | 696 ns, 0.331 rustdct |
| 3000 | 4.49 µs, 0.888 FFTW | 2.59 µs, 0.525 FFTW | 4.67 µs, 0.225 rustdct | 2.49 µs, 0.155 rustdct |
| 4096 | 6.05 µs, 0.912 FFTW | 3.75 µs, 0.596 FFTW | 5.56 µs, 0.539 rustdct | 3.06 µs, 0.307 rustdct |
| 10,000 | 16.0 µs, 0.720 FFTW | 9.44 µs, 0.488 FFTW | 16.0 µs, 0.177 rustdct | 8.96 µs, 0.133 rustdct |
| 16,384 | 27.2 µs, 0.702 FFTW | 14.1 µs, 0.474 FFTW | 27.1 µs, 0.491 rustdct | 14.2 µs, 0.298 rustdct |
| 65,536 | 123 µs, 0.667 FFTW | 65.0 µs, 0.458 FFTW | 124 µs, 0.450 rustdct | 66.3 µs, 0.301 rustdct |
| 1,048,576 | 2.60 ms, 0.563 FFTW | 1.39 ms, 0.353 FFTW | 2.64 ms, 0.430 rustdct | 1.40 ms, 0.312 rustdct |

Planning, `Dct::new(n, DctKind::II)`:

| n | `f64` | `f32` |
| ---: | --- | --- |
| 16 | 280 ns, 2.31 rustdct | 292 ns, 2.30 rustdct |
| 1000 | 3.72 µs, 0.475 rustdct | 2.90 µs, 0.384 rustdct |
| 1024 | 3.77 µs, 1.81 rustdct | 2.85 µs, 1.38 rustdct |
| 65,536 | 148 µs, 1.58 rustdct | 56.7 µs, 0.604 rustdct |

### Two and three dimensions

`Fft2::forward` and `inverse` in place, against FFTW's `plan_dft_2d` in place and `vDSP_fft2d_zip`:

| rows × cols | `forward` `f64` | `f32` | `inverse` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 16 × 16 | 203 ns, 0.651 FFTW | 120 ns, 1.27 vDSP |  |  |
| 64 × 64 | 5.58 µs, 0.665 FFTW | 2.95 µs, 0.938 vDSP | 6.03 µs, 0.681 FFTW | 3.17 µs, 1.00 vDSP |
| 128 × 128 | 34.0 µs, 0.595 FFTW | 15.6 µs, 1.10 vDSP |  |  |
| 256 × 256 | 159 µs, 0.620 FFTW | 81.5 µs, 0.976 vDSP |  |  |
| 512 × 512 | 674 µs, 0.590 FFTW | 370 µs, 0.774 vDSP |  |  |
| 1024 × 1024 | 6.61 ms, 1.12 FFTW | 4.29 ms, 1.83 vDSP | 6.85 ms, 1.24 FFTW | 4.42 ms, 1.92 vDSP |
| 2048 × 2048 | 28.8 ms, 0.949 FFTW | 24.0 ms, 2.10 vDSP |  |  |
| 4096 × 4096 | 121 ms, 0.839 FFTW | 110 ms, 2.15 vDSP |  |  |
| 480 × 640 | 965 µs, 0.680 FFTW | 489 µs, 0.381 FFTW | 1.03 ms, 0.713 FFTW | 511 µs, 0.385 FFTW |
| 1000 × 1000 | 3.75 ms, 0.624 FFTW | 2.19 ms, 0.438 FFTW |  |  |
| 4096 × 64 | 786 µs, 0.670 FFTW | 423 µs, 0.884 vDSP |  |  |
| 64 × 4096 | 741 µs, 0.642 FFTW | 481 µs, 0.990 vDSP |  |  |

`Fft3::forward` and `inverse` in place, against FFTW's `plan_dft_3d`:

| shape | `forward` `f64` | `f32` | `inverse` `f64` | `f32` |
| ---: | --- | --- | --- | --- |
| 8 × 8 × 8 | 465 ns, 0.683 FFTW | 312 ns, 0.458 FFTW |  |  |
| 16 × 16 × 16 | 4.90 µs, 0.654 FFTW | 2.86 µs, 0.383 FFTW | 5.03 µs, 0.670 FFTW | 2.95 µs, 0.402 FFTW |
| 32 × 32 × 32 | 81.9 µs, 0.744 FFTW | 37.4 µs, 0.376 FFTW |  |  |
| 64 × 64 × 64 | 752 µs, 0.719 FFTW | 381 µs, 0.358 FFTW | 775 µs, 0.724 FFTW | 387 µs, 0.369 FFTW |
| 128 × 128 × 128 | 9.30 ms, 0.700 FFTW | 4.26 ms, 0.336 FFTW |  |  |
| 256 × 256 × 256 | 86.6 ms, 0.755 FFTW | 43.1 ms, 0.419 FFTW |  |  |
| 100 × 100 × 100 | 3.66 ms, 0.668 FFTW | 2.17 ms, 0.421 FFTW |  |  |
| 8 × 256 × 256 | 1.82 ms, 0.726 FFTW | 782 µs, 0.340 FFTW |  |  |
| 256 × 256 × 8 | 1.90 ms, 0.767 FFTW | 785 µs, 0.339 FFTW |  |  |

Planning, `Fft2::new` (FFTW's `ESTIMATE` plan of two dimensions):

| rows × cols | `f64` | `f32` |
| ---: | --- | --- |
| 64 × 64 | 880 ns, 0.162 FFTW | 945 ns, 0.178 FFTW |
| 1024 × 1024 | 4.70 µs, 0.0441 FFTW | 3.78 µs, 0.0353 FFTW |

### Convolution

`convolve_into(&a, &b, &mut out)` of random operands, called again and again, so that its plans come from the cache as a repeated call finds them:

| len(a) × len(b) | `f32` | `f64` | `Complex<f32>` | `Complex<f64>` |
| ---: | --- | --- | --- | --- |
| 16 × 3 | 18.1 ns, 2.12 vDSP | 18.1 ns, 1.90 vDSP | 20.2 ns, 0.598 vDSP | 23.4 ns, 1.01 vDSP |
| 16 × 8 | 23.6 ns, 2.47 vDSP | 36.0 ns, 2.78 vDSP | 41.9 ns, 0.831 rustfft | 64.5 ns, 0.785 vDSP |
| 16 × 16 | 29.0 ns, 1.75 vDSP | 43.1 ns, 1.86 vDSP | 59.7 ns, 0.986 rustfft | 117 ns, 0.975 rustfft |
| 100 × 3 | 21.8 ns, 1.44 vDSP | 28.0 ns, 0.784 vDSP | 36.8 ns, 0.203 vDSP | 64.6 ns, 0.545 vDSP |
| 100 × 8 | 54.8 ns, 2.27 vDSP | 88.3 ns, 2.37 vDSP | 136 ns, 0.688 rustfft | 223 ns, 0.621 vDSP |
| 100 × 16 | 70.7 ns, 1.92 vDSP | 129 ns, 1.74 vDSP | 199 ns, 0.707 rustfft | 535 ns, 1.01 rustfft |
| 100 × 64 | 228 ns, 1.17 vDSP | 430 ns, 1.14 vDSP | 909 ns, 1.18 rustfft | 935 ns, 0.899 FFTW |
| 1000 × 3 | 77.6 ns, 0.788 vDSP | 126 ns, 0.443 vDSP | 274 ns, 0.189 rustfft | 536 ns, 0.470 vDSP |
| 1000 × 16 | 344 ns, 1.12 vDSP | 670 ns, 1.07 vDSP | 1.46 µs, 0.585 rustfft | 2.89 µs, 0.654 rustfft |
| 1000 × 64 | 1.46 µs, 1.19 vDSP | 2.69 µs, 1.13 vDSP | 2.87 µs, 0.748 rustfft | 5.22 µs, 0.799 rustfft |
| 1000 × 128 | 2.83 µs, 1.03 vDSP | 5.35 µs, 1.45 FFTW | 3.48 µs, 0.804 rustfft | 7.19 µs, 0.929 rustfft |
| 1000 × 256 | 2.33 µs, 0.648 realfft | 4.17 µs, 0.871 FFTW | 4.06 µs, 0.725 rustfft | 7.97 µs, 0.802 rustfft |
| 1000 × 1000 | 3.73 µs, 0.687 realfft | 7.09 µs, 0.892 FFTW | 6.87 µs, 0.801 rustfft | 14.6 µs, 0.873 rustfft |
| 10,000 × 3 | 693 ns, 0.733 vDSP | 1.09 µs, 0.393 vDSP | 2.62 µs, 0.183 rustfft | 5.22 µs, 0.437 vDSP |
| 10,000 × 16 | 3.07 µs, 1.01 vDSP | 6.05 µs, 0.982 vDSP | 12.0 µs, 0.496 rustfft | 25.8 µs, 0.606 rustfft |
| 10,000 × 128 | 17.3 µs, 0.820 realfft | 32.8 µs, 1.11 FFTW | 30.0 µs, 0.876 rustfft | 52.4 µs, 0.869 rustfft |
| 10,000 × 1000 | 24.7 µs, 0.710 realfft | 43.6 µs, 0.828 FFTW | 44.2 µs, 0.703 rustfft | 74.9 µs, 0.660 rustfft |
| 10,000 × 10,000 | 44.1 µs, 0.412 FFTW | 89.9 µs, 0.804 FFTW | 84.8 µs, 0.405 rustfft | 162 µs, 0.683 FFTW |
| 100,000 × 3 | 8.06 µs, 1.32 vDSP | 11.3 µs, 0.408 vDSP | 25.9 µs, 0.190 rustfft | 52.0 µs, 0.411 vDSP |
| 100,000 × 128 | 153 µs, 0.800 realfft | 268 µs, 0.937 FFTW | 245 µs, 0.753 rustfft | 445 µs, 0.790 rustfft |
| 100,000 × 1000 | 201 µs, 0.801 realfft | 337 µs, 0.863 FFTW | 328 µs, 0.732 rustfft | 627 µs, 0.764 rustfft |
| 100,000 × 100,000 | 564 µs, 0.521 realfft | 1.19 ms, 0.813 FFTW | 1.20 ms, 0.503 rustfft | 2.28 ms, 0.691 FFTW |
| 1,000,000 × 3 | 74.5 µs, 1.21 vDSP | 138 µs, 0.489 vDSP | 263 µs, 0.191 rustfft | 519 µs, 0.406 vDSP |
| 1,000,000 × 32 | 574 µs, 0.965 vDSP | 1.12 ms, 0.876 vDSP | 2.27 ms, 0.848 rustfft | 4.91 ms, 1.06 rustfft |
| 1,000,000 × 128 | 1.42 ms, 0.751 realfft | 2.60 ms, 0.902 FFTW | 3.17 ms, 0.959 rustfft | 4.60 ms, 0.829 rustfft |
| 1,000,000 × 1000 | 1.70 ms, 0.702 realfft | 3.44 ms, 0.900 FFTW | 2.98 ms, 0.646 rustfft | 6.03 ms, 0.752 rustfft |
| 1,000,000 × 100,000 | 3.13 ms, 0.568 realfft | 7.15 ms, 0.756 realfft | 6.82 ms, 0.677 rustfft | 12.7 ms, 0.680 rustfft |

`convolve`, which returns a new vector, at the sizes where its allocation shows:

| len(a) × len(b) | `f32` | `f64` | `Complex<f32>` | `Complex<f64>` |
| ---: | --- | --- | --- | --- |
| 16 × 3 | 27.5 ns, 3.22 vDSP | 26.7 ns, 2.82 vDSP | 30.6 ns, 0.909 vDSP | 39.5 ns, 1.70 vDSP |
| 16 × 8 | 31.8 ns, 3.33 vDSP | 47.6 ns, 3.67 vDSP | 50.2 ns, 0.997 rustfft | 79.2 ns, 0.964 vDSP |
| 100 × 3 | 39.4 ns, 2.60 vDSP | 45.2 ns, 1.27 vDSP | 58.6 ns, 0.323 vDSP | 80.6 ns, 0.680 vDSP |
| 100 × 8 | 67.8 ns, 2.80 vDSP | 106 ns, 2.86 vDSP | 154 ns, 0.779 rustfft | 237 ns, 0.660 vDSP |
| 100 × 64 | 248 ns, 1.27 vDSP | 443 ns, 1.18 vDSP | 924 ns, 1.20 rustfft | 949 ns, 0.913 FFTW |
| 1000 × 3 | 90.4 ns, 0.918 vDSP | 149 ns, 0.525 vDSP | 290 ns, 0.200 rustfft | 553 ns, 0.485 vDSP |
| 1000 × 128 | 2.84 µs, 1.03 vDSP | 5.38 µs, 1.45 FFTW | 3.51 µs, 0.811 rustfft | 7.29 µs, 0.942 rustfft |
| 10,000 × 128 | 17.3 µs, 0.820 realfft | 33.0 µs, 1.12 FFTW | 29.7 µs, 0.868 rustfft | 52.6 µs, 0.873 rustfft |
| 1,000,000 × 3 | 75.4 µs, 1.23 vDSP | 138 µs, 0.489 vDSP | 261 µs, 0.189 rustfft | 521 µs, 0.408 vDSP |
| 1,000,000 × 128 | 1.41 ms, 0.744 realfft | 2.60 ms, 0.902 FFTW | 3.19 ms, 0.966 rustfft | 4.61 ms, 0.829 rustfft |

`correlate_into`:

| len(a) × len(b) | `f32` | `f64` | `Complex<f32>` | `Complex<f64>` |
| ---: | --- | --- | --- | --- |
| 100 × 8 | 54.1 ns, 2.37 vDSP | 89.0 ns, 2.30 vDSP | 133 ns, 0.665 rustfft | 224 ns, 0.771 vDSP |
| 1000 × 16 | 344 ns, 1.12 vDSP | 668 ns, 1.05 vDSP | 1.50 µs, 0.601 rustfft | 2.94 µs, 0.634 rustfft |
| 1000 × 1000 | 3.69 µs, 0.671 realfft | 7.03 µs, 0.877 FFTW | 7.49 µs, 0.853 rustfft | 14.6 µs, 0.853 rustfft |
| 10,000 × 64 | 11.5 µs, 0.990 vDSP | 22.4 µs, 0.999 vDSP | 25.1 µs, 0.826 rustfft | 45.2 µs, 0.839 rustfft |
| 10,000 × 1000 | 25.2 µs, 0.717 realfft | 43.6 µs, 0.817 FFTW | 44.0 µs, 0.694 rustfft | 75.0 µs, 0.658 rustfft |
| 100,000 × 32 | 57.1 µs, 0.940 vDSP | 116 µs, 0.935 vDSP | 227 µs, 0.845 rustfft | 485 µs, 1.04 rustfft |
| 100,000 × 10,000 | 258 µs, 0.613 realfft | 490 µs, 0.683 realfft | 493 µs, 0.629 rustfft | 941 µs, 0.639 rustfft |
| 1,000,000 × 128 | 1.45 ms, 0.761 realfft | 2.59 ms, 0.908 FFTW | 3.18 ms, 0.998 rustfft | 4.59 ms, 0.831 rustfft |

`circular_convolve`:

| n | `f32` | `f64` | `Complex<f32>` | `Complex<f64>` |
| ---: | --- | --- | --- | --- |
| 16 | 54.0 ns, 1.49 FFTW | 58.5 ns, 1.49 FFTW | 64.7 ns, 2.06 rustfft | 92.4 ns, 1.55 rustfft |
| 100 | 278 ns, 1.12 FFTW | 551 ns, 1.78 FFTW | 413 ns, 0.815 rustfft | 580 ns, 0.929 FFTW |
| 128 | 443 ns, 1.64 realfft | 745 ns, 2.28 FFTW | 439 ns, 1.18 rustfft | 679 ns, 1.01 rustfft |
| 1000 | 1.89 µs, 0.539 FFTW | 3.33 µs, 0.849 FFTW | 3.27 µs, 0.406 FFTW | 6.19 µs, 0.688 FFTW |
| 1009 | 4.01 µs, 0.140 FFTW | 7.52 µs, 0.263 FFTW | 8.07 µs, 0.261 rustfft | 15.5 µs, 0.410 rustfft |
| 1024 | 1.65 µs, 0.668 realfft | 2.82 µs, 0.799 FFTW | 2.95 µs, 0.694 rustfft | 6.20 µs, 0.838 rustfft |
| 4096 | 8.58 µs, 0.733 realfft | 14.2 µs, 0.799 FFTW | 17.6 µs, 0.849 rustfft | 27.9 µs, 0.712 rustfft |
| 10,007 | 46.1 µs, 0.141 realfft | 87.6 µs, 0.188 FFTW | 83.7 µs, 0.265 rustfft | 160 µs, 0.317 FFTW |
| 65,536 | 159 µs, 0.642 realfft | 334 µs, 0.798 realfft | 304 µs, 0.614 rustfft | 611 µs, 0.646 FFTW |
| 100,000 | 263 µs, 0.407 FFTW | 506 µs, 0.719 FFTW | 469 µs, 0.329 FFTW | 943 µs, 0.608 FFTW |
| 1,048,576 | 3.51 ms, 0.724 realfft | 8.46 ms, 0.977 realfft | 6.70 ms, 0.647 rustfft | 15.5 ms, 0.868 rustfft |

### Where the references are faster

Every loss of the cases above (a ratio above 1.05 against the fastest reference that does the same work), by cause:

- **vDSP in `f32`**, in every transform it has: the complex transform of the powers of two from 64 (1.07 to 2.85: 2.22 at 1024, 2.85 at 4096, 1.80 at 65,536, 0.987 at 2²⁰), of `f·2^k` from 320 (1.22 to 3.15) and of 36 and 60 (1.09, 1.07), its inverse (1.10 to 3.21) and `forward_to` (1.23 to 3.04), batches of 1024 (2.19); the real transform of the powers of two from 64 (`forward` 1.11 to 3.37, `inverse` 1.27 to 3.58) and of `f·2^k` from 640 (1.42 to 3.45; 80 and 96 1.06 and 1.10); DCT-II, III and IV from 16 (1.12 to 2.06); and `Fft2` (1.10 to 2.15, the inverse 1.92). vDSP's `f32` kernels run at about five times its own `f64` speed (`vDSP_DFT_Execute` of 1024: 278 ns in `f32`, 1.37 µs in `f64`), where twice the lanes of a NEON vector account for two; `a` runs `f32` at about twice its `f64` speed (617 ns and 1.26 µs) and is as fast as vDSP or faster in `f64`. That points to Apple's matrix hardware (AMX or SME on the M4), which no stable or portable Rust interface reaches, or to an algorithm far from FFTW's family; which of the two was not determined.
- **Short complex transforms**: the powers of two 2, 8 and 32 (`f64` 1.31, 1.11 and 1.15; `f32` 1.88 at 2, 3.12 ns against 1.67 ns, and 1.15 at 32), the smooth 3, 5, 6, 7, 10, 12, 24 and 36 (1.06 to 1.32), batches of 2, 3, 6, 10, 12 and 15 against rustfft (1.09 to 1.49) and `forward_to` of 16 in `f32` (1.23): a call dispatches through the plan (its algorithm, a kernel by a function pointer) where FFTW's and rustfft's codelets are called straight, a few nanoseconds that are most of a transform this short; 32 and 36 take two passes where they have one butterfly.
- **`f64` against vDSP out of place and in the inverse**: `forward_to` of 64, 256 and 4096 (1.12, 1.06, 1.07), which takes its working memory itself, from the heap beyond 512 values, and `inverse_with_scratch` of 4096 (1.09), not investigated.
- **Short real transforms**: `forward` and `inverse` of 8 (1.07 to 1.22), the inverse of 12 (1.15, 1.17) and in `f64` of 64 and 80 (1.16, 1.09); `forward_with_scratch` and `inverse_with_scratch` of 15 and 64 (1.06 to 1.61); batches of 15 and, in `f64`, of 64 (1.37 to 1.78). FFTW has straight-line real codelets (`r2cf`, `r2cb`) for these; `a` runs an odd length up to 15 as the complex butterfly of `n` on the reals, about twice their arithmetic, and pays a plan's fixed costs at 8 to 80.
- **Odd real lengths with small factors**, 15, 45, 135 and 243 (`forward` 1.09 to 1.59, `inverse` 1.16 to 1.75, against FFTW): FFTW's odd real codelets (`hc2hc`) against the split of the residues, whose sequences go in pairs through complex transforms and one pass combines them; from 625 on `a` is faster.
- **The `f64` real transform of 65,536** (`forward` 1.71, `forward_with_scratch` 1.88 against realfft; 217 µs where the inverse takes 98.2 µs): its last complex pass, of radix 8 at the half length 32,768, is fused with the pass over the pairs, which at this length runs slower than the two separate passes. Turning the fusion off for those lengths is a known follow-up, not in this release.
- **Short and odd cosine and sine transforms**: in place against rustdct, types II and III of 16 (1.89 to 2.12), type IV of 16 (1.15 to 1.22) and the `f32` DCT-IV and DST-IV of 255 and 1023 (1.11 to 1.23); out of place in `f64` against FFTW, DCT-II and III of 8 (1.23, 1.24), DCT-IV of 64 and 255 (1.06, 1.08), DCT-I of 257 and 4097 and DST-I of 4095 (1.09 to 1.13), DCT-II of 65,536 (1.15). Types II to IV of 16 go through Makhoul's step and the real transform of 16 with their fixed costs, where rustdct has straight-line 16-point codelets; 8 is summed from the matrix against FFTW's `e10`/`e01` codelets; the odd 255 and 1023 run on the odd real transforms (above) and the odd type IV's scalar permutation; DCT-I and DST-I at `2^k ± 1` on the split-radix step, whose gathers are scalar and whose levels are plans of their own; and DCT-II's twiddle pass runs after the real transform's, not fused into it.
- **`Fft2` of 1024 × 1024 in `f64`** (1.12, the inverse 1.24, against FFTW): an array of 16 MiB is beyond the 4 MiB of the passes along the rows, and goes by tiles of columns gathered and scattered value by value.
- **Short convolutions against vDSP's direct sums**: `convolve_into` of 16 × 3 to 16 × 16 (1.75 to 2.78), 100 × 3 to 100 × 64 (1.14 to 2.37) and 1000 × 8 to 1000 × 64 (1.07 to 1.19) in `f32` and `f64`, and of 3 taps over 10⁵ and 10⁶ `f32` values (1.32, 1.21); `correlate_into` of 100 × 8 (2.30, 2.37) and 1000 × 16 in `f32` (1.12); the allocating `convolve` and `correlate` at the same sizes and the complex 16 × 3 and 16 × 16 (1.70 in `Complex<f64>`, 1.12), with the result's allocation on top (`convolve` of 16 × 8 in `f64`: 47.6 ns against 13.0 ns, 3.67). The fixed costs of a call (the model's choice of the path, the handling of the operands, the short ends) and kernel blocks of 32 or 16 outputs computed for a handful of them; for a few taps over long data vDSP's kernel is faster per output. vDSP's operands are padded before the clock.
- **Convolutions near the switch to the transforms**: `f64` 1000 × 128 and 10⁴ × 128 (1.45, 1.11, against FFTW), `Complex<f32>` 100 × 64 (1.18, rustfft) and `Complex<f64>` 10⁶ × 32 (1.06): the model charges half a plan's cost to every call and may not ask the cache whether the plans are kept (the same call must give the same bits), so near the switch it can keep the slower path.
- **Short circular convolutions and correlations** of 16, 100 and 128 values (1.11 to 2.29; `f64` of 128: 745 ns against FFTW's 327 ns): the model takes the direct `n²` path where three transforms on kept plans are faster, and the result is a new vector.
- **Planning**: `Fft::new` of 2²⁰ (3.36 and 4.50 times FFTW's `ESTIMATE`: 1.48 ms and 1.95 ms against 441 µs and 434 µs), of 16 to 4096 against vDSP's interleaved setup (1.20 to 2.24), of Rader's 1009 against rustfft (1.13, 1.38) and of Bluestein's 100,003 in `f32` (1.06); `RealFft::new` of 16 against vDSP (1.69, 2.27) and of 1001 and 1009 against realfft (1.10 to 2.27); `Dct::new` of 16, 1024 and 65,536 against rustdct (1.38 to 2.31). Every root of a plan's tables is correctly rounded (an octant table's entry corrected in double-double, or a product of a coarse and a fine table in double-double), the short lengths summed directly carry matrices of their own, and an odd real length plans the transforms of its split; FFTW, rustfft and realfft build plainer tables.

Not counted, the references that do less or more: rustfft's out-of-place form that overwrites its input takes within 2 % of the time of the form that keeps it at every length timed; and against `FFTW_MEASURE`'s planning, which times candidate plans, `Fft::new` and `RealFft::new` take 0.000257 to 0.0991 of its time.

### Accuracy

Measured with version 0.2.40, before the speed pass. The audit of the speed pass compared every changed path with double-double references on 809 complex lengths up to 2²², 1,221 real lengths, the eight cosine and sine transforms at 283 lengths, the 2-D and 3-D shapes and the convolutions, in `f32` and `f64`: the worst error was 0.097 of the tests' bounds.

The relative L2 error of the forward transform of random data in [−1, 1) against the tests' double-double references, in units of the type's `ε` (`2⁻²³` for `f32`, `2⁻⁵²` for `f64`), the largest over the lengths and the root mean square, next to FFTW's (its `FFTW_ESTIMATE` plans, the same codelets as `FFTW_MEASURE`'s) and rustfft's on the same data. The complex transform over 93 lengths from 2 to 1000003, dense up to 4096 and sparse beyond:

| Type | `a` max | `a` rms | FFTW max | FFTW rms | rustfft max | rustfft rms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `f64` | 3.06 | 1.05 | 2.98 | 1.05 | 5.50 | 1.70 |
| `f32` | 1.87 | 0.89 | 2.62 | 0.98 | 2.28 | 0.92 |

`RealFft` over 24 lengths from 15 to 4096 (powers of two, smooth, odd and primes), and the cosine and sine transforms over 17 lengths from 16 to 1025 (even, odd, `2^k ± 1`):

| Transform | `a`, `f64` max / rms | FFTW, `f64` | `a`, `f32` max / rms | FFTW, `f32` |
| --- | --- | --- | --- | --- |
| `RealFft` | 1.82 / 1.01 | 1.82 / 0.95 | 1.43 / 0.95 | 2.11 / 0.93 |
| DCT-I | 1.66 / 1.13 | 1.21 / 0.86 | 1.28 / 0.90 | 1.19 / 0.83 |
| DCT-II | 1.66 / 1.12 | 1.70 / 0.98 | 1.43 / 0.99 | 1.62 / 0.97 |
| DCT-III | 1.57 / 1.10 | 1.60 / 1.02 | 1.35 / 0.97 | 1.52 / 0.98 |
| DCT-IV | 1.69 / 1.11 | 1.63 / 1.03 | 1.48 / 0.99 | 1.75 / 1.02 |
| DST-I | 1.38 / 0.96 | 1.56 / 0.92 | 1.12 / 0.87 | 1.47 / 0.85 |
| DST-II | 1.57 / 1.13 | 1.67 / 0.99 | 1.36 / 1.01 | 1.55 / 0.96 |
| DST-III | 1.56 / 1.08 | 1.69 / 1.03 | 1.36 / 0.99 | 1.65 / 1.03 |
| DST-IV | 1.66 / 1.11 | 1.63 / 1.04 | 1.41 / 1.00 | 1.57 / 0.95 |

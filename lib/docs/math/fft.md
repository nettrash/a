[A](../../../README.md) › [Math](../../../README.md#math)

# Fourier transforms

`a::math::fft` computes discrete Fourier transforms of `f32`, `f64`, `Complex<f32>` and `Complex<f64>` sequences of any length, the cosine and sine transforms of types I to IV, transforms of two and three dimensions, and convolutions and correlations. A transform is planned once for its length and applied as often as needed; a plan never changes and may be shared between threads. The float type of a plan is `f32` or `f64` (the sealed trait `FftFloat`), and the values are the crate's [`Complex<T>`](complex.md).

- `Fft::<T>::new(n)` plans the complex transform `X[k] = Σ x[j]·e^(−2πi·jk/n)`, forward, and `x[j] = Σ X[k]·e^(2πi·jk/n)`, inverse, for any `n` (0 and 1 give the identity). `forward` and `inverse` work in place, `forward_to` and `inverse_to` from one slice into another of the same length; `forward_with_scratch` and `inverse_with_scratch` work in place with the working memory from the caller, at least `scratch_len()` values, and never allocate. `len` and `is_empty` give the length, and `Debug` the plan the length got: `Fft { n: 1009, algorithm: "Rader(4·4·3·3·7)", isa: Neon, .. }`.
- `RealFft::<T>::new(n)` plans the transform of `n` reals: `forward(input, output)` takes runs of `n` reals and gives for each the `n/2 + 1` values `X[0..=n/2]` that determine the rest (`X[n − k]` is the conjugate of `X[k]`), as FFTW's `r2c` and NumPy's `rfft` do; `inverse(input, output)` takes runs of `n/2 + 1` values and gives `n` reals. `spectrum_len()` is `n/2 + 1` (0 for `n = 0`); `scratch_len`, `forward_with_scratch` and `inverse_with_scratch` are as for `Fft`.
- `Dct::<T>::new(n, kind)` and `Dst::<T>::new(n, kind)` plan the cosine and sine transforms of the types `DctKind::{I, II, III, IV}` and `DstKind::{I, II, III, IV}`, FFTW's `REDFT00`, `REDFT10`, `REDFT01`, `REDFT11` and `RODFT00`, `RODFT10`, `RODFT01`, `RODFT11`, unscaled as FFTW's. `transform` works in place, `transform_to` from one slice into another, and `transform_with_scratch` in place with at least `scratch_len()` complex values of working memory from the caller, without allocating; `len`, `is_empty` and `kind` describe the plan.
- `Fft2::<T>::new(rows, cols)` and `Fft3::<T>::new(n0, n1, n2)` plan the transforms of complex arrays of two and three dimensions stored with the last index varying fastest; `forward` and `inverse` work in place, `shape` gives the shape.
- `convolve(&a, &b)`, `correlate(&a, &b)`, `circular_convolve(&a, &b)` and `circular_correlate(&a, &b)` take two slices of the same type, `f32`, `f64`, `Complex<f32>` or `Complex<f64>` (the sealed trait `Sample`), and return a new vector; they choose between a direct sum and transforms by the lengths, and keep the plans of their transforms for the next calls.
- The sections below give the plans and their working memory, each kind of transform and the convolutions, then the examples; the methods are after the examples, then how the module was tested and the measurements against FFTW, rustfft and realfft, and Apple's Accelerate.

## Plans, batches and working memory

- A plan is immutable, `Send` and `Sync`. `Fft`, `RealFft`, `Dct` and `Dst` are `Clone`: a clone shares the plan's tables, and costs one reference count. `Fft2` and `Fft3` are not `Clone`; they hold one `Fft` per dimension.
- Every method takes a batch: any number of sequences (or arrays) of the planned length one after the other in one slice, each transformed in turn, and panics when the length of the slice is not a multiple of the planned one, or when an output's length differs from its input's (for `RealFft`, when the output does not hold `n/2 + 1` values for each sequence of the input, or `n` for each of `n/2 + 1`). Short sequences of a batch go through the passes of the complex transform together, as many as fit in 256 values a call, when the working memory has room for them; each result is that of its sequence transformed alone, to within rounding.
- Every inverse is unscaled, as FFTW's: `Fft`'s `inverse(forward(x)) = n·x`, `RealFft`'s the same, `Fft2`'s `rows·cols·x` and `Fft3`'s `n0·n1·n2·x`; for the cosine and sine transforms see [below](#cosine-and-sine-transforms).
- `scratch_len()` is the working memory one sequence needs, in complex values (0 for some lengths: those of one pass of the complex transform, and the shortest real transforms, which are summed directly). A `_with_scratch` method panics with "scratch of … values, … needed" when given less, and never allocates; given more, `Fft::inverse_with_scratch`, `Fft::forward_with_scratch` and `RealFft::inverse_with_scratch` let the short sequences of a batch run together. The other methods take their working memory themselves: `Fft`'s `forward`, `inverse`, `forward_to` and `inverse_to` allocate one run's (or the room a batch of short runs takes together); `RealFft`'s and `Dct`'s and `Dst`'s methods take it on the stack up to 512 complex values and from the heap beyond, once for the whole batch, and `RealFft::forward` of a batch of even lengths stages its complex transforms in a buffer of up to 4096 values (one sequence's, when that is longer). `Fft2` and `Fft3` have no `_with_scratch` form: each call allocates a tile of at most 1 MiB (or one column, when a column is longer) and the scratch of its transforms.
- `forward_to`, `inverse_to` and `transform_to` have no `_with_scratch` form.

## The complex transform

- Every length is planned in time `O(n log n)`: by Stockham passes of the radices 2, 4 and 8 for the factors 2, hard-coded butterflies of 3, 5 and 7, and the odd butterfly from tables for the primes 11 to 19; a prime factor above 19 of a composite length by a plan of its own; and a prime length above 19 by Rader's algorithm or by Bluestein's with a smooth inner length, whichever the planner estimates faster. `Debug` shows the choice: `"8·8·4·4"` for 1024, `"8·8·5·5·5·5·5·5"` for 10⁶, `"Rader(4·4·3·3·7)"` for 1009, `"Bluestein(20480: 8·8·8·8·5)"` for 10007, `"2·[Rader(4·5·5)]"` for 202.
- The kernels are NEON on little-endian aarch64, SSE2 or, detected when the plan is made, AVX2 with FMA on x86_64, and portable Rust elsewhere and under Miri; the same algorithm in each, so a plan's results differ between instruction sets only by rounding.
- On an Apple M4 Max, one thread, a forward transform of `Complex<f64>` takes 1.33 µs at 1024 and 4.22 ms at 2²⁰, 0.59 and 0.52 of FFTW's time; of `Complex<f32>` 705 ns and 2.08 ms, 0.34 and 0.31 of FFTW's time and 2.3 and 1.5 times Accelerate's. The tables are [below](#measured).

## Real sequences

- `forward` gives `X[0]` with an imaginary part of exactly 0, and `X[n/2]` too for an even `n`; `inverse` ignores the imaginary parts of `X[0]` and, for an even `n`, of `X[n/2]`, as FFTW's `c2r` does.
- An even length takes the complex transform of `n/2`; an odd one with a factor up to 11 splits by it; a prime, or a length whose factors all exceed 11, takes the complex transform of `n`; odd lengths up to 15 and even ones up to 16 are summed directly. Batches are as for `Fft`.

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

`Fft2` computes `X[k, l] = Σ x[i, j]·e^(−2πi·(ik/rows + jl/cols))` for arrays of `rows × cols` values stored row after row, and `Fft3` the same over three indices: the last, contiguous dimension as one batch of rows, then each other dimension by tiles of adjacent columns gathered into a buffer, so that every pass over the array reads and writes whole cache lines.

## Convolution and correlation

- `convolve(&a, &b)` is the linear convolution, the `len(a) + len(b) − 1` values `c[k] = Σ a[i]·b[k − i]` (none when either is empty), as NumPy's `convolve` gives them in its full mode.
- `correlate(&a, &b)` is the cross-correlation, the `len(a) + len(b) − 1` values `c[k + len(b) − 1] = Σ a[i + k]·conj(b[i])` for the lags `k` from `−(len(b) − 1)` to `len(a) − 1`, NumPy's `correlate` in its full mode: the convolution with `b` reversed and conjugated, without a copy of `b`.
- `circular_convolve(&a, &b)` and `circular_correlate(&a, &b)` take two sequences of the same length `n` (and panic when the lengths differ): `c[k] = Σ a[j]·b[(k − j) mod n]` and `c[k] = Σ a[(j + k) mod n]·conj(b[j])`.
- The path is chosen by a model of the costs of the lengths: a direct sum by SIMD kernels while the shorter operand is short, the product of the transforms of the operands padded to a length `2^a·3^b·5^c·7^d`, or overlap-add when one operand is much longer than the other. With the constants of this release the direct path is taken against a long operand (10⁴ to 10⁶ values) up to a shorter one of 123 to 137 values for `f32`, 99 to 105 for `f64`, 50 to 53 for `Complex<f32>` and 38 to 40 for `Complex<f64>`, and against 1,000 values up to 187, 137, 67 and 47 ([below](#convolution)). A circular convolution is direct, by three transforms of the length `n`, or the linear convolution by transforms of a smooth length of at least `2n − 1` folded modulo `n` (for a prime `n`, whose own transforms would take Bluestein's algorithm).
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

**Stockham passes.** Every length is a sequence of passes, one per factor `r` of `n`, in the Stockham autosort form of decimation in frequency: a pass of radix `r` over sub-transforms of the length `L = r·m` of `s` interleaved sequences reads `x[q + s·(p + k·m)]` and writes `y[q + s·(r·p + j)] = W_L^(j·p) · Σ_k x[q + s·(p + k·m)]·W_r^(j·k)`. After the passes of all the factors (`s` growing from 1, `m` shrinking to 1) the result is in natural order: no bit reversal and no transposition. The passes alternate between the data and one buffer of `n` values; the last pass, which has no twiddles and its own kernel without the loop over `p`, runs in place, so an in-place transform ends in the data and an out-of-place one writes its output without a copy. The factors 2 take radix-8 passes, with one or two radix-4 passes for what is left (a radix-2 pass for a single factor 2); the factors 3, 5 and 7 have hard-coded butterflies, and the primes 11 to 19 the general odd butterfly with its cosines and sines from a table (the sums `x_k ± x_(r−k)` halve its products), each instantiated for its radix so that its loops unroll into registers. The odd factors come after the twos and the largest last. Measured on the M4 and dropped: radix-16 passes (slower at every length: 16 streams at power-of-two strides, and more vectors than registers), and four-step and six-step algorithms with blocks of columns in the caches, about twice as slow as the plain passes at 2²⁰ and 2²² in `f64`.

**Twiddles.** Each pass has its own table of `W_L^(j·p)`, stored expanded for the products as `[wr, wr, −wi, wi]` (or by blocks of SIMD lanes for a first pass that runs its vectors along `p`): a product is one multiplication, one exchange of the parts and one fused multiply-add. The inverse runs the same tables with the conjugate products and `+i` for `−i` in the butterflies, by kernels instantiated for each direction: no conjugation passes. All the tables of a plan come from one table of the roots of its length (each pass's `W_L^e` is `W_n^(e·n/L)`) that holds only the angles of at most π/4: `n/8 + 1` entries for a multiple of 4, `n/4 + 1` for twice an odd number and `n/2 + 1` for an odd `n`. Every root is an entry with its signs and parts exchanged, exactly, the octant found by comparisons and the signs set by bit operations, without a division or a branch. An entry is the system's `sin` and `cos` of the angle carried as a double-double and corrected by its low part, so each factor is as accurate as `sin` and `cos` of an `f64` and the symmetric ones are exact images of each other; `f32` tables are rounded from them.

**Primes.** A prime length up to 19 is one pass. Above 19 it takes Rader's algorithm, the cyclic convolution of length `p − 1` of the input permuted by a primitive root with the transform of the permuted roots (computed once in `f64` and divided by `p − 1`; the inverse is the forward transform written to the reversed indices), or Bluestein's, the convolution with the chirp `W_2n^(k²)` (its angle exact from `k² mod 2n`) by transforms of a smooth length `m` in `[2n − 1, 2(2n − 1))` of the form `2^a·3^b·5^c·7^d`, the cheapest by the estimate rather than a power of two. The planner estimates the time of each from the passes each inner length needs, measured per radix on the M4: Rader's for 101 and 1009 (`p − 1 = 2²·5²` and `2⁴·3²·7`), Bluestein's for 10007, 100003 and 1000003, whose `p − 1` has a large prime factor. A prime factor above 19 of a composite length is a pass of its own, taken last; its sequences are adjacent there, so Rader's or Bluestein's algorithm runs on blocks of up to 16 of them at once (a divisor of their count), with the inner transform planned for that many interleaved sequences so that the vectors run across them, and the twiddles of the pass applied on the way out; one sequence at a time when no block divides the count. Bluestein's algorithm for the whole length is taken instead when it is estimated faster. Rader's and Bluestein's kernels are computed in `f64`, through the plan's own inner transform when the type is `f64`.

**SIMD.** The butterflies and passes are written once over a block of complex values and instantiated per instruction set: NEON (`f64`: one complex value a vector; `f32`: two, with a block of one for what is left), SSE2, and AVX2 with FMA (two `f64` and four `f32` values a vector, with SSE2 blocks for what is left), chosen at run time when the plan is made; the portable block is a `Complex<T>`. The vectors run along the contiguous index `q`; the first pass (`s = 1`) of `f32` (and of AVX2's `f64`) runs them along `p` instead, with the twiddles by lanes, and stores the results by rows. AVX2's `f32` first pass takes the 128-bit block.

**Batches.** A length of one pass (2, 3, 4, 5, 7, 8, 11, 13, 17 and 19) runs in place with no working memory, its vector lanes across the runs of a batch. Longer short transforms of a batch go through the passes together, up to 256 values a call, each kernel called once for the chunk, as far as the working memory allows: `scratch_len` is what one run needs, the allocating methods take room for the chunk when given more than one run, and a `_with_scratch` caller that passes more gets the same.

**Planning.** `Fft::new` costs about 2 ns a value for the lengths of passes (two transforms' time at 1024, half of one at 2²⁰) and 13 to 23 ns a value for Rader's and Bluestein's lengths; the cost is the table of roots, each entry about 2 ns (its lookup, signs and expanded store), and for Rader's and Bluestein's lengths the transform of the kernel. The estimate of the time of Bluestein's candidate lengths is computed from their exponents, without factoring them.

### Real transforms

**Even lengths** `n = 2m` pack the reals in pairs, `z[j] = x[2j] + i·x[2j + 1]`, take the complex transform of `m` out of place straight into the output (a short one in place, its working memory on the stack; a batch of short ones as batches of the complex transform, up to 4096 values at a time, into a buffer the pass then reads), and separate the spectra of the even and odd halves in one pass over the pairs of bins `(k, m − k)`: with `a = Z[k]`, `b = conj Z[m − k]`, `u = (a + b)/2` and `t = w_k·(a − b)` for `w_k = −i·e^(−2πi·k/n)/2`, `X[k] = u + t` and `X[m − k] = conj(u − t)`; `X[0]` and `X[m]` come from `Z[0]`, and `X[m/2]` is `conj Z[m/2]`. The inverse runs the same pass with `w_k = i·e^(2πi·k/n)` for every sequence of a batch straight into the output, then the inverse complex transforms of all of them as one batch, in place, which leaves the reals there. The pass is vectorised: NEON (two `f64` or four `f32` pairs a step, the parts split into lanes by de-interleaving loads, the top bins reversed in the registers), SSE2 and AVX2 with FMA (two or four, and four or eight pairs, chosen once when the plan is made), and portable Rust elsewhere and under Miri, the same expressions lane by lane. `forward_with_scratch` runs the sequences one by one (each copied into its output and transformed there in place), since the staging of a batch needs memory its contract does not give.

**Odd lengths** have no packing into a half-length transform. One whose smallest prime factor `r` is at most 11, `n = r·m`, splits into the `r` sequences of the residues of the index modulo `r`. Their spectra come two at a time from one complex transform of `m` of `x_{2p} + i·x_{2p+1}`, all `(r − 1)/2` of them as one batch, and the last from the real transform of `m`, recursively; one pass of radix `r` (written out for 3 and 5, sums for 7 and 11) then makes only the first half of the spectrum, `X[s + q·m] = Σ_a e^(−2πi·a·s/n)·Y_a[s]·e^(−2πi·a·q/r)` for `s ≤ (m − 1)/2`, writing the bins beyond `n/2` as the conjugates they mirror. The inverse is its transpose. This costs about `1 + 1/r + …` complex transforms of `m`, near half the complex transform of `n` on reals. Odd lengths up to 15 and even ones up to 16 are summed directly from the table of `e^(−2πi·jk/n)`, the sums of the whole spectrum held in registers (faster there than any transform with its fixed costs, measured); primes and lengths whose factors all exceed 11 run the complex transform of `n` on the reals (a prime then takes Rader's or Bluestein's algorithm). Combining two real sequences of a batch into one complex one was not used: it loses accuracy for a sequence much smaller than its partner.

**The tables** of the real transforms' pass, of the odd splits and of the cosine and sine transforms of types II to IV are built as `e^(−2πi·(o + s·k)/n) = C[a]·F[j]` for `k = a·b + j`, `b` about `√len`: a fine table `F` of the powers of one root and a coarse table `C` of the powers of another, both and their products carried in double-double, the two roots summed by Taylor's series in double-double. Each value is then the correctly rounded root (the bench's `rootcheck`, against the tests' double-double roots up to `n = 2²³`, found no value off, where the complex transform's generator is off by up to one unit in the last place) and costs one product of double-doubles instead of a sine and a cosine; tables of fewer than 128 roots take the roots one by one from the complex transform's generator. These are two generators of roots, this one and the complex transform's; they are to be merged.

### Cosine and sine transforms

The algorithms are FFTW's accurate ones (those it enables by default), on the real transform:

- Types II and III by Makhoul's reordering (FFTW's `reodft010e-r2hc`): `v` holds the even-indexed values, then the odd-indexed ones backwards; with `V` its real transform and `c = e^(−πi·k/2n)·V[k]`, DCT-II is `y[k] = 2·Re c`, `y[n − k] = −2·Im c`. DCT-III is the transpose, `V[k] = e^(πi·k/2n)·(X[k] − i·X[n − k])` through the inverse real transform and the inverse reordering. For an even `n` the reordering writes straight into the packed pairs of the real transform.
- Type IV of an even `n` by the complex transform of `n/2` of `(x[2j] + i·x[n − 1 − 2j])·e^(−πi·j/n)`; with `W = 2·e^(−πi·(4p + 1)/4n)·Z[p]`, `y[2p] = Re W` and `y[n − 1 − 2p] = −Im W`, every step in place. Of an odd `n` by the real transform of `n` of a permutation of the values with signs, and sums and differences of the spectrum's parts times `√2` (Chan and Ho, FFTW's `reodft11e-r2hc-odd`).
- Type I of an odd `n` by FFTW's split-radix step (`reodft00e-splitradix`): with `N = n − 1`, `A` the DCT-I of the `N/2 + 1` even-indexed values and `B` the DCT-II of the `N/2` odd-indexed ones, `y[k] = A[k] + B[k]` and `y[N − k] = A[k] − B[k]`, recursively, which costs about one real transform of `n` where the symmetric extension would take one of `2(n − 1)`; DST-I alike, with the DST-I of the odd-indexed values and the DST-II of the even-indexed ones. An even `n` takes the real transform of the symmetric (antisymmetric) extension, `2(n − 1)` (`2(n + 1)`) values, as FFTW does.
- The sine transforms take the same steps with the signs and orders that turn them into the cosine transforms of their types, `DST-II(x)[k] = DCT-II((−1)^j·x[j])[n − 1 − k]`, `DST-III(X)[j] = (−1)^j·DCT-III(X[n − 1 − k])[j]` and `DST-IV(x)[k] = (−1)^k·DCT-IV(x[n − 1 − j])[k]`, folded into the reordering and the twiddle passes rather than run as passes of their own.
- Lengths up to 8 (16 for type I) are summed directly from the matrix, each column padded to 16 entries and the 16 sums held in registers; the crossovers were measured.

The passes around the transforms are loops over slices cut to the lengths they cover, read and written from both ends, which the compiler vectorises (NEON, and SSE2 at the baseline of x86_64); the permutation and the `√2` pass of the odd type IV, and the gathers of the type I split, are scalar. DCT-II's twiddle pass runs after the real transform's pass, not fused into it.

### Two and three dimensions

The last, contiguous dimension is transformed as one batch of rows over all the arrays of the call. Each other dimension goes by tiles of adjacent columns: up to 256 bytes of each row (two cache lines of 128 bytes: 16 `Complex<f64>` or 32 `Complex<f32>`) and at most 1 MiB in all are gathered into a buffer, column after column, transformed there as one batch of contiguous sequences and scattered back, so that every pass over the array reads and writes whole cache lines even where the stride of the rows, a power of two, maps the values of a column to the same sets of the cache. The gather and the scatter copy value by value.

### Convolution and correlation

`convolve` takes one of four paths, chosen by a model of their costs in nanoseconds:

- **Scalar**: for at most 16 products, operands of at most 16 values, the double loop on stack copies.
- **Direct**: `c[k] = Σ_t taps[t]·x[k + t]` with `taps` the shorter operand reversed and `x` the longer between `len(short) − 1` zeros, by SIMD kernels: NEON on aarch64, SSE2 and AVX2 with FMA (chosen at run time) on x86_64, and the same steps on arrays of lanes elsewhere and under Miri, from one body of macros. A kernel keeps a block of outputs in registers (8 vectors on NEON, 4 on AVX2 and SSE2) while the taps run past it two vectors at a time: for each lane `s` of the taps the vectors of `x` from `k + t + s` are loaded once, and each serves the tap `t + s` for one block of sums and `t + W + s` for the next, kept apart so that no multiply-add waits for another; the taps left over are taken one at a time. Complex values stay interleaved: the kernel sums `A = Σ x·re(tap)` and `B = Σ x·im(tap)` lane by lane and forms `A + i·B` once per block, so its steps have no shuffles. The outputs that every tap reaches (a multiple of 64 of them) are computed in place on the longer operand; only those at the two ends read short copies between zeros (on the stack up to 512 values, the taps up to 128). A correlation is `conv(op(u), v) = op(conv(u, op(v)))` for `op` a reversal, a conjugation or both, so whichever operand is reversed nothing is copied; the result is reversed or conjugated in place.
- **Transforms of the padded product**: both operands padded to a length `L ≥ len(a) + len(b) − 1` of the form `2^a·3^b·5^c·7^d` (`b ≤ 4`, `c ≤ 2`, `d ≤ 1`, and `a ≥ 1` for real data, which go through `RealFft` and its `L/2 + 1` values), the cheapest by the model rather than the shortest; the spectra multiplied with the scale `1/L` folded in (a SIMD pass), and the product transformed back.
- **Overlap-add**, when the longer operand has at least four times the shorter's length: the longer is cut into blocks of `B = L − len(short) + 1` values, each padded to `L`, transformed, multiplied by the shorter operand's spectrum (computed once) and transformed back, and the results added where they overlap; the blocks go through the transforms in batches of up to 65,536 values a call. The padded product is the case of one block.

The model prices the direct path at `direct + copy·len + mac·len·len(short)`, a transform of the length `L = 2^a·3^b·5^c·7^d` at `call + fft·L·(a + w₃b + w₅c + w₇d)` (`fft·L·log₂ L` for a power of two; factors 11 and 13 and larger primes have weights of their own), planning at half of `plan_call + plan·L`, and each pass over the buffers at `pass·L`. It takes the cheapest of the direct path, the padded product at its cheapest length, and overlap-add at every length `L` of that form from `2·len(short)` up to the padded length or `64·max(len(short), 16)`, whichever is smaller. A lower bound of every transform path skips that search when the direct path is cheaper anyway: each plans a length `L ≥ 2·len(short) − 1`, transforms the shorter operand once and its blocks there and back, at least `len(long)` values each way, and a transform of `L` values costs at least `fft·L·log₂ L` times the least weight of a radix per bit; being below every path's estimate, the bound never changes the path chosen. Half of the planning: with the plans kept, a first call pays all of it and a repeated one none, and the choice may not depend on what is kept, or the same call would give other bits after another one; charging all of it measured up to 3.2 times slower on repeated calls where kept plans make the transforms win, charging none of it up to 2.9 times slower on first calls, and a half at most 1.6 times (2.0 for repeated `f32` calls) either way, the geometric means within 1.03. The constants are measured per type by the bench's `calibrate` (`mac` by sweeps of the direct path near the crossover, where its taps are longer than calibrate's fit sees) and are to be measured again whenever the transforms or the kernels change; on x86_64 without AVX2 the cost of a product of the direct path is doubled, an estimate (SSE2's kernels have a quarter of AVX2's products a cycle and its transforms about half of AVX2's speed), not a measurement.

`circular_convolve` takes the cheapest of the direct path on `a` extended periodically (`n²` products), three transforms of the length `n`, and the linear convolution by transforms of a smooth length of at least `2n − 1`, folded modulo `n`, which the model's costs favour when `n` has a large prime factor (whose transforms of `n` would take Bluestein's algorithm). `circular_correlate` is the circular convolution with `conj(b[(n − j) mod n])`, written straight into the buffer each path fills (the folded linear path makes one copy); it takes the time of `circular_convolve`.

## Testing

The tests of `lib/tests/math/fft` (52 of them) check every algorithm of every transform against references summed in double-double arithmetic: the DFT for the complex and real transforms on dense data at every length to 64 and at chosen ones to 2062, and on sparse data at long lengths to 100,000, the definitions of the eight cosine and sine transforms, and the double-double matrix columns (impulses) at long lengths, with the relative L2 error bounded by `c·ε·(log₂ n + 1)`, `c` 3 for the complex transform, 10 for the others, 20 for round trips and 4 for batches against single runs; every batch against its runs one by one; the ignored imaginary parts of the real inverse for every algorithm; the two and three dimensions against line-by-line transforms (tiles of widths not a multiple of the tile, long columns, axes of 1); in place against out of place; every `_with_scratch` method with exactly `scratch_len()` values and with more against the allocating one, and short scratch panicking; empty batches of every algorithm (a panic of `RealFft::forward` on an empty batch of an even length of 18 and more was found by the fuzzing below and fixed); clones; `Send` and `Sync`; convolutions on every path (scalar, direct with its interior and its ends, either operand longer, padded product, overlap-add with full and partial batches, the three circular paths) against direct double-double sums, the direct kernels at every count of taps, exact small integers, and the plan cache (kept, evicted least recently used, a plan past the bytes never kept, eight threads with another churning the cache, every result bit-identical to the one-thread result).

The gates passed on the final code: `cargo fmt`, `clippy -D warnings`, clippy for `wasm32`, a check on the MSRV 1.88, rustdoc with `-D warnings`, the math tests in release and debug (1271) and the doctests (707). Under Miri, one test a process, on `aarch64-apple-darwin`, `s390x-unknown-linux-gnu` (big-endian) and `i686-unknown-linux-gnu` (32-bit), 49 of the 52 tests run and pass on each (three are too long for Miri: the long operands and two of the cache's; the threads test runs the cache under Miri with three threads); Miri runs the portable kernels, which follow the same steps as the vector ones lane by lane. It caught a violation of Stacked Borrows in the setup of Rader's and Bluestein's kernels, and on 32 bits a bound `1 << 60` overflowing `usize`; both were fixed, and with the second Rader's `a·g mod p`, which could overflow a 32-bit `usize` for `p > 2¹⁶`. On x86_64, in Docker's `linux/amd64` (`rust:1.88`), the 52 tests pass on the AVX2 and FMA path, and 52 again on the SSE2 path, by the same test binary under `qemu-x86_64 -cpu Nehalem`, where the plans report `isa: Sse2`. The x86_64 kernels were checked for correctness under emulation only; their speed has never been measured.

The module was also fuzzed differentially against FFTW and the double-double reference: every API, the `_with_scratch` forms and batches among them, `f32` and `f64`, on random, sparse, constant and impulse data of ordinary, tiny and huge scales, and on empty data and lengths 0. The first run, 5,211,326 cases on the code before its last changes, found the empty-batch panic above; the second, 3,478,231 cases on the final code, found nothing, its worst error 0.14 of the tests' bound (in DCT-IV). Its only rejected cases are convolutions with near-constant operands, where an error measured relative to `‖a‖·‖b‖` does not apply; FFTW in the same precision fails them alike.

## Measured

Measured on an Apple M4 Max on 2026-11-17, three full runs of each of three harnesses: the complex transform and the real, cosine and sine transforms with the two and three dimensions between 19:13 and 20:02, and the convolutions between 20:47 and 21:16, on the final code after a change to the lower bound of their model (above), one thread, `env -u MallocNanoZone`, fat LTO and one codegen unit, with nothing else running but the machine's own: one-minute load averages 1.2 to 8.0 at the times recorded (each table gives its range). Each figure here is the median of the three runs' figures, each of those the median of its rounds; the ratios are recomputed from the medians, `a/x` being `a`'s time over the reference's: below 1, `a` is faster. The accuracy of the real and trigonometric transforms was measured once, right after, at 20:03 (it does not depend on the load).

The references: FFTW 3.3.11 (Homebrew's static build, `FFTW_MEASURE`, one thread, planned before the clock starts; on arm64 Homebrew's FFTW has no SIMD codelets, only its scalar ones, so it is FFTW at its scalar speed here); rustfft 6.4.1 (NEON) and realfft 3.5.1; and Apple's Accelerate vDSP, on split complex data (`vDSP_fft_zip` in place for the complex powers of two, `vDSP_DFT_Execute` out of place for `f·2^k`, `f` in 1, 3, 5, 15; `vDSP_fft_zrip` and `vDSP_DFT_zrop` for real data; `vDSP_DCT_Execute` for the cosine transforms of types II, III and IV, `f32` only; `vDSP_fft2d_zip`; `vDSP_conv` and `vDSP_zconv`, direct). The x86_64 paths were not measured.

### The complex transform

`forward_with_scratch` in place for one transform and the allocating `forward` for a batch, against FFTW in place, rustfft's `process_with_scratch` and vDSP; nanoseconds per transform, the median of 7 rounds of 20 ms with the contenders interleaved; Gflop/s is FFTW's `5·n·log₂ n / t`.

`f64`, powers of two (load 2.0 to 8.0):

| n | `a` | FFTW | rustfft | vDSP | `a` Gflop/s | a/FFTW | a/rustfft | a/vDSP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.9 ns | 2.7 ns | 3.3 ns | 4.6 ns | 1.7 | 2.19 | 1.79 | 1.28 |
| 4 | 6.7 ns | 4.4 ns | 7.2 ns | 5.3 ns | 6.0 | 1.52 | 0.93 | 1.26 |
| 8 | 8.6 ns | 6.8 ns | 12.6 ns | 7.6 ns | 14.0 | 1.26 | 0.68 | 1.13 |
| 16 | 19 ns | 12 ns | 16.9 ns | 16.7 ns | 16.8 | 1.58 | 1.12 | 1.14 |
| 32 | 32.2 ns | 30.4 ns | 27.4 ns | 33.8 ns | 24.8 | 1.06 | 1.18 | 0.95 |
| 64 | 57.3 ns | 85.7 ns | 81.4 ns | 60.1 ns | 33.5 | 0.67 | 0.70 | 0.95 |
| 128 | 122 ns | 216 ns | 181 ns | 135 ns | 36.6 | 0.57 | 0.68 | 0.91 |
| 256 | 265 ns | 433 ns | 404 ns | 272 ns | 38.6 | 0.61 | 0.66 | 0.98 |
| 512 | 588 ns | 962 ns | 905 ns | 658 ns | 39.2 | 0.61 | 0.65 | 0.89 |
| 1024 | 1.33 µs | 2.23 µs | 1.99 µs | 1.41 µs | 38.6 | 0.59 | 0.67 | 0.94 |
| 2048 | 3.05 µs | 4.81 µs | 4.42 µs | 3.22 µs | 36.9 | 0.63 | 0.69 | 0.95 |
| 4096 | 7.17 µs | 11.9 µs | 10.5 µs | 6.9 µs | 34.3 | 0.60 | 0.68 | 1.04 |
| 8192 | 17 µs | 27 µs | 23.9 µs | 16.7 µs | 31.2 | 0.63 | 0.71 | 1.02 |
| 16,384 | 37.5 µs | 57.6 µs | 63.7 µs | 52.1 µs | 30.6 | 0.65 | 0.59 | 0.72 |
| 32,768 | 80.1 µs | 127 µs | 119 µs | 140 µs | 30.7 | 0.63 | 0.67 | 0.57 |
| 65,536 | 189 µs | 288 µs | 295 µs | 299 µs | 27.8 | 0.65 | 0.64 | 0.63 |
| 131,072 | 427 µs | 621 µs | 534 µs | 707 µs | 26.1 | 0.69 | 0.80 | 0.60 |
| 262,144 | 956 µs | 1.36 ms | 1.31 ms | 1.66 ms | 24.7 | 0.70 | 0.73 | 0.58 |
| 524,288 | 1.96 ms | 3.56 ms | 2.48 ms | 3.39 ms | 25.4 | 0.55 | 0.79 | 0.58 |
| 1,048,576 | 4.22 ms | 8.04 ms | 5.51 ms | 7.31 ms | 24.9 | 0.52 | 0.77 | 0.58 |
| 2,097,152 | 8.9 ms | 15.9 ms | 12.2 ms | 16.1 ms | 24.8 | 0.56 | 0.73 | 0.55 |
| 4,194,304 | 20.3 ms | 39.2 ms | 26.6 ms | 32.9 ms | 22.7 | 0.52 | 0.76 | 0.62 |

`f32`, powers of two (load 2.7 to 6.8):

| n | `a` | FFTW | rustfft | vDSP | `a` Gflop/s | a/FFTW | a/rustfft | a/vDSP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 6.2 ns | 2.5 ns | 1.7 ns | 5.1 ns | 1.6 | 2.48 | 3.65 | 1.22 |
| 4 | 6.7 ns | 4.4 ns | 6 ns | 7.3 ns | 6.0 | 1.52 | 1.12 | 0.92 |
| 8 | 9 ns | 6.6 ns | 10.8 ns | 9 ns | 13.3 | 1.36 | 0.83 | 1.00 |
| 16 | 15.2 ns | 12.1 ns | 10.7 ns | 16 ns | 21.1 | 1.26 | 1.42 | 0.95 |
| 32 | 20 ns | 30.3 ns | 16.5 ns | 19.8 ns | 40.0 | 0.66 | 1.21 | 1.01 |
| 64 | 32.4 ns | 85.5 ns | 51.3 ns | 35.6 ns | 59.3 | 0.38 | 0.63 | 0.91 |
| 128 | 72.2 ns | 203 ns | 103 ns | 60.7 ns | 62.0 | 0.36 | 0.70 | 1.19 |
| 256 | 149 ns | 419 ns | 239 ns | 120 ns | 68.8 | 0.36 | 0.62 | 1.24 |
| 512 | 314 ns | 960 ns | 521 ns | 196 ns | 73.5 | 0.33 | 0.60 | 1.60 |
| 1024 | 705 ns | 2.1 µs | 1.2 µs | 306 ns | 72.6 | 0.34 | 0.59 | 2.30 |
| 2048 | 1.53 µs | 4.78 µs | 2.52 µs | 680 ns | 73.4 | 0.32 | 0.61 | 2.26 |
| 4096 | 4.71 µs | 10.4 µs | 5.76 µs | 1.28 µs | 52.2 | 0.45 | 0.82 | 3.69 |
| 8192 | 10.5 µs | 23.3 µs | 13.1 µs | 3.31 µs | 50.9 | 0.45 | 0.80 | 3.16 |
| 16,384 | 17.8 µs | 51.4 µs | 30 µs | 8.23 µs | 64.4 | 0.35 | 0.59 | 2.16 |
| 32,768 | 39.7 µs | 121 µs | 63.8 µs | 18.2 µs | 62.0 | 0.33 | 0.62 | 2.18 |
| 65,536 | 94.1 µs | 258 µs | 148 µs | 47.8 µs | 55.7 | 0.36 | 0.64 | 1.97 |
| 131,072 | 210 µs | 584 µs | 284 µs | 112 µs | 53.0 | 0.36 | 0.74 | 1.88 |
| 262,144 | 463 µs | 1.28 ms | 748 µs | 231 µs | 51.0 | 0.36 | 0.62 | 2.00 |
| 524,288 | 971 µs | 3.24 ms | 1.26 ms | 495 µs | 51.3 | 0.30 | 0.77 | 1.96 |
| 1,048,576 | 2.08 ms | 6.8 ms | 3.27 ms | 1.38 ms | 50.5 | 0.31 | 0.63 | 1.50 |
| 2,097,152 | 4.33 ms | 13.8 ms | 5.89 ms | 3.53 ms | 50.8 | 0.31 | 0.74 | 1.23 |
| 4,194,304 | 9.57 ms | 32 ms | 13.6 ms | 7.76 ms | 48.2 | 0.30 | 0.70 | 1.23 |

Smooth lengths: `3·2^k` and `5·2^k` (vDSP's DFT has them from 24 and 40), then `3^k`, `5^k`, `7^k` and mixed ones (15,360 is `15·2^10`):

`f64`, smooth lengths (load 2.0 to 7.9):

| n | `a` | FFTW | rustfft | vDSP | `a` Gflop/s | a/FFTW | a/rustfft | a/vDSP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 11.6 ns | 6.1 ns | 8.9 ns |  | 6.7 | 1.90 | 1.30 |  |
| 12 | 15.8 ns | 9.7 ns | 12.6 ns |  | 13.6 | 1.63 | 1.25 |  |
| 24 | 27.1 ns | 36.8 ns | 20.6 ns | 40.1 ns | 20.3 | 0.74 | 1.32 | 0.68 |
| 48 | 50.4 ns | 74.7 ns | 88.7 ns | 84.4 ns | 26.6 | 0.67 | 0.57 | 0.60 |
| 96 | 96 ns | 174 ns | 186 ns | 178 ns | 32.9 | 0.55 | 0.52 | 0.54 |
| 192 | 205 ns | 334 ns | 277 ns | 388 ns | 35.6 | 0.61 | 0.74 | 0.53 |
| 384 | 457 ns | 720 ns | 634 ns | 838 ns | 36.1 | 0.63 | 0.72 | 0.55 |
| 768 | 1 µs | 1.51 µs | 1.41 µs | 1.91 µs | 36.8 | 0.66 | 0.71 | 0.52 |
| 1536 | 2.19 µs | 3.5 µs | 3.09 µs | 3.99 µs | 37.0 | 0.63 | 0.71 | 0.55 |
| 3072 | 5.15 µs | 7.98 µs | 7.54 µs | 9.5 µs | 34.5 | 0.65 | 0.68 | 0.54 |
| 6144 | 12 µs | 17.5 µs | 15.9 µs | 21.9 µs | 32.3 | 0.68 | 0.75 | 0.55 |
| 12,288 | 25.9 µs | 40 µs | 40.3 µs | 55.4 µs | 32.3 | 0.65 | 0.64 | 0.47 |
| 24,576 | 69.8 µs | 88.5 µs | 86.8 µs | 156 µs | 25.7 | 0.79 | 0.80 | 0.45 |
| 49,152 | 149 µs | 200 µs | 206 µs | 333 µs | 25.7 | 0.75 | 0.72 | 0.45 |
| 98,304 | 329 µs | 433 µs | 395 µs | 1.07 ms | 24.8 | 0.76 | 0.83 | 0.31 |
| 196,608 | 634 µs | 939 µs | 956 µs | 1.46 ms | 27.3 | 0.67 | 0.66 | 0.43 |
| 393,216 | 1.41 ms | 2.2 ms | 1.89 ms | 4.61 ms | 25.9 | 0.64 | 0.74 | 0.31 |
| 786,432 | 3.04 ms | 4.73 ms | 4.22 ms | 6.22 ms | 25.3 | 0.64 | 0.72 | 0.49 |
| 10 | 15.4 ns | 9.6 ns | 11.5 ns |  | 10.8 | 1.60 | 1.34 |  |
| 20 | 22.2 ns | 16.8 ns | 34.3 ns |  | 19.5 | 1.32 | 0.65 |  |
| 40 | 39.2 ns | 61 ns | 60 ns | 69.3 ns | 27.2 | 0.64 | 0.65 | 0.57 |
| 80 | 80 ns | 134 ns | 158 ns | 139 ns | 31.6 | 0.60 | 0.51 | 0.57 |
| 160 | 165 ns | 284 ns | 352 ns | 294 ns | 35.5 | 0.58 | 0.47 | 0.56 |
| 320 | 357 ns | 581 ns | 803 ns | 669 ns | 37.3 | 0.62 | 0.45 | 0.53 |
| 640 | 812 ns | 1.35 µs | 1.71 µs | 1.44 µs | 36.8 | 0.60 | 0.48 | 0.57 |
| 1280 | 1.79 µs | 2.97 µs | 3.59 µs | 3.29 µs | 36.9 | 0.60 | 0.50 | 0.54 |
| 2560 | 4.03 µs | 6.64 µs | 7.91 µs | 7.23 µs | 36.0 | 0.61 | 0.51 | 0.56 |
| 5120 | 9.52 µs | 15.4 µs | 18.6 µs | 17.5 µs | 33.2 | 0.62 | 0.51 | 0.54 |
| 10,240 | 20.6 µs | 33.9 µs | 40.4 µs | 54.7 µs | 33.1 | 0.61 | 0.51 | 0.38 |
| 20,480 | 44.2 µs | 74.2 µs | 87.1 µs | 129 µs | 33.2 | 0.60 | 0.51 | 0.34 |
| 40,960 | 119 µs | 170 µs | 187 µs | 323 µs | 26.3 | 0.70 | 0.64 | 0.37 |
| 81,920 | 251 µs | 362 µs | 449 µs | 645 µs | 26.7 | 0.69 | 0.56 | 0.39 |
| 163,840 | 559 µs | 771 µs | 855 µs | 2 ms | 25.4 | 0.73 | 0.65 | 0.28 |
| 327,680 | 1.07 ms | 1.89 ms | 2 ms | 2.8 ms | 28.1 | 0.56 | 0.53 | 0.38 |
| 655,360 | 2.37 ms | 3.89 ms | 3.75 ms | 8.52 ms | 26.8 | 0.61 | 0.63 | 0.28 |
| 9 | 14 ns | 10.9 ns | 14.3 ns |  | 10.2 | 1.28 | 0.98 |  |
| 27 | 35.2 ns | 43.5 ns | 56.6 ns |  | 18.2 | 0.81 | 0.62 |  |
| 81 | 103 ns | 151 ns | 175 ns |  | 24.9 | 0.68 | 0.59 |  |
| 243 | 344 ns | 552 ns | 948 ns |  | 28.0 | 0.62 | 0.36 |  |
| 729 | 1.22 µs | 1.94 µs | 3.89 µs |  | 28.4 | 0.63 | 0.31 |  |
| 2187 | 4.27 µs | 6.96 µs | 11.1 µs |  | 28.4 | 0.61 | 0.39 |  |
| 6561 | 19 µs | 24.2 µs | 37 µs |  | 21.9 | 0.78 | 0.51 |  |
| 19,683 | 65.1 µs | 81.4 µs | 146 µs |  | 21.6 | 0.80 | 0.44 |  |
| 59,049 | 245 µs | 275 µs | 542 µs |  | 19.1 | 0.89 | 0.45 |  |
| 177,147 | 778 µs | 969 µs | 1.91 ms |  | 19.8 | 0.80 | 0.41 |  |
| 531,441 | 2.78 ms | 3.35 ms | 6.58 ms |  | 18.2 | 0.83 | 0.42 |  |
| 25 | 25.7 ns | 29.8 ns | 52.6 ns |  | 22.6 | 0.86 | 0.49 |  |
| 125 | 136 ns | 271 ns | 463 ns |  | 32.1 | 0.50 | 0.29 |  |
| 625 | 910 ns | 1.75 µs | 3.29 µs |  | 31.9 | 0.52 | 0.28 |  |
| 3125 | 6.03 µs | 11.2 µs | 21.5 µs |  | 30.1 | 0.54 | 0.28 |  |
| 15,625 | 37.2 µs | 65.8 µs | 141 µs |  | 29.2 | 0.57 | 0.26 |  |
| 78,125 | 215 µs | 419 µs | 826 µs |  | 29.5 | 0.51 | 0.26 |  |
| 390,625 | 1.26 ms | 2.71 ms | 4.79 ms |  | 28.8 | 0.47 | 0.26 |  |
| 49 | 47.5 ns | 87.4 ns | 93.6 ns |  | 29.0 | 0.54 | 0.51 |  |
| 343 | 447 ns | 866 ns | 1.14 µs |  | 32.3 | 0.52 | 0.39 |  |
| 2401 | 4.33 µs | 8.16 µs | 11.4 µs |  | 31.2 | 0.53 | 0.38 |  |
| 16,807 | 39 µs | 72.6 µs | 119 µs |  | 30.3 | 0.54 | 0.33 |  |
| 117,649 | 331 µs | 746 µs | 1.02 ms |  | 30.0 | 0.44 | 0.32 |  |
| 823,543 | 2.74 ms | 7.07 ms | 8.69 ms |  | 29.5 | 0.39 | 0.32 |  |
| 60 | 62.6 ns | 96.3 ns | 116 ns |  | 28.3 | 0.65 | 0.54 |  |
| 360 | 463 ns | 790 ns | 885 ns |  | 33.0 | 0.59 | 0.52 |  |
| 1000 | 1.48 µs | 2.53 µs | 5.34 µs |  | 33.6 | 0.59 | 0.28 |  |
| 2160 | 4 µs | 5.78 µs | 7.97 µs |  | 29.9 | 0.69 | 0.50 |  |
| 6000 | 13.3 µs | 19 µs | 43.4 µs |  | 28.3 | 0.70 | 0.31 |  |
| 10,000 | 23.8 µs | 34.2 µs | 55.5 µs |  | 27.9 | 0.70 | 0.43 |  |
| 15,360 | 42.1 µs | 48.6 µs | 72.1 µs | 66 µs | 25.4 | 0.87 | 0.58 | 0.64 |
| 30,000 | 78.5 µs | 111 µs | 259 µs |  | 28.4 | 0.71 | 0.30 |  |
| 48,000 | 124 µs | 201 µs | 388 µs |  | 30.1 | 0.62 | 0.32 |  |
| 100,000 | 274 µs | 495 µs | 917 µs |  | 30.3 | 0.55 | 0.30 |  |
| 216,000 | 697 µs | 1.07 ms | 2.14 ms |  | 27.5 | 0.65 | 0.33 |  |
| 1,000,000 | 3.38 ms | 7.64 ms | 12.7 ms |  | 29.5 | 0.44 | 0.27 |  |

`f32`, smooth lengths (load 2.2 to 6.4):

| n | `a` | FFTW | rustfft | vDSP | `a` Gflop/s | a/FFTW | a/rustfft | a/vDSP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 11.7 ns | 6.1 ns | 7.3 ns |  | 6.6 | 1.92 | 1.60 |  |
| 12 | 15.2 ns | 9.6 ns | 10.3 ns |  | 14.2 | 1.58 | 1.48 |  |
| 24 | 19.3 ns | 37 ns | 13.9 ns | 24.9 ns | 28.5 | 0.52 | 1.39 | 0.78 |
| 48 | 32.2 ns | 68 ns | 69.3 ns | 47.1 ns | 41.6 | 0.47 | 0.46 | 0.68 |
| 96 | 56.9 ns | 174 ns | 134 ns | 85.8 ns | 55.5 | 0.33 | 0.42 | 0.66 |
| 192 | 117 ns | 317 ns | 172 ns | 173 ns | 62.0 | 0.37 | 0.68 | 0.68 |
| 384 | 250 ns | 654 ns | 375 ns | 332 ns | 66.0 | 0.38 | 0.67 | 0.75 |
| 768 | 534 ns | 1.55 µs | 873 ns | 250 ns | 69.0 | 0.35 | 0.61 | 2.14 |
| 1536 | 1.15 µs | 3.22 µs | 1.84 µs | 454 ns | 70.6 | 0.36 | 0.63 | 2.53 |
| 3072 | 2.58 µs | 7.44 µs | 4.2 µs | 893 ns | 69.0 | 0.35 | 0.61 | 2.89 |
| 6144 | 7.72 µs | 16.3 µs | 9.54 µs | 2.11 µs | 50.1 | 0.47 | 0.81 | 3.66 |
| 12,288 | 13 µs | 35.6 µs | 20.3 µs | 5.45 µs | 64.1 | 0.37 | 0.64 | 2.39 |
| 24,576 | 33 µs | 79.8 µs | 45.8 µs | 12 µs | 54.2 | 0.41 | 0.72 | 2.75 |
| 49,152 | 71.4 µs | 171 µs | 100 µs | 30.9 µs | 53.7 | 0.42 | 0.71 | 2.31 |
| 98,304 | 161 µs | 408 µs | 215 µs | 70.8 µs | 50.8 | 0.39 | 0.75 | 2.27 |
| 196,608 | 313 µs | 884 µs | 495 µs | 159 µs | 55.2 | 0.35 | 0.63 | 1.97 |
| 393,216 | 697 µs | 1.99 ms | 939 µs | 343 µs | 52.4 | 0.35 | 0.74 | 2.03 |
| 786,432 | 1.51 ms | 4.21 ms | 2.37 ms | 837 µs | 51.1 | 0.36 | 0.64 | 1.80 |
| 10 | 14.1 ns | 9.3 ns | 9.3 ns |  | 11.8 | 1.52 | 1.52 |  |
| 20 | 18 ns | 16.8 ns | 30.3 ns |  | 24.0 | 1.07 | 0.59 |  |
| 40 | 26.3 ns | 60.8 ns | 48.7 ns | 39.1 ns | 40.5 | 0.43 | 0.54 | 0.67 |
| 80 | 47.1 ns | 125 ns | 114 ns | 65.8 ns | 53.7 | 0.38 | 0.41 | 0.72 |
| 160 | 96.4 ns | 279 ns | 252 ns | 131 ns | 60.8 | 0.35 | 0.38 | 0.73 |
| 320 | 204 ns | 543 ns | 539 ns | 258 ns | 65.4 | 0.38 | 0.38 | 0.79 |
| 640 | 455 ns | 1.23 µs | 1.1 µs | 552 ns | 65.6 | 0.37 | 0.41 | 0.82 |
| 1280 | 975 ns | 2.83 µs | 2.34 µs | 412 ns | 67.7 | 0.34 | 0.42 | 2.37 |
| 2560 | 2.13 µs | 6.29 µs | 4.87 µs | 771 ns | 68.2 | 0.34 | 0.44 | 2.76 |
| 5120 | 6.4 µs | 13.4 µs | 10.7 µs | 1.77 µs | 49.3 | 0.48 | 0.60 | 3.63 |
| 10,240 | 11.2 µs | 30 µs | 24 µs | 4.41 µs | 60.7 | 0.37 | 0.47 | 2.55 |
| 20,480 | 21.9 µs | 64.3 µs | 50 µs | 9.1 µs | 66.9 | 0.34 | 0.44 | 2.41 |
| 40,960 | 57.1 µs | 144 µs | 105 µs | 21.9 µs | 54.9 | 0.40 | 0.54 | 2.61 |
| 81,920 | 124 µs | 330 µs | 232 µs | 57 µs | 53.8 | 0.38 | 0.53 | 2.18 |
| 163,840 | 275 µs | 730 µs | 474 µs | 132 µs | 51.6 | 0.38 | 0.58 | 2.08 |
| 327,680 | 546 µs | 1.6 ms | 1.06 ms | 271 µs | 55.0 | 0.34 | 0.52 | 2.01 |
| 655,360 | 1.2 ms | 3.42 ms | 2.08 ms | 697 µs | 52.9 | 0.35 | 0.57 | 1.72 |
| 9 | 14.9 ns | 10.6 ns | 11.6 ns |  | 9.6 | 1.41 | 1.28 |  |
| 27 | 27.1 ns | 43.4 ns | 48.5 ns |  | 23.7 | 0.62 | 0.56 |  |
| 81 | 69.4 ns | 132 ns | 130 ns |  | 37.0 | 0.53 | 0.53 |  |
| 243 | 207 ns | 526 ns | 780 ns |  | 46.4 | 0.39 | 0.27 |  |
| 729 | 706 ns | 1.88 µs | 3.33 µs |  | 49.1 | 0.38 | 0.21 |  |
| 2187 | 2.45 µs | 6.58 µs | 8.68 µs |  | 49.5 | 0.37 | 0.28 |  |
| 6561 | 8.66 µs | 22.9 µs | 24.5 µs |  | 48.0 | 0.38 | 0.35 |  |
| 19,683 | 35.4 µs | 78 µs | 109 µs |  | 39.7 | 0.45 | 0.32 |  |
| 59,049 | 128 µs | 263 µs | 419 µs |  | 36.5 | 0.49 | 0.31 |  |
| 177,147 | 407 µs | 911 µs | 1.5 ms |  | 37.9 | 0.45 | 0.27 |  |
| 531,441 | 1.43 ms | 2.96 ms | 5.26 ms |  | 35.3 | 0.48 | 0.27 |  |
| 25 | 20.6 ns | 29.8 ns | 41.5 ns |  | 28.2 | 0.69 | 0.50 |  |
| 125 | 86.6 ns | 243 ns | 370 ns |  | 50.3 | 0.36 | 0.23 |  |
| 625 | 542 ns | 1.65 µs | 2.66 µs |  | 53.6 | 0.33 | 0.20 |  |
| 3125 | 3.34 µs | 10.2 µs | 16.3 µs |  | 54.3 | 0.33 | 0.21 |  |
| 15,625 | 20.8 µs | 63.3 µs | 105 µs |  | 52.4 | 0.33 | 0.20 |  |
| 78,125 | 119 µs | 407 µs | 644 µs |  | 53.3 | 0.29 | 0.19 |  |
| 390,625 | 687 µs | 2.4 ms | 3.64 ms |  | 52.8 | 0.29 | 0.19 |  |
| 49 | 35.2 ns | 78 ns | 74.2 ns |  | 39.1 | 0.45 | 0.47 |  |
| 343 | 286 ns | 824 ns | 871 ns |  | 50.5 | 0.35 | 0.33 |  |
| 2401 | 2.55 µs | 7.69 µs | 8.4 µs |  | 53.0 | 0.33 | 0.30 |  |
| 16,807 | 22.3 µs | 69.1 µs | 82.1 µs |  | 53.0 | 0.32 | 0.27 |  |
| 117,649 | 184 µs | 657 µs | 726 µs |  | 53.7 | 0.28 | 0.25 |  |
| 823,543 | 1.52 ms | 6.08 ms | 5.85 ms |  | 53.4 | 0.25 | 0.26 |  |
| 60 | 39.7 ns | 95.8 ns | 86.8 ns | 30.7 ns | 44.6 | 0.41 | 0.46 | 1.29 |
| 360 | 262 ns | 728 ns | 609 ns |  | 58.4 | 0.36 | 0.43 |  |
| 1000 | 821 ns | 2.34 µs | 3.99 µs |  | 60.7 | 0.35 | 0.21 |  |
| 2160 | 2.01 µs | 5.09 µs | 5.47 µs |  | 59.5 | 0.40 | 0.37 |  |
| 6000 | 6.47 µs | 17 µs | 30.2 µs |  | 58.2 | 0.38 | 0.21 |  |
| 10,000 | 11.8 µs | 31.8 µs | 36.5 µs |  | 56.5 | 0.37 | 0.32 |  |
| 15,360 | 19.1 µs | 45.6 µs | 39.4 µs | 7.2 µs | 55.8 | 0.42 | 0.49 | 2.66 |
| 30,000 | 39.5 µs | 107 µs | 194 µs |  | 56.5 | 0.37 | 0.20 |  |
| 48,000 | 62.7 µs | 165 µs | 264 µs |  | 59.5 | 0.38 | 0.24 |  |
| 100,000 | 139 µs | 472 µs | 649 µs |  | 59.9 | 0.29 | 0.21 |  |
| 216,000 | 338 µs | 953 µs | 1.5 ms |  | 56.6 | 0.35 | 0.23 |  |
| 1,000,000 | 1.71 ms | 7.01 ms | 8.58 ms |  | 58.2 | 0.24 | 0.20 |  |

Primes, by Rader's algorithm (17 by one pass; 101 and 1009) and Bluestein's (10007 and above), and twice a prime:

`f64`, primes (load 2.6 to 7.4):

| n | `a` | FFTW | rustfft | `a` Gflop/s | a/FFTW | a/rustfft |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 17.4 ns | 33.3 ns | 21.1 ns | 20.0 | 0.52 | 0.82 |
| 101 | 301 ns | 969 ns | 1.01 µs | 11.2 | 0.31 | 0.30 |
| 1009 | 4.13 µs | 13 µs | 12 µs | 12.2 | 0.32 | 0.34 |
| 10,007 | 116 µs | 171 µs | 196 µs | 5.7 | 0.68 | 0.60 |
| 100,003 | 1.6 ms | 2.24 ms | 2.85 ms | 5.2 | 0.72 | 0.56 |
| 1,000,003 | 20.7 ms | 30.4 ms | 25.6 ms | 4.8 | 0.68 | 0.81 |

`f32`, primes (load 2.2 to 5.5):

| n | `a` | FFTW | rustfft | `a` Gflop/s | a/FFTW | a/rustfft |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 16.9 ns | 32.9 ns | 16.9 ns | 20.6 | 0.51 | 1.00 |
| 101 | 191 ns | 922 ns | 847 ns | 17.6 | 0.21 | 0.23 |
| 1009 | 2.41 µs | 12.4 µs | 9.78 µs | 20.9 | 0.19 | 0.25 |
| 10,007 | 59.9 µs | 157 µs | 101 µs | 11.1 | 0.38 | 0.59 |
| 100,003 | 800 µs | 2 ms | 1.6 ms | 10.4 | 0.40 | 0.50 |
| 1,000,003 | 10.3 ms | 26.5 ms | 13.1 ms | 9.7 | 0.39 | 0.79 |

`f64`, twice a prime (load 2.6 to 7.3):

| n | `a` | FFTW | rustfft | `a` Gflop/s | a/FFTW | a/rustfft |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 202 | 755 ns | 2.09 µs | 2.35 µs | 10.2 | 0.36 | 0.32 |
| 2018 | 10.3 µs | 28.3 µs | 26.6 µs | 10.8 | 0.36 | 0.39 |
| 20,014 | 266 µs | 351 µs | 444 µs | 5.4 | 0.76 | 0.60 |
| 200,006 | 3.49 ms | 4.57 ms | 6.27 ms | 5.0 | 0.76 | 0.56 |
| 2,000,006 | 42.4 ms | 63 ms | 55.3 ms | 4.9 | 0.67 | 0.77 |

`f32`, twice a prime (load 2.5 to 5.4):

| n | `a` | FFTW | rustfft | `a` Gflop/s | a/FFTW | a/rustfft |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 202 | 537 ns | 1.95 µs | 2 µs | 14.4 | 0.28 | 0.27 |
| 2018 | 6.56 µs | 26.7 µs | 21.4 µs | 16.9 | 0.25 | 0.31 |
| 20,014 | 132 µs | 321 µs | 229 µs | 10.8 | 0.41 | 0.58 |
| 200,006 | 1.76 ms | 4.13 ms | 3.3 ms | 10.0 | 0.43 | 0.53 |
| 2,000,006 | 21.9 ms | 53.4 ms | 28.4 ms | 9.5 | 0.41 | 0.77 |

Batches of 1000 transforms in one call, against FFTW's `plan_many_dft` and rustfft's processing of the whole buffer; ns per transform, as printed to 0.1 ns:

`f64`, batches of 1000 transforms, ns per transform (load 3.3 to 7.0):

| n | `a` | FFTW | rustfft | a/FFTW | a/rustfft |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.4 | 0.5 | 0.3 | 0.80 | 1.33 |
| 4 | 0.8 | 1.1 | 0.7 | 0.73 | 1.14 |
| 8 | 2.1 | 3.5 | 2.7 | 0.60 | 0.78 |
| 16 | 9.6 | 10.0 | 8.1 | 0.96 | 1.19 |
| 32 | 21.4 | 26.8 | 23.7 | 0.80 | 0.90 |
| 64 | 52.3 | 70.7 | 84.5 | 0.74 | 0.62 |

`f32`, batches of 1000 transforms, ns per transform (load 2.4 to 5.4):

| n | `a` | FFTW | rustfft | a/FFTW | a/rustfft |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.2 | 0.5 | 0.2 | 0.40 | 1.00 |
| 4 | 0.6 | 1.1 | 0.6 | 0.55 | 1.00 |
| 8 | 1.6 | 3.5 | 1.7 | 0.46 | 0.94 |
| 16 | 5.5 | 9.8 | 4.3 | 0.56 | 1.28 |
| 32 | 13.0 | 26.4 | 11.4 | 0.49 | 1.14 |
| 64 | 26.8 | 75.4 | 49.4 | 0.36 | 0.54 |

The inverse, `inverse_with_scratch`:

`f64`, inverse transforms (load 3.3 to 7.0):

| n | `a` | FFTW | rustfft | vDSP | `a` Gflop/s | a/FFTW | a/rustfft | a/vDSP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 58.4 ns | 85.3 ns | 81.6 ns | 60.2 ns | 32.9 | 0.68 | 0.72 | 0.97 |
| 1024 | 1.47 µs | 2.24 µs | 2 µs | 1.41 µs | 34.8 | 0.66 | 0.74 | 1.04 |
| 65,536 | 199 µs | 292 µs | 295 µs | 302 µs | 26.4 | 0.68 | 0.67 | 0.66 |
| 1000 | 1.63 µs | 2.56 µs | 5.2 µs |  | 30.5 | 0.64 | 0.31 |  |
| 10,007 | 118 µs | 169 µs | 195 µs |  | 5.7 | 0.69 | 0.60 |  |

`f32`, inverse transforms (load 2.3 to 6.2):

| n | `a` | FFTW | rustfft | vDSP | `a` Gflop/s | a/FFTW | a/rustfft | a/vDSP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 33.7 ns | 84.8 ns | 50.9 ns | 35.5 ns | 57.0 | 0.40 | 0.66 | 0.95 |
| 1024 | 765 ns | 2.16 µs | 1.19 µs | 307 ns | 66.9 | 0.36 | 0.64 | 2.49 |
| 65,536 | 100 µs | 258 µs | 147 µs | 47.8 µs | 52.4 | 0.39 | 0.68 | 2.09 |
| 1000 | 890 ns | 2.43 µs | 4.04 µs |  | 56.0 | 0.37 | 0.22 |  |
| 10,007 | 60.3 µs | 155 µs | 102 µs |  | 11.0 | 0.39 | 0.59 |  |

**Planning.** `Fft::new`, the median of up to 200 plans in each of three interleaved runs at a load of 3.3 to 3.4, on this code (measured with the change that made planning cheaper, not in the campaign), next to the campaign's time of one forward transform of `Complex<f64>`:

| n | plan `f64` | plan `f32` | ns a value (`f64`) | transform `f64` | plan / transform |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.17 µs | 0.17 µs | 10.4 | 19 ns | 8.9 |
| 64 | 0.33 µs | 0.29 µs | 5.2 | 57.3 ns | 5.8 |
| 1024 | 2.58 µs | 2.54 µs | 2.52 | 1.33 µs | 1.9 |
| 4096 | 9.04 µs | 9.29 µs | 2.21 | 7.17 µs | 1.3 |
| 65,536 | 136 µs | 142 µs | 2.08 | 189 µs | 0.72 |
| 2²⁰ | 2.13 ms | 2.21 ms | 2.03 | 4.22 ms | 0.50 |
| 10⁶ | 2.02 ms | 2.11 ms | 2.02 | 3.38 ms | 0.60 |
| 1009 (Rader) | 13.0 µs | 15.5 µs | 12.8 | 4.13 µs | 3.1 |
| 10007 (Bluestein) | 174 µs | 209 µs | 17.4 | 116 µs | 1.5 |
| 100003 (Bluestein) | 1.93 ms | 2.28 ms | 19.3 | 1.60 ms | 1.2 |

The calibration in the convolutions' runs fits the planning of the lengths `2^k`, `3·2^k`, `5·2^k` and `7·2^k` from 1024 to 2²¹ (median of the three runs): `Fft::<f32>::new` 189 ns + 2.14 ns a value, `Fft::<f64>::new` 203 ns + 2.08 ns, `RealFft::<f32>::new` 481 ns + 1.26 ns and `RealFft::<f64>::new` 399 ns + 1.60 ns.

### Real transforms

`RealFft::forward` and `inverse` (the allocating forms, out of place), against FFTW's `r2c` and `c2r`, realfft, and vDSP's `zrip` (powers of two) and `DFT_zrop` (`f·2^k`); the last column, `Fft` of `n/2`, is `a`'s complex transform of half the length alone, the core of an even length. The median of 9 rounds of 4 ms, the contenders interleaved, a different one first in each round. Powers of two, then `f·2^k`, then other even lengths, odd lengths with small factors, and primes:

`f64`, `RealFft::forward` (load 1.5 to 5.3):

| n | `a` | FFTW | realfft | vDSP | `Fft` of n/2 | a/fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 23.5 ns | 7.8 ns | 8.1 ns | 14.7 ns | 8.3 ns | 3.01 |
| 32 | 39.1 ns | 15.7 ns | 18.1 ns | 28.8 ns | 18.3 ns | 2.49 |
| 64 | 58.6 ns | 42.7 ns | 40.9 ns | 53.6 ns | 32.3 ns | 1.43 |
| 128 | 105 ns | 98.9 ns | 103 ns | 93.9 ns | 57.4 ns | 1.12 |
| 256 | 222 ns | 195 ns | 228 ns | 197 ns | 122 ns | 1.14 |
| 512 | 453 ns | 468 ns | 493 ns | 402 ns | 264 ns | 1.13 |
| 1024 | 954 ns | 987 ns | 1.09 µs | 899 ns | 587 ns | 1.06 |
| 2048 | 1.93 µs | 2.33 µs | 2.37 µs | 1.9 µs | 1.33 µs | 1.02 |
| 4096 | 4.67 µs | 5.13 µs | 5.22 µs | 4.18 µs | 3.07 µs | 1.12 |
| 8192 | 9.92 µs | 11.2 µs | 11.3 µs | 8.85 µs | 7.26 µs | 1.12 |
| 16,384 | 21.5 µs | 28 µs | 26.3 µs | 21.9 µs | 17.3 µs | 0.98 |
| 65,536 | 96.4 µs | 135 µs | 128 µs | 160 µs | 80.3 µs | 0.75 |
| 262,144 | 471 µs | 603 µs | 574 µs | 782 µs | 426 µs | 0.82 |
| 1,048,576 | 2.2 ms | 2.98 ms | 2.72 ms | 3.67 ms | 1.95 ms | 0.81 |
| 96 | 93.5 ns | 64.4 ns | 107 ns | 106 ns | 49.6 ns | 1.45 |
| 384 | 351 ns | 292 ns | 347 ns | 469 ns | 205 ns | 1.20 |
| 1536 | 1.44 µs | 1.53 µs | 1.69 µs | 2.22 µs | 999 ns | 0.94 |
| 6144 | 7.42 µs | 7.49 µs | 8.02 µs | 10.8 µs | 5.13 µs | 0.99 |
| 80 | 71.6 ns | 54.4 ns | 72.1 ns | 89.1 ns | 38.6 ns | 1.32 |
| 640 | 592 ns | 563 ns | 1.01 µs | 801 ns | 356 ns | 1.05 |
| 5120 | 5.93 µs | 6.15 µs | 9.53 µs | 8.63 µs | 4.19 µs | 0.96 |
| 1000 | 1.06 µs | 1.13 µs | 2.77 µs |  | 686 ns | 0.94 |
| 10,000 | 12.9 µs | 15.8 µs | 37.4 µs |  | 9.98 µs | 0.81 |
| 100,000 | 159 µs | 210 µs | 464 µs |  | 134 µs | 0.76 |
| 1,000,000 | 1.87 ms | 2.85 ms | 6.06 ms |  | 1.64 ms | 0.66 |
| 6000 | 7.76 µs | 8.22 µs | 17.9 µs |  | 5.57 µs | 0.94 |
| 30,000 | 41.7 µs | 52.6 µs | 116 µs |  | 34.6 µs | 0.79 |
| 15 | 20.6 ns | 7.5 ns | 13.9 ns |  |  | 2.75 |
| 45 | 92.5 ns | 59.1 ns | 81.4 ns |  |  | 1.57 |
| 135 | 272 ns | 142 ns | 348 ns |  |  | 1.91 |
| 243 | 559 ns | 302 ns | 997 ns |  |  | 1.85 |
| 625 | 1.18 µs | 975 ns | 3.53 µs |  |  | 1.21 |
| 2187 | 5 µs | 3.53 µs | 12 µs |  |  | 1.41 |
| 3125 | 6.35 µs | 6.37 µs | 22.9 µs |  |  | 1.00 |
| 2401 | 8.04 µs | 4.03 µs | 11.7 µs |  |  | 1.99 |
| 16,807 | 61.2 µs | 38.8 µs | 123 µs |  |  | 1.58 |
| 19,683 | 53.6 µs | 42.5 µs | 157 µs |  |  | 1.26 |
| 177,147 | 582 µs | 499 µs | 1.97 ms |  |  | 1.17 |
| 1155 | 2.47 µs | 1.8 µs | 4.76 µs |  |  | 1.37 |
| 15,015 | 38.1 µs | 35.5 µs | 84.3 µs |  |  | 1.07 |
| 99,225 | 262 µs | 247 µs | 806 µs |  |  | 1.06 |
| 101 | 390 ns | 832 ns | 1.04 µs |  |  | 0.47 |
| 1009 | 4.98 µs | 9.24 µs | 12.1 µs |  |  | 0.54 |
| 10,007 | 130 µs | 153 µs | 205 µs |  |  | 0.85 |
| 100,003 | 1.69 ms | 2.65 ms | 2.91 ms |  |  | 0.64 |

`f32`, `RealFft::forward` (load 1.5 to 5.3):

| n | `a` | FFTW | realfft | vDSP | `Fft` of n/2 | a/fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 15.3 ns | 7.8 ns | 7.9 ns | 16.6 ns | 8.6 ns | 1.96 |
| 32 | 35.7 ns | 15.8 ns | 14.5 ns | 22.1 ns | 14.9 ns | 2.46 |
| 64 | 44.5 ns | 35.8 ns | 32.8 ns | 30.4 ns | 19.6 ns | 1.46 |
| 128 | 66.6 ns | 94 ns | 75.1 ns | 49.3 ns | 32.3 ns | 1.35 |
| 256 | 131 ns | 196 ns | 159 ns | 81.4 ns | 71.4 ns | 1.61 |
| 512 | 248 ns | 419 ns | 352 ns | 270 ns | 144 ns | 0.92 |
| 1024 | 510 ns | 973 ns | 740 ns | 382 ns | 308 ns | 1.34 |
| 2048 | 1.04 µs | 2.15 µs | 1.63 µs | 877 ns | 694 ns | 1.18 |
| 4096 | 2.2 µs | 4.5 µs | 3.42 µs | 846 ns | 1.51 µs | 2.60 |
| 8192 | 5.82 µs | 10.4 µs | 7.53 µs | 1.62 µs | 4.08 µs | 3.59 |
| 16,384 | 11.7 µs | 23.3 µs | 16.1 µs | 7.03 µs | 9.78 µs | 1.66 |
| 65,536 | 50.9 µs | 126 µs | 76.8 µs | 30.1 µs | 39.3 µs | 1.69 |
| 262,144 | 239 µs | 601 µs | 340 µs | 144 µs | 209 µs | 1.66 |
| 1,048,576 | 1.14 ms | 2.7 ms | 1.49 ms | 614 µs | 958 µs | 1.86 |
| 96 | 58.6 ns | 69.1 ns | 87.9 ns | 57.8 ns | 31.2 ns | 1.01 |
| 384 | 197 ns | 311 ns | 253 ns | 197 ns | 118 ns | 1.00 |
| 1536 | 777 ns | 1.53 µs | 1.19 µs | 344 ns | 527 ns | 2.25 |
| 6144 | 4.37 µs | 7.26 µs | 5.53 µs | 1.18 µs | 2.51 µs | 3.70 |
| 80 | 51.1 ns | 54.3 ns | 65.2 ns | 50 ns | 26.5 ns | 1.02 |
| 640 | 327 ns | 561 ns | 714 ns | 181 ns | 197 ns | 1.81 |
| 5120 | 3.06 µs | 5.98 µs | 6.28 µs | 979 ns | 2.08 µs | 3.13 |
| 1000 | 581 ns | 1.04 µs | 2.18 µs |  | 383 ns | 0.56 |
| 10,000 | 7.59 µs | 15.5 µs | 28.7 µs |  | 6.28 µs | 0.49 |
| 100,000 | 82.5 µs | 202 µs | 334 µs |  | 68.1 µs | 0.41 |
| 1,000,000 | 953 µs | 2.62 ms | 4.24 ms |  | 808 µs | 0.36 |
| 6000 | 4.24 µs | 7.98 µs | 13.3 µs |  | 2.89 µs | 0.53 |
| 30,000 | 22.7 µs | 48.7 µs | 90.7 µs |  | 18.9 µs | 0.47 |
| 15 | 13.2 ns | 7.7 ns | 11.4 ns |  |  | 1.71 |
| 45 | 102 ns | 52 ns | 75 ns |  |  | 1.96 |
| 135 | 282 ns | 138 ns | 254 ns |  |  | 2.04 |
| 243 | 544 ns | 337 ns | 785 ns |  |  | 1.62 |
| 625 | 983 ns | 986 ns | 2.81 µs |  |  | 1.00 |
| 2187 | 4.89 µs | 3.66 µs | 9.16 µs |  |  | 1.34 |
| 3125 | 4.94 µs | 5.95 µs | 17.4 µs |  |  | 0.83 |
| 2401 | 7.26 µs | 3.98 µs | 9.05 µs |  |  | 1.83 |
| 16,807 | 52.7 µs | 35.2 µs | 87.8 µs |  |  | 1.50 |
| 19,683 | 46.9 µs | 41.4 µs | 113 µs |  |  | 1.13 |
| 177,147 | 474 µs | 483 µs | 1.51 ms |  |  | 0.98 |
| 1155 | 2.37 µs | 1.73 µs | 3.97 µs |  |  | 1.37 |
| 15,015 | 33 µs | 34.9 µs | 60.5 µs |  |  | 0.95 |
| 99,225 | 239 µs | 240 µs | 609 µs |  |  | 0.99 |
| 101 | 227 ns | 806 ns | 903 ns |  |  | 0.28 |
| 1009 | 2.76 µs | 9.18 µs | 9.88 µs |  |  | 0.30 |
| 10,007 | 71.7 µs | 143 µs | 107 µs |  |  | 0.67 |
| 100,003 | 938 µs | 2.62 ms | 1.6 ms |  |  | 0.59 |

`f64`, `RealFft::inverse` (load 1.5 to 4.8):

| n | `a` | FFTW | realfft | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 44.7 ns | 6.3 ns | 9.7 ns | 7.10 |
| 32 | 29.8 ns | 20.7 ns | 17.9 ns | 1.66 |
| 64 | 50 ns | 38.9 ns | 40.3 ns | 1.29 |
| 128 | 93.8 ns | 88.5 ns | 99.2 ns | 1.06 |
| 256 | 198 ns | 188 ns | 222 ns | 1.05 |
| 512 | 418 ns | 464 ns | 482 ns | 0.90 |
| 1024 | 905 ns | 1 µs | 1.06 µs | 0.90 |
| 2048 | 2 µs | 2.14 µs | 2.32 µs | 0.94 |
| 4096 | 4.45 µs | 5.07 µs | 5.02 µs | 0.89 |
| 8192 | 10.5 µs | 11.6 µs | 11.4 µs | 0.92 |
| 16,384 | 22.4 µs | 26.7 µs | 26.1 µs | 0.86 |
| 65,536 | 104 µs | 135 µs | 126 µs | 0.82 |
| 262,144 | 524 µs | 594 µs | 564 µs | 0.93 |
| 1,048,576 | 2.36 ms | 2.99 ms | 2.63 ms | 0.90 |
| 96 | 78.1 ns | 61.4 ns | 110 ns | 1.27 |
| 384 | 316 ns | 313 ns | 338 ns | 1.01 |
| 1536 | 1.51 µs | 1.54 µs | 1.65 µs | 0.98 |
| 6144 | 7.52 µs | 7.57 µs | 7.89 µs | 0.99 |
| 80 | 62.2 ns | 55.4 ns | 72.7 ns | 1.12 |
| 640 | 550 ns | 621 ns | 1000 ns | 0.88 |
| 5120 | 5.98 µs | 6.51 µs | 9.49 µs | 0.92 |
| 1000 | 1.01 µs | 1.04 µs | 2.75 µs | 0.97 |
| 10,000 | 14.1 µs | 15.9 µs | 36.9 µs | 0.89 |
| 100,000 | 171 µs | 214 µs | 462 µs | 0.80 |
| 1,000,000 | 2 ms | 2.71 ms | 5.91 ms | 0.74 |
| 6000 | 7.82 µs | 8.13 µs | 17.4 µs | 0.96 |
| 30,000 | 46 µs | 55.4 µs | 116 µs | 0.83 |
| 15 | 40.7 ns | 6.6 ns | 16 ns | 6.17 |
| 45 | 128 ns | 51.5 ns | 85.2 ns | 2.49 |
| 135 | 346 ns | 138 ns | 350 ns | 2.50 |
| 243 | 615 ns | 278 ns | 1.01 µs | 2.21 |
| 625 | 1.3 µs | 842 ns | 3.48 µs | 1.55 |
| 2187 | 5.82 µs | 3.41 µs | 12.1 µs | 1.71 |
| 3125 | 7 µs | 5.31 µs | 22.1 µs | 1.32 |
| 2401 | 9.18 µs | 3.97 µs | 11.8 µs | 2.31 |
| 16,807 | 70.8 µs | 44.9 µs | 123 µs | 1.58 |
| 19,683 | 60.9 µs | 49.6 µs | 157 µs | 1.23 |
| 177,147 | 618 µs | 582 µs | 1.97 ms | 1.06 |
| 1155 | 2.87 µs | 1.82 µs | 4.77 µs | 1.58 |
| 15,015 | 42.8 µs | 41.7 µs | 83.6 µs | 1.03 |
| 99,225 | 302 µs | 315 µs | 810 µs | 0.96 |
| 101 | 362 ns | 836 ns | 1.06 µs | 0.43 |
| 1009 | 4.93 µs | 9.58 µs | 12.1 µs | 0.51 |
| 10,007 | 129 µs | 153 µs | 205 µs | 0.84 |
| 100,003 | 1.72 ms | 2.69 ms | 2.91 ms | 0.64 |

`f32`, `RealFft::inverse` (load 1.5 to 4.8):

| n | `a` | FFTW | realfft | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 38.2 ns | 6.4 ns | 8.5 ns | 5.97 |
| 32 | 24.3 ns | 17.8 ns | 12.9 ns | 1.88 |
| 64 | 32.7 ns | 38.9 ns | 28.3 ns | 1.16 |
| 128 | 55.6 ns | 93.8 ns | 71.7 ns | 0.78 |
| 256 | 116 ns | 216 ns | 152 ns | 0.76 |
| 512 | 234 ns | 460 ns | 338 ns | 0.69 |
| 1024 | 490 ns | 1.01 µs | 712 ns | 0.69 |
| 2048 | 1.09 µs | 2.24 µs | 1.59 µs | 0.69 |
| 4096 | 2.34 µs | 5.01 µs | 3.3 µs | 0.71 |
| 8192 | 6.72 µs | 11.2 µs | 7.31 µs | 0.92 |
| 16,384 | 11.4 µs | 24.3 µs | 15.7 µs | 0.73 |
| 65,536 | 52.2 µs | 124 µs | 74.5 µs | 0.70 |
| 262,144 | 265 µs | 588 µs | 332 µs | 0.80 |
| 1,048,576 | 1.18 ms | 2.62 ms | 1.47 ms | 0.81 |
| 96 | 49.3 ns | 62.2 ns | 92 ns | 0.79 |
| 384 | 180 ns | 340 ns | 245 ns | 0.73 |
| 1536 | 820 ns | 1.56 µs | 1.16 µs | 0.71 |
| 6144 | 4.85 µs | 7.77 µs | 5.36 µs | 0.90 |
| 80 | 40.8 ns | 53.3 ns | 65.9 ns | 0.77 |
| 640 | 313 ns | 584 ns | 713 ns | 0.54 |
| 5120 | 3.36 µs | 6.31 µs | 6.26 µs | 0.54 |
| 1000 | 563 ns | 1.03 µs | 2.18 µs | 0.55 |
| 10,000 | 7.87 µs | 15.5 µs | 29 µs | 0.51 |
| 100,000 | 88 µs | 207 µs | 332 µs | 0.43 |
| 1,000,000 | 1.02 ms | 2.54 ms | 4.21 ms | 0.40 |
| 6000 | 4.56 µs | 8.12 µs | 13.9 µs | 0.56 |
| 30,000 | 24.4 µs | 52.3 µs | 88.5 µs | 0.47 |
| 15 | 35.7 ns | 6.8 ns | 13.4 ns | 5.25 |
| 45 | 118 ns | 48.3 ns | 78.7 ns | 2.45 |
| 135 | 323 ns | 142 ns | 259 ns | 2.27 |
| 243 | 545 ns | 289 ns | 789 ns | 1.88 |
| 625 | 1.11 µs | 835 ns | 2.85 µs | 1.33 |
| 2187 | 4.67 µs | 3.38 µs | 9.14 µs | 1.38 |
| 3125 | 5.61 µs | 5.87 µs | 17.1 µs | 0.95 |
| 2401 | 8.51 µs | 3.95 µs | 9.22 µs | 2.16 |
| 16,807 | 61.6 µs | 35.1 µs | 95.3 µs | 1.76 |
| 19,683 | 44.7 µs | 39.9 µs | 122 µs | 1.12 |
| 177,147 | 452 µs | 485 µs | 1.55 ms | 0.93 |
| 1155 | 2.4 µs | 1.78 µs | 4.06 µs | 1.35 |
| 15,015 | 33.7 µs | 35.5 µs | 61.8 µs | 0.95 |
| 99,225 | 230 µs | 257 µs | 610 µs | 0.90 |
| 101 | 234 ns | 787 ns | 876 ns | 0.30 |
| 1009 | 2.89 µs | 9.19 µs | 10 µs | 0.31 |
| 10,007 | 71.3 µs | 144 µs | 109 µs | 0.66 |
| 100,003 | 990 µs | 2.56 ms | 1.6 ms | 0.62 |

Batches, `n × count` sequences in one call, against FFTW's `plan_many_dft_r2c` and realfft in a loop:

`f64`, batches of `RealFft::forward`, n × count (load 2.4 to 4.5):

| n | `a` | FFTW | realfft | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 8 × 2048 | 20.3 µs | 4.67 µs | 8.11 µs | 4.35 |
| 16 × 1024 | 21.1 µs | 5.34 µs | 8.3 µs | 3.95 |
| 32 × 512 | 9.04 µs | 6.39 µs | 9.26 µs | 1.42 |
| 64 × 256 | 9.73 µs | 7.74 µs | 10.6 µs | 1.26 |
| 15 × 1092 | 18 µs | 5.95 µs | 15.2 µs | 3.02 |
| 60 × 272 | 12 µs | 11.9 µs | 15.7 µs | 1.01 |
| 100 × 160 | 12.3 µs | 14 µs | 19.7 µs | 0.88 |

`f32`, batches of `RealFft::forward`, n × count (load 2.4 to 4.5):

| n | `a` | FFTW | realfft | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 8 × 2048 | 14.8 µs | 3.58 µs | 7.41 µs | 4.14 |
| 16 × 1024 | 11.8 µs | 4.53 µs | 7.93 µs | 2.60 |
| 32 × 512 | 5.87 µs | 5.92 µs | 7.46 µs | 0.99 |
| 64 × 256 | 5.98 µs | 7.45 µs | 8.29 µs | 0.80 |
| 15 × 1092 | 10.7 µs | 5.44 µs | 12.1 µs | 1.96 |
| 60 × 272 | 7.23 µs | 11.7 µs | 13.7 µs | 0.62 |
| 100 × 160 | 7.36 µs | 13.3 µs | 15.8 µs | 0.55 |

### Cosine and sine transforms

`Dct::transform_to` and `Dst::transform_to` against FFTW's `r2r` kinds and, for `f32` and the lengths it has, `vDSP_DCT_Execute`; the same rounds as the real transforms:

`f64`, DCT-II (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 37.1 ns | 24.4 ns | 1.52 |
| 64 | 96.9 ns | 76 ns | 1.28 |
| 256 | 362 ns | 326 ns | 1.11 |
| 1024 | 1.3 µs | 1.45 µs | 0.90 |
| 4096 | 6.93 µs | 7 µs | 0.99 |
| 16,384 | 31.9 µs | 41.4 µs | 0.77 |
| 65,536 | 135 µs | 216 µs | 0.63 |
| 1,048,576 | 3.33 ms | 4.99 ms | 0.67 |
| 48 | 66.5 ns | 77.8 ns | 0.85 |
| 1000 | 1.39 µs | 1.64 µs | 0.85 |
| 10,000 | 19.2 µs | 26.3 µs | 0.73 |
| 3000 | 5.13 µs | 5.23 µs | 0.98 |
| 255 | 617 ns | 630 ns | 0.98 |
| 1023 | 4.75 µs | 4.13 µs | 1.15 |

`f32`, DCT-II (load 2.4 to 4.4):

| n | `a` | FFTW | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 22.8 ns | 24.1 ns | 10.3 ns | 2.21 |
| 64 | 55.8 ns | 67.6 ns | 27.5 ns | 2.03 |
| 256 | 178 ns | 325 ns | 98.5 ns | 1.81 |
| 1024 | 729 ns | 1.39 µs | 412 ns | 1.77 |
| 4096 | 3.65 µs | 6.61 µs | 1.82 µs | 2.00 |
| 16,384 | 15.9 µs | 31.3 µs | 8.53 µs | 1.86 |
| 65,536 | 73 µs | 158 µs | 106 µs | 0.69 |
| 1,048,576 | 1.75 ms | 4.18 ms | 2.22 ms | 0.79 |
| 48 | 50 ns | 73.7 ns | 39 ns | 1.28 |
| 1000 | 780 ns | 1.52 µs |  | 0.51 |
| 10,000 | 9.97 µs | 19.6 µs |  | 0.51 |
| 3000 | 2.63 µs | 5.12 µs |  | 0.51 |
| 255 | 562 ns | 625 ns |  | 0.90 |
| 1023 | 3.85 µs | 4.31 µs |  | 0.89 |

`f64`, DCT-III (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 55.1 ns | 23.9 ns | 2.31 |
| 64 | 96.2 ns | 81.4 ns | 1.18 |
| 256 | 319 ns | 332 ns | 0.96 |
| 1024 | 1.32 µs | 1.48 µs | 0.90 |
| 4096 | 6.56 µs | 7.15 µs | 0.92 |
| 16,384 | 29.4 µs | 35.7 µs | 0.82 |
| 65,536 | 130 µs | 191 µs | 0.68 |
| 1,048,576 | 2.76 ms | 4.55 ms | 0.61 |
| 48 | 64.3 ns | 79.6 ns | 0.81 |
| 1000 | 1.51 µs | 1.64 µs | 0.92 |
| 10,000 | 17.9 µs | 22.6 µs | 0.79 |
| 3000 | 5.05 µs | 5.13 µs | 0.98 |
| 255 | 679 ns | 684 ns | 0.99 |
| 1023 | 5.04 µs | 4.16 µs | 1.21 |

`f32`, DCT-III (load 2.4 to 4.4):

| n | `a` | FFTW | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 46.7 ns | 23.2 ns | 13.1 ns | 3.56 |
| 64 | 55.3 ns | 72 ns | 41.4 ns | 1.34 |
| 256 | 172 ns | 325 ns | 130 ns | 1.32 |
| 1024 | 721 ns | 1.41 µs | 548 ns | 1.32 |
| 4096 | 3.6 µs | 6.62 µs | 2.48 µs | 1.45 |
| 16,384 | 16.5 µs | 31.4 µs | 11.5 µs | 1.44 |
| 65,536 | 72.7 µs | 158 µs | 124 µs | 0.59 |
| 1,048,576 | 1.63 ms | 4.19 ms | 2.82 ms | 0.58 |
| 48 | 50.5 ns | 75.4 ns | 47 ns | 1.07 |
| 1000 | 797 ns | 1.52 µs |  | 0.53 |
| 10,000 | 9.45 µs | 19.8 µs |  | 0.48 |
| 3000 | 2.79 µs | 5.23 µs |  | 0.53 |
| 255 | 563 ns | 628 ns |  | 0.90 |
| 1023 | 3.96 µs | 4.31 µs |  | 0.92 |

`f64`, DCT-IV (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 24.1 ns | 25.6 ns | 0.94 |
| 64 | 77.2 ns | 78.3 ns | 0.99 |
| 256 | 302 ns | 294 ns | 1.03 |
| 1024 | 1.3 µs | 1.49 µs | 0.87 |
| 4096 | 6.36 µs | 6.63 µs | 0.96 |
| 16,384 | 28.2 µs | 34.5 µs | 0.82 |
| 65,536 | 124 µs | 182 µs | 0.68 |
| 1,048,576 | 2.67 ms | 4.67 ms | 0.57 |
| 48 | 63.7 ns | 102 ns | 0.63 |
| 1000 | 1.39 µs | 1.55 µs | 0.90 |
| 10,000 | 16.8 µs | 20.3 µs | 0.83 |
| 3000 | 4.61 µs | 5.35 µs | 0.86 |
| 255 | 956 ns | 673 ns | 1.42 |
| 1023 | 5.92 µs | 4.4 µs | 1.34 |

`f32`, DCT-IV (load 2.4 to 4.4):

| n | `a` | FFTW | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 21.2 ns | 24.8 ns | 14.2 ns | 1.49 |
| 64 | 60.7 ns | 68.9 ns | 41 ns | 1.48 |
| 256 | 210 ns | 298 ns | 132 ns | 1.59 |
| 1024 | 837 ns | 1.37 µs | 545 ns | 1.54 |
| 4096 | 4.04 µs | 6.31 µs | 2.4 µs | 1.68 |
| 16,384 | 15.7 µs | 30 µs | 11.8 µs | 1.32 |
| 65,536 | 70.8 µs | 144 µs | 138 µs | 0.51 |
| 1,048,576 | 1.48 ms | 3.89 ms | 2.87 ms | 0.52 |
| 48 | 50.9 ns | 80.2 ns | 51.1 ns | 1.00 |
| 1000 | 896 ns | 1.52 µs |  | 0.59 |
| 10,000 | 10.2 µs | 19 µs |  | 0.54 |
| 3000 | 2.88 µs | 4.97 µs |  | 0.58 |
| 255 | 930 ns | 684 ns |  | 1.36 |
| 1023 | 5.32 µs | 4.61 µs |  | 1.15 |

`f64`, DST-II (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 37.2 ns | 27.6 ns | 1.35 |
| 64 | 97.2 ns | 75.9 ns | 1.28 |
| 256 | 362 ns | 325 ns | 1.11 |
| 1024 | 1.32 µs | 1.45 µs | 0.91 |
| 4096 | 7.06 µs | 6.95 µs | 1.02 |
| 16,384 | 31.3 µs | 40.9 µs | 0.77 |
| 65,536 | 136 µs | 215 µs | 0.63 |
| 1,048,576 | 3.33 ms | 4.98 ms | 0.67 |
| 48 | 66.2 ns | 76.6 ns | 0.86 |
| 1000 | 1.39 µs | 1.63 µs | 0.86 |
| 10,000 | 19.4 µs | 26.3 µs | 0.74 |
| 3000 | 5.36 µs | 5.18 µs | 1.03 |
| 255 | 616 ns | 614 ns | 1.00 |
| 1023 | 4.62 µs | 4.15 µs | 1.11 |

`f32`, DST-II (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 23 ns | 24.6 ns | 0.93 |
| 64 | 55.6 ns | 67.6 ns | 0.82 |
| 256 | 176 ns | 325 ns | 0.54 |
| 1024 | 717 ns | 1.4 µs | 0.51 |
| 4096 | 4.36 µs | 6.59 µs | 0.66 |
| 16,384 | 15.3 µs | 31.3 µs | 0.49 |
| 65,536 | 72.9 µs | 158 µs | 0.46 |
| 1,048,576 | 1.74 ms | 4.15 ms | 0.42 |
| 48 | 50.4 ns | 75.9 ns | 0.66 |
| 1000 | 786 ns | 1.52 µs | 0.52 |
| 10,000 | 9.88 µs | 19.8 µs | 0.50 |
| 3000 | 2.62 µs | 5.17 µs | 0.51 |
| 255 | 563 ns | 626 ns | 0.90 |
| 1023 | 3.82 µs | 4.31 µs | 0.89 |

`f64`, DST-IV (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 24.2 ns | 26.2 ns | 0.92 |
| 64 | 73.6 ns | 76.5 ns | 0.96 |
| 256 | 283 ns | 295 ns | 0.96 |
| 1024 | 1.24 µs | 1.5 µs | 0.82 |
| 4096 | 6.23 µs | 6.66 µs | 0.94 |
| 16,384 | 28.1 µs | 34.7 µs | 0.81 |
| 65,536 | 124 µs | 181 µs | 0.69 |
| 1,048,576 | 2.66 ms | 4.66 ms | 0.57 |
| 48 | 60.9 ns | 101 ns | 0.60 |
| 1000 | 1.34 µs | 1.57 µs | 0.85 |
| 10,000 | 16.2 µs | 20.4 µs | 0.80 |
| 3000 | 4.51 µs | 5.36 µs | 0.84 |
| 255 | 989 ns | 707 ns | 1.40 |
| 1023 | 6.09 µs | 4.44 µs | 1.37 |

`f32`, DST-IV (load 2.4 to 4.4):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 | 21.2 ns | 24.5 ns | 0.87 |
| 64 | 60.6 ns | 68.9 ns | 0.88 |
| 256 | 191 ns | 297 ns | 0.64 |
| 1024 | 779 ns | 1.38 µs | 0.57 |
| 4096 | 3.91 µs | 6.31 µs | 0.62 |
| 16,384 | 15.1 µs | 30 µs | 0.50 |
| 65,536 | 68.8 µs | 146 µs | 0.47 |
| 1,048,576 | 1.42 ms | 3.92 ms | 0.36 |
| 48 | 50.9 ns | 80.9 ns | 0.63 |
| 1000 | 850 ns | 1.52 µs | 0.56 |
| 10,000 | 9.77 µs | 19.1 µs | 0.51 |
| 3000 | 2.7 µs | 4.94 µs | 0.55 |
| 255 | 957 ns | 696 ns | 1.37 |
| 1023 | 5.4 µs | 4.63 µs | 1.17 |

Types I, at `2^k ± 1` (the split-radix step for these odd lengths) and at even lengths (the real transform of the extension):

`f64`, DCT-I (load 3.0 to 4.3):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 17 | 40.2 ns | 36.9 ns | 1.09 |
| 65 | 168 ns | 112 ns | 1.49 |
| 257 | 589 ns | 404 ns | 1.46 |
| 1025 | 2.36 µs | 1.71 µs | 1.38 |
| 4097 | 9.87 µs | 7.52 µs | 1.31 |
| 16,385 | 44 µs | 36.2 µs | 1.22 |
| 65,537 | 212 µs | 177 µs | 1.20 |
| 64 | 151 ns | 180 ns | 0.84 |
| 1000 | 5.03 µs | 7.73 µs | 0.65 |
| 1024 | 5.19 µs | 8.18 µs | 0.63 |
| 4096 | 13.8 µs | 16.9 µs | 0.82 |

`f32`, DCT-I (load 3.0 to 4.3):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 17 | 35 ns | 38.1 ns | 0.92 |
| 65 | 150 ns | 99.3 ns | 1.51 |
| 257 | 512 ns | 361 ns | 1.42 |
| 1025 | 1.91 µs | 1.59 µs | 1.21 |
| 4097 | 7.32 µs | 6.83 µs | 1.07 |
| 16,385 | 30.4 µs | 31.4 µs | 0.97 |
| 65,537 | 130 µs | 149 µs | 0.87 |
| 64 | 115 ns | 160 ns | 0.72 |
| 1000 | 3.21 µs | 7.3 µs | 0.44 |
| 1024 | 3.31 µs | 7.71 µs | 0.43 |
| 4096 | 8.52 µs | 15.2 µs | 0.56 |

`f64`, DST-I (load 3.0 to 4.3):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 15 | 22.9 ns | 35.4 ns | 0.65 |
| 63 | 153 ns | 115 ns | 1.33 |
| 255 | 578 ns | 411 ns | 1.41 |
| 1023 | 2.37 µs | 1.71 µs | 1.38 |
| 4095 | 10.9 µs | 7.51 µs | 1.45 |
| 16,383 | 44 µs | 36.3 µs | 1.21 |
| 65,535 | 188 µs | 177 µs | 1.06 |
| 64 | 162 ns | 186 ns | 0.87 |
| 1000 | 2.9 µs | 4.36 µs | 0.67 |
| 1024 | 5.01 µs | 8.81 µs | 0.57 |
| 4096 | 27.2 µs | 93.6 µs | 0.29 |

`f32`, DST-I (load 3.0 to 4.3):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 15 | 18.1 ns | 39.9 ns | 0.45 |
| 63 | 130 ns | 104 ns | 1.24 |
| 255 | 455 ns | 362 ns | 1.26 |
| 1023 | 1.81 µs | 1.6 µs | 1.13 |
| 4095 | 6.77 µs | 6.77 µs | 1.00 |
| 16,383 | 29.6 µs | 31.2 µs | 0.95 |
| 65,535 | 121 µs | 150 µs | 0.80 |
| 64 | 109 ns | 226 ns | 0.48 |
| 1000 | 1.72 µs | 4.26 µs | 0.40 |
| 1024 | 3.29 µs | 8.01 µs | 0.41 |
| 4096 | 16.1 µs | 85.2 µs | 0.19 |

### Two and three dimensions

`Fft2::forward` and `Fft3::forward` in place, against FFTW's `plan_dft_2d` and `plan_dft_3d` in place and, in two dimensions, `vDSP_fft2d_zip` (powers of two):

`f64`, `Fft2`, rows × columns (load 2.3 to 4.9):

| n | `a` | FFTW | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 64 × 64 | 9.76 µs | 8.82 µs | 20.5 µs | 1.11 |
| 128 × 128 | 42.1 µs | 60.4 µs | 285 µs | 0.70 |
| 256 × 256 | 181 µs | 270 µs | 2.2 ms | 0.67 |
| 512 × 512 | 839 µs | 1.18 ms | 11 ms | 0.71 |
| 1024 × 1024 | 6.71 ms | 5.63 ms | 64.9 ms | 1.19 |
| 2048 × 2048 | 29 ms | 29.6 ms | 383 ms | 0.98 |
| 4096 × 4096 | 125 ms | 144 ms | 1.61 s | 0.86 |
| 480 × 640 | 1.01 ms | 1.41 ms |  | 0.72 |
| 1000 × 1000 | 3.8 ms | 5.47 ms |  | 0.69 |
| 4096 × 64 | 1.58 ms | 1.14 ms | 6.05 ms | 1.38 |
| 64 × 4096 | 952 µs | 1.19 ms | 14.3 ms | 0.80 |

`f32`, `Fft2`, rows × columns (load 2.3 to 4.9):

| n | `a` | FFTW | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: |
| 64 × 64 | 6.34 µs | 9.56 µs | 3.17 µs | 2.00 |
| 128 × 128 | 27.4 µs | 56.8 µs | 14.3 µs | 1.91 |
| 256 × 256 | 115 µs | 254 µs | 84.4 µs | 1.37 |
| 512 × 512 | 524 µs | 1.13 ms | 458 µs | 1.15 |
| 1024 × 1024 | 4.43 ms | 5.31 ms | 2.46 ms | 1.80 |
| 2048 × 2048 | 24.5 ms | 24.7 ms | 11.5 ms | 2.13 |
| 4096 × 4096 | 112 ms | 118 ms | 51.3 ms | 2.19 |
| 480 × 640 | 626 µs | 1.4 ms |  | 0.45 |
| 1000 × 1000 | 2.27 ms | 5.33 ms |  | 0.43 |
| 4096 × 64 | 1.49 ms | 1.09 ms | 576 µs | 2.58 |
| 64 × 4096 | 599 µs | 1.14 ms | 561 µs | 1.07 |

`f64`, `Fft3` (load 2.3 to 4.9):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 × 16 × 16 | 16.1 µs | 7.56 µs | 2.14 |
| 32 × 32 × 32 | 136 µs | 108 µs | 1.26 |
| 64 × 64 × 64 | 1.2 ms | 1.07 ms | 1.12 |
| 128 × 128 × 128 | 10.5 ms | 13.4 ms | 0.78 |
| 256 × 256 × 256 | 96.7 ms | 117 ms | 0.82 |
| 100 × 100 × 100 | 4.39 ms | 5.41 ms | 0.81 |
| 8 × 256 × 256 | 1.97 ms | 2.55 ms | 0.77 |
| 256 × 256 × 8 | 2.38 ms | 2.46 ms | 0.97 |

`f32`, `Fft3` (load 2.3 to 4.9):

| n | `a` | FFTW | a/fastest |
| ---: | ---: | ---: | ---: |
| 16 × 16 × 16 | 13.8 µs | 7.49 µs | 1.84 |
| 32 × 32 × 32 | 88.3 µs | 100 µs | 0.88 |
| 64 × 64 × 64 | 673 µs | 1.06 ms | 0.63 |
| 128 × 128 × 128 | 5.88 ms | 12 ms | 0.49 |
| 256 × 256 × 256 | 54.5 ms | 106 ms | 0.51 |
| 100 × 100 × 100 | 3.01 ms | 4.84 ms | 0.62 |
| 8 × 256 × 256 | 1.33 ms | 2.39 ms | 0.56 |
| 256 × 256 × 8 | 1.35 ms | 2.37 ms | 0.57 |

### Convolution

`convolve(&a, &b)` of random operands, called again and again, so that its plans come from the cache (as a repeated call finds them; a plan larger than the cache's 16 MiB would be made in every call). The references: FFTW (`r2c` and `c2r` for real data, complex transforms in place) and rustfft or realfft, planned before the clock starts, each timed at the padded length (the cheapest 7-smooth length of at least `len(a) + len(b) − 1`) and at overlap-add lengths `2^k` from twice the shorter operand up to 64 times that, the fastest of them kept (the best case of each library); and vDSP's direct `vDSP_conv` and `vDSP_zconv`, with the copy of the input between zeros (and the split of complex values) they need timed with them. `a`, plan reused, is the same overlap-add naively on `a`'s public `RealFft` and `Fft` planned once, block by block, at the same lengths as the references: it shows `a`'s transforms in the place of the others'. The median of 5 rounds of 20 ms (`a` and vDSP interleaved; the lengths of each FFT-based reference raced among themselves), times a call; "–" where vDSP's direct sums would take too long:

`convolve` of `f32` (load 2.1 to 3.8):

| len(a) | len(b) | `a` | `a`, plan reused | FFTW | rustfft | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 30 ns | 113 ns | 76 ns | 73 ns | 24 ns | 1.25 (vDSP) |
| 16 | 8 | 32 ns | 121 ns | 78 ns | 75 ns | 26 ns | 1.23 (vDSP) |
| 16 | 16 | 37 ns | 136 ns | 83 ns | 84 ns | 35 ns | 1.06 (vDSP) |
| 100 | 3 | 61 ns | 274 ns | 223 ns | 224 ns | 40 ns | 1.52 (vDSP) |
| 100 | 8 | 69 ns | 270 ns | 229 ns | 223 ns | 51 ns | 1.35 (vDSP) |
| 100 | 16 | 91 ns | 282 ns | 324 ns | 305 ns | 65 ns | 1.40 (vDSP) |
| 100 | 32 | 134 ns | 336 ns | 424 ns | 381 ns | 110 ns | 1.22 (vDSP) |
| 100 | 64 | 254 ns | 353 ns | 509 ns | 491 ns | 224 ns | 1.13 (vDSP) |
| 1000 | 3 | 139 ns | 1.36 µs | 1.57 µs | 1.42 µs | 189 ns | 0.74 (vDSP) |
| 1000 | 8 | 200 ns | 1.53 µs | 1.92 µs | 1.56 µs | 273 ns | 0.73 (vDSP) |
| 1000 | 16 | 350 ns | 1.53 µs | 2.13 µs | 1.77 µs | 398 ns | 0.88 (vDSP) |
| 1000 | 32 | 636 ns | 1.75 µs | 2.51 µs | 2.14 µs | 723 ns | 0.88 (vDSP) |
| 1000 | 64 | 1.48 µs | 2.07 µs | 3.03 µs | 2.54 µs | 1.33 µs | 1.12 (vDSP) |
| 1000 | 128 | 2.86 µs | 2.16 µs | 3.6 µs | 2.86 µs | 2.91 µs | 1.00 (rustfft) |
| 1000 | 256 | 2.58 µs | 2.77 µs | 4.6 µs | 3.66 µs | 6.91 µs | 0.70 (rustfft) |
| 1000 | 1000 | 3.97 µs | 4.22 µs | 7.69 µs | 14.5 µs | 45.5 µs | 0.52 (FFTW) |
| 10,000 | 3 | 1.11 µs | 13 µs | 15.5 µs | 13.3 µs | 2.28 µs | 0.49 (vDSP) |
| 10,000 | 8 | 1.63 µs | 13.2 µs | 18.9 µs | 14.8 µs | 2.96 µs | 0.55 (vDSP) |
| 10,000 | 16 | 3.15 µs | 13.6 µs | 19.6 µs | 16.9 µs | 4.08 µs | 0.77 (vDSP) |
| 10,000 | 32 | 6.08 µs | 13.8 µs | 21.5 µs | 18.1 µs | 7.03 µs | 0.87 (vDSP) |
| 10,000 | 64 | 11.8 µs | 14.9 µs | 24.4 µs | 19.7 µs | 12.9 µs | 0.91 (vDSP) |
| 10,000 | 128 | 23.3 µs | 16.5 µs | 28.1 µs | 22.3 µs | 26.6 µs | 1.05 (rustfft) |
| 10,000 | 256 | 20.5 µs | 18.5 µs | 33.4 µs | 25.6 µs | 57.6 µs | 0.80 (rustfft) |
| 10,000 | 1000 | 27.3 µs | 29 µs | 49.5 µs | 36.6 µs | 258 µs | 0.75 (rustfft) |
| 10,000 | 10,000 | 48.9 µs | 54 µs | 110 µs | 136 µs | 4.71 ms | 0.45 (FFTW) |
| 100,000 | 3 | 9.9 µs | 126 µs | 153 µs | 130 µs | 16.1 µs | 0.62 (vDSP) |
| 100,000 | 8 | 15.7 µs | 126 µs | 185 µs | 142 µs | 25.4 µs | 0.62 (vDSP) |
| 100,000 | 16 | 32.5 µs | 129 µs | 191 µs | 161 µs | 38.3 µs | 0.85 (vDSP) |
| 100,000 | 32 | 58.1 µs | 132 µs | 209 µs | 172 µs | 68.6 µs | 0.85 (vDSP) |
| 100,000 | 64 | 114 µs | 139 µs | 233 µs | 187 µs | 127 µs | 0.90 (vDSP) |
| 100,000 | 128 | 165 µs | 147 µs | 264 µs | 199 µs | 259 µs | 0.83 (rustfft) |
| 100,000 | 256 | 175 µs | 163 µs | 288 µs | 218 µs | 558 µs | 0.80 (rustfft) |
| 100,000 | 1000 | 219 µs | 189 µs | 349 µs | 256 µs | 2.35 ms | 0.85 (rustfft) |
| 100,000 | 10,000 | 278 µs | 308 µs | 671 µs | 432 µs | 26 ms | 0.64 (rustfft) |
| 100,000 | 100,000 | 603 µs | 592 µs | 1.47 ms | 2.21 ms | – | 0.41 (FFTW) |
| 1,000,000 | 3 | 95.6 µs | 1.23 ms | 1.52 ms | 1.27 ms | 137 µs | 0.70 (vDSP) |
| 1,000,000 | 8 | 154 µs | 1.23 ms | 1.83 ms | 1.42 ms | 242 µs | 0.64 (vDSP) |
| 1,000,000 | 16 | 310 µs | 1.27 ms | 1.88 ms | 1.6 ms | 377 µs | 0.82 (vDSP) |
| 1,000,000 | 32 | 569 µs | 1.3 ms | 2.08 ms | 1.72 ms | 661 µs | 0.86 (vDSP) |
| 1,000,000 | 64 | 1.17 ms | 1.35 ms | 2.33 ms | 1.83 ms | 1.26 ms | 0.93 (vDSP) |
| 1,000,000 | 128 | 1.5 ms | 1.45 ms | 2.56 ms | 1.95 ms | 2.55 ms | 0.77 (rustfft) |
| 1,000,000 | 256 | 1.74 ms | 1.57 ms | 2.79 ms | 2.09 ms | 5.51 ms | 0.84 (rustfft) |
| 1,000,000 | 1000 | 1.81 ms | 1.74 ms | 3.35 ms | 2.45 ms | 23 ms | 0.74 (rustfft) |
| 1,000,000 | 10,000 | 2.33 ms | 2.43 ms | 5.28 ms | 3.36 ms | – | 0.69 (rustfft) |
| 1,000,000 | 100,000 | 3.3 ms | 4.27 ms | 9.28 ms | 5.63 ms | – | 0.59 (rustfft) |

`convolve` of `f64` (load 1.6 to 4.9):

| len(a) | len(b) | `a` | `a`, plan reused | FFTW | rustfft | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 43 ns | 128 ns | 85 ns | 88 ns | 24 ns | 1.79 (vDSP) |
| 16 | 8 | 46 ns | 143 ns | 87 ns | 89 ns | 30 ns | 1.53 (vDSP) |
| 16 | 16 | 53 ns | 156 ns | 93 ns | 115 ns | 38 ns | 1.39 (vDSP) |
| 100 | 3 | 89 ns | 373 ns | 248 ns | 310 ns | 64 ns | 1.39 (vDSP) |
| 100 | 8 | 109 ns | 374 ns | 256 ns | 311 ns | 71 ns | 1.54 (vDSP) |
| 100 | 16 | 147 ns | 428 ns | 354 ns | 421 ns | 109 ns | 1.35 (vDSP) |
| 100 | 32 | 234 ns | 519 ns | 436 ns | 529 ns | 201 ns | 1.16 (vDSP) |
| 100 | 64 | 443 ns | 575 ns | 552 ns | 620 ns | 417 ns | 1.06 (vDSP) |
| 1000 | 3 | 246 ns | 2.24 µs | 1.81 µs | 2.1 µs | 451 ns | 0.55 (vDSP) |
| 1000 | 8 | 404 ns | 2.49 µs | 2.14 µs | 2.21 µs | 515 ns | 0.78 (vDSP) |
| 1000 | 16 | 679 ns | 2.47 µs | 2.24 µs | 2.57 µs | 794 ns | 0.86 (vDSP) |
| 1000 | 32 | 1.22 µs | 2.96 µs | 2.75 µs | 3.13 µs | 1.5 µs | 0.81 (vDSP) |
| 1000 | 64 | 2.66 µs | 3.46 µs | 3.5 µs | 3.72 µs | 2.61 µs | 1.02 (vDSP) |
| 1000 | 128 | 5.34 µs | 3.9 µs | 3.98 µs | 4.3 µs | 5.16 µs | 1.34 (FFTW) |
| 1000 | 256 | 4.53 µs | 4.99 µs | 5.09 µs | 5.44 µs | 11.2 µs | 0.89 (FFTW) |
| 1000 | 1000 | 7.97 µs | 8.24 µs | 8.69 µs | 18.6 µs | 68.4 µs | 0.92 (FFTW) |
| 10,000 | 3 | 1.92 µs | 21.8 µs | 18.2 µs | 20.1 µs | 5.16 µs | 0.37 (vDSP) |
| 10,000 | 8 | 3.72 µs | 22.7 µs | 21.2 µs | 21.9 µs | 5.4 µs | 0.69 (vDSP) |
| 10,000 | 16 | 6.23 µs | 23.8 µs | 22.2 µs | 25.1 µs | 8.35 µs | 0.75 (vDSP) |
| 10,000 | 32 | 11.7 µs | 25.4 µs | 25.2 µs | 27.2 µs | 14.8 µs | 0.79 (vDSP) |
| 10,000 | 64 | 22.5 µs | 27.8 µs | 28.2 µs | 30.4 µs | 24.5 µs | 0.92 (vDSP) |
| 10,000 | 128 | 33.9 µs | 31 µs | 31.6 µs | 33.8 µs | 46.6 µs | 1.07 (FFTW) |
| 10,000 | 256 | 36.2 µs | 35.1 µs | 35.3 µs | 38.5 µs | 92.6 µs | 1.03 (FFTW) |
| 10,000 | 1000 | 46.4 µs | 52.1 µs | 54.3 µs | 57.6 µs | 375 µs | 0.85 (FFTW) |
| 10,000 | 10,000 | 92.4 µs | 102 µs | 121 µs | 205 µs | 7.16 ms | 0.76 (FFTW) |
| 100,000 | 3 | 18.2 µs | 208 µs | 174 µs | 192 µs | 42.1 µs | 0.43 (vDSP) |
| 100,000 | 8 | 34.6 µs | 218 µs | 201 µs | 210 µs | 49 µs | 0.71 (vDSP) |
| 100,000 | 16 | 62.1 µs | 227 µs | 215 µs | 242 µs | 76.5 µs | 0.81 (vDSP) |
| 100,000 | 32 | 113 µs | 242 µs | 239 µs | 260 µs | 144 µs | 0.79 (vDSP) |
| 100,000 | 64 | 220 µs | 255 µs | 264 µs | 281 µs | 238 µs | 0.93 (vDSP) |
| 100,000 | 128 | 279 µs | 275 µs | 285 µs | 307 µs | 453 µs | 0.98 (FFTW) |
| 100,000 | 256 | 296 µs | 305 µs | 305 µs | 330 µs | 896 µs | 0.97 (FFTW) |
| 100,000 | 1000 | 351 µs | 353 µs | 403 µs | 422 µs | 3.44 ms | 0.87 (FFTW) |
| 100,000 | 10,000 | 521 µs | 586 µs | 766 µs | 731 µs | 39.2 ms | 0.71 (rustfft) |
| 100,000 | 100,000 | 1.15 ms | 1.14 ms | 1.46 ms | 2.98 ms | – | 0.78 (FFTW) |
| 1,000,000 | 3 | 182 µs | 2.05 ms | 1.71 ms | 1.91 ms | 418 µs | 0.43 (vDSP) |
| 1,000,000 | 8 | 335 µs | 2.15 ms | 1.99 ms | 2.08 ms | 483 µs | 0.69 (vDSP) |
| 1,000,000 | 16 | 631 µs | 2.26 ms | 2.12 ms | 2.41 ms | 762 µs | 0.83 (vDSP) |
| 1,000,000 | 32 | 1.13 ms | 2.43 ms | 2.38 ms | 2.57 ms | 1.42 ms | 0.79 (vDSP) |
| 1,000,000 | 64 | 2.21 ms | 2.57 ms | 2.62 ms | 2.78 ms | 2.38 ms | 0.93 (vDSP) |
| 1,000,000 | 128 | 2.67 ms | 2.76 ms | 2.83 ms | 3.05 ms | 4.53 ms | 0.94 (FFTW) |
| 1,000,000 | 256 | 2.96 ms | 2.99 ms | 3.01 ms | 3.28 ms | 8.9 ms | 0.98 (FFTW) |
| 1,000,000 | 1000 | 3.48 ms | 3.45 ms | 3.94 ms | 4.12 ms | 34.1 ms | 0.88 (FFTW) |
| 1,000,000 | 10,000 | 4.58 ms | 4.61 ms | 6.16 ms | 5.82 ms | – | 0.79 (rustfft) |
| 1,000,000 | 100,000 | 7.35 ms | 8.39 ms | 10.4 ms | 9.67 ms | – | 0.76 (rustfft) |

`convolve` of `Complex<f32>` (load 1.2 to 3.4):

| len(a) | len(b) | `a` | `a`, plan reused | FFTW | rustfft | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 42 ns | 109 ns | 118 ns | 89 ns | 56 ns | 0.75 (vDSP) |
| 16 | 8 | 52 ns | 104 ns | 120 ns | 74 ns | 125 ns | 0.70 (rustfft) |
| 16 | 16 | 73 ns | 114 ns | 142 ns | 89 ns | 302 ns | 0.82 (rustfft) |
| 100 | 3 | 94 ns | 284 ns | 358 ns | 233 ns | 235 ns | 0.40 (rustfft) |
| 100 | 8 | 143 ns | 281 ns | 402 ns | 234 ns | 525 ns | 0.61 (rustfft) |
| 100 | 16 | 215 ns | 341 ns | 562 ns | 324 ns | 1.07 µs | 0.66 (rustfft) |
| 100 | 32 | 488 ns | 451 ns | 872 ns | 617 ns | 2.28 µs | 0.79 (rustfft) |
| 100 | 64 | 947 ns | 464 ns | 1.01 µs | 701 ns | 5.57 µs | 1.35 (rustfft) |
| 1000 | 3 | 330 ns | 1.81 µs | 2.75 µs | 1.57 µs | 2.21 µs | 0.21 (rustfft) |
| 1000 | 8 | 744 ns | 1.92 µs | 3.46 µs | 1.82 µs | 4.63 µs | 0.41 (rustfft) |
| 1000 | 16 | 1.53 µs | 2.02 µs | 4.38 µs | 2.61 µs | 9.33 µs | 0.59 (rustfft) |
| 1000 | 32 | 2.7 µs | 2.36 µs | 5.27 µs | 3.14 µs | 17.9 µs | 0.86 (rustfft) |
| 1000 | 64 | 5.25 µs | 2.78 µs | 6.43 µs | 3.92 µs | 36.2 µs | 1.34 (rustfft) |
| 1000 | 128 | 3.82 µs | 3.13 µs | 7.75 µs | 4.46 µs | 79.3 µs | 0.86 (rustfft) |
| 1000 | 256 | 4.26 µs | 4.03 µs | 9.99 µs | 5.76 µs | 180 µs | 0.74 (rustfft) |
| 1000 | 1000 | 7.76 µs | 7.13 µs | 16.9 µs | 24.5 µs | 1.23 ms | 0.46 (FFTW) |
| 10,000 | 3 | 2.77 µs | 17.2 µs | 27.4 µs | 15.3 µs | 22.4 µs | 0.18 (rustfft) |
| 10,000 | 8 | 6.8 µs | 18.2 µs | 33.9 µs | 18.4 µs | 47.1 µs | 0.37 (rustfft) |
| 10,000 | 16 | 12.3 µs | 19.3 µs | 41.9 µs | 25 µs | 92.3 µs | 0.49 (rustfft) |
| 10,000 | 32 | 23.4 µs | 20.5 µs | 46.2 µs | 28.7 µs | 174 µs | 0.82 (rustfft) |
| 10,000 | 64 | 26.9 µs | 22.3 µs | 51.6 µs | 31.2 µs | 339 µs | 0.86 (rustfft) |
| 10,000 | 128 | 31.4 µs | 25.2 µs | 59.6 µs | 35.3 µs | 711 µs | 0.89 (rustfft) |
| 10,000 | 256 | 36.2 µs | 29.7 µs | 71.5 µs | 40.7 µs | 1.48 ms | 0.89 (rustfft) |
| 10,000 | 1000 | 47.4 µs | 47.1 µs | 106 µs | 64.1 µs | 6.82 ms | 0.74 (rustfft) |
| 10,000 | 10,000 | 88.7 µs | 91 µs | 222 µs | 350 µs | 137 ms | 0.40 (FFTW) |
| 100,000 | 3 | 26.6 µs | 165 µs | 267 µs | 146 µs | 221 µs | 0.18 (rustfft) |
| 100,000 | 8 | 66.7 µs | 175 µs | 329 µs | 180 µs | 468 µs | 0.37 (rustfft) |
| 100,000 | 16 | 121 µs | 184 µs | 415 µs | 240 µs | 924 µs | 0.50 (rustfft) |
| 100,000 | 32 | 229 µs | 196 µs | 452 µs | 273 µs | 1.72 ms | 0.84 (rustfft) |
| 100,000 | 64 | 259 µs | 209 µs | 504 µs | 296 µs | 3.38 ms | 0.88 (rustfft) |
| 100,000 | 128 | 266 µs | 232 µs | 553 µs | 331 µs | 6.93 ms | 0.80 (rustfft) |
| 100,000 | 256 | 319 µs | 266 µs | 609 µs | 349 µs | 14.2 ms | 0.91 (rustfft) |
| 100,000 | 1000 | 343 µs | 307 µs | 747 µs | 461 µs | 62.5 ms | 0.75 (rustfft) |
| 100,000 | 10,000 | 526 µs | 563 µs | 1.38 ms | 792 µs | – | 0.66 (rustfft) |
| 100,000 | 100,000 | 1.26 ms | 1.06 ms | 3 ms | 4.25 ms | – | 0.42 (FFTW) |
| 1,000,000 | 3 | 264 µs | 1.63 ms | 2.67 ms | 1.46 ms | 2.17 ms | 0.18 (rustfft) |
| 1,000,000 | 8 | 669 µs | 1.73 ms | 3.32 ms | 1.75 ms | 4.68 ms | 0.38 (rustfft) |
| 1,000,000 | 16 | 1.17 ms | 1.84 ms | 4.12 ms | 2.41 ms | 9.21 ms | 0.49 (rustfft) |
| 1,000,000 | 32 | 2.29 ms | 1.95 ms | 4.44 ms | 2.78 ms | 17.2 ms | 0.82 (rustfft) |
| 1,000,000 | 64 | 2.53 ms | 2.09 ms | 5 ms | 2.94 ms | 34.6 ms | 0.86 (rustfft) |
| 1,000,000 | 128 | 3.18 ms | 2.3 ms | 5.49 ms | 3.23 ms | 70.3 ms | 0.99 (rustfft) |
| 1,000,000 | 256 | 2.91 ms | 2.6 ms | 6.03 ms | 3.46 ms | 142 ms | 0.84 (rustfft) |
| 1,000,000 | 1000 | 3.19 ms | 3.03 ms | 7.35 ms | 4.44 ms | – | 0.72 (rustfft) |
| 1,000,000 | 10,000 | 4.28 ms | 4.59 ms | 11 ms | 6.12 ms | – | 0.70 (rustfft) |
| 1,000,000 | 100,000 | 7.09 ms | 7.92 ms | 19.3 ms | 9.93 ms | – | 0.71 (rustfft) |

`convolve` of `Complex<f64>` (load 1.2 to 3.9):

| len(a) | len(b) | `a` | `a`, plan reused | FFTW | rustfft | vDSP | a/fastest |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 56 ns | 139 ns | 132 ns | 130 ns | 53 ns | 1.06 (vDSP) |
| 16 | 8 | 77 ns | 139 ns | 139 ns | 114 ns | 116 ns | 0.68 (rustfft) |
| 16 | 16 | 117 ns | 166 ns | 170 ns | 151 ns | 245 ns | 0.77 (rustfft) |
| 100 | 3 | 144 ns | 438 ns | 399 ns | 385 ns | 195 ns | 0.74 (vDSP) |
| 100 | 8 | 219 ns | 453 ns | 449 ns | 399 ns | 444 ns | 0.55 (rustfft) |
| 100 | 16 | 560 ns | 559 ns | 631 ns | 559 ns | 819 ns | 1.00 (rustfft) |
| 100 | 32 | 811 ns | 700 ns | 946 ns | 1.01 µs | 1.72 µs | 0.86 (FFTW) |
| 100 | 64 | 981 ns | 798 ns | 1.14 µs | 1.14 µs | 4.83 µs | 0.86 (rustfft) |
| 1000 | 3 | 628 ns | 3.06 µs | 3.2 µs | 2.89 µs | 1.91 µs | 0.33 (vDSP) |
| 1000 | 8 | 1.39 µs | 3.23 µs | 3.8 µs | 3.4 µs | 4.12 µs | 0.41 (rustfft) |
| 1000 | 16 | 2.68 µs | 3.52 µs | 4.8 µs | 4.5 µs | 7.27 µs | 0.60 (rustfft) |
| 1000 | 32 | 5 µs | 4.24 µs | 6.11 µs | 5.49 µs | 13.5 µs | 0.91 (rustfft) |
| 1000 | 64 | 5.64 µs | 5.11 µs | 7.25 µs | 6.84 µs | 32.1 µs | 0.83 (rustfft) |
| 1000 | 128 | 7.11 µs | 5.84 µs | 8.54 µs | 7.99 µs | 74.6 µs | 0.89 (rustfft) |
| 1000 | 256 | 8.73 µs | 7.5 µs | 11 µs | 10.2 µs | 190 µs | 0.86 (rustfft) |
| 1000 | 1000 | 14.6 µs | 13.9 µs | 19.1 µs | 34 µs | 1.31 ms | 0.77 (FFTW) |
| 10,000 | 3 | 5.71 µs | 29.2 µs | 32.2 µs | 28.8 µs | 21.1 µs | 0.27 (vDSP) |
| 10,000 | 8 | 13.2 µs | 31.5 µs | 38.5 µs | 34 µs | 44.3 µs | 0.39 (rustfft) |
| 10,000 | 16 | 23.8 µs | 34.5 µs | 47.6 µs | 44.1 µs | 73.9 µs | 0.54 (rustfft) |
| 10,000 | 32 | 45.5 µs | 37.7 µs | 52.7 µs | 49.2 µs | 135 µs | 0.93 (rustfft) |
| 10,000 | 64 | 47.7 µs | 41.3 µs | 59.7 µs | 55.1 µs | 303 µs | 0.87 (rustfft) |
| 10,000 | 128 | 54.3 µs | 46.7 µs | 66.9 µs | 62 µs | 680 µs | 0.88 (rustfft) |
| 10,000 | 256 | 59.4 µs | 56.1 µs | 78.6 µs | 73.1 µs | 1.56 ms | 0.81 (rustfft) |
| 10,000 | 1000 | 80 µs | 87.4 µs | 123 µs | 115 µs | 7.27 ms | 0.69 (rustfft) |
| 10,000 | 10,000 | 163 µs | 175 µs | 242 µs | 476 µs | 135 ms | 0.67 (FFTW) |
| 100,000 | 3 | 53.4 µs | 285 µs | 316 µs | 282 µs | 201 µs | 0.27 (vDSP) |
| 100,000 | 8 | 132 µs | 309 µs | 379 µs | 336 µs | 423 µs | 0.39 (rustfft) |
| 100,000 | 16 | 242 µs | 336 µs | 477 µs | 436 µs | 734 µs | 0.56 (rustfft) |
| 100,000 | 32 | 453 µs | 367 µs | 513 µs | 487 µs | 1.35 ms | 0.93 (rustfft) |
| 100,000 | 64 | 427 µs | 396 µs | 565 µs | 536 µs | 2.98 ms | 0.80 (rustfft) |
| 100,000 | 128 | 470 µs | 441 µs | 635 µs | 574 µs | 6.68 ms | 0.82 (rustfft) |
| 100,000 | 256 | 532 µs | 503 µs | 683 µs | 663 µs | 15.3 ms | 0.80 (rustfft) |
| 100,000 | 1000 | 649 µs | 656 µs | 914 µs | 831 µs | 66.2 ms | 0.78 (rustfft) |
| 100,000 | 10,000 | 960 µs | 1.11 ms | 1.6 ms | 1.51 ms | – | 0.64 (rustfft) |
| 100,000 | 100,000 | 2.23 ms | 2.08 ms | 3.37 ms | 6.12 ms | – | 0.66 (FFTW) |
| 1,000,000 | 3 | 531 µs | 2.86 ms | 3.15 ms | 2.82 ms | 2.06 ms | 0.26 (vDSP) |
| 1,000,000 | 8 | 1.33 ms | 3.11 ms | 3.78 ms | 3.38 ms | 4.29 ms | 0.39 (rustfft) |
| 1,000,000 | 16 | 2.43 ms | 3.36 ms | 4.73 ms | 4.35 ms | 7.21 ms | 0.56 (rustfft) |
| 1,000,000 | 32 | 4.53 ms | 3.71 ms | 5.14 ms | 4.78 ms | 13.5 ms | 0.95 (rustfft) |
| 1,000,000 | 64 | 4.33 ms | 4.03 ms | 5.83 ms | 5.32 ms | 30.2 ms | 0.81 (rustfft) |
| 1,000,000 | 128 | 4.88 ms | 4.49 ms | 6.43 ms | 5.85 ms | 66.7 ms | 0.83 (rustfft) |
| 1,000,000 | 256 | 5.09 ms | 4.96 ms | 6.99 ms | 6.53 ms | 153 ms | 0.78 (rustfft) |
| 1,000,000 | 1000 | 6.3 ms | 6.38 ms | 8.88 ms | 8.21 ms | – | 0.77 (rustfft) |
| 1,000,000 | 10,000 | 8.35 ms | 9.08 ms | 13 ms | 11.8 ms | – | 0.71 (rustfft) |
| 1,000,000 | 100,000 | 13.4 ms | 16 ms | 22.7 ms | 18.8 ms | – | 0.72 (rustfft) |

**The crossover.** The shorter operand up to which the model of this release takes the direct path, and the path it takes beyond (`bench/crossover.py` on the constants of `conv.rs`, which `calibrate` measured on the final transforms; `c32` and `c64` are `Complex<f32>` and `Complex<f64>`):

| type | len(long) | direct up to len(short) | then |
| --- | ---: | ---: | --- |
| f32 | 100 | all | - |
| f32 | 1000 | 187 | overlap-add L=700 |
| f32 | 10000 | 137 | overlap-add L=1280 |
| f32 | 100000 | 123 | overlap-add L=2048 |
| f32 | 1000000 | 123 | overlap-add L=2048 |
| f64 | 100 | all | - |
| f64 | 1000 | 137 | overlap-add L=640 |
| f64 | 10000 | 105 | overlap-add L=1024 |
| f64 | 100000 | 99 | overlap-add L=1280 |
| f64 | 1000000 | 99 | overlap-add L=1024 |
| c32 | 100 | 89 | padded L=192 |
| c32 | 1000 | 67 | overlap-add L=320 |
| c32 | 10000 | 53 | overlap-add L=512 |
| c32 | 100000 | 50 | overlap-add L=512 |
| c32 | 1000000 | 50 | overlap-add L=1024 |
| c64 | 100 | 57 | padded L=160 |
| c64 | 1000 | 47 | overlap-add L=256 |
| c64 | 10000 | 40 | overlap-add L=320 |
| c64 | 100000 | 38 | overlap-add L=512 |
| c64 | 1000000 | 38 | overlap-add L=512 |

The grid above follows it: in `f32` against 10⁴ to 10⁶ values the direct path takes 0.49 to 0.93 of the fastest reference's time up to 64 taps and the transforms 0.41 to 1.05 from 128; three cells near the switch are 1.34 to 1.35 times the fastest reference ([below](#where-the-references-are-faster)). The calibration in these runs (median of the three) measured the direct kernels at 0.0130 (`f32`), 0.0331 (`f64`), 0.0683 (`Complex<f32>`) and 0.134 (`Complex<f64>`) ns a product over 8 to 32 taps, and the transforms, the mean of a forward and an inverse one, at 0.0517, 0.0966, 0.0954 and 0.177 ns times `L·log₂ L` (the median over the powers of two from 1024; FFTW's, the same way from 2¹⁰ to 2¹⁸: 0.103, 0.122, 0.227 and 0.258).

### Accuracy

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

### Where the references are faster

Every row group above where `a` takes more than 1.10 times the fastest reference's time, by cause:

- **vDSP in `f32`**: the complex transform of powers of two from 128 (1.19 to 3.69 times; 2.30 at 1024, 1.50 at 2²⁰) and of `f·2^k` from 768 (1.72 to 3.66; and 60, 1.29), its inverse (2.49 at 1024, 2.09 at 65,536); the real transform of powers of two from 64 but 512 (1.18 to 3.59) and of `f·2^k` from 640 (1.81 to 3.70); DCT-II at 16 to 16,384 and 48 (1.28 to 2.21), DCT-III at 16 to 16,384 (1.32 to 3.56), DCT-IV at 16 to 16,384 (1.32 to 1.68); and `Fft2` at every size it has but 64 × 4096 (1.15 to 2.58; 1.07 there). vDSP works on split complex data with Apple's own kernels, where `a` keeps `Complex<T>` interleaved: a split layout through NEON's de-interleaving loads would save about a quarter of the arithmetic of a butterfly by the implementer's count, and is the main idea for this gap. In `f64` vDSP is matched or beaten (powers of two 0.55 to 1.04, `f·2^k` 0.28 to 0.68), and its cosine transforms from 65,536 values in `f32` are slower than `a`'s (0.51 to 0.79).
- **Single short complex transforms**: powers of two up to 32 (`f64` 1.18 to 2.19, `f32` 1.21 to 3.65, the worst `n = 2` in `f32`, 6.2 ns against rustfft's 1.7) and the smooth lengths 6, 9, 10, 12, 20 and 24 (1.28 to 1.92): a call costs about 5 ns of dispatch (the shared plan, the match on its algorithm, a function pointer a pass, the set-up of the passes), and 6, 9, 10, 12, 15 and 16 have no composite one-pass codelets, so 16 takes two passes of 4 and 24 two passes, where FFTW and rustfft have one butterfly for each.
- **Batches of short complex transforms against rustfft**: `f64` at 2, 4 and 16 (1.33, 1.14, 1.19; at 2 and 4 the times are 0.4 against 0.3 and 0.8 against 0.7 ns, printed to 0.1 ns) and `f32` at 16 and 32 (1.28, 1.14): rustfft's 16- and 32-point butterflies, where `a` runs two passes.
- **Short real transforms**: `RealFft::forward` up to 64 (`f64` 1.43 to 3.01, `f32` 1.46 to 2.46), at 80, 96 and 384 in `f64` (1.20 to 1.45), and `inverse` up to 64 (`f64` 1.29 to 7.10, `f32` 1.16 to 5.97); batches of 8, 15 and 16 values (1.96 to 4.35) and of 32 and 64 in `f64` (1.42, 1.26): fixed costs and no real codelets (FFTW's `r2cf` and `r2cb` for these sizes), the lengths up to 16 summed directly sequence by sequence. The inverse's direct sums at 15 and 16 take 1.9 to 2.7 times the forward's time and more than the inverse of 32 (41 and 45 ns against 30 in `f64`): not investigated, and DCT-III of 16 (2.31 and 3.56 times) runs on them.
- **Even real transforms of 128 to 8192 values in `f64`**: 1.12 to 1.14 at 128, 256, 512, 4096 and 8192 against vDSP or FFTW (1.02 to 1.06 at the others, faster from 16,384): the pass that separates the two halves' spectra ran at about 5 cycles a step of two pairs where 4 would do by the count of its operations (measured while the machine was loaded), and is a pass of its own over the data rather than fused into the complex transform's last.
- **Odd real lengths**: 15 to 177,147 with factors of 3, 5, 7 and 11 (`forward` `f64` 1.17 to 2.75, `f32` 1.13 to 2.04; `inverse` 1.12 to 6.17), against FFTW, which has real codelets for odd radices: here the residue sequences go in pairs through the complex transform and one pass combines them, scalar, with sums rather than written-out butterflies for 7 and 11; part of it is the complex transform's own odd lengths. The cosine and sine transforms of types II to IV at 255 and 1023 (1.11 to 1.42) run on these, and the odd type IV adds a scalar permutation and `√2` pass.
- **Short cosine and sine transforms in `f64`**: DCT-II and DST-II at 16, 64 and 256 (1.11 to 1.52) and DCT-III at 64 (1.18): the fixed costs of the short real transforms they run on, and the DCT-II twiddle pass run as a pass of its own after the real transform's (fusing the two would save about 15 to 20 % of a DCT-II at 1024, by the implementer's estimate).
- **DCT-I and DST-I at `2^k ± 1`**: `f64` 63 to 65,537 (1.20 to 1.49), `f32` 63 to 1025 (1.13 to 1.51): the split-radix step's gathers are scalar and each level of its recursion is a plan of its own with its fixed costs; FFTW's planner measures several algorithms for these lengths, and which one it took was not recorded. At even lengths `a` is 0.19 to 0.87 times FFTW's time.
- **Two and three dimensions in `f64`, and small arrays**: `Fft2` at 64 × 64 (1.11), 1024 × 1024 (1.19) and 4096 × 64 (1.38; `f32` 1.37 against FFTW, besides vDSP), `Fft3` at 16³ (2.14, `f32` 1.84), 32³ and 64³ (1.26, 1.12): the tiles of columns are gathered and scattered value by value, with no SIMD transpose, and their sizes were tuned on a loaded machine; the small arrays pay the working memory each call allocates and a call of the transform per tile.
- **Short convolutions, direct**: `f32` at 16 × 3 and 8 (1.25, 1.23) and 100 × 3 to 64 (1.13 to 1.52 against vDSP), and `f64` at 16 × 3 to 16 and 100 × 3 to 32 (1.16 to 1.79): fixed costs, the allocation of the result and a whole kernel block of outputs (32 or 16) computed for a few. At 1000 × 64 in `f32` (1.12, 1.48 against 1.33 µs) it is the kernels' rate, about 77 % of the FMA peak by the implementer's measurement under load, against vDSP's.
- **Convolutions near the crossover**: `f64` at 1000 × 128 (1.34 against FFTW), `Complex<f32>` at 100 × 64 and 1000 × 64 (1.35, 1.34 against rustfft): the model takes the direct path where the transforms on kept plans would be faster (`a`'s own transforms, plan reused, take 3.90 against 5.34 µs, 464 against 947 ns and 2.78 against 5.25 µs), because it charges half a plan's cost to every call and may not ask the cache; the implementer's sweeps put it within 1.6 times of the best path at the switch.

The first runs of the convolutions, before the change to the model's lower bound, had a third group: the first sizes at which the model searched the transform lengths and then took the direct path anyway (`f32` 1000 × 8 to 64, `f64` 1000 × 8 and 16, `Complex<f32>` 100 × 3 to 16 and `Complex<f64>` 100 × 3, 1.11 to 1.72 times); the tighter bound skips those searches, and `f32` 1000 × 8 went from 443 to 200 ns. Nothing of that group is left.

### For the optimisation pass

The measured gaps above, with the ideas the implementers named for them:

- **The split-complex layout for vDSP-level `f32`** (1.2 to 3.7 times vDSP from 128 values in the complex, real, cosine and 2D transforms): planar data through `ld2`/`st2`, radix-16 and radix-32 codelets with register-blocked twiddles (genfft's style), and FCMA's `fcadd`/`fcmla` for rotations and products (inline assembly or unstable intrinsics).
- **Composite small codelets** (6, 9, 10, 12, 15, 16, and 16 and 32 for batches; 1.2 to 3.7 times at `n ≤ 32`) and a direct path for a single short call, without the dispatch of about 5 ns; **real codelets** (`r2cf`-style) for `RealFft` up to 32 or 64, and its batches.
- **The fused DCT-II pass**: the twiddles of DCT-II's outputs in the real transform's pass over the pairs (and DCT-III's in its inverse's), one pass of `O(n)` instead of two, about 15 to 20 % of a DCT-II at 1024; it needs a new kernel in NEON, SSE2, AVX2 and portable form.
- **Odd real lengths** (1.1 to 2.75 times FFTW): the combine pass vectorised across `s` (its twiddles and outputs are contiguous, the mirrored ones reversed), radix 7 and 11 written out, real (`hc2hc`) codelets; the odd type IV's permutation and `√2` pass are scalar too. The inverse's direct sums at 15 and 16, slower than the forward's.
- **The pair pass of the even real transforms**: about 0.84 ns a pair in `f64` and 0.49 in `f32` (measured under load), 5 cycles a step against 4; scheduling, or fusing it into the complex transform's last pass.
- **Two and three dimensions**: a 2 × 2 or 4 × 4 SIMD transpose for the tiles, the tiles tuned on a quiet machine, and the per-call allocation of the work.
- **Merging the two root generators**, `twiddle::Roots` (the complex transform's, an octant table of `sin` and `cos` corrected in double-double) and `real/roots.rs` (the real and trigonometric transforms', products of a coarse and a fine table in double-double, correctly rounded); planning still costs about 2 ns a twiddle entry, and Bluestein's `f32` plans transform their kernel in `f64`.
- **Long transforms**: from 2²⁰ the passes run at 23 to 25 Gflop/s in `f64` and 48 to 51 in `f32`; a properly blocked six-step algorithm (columns in L1, two-level twiddles, prefetching) might gain 10 to 20 %, where the two attempts lost twice.
- **Primes**: 23 to 31 as passes (hand-scheduled against the spills of 23), and the product of Rader's and Bluestein's kernels fused into the last pass of the inner forward transform and the first of the inverse.
- **Kernels**: the odd butterflies 11 to 19 by-element FMAs with two constants a register; AVX2's `f32` first pass on 256-bit vectors (a 4 × 4 lane transpose); the convolutions' direct kernels at about 77 % of the FMA peak (measured under load; hand scheduling, and 12 vectors of outputs for long taps, measured 6 % faster there and slower at short ones), a one-vector kernel for tiny outputs, overlap-save in place of overlap-add, and the spectral product fused into the transforms' passes.
- **The convolutions' choice**: its constants near the switch (1.34 to 1.35 times the fastest reference in three cells); `calibrate`'s fit of the direct path under-reads its cost per product in `f32` at long taps.
- **x86 tuning**: nothing on x86_64 has been timed. The planner's estimates and the convolutions' constants are the M4's (the doubled cost of SSE2's products an estimate), the AVX2 kernels' assembly was checked only for the intrinsics being inlined, and every crossover is to be measured on x86 hardware.

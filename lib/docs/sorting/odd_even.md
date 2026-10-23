[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Odd-Even

Odd-even sort, also called brick sort or odd-even transposition sort, is a simple, stable, in-place sorting algorithm that compares neighbours in alternating phases: an odd phase compare-exchanges the pairs (1, 2), (3, 4), …, and an even phase the pairs (0, 1), (2, 3), …, swapping a pair if its second element is smaller. An odd and an even phase make a round, and the textbook version repeats rounds until one swaps nothing. No two pairs of a phase share an element, so all compare-exchanges of a phase are independent: the algorithm comes from parallel computing, where a row of processors that only talk to their neighbours sorts n elements in n phases (Habermann, 1972). On one processor it is bubble sort with each pass cut into independent pairs; `odd_even_parallel` runs it on several threads. This implementation uses seven optimizations, none of which changes which compare-exchanges run in which phase (every swap is one the textbook version makes):

- Bounds from the last swaps: a pair that is in order stays in order until one of its elements changes, so the odd phase of a round only compares the pairs from one before the first to one after the last pair the previous even phase swapped, and the even phase one pair more at each end; the textbook compares all n−1 pairs every round. This saves 7% of the comparisons on 1000 random elements, makes nearly sorted input 1.25–1.8 times faster and input with few distinct keys 1.2–1.4 times, and an element out of place at either end (a turtle or a rabbit) costs about 3.5n comparisons instead of about n²/2.
- At most ⌈n/2⌉ rounds: n phases sort any input (the 0-1 principle), so the sort stops then even if the last round swapped. This saves the textbook's final round on input that needs every phase, such as reverse-sorted input, and guarantees that the sort ends with an inconsistent `Ord`, whose answers would otherwise almost never let a round pass without a swap.
- One sweep per round: the even pair (k, k+1) depends only on the odd pairs on either side of it, so one sweep compares an odd pair and then the even pair just behind it — a wavefront making the same compare-exchanges on the same elements as two separate phases, while loading each element once per round instead of twice: 2–2.2 times faster for integers, up to 1.7 times for strings and 1.6 times for 64-byte elements.
- Branchless compare-exchanges for elements of at most 8 bytes: both elements are read and the smaller and the larger written back, chosen by the comparison, so that a chunk of 32 odd and 32 even pairs compiles to vector code for integers and reports its swaps as a bit mask. 1.3–2 times faster than branching on random input or few distinct keys (2–10% slower on nearly sorted input, where branches are predictable); 16-byte elements are 2.4–3.9 times slower without branches, so larger elements keep them.
- Elements of at most 4 bytes split into even and odd positions: a chunk of 32 steps is moved out of the slice into two small arrays, one for the elements at even positions and one for those at odd ones, and back after both phases, with the element where the next chunk starts carried over in a register. Every pair of a phase then sits at the same place of the two arrays (one place apart for the odd phase), and the compiler runs each phase as a few vector minimums and maximums, where in the slice it picked lanes apart and widened every swap bit to 64 bits. The same pairs are compared and the same swaps made (the elements are compared where the chunk holds them, outside the slice): 1.3–1.5 times faster for 4-byte elements from 1000 elements on, 1.7–2.35 times for 2-byte and 3.3–4.3 times for 1-byte ones (1.05–1.5 times on 100); for `u64` it made no difference.
- For larger elements, swaps by plain copies with the even phase two pairs behind the odd one: `mem::swap` compiled to copies the processor could not forward to the next comparison (1.2–1.9 times slower for 16- and 64-byte elements), and an even phase right behind the odd one was 1.07–1.45 times slower for 64-byte elements.
- Parallel form: `odd_even_parallel` cuts the slice into blocks that threads (`std::thread::scope`, no dependencies) claim in turn; on its block a thread runs a super-step of K phases, each leaving out one more pair at each interior end (a trapezoid), and in the next super-step the slice is cut at the midpoints of the previous blocks, so that each thread first completes the triangle of pairs left out around one old boundary and then runs the next trapezoid. Every compare-exchange still runs exactly once, after the two it depends on, so this is still odd-even transposition sort (not block merge-split sort), with one barrier per super-step instead of one per phase. Short slices and input that is sorted within the first rounds never start a thread; a panic in any thread reaches the caller with its payload.

## Examples

Rust usage with integers:

```rust
use a::sort::odd_even;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  odd_even(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

A turtle (the 0 at the end moves two slots per round, and each round only compares around it):

```rust
use a::sort::odd_even;

fn main() {
  let mut v = vec![1, 2, 3, 4, 5, 0];
  odd_even(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
}
```

On several threads:

```rust
use a::sort::odd_even_parallel;

fn main() {
  let mut v: Vec<u32> = (0..10_000).rev().collect();
  odd_even_parallel(&mut v);
  assert!(v.windows(2).all(|w| w[0] <= w[1]));
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2)
  - Average: O(n^2)
  - Best: O(n), when the input is already sorted
  - Parallel: O(n^2 / p) work per thread on p threads, with one barrier per super-step
- Comparisons and swaps (`I` = number of inversions):
  - Any input: exactly I swaps (each removes one inversion), as the textbook version; never more comparisons than the textbook version
  - Best (sorted): n−1 comparisons, 0 swaps
  - Reverse sorted: n(n−1)/2 comparisons for even n, (n²−1)/2 for odd n, every one a swap; the textbook version makes n−1 more
  - Turtle `[1, 2, …, n−1, 0]`: (7n−16)/2 comparisons for even n, (7n−13)/2 for odd n, against (n−1)(⌊n/2⌋+1) for the textbook version (3,492 against 500,499 for n = 1000)
  - Random input: about 453,400 comparisons for n = 1000 (textbook ≈ 486,800) and about n²/4 swaps
- Space complexity: O(1) extra (in-place); the parallel form allocates only for its threads and a block plan
- Stability: stable (only neighbours are swapped, and only when the second is strictly smaller)
- Adaptivity: the bounds shrink to the part still changing; good for nearly sorted input whose displaced elements have not far to go, but a far-displaced element still costs a phase per slot
- Against the textbook version (1000 and 5000 elements): 1.15–3.4 times faster for integers (2–2.6 on random), 1.03–1.75 times for strings, 1.0–2.1 times for 64-byte elements; elements of at most 4 bytes gain another 1.3–4.3 times from splitting chunks into even and odd positions
- Against [Bubble](bubble.md): 1.6–1.7 times faster on random integers, but on strings and 64-byte elements from 1.1 times faster to 2 times slower (bubble sort shifts an element with one write per slot instead of swapping). Against [Cocktail Shaker](cocktail_shaker.md): 1.45 times faster on random integers and 2.1 times on few distinct keys, but 1.2–1.4 times slower on random strings, 1.5–2.2 times on 64-byte elements and 20–50 times on nearly sorted input, which cocktail shaker sort fixes in a few passes
- Against `slice::sort_unstable`: about 20 times slower on 1000 random integers and 85 times on 5000 (on random `u32`s about 11 and 45 times, on `u8`s 8 and 30 times) — prefer `slice::sort` for real workloads
- Parallel form against the sequential one (16-core machine, 12 performance cores, under other load): 4.3–6 times faster on 20,000 integers, 7–8 times on strings and 6.6–10.5 times on 64-byte elements; 1.9–3.8, 4.4–6.5 and 3.5–6.6 times on 8192; 1.3–2.1, 2.7–4 and 2–4.4 times on 4096. It uses threads from 4096 elements of at most 8 bytes and 2048 larger ones, after the first 64 rounds (or 2²⁰ / 2¹⁶ compared pairs) on the calling thread. Splitting chunks into even and odd positions makes it 1.2–1.25 times faster for `u32` and 2.1–2.2 times for `u8` on 8192 and 20,000 elements

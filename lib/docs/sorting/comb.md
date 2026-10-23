[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Comb

Comb sort is a simple, unstable, in-place sorting algorithm: bubble sort with a gap. A pass compares every element with the one `gap` slots after it and swaps the two if they are out of order, so a small element near the end (a "turtle") moves `gap` slots left in one step instead of one. The gap starts at the length of the slice and shrinks by a factor of 1.3 after each pass; once it reaches 1, the slice is nearly sorted and bubble sort finishes it. This implementation uses four optimizations:

- Rule of 11 (Combsort11): a gap of 9 or 10 is replaced by 11, which leaves fewer turtles for the final bubble sort; on random input of 10,000 elements this saves about 6% of the comparisons and narrows their spread.
- Adaptive finish: the final phase is this crate's [Bubble](bubble.md) sort, which scans only up to the last element it moved and moves elements by hole shifting, instead of the repeated full passes of classic comb sort.
- Branchless swaps for small elements: whether a gapped pass swaps a pair is close to a coin flip, which defeats the branch predictor, so for elements of at most 32 bytes both slots are rewritten from addresses chosen by the comparison, and each pass is walked in blocks of `gap` pairs over two non-overlapping slices. On random input this is 3–4.5 times faster for integers and 1.5–1.8 times for strings. Larger elements are swapped only when needed, since rewriting them costs more than a misprediction.
- Carrying in registers at small gaps, for elements of at most 16 bytes: at a gap of 14 or less, each pair overlaps the one `gap` slots before it, so a swap has to reach memory before the next comparison can read it back. Instead, the pass holds the largest element seen so far of every chain `c, c + gap, c + 2·gap, …` in a register, one per chain, and writes back only the smaller of it and the next element: the comparisons of a plain pass, in its order and with the same outcomes and result, with one move per swap plus one per carried element instead of two per swap, and the `gap` chains overlapping. Together with the faster final bubble sort this makes the whole sort 1.15–1.5 times faster for `u64`, 1.4–1.8 times for `u32` and 1.2–1.5 times for pairs of integers than before; `String`, whose comparisons usually read memory, keeps the plain pass.

## Examples

Rust usage with integers:

```rust
use a::sort::comb;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  comb(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

Turtles (the small elements at the end move many slots per step):

```rust
use a::sort::comb;

fn main() {
  let mut v = vec![5, 6, 7, 8, 9, 0, 1, 2, 3, 4];
  comb(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2) in theory (no known input reaches it in practice, but no O(n log n) bound has been proved)
  - Average: about n·log₁.₃ n comparisons
  - Best: O(n log n); the gapped passes cost the same on every input
- Comparisons (`g₁ > g₂ > … > gₖ ≥ 2` = the gaps, `k ≈ log₁.₃ n − 4`):
  - Gapped passes: exactly Σ (n − gᵢ) ≈ n·(log₁.₃ n − 7.3) on every input
  - Best (already sorted): Σ (n − gᵢ) + n − 1 (19,706 for n = 1000); best moves: 0
  - Random input: about n·(log₁.₃ n − 4.5) (≈ 21,300 for n = 1000, against ≈ 499,000 for bubble sort and ≈ 22,500 for classic comb sort without the rule of 11 and the adaptive finish)
- Space complexity: O(1) extra (in-place)
- Stability: unstable (a gapped swap can carry an element past an equal one)
- Adaptivity: not adaptive (only the final bubble sort gets cheaper on nearly-sorted input); much faster than the other quadratic sorts here on random input, about 1.5 times faster than `slice::sort_unstable` on 10,000 random `u32` and about as fast on `u64` (1.2 times slower at 1,000,000)

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Bubble

Bubble sort is a simple, stable, in-place sorting algorithm. It repeatedly scans the slice, swapping adjacent elements that are out of order. This implementation uses these optimizations:

- Early exit: stop when a pass performs no swaps (already sorted).
- Last-swap bound: the next pass only scans up to the last index that swapped (the tail is already sorted).
- Hole shifting: an element travelling `k` slots is read once, the `k` elements it overtakes shift one slot each, and it is written once: `k + 1` moves instead of the `3k` of `k` swaps.
- Branchless passes for small elements (at most 8 bytes, no drop glue): the running maximum is held in a register and every comparison writes the smaller element into the slot behind it, with no branch on the outcome, and two consecutive passes run at once, the second trailing the first, so that their comparisons overlap. Each pass picks branchy or branchless from what the previous one found, so sorted input still costs one branchy pass. The comparisons and their outcomes are exactly those of classic bubble sort; on random `u64` and `u32` input this is 1.2–2.9 times faster, and up to 3.3 times on elements that are each a few slots from their places.

## Examples

Rust usage with integers:

```rust
use a::sort::bubble;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  bubble(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

Sorting strings:

```rust
use a::sort::bubble;

fn main() {
  let mut fruits = vec!["pear", "apple", "banana", "apple"]; 
  bubble(&mut fruits);
  assert_eq!(fruits, vec!["apple", "apple", "banana", "pear"]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2)
  - Average: O(n^2)
  - Best: O(n) with early-exit when input is already sorted
- Comparisons and swaps (tight bounds for the canonical variant):
  - Worst comparisons: n(n−1)/2; worst swaps: n(n−1)/2
  - Best comparisons: n−1; best swaps: 0
- Space complexity: O(1) extra (in-place)
- Stability: stable (equal elements keep their relative order)
- Adaptivity: good for nearly-sorted data due to early exit and last-swap bound; unsuitable for large, random inputs compared to O(n log n) sorts

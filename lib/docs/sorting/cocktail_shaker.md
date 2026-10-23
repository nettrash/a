[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Cocktail Shaker

Cocktail shaker sort is a simple, stable, in-place sorting algorithm: bubble sort that alternates direction. A forward pass carries the largest element seen so far towards the end, then a backward pass carries the smallest element seen so far towards the front, so the unsorted range shrinks from both ends. Unlike bubble sort, a small element near the end (a "turtle") reaches its place in a single backward pass instead of moving one slot per pass. This implementation uses three optimizations:

- Last-move bounds at both ends: a forward pass ends the unsorted range at the last element it moved and a backward pass starts it just after the last element it moved, so sorted runs at either end are never rescanned; a pass that moves nothing ends the sort.
- Hole shifting instead of swapping: an element travelling `k` slots is read once, the `k` elements it overtakes shift one slot each as they are compared with it, and it is written once into its final slot: `k + 1` moves instead of the `3k` of `k` swaps. Shifting while comparing touches each element once and avoids a `memmove` call per (mostly very short) carry, which makes it about twice as fast on random integers as finding the distance first and moving a block afterwards.
- Branchless passes for small elements (at most 8 bytes, no drop glue): the running extreme is held in a register and every comparison writes the other element into the slot behind it, with no branch on the outcome. Each pass picks branchy or branchless from what the previous one found, so sorted input still costs one branchy pass. The comparisons are exactly those of classic cocktail shaker sort; on random `u64` input this is 1.8–2.6 times faster up to a few hundred elements and 1.2 times at 1,000, and 1.5–2.6 times on elements that are each a few slots from their places.

## Examples

Rust usage with integers:

```rust
use a::sort::cocktail_shaker;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  cocktail_shaker(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

A turtle (the 0 at the end reaches the front in one backward pass; bubble sort would need five passes):

```rust
use a::sort::cocktail_shaker;

fn main() {
  let mut v = vec![1, 2, 3, 4, 5, 0];
  cocktail_shaker(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2)
  - Average: O(n^2)
  - Best: O(n) with early-exit when input is already sorted
- Comparisons and moves (`I` = number of inversions, `C` = number of carries, i.e. elements picked up and moved at least one slot, counted once per pass):
  - Worst comparisons: n(n−1)/2 (reverse sorted); worst moves: n(n+1)/2 − 1
  - Any input: at most n(n−1)/2 comparisons and exactly I + C ≤ 2I moves
  - Best comparisons: n−1; best moves: 0
  - Random input: about n²/3 comparisons, a third fewer than bubble sort's n²/2 (≈ 332,000 against ≈ 496,000 for n = 1000)
  - Turtle `[1, 2, …, n−1, 0]`: 3n−6 comparisons, against n(n−1)/2 for bubble sort (2,994 against 499,500 for n = 1000)
- Space complexity: O(1) extra (in-place)
- Stability: stable (equal elements keep their relative order)
- Adaptivity: good for nearly-sorted data, including elements that are out of place in either direction; unsuitable for large, random inputs compared to O(n log n) sorts

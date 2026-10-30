[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Gnome

Gnome sort is a simple, stable, in-place sorting algorithm. A gnome walks along the slice comparing each element with the one before it: while they are in order it steps forward, and when they are not it carries the element back, one slot at a time, until it meets an element that is not greater than it or reaches the front. The slice behind the gnome is always sorted. This implementation uses two optimizations:

- Teleportation: classic gnome sort walks forward again over the slots it just carried an element back across, re-comparing pairs it already knows are in order; this gnome returns to where it turned back at once. Every comparison it makes is one classic gnome sort also makes, in the same order, but it saves exactly one per inversion: about half of them on random input.
- Hole shifting instead of swapping: an element travelling `d` slots is read once, the `d` greater elements it passes shift right by one slot each as they are compared with it, and it is written once into its final slot: `d + 1` moves instead of the `3d` of `d` swaps. The walk back is unrolled four times, which makes small elements up to 1.2 times faster (reverse-sorted input up to 1.5 times).

With both optimizations gnome sort makes the same comparisons and moves as insertion sort with a linear search; the [Insertion](insertion.md) sort above uses a galloping search instead and makes far fewer comparisons.

## Examples

Rust usage with integers:

```rust
use a::sort::gnome;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  gnome(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

A turtle (the 0 at the end is carried to the front in a single walk back):

```rust
use a::sort::gnome;

fn main() {
  let mut v = vec![1, 2, 3, 4, 5, 0];
  gnome(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2)
  - Average: O(n^2)
  - Best: O(n), when the input is already sorted
- Comparisons and moves (`I` = number of inversions, `C` = number of elements that move, `F` = number of elements other than the first that are smaller than everything before them, i.e. carried all the way to the front):
  - Any input: exactly n−1 + I − F comparisons and exactly I + C moves
  - Worst comparisons: n(n−1)/2 (reverse sorted), against n(n−1) for classic gnome sort; worst moves: n(n+1)/2 − 1
  - Best comparisons: n−1; best moves: 0
  - Random input: about n²/4 comparisons and n²/4 moves, against n²/2 comparisons and n²/4 swaps (3n²/4 moves) for classic gnome sort (≈ 250,800 against ≈ 500,600 comparisons for n = 1000)
  - Turtle `[1, 2, …, n−1, 0]`: 2n−3 comparisons, against 3n−4 for classic gnome sort and n(n−1)/2 for bubble sort
- Space complexity: O(1) extra (in-place)
- Stability: stable (equal elements keep their relative order)
- Adaptivity: good for nearly-sorted data; cost grows with the number of inversions; unsuitable for large, random inputs compared to O(n log n) sorts

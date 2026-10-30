[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Selection

Selection sort is a simple, in-place, unstable sorting algorithm. Each pass selects the extremes of the unsorted part of the slice and moves them to their final positions. This implementation is double-ended and uses three optimizations:

- Pairwise min-max scan: elements are taken two at a time, compared with each other, and only the smaller one is compared with the running minimum and the larger one with the running maximum, for `⌈3m/2⌉ − 2` comparisons per pass over `m` elements (the fewest possible for finding both extremes) instead of `2m − 2`.
- Duplicate gathering: every comparison is three-way, so all copies of the minimum and maximum keys are placed in the same pass at no extra comparisons; input with `k` distinct keys finishes in at most `⌈k/2⌉` passes, and a range found to be all equal ends the sort.
- Lazy placement: an extreme with no duplicates is only remembered by index and moved once the pass is over, so distinct keys cost at most `n − 1` swaps and already sorted input is never moved.

The scan picks which element of a pair to compare with each extreme without a branch, since which one is smaller is a coin flip on random input; with the same comparisons, random input with distinct keys sorts 1.7–2.5 times faster from 1,000 elements up. For elements of 9 to 32 bytes (strings, for one), whose comparisons are slow to start, the first pass counts how predictable the pairs are and later passes branch when they are.

## Examples

Rust usage with integers:

```rust
use a::sort::selection;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  selection(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

Many duplicates (three distinct keys, sorted in two passes):

```rust
use a::sort::selection;

fn main() {
  let mut v = vec![3, 1, 2, 3, 1, 2, 2, 1];
  selection(&mut v);
  assert_eq!(v, vec![1, 1, 1, 2, 2, 2, 3, 3]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2), when all keys are distinct
  - Average: O(n^2) for distinct keys; O(nk) with `k` distinct keys
  - Best: O(n), when all elements are equal
- Comparisons and swaps:
  - Worst comparisons: n(3n−2)/8 for even n, 3(n²−1)/8 for odd n (about 25% fewer than the n(n−1)/2 of classic selection sort); swaps with distinct keys: at most n−1
  - Best comparisons: n−1 (all equal); best swaps: 0 (already non-decreasing input)
  - With repeated keys: at most 2m swaps per pass over m elements, trading the n−1 swap bound for far fewer comparisons and passes
- Space complexity: O(1) extra (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: not adaptive to existing order (sorted input with distinct keys costs as much as any other), but adaptive to duplicates: the fewer distinct keys, the fewer passes; unsuitable for large inputs compared to O(n log n) sorts

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Quick

Quick sort is an unstable, in-place, `O(n log n)` sorting algorithm. It picks a pivot, partitions the slice into the elements smaller than the pivot and the rest, puts the pivot between them, and sorts the two parts the same way. This implementation uses seven optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input).
- Pivots from a median of samples: the median of three elements spread over the slice, from 64 elements on the median of three such medians (the "ninther"), and from 4096 elements on a pseudo-median of 27 evenly spaced samples, three times as many for every 16 times the length, which keeps comparisons at about n·log₂ n on random input, against 1.39·n·log₂ n for a random pivot.
- Branchless partitioning for elements of at most 64 bytes: whether an element is smaller than the pivot is a coin flip on random input, so instead of branching, a vacant slot cycles through the slice and every element is moved unconditionally, the comparison only choosing where (Lomuto's scheme, made branchless). That is 1.9–4.3 times faster than Hoare's scheme for integers and 1.4–2.5 times for strings; larger elements, for which two moves per element cost more than a misprediction, use Hoare's scheme, which moves only misplaced pairs (two moves per pair, cycled through a vacant slot, instead of three for a swap).
- Runs of equal elements in one pass: if the pivot equals the pivot of the enclosing partition, the part is split into the elements equal to it, which are done, and the rest, so `k` distinct keys cost O(n log k) and all-equal input is never quadratic.
- Recursion into the smaller part only, looping on the larger one, so the stack never holds more than log₂ n frames.
- Sorting networks for short parts: for elements of at most 16 bytes, parts of at most 8 elements are sorted by optimal sorting networks made of branchless compare-exchanges, unrolled for each length so the elements stay in registers (1.3–1.85 times faster than insertion sort for random integers); larger elements, for which a compare-exchange's three moves cost more, sort parts of at most 16 elements by insertion sort.
- A guaranteed O(n log n): a part still being partitioned after 2·log₂ n levels means the pivots were consistently bad, which an adversary that answers comparisons adaptively can force on any pivot rule; such a part is finished by heap sort (the "introsort" idea). Against McIlroy's adversary, 10,000 elements cost about 480,000 comparisons instead of 8.4 million.

## Examples

Rust usage with integers:

```rust
use a::sort::quick;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  quick(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Many duplicates (each distinct pivot takes all its copies out in one pass):

```rust
use a::sort::quick;

fn main() {
  let mut v: Vec<u8> = (0..1000).map(|i| (i * 7 % 3) as u8).collect();
  quick(&mut v);
  assert!(v.windows(2).all(|w| w[0] <= w[1]));
}
```

## Efficiency

- Time complexity:
  - Worst: O(n log n), thanks to the [Heap](heap.md) sort fallback
  - Average: O(n log n)
  - Best: O(n), when the input is sorted, reverse sorted or all equal
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Random input: about n·log₂ n + 0.2n to n·log₂ n + 0.6n (≈ 10,200 for n = 1000, n·log₂ n + 0.6n for n = 100,000 and 1,000,000), up to 0.4n more for elements larger than 16 bytes
  - `k` distinct keys: O(n log k), about 1.1–1.45·n·(log₂ k + 1) in practice
- Space complexity: O(log n) stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: adaptive to sorted and reverse-sorted input and to repeated keys; on nearly sorted input (a few elements out of place) Hoare's scheme would be 1.5–1.75 times faster for integers, since its branches are then predictable. `slice::sort_unstable`, which shares several of these ideas, is 1.1–1.2 times faster on random integers from 100 elements on (it merges short parts through a buffer, which is only sound for types without interior mutability), but this sort is 1.9 times faster on 20 integers, 1.03–1.14 times faster on random strings and as fast or faster with 256-byte elements

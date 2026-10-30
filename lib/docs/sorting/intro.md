[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Intro

Intro sort (introspective sort) is an unstable, in-place, `O(n log n)` sorting algorithm designed by David Musser in 1997: a quick sort that watches its own progress. Quick sort partitions the slice around a pivot and sorts the two parts the same way; with good pivots each level halves the parts, but no cheap pivot rule is good on every input, and bad pivots make it quadratic. Intro sort counts the levels, and a part still being partitioned after `2·⌊log₂ n⌋` levels is handed to [Heap](heap.md) sort, which is `O(n log n)` on every input; typical input never reaches the limit and keeps quick sort's speed. Short parts are left to insertion sort. This implementation keeps Hoare's partitioning comparisons and uses five optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input).
- Branchless block partitioning: Hoare's scheme scans from both ends for elements on the wrong side of the pivot and swaps them, and each scan stops at an unpredictable element, a mispredicted branch. Here each end compares a block of 128 elements first (Edelkamp and Weiss's choice; 1.02–1.06 times faster than 64 on random input from 1000 elements on), recording the misplaced ones by offset without branching, and then exchanges as many misplaced pairs as both blocks have (Edelkamp and Weiss's BlockQuicksort), in a cycle of two moves per pair instead of a swap's three. Same comparisons as Hoare's scheme; on random input 1.8–2.7 times faster for integers, 1.7–2.2 times for strings and 1.4–2.1 times for 64-byte elements. The block scans are unrolled by 8, another 1.03–1.08 times on random input and 1.1–1.25 times on nearly sorted input, where the scans are nearly all the work.
- Equal keys split evenly: as in Hoare's and Musser's scheme, an element equal to the pivot stops the scans on both sides, so equal keys are spread over both parts, and input with few distinct keys, even all equal but one, costs O(n log n) without ever reaching the depth limit.
- Pivots away from the ends, from a ninther on long parts: the median of the elements at a quarter, half and three quarters of the part, and from 128 elements on the median of three medians of three (Tukey's ninther), 1.04–1.13 times faster on sawtooth input of 100,000 elements or more. Musser samples the first, middle and last elements; swapping the pivot to the front then moves the smallest element of a sorted stretch into the middle, where the next sample finds it again, so sorted input with a few elements appended keeps getting the second smallest element as its pivot until the depth limit gives up: quartiles are 1.2–4 times faster there.
- Insertion sort while the part is fresh: parts of at most 24 elements (Musser: 16) are insertion sorted as soon as partitioning leaves them, still in the cache, instead of in one pass over the whole slice at the end, which reads the whole slice once more and is 1.05–1.15 times slower for strings.

## Examples

Rust usage with integers:

```rust
use a::sort::intro;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  intro(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Rust usage with strings:

```rust
use a::sort::intro;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  intro(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n log n), guaranteed by the heap sort fallback
  - Average: O(n log n)
  - Best: O(n), when the input is sorted or reverse sorted
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Random input: about (1.1–1.19)·n·log₂ n (≈ 11,800 for n = 1000); insertion into parts of up to 24 elements costs more comparisons than partitioning them further would, but less time
  - Few distinct keys: about 0.85·n·log₂ n; unlike [Quick](quick.md) sort, intro sort does not take all copies of a pivot out at once
  - Adversarial input: McIlroy's adversary, which answers comparisons so as to make every pivot bad, drives partitioning alone to 9.1 million comparisons for n = 10,000; with the heap sort fallback the whole sort makes about 470,000
- Space complexity: O(log n) stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: adaptive to sorted and reverse-sorted input, and cheap on nearly sorted input, where Hoare's scheme moves only what is misplaced. Against [Quick](quick.md) sort: 1.35–1.55 times slower on random integers from 1000 elements on, 1.06–1.13 times on random strings and 1.5–3.8 times with few distinct keys, but 1.2–1.65 times faster on random 64-byte elements and 1.3–3.6 times faster on nearly sorted input. Against `slice::sort_unstable`: 1.45–1.75 times slower on random integers and within 1.07 times on random strings, but 1.08–1.55 times faster on random 64-byte elements and 1.5–3.7 times on nearly sorted input. Against [Heap](heap.md) sort, which it falls back to: 1.3–2.4 times faster on random integers, 2.1–6.7 times on strings and 1.5–4.4 times on 64-byte elements

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Default

`sort` is this crate's default sort: use it when there is no reason to prefer a particular algorithm. It is [Drift](drift.md) sort, the same choice the standard library made for `slice::sort`: stable, O(n log n) on every input, O(n log k) with `k` distinct keys, O(n) on sorted or strictly descending input, and able to take advantage of runs of `√n` elements or more.

The choice was measured against this crate's other general-purpose comparison sorts ([Quick](quick.md), [Merge](merge.md), [Tim](tim.md) and [Intro](intro.md)) on an Apple M4 Max with new input for every sort. The test covered 210 cases: `u64`, `u32`, `u8`, strings and 64-byte elements; random, sorted, reverse-sorted, nearly sorted, 16 distinct keys, sawtooth, runs of `√n` and sorted with 1% appended; 16 to a million elements. Each sort was compared with the fastest of the five in every case:

| Sort | Average (geometric mean) | Worst | Cases more than 1.5 times slower | Stable |
| --- | --- | --- | --- | --- |
| Drift (default) | 1.21 | 2.9 | 32 of 210 | yes |
| Tim | 1.45 | 7.8 | 68 | yes |
| Quick | 1.51 | 13.8 | 64 | no |
| Intro | 1.60 | 12.5 | 90 | no |
| Merge | 1.74 | 10.4 | 101 | yes |

No sort is the fastest everywhere, but drift sort is never far off:

- [Quick](quick.md) sort is 1.1–1.3 times faster on random integers from 100 elements on and 1.1–1.2 times on random strings. Drift sort is 1.3 times faster on 16 random integers and on 100,000 or more random bytes, 1.15–1.2 times faster on random 64-byte elements from 1000 on, up to 2.1 times on nearly sorted input and 2.8–13 times on sorted input with elements appended.
- [Intro](intro.md) sort is 1.1–1.4 times faster on random 64-byte elements and up to 2.9 times on nearly sorted ones. Drift sort is 1.15–1.3 times faster on random integers from 100 elements on, 2–4.3 times with 16 distinct keys and 3.4–11 times on sorted input with elements appended, from 1000 elements on.
- [Tim](tim.md) sort is up to 1.8 times faster on nearly sorted input. Drift sort is 1.05–1.6 times faster on random input (5 times on a million bytes) and 2–8 times with 16 distinct keys from 1000 elements on.
- [Merge](merge.md) sort is up to 1.9 times faster on runs of `√n` elements. Drift sort is 2–8 times faster with 16 distinct keys from 1000 elements on and up to 10 times on reverse-sorted 64-byte elements.

The non-comparison sorts ([Counting](counting.md), [Radix](radix.md), [Bucket](bucket.md), [Pigeonhole](pigeonhole.md)) are not candidates because they sort integer or float keys only, not every `T: Ord`.

## Examples

Rust usage with integers:

```rust
use a::sort::sort;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  sort(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Sorting strings:

```rust
use a::sort::sort;

fn main() {
  let mut words = vec!["pear", "apple", "fig", "apple"];
  sort(&mut words);
  assert_eq!(words, vec!["apple", "apple", "fig", "pear"]);
}
```

## Efficiency

The same as [Drift](drift.md) sort:

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n), and O(n log k) with `k` distinct keys
  - Best: O(n), when the input is sorted or strictly descending
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once and only when needed) plus O(log n) stack
- Stability: stable (equal elements keep their relative order)
- Soundness: a comparison that panics or answers inconsistently leaves every element in the slice exactly once

# A

![rust workflow](https://github.com/nettrash/a/actions/workflows/rust.yml/badge.svg)

The A Library is a set of algorithms.

## Table of contents

- [A](#a)
  - [Table of contents](#table-of-contents)
  - [Sorting](#sorting)
    - [Bubble](#bubble)
      - [Examples](#examples)
      - [Sorting strings:](#sorting-strings)
      - [Efficiency](#efficiency)

## Sorting

### Bubble

Bubble sort is a simple, stable, in-place sorting algorithm. It repeatedly scans the slice, swapping adjacent elements that are out of order. This implementation uses two common optimizations:

- Early exit: stop when a pass performs no swaps (already sorted).
- Last-swap bound: the next pass only scans up to the last index that swapped (the tail is already sorted).

#### Examples

Rust usage with integers:

```rust
use a::sort::bubble;

fn main() {
    let mut v = vec![5, 1, 4, 2, 8];
    bubble(&mut v);
    assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

#### Sorting strings:

```rust
use a::sort::bubble;

fn main() {
    let mut fruits = vec!["pear", "apple", "banana", "apple"]; 
    bubble(&mut fruits);
    assert_eq!(fruits, vec!["apple", "apple", "banana", "pear"]);
}
```

#### Efficiency

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

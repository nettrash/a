# A

![rust workflow](https://github.com/nettrash/a/actions/workflows/rust.yml/badge.svg)

The A Library is a set of algorithms.

## Table of contents

- [A](#a)
  - [Table of contents](#table-of-contents)
  - [Sorting](#sorting)
    - [Bubble](#bubble)
      - [Examples](#examples)
      - [Efficiency](#efficiency)
    - [Selection](#selection)
      - [Examples](#examples-1)
      - [Efficiency](#efficiency-1)
    - [Insertion](#insertion)
      - [Examples](#examples-2)
      - [Efficiency](#efficiency-2)

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

Sorting strings:

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

### Selection

Selection sort is a simple, in-place, unstable sorting algorithm. Each pass selects the extremes of the unsorted part of the slice and moves them to their final positions. This implementation is double-ended and uses three optimizations:

- Pairwise min-max scan: elements are taken two at a time, compared with each other, and only the smaller one is compared with the running minimum and the larger one with the running maximum, for `⌈3m/2⌉ − 2` comparisons per pass over `m` elements (the fewest possible for finding both extremes) instead of `2m − 2`.
- Duplicate gathering: every comparison is three-way, so all copies of the minimum and maximum keys are placed in the same pass at no extra comparisons; input with `k` distinct keys finishes in at most `⌈k/2⌉` passes, and a range found to be all equal ends the sort.
- Lazy placement: an extreme with no duplicates is only remembered by index and moved once the pass is over, so distinct keys cost at most `n − 1` swaps and already sorted input is never moved.

#### Examples

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

#### Efficiency

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

### Insertion

Insertion sort is a simple, stable, in-place sorting algorithm. It grows a sorted prefix from the left, inserting each new element after every prefix element that is not greater than it. This implementation is a pair insertion sort (the variant Java's dual-pivot quicksort uses for short ranges) with three optimizations:

- Sorted fast path: the smaller element of each new pair is first compared with the last element of the prefix; if it is not smaller, the pair is already in place, so sorted input costs exactly n−1 comparisons.
- Galloping search: an insertion point is found by probing 1, 2, 4, 8, … slots to the left and then binary searching the last gap, so an element that moves `d` slots costs at most `2⌊log₂ d⌋ + 2` comparisons instead of `d`.
- Pair moves: two elements are inserted at once; the smaller one is only searched for left of the larger one's slot, and prefix elements greater than both shift right by two slots in a single move, so they move once instead of twice.

#### Examples

Rust usage with integers:

```rust
use a::sort::insertion;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8];
  insertion(&mut v);
  assert_eq!(v, vec![1, 2, 4, 5, 8]);
}
```

Equal keys keep their original order (cards are compared by rank only):

```rust
use a::sort::insertion;
use std::cmp::Ordering;

#[derive(Debug, PartialEq, Eq)]
struct Card(u8, char);

impl Ord for Card {
  fn cmp(&self, other: &Self) -> Ordering {
    self.0.cmp(&other.0)
  }
}

impl PartialOrd for Card {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

fn main() {
  let mut cards = vec![Card(2, 'a'), Card(1, 'x'), Card(2, 'b'), Card(0, 'y')];
  insertion(&mut cards);
  assert_eq!(cards, vec![Card(0, 'y'), Card(1, 'x'), Card(2, 'a'), Card(2, 'b')]);
}
```

#### Efficiency

- Time complexity:
  - Worst: O(n^2) element moves, but only O(n log n) comparisons
  - Average: O(n^2) moves, O(n log n) comparisons
  - Best: O(n), when the input is already sorted
- Comparisons and moves (`dₓ` = how many slots element `x` moves left, `I = Σ dₓ` = number of inversions):
  - Any input: at most n−1 + Σ 2⌈log₂(dₓ+1)⌉ comparisons and at most I + n moves
  - Worst comparisons: at most 2n·log₂ n (17,020 for n = 1000, against 499,500 for classic insertion sort)
  - Reverse sorted: about (n/2)·log₂ n + n comparisons and n²/4 moves
  - Best comparisons: n−1; best moves: 0
  - Against inserting one element at a time with the same search: about a third fewer moves on random input and half on reverse-sorted input
- Space complexity: O(1) extra (in-place)
- Stability: stable (equal elements keep their relative order)
- Adaptivity: very good; cost grows with the logarithm of how far elements move, so nearly-sorted input sorts in close to linear time; unsuitable for large, random inputs compared to O(n log n) sorts because of the O(n^2) moves

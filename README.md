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
    - [Cocktail Shaker](#cocktail-shaker)
      - [Examples](#examples-3)
      - [Efficiency](#efficiency-3)
    - [Gnome](#gnome)
      - [Examples](#examples-4)
      - [Efficiency](#efficiency-4)
    - [Comb](#comb)
      - [Examples](#examples-5)
      - [Efficiency](#efficiency-5)

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

### Cocktail Shaker

Cocktail shaker sort is a simple, stable, in-place sorting algorithm: bubble sort that alternates direction. A forward pass carries the largest element seen so far towards the end, then a backward pass carries the smallest element seen so far towards the front, so the unsorted range shrinks from both ends. Unlike bubble sort, a small element near the end (a "turtle") reaches its place in a single backward pass instead of moving one slot per pass. This implementation uses two optimizations:

- Last-move bounds at both ends: a forward pass ends the unsorted range at the last element it moved and a backward pass starts it just after the last element it moved, so sorted runs at either end are never rescanned; a pass that moves nothing ends the sort.
- Hole shifting instead of swapping: an element travelling `k` slots is read once, the `k` elements it overtakes shift one slot each as they are compared with it, and it is written once into its final slot: `k + 1` moves instead of the `3k` of `k` swaps. Shifting while comparing touches each element once and avoids a `memmove` call per (mostly very short) carry, which makes it about twice as fast on random integers as finding the distance first and moving a block afterwards.

#### Examples

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

#### Efficiency

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

### Gnome

Gnome sort is a simple, stable, in-place sorting algorithm. A gnome walks along the slice comparing each element with the one before it: while they are in order it steps forward, and when they are not it carries the element back, one slot at a time, until it meets an element that is not greater than it or reaches the front. The slice behind the gnome is always sorted. This implementation uses two optimizations:

- Teleportation: classic gnome sort walks forward again over the slots it just carried an element back across, re-comparing pairs it already knows are in order; this gnome returns to where it turned back at once. Every comparison it makes is one classic gnome sort also makes, in the same order, but it saves exactly one per inversion: about half of them on random input.
- Hole shifting instead of swapping: an element travelling `d` slots is read once, the `d` greater elements it passes shift right by one slot each as they are compared with it, and it is written once into its final slot: `d + 1` moves instead of the `3d` of `d` swaps.

With both optimizations gnome sort makes the same comparisons and moves as insertion sort with a linear search; the [Insertion](#insertion) sort above uses a galloping search instead and makes far fewer comparisons.

#### Examples

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

#### Efficiency

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

### Comb

Comb sort is a simple, unstable, in-place sorting algorithm: bubble sort with a gap. A pass compares every element with the one `gap` slots after it and swaps the two if they are out of order, so a small element near the end (a "turtle") moves `gap` slots left in one step instead of one. The gap starts at the length of the slice and shrinks by a factor of 1.3 after each pass; once it reaches 1, the slice is nearly sorted and bubble sort finishes it. This implementation uses four optimizations:

- Rule of 11 (Combsort11): a gap of 9 or 10 is replaced by 11, which leaves fewer turtles for the final bubble sort; on random input of 10,000 elements this saves about 6% of the comparisons and narrows their spread.
- Adaptive finish: the final phase is this crate's [Bubble](#bubble) sort, which scans only up to the last element it moved and moves elements by hole shifting, instead of the repeated full passes of classic comb sort.
- Branchless swaps for small elements: whether a gapped pass swaps a pair is close to a coin flip, which defeats the branch predictor, so for elements of at most 32 bytes both slots are rewritten from addresses chosen by the comparison, and each pass is walked in blocks of `gap` pairs over two non-overlapping slices. On random input this is 3–4.5 times faster for integers and 1.5–1.8 times for strings. Larger elements are swapped only when needed, since rewriting them costs more than a misprediction.
- Carrying along each chain at small gaps, for elements of at most 8 bytes: at a gap of 8 or less, each pair overlaps the one `gap` slots before it, so a swap has to reach memory before the next comparison can read it back. Instead, each chain `c, c + gap, c + 2·gap, …` is walked holding the largest element seen so far in a register, writing back only the smaller of it and the next element: the same comparisons and the same result, with one move per swap plus one per carried element instead of two per swap. This makes the whole sort 10–50% faster for integers and pointers; for larger elements, whose comparisons usually read memory, it would be slower, because each comparison has to wait for the previous one.

#### Examples

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

#### Efficiency

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
- Adaptivity: not adaptive (only the final bubble sort gets cheaper on nearly-sorted input); much faster than the other quadratic sorts here on random input, about 1.2 times slower than `slice::sort_unstable` on 10,000 random `u32` and 1.6 times on `u64`

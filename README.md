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
    - [Merge](#merge)
      - [Examples](#examples-6)
      - [Efficiency](#efficiency-6)
    - [Quick](#quick)
      - [Examples](#examples-7)
      - [Efficiency](#efficiency-7)
    - [Heap](#heap)
      - [Examples](#examples-8)
      - [Efficiency](#efficiency-8)
    - [Tim](#tim)
      - [Examples](#examples-9)
      - [Efficiency](#efficiency-9)
    - [Intro](#intro)
      - [Examples](#examples-10)
      - [Efficiency](#efficiency-10)
    - [Shell](#shell)
      - [Examples](#examples-11)
      - [Efficiency](#efficiency-11)
    - [Drift](#drift)
      - [Examples](#examples-12)
      - [Efficiency](#efficiency-12)

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

### Merge

Merge sort is a stable, `O(n log n)` sorting algorithm that uses a buffer. It splits the slice into two halves, sorts each half recursively, and merges the two sorted halves into one. Every element takes part in one merge per level of recursion and there are about `log₂ n` levels, so the cost hardly depends on the input. This implementation uses a buffer of `n` elements, kept on the stack when it fits in 4 KiB and allocated once otherwise, and five optimizations:

- Short runs sorted without branches: halves of at most 8 elements are not split further. A run is first scanned for order, which leaves a sorted run alone after len−1 comparisons and otherwise stops at the first descent, usually after one or two; then its two halves of at most 4 elements are sorted straight into the buffer by fixed sequences of comparisons that select where each element goes instead of branching on the outcome (1, 3 and 5 comparisons for 2, 3 and 4 elements, the last being the standard library's stable four-element sort), and merged back. Short runs took up to 60% of the time when they were insertion sorted; together with the stack buffer below, this makes the whole sort 1.2–2.8 times faster for integers (most for small slices) and 1.02–1.9 times for strings and large elements.
- Ordered halves are not merged: if the first element of the right half is not smaller than the last element of the left half, one comparison replaces the merge, so already-sorted input costs exactly n−1 comparisons and is never moved.
- Reversed halves are swapped: if the last element of the right half is smaller than the first element of the left half, the halves trade places as blocks with no further comparison, so reverse-sorted input costs about 2.6n comparisons instead of n·log₂ n.
- Bidirectional, branchless merging: each merge proceeds from both ends at once, the front taking the smaller of the two first elements and the back the larger of the two last ones. The two independent chains of comparisons overlap in the processor, and every step copies from an address chosen by the comparison instead of branching on it. On random input this is 1.2–1.6 times faster than the classic merge that copies only the left half out and merges from the front.
- No allocation for small slices: a buffer of up to 4 KiB lives on the stack, so a slice that fits in it (512 `u64`s, say) is sorted without touching the allocator.

#### Examples

Rust usage with integers:

```rust
use a::sort::merge;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  merge(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Equal keys keep their original order (cards are compared by rank only):

```rust
use a::sort::merge;
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
  let ranks = [3, 1, 3, 0, 1, 3, 0, 2, 1, 0];
  let mut cards: Vec<Card> = "abcdefghij".chars().zip(ranks).map(|(c, r)| Card(r, c)).collect();
  merge(&mut cards);
  let order: String = cards.iter().map(|c| c.1).collect();
  assert_eq!(order, "dgjbeihacf");
}
```

#### Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when the input is already sorted (or made of ordered or reversed halves at every level)
- Comparisons and moves (a merge of m elements makes at most 2⌊m/2⌋ comparisons after the two checks, exactly that many for elements of at most 8 bytes, and moves every element twice: out to the buffer and back):
  - Best (already sorted): n−1 comparisons, 0 moves
  - Reverse sorted: about 2.6n comparisons (2,590 for n = 1000, against 5,044 for the textbook merge sort)
  - Random input: about n·log₂ n − 0.35n comparisons (≈ 9,620 for n = 1000; the textbook merge sort makes ≈ 8,700, since the checks cost a comparison or two per merge and a bidirectional merge does not stop when one half runs out) and about 2n·log₂(n/8) + 2n moves
  - Worst: at most n·log₂ n + O(n) comparisons
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once) plus O(log n) stack
- Stability: stable (equal elements keep their relative order)
- Adaptivity: adaptive at the scale of runs and halves (sorted and reverse-sorted input are nearly linear); faster than `slice::sort` on slices of up to about 20 random integers, but about 1.8 times slower on 100,000 and 1.2 times on large elements, since the standard library's stable sort also finds natural runs and merges them more cleverly

### Quick

Quick sort is an unstable, in-place, `O(n log n)` sorting algorithm. It picks a pivot, partitions the slice into the elements smaller than the pivot and the rest, puts the pivot between them, and sorts the two parts the same way. This implementation uses seven optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input).
- Pivots from a median of samples: the median of three elements spread over the slice, from 64 elements on the median of three such medians (the "ninther"), and from 4096 elements on a pseudo-median of 27 evenly spaced samples, three times as many for every 16 times the length, which keeps comparisons at about n·log₂ n on random input, against 1.39·n·log₂ n for a random pivot.
- Branchless partitioning for elements of at most 64 bytes: whether an element is smaller than the pivot is a coin flip on random input, so instead of branching, a vacant slot cycles through the slice and every element is moved unconditionally, the comparison only choosing where (Lomuto's scheme, made branchless). That is 1.9–4.3 times faster than Hoare's scheme for integers and 1.4–2.5 times for strings; larger elements, for which two moves per element cost more than a misprediction, use Hoare's scheme, which moves only misplaced pairs (two moves per pair, cycled through a vacant slot, instead of three for a swap).
- Runs of equal elements in one pass: if the pivot equals the pivot of the enclosing partition, the part is split into the elements equal to it, which are done, and the rest, so `k` distinct keys cost O(n log k) and all-equal input is never quadratic.
- Recursion into the smaller part only, looping on the larger one, so the stack never holds more than log₂ n frames.
- Sorting networks for short parts: for elements of at most 16 bytes, parts of at most 8 elements are sorted by optimal sorting networks made of branchless compare-exchanges, unrolled for each length so the elements stay in registers (1.3–1.85 times faster than insertion sort for random integers); larger elements, for which a compare-exchange's three moves cost more, sort parts of at most 16 elements by insertion sort.
- A guaranteed O(n log n): a part still being partitioned after 2·log₂ n levels means the pivots were consistently bad, which an adversary that answers comparisons adaptively can force on any pivot rule; such a part is finished by heap sort (the "introsort" idea). Against McIlroy's adversary, 10,000 elements cost about 480,000 comparisons instead of 8.4 million.

#### Examples

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

#### Efficiency

- Time complexity:
  - Worst: O(n log n), thanks to the [Heap](#heap) sort fallback
  - Average: O(n log n)
  - Best: O(n), when the input is sorted, reverse sorted or all equal
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Random input: about n·log₂ n + 0.2n to n·log₂ n + 0.6n (≈ 10,200 for n = 1000, n·log₂ n + 0.6n for n = 100,000 and 1,000,000), up to 0.4n more for elements larger than 16 bytes
  - `k` distinct keys: O(n log k), about 1.1–1.45·n·(log₂ k + 1) in practice
- Space complexity: O(log n) stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: adaptive to sorted and reverse-sorted input and to repeated keys; on nearly sorted input (a few elements out of place) Hoare's scheme would be 1.5–1.75 times faster for integers, since its branches are then predictable. `slice::sort_unstable`, which shares several of these ideas, is 1.1–1.2 times faster on random integers from 100 elements on (it merges short parts through a buffer, which is only sound for types without interior mutability), but this sort is 1.9 times faster on 20 integers, 1.03–1.14 times faster on random strings and as fast or faster with 256-byte elements

### Heap

Heap sort is an unstable, in-place, `O(n log n)` sorting algorithm with constant stack. It arranges the slice into a max-heap, a tree stored in the slice itself in which no element is smaller than its children, so the largest element is at the root; it then repeatedly moves the root to the end of the heap, shrinks the heap by one and restores the heap order. Building the heap takes O(n) comparisons and each of the n removals O(log n), whatever the input. This implementation uses four optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input). Heap sort itself would spend n·log₂ n comparisons on sorted input.
- Bottom-up removal: the textbook sift-down compares the element moved to the root with the larger child at every level, but that element comes from the end of the heap and usually sinks almost to the bottom anyway. Instead, the vacant root moves down to a leaf along the larger children, one comparison per level, and the element moves up from there, usually by a level or two (Wegener's bottom-up heap sort): about n·log₂ n comparisons instead of 1.7–1.85·n·log₂ n.
- Holes instead of swaps: the element being placed is held outside the slice, and each level moves one element into the vacant slot instead of swapping two, one move instead of three.
- Four children per node for large heaps: from 64 KiB of elements on, every level of the heap costs a cache miss, and a heap with four children per node has half as many levels, at three comparisons per level instead of one (about 1.5·n·log₂ n in all). On random input this is 1.15–1.35 times faster for integers from 100,000 elements on, 1.1–1.4 times for strings and 1.25–2 times for 64-byte elements; on nearly sorted strings it is 1.05–1.15 times slower.

Together, against the textbook heap sort, this is 1.25–1.35 times faster for integers from 100,000 elements on, 1.05–1.4 times for strings from 1000 elements on and 1.3–2.2 times for 256-byte elements; on slices of up to 10,000 integers, where comparisons are nearly free, the textbook heap sort is 1.03–1.15 times faster.

#### Examples

Rust usage with integers:

```rust
use a::sort::heap;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  heap(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Rust usage with strings:

```rust
use a::sort::heap;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  heap(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

#### Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when the input is sorted or reverse sorted
- Comparisons:
  - Best (sorted, reverse sorted): n−1
  - Random input: about n·log₂ n + 0.6n (≈ 10,500 for n = 1000, against ≈ 16,800 for the textbook heap sort); about 1.5·n·log₂ n from 64 KiB of elements on, with four children per node
  - Worst: O(n log n), on every input
- Moves: about n·log₂ n, and about 0.5·n·log₂ n with four children per node
- Space complexity: O(1) stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: not adaptive, except to sorted and reverse-sorted input. Heap sort jumps around the slice, so `slice::sort_unstable` and this crate's [Quick](#quick) sort, which falls back to heap sort only when its pivots keep failing, are faster: 3–4 times for random integers, 1.6–6 times for strings and 1.1–3.5 times for 256-byte elements. Heap sort's strength is its guarantee: O(n log n) on every input, in place, with constant stack

### Tim

Tim sort is a stable, `O(n log n)` merge sort that makes use of the order already in the input; Tim Peters designed it for Python's `list.sort` in 2002. It splits the slice into runs, the longest stretches that are ascending or strictly descending, and reverses the descending ones. Runs shorter than a minimum length (16 to 32 elements, chosen so that the slice splits into a power of two runs or slightly fewer) are extended to it by insertion sort, and runs are merged as they are found, in an order kept on a stack, until one is left. Random input has short runs and costs as much as any merge sort, but input made of a few long runs costs little more than finding them. This implementation uses a buffer of `n` elements, kept on the stack when it fits in 4 KiB and otherwise allocated when the first merge needs it, and six optimizations:

- Galloping: before a merge, two searches find the elements of the left run that are not greater than the first of the right run, and those of the right run that are not smaller than the last of the left run; they are already in place and stay out of the merge. During a merge, once one run has supplied seven elements in a row, the merge gallops: it finds how many more of that run's elements come next by probing 1, 2, 4, … elements ahead and binary searching the last gap, and moves them as a block, `k` elements for about `2·log₂ k` comparisons. The threshold drops while galloping pays off and rises when it does not, so random input hardly ever gallops.
- Powersort merge policy: two runs are merged once the boundary between them lies deeper in a perfectly balanced merge tree over the slice than the boundary after them (Munro and Wild's powersort, used by CPython's Tim sort since Python 3.11). This merges in close to the cheapest order possible and never keeps more than 65 runs on the stack; Peters' original stack rules are 1.04–1.12 times slower on input made of runs of varying lengths and no faster elsewhere.
- Bidirectional, branchless merging: like [Merge](#merge) sort, a merge copies both runs out and merges them back from both ends at once, two independent chains of comparisons that copy from an address chosen by the comparison instead of branching on it; each end counts its own winning streaks and gallops on its own. Tim sort's classic merge, which copies only the shorter run out and merges from one end, is 1.15–1.45 times slower on random input, but moves less when one run is much longer than the other, so merges of runs more than four times as long as each other still use it (1.05–1.55 times faster on sorted input with elements appended).
- Pair insertion: short runs are extended by inserting elements two at a time, scanning from the right; the larger one is carried left first and the smaller one goes on from where it stopped, so the elements both pass move once. Peters' binary insertion makes fewer comparisons, but each probe waits for the previous one and its moves go through `memmove`: it is 1.2–1.9 times slower on random input, strings included. Runs of 16 to 32 elements (CPython uses 32 to 64) are 1.1–1.2 times faster on 100 to 1000 random elements.
- Branchless short runs for small elements: each insertion ends in a mispredicted branch, which for elements of at most 8 bytes costs more than everything else, so a short run that starts with fewer than four sorted elements is instead sorted by [Merge](#merge) sort's branchless short-run sort. This is 1.1–1.3 times faster for integers on random input and 1.1–1.2 times with few distinct keys, and up to 1.25 times slower on short slices that are nearly sorted in small steps; for integers it also brings random input down to about n·log₂ n comparisons.
- No allocation for small slices or single runs: the buffer lives on the stack while it fits in 4 KiB (512 `u64`s, say), and is otherwise allocated when the first merge needs it, so sorted and strictly descending input never allocates.

#### Examples

Rust usage with integers:

```rust
use a::sort::tim;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  tim(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

A sorted log with late entries appended is two runs, merged mostly by galloping:

```rust
use a::sort::tim;

fn main() {
  let mut times: Vec<u32> = (0..1000).map(|i| i * 10).collect();
  times.extend([15, 4005, 9999]);
  tim(&mut times);
  assert!(times.windows(2).all(|w| w[0] <= w[1]));
}
```

Equal keys keep their original order (cards are compared by rank only):

```rust
use a::sort::tim;
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
  let ranks = [3, 1, 3, 0, 1, 3, 0, 2, 1, 0];
  let mut cards: Vec<Card> = "abcdefghij".chars().zip(ranks).map(|(c, r)| Card(r, c)).collect();
  tim(&mut cards);
  let order: String = cards.iter().map(|c| c.1).collect();
  assert_eq!(order, "dgjbeihacf");
}
```

#### Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when the input is sorted or strictly descending; O(n + n·H) for input made of runs, `H ≤ log₂ r` being the entropy of the lengths of its `r` runs
- Comparisons:
  - Best (sorted, strictly descending): n−1, with no moves (strictly descending input is reversed)
  - Nearly sorted: about 1,180 comparisons for n = 1000 with 10 random elements appended and 1,800 with 10 random pairs swapped (merge sort: about 3,020 and 6,100)
  - Random input: about n·log₂ n for elements of at most 8 bytes (≈ 9,770 for n = 1000, against ≈ 9,640 for merge sort) and about (1.05–1.17)·n·log₂ n for larger ones (≈ 11,700), whose short runs are built by linear insertion: more comparisons than a binary search would make, but less time
  - Worst: O(n log n)
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once, and only if a merge is needed) plus a stack of at most 65 runs
- Stability: stable (equal elements keep their relative order)
- Adaptivity: very good. Against this crate's merge sort, as fast or up to 1.3 times faster on random integers, 1.05–1.4 times faster on random strings and 1.03–1.1 times on 64-byte elements from 1000 elements on, and 2.7–4.3 times faster on nearly sorted input, appended elements or two interleaved runs (integers and strings) and 7–11 times on reverse-sorted input. Against `slice::sort`, which also finds runs: 1.35–1.7 times slower on random integers and 1.3–1.55 times on random strings, but 1.6–4.2 times faster on nearly sorted input, 2.6–3.9 times on appended elements or two interleaved runs and 1.4–1.8 times on input made of many runs

### Intro

Intro sort (introspective sort) is an unstable, in-place, `O(n log n)` sorting algorithm designed by David Musser in 1997: a quick sort that watches its own progress. Quick sort partitions the slice around a pivot and sorts the two parts the same way; with good pivots each level halves the parts, but no cheap pivot rule is good on every input, and bad pivots make it quadratic. Intro sort counts the levels, and a part still being partitioned after `2·⌊log₂ n⌋` levels is handed to [Heap](#heap) sort, which is `O(n log n)` on every input; typical input never reaches the limit and keeps quick sort's speed. Short parts are left to insertion sort. This implementation keeps Hoare's partitioning comparisons and uses five optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input).
- Branchless block partitioning: Hoare's scheme scans from both ends for elements on the wrong side of the pivot and swaps them, and each scan stops at an unpredictable element, a mispredicted branch. Here each end compares a block of 64 elements first, recording the misplaced ones by offset without branching, and then exchanges as many misplaced pairs as both blocks have (Edelkamp and Weiss's BlockQuicksort), in a cycle of two moves per pair instead of a swap's three. Same comparisons as Hoare's scheme; on random input 1.8–2.7 times faster for integers, 1.7–2.2 times for strings and 1.4–2.1 times for 64-byte elements. The block scans are unrolled by 8, another 1.03–1.08 times on random input and 1.1–1.25 times on nearly sorted input, where the scans are nearly all the work.
- Equal keys split evenly: as in Hoare's and Musser's scheme, an element equal to the pivot stops the scans on both sides, so equal keys are spread over both parts, and input with few distinct keys, even all equal but one, costs O(n log n) without ever reaching the depth limit.
- Pivots away from the ends, from a ninther on long parts: the median of the elements at a quarter, half and three quarters of the part, and from 128 elements on the median of three medians of three (Tukey's ninther), 1.04–1.13 times faster on sawtooth input of 100,000 elements or more. Musser samples the first, middle and last elements; swapping the pivot to the front then moves the smallest element of a sorted stretch into the middle, where the next sample finds it again, so sorted input with a few elements appended keeps getting the second smallest element as its pivot until the depth limit gives up: quartiles are 1.2–4 times faster there.
- Insertion sort while the part is fresh: parts of at most 24 elements (Musser: 16) are insertion sorted as soon as partitioning leaves them, still in the cache, instead of in one pass over the whole slice at the end, which reads the whole slice once more and is 1.05–1.15 times slower for strings.

#### Examples

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

#### Efficiency

- Time complexity:
  - Worst: O(n log n), guaranteed by the heap sort fallback
  - Average: O(n log n)
  - Best: O(n), when the input is sorted or reverse sorted
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Random input: about (1.1–1.19)·n·log₂ n (≈ 11,800 for n = 1000); insertion into parts of up to 24 elements costs more comparisons than partitioning them further would, but less time
  - Few distinct keys: about 0.85·n·log₂ n; unlike [Quick](#quick) sort, intro sort does not take all copies of a pivot out at once
  - Adversarial input: McIlroy's adversary, which answers comparisons so as to make every pivot bad, drives partitioning alone to 9.1 million comparisons for n = 10,000; with the heap sort fallback the whole sort makes about 470,000
- Space complexity: O(log n) stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: adaptive to sorted and reverse-sorted input, and cheap on nearly sorted input, where Hoare's scheme moves only what is misplaced. Against [Quick](#quick) sort: 1.4–1.5 times slower on random integers, 1.06–1.13 times on random strings and 1.5–3.8 times with few distinct keys, but 1.2–1.65 times faster on random 64-byte elements and 1.3–3.6 times faster on nearly sorted input. Against `slice::sort_unstable`: 1.6–1.8 times slower on random integers and within 1.07 times on random strings, but 1.08–1.55 times faster on random 64-byte elements and 1.5–3.7 times on nearly sorted input. Against [Heap](#heap) sort, which it falls back to: 1.75–2.4 times faster on random integers, 2.1–6.7 times on strings and 1.5–4.4 times on 64-byte elements

### Shell

Shell sort, published by Donald Shell in 1959, is an unstable, in-place insertion sort with a gap. A pass with gap `h` insertion sorts every chain of elements `h` apart, so an element far from its place travels there in steps of `h` instead of one slot at a time; the passes use a decreasing sequence of gaps ending with 1, a plain insertion sort that the earlier passes leave with little to do. How fast it is depends almost entirely on the gaps. This implementation uses five optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input).
- Hole shifting: an element is read out once, the greater elements before it on its chain shift one place along the chain each, and it is written once into the slot left free, one move per step instead of a swap's three.
- The gaps that are fastest, not the fewest comparisons: Ciura's sequence (1, 4, 10, 23, 57, 132, 301, 701, then times 2.25) makes 1.1–1.27 times fewer comparisons than Sedgewick's 1982 one (1, 8, 23, 77, 281, 1073, …), but nearly every insertion ends in a mispredicted branch, and Sedgewick's gaps grow by a factor of about 4, so they take fewer passes: for elements larger than 8 bytes they are 1.05–1.35 times faster, strings included (except random strings from a million elements on, 1.1 times slower). Elements of at most 8 bytes use Ciura's gaps, which suit the window below.
- A branchless window for small elements: in a pass over a slice far from sorted, how far an element moves along its chain is about as hard to predict as a coin toss (with Ciura's gaps 0 steps 44% of the time, 1 step 26%, 2 steps 14%, 3 steps 8%, 4 or more 8%). For elements of at most 8 bytes, each element is compared with the 4 elements before it on its chain at once, how many it passes is computed as a number, and the 5 slots are rewritten from copies chosen by an index computed from it; only the 8% that pass all 4 go on with a branch per step. On random input this is 1.4–1.8 times faster for integers.
- The window only while it pays: on a nearly sorted slice insertion's branches are predictable and the window's extra comparisons and writes are wasted (up to 2.6 times slower), so a pass takes the window only if the pass before it moved more than a quarter of the elements.

#### Examples

Rust usage with integers:

```rust
use a::sort::shell;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  shell(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Rust usage with strings:

```rust
use a::sort::shell;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  shell(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

#### Efficiency

- Time complexity:
  - Worst: O(n^(4/3)) with Sedgewick's gaps; no bound is known for Ciura's, which were found by experiment
  - Average: no exact bound is known; measured, 1.6–2.8·n·log₂ n comparisons up to a million elements (no gap sequence is known to make Shell sort O(n log n))
  - Best: O(n), when the input is sorted or reverse sorted
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Random input: about (1.6–2)·n·log₂ n with Sedgewick's gaps (≈ 15,900 for n = 1000) and (2.1–2.8)·n·log₂ n with Ciura's gaps and the window (≈ 21,300), whose unnecessary comparisons are cheap for small elements
- Space complexity: O(1) extra (in-place)
- Stability: unstable (an insertion along a chain can carry an element past an equal one between the two)
- Adaptivity: the passes do less on nearly sorted input. Against classic Shell sort with Ciura's gaps: 1.2–1.75 times faster on random integers, 1.1–1.65 times on nearly sorted ones, 1.05–1.4 times on strings and 1.03–1.3 times on 64-byte elements. Against [Comb](#comb) sort: 2.4–3.3 times slower on random integers from 1000 elements on, but 1.3–2.5 times faster on strings that are nearly sorted, have few distinct keys or follow a pattern. Against [Heap](#heap) sort: 1.3–1.7 times slower on random integers from 1000 elements on, about as fast on random strings, and faster on everything nearly sorted or patterned

### Drift

Drift sort is a stable, `O(n log n)` sorting algorithm designed by Orson Peters and Lukas Bergdoll in 2024; it has been the stable sort of Rust's standard library since 1.81. It drifts between merge sort and quick sort: it scans for natural runs, ascending or strictly descending, but keeps only those of at least about `√n` elements (`min(n/2, 32)` below 4096 elements), and leaves everything else as unsorted chunks. Runs and chunks are combined in powersort's merge order: two adjacent unsorted chunks are simply taken together, while an unsorted chunk that meets a sorted run is first sorted by a stable quick sort and then merged. Random input ends up as one chunk for quick sort, which handles repeated keys in `O(n log k)`; input made of long runs is merged like [Tim](#tim) sort's. This implementation uses a buffer of `n` elements, kept on the stack when it fits in 4 KiB and otherwise allocated when first needed, and nine optimizations (ratios measured with new input for every sort):

- This crate's best parts: runs are merged by [Tim](#tim) sort's merge (skipping elements already in place, merging from both ends at once without branching, galloping through runs that interleave in blocks, and running two merges at once once galloping has not paid off: 1.25–1.5 times faster on runs of random content); short parts are sorted by [Merge](#merge) sort's branchless short-run sort; and a quick sort still partitioning after `2·log₂ n` levels hands the part to merge sort, so the sort is `O(n log n)` on any input.
- Pivot samples at several scales: [Quick](#quick) sort's ninther and evenly spaced pseudo-median sample at one scale, which periodic input whose period divides the spacing defeats (a sawtooth of 1000 elements cost 2.4 times the standard library's time). The samples lie in three sections of an eighth of the width, centred at the middle and `3/8` of the width to either side, recursively: 9 samples at two scales from 64 elements, 27 at three from 1024, three times as many for every 16 times the length. The sawtooth then costs 3.3–3.7 times less and random input 1–2% fewer comparisons.
- Branchless stable partitioning: each element is compared with the pivot and copied into the buffer, to the front if it goes left and to the back otherwise, from an address the comparison selects, and both parts are copied back, the back one reversed; a partition that moved nothing skips the copies back (1.05–1.3 times faster for integers with 2 distinct keys). Elements larger than 8 bytes are compacted in place as they go left, only the right part passing through the buffer, with the pivot moved into a local as its only live copy and a guard that restores every element if a comparison panics: 1.2–1.3 times faster on random 64-byte elements, 1.2–1.5 times with repeated keys, 1.1–1.2 times on random strings from 10,000 elements.
- Repeated keys in one pass, without copying the pivot: when a part's pivot is no greater than the pivot of the partition that produced the part, every element equal to it is taken out in one pass; the earlier pivot is found in place (the standard library compares a bitwise copy, sound only for types without interior mutability). Where no earlier pivot bounds a part, it is first scanned for an element smaller than the pivot, a comparison or two on most input, which spares a partition that moves nothing: 1.08–1.25 times faster for integers with 2 distinct keys.
- No per-element bookkeeping: the partition loop is split at the pivot and at the earlier pivot instead of checking for them at every element, and for elements of at most 8 bytes it is unrolled by 4.
- Short parts noted for order and merged in place: a part of up to 32 elements of at most 8 bytes is scanned for descents with one comparison per element and no branch on the answers; runs and stretches of the merge tree without one are left alone, the rest is sorted by the short-run sort, and every merge reads its halves where the level below left them and writes the other place, so no level is copied out before it is merged. The merges leave no stale copy behind when a comparison mutates its arguments: a merge of up to 16 elements decides the order before it copies anything, a longer one copies the few elements compared after their copy was written once more (1.03–1.05 times slower on random integers of 100 to 10,000 elements than merges that compare stale copies, which are sound only for types without interior mutability).
- Short slices adapt too: a slice of up to 64 elements of at most 8 bytes gets the same scan for descents: none and it is left alone, one at every element and it is reversed, at most one per 8 elements and the rest is inserted into the sorted prefix, otherwise it is sorted with the descents found (1.2–3 times faster than scanning for a prefix run and sorting the rest on random, few-keys, sorted, reverse-sorted or appended input of 8–64 integers; 1.1–1.25 times slower on 8–28 nearly sorted ones). Larger elements are scanned for a sorted or strictly descending prefix and the rest inserted, without a buffer, up to 28 elements (40 beyond 32 bytes), and up to twice the small sort's length sort each chunk as soon as it is found and merge the chunks, with no chunk scanned twice.
- Runs cut by a chunk boundary are taken back: a natural run whose rest falls short of the length a run must have is probed with one or two comparisons and, if they are in order, scanned backwards into the unsorted chunk before it, the boundary moving with it: 1.05–1.2 times faster on input made of runs of about `√n` elements.
- No allocation for small slices or single runs: the buffer lives on the stack while it fits in 4 KiB and is otherwise allocated when first needed; short slices of large elements are insertion sorted and need none.

#### Examples

Rust usage with integers:

```rust
use a::sort::drift;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  drift(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Equal keys keep their original order (cards are compared by rank only):

```rust
use a::sort::drift;
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
  let ranks = [3, 1, 3, 0, 1, 3, 0, 2, 1, 0];
  let mut cards: Vec<Card> = "abcdefghij".chars().zip(ranks).map(|(c, r)| Card(r, c)).collect();
  drift(&mut cards);
  let order: String = cards.iter().map(|c| c.1).collect();
  assert_eq!(order, "dgjbeihacf");
}
```

#### Efficiency

- Time complexity:
  - Worst: O(n log n), guaranteed by the merge sort fallback of the quick sort
  - Average: O(n log n), and O(n log k) with `k` distinct keys
  - Best: O(n), when the input is sorted or strictly descending
- Comparisons:
  - Best (sorted, strictly descending): n−1, with no moves (strictly descending input is reversed)
  - Random input: about (1.06–1.12)·n·log₂ n (≈ 11,200 for n = 1000, the standard library 10,600; elements of at most 8 bytes trade about 4% more comparisons for fewer mispredicted branches)
  - Few distinct keys: about 0.22–0.28·n·log₂ n with 2 keys and 0.38–0.57·n·log₂ n with 16 (≈ 2,800 and 5,700 for n = 1000, against 5,610 and 8,780 for [Tim](#tim) sort); about 2,050 with 10 random pairs swapped and 7,100 on a sawtooth of 1000
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once and only when needed; the standard library makes do with about n/2 for large inputs) plus O(log n) stack
- Stability: stable (equal elements keep their relative order)
- Soundness: a comparison that panics, answers inconsistently or mutates its arguments through interior mutability never leaves a stale copy, a duplicate or a hole behind
- Adaptivity: adaptive to runs of `√n` elements or more, even where a chunk boundary cuts them, and to repeated keys. Against `slice::sort`, the standard library's drift sort, with new input for every sort: 1.3–3 times faster on 8–64 random integers, 1.03–1.06 times slower on 100–100,000 and as fast beyond; 1.05–1.07 times slower on random strings up to 1000 elements, 1.03–1.1 times faster from 10,000 and 1.7 times on a million; 1.1 times slower on random 64-byte elements up to 100 and 1.03–1.25 times faster from 1000; 1.05–1.9 times faster with few or `√n` distinct keys from 1000 elements and 1.25–1.65 times on a sawtooth of 1000; and 1.4–8 times faster on nearly sorted, appended, block-wise sorted input or runs of `√n` elements (runs from 10,000 elements on). Against [Tim](#tim) sort: 1.3–2.1 times faster on random integers and strings, 4.5–7 times with few distinct keys and 1.3–2.4 times on sawtooth input up to 100,000 elements; within 1.05 times on nearly sorted input from 10,000 elements (1.15 times slower below) and 1.05–1.4 times slower on runs of `√n` elements, which Tim sort keeps whole

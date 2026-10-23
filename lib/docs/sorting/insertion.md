[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Insertion

Insertion sort is a simple, stable, in-place sorting algorithm. It grows a sorted prefix from the left, inserting each new element after every prefix element that is not greater than it. This implementation is a pair insertion sort (the variant Java's dual-pivot quicksort uses for short ranges) with three optimizations:

- Sorted fast path: the smaller element of each new pair is first compared with the last element of the prefix; if it is not smaller, the pair is already in place, so sorted input costs exactly n−1 comparisons.
- Galloping search: an insertion point is found by probing 1, 2, 4, 8, … slots to the left and then binary searching the last gap, so an element that moves `d` slots costs at most `2⌊log₂ d⌋ + 2` comparisons instead of `d`.
- Pair moves: two elements are inserted at once; the smaller one is only searched for left of the larger one's slot, and prefix elements greater than both shift right by two slots in a single move, so they move once instead of twice. Short moves of elements of up to 32 bytes are made in place, four elements at a time, instead of through `memmove`, which costs them more than the copying (1.05–1.3 times faster for integers on short or nearly sorted slices and up to 1.5 times on short reverse-sorted ones, within 1.03 times either way on long random ones).

## Examples

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

## Efficiency

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

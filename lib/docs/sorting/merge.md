[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Merge

Merge sort is a stable, `O(n log n)` sorting algorithm that uses a buffer. It splits the slice into two halves, sorts each half recursively, and merges the two sorted halves into one. Every element takes part in one merge per level of recursion and there are about `log₂ n` levels, so the cost hardly depends on the input. This implementation uses a buffer of `n` elements, kept on the stack when it fits in 4 KiB and allocated once otherwise, and seven optimizations:

- Short runs sorted without branches: halves of at most 8 elements are not split further. A run is first scanned for order, which leaves a sorted run alone after len−1 comparisons and otherwise stops at the first descent, usually after one or two; then its two halves of at most 4 elements are sorted straight into the buffer by fixed sequences of comparisons that select where each element goes instead of branching on the outcome (1, 3 and 5 comparisons for 2, 3 and 4 elements, the last being the standard library's stable four-element sort), and merged back. Short runs took up to 60% of the time when they were insertion sorted; together with the stack buffer below, this makes the whole sort 1.2–2.8 times faster for integers (most for small slices) and 1.02–1.9 times for strings and large elements.
- Short slices of small elements noted for order: for elements of at most 8 bytes, a half of up to 64 elements is first scanned for descents (elements smaller than the one before) with one comparison per element and no branch on the answers. None and it is left alone, one at every element and it is reversed, otherwise runs and stretches of the recursion without a descent are left alone and the rest is sorted as above (random integers 1.1 times faster for about n more comparisons, sorted input 3.3–4 times, reverse-sorted 2.8–8 times).
- Ordered halves are not merged: if the first element of the right half is not smaller than the last element of the left half, one comparison replaces the merge, so already-sorted input costs exactly n−1 comparisons and is never moved.
- Reversed halves are swapped: if the last element of the right half is smaller than the first element of the left half, the halves trade places as blocks with no further comparison, so reverse-sorted input costs about 2.6n comparisons instead of n·log₂ n (about n for elements of at most 8 bytes).
- No level copied out before it is merged: every merge reads its halves where the level below left them, in the slice or in the buffer, and writes the other place, so each level moves every element once instead of twice, and the result is copied back at most once (1.03–1.1 times faster for integers and strings on random input, 1.06–1.45 times for 64-byte elements).
- Bidirectional, branchless merging: each merge proceeds from both ends at once, the front taking the smaller of the two first elements and the back the larger of the two last ones. The two independent chains of comparisons overlap in the processor, and every step copies from an address chosen by the comparison instead of branching on it. On random input this is 1.2–1.6 times faster than the classic merge that copies only the left half out and merges from the front.
- Long merges split in two or four: a merge of 128 elements or more is split at the middle of its result by a binary search (co-ranking) into two independent merges from both ends, four chains the processor overlaps, and from 4096 elements of at most 32 bytes into four; each part stops when a half has at most one element left, so it never compares an element whose copy is written already (1.13–1.4 times faster for integers, 1.12–1.55 times for strings and 1.08–1.2 times for 64-byte elements from 1000 elements on).
- No allocation for small slices: a buffer of up to 4 KiB lives on the stack, so a slice that fits in it (512 `u64`s, say) is sorted without touching the allocator.

## Examples

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

## Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when the input is already sorted (or made of ordered or reversed halves at every level)
- Comparisons and moves (a merge of m elements makes at most 2⌊m/2⌋ comparisons after the two checks, fewer when a half runs out early, plus about 3·log₂ m for the binary searches of a split merge, and moves every element once, from where its halves are to the other place):
  - Best (already sorted): n−1 comparisons, 0 moves
  - Reverse sorted: about 2.6n comparisons (2,590 for n = 1000, against 5,044 for the textbook merge sort; about n for elements of at most 8 bytes, 1,010 for n = 1000)
  - Random input: about n·log₂ n − 0.35n comparisons (≈ 9,570 for n = 1000; ≈ 10,290 for elements of at most 8 bytes, the scans for descents included; the textbook merge sort makes ≈ 8,700, since the checks cost a comparison or two per merge and a bidirectional merge does not stop when one half runs out) and about n·log₂(n/8) + 2n moves
  - Worst: at most n·log₂ n + O(n) comparisons and 1.5n·⌈log₂(n/8)⌉ + 3n moves
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once) plus O(log n) stack
- Stability: stable (equal elements keep their relative order)
- Adaptivity: adaptive at the scale of runs and halves (sorted and reverse-sorted input are nearly linear). Against `slice::sort`, with new input for every sort: 1.6–2.4 times faster on up to 64 random integers and 1.08–1.23 times slower from 1000 on, 1.07–1.09 times slower on random strings of 1000 to 100,000 (1.65 times faster on a million), 1.1–1.35 times faster on random 64-byte elements from 1000 on and 1.2–2.6 times on nearly sorted integers, but 3–6 times slower with few distinct keys and 1.6–10 times on reverse-sorted input from 1000 elements on, which the standard library's stable sort handles in one pass

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Drift

Drift sort is a stable, `O(n log n)` sorting algorithm designed by Orson Peters and Lukas Bergdoll in 2024; it has been the stable sort of Rust's standard library since 1.81. It drifts between merge sort and quick sort: it scans for natural runs, ascending or strictly descending, but keeps only those of at least about `√n` elements (`min(n/2, 32)` below 4096 elements), and leaves everything else as unsorted chunks. Runs and chunks are combined in powersort's merge order: two adjacent unsorted chunks are simply taken together, while an unsorted chunk that meets a sorted run is first sorted by a stable quick sort and then merged. Random input ends up as one chunk for quick sort, which handles repeated keys in `O(n log k)`; input made of long runs is merged like [Tim](tim.md) sort's. This implementation uses a buffer of `n` elements, kept on the stack when it fits in 4 KiB and otherwise allocated when first needed, and nine optimizations (ratios measured with new input for every sort):

- This crate's best parts: runs are merged by [Tim](tim.md) sort's merge (skipping elements already in place, merging from both ends at once without branching, galloping through runs that interleave in blocks, and running two merges at once once galloping has not paid off: 1.25–1.5 times faster on runs of random content); short parts are sorted by [Merge](merge.md) sort's branchless short-run sort; and a quick sort still partitioning after `2·log₂ n` levels hands the part to merge sort, so the sort is `O(n log n)` on any input.
- Pivot samples at several scales: [Quick](quick.md) sort's ninther and evenly spaced pseudo-median sample at one scale, which periodic input whose period divides the spacing defeats (a sawtooth of 1000 elements cost 2.4 times the standard library's time). The samples lie in three sections of an eighth of the width, centred at the middle and `3/8` of the width to either side, recursively: 9 samples at two scales from 64 elements, 27 at three from 1024, three times as many for every 16 times the length. The sawtooth then costs 3.3–3.7 times less and random input 1–2% fewer comparisons.
- Branchless stable partitioning: each element is compared with the pivot and copied into the buffer, to the front if it goes left and to the back otherwise, from an address the comparison selects, and both parts are copied back, the back one reversed; a partition that moved nothing skips the copies back (1.05–1.3 times faster for integers with 2 distinct keys). Elements larger than 8 bytes are compacted in place as they go left, only the right part passing through the buffer, with the pivot moved into a local as its only live copy and a guard that restores every element if a comparison panics: 1.2–1.3 times faster on random 64-byte elements, 1.2–1.5 times with repeated keys, 1.1–1.2 times on random strings from 10,000 elements.
- Repeated keys in one pass, without copying the pivot: when a part's pivot is no greater than the pivot of the partition that produced the part, every element equal to it is taken out in one pass; the earlier pivot is found in place (the standard library compares a bitwise copy, sound only for types without interior mutability). Where no earlier pivot bounds a part, it is first scanned for an element smaller than the pivot, a comparison or two on most input, which spares a partition that moves nothing: 1.08–1.25 times faster for integers with 2 distinct keys.
- No per-element bookkeeping: the partition loop is split at the pivot and at the earlier pivot instead of checking for them at every element, and for elements of at most 8 bytes it is unrolled by 4.
- Short parts noted for order and merged in place: a part of up to 32 elements of at most 8 bytes is scanned for descents with one comparison per element and no branch on the answers; runs and stretches of the merge tree without one are left alone, the rest is sorted by the short-run sort, and every merge reads its halves where the level below left them and writes the other place, so no level is copied out before it is merged. The merges leave no stale copy behind when a comparison mutates its arguments: a merge of up to 16 elements decides the order before it copies anything, a longer one copies the few elements compared after their copy was written once more (1.03–1.05 times slower on random integers of 100 to 10,000 elements than merges that compare stale copies, which are sound only for types without interior mutability).
- Short slices adapt too: a slice of up to 64 elements of at most 8 bytes gets the same scan for descents: none and it is left alone, one at every element and it is reversed, at most one per 8 elements and the rest is inserted into the sorted prefix, otherwise it is sorted with the descents found (1.2–3 times faster than scanning for a prefix run and sorting the rest on random, few-keys, sorted, reverse-sorted or appended input of 8–64 integers; 1.1–1.25 times slower on 8–28 nearly sorted ones). Larger elements are scanned for a sorted or strictly descending prefix and the rest inserted, without a buffer, up to 28 elements (40 beyond 32 bytes), and up to twice the small sort's length sort each chunk as soon as it is found and merge the chunks, with no chunk scanned twice.
- Runs cut by a chunk boundary are taken back: a natural run whose rest falls short of the length a run must have is probed with one or two comparisons and, if they are in order, scanned backwards into the unsorted chunk before it, the boundary moving with it: 1.05–1.2 times faster on input made of runs of about `√n` elements.
- No allocation for small slices or single runs: the buffer lives on the stack while it fits in 4 KiB and is otherwise allocated when first needed; short slices of large elements are insertion sorted and need none.

## Examples

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

## Efficiency

- Time complexity:
  - Worst: O(n log n), guaranteed by the merge sort fallback of the quick sort
  - Average: O(n log n), and O(n log k) with `k` distinct keys
  - Best: O(n), when the input is sorted or strictly descending
- Comparisons:
  - Best (sorted, strictly descending): n−1, with no moves (strictly descending input is reversed)
  - Random input: about (1.06–1.12)·n·log₂ n (≈ 11,200 for n = 1000, the standard library 10,600; elements of at most 8 bytes trade about 4% more comparisons for fewer mispredicted branches)
  - Few distinct keys: about 0.22–0.28·n·log₂ n with 2 keys and 0.38–0.57·n·log₂ n with 16 (≈ 2,800 and 5,700 for n = 1000, against 7,050 and 9,390 for [Tim](tim.md) sort with elements of at most 8 bytes and 5,290 and 10,380 with larger ones); about 2,050 with 10 random pairs swapped and 7,100 on a sawtooth of 1000
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once and only when needed; the standard library makes do with about n/2 for large inputs) plus O(log n) stack
- Stability: stable (equal elements keep their relative order)
- Soundness: a comparison that panics, answers inconsistently or mutates its arguments through interior mutability never leaves a stale copy, a duplicate or a hole behind
- Adaptivity: adaptive to runs of `√n` elements or more, even where a chunk boundary cuts them, and to repeated keys. Against `slice::sort`, the standard library's drift sort, with new input for every sort: 1.3–3 times faster on 8–64 random integers, 1.03–1.06 times slower on 100–100,000 and as fast beyond; 1.05–1.07 times slower on random strings up to 1000 elements, 1.03–1.1 times faster from 10,000 and 1.7 times on a million; 1.1 times slower on random 64-byte elements up to 100 and 1.03–1.25 times faster from 1000; 1.05–1.9 times faster with few or `√n` distinct keys from 1000 elements and 1.25–1.65 times on a sawtooth of 1000; and 1.4–8 times faster on nearly sorted, appended, block-wise sorted input or runs of `√n` elements (runs from 10,000 elements on). Against [Tim](tim.md) sort: 1.05–1.3 times faster on random integers from 1000 elements on, 1.4–1.65 times on random strings and 1.2–1.3 times on 64-byte elements, 1.6–9.3 times with few distinct keys (3.4–9.3 times for integers) and 1.25–2.55 times on sawtooth input up to 100,000 elements; 1.1–1.4 times slower on nearly sorted integers (1.15–2 times for strings) and 1.05–1.4 times slower on runs of `√n` elements, which Tim sort keeps whole

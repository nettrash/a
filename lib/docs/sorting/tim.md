[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Tim

Tim sort is a stable, `O(n log n)` merge sort that makes use of the order already in the input; Tim Peters designed it for Python's `list.sort` in 2002. It splits the slice into runs, the longest stretches that are ascending or strictly descending, and reverses the descending ones. Runs shorter than a minimum length (16 to 32 elements, chosen so that the slice splits into a power of two runs or slightly fewer) are extended to it by insertion sort (to twice it for elements of at most 8 bytes), and runs are merged as they are found, in an order kept on a stack, until one is left. Random input has short runs and costs as much as any merge sort, but input made of a few long runs costs little more than finding them. This implementation uses a buffer of `n` elements, kept on the stack when it fits in 4 KiB and otherwise allocated when the first merge needs it, and six optimizations:

- Galloping: before a merge, two searches find the elements of the left run that are not greater than the first of the right run, and those of the right run that are not smaller than the last of the left run; they are already in place and stay out of the merge. During a merge, once one run has supplied seven elements in a row, the merge gallops: it finds how many more of that run's elements come next by probing 1, 2, 4, … elements ahead and binary searching the last gap, and moves them as a block, `k` elements for about `2·log₂ k` comparisons. The threshold drops while galloping pays off and rises when it does not, so random input hardly ever gallops.
- Powersort merge policy: two runs are merged once the boundary between them lies deeper in a perfectly balanced merge tree over the slice than the boundary after them (Munro and Wild's powersort, used by CPython's Tim sort since Python 3.11). This merges in close to the cheapest order possible and never keeps more than 65 runs on the stack; Peters' original stack rules are 1.04–1.12 times slower on input made of runs of varying lengths and no faster elsewhere.
- Bidirectional, branchless merging: like [Merge](merge.md) sort, a merge copies both runs out and merges them back from both ends at once, two independent chains of comparisons that copy from an address chosen by the comparison instead of branching on it; each end counts its own winning streaks and gallops on its own. Tim sort's classic merge, which copies only the shorter run out and merges from one end, is 1.15–1.45 times slower on random input, but moves less when one run is much longer than the other, so merges of runs more than four times as long as each other still use it (1.05–1.55 times faster on sorted input with elements appended).
- Pair insertion: short runs are extended by inserting elements two at a time, scanning from the right; the larger one is carried left first and the smaller one goes on from where it stopped, so the elements both pass move once. Peters' binary insertion makes fewer comparisons, but each probe waits for the previous one and its moves go through `memmove`: it is 1.2–1.9 times slower on random input, strings included. Runs of 16 to 32 elements (CPython uses 32 to 64) are 1.1–1.2 times faster on 100 to 1000 random elements when every short run is extended by insertion.
- Branchless short runs for small elements: each insertion ends in a mispredicted branch, which for elements of at most 8 bytes costs more than everything else, so for them a natural run shorter than the minimum length is extended to twice it (32 to 64 elements, as in CPython; natural runs of the minimum length are kept), and a short run that starts with fewer than four sorted elements, or whose next elements dip below its end, is sorted by [Merge](merge.md) sort's branchless sort for short slices, which notes the descents and leaves alone what is in order. Against insertion this is 1.1–1.3 times faster for integers on random input and 1.1–1.2 times with few distinct keys, and up to 1.25 times slower on short slices that are nearly sorted in small steps; the runs of twice the length are another 1.08–1.3 times faster on random input and with few distinct keys and 1.05–1.45 times on input made of runs up to 1000 elements, up to 1.07 times slower on runs of about 45 elements and 1.05–1.75 times on a sawtooth of 40 to 1000 elements.
- No allocation for small slices or single runs: the buffer lives on the stack while it fits in 4 KiB (512 `u64`s, say), and is otherwise allocated when the first merge needs it, so sorted and strictly descending input never allocates.

## Examples

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

## Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when the input is sorted or strictly descending; O(n + n·H) for input made of runs, `H ≤ log₂ r` being the entropy of the lengths of its `r` runs
- Comparisons:
  - Best (sorted, strictly descending): n−1, with no moves (strictly descending input is reversed)
  - Nearly sorted: about 1,180 comparisons for n = 1000 with 10 random elements appended and 1,800 with 10 random pairs swapped (2,060 for elements of at most 8 bytes; merge sort: about 1,800 and 3,900)
  - Random input: about 1.06·n·log₂ n for elements of at most 8 bytes (≈ 10,580 for n = 1000, against ≈ 10,290 for merge sort) and about (1.05–1.17)·n·log₂ n for larger ones (≈ 11,700, merge sort ≈ 9,570), whose short runs are built by linear insertion: more comparisons than a binary search would make, but less time
  - Worst: O(n log n)
- Space complexity: O(n) extra (one buffer of n elements, on the stack up to 4 KiB, otherwise allocated once, and only if a merge is needed) plus a stack of at most 65 runs
- Stability: stable (equal elements keep their relative order)
- Adaptivity: very good. Against this crate's merge sort, which no longer copies a level out before merging it and splits long merges, and so is the faster of the two on input without long runs: 1.1–1.5 times slower on random input from 1000 elements on and up to 1.35 times on short runs of random content, but 1.2–1.5 times faster on nearly sorted integers and strings, 1.45–2.2 times on sorted input with elements appended and 1.7–11 times on reverse-sorted input from 1000 elements on. Against `slice::sort`, which also finds runs: 1.2–1.45 times slower on random integers from 1000 elements on and 1.35–1.45 times on random strings (1.5 times faster on a million), but 1.5–3.7 times faster on nearly sorted input, 1.9–3.4 times on appended elements and 1.6–3.6 times on input made of runs of `√n` elements from 10,000 on, for integers and strings

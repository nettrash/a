[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Pigeonhole

Pigeonhole sort gives every possible key from the smallest to the largest a hole of its own, puts each element into the hole of its key, and reads the holes out in order: `O(n + k)` time for `k` keys. This implementation keeps the holes in the slice itself: the keys are counted, which gives every hole its place, and each element out of place is then moved straight into the next free place of its hole, the element there taken out in turn, around the cycle until an element lands where the first one was. Every element moves at most once, in place, with no buffer for the elements; it is not stable. Keys are the same as counting sort's (the sealed `IntegerKey` trait), and so is the range rule. It comes in two forms: `pigeonhole` sorts the keys themselves, for which [Counting](counting.md) sort, which rewrites equal values from their counts, is 2.2–7 times faster on random values and 1.4–2.25 times on nearly sorted ones, and `pigeonhole_by_key` sorts any type in place by such a key, calling `key` exactly once per element. This implementation uses eight optimizations:

- Sorted and reverse-sorted input is recognised (non-increasing input too, as the sort is not stable), and left alone or reversed.
- Large slices are placed by swaps: following a cycle, each step waits for the one before it, and for memory once the slice no longer fits the caches. From 32,768 elements on (more for large elements), unless the input is mostly in order, the element at every free place is swapped with the next free place of its own hole instead; the target of a swap depends on its own key only, so the processor overlaps the swaps' waits for memory (1.1–4.3 times faster from 100,000 elements on). Two holes count as one more element towards that length, the counts become the starts of the holes in place, and the holes with free places are listed without a branch (1.35–1.65 times faster from 40,000 elements on, 1.2–1.4 times from 20,000 with four keys per element).
- A few distinct keys are counted in eight tables (at most 256 keys, 16 elements per key) and, with at most 16, placed by swaps at any length, as each step of a cycle would wait on the step before it in the same few holes (2–3 times faster with four distinct keys up to 10,000 elements, 1.5 times beyond).
- Input mostly in order, and ranges of two keys per element or more, walk the places rather than the holes: the cycles start from each place in turn, with no loop per hole whose end mispredicts (3.5–4.5 times faster on nearly sorted input from 100 elements on); when the first keys are mostly in order they are also counted four at a time, so that a run of equal keys waits for memory once per four (twice as fast with sixteen keys per hole).
- Holes move instead of keys: `pigeonhole_by_key` rewrites each key, as it counts it, as its hole in the fewest bytes that hold every hole of the range (1.2–1.8 times faster at 10,000 elements).
- Holes are counted in `u16` up to 65,535 elements, the next free place of each hole needs no clearing, and a range of more than 32 keys per element skips its empty holes eight at a time (1.1–2.2 times faster on a few elements over a wide range).
- Dense ranges walk their holes in a plain loop with no branch on whether a hole is empty, which mispredicts where about a third of them are.
- No allocation for small sorts: keys and counts of up to 4 KiB each live on the stack.

## Examples

Rust usage with integers:

```rust
use a::sort::pigeonhole;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  pigeonhole(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Sorting by a key, in place (elements with equal keys may come in any order):

```rust
use a::sort::pigeonhole_by_key;

fn main() {
  let mut v = vec![(3, 'a'), (1, 'b'), (0, 'c'), (2, 'd')];
  pigeonhole_by_key(&mut v, |&(k, _)| k);
  assert_eq!(v, vec![(0, 'c'), (1, 'b'), (2, 'd'), (3, 'a')]);
}
```

## Efficiency

- Time complexity:
  - Worst, average and best: O(n + k) for `k` possible keys between the smallest and the largest
  - Sorted and reverse-sorted input (also non-increasing for `pigeonhole_by_key`): O(n), one scan
- Moves: each element at most once along the cycles, or twice by swaps for large slices and few distinct keys; none for elements already in their holes
- Calls to `key` (`pigeonhole_by_key`): exactly n
- Space complexity: O(k) counts for `pigeonhole`; O(n + k) for `pigeonhole_by_key` (a buffer of n keys, rewritten as holes, and the counts), with no buffer for the elements; each on the stack up to 4 KiB, otherwise allocated once
- Stability: not stable (`pigeonhole` cannot tell equal keys apart anyway)
- Against the textbook pigeonhole sort (a vector per hole, the elements pushed in and read out): 2–6 times faster on random keys and 160–340 times on sorted `u32` input
- Against the standard library, from 100 elements on: `pigeonhole` is 1.15–2.2 times faster than `slice::sort_unstable` on random `u32` keys in a range of about `n` keys from 1000 elements on and 4–7 times on nearly sorted input, between 1.25 times faster and 1.2 times slower on a range of `4n` keys, and as fast to 1.17 times slower with four distinct keys; `pigeonhole_by_key` is 1.05–2.8 times faster than `slice::sort_unstable_by_key` on random `u64` keys of 16-byte elements up to 10,000 elements (1.35–1.45 times beyond), 1.7–5 times for 72-byte elements and 2–16 times on nearly sorted input, but 1.2–2.1 times slower on sorted and reverse-sorted input and 1.1–1.3 times slower with four distinct keys of 16-byte elements. [Counting](counting.md) sort's stable `counting_by_key`, which moves the elements through a buffer, is 1.15–3.1 times faster on random keys; `pigeonhole_by_key` is 1.2–2.35 times faster on nearly sorted input of 72- and 128-byte elements, and needs no buffer for them

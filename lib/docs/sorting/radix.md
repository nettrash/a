[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Radix

LSD radix sort, used on punched-card sorters since the 1920s, does not compare elements either. It reads every key as a number of a few digits and sorts the slice by each digit in turn, from the least significant to the most, with a stable counting pass; as each pass keeps the order the earlier ones made among equal digits, the slice ends up sorted by the whole key. For `w`-bit keys and `r`-bit digits that is `⌈w / r⌉` passes of `O(n + 2^r)` each, whatever the order of the input: unlike [Counting](counting.md) sort, any range of keys can be sorted. Keys are the same as counting sort's (the sealed `IntegerKey` trait), signed integers read with their sign bit flipped. It comes in two forms: `radix` sorts the keys themselves, and `radix_by_key` sorts any type by such a key, stably, calling `key` exactly once per element. This implementation uses eight optimizations:

- Sorted and reverse-sorted input is recognised: `radix` scans for one ascending or one non-increasing run, 16 neighbours at a time with no exit inside a block, and `radix_by_key` checks for one ascending or one strictly descending run in the pass that reads the keys; a slice that is one run is left alone or reversed, and nothing is counted.
- Only the bits in which keys differ are sorted, in digits sized by the slice: one pass ORs together every key's difference from the first, and the span from its lowest to its highest set bit is cut into the fewest digits of equal width of at most `log₂ n` bits (at least 8 from 64 elements on, at most 12), the digit size CLRS derives for radix sort. Against byte digits with shared bytes skipped: 1.1–1.25 times faster on random integers from 10,000 elements on and 1.75–3.4 times on integers below `n`.
- Short slices count their digits side by side: every pass costs the sum of its counts however few elements it moves, 256 of them for an 8-bit digit. Below 80 elements per 8-bit digit (640 for 64-bit keys), the counts of up to eight 8-bit digits are kept in rows of eight lanes, one per digit (4 KiB), and turned into places by one vectorized running sum per lane for all digits at once, the first half of the rows summed upwards and the second downwards: fewer passes than `log₂ n`-bit digits take, for about the cost of one sum. This pays when the `s` passes saved against 6-bit digits make `n·s²` at least 144; otherwise, from 24 elements on, 6-bit digits are taken, whose 64 counts are summed while the pass before waits on memory. In lanes, `radix` sorts signed values as their bits, flipping the sign bit once before and once after the passes rather than in every pass. 1.15–1.8 times faster on 16 to 256 random `u64`s (`radix_by_key`: 1.15–1.6 times for 16-byte elements, 1.35–1.5 times for 72-byte ones), 1.1–1.45 times on 32 to 256 random `u32`s.
- One digit is a counting sort: when a single digit holds every bit in which the keys differ, its pass is this crate's [Counting](counting.md) sort, with its count tables (`radix`, 2.4–5.4 times faster with four distinct values) or its segments (`radix_by_key`, 1.3–1.8 times faster with four distinct keys).
- Every count in one pass, and no copy back: the counts of all digits are made in one pass, a digit all keys share is skipped, and the passes move the slice into the buffer and back alternately. The counts of a digit become places four at a time, one addition per four waiting for the one before it (1.15–1.2 times faster on 100 random keys).
- Runs of a digit do not wait on each other: the counts walk four quarters of the slice side by side, as they do not depend on the order of the keys (1.6 times faster on nearly sorted input, whose high digits neighbours share), and every pass places four keys at a time, all four counts read before any is written back (1.7–2.2 times faster with four distinct wide keys).
- Large elements are moved once: when `radix_by_key`'s passes would move at least 96 bytes of every element and the slice takes at most 16 MiB, the keys are sorted with the indices of their elements, which are then gathered once (1.25 times faster for 32-byte elements, up to 1.6 times for 72-byte ones).
- No allocation for small sorts: keys, elements and counts (`u16` up to 65,535 elements, `u32` up to `u32::MAX`) each live on the stack up to 4 KiB.

## Examples

Rust usage with integers:

```rust
use a::sort::radix;

fn main() {
  let mut v = vec![i64::MAX, -1, i64::MIN, 0, 1 << 40];
  radix(&mut v);
  assert_eq!(v, vec![i64::MIN, -1, 0, 1 << 40, i64::MAX]);
}
```

Sorting by a key, stably:

```rust
use a::sort::radix_by_key;

fn main() {
  let mut v = vec![(3, 'a'), (1, 'b'), (3, 'c'), (0, 'd'), (1, 'e')];
  radix_by_key(&mut v, |&(k, _)| k);
  assert_eq!(v, vec![(0, 'd'), (1, 'b'), (1, 'e'), (3, 'a'), (3, 'c')]);
}
```

## Efficiency

- Time complexity:
  - Worst, average and best: O(n·b/r) for keys that differ in `b` bits and digits of `r` bits (`r` ≈ log₂ n, between 8 and 12 from 64 elements on; 6 or 8 for short slices)
  - Sorted and reverse-sorted input (strictly descending for `radix_by_key`): O(n), one scan
- Comparisons: none between elements, except the scan for a sorted or reverse-sorted run
- Calls to `key` (`radix_by_key`): exactly n
- Space complexity: O(n) buffers (`radix`: n values; `radix_by_key`: n keys, n elements and, when gathering, 2n keys with indices) plus the counts; each on the stack up to 4 KiB, otherwise allocated once
- Stability: `radix_by_key` is stable; `radix` cannot tell equal values apart
- Against the textbook LSD radix sort (a pass per byte through a buffer copied back each time): `radix` is 1.15–1.8 times faster on random integers from 1000 elements on (1.9–2.3 times at 100 elements and 2.7–3.4 times at 16 against one that allocates its buffer), 2.6–16 times on integers below `n`, 12–35 times with four distinct values and 58–190 times on sorted input; `radix_by_key` is as fast to 1.2 times faster on random `u32` keys of 16-byte elements, 1.45–2.4 times for 72-byte elements and 6–14 times with four distinct keys
- Against the standard library, from 1000 elements on: `radix` is 1.8–5 times faster than `slice::sort_unstable` on random `u32`s, up to 2.3 times on random `u64`s, 3–7.5 times on integers below `n` and 1.1–5 times on nearly sorted input; `radix_by_key` is 1.7–6 times faster than `slice::sort_by_key` on random `u32` keys (16- to 72-byte elements) and up to 2 times on random `u64` keys. On 100 to 256 random `u32`s `radix` is 1.15–1.75 times faster too. Its weaknesses are short slices of wide keys, where every pass costs its counts however few elements it moves (100 random `u64`s sort 1.4 times slower than with `slice::sort_unstable` on fresh input, 2.3 times on the same input over and over; 16 to 32 elements 1.6 (`u32`) to 3.2 (`u64`) times slower; `radix_by_key` on 100 random `u64` keys of 16-byte elements is 1.15 times slower than `slice::sort_by_key`, of 72-byte elements 1.4 times faster), and a few distinct values spread over wide keys, which take every pass while the standard library partitions them away (four distinct random `u32`s take 2–2.9 times as long, `u64`s 2.8–5 times). For keys in a range of about `n` values, [Counting](counting.md) sort is 1.07–1.8 times faster

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Counting

Counting sort, described by Harold Seward in 1954, is the first sort of this library that does not compare elements with each other. It finds the smallest and the largest key, counts how many times each of the `k` keys in between occurs, and turns the counts into the sorted order, in `O(n + k)` time whatever the order of the input: faster than any comparison sort when `k` is not much larger than `n`, as with bytes, small integers or values in a known range. Keys are primitive integers, `char`s and `bool`s (the sealed `IntegerKey` trait). It comes in two forms:

- `counting` sorts the keys themselves. Equal keys cannot be told apart, so the slice is simply rewritten from the counts: so many copies of the smallest key, then so many of the next.
- `counting_by_key` sorts any type by such a key, stably: the counts become the place where each key's elements begin, and every element is moved, in order, to the next place of its key in a buffer, which is then copied back. This is counting sort as radix sort uses it. `key` is called exactly once per element.

A range of keys too large to count panics, leaving the slice as it was: one of more than 65,536 keys and more than four per element. Any range of a one- or two-byte type is allowed. This implementation uses six optimizations:

- Sorted and reverse-sorted input is recognised: `counting` scans for one ascending or one non-increasing run, 16 neighbours at a time with no exit inside a block, which the compiler vectorizes, and `counting_by_key` checks for one ascending or one strictly descending run in the pass that reads the keys; a slice that is one run is left alone or reversed, and nothing is counted.
- Eight count tables or segments for small ranges: incrementing a count waits for the increment before it to reach memory, so a few keys repeating make a chain of dependent increments. `counting` counts into eight tables, one element after the other into the next, with at most 2048 keys and 16 elements per key (3–5 times faster with a few distinct keys and on runs of bytes); `counting_by_key` cuts the slice into eight segments with their own places for every key, after those of the segments before them, with at most 256 keys and 16 elements per key, which keeps it stable (1.6–2.3 times faster with four distinct keys, 2.4–2.9 times on runs of bytes).
- Runs of equal keys do not wait on each other with one table or segment either: counts do not depend on the order of the keys, so one table is counted walking four quarters of the slice side by side (for `counting`, with `u16` counts, or when the first values are mostly in order), and `counting_by_key` places four keys at a time, all four places read before any is written back: 2.2–2.6 times faster on nearly sorted input with sixteen elements per key, 1.4–1.7 times on nearly sorted `counting_by_key` input, and `counting` 1.05–1.3 times faster on random `u8` and `u32` values.
- Keys written ahead of their counts: `counting` writes each key 4 times (8 for one- and two-byte types, in one store) whatever its count, and only a count above that takes a branch; branching on every count mispredicts when there are about as many keys as elements and counts of 0, 1 and 2 are all common (2.5–5 times faster there). A range with 32 or more keys per element skips blocks of 8 empty counts at once instead.
- Keys read once: `counting_by_key` keeps the keys in a buffer as it reads them, and counts and places from it (1.05–1.25 times faster than calling `key` again on random keys); the counts and places can then be trusted without checking, and if `key` panics, no element has moved yet.
- No allocation for small sorts: counts (`u16` up to 65,535 elements, `u32` up to `u32::MAX`), keys and the element buffer each live on the stack up to 4 KiB; the places of `counting_by_key` are summed four at a time, one addition per four waiting for the one before it (1.1–1.25 times faster on random keys over four times as many keys as elements).

## Examples

Rust usage with integers:

```rust
use a::sort::counting;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  counting(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Sorting by a key, stably (equal lengths keep their order):

```rust
use a::sort::counting_by_key;

fn main() {
  let mut words = vec!["pear", "fig", "banana", "kiwi", "apple", "lime"];
  counting_by_key(&mut words, |w| w.len());
  assert_eq!(words, vec!["fig", "pear", "kiwi", "lime", "apple", "banana"]);
}
```

## Efficiency

- Time complexity:
  - Worst, average and best: O(n + k) for `k` possible keys between the smallest and the largest
  - Sorted and reverse-sorted input (strictly descending for `counting_by_key`): O(n), one scan
- Comparisons: none between elements, except the scan for a sorted or reverse-sorted run, which leaves random input after a block or two
- Calls to `key` (`counting_by_key`): exactly n
- Space complexity: O(k) counts for `counting`; O(n + k) for `counting_by_key` (a buffer of n elements, one of n keys and the places); each on the stack up to 4 KiB, otherwise allocated once
- Stability: `counting_by_key` is stable; `counting` cannot tell equal keys apart
- Against the textbook counting sort: `counting` is 3–5.5 times faster on random integers with about as many distinct keys as elements, 1.7–3.1 times with four distinct keys and 9–80 times on sorted, reverse-sorted and all-equal input; `counting_by_key` is 1.05–1.6 times faster on random keys, 1.35–2.2 times with four distinct keys and 6–20 times on sorted and all-equal input, but 1.02–1.15 times slower on reverse-sorted input with repeated keys, random bytes from 10,000 elements on and 1000–10,000 random keys with sixteen times fewer keys than elements, where keeping the keys costs more than it saves (nearly sorted input, which was 1.02–1.15 times slower, now takes 0.6–0.85 times as long as it did)
- Against the standard library, from 100 elements on: `counting` is 3.2–10 times faster than `slice::sort_unstable` on random `u32` and `u64` with about as many distinct keys as elements, 1.9–4 times with four times as many, 2.8–4 times with four distinct keys, 2.2–12 times on nearly sorted input and 9–15 times on random bytes from 1000 elements on; `counting_by_key` is 2.9–7 times faster than `slice::sort_by_key` on random `u32` keys with about as many distinct keys as elements and 1.6–2.3 times with four distinct keys, and as fast on sorted input. On 16 elements both are up to 2 times slower on sorted or nearly sorted input, where the standard library's insertion sort shines

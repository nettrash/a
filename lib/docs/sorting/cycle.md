[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Cycle

Cycle sort, described by B. K. Haddon in 1990, is an unstable, in-place sorting algorithm that writes to the slice as few times as any sort can. The permutation that sorts the slice splits into cycles (the element in slot `a` belongs in slot `b`, the one in `b` in `c`, and so on back to `a`), and each cycle is rotated once: the element of the cycle's first slot is taken out, its final slot is found by counting the elements smaller than it, it is written there and the element it displaces is taken in turn, until the element in hand belongs in the first slot. Equal keys share a block of slots, and an element goes to the first slot of its key's block that does not already hold an equal element. Every element is written at most once, straight into its final slot, and an element whose slot already holds its key is not written at all, which is the point where writes are slow or wear the memory out (flash, EEPROM). The price is the counting: `Θ(n²)` comparisons on any input that is not in order already. This implementation writes the same elements to the same slots as the textbook version (in the same order, except on strictly decreasing input) and uses six optimizations:

- Written cycle starts are passed over: after a cycle, the textbook takes the next slot as the next cycle start and counts for its element even if an earlier cycle wrote it, though such an element is final and the count finds nothing smaller. That is a third of all comparisons on random input (about 0.5n² of 1.5n²). A 4 KiB bitmap on the stack (a window of 32,768 slots) remembers the written slots: 1.23–1.25 times faster on 100 random `u64`s, 1.45–1.5 times from 1000 on and for strings and 64-byte elements, 1.15–1.25 times with four distinct keys from 1000 elements on. On nearly sorted input it gains nothing, and its bookkeeping (about 40 ns a sort) makes 100 `u64`s 1.08 times slower.
- Runs of written starts are passed over at once: a written start is recognised by its bit with a branch, and the whole run of written slots after it is passed over a word at a time (1.03–1.04 times faster on 100 `u64`s than one bit per start; scanning the bits from every start with no branch made each count wait and was up to 1.3 times slower on nearly sorted input).
- Sorted and reverse-sorted input is recognised: a non-decreasing slice is left alone and a strictly decreasing one is reversed, which writes the same elements to the same slots as the cycles would, at n−1 or n comparisons instead of `Θ(n²)` (18–460 times faster from 100 to 3000 elements; usually two comparisons on other input).
- The element in hand: the textbook in safe generic Rust keeps the element being placed in the cycle's first slot and swaps, writing that slot again at every step; here the element is moved out once and the first slot stays vacant until the cycle closes, so each element placed costs exactly one write (a guard puts the element back if a comparison panics). The speed is the same, within 4%.
- Long runs of equal small elements are skipped 16 at a time: for elements of up to 8 bytes, after 32 equal elements one at a time, they are compared 16 at a time in a few vector instructions (1.06 times faster on 1000 `u64`s with four distinct keys, 1.17 times on 3000; at most 1.06 times slower on 200–300 of them).
- Narrow elements are counted in counters as narrow as they are: the counts compare every element with no branch on the answer, so integers are compared several at a time, but into a `usize` the compiler widened the answer of every 1-, 2- and 4-byte element to 64 bits, two elements per vector instruction. Counting them into `u8`, `u16` or `u32` counters, added to the total every 192 elements for `u8` and every 32,768 for the others, compares 16, 8 or 4 at a time: on random input 2.6–3.3 times faster for `u32` from 1000 elements on, 4.1–6.6 times for `u16` and 3.9–5 times for `u8` (1.3–1.65 times on 100 elements), and 2.2–5.2 times with four distinct keys.

## Examples

Rust usage with integers:

```rust
use a::sort::cycle;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  cycle(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Only the elements in the wrong slot are written (here the 0 and the 2):

```rust
use a::sort::cycle;

fn main() {
  let mut v = vec![2, 1, 0, 3];
  cycle(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n^2)
  - Average: O(n^2)
  - Best: O(n), when the input is non-decreasing or strictly decreasing
- Comparisons:
  - Non-decreasing input: n−1; strictly decreasing: n
  - Random distinct keys: about n² (the textbook cycle sort: about 1.5n²); with duplicates, placing an element also compares it with the equal elements already in its key's block
  - Worst: Θ(n²)
- Writes: exactly one per element whose slot does not already hold its key, and none for any other element: the fewest any sort can make (at most n; none for sorted input, 2⌊n/2⌋ for strictly decreasing input)
- Space complexity: O(1): a 4 KiB bitmap on the stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Against the textbook cycle sort: 1.2–1.3 times faster on 100 random `u64`s, 1.45–1.5 times from 1000 on and on random strings and 64-byte elements, 1.15–1.45 times with four distinct keys from 1000 elements on, 11–7000 times on sorted and reverse-sorted input, and as fast on nearly sorted input from 1000 elements on; on up to 200 nearly sorted `u64`s the textbook is 1.08–1.2 times faster, and on 100 `u64`s with four distinct keys 1.04 times. Elements of at most 4 bytes gain another 2.6–6.6 times from 1000 elements on (see above)
- Against [Selection](selection.md) sort (at most n−1 swaps, so up to 2(n−1) writes): 2–2.6 times faster on random `u64`s and 4–5.1 times on nearly sorted ones; but selection sort is 1.1–2.3 times faster on random strings and 64-byte elements, 1.05–1.2 times on nearly sorted ones and 1.6–2.5 times on sawtooth ones, and, as it gathers equal keys in each pass, 2.25–700 times faster with four distinct keys
- Against the standard library: quadratic, so only worth it where a write costs far more than a comparison; `slice::sort_unstable` is 3–6 times faster on 100 random elements, 12–50 times on 1000 and 90–200 times on 10,000 (125 times on 3000 strings; 17–30 times on 5000 random `u32`s, `u16`s or `u8`s), and as fast on sorted and reverse-sorted input

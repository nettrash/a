[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Shell

Shell sort, published by Donald Shell in 1959, is an unstable, in-place insertion sort with a gap. A pass with gap `h` insertion sorts every chain of elements `h` apart, so an element far from its place travels there in steps of `h` instead of one slot at a time; the passes use a decreasing sequence of gaps ending with 1, a plain insertion sort that the earlier passes leave with little to do. How fast it is depends almost entirely on the gaps. This implementation uses five optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input).
- Hole shifting: an element is read out once, the greater elements before it on its chain shift one place along the chain each, and it is written once into the slot left free, one move per step instead of a swap's three.
- The gaps that are fastest, not the fewest comparisons: Ciura's sequence (1, 4, 10, 23, 57, 132, 301, 701, then times 2.25) makes 1.1–1.27 times fewer comparisons than Sedgewick's 1982 one (1, 8, 23, 77, 281, 1073, …), but nearly every insertion ends in a mispredicted branch, and Sedgewick's gaps grow by a factor of about 4, so they take fewer passes: for elements larger than 8 bytes they are 1.05–1.35 times faster, strings included (except random strings from a million elements on, 1.1 times slower). Elements of at most 8 bytes use Ciura's gaps, which suit the window below.
- A branchless window for small elements: in a pass over a slice far from sorted, how far an element moves along its chain is about as hard to predict as a coin toss (with Ciura's gaps 0 steps 44% of the time, 1 step 26%, 2 steps 14%, 3 steps 8%, 4 or more 8%). For elements of at most 8 bytes, each element is compared with the 4 elements before it on its chain at once, how many it passes is computed as a number, and the 4 slots it can move into are rewritten from copies held in registers, each chosen by two selects on that number; only the 8% that pass all 4 go on with a branch per step. On random input this is 1.65–1.9 times faster for integers.
- The window only where it pays, judged by the insertions themselves: where the slice is nearly sorted or its order follows a pattern, insertion's branches are predictable and the window's extra comparisons and writes are wasted (a pass can be up to 5 times slower with it), and how many elements move does not tell the two apart, but how regularly they move does. So a pass is made in blocks of 512 elements, and each block counts, at no measurable cost, how predictable its insertions were and chooses the window for the next block only if they were not: inserted plainly, the runs of consecutive elements that moved and the steps they took beyond the first (`2·runs + 0.35·steps > 0.42` per element); through the window, how often the number of elements passed differs from the one before (more than 0.42 per element). As branch predictors also learn short patterns, the last 32 elements of each block note their steps too, and a block whose steps repeat with a period of at most 16 is followed by plain insertion. Against taking the window whenever the pass before moved more than a quarter of the elements: 1.2–1.85 times faster on sawtooth input of 1000 to a million integers with periods from 4 to n/8 (but a period of √n from 100,000 elements on, as fast to 1.1 times slower), 2.1–2.45 times on organ-pipe input, 1.05–1.15 times on nearly sorted input, 1.05–1.25 times with 4 distinct keys, and as fast on random input (sorted runs of about √n random elements are 1.03–1.25 times slower below 10,000 elements).

## Examples

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

## Efficiency

- Time complexity:
  - Worst: O(n^(4/3)) with Sedgewick's gaps; no bound is known for Ciura's, which were found by experiment
  - Average: no exact bound is known; measured, 1.6–2.8·n·log₂ n comparisons up to a million elements (no gap sequence is known to make Shell sort O(n log n))
  - Best: O(n), when the input is sorted or reverse sorted
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Random input: about (1.6–2)·n·log₂ n with Sedgewick's gaps (≈ 15,900 for n = 1000) and (2.1–2.8)·n·log₂ n with Ciura's gaps and the window (≈ 21,300), whose unnecessary comparisons are cheap for small elements
- Space complexity: O(1) extra (in-place)
- Stability: unstable (an insertion along a chain can carry an element past an equal one between the two)
- Adaptivity: the passes do less on nearly sorted input. Against classic Shell sort with Ciura's gaps: 1.65–1.9 times faster on random integers from 1000 elements on, up to 1.75 times on nearly sorted ones, 1.3–1.4 times with few distinct keys and 1.1–1.2 times on organ-pipe input; on sawtooth input of 1000–100,000 integers with periods from 4 to n/8 from 1.15 times slower (a period of 16 at 1000 elements) to 1.35 times faster, mostly within 1.1 times either way; 1.05–1.4 times faster on strings and 1.03–1.3 times on 64-byte elements. Against [Comb](comb.md) sort: 3.55–4 times slower on random integers from 1000 elements on and 1.8–2.65 times on nearly sorted ones or with few distinct keys (1.15 times faster on 100 nearly sorted ones), but 1.1–2.5 times faster on strings that have few distinct keys or follow a pattern, and 1.6 times on 1000 nearly sorted strings (as fast on longer ones). Against [Heap](heap.md) sort: 1.45–1.8 times slower on random integers from 1000 elements on, 1.1–1.6 times on nearly sorted ones and 1.1–1.25 times with few distinct keys, about as fast on random strings, and faster on strings that are nearly sorted or patterned

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Bitonic

Bitonic sort, published by Kenneth Batcher in 1968, is a sorting network: a fixed sequence of compare-exchanges, each putting the elements at two given positions in order, chosen in advance and never depending on what the comparisons answer. A sequence that first rises and then falls is bitonic; one stage of compare-exchanges, of each element of the first half with the element half a slice further on, splits it into two bitonic halves with nothing in the first greater than anything in the second, and repeating this on the halves merges it. Bitonic sort sorts the two halves in opposite directions, which makes the whole slice bitonic, and merges it: `n·k·(k + 1) / 4` compare-exchanges for `n = 2^k`, in `k·(k + 1) / 2` stages whose compare-exchanges are independent of each other, which is why it suits hardware, vector units, GPUs and threads. Batcher's network is for powers of two; for other lengths this is H. W. Lang's formulation (the first `⌊n/2⌋` elements sorted descending, the rest ascending, and the merge of `n` elements comparing `i` with `i + m` for `m` the largest power of two below `n`), which is the network of the next power of two without the compare-exchanges that would involve missing elements, so nothing is padded and any `T` can be sorted. It is unstable and in place, and not adaptive: every input of a length gets the same compare-exchanges. It comes in two forms: `bitonic` on the calling thread, and `bitonic_parallel`, which makes exactly the same compare-exchanges on several threads. This implementation uses five optimizations, all of which change only how the network is carried out, never which compare-exchanges it makes:

- Small networks in straight-line code: sorts of at most 16 elements and merges of at most 16 run their network with every compare-exchange written out at constant positions, so small elements stay in registers (2.2–4.2 times faster for integers than recursing down to single elements, 1.6–2 times for strings, 1.35–1.6 times for 64-byte elements; 1.9–2.4, 1.1–1.15 and 1.35–1.55 times faster than a loop over the same networks).
- Two or three stages per pass: a merge of a power-of-two length runs its first two or three stages as one pass over groups of 4 or 8 elements, each read once, put through the 4 or 12 compare-exchanges of those stages and written once, then merges its parts depth first, so the later stages run in cache (three stages make integers 1.04–1.14 times faster, two make 64-byte elements 1.03–1.08 times faster; strings, whose comparisons cost more than the passes, take one).
- Compare-exchanges without branches: elements of at most 8 bytes are exchanged by selecting the values to write back, which lets the compiler run a pass's independent compare-exchanges in vector registers (4.6–5 times faster than branching). Within a merge, whether a compare-exchange swaps comes in long runs, but in the small networks it is a coin toss on random input, so elements of 9 to 24 bytes, such as `String`, are exchanged there by selecting the addresses to copy from and by a branch in the passes (1.25–1.45 times faster for random strings, 1.05–1.1 times for nearly sorted ones; `u128` and pairs gain 1.4–1.9 times on random input but lose up to 1.8 times on nearly sorted input, and arrays, whose comparison is itself a chain of branches, lose up to 1.4 times). Larger elements branch.
- The direction is a type: each part is sorted ascending or descending through a type parameter rather than a flag, so every compare-exchange is compiled for its direction.
- Narrow elements are merged stage by stage: for elements of at most 4 bytes, a merge of a power-of-two length of at most 2048 runs its stages one after the other, each as one loop over the whole merge, which stays in the first-level cache, so that every stage, down to the one pairing neighbours, runs as vector minimums and maximums of 4 to 16 elements at once instead of the one pair at a time of the straight-line merges of 16 (1.25 times faster for `u32`, `u16` and `u8` at 1000 elements, 1.45–1.6 times at 16,384 and 1.3–1.55 times at a million, on every input; for `u64`, two to a vector, it was 1.1–1.25 times slower).

`bitonic_parallel` uses one thread per 16,384 elements of at most 8 bytes, or per 4,096 larger elements, up to `std::thread::available_parallelism`; shorter slices are sorted by `bitonic`, with no allocation. It cuts the slice into parts along the sort's own halving, two or four per thread, sorts them in a first round, and then takes two rounds per level of merges: the first runs the stages that compare elements of different parts, one range of positions at a time across all parts, the second merges each part on its own. The threads are spawned once and kept for all rounds (spawning them for each round instead was 1.15–1.6 times slower from 32,768 to 131,072 elements, and 1.2 times for a million integers), take the tasks of a round, which touch disjoint elements, one at a time until none is left, and meet at a barrier between rounds. A panic on any thread lets the others finish their current tasks, skips the rest and is propagated to the caller, with every element still in the slice exactly once.

## Examples

Rust usage with integers:

```rust
use a::sort::bitonic;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  bitonic(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Rust usage with strings:

```rust
use a::sort::bitonic;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  bitonic(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

On all cores (the same compare-exchanges, so the same result, equal elements included):

```rust
use a::sort::bitonic_parallel;

fn main() {
  let mut v: Vec<u64> = (0..1_000_000).map(|i| i * 7919 % 1_000_003).collect();
  bitonic_parallel(&mut v);
  assert!(v.windows(2).all(|w| w[0] <= w[1]));
}
```

## Efficiency

- Time complexity:
  - Worst, average and best: O(n log² n), the same on every input
  - Parallel: `k·(k + 1) / 2` stages of independent compare-exchanges for `2^(k−1) < n ≤ 2^k`, O(log² n) with enough processors
- Comparisons: exactly the network's, on every input: `n·k·(k + 1) / 4` for `n = 2^k`, between `n·(k − 1)² / 4` and `n·k·(k + 1) / 4` for `2^(k−1) < n < 2^k` (26,984 for n = 1000 against about 10,500 for [Heap](heap.md) sort, 14,600 for elements of at most 8 bytes; 102,931,040 for n = 1,000,000, about 5.2·n·log₂ n)
- Space complexity: O(log n) stack, no allocation (in-place); `bitonic_parallel` allocates its list of tasks and its threads
- Stability: unstable (equal elements may be reordered)
- Adaptivity: none; sorted input costs as much as random input
- Against the textbook bitonic sort (Lang's recursion, with the direction as a flag and a branch and a swap for each compare-exchange): 10.5–11.8 times faster on random integers and 4.8–9.8 times on other input, 1.75–2.1 times on random strings and 1.3–1.6 times on other strings, 1.4–1.8 times for random 64-byte elements and 1.05–1.5 times on other input
- Against the standard library: as fast as `slice::sort_unstable` on random integers around 1000 elements (and 1.25 times faster than [Quick](quick.md) sort), but its `log₂ n` factor shows from there on: `sort_unstable` is 1.35 times faster at 16,384 integers and 1.85 times at a million, 20–70 times faster on sorted input (random `u32`s and `u16`s, merged stage by stage, are sorted 1.1–1.3 times faster than by `sort_unstable` from 1000 to 16,384 elements and 1.25–1.45 times slower at a million), 2.5–5 times faster on random strings and 2.3–4.1 times for random 64-byte elements. Against [Shell](shell.md) sort: 2.25–3.6 times faster on random integers from 1000 elements to a million, as fast on random strings, 1.35–1.6 times slower for 64-byte elements
- `bitonic_parallel` on the 16 threads of an Apple M4 Max (12 performance and 4 efficiency cores), against `bitonic`: integers 1.6 times faster at 32,768 elements, 2.5 times at 65,536, 2.9–3.2 times up to 262,144, 7.2 times at a million and 8.8 times at four million; strings 1.8 times faster at 8,192 elements, 3.3 times at 16,384, 4.4–5.7 times at 32,768 and 10–11 times from 262,144 on; 64-byte elements 1.7 times faster at 8,192 elements, 2.9 times at 16,384 and 8–9.7 times from 262,144 on (between 262,144 and a million integers, 8 to 12 threads would be up to 1.25 times faster than 16, as each round waits for its slowest thread). Against the single-threaded `slice::sort_unstable` it is faster from 32,768 elements on: for integers 1.2 times there and 3.9 times at a million, for strings 1.15 and 2.2 times, for 64-byte elements 1.05 and 2.4 times. The stage-by-stage merges make `bitonic_parallel` 1.05–1.25 times faster for `u32` and 1.15–1.35 times for `u8` from 65,536 to a million elements

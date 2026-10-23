[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Heap

Heap sort is an unstable, in-place, `O(n log n)` sorting algorithm with constant stack. It arranges the slice into a max-heap, a tree stored in the slice itself in which no element is smaller than its children, so the largest element is at the root; it then repeatedly moves the root to the end of the heap, shrinks the heap by one and restores the heap order. Building the heap takes O(n) comparisons and each of the n removals O(log n), whatever the input. This implementation uses five optimizations:

- Sorted and reverse-sorted input is recognised: the slice is first scanned for one ascending or strictly descending run; if it covers the whole slice, the slice is left alone or reversed, at n−1 comparisons (usually two or three on other input). Heap sort itself would spend n·log₂ n comparisons on sorted input.
- Bottom-up removal: the textbook sift-down compares the element moved to the root with the larger child at every level, but that element comes from the end of the heap and usually sinks almost to the bottom anyway. Instead, the vacant root moves down to a leaf along the larger children, one comparison per level, and the element moves up from there, usually by a level or two (Wegener's bottom-up heap sort): about n·log₂ n comparisons instead of 1.7–1.85·n·log₂ n.
- Holes instead of swaps: the element being placed is held outside the slice, and each level moves one element into the vacant slot instead of swapping two, one move instead of three.
- Four children per node for large heaps and small elements: a heap with four children per node has half as many levels, at three comparisons per level instead of one (about 1.5·n·log₂ n in all). That pays from 64 KiB of elements on, where every level of the heap costs a cache miss: on random input 1.15–1.35 times faster for integers from 100,000 elements on, 1.1–1.4 times for strings and 1.25–2 times for 64-byte elements; on nearly sorted strings 1.05–1.15 times slower. It also pays for elements of at most 8 bytes at any length, whose comparisons are so cheap that a level costs about the time to load the children, compare them and find the next ones: 1.15–1.25 times faster for integers in cache (`&str`, as small as a `u128` but slower to compare, would be 1.05–1.15 times slower, so 16-byte elements keep two children below 64 KiB).
- Short chains from one level to the next: each level of a removal waits for the comparison before it, which says where the next children are. Every child is loaded straight from the first child's index, and for elements of up to 16 bytes the larger child and its own first child are chosen by selects between values computed beforehand; with four children the winners of the first round are selected as pointers, so the values already loaded are compared instead of being loaded again. This alone makes integers sort 1.25–1.3 times faster on four children per node. Larger elements, which compare by chains of branches that would turn the selects back into branches, keep arithmetic.

Together, against the textbook heap sort, this is 1.1–1.25 times faster for integers up to 1000 elements and 1.4–1.85 times from 10,000 on, 1.05–1.4 times for strings from 1000 elements on and 1.3–2.2 times for 256-byte elements. Only sawtooth input of 1000–2000 integers, whose short periods make three comparisons per level cost more than the levels saved, would sort 1.02–1.05 times faster on two children per node.

## Examples

Rust usage with integers:

```rust
use a::sort::heap;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  heap(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Rust usage with strings:

```rust
use a::sort::heap;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  heap(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

## Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when the input is sorted or reverse sorted
- Comparisons:
  - Best (sorted, reverse sorted): n−1
  - Random input: about n·log₂ n + 0.6n (≈ 10,500 for n = 1000, against ≈ 16,800 for the textbook heap sort); about 1.5·n·log₂ n with four children per node, from 64 KiB of elements on and for elements of at most 8 bytes (≈ 14,600 for n = 1000)
  - Worst: O(n log n), on every input
- Moves: about n·log₂ n, and about 0.5·n·log₂ n with four children per node
- Space complexity: O(1) stack, no allocation (in-place)
- Stability: unstable (equal elements may be reordered)
- Adaptivity: not adaptive, except to sorted and reverse-sorted input. Heap sort jumps around the slice, so `slice::sort_unstable` and this crate's [Quick](quick.md) sort, which falls back to heap sort only when its pivots keep failing, are faster: 2–3.4 times for random integers from 64 elements on, 1.6–6 times for strings and 1.1–3.5 times for 256-byte elements. Heap sort's strength is its guarantee: O(n log n) on every input, in place, with constant stack

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Stooge

Stooge sort is a deliberately slow, unstable, in-place sorting algorithm, mostly of educational value: an example of a short, correct algorithm with a surprising running time. It compares the first and the last element and swaps them if they are out of order; then, if the slice has at least three elements, it stooge-sorts the first two thirds (rounded up), the last two thirds, and the first two thirds again. The first call moves every element of the largest third that starts in the first two thirds into the middle, so after the second call all of them sit at the end, in order, and the third call sorts the rest. It is Problem 7-3 of Cormen, Leiserson, Rivest and Stein's *Introduction to Algorithms*, where Professors Howard, Fine and Howard (the Three Stooges) propose it as an "elegant" sorting algorithm, and the reader shows that it takes Θ(n^(log 3 / log 1.5)) ≈ Θ(n^2.71) time, worse than bubble sort. Which elements are compared depends only on the length of the slice, never on the answers: stooge sort is a sorting network, one fixed sequence of compare-exchanges for each length. This implementation makes exactly the textbook's comparisons, in the same order, with the same swaps, and uses three optimizations in executing them:

- Unrolled bottom levels: all parts at one depth of the recursion have the same length, so the parts four levels above the bottom have 7, 8 or 9 elements throughout one sort; they run as straight-line code, the 121 compare-exchanges below them at constant indices, with no calls and no index checks. The textbook makes one call per compare-exchange, and the calls are its cost: recursing down to two elements without index checks is only 1.05–1.15 times faster than it, while unrolling four levels is 5–8 times faster for `u64`, 1.4–1.65 times for strings and 1.4–2.3 times for 64-byte elements than recursing down to two elements.
- A branch, not a branchless select: only about 1% of the compare-exchanges swap anything (4.5% at 16 random elements, 0.2% with few distinct keys), so the branch is predicted almost always and the compare-exchanges do not wait for one another; a branchless select, which chains each to the one before, was 2 times slower than the textbook for `u64`, 4 times for strings and 1.8 times for 64-byte elements.
- The swap is marked cold, so the compare-exchanges that do not swap are laid out as one straight line: 1.5–1.85 times faster for 64-byte elements, 1.05–1.25 times for strings and 1.1–1.25 times for `u64` from 100 elements on.

The code is safe Rust, with no `unsafe` block.

## Examples

Rust usage with integers:

```rust
use a::sort::stooge;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0];
  stooge(&mut v);
  assert_eq!(v, vec![0, 1, 2, 4, 5, 8]);
}
```

Sorting strings:

```rust
use a::sort::stooge;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  stooge(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

## Efficiency

- Time complexity: Θ(n^(log 3 / log 1.5)) ≈ Θ(n^2.71) on every input (worst, average and best alike)
- Comparisons: exactly C(n) = (3^(D+1) − 1)/2 on every input of n ≥ 2 elements, where D is the number of times m ↦ m − ⌊m/3⌋ takes n to 2 (equivalently C(2) = 1, C(n) = 1 + 3·C(n − ⌊n/3⌋)): 13 for n = 4, 121 for n = 8, 1,093 for n = 16, 265,720 for n = 100 (and for 128: D grows in steps), 64,570,081 for n = 1000, against at most n(n−1)/2 = 499,500 for bubble sort
- Swaps: at most C(n), depending on the input; about 1% of the comparisons on random input; none on already-sorted input
- Space complexity: O(log n) stack, no allocation (in-place); the recursion nests D − 4 levels for n ≥ 10 (12 for n = 1000)
- Stability: unstable (`[1a, 1b, 0]` becomes `[0, 1b, 1a]`)
- Adaptivity: none; the same comparisons on every input of a given length
- Against the textbook stooge sort (recursive, one call per compare-exchange): 9–11 times faster for `u64` from 64 elements on (3–10 times below), 1.7–2.2 times for strings, and 4.3–5 times for 64-byte elements from 32 on (2.5–4.5 times below; 1.4–2 times with few distinct keys, where every comparison reads all 64 bytes)
- Against the standard library: `slice::sort_unstable` is 1.2–4 times faster on 8 elements, 50–130 times on 100 random elements, 900–2400 times on 1000 and about 4000 times on 2000 random `u64`; on sorted input, which std recognises in one pass, 10,000–70,000 times from 500 elements on

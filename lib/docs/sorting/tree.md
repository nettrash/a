[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Tree

Tree sort is a stable, `O(n log n)` sorting algorithm that inserts the elements one by one into a binary search tree, in which the left subtree of every node holds the elements that go before it and the right subtree those that go after, and then reads the tree out in order (an in-order traversal) back into the slice. Binary search trees, and sorting by building one, go back to around 1960 (Windley; Booth and Colin; Hibbard). A plain binary search tree is only as good as the order the elements come in: sorted input builds a list and costs n²/2 comparisons. This implementation keeps the tree an AVL tree (Adelson-Velsky and Landis, 1962), the first self-balancing binary search tree: the two subtrees of every node differ in height by at most one, restored after each insertion by at most one single or double rotation, so the tree stays below 1.44·log₂ n levels and every insertion costs O(log n) whatever the input. An element equal to one already in the tree goes to its right, after it, which makes the sort stable. AVL trees were measured against red-black trees, whose looser balance makes them up to twice as high: within 10% of each other on random input, but red-black trees were 1.15–2.4 times slower on sorted, nearly sorted and sawtooth input and on input with few distinct keys. This implementation uses seven optimizations:

- One array of nodes instead of an allocation per node: node `i` belongs to the element that started at index `i`, so the nodes form one array with 32-bit indices as links and the balance factors in a separate byte array that only rebalancing reads; searches and the in-order traversal are loops with an explicit stack. Elements are moved twice and only twice, whatever the input. Against the textbook AVL tree sort, which allocates a node per element and inserts recursively, this alone (the crate's first version of this sort) is 1.2–4 times faster on random input and up to 30 times on sorted strings.
- Small elements live in the nodes: elements of 5 to 8 and of 17 to 24 bytes are copied into nodes of 16 and 32 bytes, so a search step loads a node and its key from one place; other elements stay in the slice, where they are compared, and are gathered into a buffer in order at the end (their nodes would waste a quarter or more on padding, and elements of up to 4 bytes or of 16 bytes are addressed by a shift in the slice too). This is 1.03–1.07 times faster on random input of 10,000 elements and more.
- Branchless search steps: a step loads both links of a node with its key and the comparison selects one, instead of a branch that random input mispredicts half the time or a load that waits for the comparison: 1.02–1.11 times faster up to 10,000 elements.
- Searches side by side: a search is a chain of dependent loads and comparisons that the processor cannot overlap with itself, so the places of the next 2 elements (4 from 1024 elements on, 8 from 262,144 on) are searched for at once in the same tree, and the elements are then inserted one after the other, in order. An insertion can only invalidate a later search through the slot it takes or a node it rotates; such a search goes on from the deepest node of its path still in place, usually a level or two above its slot, so the tree is exactly the one that one-by-one insertion builds, at up to 2% more comparisons with few distinct keys. On random input this is 1.1–3.7 times faster than one search at a time (most for strings and large slices), and 1.03–1.6 times with 16 distinct keys, except for 64-byte elements at 1,000,000 (1.08 times slower). Searching again from the root instead of going on is 1.05–1.2 times slower with few distinct keys.
- Insertion beside the last insertion: if the last element inserted went right after or right before the one inserted before it, the next one is first tried right after or right before it in turn. Two comparisons, with the last element and its neighbour in order, say whether it belongs there; the path to that slot is the last element's, kept up to date through the rotations, so no search is needed. Sorted, reverse-sorted and all-equal input costs n−1 comparisons, and an ascending or descending run costs two per element after its first, wherever it lands. This is 1.9–8 times faster on sorted and reverse-sorted input and 1.6–3.6 times on nearly sorted input, at up to 5% on random input.
- Short shifts of the kept path inline: a rotation removes a node from the path to the last insertion, usually a level or two above it; moving the one or two entries below it inline instead of calling `memmove` is 1.1–1.3 times faster on sorted and reverse-sorted input.
- No allocation for small slices: up to 4 KiB of nodes (240 nodes of 8-byte elements) live on the stack, which is 2.1–3.4 times faster for 2 elements, 1.1–1.4 times for 10 and up to 1.27 times for 30; beyond, nodes, balance factors and the gather buffer take one allocation.

Together, against the textbook tree sort, which builds an unbalanced binary search tree with a node allocated per element: 1.5–4.4 times faster on random input of 100 to 1,000,000 elements, and 1,600–2,500 times on 10,000 sorted elements, where the textbook sort makes n²/2 comparisons.

## Examples

Rust usage with integers:

```rust
use a::sort::tree;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  tree(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}
```

Rust usage with strings:

```rust
use a::sort::tree;

fn main() {
  let mut words = vec!["pear", "apple", "fig"];
  tree(&mut words);
  assert_eq!(words, vec!["apple", "fig", "pear"]);
}
```

Equal keys keep their original order (cards are compared by rank only):

```rust
use a::sort::tree;
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
  tree(&mut cards);
  let order: String = cards.iter().map(|c| c.1).collect();
  assert_eq!(order, "dgjbeihacf");
}
```

## Efficiency

- Time complexity:
  - Worst: O(n log n)
  - Average: O(n log n)
  - Best: O(n), when every element goes right after or right before the one before it (sorted, reverse-sorted or all-equal input)
- Comparisons:
  - Best (sorted, reverse sorted, all equal): n−1
  - Runs: at most 2 per element, plus a search for the first element of each run
  - Random input: about n·log₂ n − 1.2n (≈ 8,730 for n = 1000, against ≈ 10,000–11,000 for the textbook tree sort, which makes 499,500 on sorted input)
  - Worst: O(n log n), since a search takes fewer than 1.44·log₂ n steps, the height of an AVL tree, on any input
- Moves: exactly 2n, whatever the input (into the nodes and back, or gathered into a buffer and copied back)
- Space complexity: O(n) extra: a node of 8 bytes per element (16 or 32 bytes with the element inside) and a balance factor, on the stack up to 4 KiB and allocated once beyond, plus a buffer of n elements for elements that stay in the slice; O(log n) stack
- Stability: stable (equal elements keep their relative order)
- Adaptivity: adaptive to runs, which cost at most two comparisons per element. A binary search tree is a poor way to sort what fits in an array, though: every search step in a large tree is a cache miss and every insertion pays for rebalancing. `slice::sort` is 1.5–5 times faster on random input, 2–17 times with few distinct keys and 6–30 times on sorted input; this crate's [Merge](merge.md) and [Tim](tim.md) sorts are 1.25–4.5 times faster on random integers and 64-byte elements, and on random strings Merge sort is 1.3–1.8 times faster and Tim sort 1.05–1.4 times. Tree sort moves every element exactly twice, against n·log₂ n times for a merge sort, and is up to 1.9 times faster than `slice::sort` on nearly sorted strings and 1.4–1.5 times on nearly sorted 64-byte elements

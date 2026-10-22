use core::alloc::Layout;
use core::hint::select_unpredictable;
use core::mem::{self, MaybeUninit};
use core::ptr;

/// Sorts a slice using tree sort.
///
/// Tree sort inserts the elements one by one into a binary search tree, in which the left
/// subtree of every node holds the elements that go before it and the right subtree those that
/// go after, and then reads the tree out in order (an in-order traversal) back into the slice.
/// Binary search trees, and sorting by building one, go back to around 1960 (Windley; Booth and
/// Colin; Hibbard). A plain binary search tree is only as good as the order the elements come
/// in: sorted input builds a list, `n²/2` comparisons. This implementation keeps the tree an
/// AVL tree (Adelson-Velsky and Landis, 1962), the first self-balancing one: the two subtrees
/// of every node differ in height by at most one, restored after each insertion by at most one
/// single or double rotation, so the tree is less than `1.44·log₂ n` levels high and every
/// insertion costs `O(log n)` whatever the input. An element equal to one in the tree goes to
/// its right, after it, which makes the sort stable.
///
/// AVL trees were measured against red-black trees, whose rebalancing is looser, both without
/// the refinements below: on random input they are within 5–10% of each other, but red-black
/// trees grow up to twice as high on ordered input, which made them 1.15–2.4 times slower on
/// sorted, nearly sorted and sawtooth input and on input with few distinct keys
/// (1.45–1.7·n·log₂ n comparisons on sorted input, against 0.9–0.95·n·log₂ n).
///
/// Seven refinements keep the algorithm as cheap as tree sort can be:
///
/// * **One array of nodes, no allocation per node.** The textbook allocates a node per element
///   and links nodes by pointers. Here node `i` belongs to the element that started at index
///   `i`, so the nodes form one array with 32-bit indices as links, and the balance factors a
///   separate byte array that only rebalancing reads. A search is a loop, and so is the
///   in-order traversal, with an explicit stack (below `2^32` elements the tree is never more
///   than 45 levels high).
///   Elements are moved twice and only twice, whatever the input. Against the textbook AVL
///   tree sort, which allocates a node per element and inserts recursively, this alone (this
///   crate's first version of the sort) is 1.2–4 times faster on random input and up to 30
///   times on sorted strings.
/// * **Small elements live in the nodes.** Elements of 5 to 8 and of 17 to 24 bytes are
///   copied into nodes of 16 and 32 bytes, so a search step loads a node and its key from one
///   place, with an index scaled by a shift. Other elements stay in the slice, where they are
///   compared, and are gathered into a buffer in order at the end: their nodes would waste a
///   quarter or more on padding, and elements of up to 4 bytes or of 16 bytes are addressed by
///   a shift in the slice too. This is 1.03–1.07 times faster on random input of 10,000
///   elements and more, and within 5% either way with few distinct keys.
/// * **Branchless search steps.** A search step loads both links of the node with its key, and
///   the comparison selects one of them, instead of a branch, which random input mispredicts
///   half the time, or a load that waits for the comparison: 1.02–1.11 times faster up to
///   10,000 elements. With eight searches side by side (below) the two are within 10% of each
///   other, either way depending on the element type.
/// * **Searches side by side.** A search is a chain of dependent loads and comparisons, which
///   the processor cannot overlap with itself. So the places of the next 2 elements (4 from
///   1024 elements on, 8 from 262,144 on) are searched for at once, in the same tree, and
///   the elements are then inserted one after the other, in order. An insertion can only
///   invalidate a later search through the slot it takes or a node it rotates; such a search
///   goes on from the deepest node of its path still in place, usually a level or two above
///   its slot (searching again from the root is 1.05–1.2 times slower with few distinct
///   keys). The tree is exactly the one that one-by-one insertion builds, at up to 2% more
///   comparisons with few distinct keys. On random input this is 1.1–3.7 times faster than
///   one search at a time, most for strings and large slices, and 1.03–1.6 times with 16
///   distinct keys, except for 64-byte elements at 1,000,000 (1.08 times slower).
/// * **Insertion beside the last insertion.** If the last element inserted went right after
///   or right before the one inserted before it, the next one is first tried right after or
///   right before it in turn. Two comparisons, with the last element and its neighbour in
///   order, say whether it belongs there, and the path to that slot is the last element's,
///   kept up to date through the rotations, so no search is needed. Sorted, reverse-sorted and
///   all-equal input then costs `n − 1` comparisons, and an ascending or descending run two
///   per element after its first, wherever it lands in the tree. This is 1.9–8 times faster
///   on sorted and reverse-sorted input and 1.6–3.6 times on nearly sorted input, at up to 5%
///   on random input.
/// * **Short shifts inline.** A rotation removes a node from the kept path, usually a level or
///   two above its end: moving the one or two entries below it inline, instead of calling
///   `memmove`, is 1.1–1.3 times faster on sorted and reverse-sorted input.
/// * **No allocation for small slices.** Up to 4 KiB of nodes (240 nodes of 8-byte elements)
///   live on the stack: 2.1–3.4 times faster for 2 elements, 1.1–1.4 times for 10 and up to
///   1.27 times for 30. Beyond, the nodes, balance factors and the gather buffer take one
///   allocation.
///
/// Together, against the textbook tree sort, which builds an unbalanced binary search tree
/// with a node allocated per element: 1.5–4.4 times faster on random input of 100 to
/// 1,000,000 elements, and 1,600–2,500 times on 10,000 sorted elements, where the textbook
/// sort makes `n²/2` comparisons.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order. An element goes left of a node
///   only if it is strictly smaller, and equal elements are inserted in their original order.
/// * **Not in place**: `n` nodes of 8 bytes, 16 or 32 bytes with the element inside, and a
///   byte each, on the stack up to 4 KiB and allocated once beyond; plus a buffer of `n`
///   elements for elements kept in the slice. `O(log n)` stack.
/// * **Adaptive** to runs: an element that goes right after or right before the one inserted
///   before it costs at most two comparisons and no search.
///
/// | Case                              | Comparisons                    | Moves |
/// |-----------------------------------|--------------------------------|-------|
/// | Best (sorted, reverse, all equal) | `n − 1`                        | `2n`  |
/// | Runs                              | `≤ 2n` plus a search per run   | `2n`  |
/// | Average (random)                  | `≈ n·log₂ n − 1.2n`            | `2n`  |
/// | Worst                             | `O(n log n)`                   | `2n`  |
///
/// A search takes fewer than `1.44·log₂ n` steps, the height of the tree, on any input; one
/// overtaken by an earlier insertion goes on for a few more.
/// For `n = 1000`: 999 comparisons on sorted or reverse-sorted input and about 8,730 on random
/// input, against about 10,000–11,000 for the textbook tree sort on random input and 499,500 on
/// sorted input.
///
/// A binary search tree is a poor way to sort what fits in an array: every search step in a
/// large tree is a cache miss, and every insertion pays for rebalancing. [`slice::sort`] is
/// 1.5–5 times faster on random input, 2–17 times with few distinct keys and 6–30 times on
/// sorted input; this crate's [`merge`](fn@super::merge) and [`tim`](fn@super::tim) sorts are
/// 1.3–3.7 times faster on random integers and 64-byte elements (on random strings merge sort
/// is up to 1.1 times slower and Tim sort 1.1–1.4 times faster). Tree sort moves every element
/// exactly twice, though, against `n·log₂ n` times for a merge sort, and it is up to 1.9 times
/// faster than [`slice::sort`] on nearly sorted strings and 1.4–1.5 times on nearly sorted
/// 64-byte elements.
///
/// # Panics
///
/// Panics only if allocating the tree fails. If the [`Ord`] implementation of `T` panics, the
/// panic is propagated and `arr` is left in its original order: elements are only moved once
/// every comparison has been made. An [`Ord`] that is not a total order leaves `arr` in an
/// unspecified order with every element exactly once.
///
/// # Examples
///
/// ```
/// use a::sort::tree;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// tree(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// tree(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
///
/// // Equal keys keep their order: cards are compared by rank only.
/// #[derive(Debug, PartialEq, Eq)]
/// struct Card(u8, char);
/// impl Ord for Card {
///     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
///         self.0.cmp(&other.0)
///     }
/// }
/// impl PartialOrd for Card {
///     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
///         Some(self.cmp(other))
///     }
/// }
/// let mut cards: Vec<Card> = "abcdefghij".chars().zip([3, 1, 3, 0, 1, 3, 0, 2, 1, 0])
///     .map(|(c, rank)| Card(rank, c))
///     .collect();
/// tree(&mut cards);
/// let order: String = cards.iter().map(|c| c.1).collect();
/// assert_eq!(order, "dgjbeihacf");
/// ```
pub fn tree<T: Ord>(arr: &mut [T]) {
    // Zero-sized elements are all alike: any order is sorted.
    if arr.len() < 2 || mem::size_of::<T>() == 0 {
        return;
    }
    // Elements of 5 to 8 and of 17 to 24 bytes fill nodes of 16 and 32 bytes with little
    // padding; the room in a node is made of `u64`s, so it suits alignments up to 8.
    let (size, align) = (mem::size_of::<T>(), mem::align_of::<T>());
    if size > 4 && size <= 8 && align <= 8 {
        lanes::<T, [MaybeUninit<u64>; 1]>(arr);
    } else if size > 16 && size <= 24 && align <= 8 {
        lanes::<T, [MaybeUninit<u64>; 3]>(arr);
    } else {
        lanes::<T, ()>(arr);
    }
}

/// Sorts `arr` with nodes of `V` (an element's room, or nothing if elements stay in the
/// slice), searching for as many places at once as pays off for its length.
fn lanes<T: Ord, V>(arr: &mut [T]) {
    let len = arr.len();
    if len >= u32::MAX as usize {
        // Too many nodes for 32-bit links (`u32::MAX` itself means no node).
        sort::<T, usize, V, 8>(arr);
    } else if len >= LANES_8_MIN {
        sort::<T, u32, V, 8>(arr);
    } else if len >= LANES_4_MIN {
        sort::<T, u32, V, 4>(arr);
    } else {
        sort::<T, u32, V, 2>(arr);
    }
}

/// From this many elements on, four searches run side by side instead of two.
///
/// Measured: at 1000 elements four are 1.06–1.24 times faster on random input, 1.1 times for
/// integers with 16 distinct keys, and up to 1.06 times slower for 64-byte elements with 16
/// distinct keys; at 100 elements two are up to 1.09 times faster.
const LANES_4_MIN: usize = 1024;

/// From this many elements on, eight searches run side by side instead of four.
///
/// Measured on random input: at 1,000,000 elements, where most search steps miss the cache,
/// eight are 1.08–1.27 times faster than four; at 100,000 they are 1.03–1.14 times faster for
/// strings and 64-byte elements and 1.03 times slower for integers. With 16 distinct keys
/// they are 1.1–1.2 times slower, since more searches go to the same place.
const LANES_8_MIN: usize = 1 << 18;

/// Room for the longest path from the root of an AVL tree of fewer than `2^64` nodes, which is
/// less than `1.4405·log₂(n + 2)` levels high: at most 91 (45 below `2^32` nodes).
const HEIGHT: usize = 96;

/// No node, for the in-order neighbours of a node at an end of the tree.
const NONE: usize = usize::MAX;

/// A link to a child: a node index, or `NIL` for none.
trait Link: Copy + Eq {
    const NIL: Self;
    fn new(i: usize) -> Self;
    fn get(self) -> usize;
    /// Returns the right link of the pair at `p` if `right`, else the left one.
    ///
    /// Both are loaded before the choice, which then takes one instruction (a conditional
    /// select) after the comparison that decides it, instead of a load or a branch.
    ///
    /// # Safety
    ///
    /// `p` must be valid for reads.
    unsafe fn pick(p: *const [Self; 2], right: bool) -> Self;
}

impl Link for u32 {
    const NIL: Self = u32::MAX;
    #[inline(always)]
    fn new(i: usize) -> Self {
        i as u32
    }
    #[inline(always)]
    fn get(self) -> usize {
        self as usize
    }
    #[inline(always)]
    unsafe fn pick(p: *const [Self; 2], right: bool) -> Self {
        // One 8-byte load: two separate loads let LLVM select the address and load once after
        // the comparison, which puts the load back on the chain from one step to the next.
        // SAFETY: `p` is valid for reads of 8 bytes, as the caller guarantees.
        let pair = unsafe { p.cast::<u64>().read_unaligned() };
        let (l, r) = if cfg!(target_endian = "little") {
            (pair as u32, (pair >> 32) as u32)
        } else {
            ((pair >> 32) as u32, pair as u32)
        };
        select_unpredictable(right, r, l)
    }
}

impl Link for usize {
    const NIL: Self = usize::MAX;
    #[inline(always)]
    fn new(i: usize) -> Self {
        i
    }
    #[inline(always)]
    fn get(self) -> usize {
        self
    }
    #[inline(always)]
    unsafe fn pick(p: *const [Self; 2], right: bool) -> Self {
        // SAFETY: `p` is valid for reads, as the caller guarantees.
        let [l, r] = unsafe { *p };
        select_unpredictable(right, r, l)
    }
}

/// A node: the links to its left and right children, and room `V` for its element, or
/// nothing if the element stays in the slice.
#[repr(C)]
struct Node<L, V> {
    link: [L; 2],
    val: V,
}

/// The path from the root to a node or slot: each entry is a node and the side taken from it,
/// `node << 1 | right`.
type Path = [MaybeUninit<usize>; HEIGHT];

/// An AVL tree over the elements of a slice.
struct Tree<T, L, V> {
    /// Node `i` is the node of the element that started at index `i`.
    nodes: *mut Node<L, V>,
    /// The balance factor of node `i`: the height of its right subtree minus that of its
    /// left one, −1, 0 or 1. Kept apart from the nodes, since searches never read it.
    bal: *mut i8,
    /// The slice, where elements are compared if they are not in the nodes.
    keys: *const T,
    root: usize,
}

impl<T: Ord, L: Link, V> Tree<T, L, V> {
    /// Whether the elements are in the nodes.
    const EMBED: bool = mem::size_of::<V>() != 0;

    /// Returns where the element of node `i` is.
    ///
    /// # Safety
    ///
    /// `i` must be a node.
    #[inline(always)]
    unsafe fn key(&self, i: usize) -> *const T {
        // SAFETY: `i` is a node, as the caller guarantees, and so an index into the slice.
        unsafe {
            if Self::EMBED {
                (&raw const (*self.nodes.add(i)).val).cast::<T>()
            } else {
                self.keys.add(i)
            }
        }
    }

    /// Returns the links of node `i`.
    ///
    /// # Safety
    ///
    /// `i` must be a node.
    #[inline(always)]
    unsafe fn link(&self, i: usize) -> *mut [L; 2] {
        // SAFETY: `i` is a node, as the caller guarantees.
        unsafe { &raw mut (*self.nodes.add(i)).link }
    }

    /// Returns the child of node `i` on side `dir` (1 for right).
    ///
    /// # Safety
    ///
    /// `i` must be a node and `dir` 0 or 1.
    #[inline(always)]
    unsafe fn child(&self, i: usize, dir: usize) -> L {
        // SAFETY: as the caller guarantees.
        unsafe { *self.link(i).cast::<L>().add(dir) }
    }

    /// Sets the child of node `i` on side `dir`.
    ///
    /// # Safety
    ///
    /// `i` must be a node and `dir` 0 or 1.
    #[inline(always)]
    unsafe fn set_child(&self, i: usize, dir: usize, c: L) {
        // SAFETY: as the caller guarantees.
        unsafe { *self.link(i).cast::<L>().add(dir) = c }
    }

    /// Returns the balance factor of node `i`.
    ///
    /// # Safety
    ///
    /// `i` must be a node.
    #[inline(always)]
    unsafe fn bal(&self, i: usize) -> i8 {
        // SAFETY: `i` is a node, as the caller guarantees.
        unsafe { *self.bal.add(i) }
    }

    /// Sets the balance factor of node `i`.
    ///
    /// # Safety
    ///
    /// `i` must be a node.
    #[inline(always)]
    unsafe fn set_bal(&self, i: usize, b: i8) {
        // SAFETY: `i` is a node, as the caller guarantees.
        unsafe { *self.bal.add(i) = b }
    }

    /// Searches for the slot of `x` from node `c`, whose ancestors are `path[..depth]`,
    /// recording the path; returns its new length. The slot is the last entry.
    ///
    /// # Safety
    ///
    /// `c` must be a node of the tree, with `path[..depth]` the path from the root to it.
    #[inline(always)]
    unsafe fn search_from(&self, x: &T, path: &mut Path, mut depth: usize, mut c: usize) -> usize {
        // SAFETY: `c` is a node, and every link followed leads to another one. The path never
        // grows beyond the height of the tree, which `HEIGHT` bounds; indexing checks it all
        // the same.
        unsafe {
            loop {
                let right = !(*x < *self.key(c));
                let next = L::pick(self.link(c), right);
                path[depth].write(c << 1 | usize::from(right));
                depth += 1;
                if next == L::NIL {
                    return depth;
                }
                c = next.get();
            }
        }
    }

    /// Searches for the slot of `x` from the root, recording the path; returns its length.
    ///
    /// # Safety
    ///
    /// The tree must have a node.
    #[inline(always)]
    unsafe fn search(&self, x: &T, path: &mut Path) -> usize {
        // SAFETY: the root is a node, with no ancestors.
        unsafe { self.search_from(x, path, 0, self.root) }
    }

    /// Searches for the slot of `x` below `path[..from]`, which is still the path from the
    /// root to the last node in it; returns the length of the new path.
    ///
    /// # Safety
    ///
    /// The tree must have a node, and `path[..from]` must be a path from the root.
    #[inline(always)]
    unsafe fn resume(&self, x: &T, path: &mut Path, from: usize) -> usize {
        // SAFETY: as the caller guarantees: the child of the last entry is a node, if any.
        unsafe {
            if from == 0 {
                return self.search(x, path);
            }
            let e = path[from - 1].assume_init();
            let c = self.child(e >> 1, e & 1);
            if c == L::NIL {
                return from;
            }
            self.search_from(x, path, from, c.get())
        }
    }

    /// Searches for the slots of the `LANES` elements `xs` at once, recording their paths;
    /// returns their lengths.
    ///
    /// Each step of each search waits for the one before it, so one search at a time leaves
    /// the processor idle; the steps of several searches overlap. Each comparison is made
    /// once: a search that has found its slot is not stepped again.
    ///
    /// # Safety
    ///
    /// The tree must have a node.
    #[inline(always)]
    unsafe fn search_lanes<const LANES: usize>(
        &self,
        xs: [&T; LANES],
        paths: &mut [Path; LANES],
    ) -> [usize; LANES] {
        // SAFETY: as in `search_from`, for each search.
        unsafe {
            let mut c = [self.root; LANES];
            let mut depth = [0; LANES];
            let mut active = [true; LANES];
            loop {
                for j in 0..LANES {
                    if active[j] {
                        let right = !(*xs[j] < *self.key(c[j]));
                        let next = L::pick(self.link(c[j]), right);
                        paths[j][depth[j]].write(c[j] << 1 | usize::from(right));
                        depth[j] += 1;
                        // A finished search's node is never read again, so it is overwritten
                        // without a select on the chain of the next step.
                        active[j] = next != L::NIL;
                        c[j] = next.get();
                    }
                }
                if active == [false; LANES] {
                    return depth;
                }
            }
        }
    }

    /// Records the path from `node`, whose ancestors are `path[..depth]`, to the slot right
    /// after it (`dir` 1) or right before it (`dir` 0); returns the new length of the path.
    ///
    /// # Safety
    ///
    /// `node` must be a node, with `path[..depth]` the path from the root to it.
    #[inline(always)]
    unsafe fn beside(&self, node: usize, dir: usize, path: &mut Path, mut depth: usize) -> usize {
        // SAFETY: `node` is a node, and every link followed leads to another one.
        unsafe {
            path[depth].write(node << 1 | dir);
            depth += 1;
            let mut c = self.child(node, dir);
            while c != L::NIL {
                path[depth].write(c.get() << 1 | (dir ^ 1));
                depth += 1;
                c = self.child(c.get(), dir ^ 1);
            }
            depth
        }
    }

    /// Attaches node `q` at the slot `path[depth - 1]` and restores the balance of the tree.
    /// If `keep`, `path` is rewritten to the ancestors of `q` after the rotation; otherwise
    /// only `path[..k]` is still valid. Returns the node and the depth `k` of the rotation
    /// (depth `HEIGHT` if there was none) and the number of ancestors of `q` (if `keep`).
    ///
    /// Going up from the slot, each node whose subtrees were as high as each other leans to
    /// the side that grew, and the tree is one level higher there: the first node that
    /// leaned the other way is now balanced, and one that leaned that way already is rotated,
    /// once or twice, which brings its subtree back to its height before the insertion.
    ///
    /// # Safety
    ///
    /// `q` must be a node not in the tree yet, without children, and `path[..depth]` a path
    /// from the root to an empty slot.
    #[inline(always)]
    unsafe fn attach(
        &mut self,
        q: usize,
        path: &mut Path,
        depth: usize,
        keep: bool,
    ) -> (usize, usize, usize) {
        // SAFETY: every node read or written is on the path or a child of one on it, as AVL
        // rotations need: a node that leans to a side has a child on it. The path entries
        // moved stay within `path[..depth]`.
        unsafe {
            let last = path[depth - 1].assume_init();
            self.set_child(last >> 1, last & 1, L::new(q));
            let mut k = depth;
            while k > 0 {
                k -= 1;
                let e = path[k].assume_init();
                let (p, dir) = (e >> 1, e & 1);
                let s: i8 = if dir == 1 { 1 } else { -1 };
                let b = self.bal(p);
                if b == 0 {
                    self.set_bal(p, s);
                    continue;
                }
                if b == -s {
                    self.set_bal(p, 0);
                    return (0, HEIGHT, depth);
                }
                // `p` leans to side `dir` by two levels now.
                let c = self.child(p, dir).get();
                let (r, ancestors);
                if self.bal(c) == s {
                    // Single rotation: `c` takes the place of `p`, which becomes its child.
                    self.set_child(p, dir, self.child(c, 1 - dir));
                    self.set_child(c, 1 - dir, L::new(p));
                    self.set_bal(p, 0);
                    self.set_bal(c, 0);
                    r = c;
                    // `p` leaves the path; `c` takes its place, still going `dir`.
                    if keep {
                        shift_down(path, k + 1, depth);
                    }
                    ancestors = depth - 1;
                } else {
                    // Double rotation: `g`, the inner child of `c`, takes the place of `p`,
                    // with `c` and `p` as its children.
                    let g = self.child(c, 1 - dir).get();
                    self.set_child(c, 1 - dir, self.child(g, dir));
                    self.set_child(p, dir, self.child(g, 1 - dir));
                    self.set_child(g, dir, L::new(c));
                    self.set_child(g, 1 - dir, L::new(p));
                    let gb = self.bal(g);
                    self.set_bal(p, if gb == s { -s } else { 0 });
                    self.set_bal(c, if gb == -s { s } else { 0 });
                    self.set_bal(g, 0);
                    r = g;
                    if k + 2 == depth {
                        // `g` is `q`, now at depth `k`.
                        ancestors = k;
                    } else if keep {
                        // The path went on from `g` into one of its subtrees, which now
                        // hangs below `c` or `p`.
                        let dg = path[k + 2].assume_init() & 1;
                        if dg == dir {
                            path[k].write(g << 1 | dir);
                            path[k + 1].write(c << 1 | (1 - dir));
                        } else {
                            path[k].write(g << 1 | (1 - dir));
                            path[k + 1].write(p << 1 | dir);
                        }
                        shift_down(path, k + 3, depth);
                        ancestors = depth - 1;
                    } else {
                        ancestors = depth - 1;
                    }
                }
                if k == 0 {
                    self.root = r;
                } else {
                    // Entry `k - 1` is above the rotation: never rewritten.
                    let e = path[k - 1].assume_init();
                    self.set_child(e >> 1, e & 1, L::new(r));
                }
                return (p, k, ancestors);
            }
            (0, HEIGHT, depth)
        }
    }
}

/// Moves `path[from..to]` down by one entry, overwriting `path[from - 1]`.
///
/// Rotations usually happen a level or two above the slot, so this usually moves one or two
/// entries: inline, rather than through a call to `memmove`, which `ptr::copy` makes for a
/// length unknown at compile time: 1.1–1.3 times faster on sorted and reverse-sorted input.
#[inline(always)]
fn shift_down(path: &mut Path, from: usize, to: usize) {
    match to - from {
        0 => {}
        1 => path[from - 1] = path[from],
        2 => {
            path[from - 1] = path[from];
            path[from] = path[from + 1];
        }
        _ => path.copy_within(from..to, from - 1),
    }
}

/// Returns the in-order neighbours of the slot at the end of `path`: the deepest nodes on it
/// from which the path goes right and left, or `NONE`.
#[inline(always)]
fn neighbours(path: &Path, depth: usize) -> (usize, usize) {
    let (mut pred, mut succ) = (NONE, NONE);
    for k in (0..depth).rev() {
        // SAFETY: `path[..depth]` has been written.
        let e = unsafe { path[k].assume_init() };
        if e & 1 == 1 {
            if pred == NONE {
                pred = e >> 1;
                if succ != NONE {
                    break;
                }
            }
        } else if succ == NONE {
            succ = e >> 1;
            if pred != NONE {
                break;
            }
        }
    }
    (pred, succ)
}

/// Puts the elements copied into the nodes back where they started, if a comparison panics
/// while they are there.
struct Guard<T, L, V> {
    nodes: *mut Node<L, V>,
    arr: *mut T,
    /// How many nodes hold an element.
    len: usize,
}

impl<T, L, V> Drop for Guard<T, L, V> {
    fn drop(&mut self) {
        if mem::size_of::<V>() == 0 {
            return;
        }
        for i in 0..self.len {
            // SAFETY: node `i` holds the element that started at `arr[i]`, the only copy in
            // use: the slice holds a stale one, which this overwrites.
            unsafe {
                ptr::copy_nonoverlapping(
                    (&raw const (*self.nodes.add(i)).val).cast::<T>(),
                    self.arr.add(i),
                    1,
                )
            }
        }
    }
}

/// The size of the memory kept on the stack for the tree, in bytes.
///
/// Measured against always allocating: 2.1–3.4 times faster on 2 elements, 1.1–1.4 times on 10
/// and up to 1.27 times on 30; from 100 elements on within 5% either way.
const STACK_BYTES: usize = 4096;

/// Room for `STACK_BYTES` bytes, aligned for `A`.
#[repr(C)]
struct StackBuffer<A> {
    _align: [A; 0],
    bytes: [MaybeUninit<u8>; STACK_BYTES],
}

/// Returns the memory for `len` nodes `N`, then, unless `embed`, `len` elements `T` to gather
/// the sorted elements into, then `len` balance factors, with the offsets of the last two; or
/// `None` if it overflows.
fn regions<T, N>(len: usize, embed: bool) -> Option<(Layout, usize, usize)> {
    let mut layout = Layout::array::<N>(len).ok()?;
    let mut gather = 0;
    if !embed {
        (layout, gather) = layout.extend(Layout::array::<T>(len).ok()?).ok()?;
    }
    let (layout, balance) = layout.extend(Layout::array::<i8>(len).ok()?).ok()?;
    Some((layout, gather, balance))
}

/// Sorts `arr`, of at least two elements, with nodes of `V` and links of `L`, searching for
/// `LANES` places at once.
fn sort<T: Ord, L: Link, V, const LANES: usize>(arr: &mut [T]) {
    let len = arr.len();
    let embed = Tree::<T, L, V>::EMBED;
    let (layout, gather, balance) =
        regions::<T, Node<L, V>>(len, embed).expect("capacity overflow");
    // The memory is only raw storage: the vector's length stays 0, so it never drops what
    // passes through it.
    let mut stack = StackBuffer::<(Node<L, V>, T)> {
        _align: [],
        bytes: [MaybeUninit::uninit(); STACK_BYTES],
    };
    let mut heap: Vec<MaybeUninit<(Node<L, V>, T)>> = Vec::new();
    let region: *mut u8 = if layout.size() <= STACK_BYTES {
        stack.bytes.as_mut_ptr().cast()
    } else {
        heap.reserve_exact(layout.size().div_ceil(mem::size_of::<(Node<L, V>, T)>()));
        heap.as_mut_ptr().cast()
    };
    let base = arr.as_mut_ptr();
    // SAFETY: `region` has room for `layout`, aligned for nodes, elements and bytes alike, and
    // `balance` is the offset of its part for balance factors.
    let mut t = unsafe {
        Tree::<T, L, V> {
            nodes: region.cast(),
            bal: region.add(balance).cast(),
            keys: base,
            root: 0,
        }
    };
    // SAFETY: the part of `region` for balance factors has room for `len` of them, and that at
    // `gather`, if elements stay in the slice, for `len` elements.
    let out: *mut T = unsafe {
        ptr::write_bytes(t.bal, 0, len);
        if embed {
            base
        } else {
            region.add(gather).cast()
        }
    };
    let mut guard = Guard::<T, L, V> {
        nodes: t.nodes,
        arr: base,
        len: 0,
    };
    // SAFETY: nodes `0..len` all exist, and every index below is one of them or a slot of
    // `paths`, which indexing checks. Elements copied into the nodes are only compared there
    // until they are copied back, and `guard` copies them back if a comparison panics; elements
    // in the slice are only read until every comparison has been made.
    unsafe {
        for i in 0..len {
            let n = t.nodes.add(i);
            ptr::write(&raw mut (*n).link, [L::NIL; 2]);
            if embed {
                ptr::copy_nonoverlapping(base.add(i), (&raw mut (*n).val).cast::<T>(), 1);
            }
        }
        guard.len = len;
        let mut paths: [Path; LANES] = [[MaybeUninit::uninit(); HEIGHT]; LANES];
        // Node 0 is the root. The last node inserted, the path holding its ancestors and their
        // number, and its in-order neighbours, all kept up to date while `finger`, which says
        // that it went beside the one inserted before it.
        let mut last = 0;
        let mut f = 0;
        let mut ancestors = 0;
        let (mut pred, mut succ) = (NONE, NONE);
        let mut finger = true;
        let mut q = 1;
        while q < len {
            let x = &*t.key(q);
            if finger {
                let path = &mut paths[f];
                if !(*x < *t.key(last)) {
                    if succ == NONE || *x < *t.key(succ) {
                        let depth = t.beside(last, 1, path, ancestors);
                        ancestors = t.attach(q, path, depth, true).2;
                        pred = last;
                        last = q;
                        q += 1;
                        continue;
                    }
                } else if pred == NONE || !(*x < *t.key(pred)) {
                    let depth = t.beside(last, 0, path, ancestors);
                    ancestors = t.attach(q, path, depth, true).2;
                    succ = last;
                    last = q;
                    q += 1;
                    continue;
                }
                finger = false;
            }
            if q + LANES > len {
                let path = &mut paths[0];
                let depth = t.search(x, path);
                if path[depth - 1].assume_init() >> 1 == last {
                    finger = true;
                    (pred, succ) = neighbours(path, depth);
                }
                ancestors = t.attach(q, path, depth, finger).2;
                f = 0;
                last = q;
                q += 1;
                continue;
            }
            let xs: [&T; LANES] = core::array::from_fn(|j| &*t.key(q + j));
            let depths = t.search_lanes(xs, &mut paths);
            // The slot each element took, and the node and depth of the rotation it caused.
            let mut slots = [0usize; LANES];
            let mut rotated = [(0usize, HEIGHT); LANES];
            for j in 0..LANES {
                let path = &mut paths[j];
                let mut depth = depths[j];
                let slot = path[depth - 1].assume_init();
                // The path is still the path from the root down to the shallowest node an
                // earlier insertion rotated, if any, and otherwise all the way, but its slot
                // may have been taken. Either way the search goes on from there. A path that
                // the earlier insertions left alone is a path in the tree as they left it, so
                // the depths of their rotations index it.
                let mut from = depth;
                let mut taken = false;
                for i in 0..j {
                    let (z, k) = rotated[i];
                    taken |= slots[i] == slot;
                    if k < from && path[k].assume_init() >> 1 == z {
                        from = k;
                    }
                }
                if from < depth || taken {
                    depth = t.resume(xs[j], path, from);
                }
                let slot = path[depth - 1].assume_init();
                slots[j] = slot;
                let keep = j == LANES - 1 && slot >> 1 == q + j - 1;
                if keep {
                    finger = true;
                    (pred, succ) = neighbours(path, depth);
                }
                let (z, k, a) = t.attach(q + j, path, depth, keep);
                rotated[j] = (z, k);
                ancestors = a;
            }
            f = LANES - 1;
            last = q + LANES - 1;
            q += LANES;
        }
        // No comparisons from here on: nothing can panic.
        mem::forget(guard);
        // In-order traversal, with `paths[0]` as the stack of nodes whose left subtree is
        // being read out.
        let pending = &mut paths[0];
        let mut sp = 0;
        let mut cur = L::new(t.root);
        let mut k = 0;
        loop {
            while cur != L::NIL {
                pending[sp].write(cur.get());
                sp += 1;
                cur = t.child(cur.get(), 0);
            }
            if sp == 0 {
                break;
            }
            sp -= 1;
            let n = pending[sp].assume_init();
            ptr::copy_nonoverlapping(t.key(n), out.add(k), 1);
            k += 1;
            cur = t.child(n, 1);
        }
        if !embed {
            ptr::copy_nonoverlapping(out, base, len);
        }
    }
}

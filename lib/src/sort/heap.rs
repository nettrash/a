use core::hint::select_unpredictable;
use core::mem::{self, ManuallyDrop};
use core::ptr;

/// Sorts a slice in place using heap sort.
///
/// Heap sort arranges the slice into a max-heap: a tree stored in the slice itself, in which
/// no element is smaller than its children, so the largest element is at the root. It then
/// repeatedly moves the root to the end of the heap, shrinks the heap by one element and
/// restores the heap order. Building the heap takes `O(n)` comparisons and each of the `n`
/// removals `O(log n)`, so the sort makes `O(n log n)` comparisons whatever the input, needs no
/// buffer and does not recurse.
///
/// Five refinements keep the algorithm as cheap as heap sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or strictly descending run; if it covers the whole slice, the slice is left
///   alone or reversed. This costs `n − 1` comparisons on such input and usually two or three
///   on any other. Heap sort itself is not adaptive: it would spend `n·log₂ n` comparisons on
///   sorted input.
/// * **Bottom-up removal.** The textbook sift-down compares the element moved to the root with
///   the larger child at every level, two comparisons per level. That element comes from the
///   end of the heap and is usually small, so it sinks almost to the bottom anyway. Instead,
///   the vacant root is moved down to a leaf along the larger children, one comparison per
///   level, and the element is moved up from there, usually by a level or two (Wegener's
///   bottom-up heap sort). This takes the comparisons from about `1.7–1.85·n·log₂ n` down to
///   about `n·log₂ n`.
/// * **Holes instead of swaps.** The element being placed is held outside the slice, and each
///   level moves one element into the vacant slot instead of swapping two: one move instead of
///   three.
/// * **Four children per node for large heaps and small elements.** A heap with four children
///   per node has half as many levels, whose children are adjacent in memory, at the cost of
///   three comparisons per level instead of one, about `1.5·n·log₂ n` in all. That pays from
///   64 KiB of elements on, where the heap no longer fits in the processor's fastest cache and
///   every level costs a cache miss: on random input 1.15–1.35 times faster than two children
///   for integers from 100,000 elements on, 1.1–1.4 times for strings and 1.25–2 times for
///   64-byte elements; on nearly sorted strings, whose comparisons cost the most, 1.05–1.15
///   times slower. It also pays for elements of at most 8 bytes at any length: their
///   comparisons are so cheap that a level costs about the time to load the children, compare
///   them and find the next ones, so half as many levels make integers sort 1.15–1.25 times
///   faster in cache.
/// * **Short chains from one level to the next.** Each level of a removal waits for the
///   comparison before it, which says where the next children are. Every child is loaded from
///   the first child's index with no address to compute first, and for elements of up to 16
///   bytes the larger child and its own first child are chosen by selects between values
///   computed beforehand; with four children the winners of the first round are selected as
///   pointers, so that the values already loaded are compared instead of being loaded again.
///   This alone makes integers sort 1.25–1.3 times faster on four children per node.
///
/// Together, against the textbook heap sort: 1.1–1.25 times faster for integers up to 1000
/// elements and 1.4–1.85 times from 10,000 on, 1.05–1.4 times for strings from 1000 elements
/// on, and 1.3–2.2 times for 256-byte elements. Only sawtooth input of 1000–2000 integers, in
/// whose short periods three comparisons per level cost more than the levels saved, would sort
/// 1.02–1.05 times faster on two children per node.
///
/// # Properties
///
/// * **Unstable**: equal elements may be reordered.
/// * **In place**: no allocation, `O(1)` stack.
/// * **Not adaptive**, except to sorted and reverse-sorted input.
///
/// | Case                     | Comparisons                     | Moves              |
/// |--------------------------|---------------------------------|--------------------|
/// | Best (sorted, reverse)   | `n − 1`                         | `0` (`n` reversed) |
/// | Average (random)         | `≈ n·log₂ n + 0.6n`             | `≈ n·log₂ n`       |
/// | Four children per node   | `≈ 1.5·n·log₂ n`                | `≈ 0.5·n·log₂ n`   |
/// | Worst                    | `O(n log n)`                    | `O(n log n)`       |
///
/// Four children per node: from 64 KiB of elements on, and for elements of at most 8 bytes.
/// For `n = 1000`: 999 comparisons on sorted or reverse-sorted input and about 10,500 on
/// random input (about 14,600 for elements of at most 8 bytes), against about 16,800 for the
/// textbook heap sort.
///
/// Heap sort's strength is its guarantee: `O(n log n)` on every input, in place, with constant
/// stack. It jumps around the slice, though, so [`slice::sort_unstable`] and this crate's
/// [`quick`](fn@super::quick) sort, which falls back to it only when its pivots keep failing,
/// are faster: 2–3.4 times for random integers from 64 elements on, 1.6–6 times for strings
/// and 1.1–3.5 times for 256-byte elements.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: nothing is dropped twice or leaked. An [`Ord`] that is not a total
/// order leaves `arr` in an unspecified order with every element exactly once as well.
///
/// # Examples
///
/// ```
/// use a::sort::heap;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// heap(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// heap(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn heap<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    // An ascending run stops at the first descent; a strictly descending one at the first
    // element that is not smaller, so reversing it is all it takes.
    let descending = arr[1] < arr[0];
    let run = 2 + if descending {
        arr[1..].windows(2).take_while(|w| w[1] < w[0]).count()
    } else {
        arr[1..].windows(2).take_while(|w| w[1] >= w[0]).count()
    };
    if run == len {
        if descending {
            arr.reverse();
        }
        return;
    }
    if mem::size_of::<T>() > QUATERNARY_MAX_SIZE
        && len.saturating_mul(mem::size_of::<T>()) < QUATERNARY_MIN_BYTES
    {
        heap_sort::<T, 2>(arr);
    } else {
        heap_sort::<T, 4>(arr);
    }
}

/// From this many bytes of elements on, the heap has four children per node instead of two,
/// whatever the element size.
///
/// Measured: below it, two children are 1.02–1.15 times faster for strings; from about this
/// size on, four are, for 64-byte elements from 1000 and for `u64`s (before they took four at
/// every length) from 100,000, where the heap leaves the 128 KiB first-level cache of the
/// Apple M4 measured on.
const QUATERNARY_MIN_BYTES: usize = 64 * 1024;

/// Elements of at most this many bytes use a heap with four children per node at any length.
///
/// Their comparisons are so cheap that a level costs about the latency of loading the
/// children, comparing them and working out where the next children are, and half as many
/// levels outweigh three comparisons per level instead of one: measured below 64 KiB,
/// integers sort 1.15–1.25 times faster with four children, `u128`s 1.15–1.4 times, but
/// `&str`s, as small as a `u128` and slower to compare, 1.05–1.15 times slower.
const QUATERNARY_MAX_SIZE: usize = 8;

/// The largest element size, in bytes, for which [`largest_child`] chooses its results by
/// selects instead of arithmetic.
///
/// Larger elements compare by chains of branches (64-byte arrays element by element), after
/// which the compiler turns the selects into a branch on the answer: 1.35–1.5 times slower
/// for 64-byte elements from 10,000 on. With arithmetic, they sort as before.
const SELECT_MAX_SIZE: usize = 16;

/// Sorts `v`, of at least two elements, by heap sort on a heap with `D` children per node, `D`
/// being 2 or 4. The children of `v[i]` are `v[D·i + 1..=D·i + D]`.
fn heap_sort<T: Ord, const D: usize>(v: &mut [T]) {
    let len = v.len();
    let base = v.as_mut_ptr();
    // SAFETY: `len >= 2`. Every node passed to `sift_down` has a child, so it is below `len`,
    // and every `end` passed to `pop` is in `1..len`: both stay inside `v`.
    unsafe {
        // Every node with a child, from the last one up: each is sifted down into the heap
        // its subtrees already form.
        for node in (0..(len - 1).div_ceil(D)).rev() {
            sift_down::<T, D>(base, node, len);
        }
        for end in (1..len).rev() {
            pop::<T, D>(base, end);
        }
    }
}

/// Returns the index of the largest of the `D` children starting at `c`, with one comparison
/// for two children and three, in two rounds, for four, and the index of that child's first
/// child.
///
/// Each level of a removal waits for the one before: the children's addresses depend on the
/// last comparison. `bases[k]` is `base.add(k)`, so that every child is loaded from `c`
/// directly, with no address to compute first. For elements of at most [`SELECT_MAX_SIZE`]
/// bytes, both results are chosen by selects between values computed before the comparison,
/// so that a single select lies between a comparison and the next level's loads; with four
/// children, the winners of the first round are selected as pointers, so that the compiler
/// compares the values it has already loaded instead of loading the winners again. Together,
/// this makes a removal from a heap with four children 1.25–1.3 times faster for integers,
/// and one from a heap with two 1.02–1.05 times.
///
/// # Safety
///
/// `base` must be valid for `c + D` elements.
#[inline(always)]
unsafe fn largest_child<T: Ord, const D: usize>(bases: &[*const T; D], c: usize) -> (usize, usize) {
    // SAFETY: `c..c + D` is in bounds, as the caller guarantees.
    unsafe {
        if mem::size_of::<T>() > SELECT_MAX_SIZE {
            let (base, last) = (bases[0], c + (D - 1));
            let child = if D == 2 {
                c + usize::from(*base.add(c) < *base.add(last))
            } else {
                let left = c + usize::from(*base.add(c) < *base.add(c + 1));
                let right = c + 2 + usize::from(*base.add(c + 2) < *base.add(last));
                if *base.add(left) < *base.add(right) {
                    right
                } else {
                    left
                }
            };
            return (child, D * child + 1);
        }
        if D == 2 {
            let right = *bases[0].add(c) < *bases[1].add(c);
            return (
                select_unpredictable(right, c + 1, c),
                select_unpredictable(right, 2 * c + 3, 2 * c + 1),
            );
        }
        let (p0, p1) = (bases[0].add(c), bases[1].add(c));
        let (p2, p3) = (bases[2].add(c), bases[3].add(c));
        let (b0, b1) = (*p0 < *p1, *p2 < *p3);
        // The winners are selected as pointers, so the compiler compares the values it has
        // already loaded instead of loading the winners again.
        let left = select_unpredictable(b0, p1, p0);
        let right = select_unpredictable(b1, p3, p2);
        let (li, ri) = (
            select_unpredictable(b0, c + 1, c),
            select_unpredictable(b1, c + 3, c + 2),
        );
        let b2 = *left < *right;
        (
            select_unpredictable(b2, ri, li),
            select_unpredictable(b2, D * ri + 1, D * li + 1),
        )
    }
}

/// Returns `[base, base + 1, …, base + D − 1]`, the pointers [`largest_child`] addresses the
/// children from.
#[inline(always)]
fn child_bases<T, const D: usize>(base: *const T) -> [*const T; D] {
    core::array::from_fn(|k| base.wrapping_add(k))
}

/// Returns the index of the largest of the `k` children starting at `c`, `1 <= k < D`: the
/// last node with children may have fewer than `D`.
///
/// # Safety
///
/// `base` must be valid for `c + k` elements.
#[inline(always)]
unsafe fn largest_of_few<T: Ord, const D: usize>(base: *const T, c: usize, k: usize) -> usize {
    if D == 2 {
        return c;
    }
    let mut largest = c;
    for i in c + 1..c + k {
        // SAFETY: `i < c + k`, which is in bounds, as the caller guarantees.
        if unsafe { *base.add(largest) < *base.add(i) } {
            largest = i;
        }
    }
    largest
}

/// Moves `v[node]` down the heap `v[..len]` until no child is larger, `v` starting at `base`;
/// the subtrees of `node` must be heaps already.
///
/// # Safety
///
/// `node < len` and `base` must be valid for `len` elements.
unsafe fn sift_down<T: Ord, const D: usize>(base: *mut T, mut node: usize, len: usize) {
    // SAFETY: every index is a node below `len` or one of its children, checked against `len`
    // before use. The element at `node` is read out, which leaves its slot vacant; each level
    // moves a child up into the vacant slot, so the vacant slot moves down. The held element is
    // compared as the copy that gets written back, and if a comparison panics, `hole` writes it
    // into the vacant slot, so `v` holds every element exactly once.
    unsafe {
        let held = ManuallyDrop::new(ptr::read(base.add(node)));
        let mut hole = Hole {
            src: &*held,
            dst: base.add(node),
        };
        let bases = child_bases::<T, D>(base);
        loop {
            let c = D * node + 1;
            let child = if c + (D - 1) < len {
                largest_child::<T, D>(&bases, c).0
            } else if c < len {
                largest_of_few::<T, D>(base, c, len - c)
            } else {
                break;
            };
            if *held >= *base.add(child) {
                break;
            }
            ptr::copy_nonoverlapping(base.add(child), base.add(node), 1);
            node = child;
            hole.dst = base.add(node);
        }
        // Dropping `hole` writes the held element into the vacant slot.
    }
}

/// Moves the largest element of the heap `v[..=end]` to `v[end]` and makes `v[..end]` a heap
/// again, `v` starting at `base`.
///
/// The last element is held out and the root moved into its slot, leaving the root vacant.
/// The vacant slot then moves down to a leaf along the larger children, which move up, and the
/// held element moves up from there past the elements smaller than it.
///
/// # Safety
///
/// `1 <= end` and `base` must be valid for `end + 1` elements.
unsafe fn pop<T: Ord, const D: usize>(base: *mut T, end: usize) {
    // SAFETY: every index is the vacant node, below `end`, one of its children, checked
    // against `end` before use, or its parent. The descent compares children in place, and the
    // ascent compares the held element as the copy that gets written back. If a comparison
    // panics, `hole` writes the held element into the vacant slot, so `v` holds every element
    // exactly once.
    unsafe {
        let held = ManuallyDrop::new(ptr::read(base.add(end)));
        ptr::copy_nonoverlapping(base, base.add(end), 1);
        let mut hole = Hole {
            src: &*held,
            dst: base,
        };
        let bases = child_bases::<T, D>(base);
        let (mut node, mut c) = (0, 1);
        while c + (D - 1) < end {
            let (child, next) = largest_child::<T, D>(&bases, c);
            ptr::copy_nonoverlapping(base.add(child), base.add(node), 1);
            node = child;
            c = next;
            hole.dst = base.add(node);
        }
        // Only the last node with children can have fewer than `D`, and their children are
        // leaves.
        if c < end {
            let child = largest_of_few::<T, D>(base, c, end - c);
            ptr::copy_nonoverlapping(base.add(child), base.add(node), 1);
            node = child;
            hole.dst = base.add(node);
        }
        while node > 0 {
            let parent = (node - 1) / D;
            if *base.add(parent) >= *held {
                break;
            }
            ptr::copy_nonoverlapping(base.add(parent), base.add(node), 1);
            node = parent;
            hole.dst = base.add(node);
        }
        // Dropping `hole` writes the held element into the vacant slot.
    }
}

/// A vacant slot of a slice, plus the element held outside the slice that belongs there.
///
/// Writing the held element on drop is what keeps [`sift_down`] and [`pop`] panic-safe: the
/// slice is whole again even if a comparison unwinds while the element is out.
struct Hole<T> {
    src: *const T,
    dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to the single vacant slot of the slice being sorted, and `src`
        // to the held element, which is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) }
    }
}

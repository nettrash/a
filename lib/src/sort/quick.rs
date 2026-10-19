use core::mem::{self, ManuallyDrop};
use core::ptr;

/// Sorts a slice in place using quick sort.
///
/// Quick sort picks a pivot element, partitions the slice into the elements smaller than the
/// pivot and the rest, puts the pivot between them, and sorts the two parts the same way. With
/// pivots near the median every level halves the parts, so the sort makes `O(n log n)`
/// comparisons, and it needs no buffer.
///
/// Seven refinements keep the algorithm as cheap as quick sort can be:
///
/// * **Sorted and reverse-sorted input is recognised.** The slice is first scanned for one
///   ascending or strictly descending run; if it covers the whole slice, the slice is left
///   alone or reversed. This costs `n − 1` comparisons on such input and usually two or three
///   on any other.
/// * **Pivots from a median of samples.** The pivot is the median of three elements spread over
///   the slice, from 64 elements on the median of three such medians (the "ninther"), and from
///   4096 elements on a pseudo-median of 27 evenly spaced samples, three times as many for
///   every 16 times the length (243 for a million elements). On random input this keeps
///   comparisons at about `n·log₂ n`, against `1.39·n·log₂ n` for a random pivot.
/// * **Branchless partitioning for elements of at most 64 bytes.** Whether an element is
///   smaller than the pivot is a coin flip on random input, which defeats the branch predictor.
///   Instead, every element is moved unconditionally, with the comparison only choosing where:
///   a vacant slot cycles through the slice, and each element either joins the smaller part or
///   moves to the end of the larger one (Lomuto's scheme, made branchless). That costs two
///   moves per element, which larger elements pay for more than a misprediction, so they are
///   partitioned by Hoare's scheme, which moves only misplaced pairs, two moves per pair by
///   cycling them through a vacant slot instead of three by swapping. On random input the
///   branchless scheme is 1.9–4.3 times faster than Hoare's for integers and 1.4–2.5 times for
///   strings.
/// * **Runs of equal elements in one pass.** If the pivot equals the pivot of the enclosing
///   partition, which is no larger than any element of this part, the part is partitioned into
///   elements not larger than the pivot, all equal to it and done, and the rest. Input with `k`
///   distinct keys costs `O(n log k)`, and all-equal input is never quadratic.
/// * **Recursion into the smaller part only.** The larger part is sorted by looping, so the
///   stack never holds more than `log₂ n` frames.
/// * **Sorting networks for short parts.** For elements of at most 16 bytes, parts of at most
///   8 elements are sorted by optimal sorting networks (fixed sequences of 1 to 19
///   compare-exchanges) made of branchless swaps, unrolled for each length so the elements
///   stay in registers. On random input this makes integers sort 1.3–1.85 times faster than
///   with insertion sort. Larger elements pay for a compare-exchange's three moves, so their
///   parts of at most 16 elements are sorted by insertion sort.
/// * **A guaranteed `O(n log n)`.** A part still being partitioned after `2·log₂ n` levels
///   means the pivots were consistently bad, which no sampling rules out (an adversary that
///   answers comparisons adaptively can force it). Such a part is finished by heap sort, so
///   no input makes the sort quadratic (the "introsort" idea).
///
/// # Properties
///
/// * **Unstable**: equal elements may be reordered.
/// * **In place**: no allocation, `O(log n)` stack.
/// * **Adaptive** to sorted and reverse-sorted input, and to repeated keys.
///
/// | Case                              | Comparisons                     | Moves              |
/// |-----------------------------------|---------------------------------|--------------------|
/// | Best (sorted, reverse, all equal) | `n − 1`                         | `0` (`n` reversed) |
/// | Average (random)                  | `≈ n·log₂ n + (0.2–1)·n`        | `≈ 2n·log₂(n/8)`   |
/// | `k` distinct keys                 | `O(n log k)`                    | `O(n log k)`       |
/// | Worst                             | `O(n log n)`                    | `O(n log n)`       |
///
/// For `n = 1000`: 999 comparisons on sorted or reverse-sorted input and about 10,200 on
/// random input (`n·log₂ n + 0.22n`); for `n = 100,000` and `n = 1,000,000`, about
/// `n·log₂ n + 0.6n`. Elements larger than 16 bytes, whose short parts are sorted by insertion
/// sort, take up to `0.4n` more.
///
/// On nearly sorted input (a few elements out of place), Hoare's scheme would be 1.5–1.75 times
/// faster for integers and 1.2–1.5 times for strings, since its branches are then predictable;
/// on random input it is 1.9–4.3 times slower. [`slice::sort_unstable`], which shares several
/// of these ideas, is 1.1–1.2 times faster on random integers from 100 elements on (its short
/// parts are merged through a buffer, which is only sound for types without interior
/// mutability, something stable Rust cannot test for). This sort is 1.9 times faster on 20
/// integers, 1.03–1.14 times faster on random strings and as fast or faster with 256-byte
/// elements.
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
/// use a::sort::quick;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// quick(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Many duplicates are handled in one pass per distinct pivot.
/// let mut v: Vec<u8> = (0..1000).map(|i| (i * 7 % 3) as u8).collect();
/// quick(&mut v);
/// assert!(v.windows(2).all(|w| w[0] <= w[1]));
///
/// let mut words = vec!["pear", "apple", "fig"];
/// quick(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn quick<T: Ord>(arr: &mut [T]) {
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
    sort(arr, None, 2 * (len | 1).ilog2());
}

/// Parts of at most this many elements are sorted by [`sort_short`] instead of partitioned,
/// when that uses sorting networks.
const SHORT: usize = 8;

/// Parts of at most this many elements are sorted by [`sort_short`] instead of partitioned,
/// when that uses insertion sort.
///
/// Measured against 8 and 12: 1.02–1.03 times faster on random input for elements of 32 to
/// 256 bytes, and up to 1.25 times on nearly sorted input; 1.03 times slower for random
/// `String`s.
const SHORT_INSERTION: usize = 16;

/// From this length on, the pivot is the median of three medians of three.
const NINTHER: usize = 64;

/// From this length on, the pivot is the pseudo-median of 27 samples, and of three times as
/// many for every 16 times the length.
///
/// Measured against the ninther: 1.02–1.03 times faster on random input from 100,000
/// elements, 1.04 times with about `√n` distinct keys and 1.15 times on 1,000,000 elements of
/// a sawtooth pattern; no change on nearly sorted input or below this length.
const PSEUDO_MEDIAN: usize = 4096;

/// The largest element size, in bytes, partitioned by [`partition_lomuto`]; larger elements
/// use [`partition_hoare`].
///
/// Measured: Lomuto's branchless scheme is 1.05–1.35 times faster on random input with
/// 64-byte elements. With 80 and 96 bytes it is still up to 1.15 times faster on random input
/// of 10,000 elements or more, but 1.1 times slower below that, 1.15–1.25 times slower with
/// few distinct keys, and 3–4 times slower on nearly sorted input, whose branches Hoare's
/// scheme predicts.
const LOMUTO_MAX_SIZE: usize = 64;

/// The largest element size, in bytes, for which short parts are sorted by sorting networks.
///
/// Measured on random input: integers, `u128`, pairs and `&str` sort 1.1–1.85 times faster
/// with networks. From 24 bytes on, the three moves per compare-exchange cost more than
/// insertion sort's mispredicted branches: `[u64; 3]` and `[u64; 4]` sort 1.1–1.3 times
/// faster by insertion. `String` is a toss-up: networks are 1.05–1.13 times faster on random
/// input, insertion sort 1.06–1.2 times faster on nearly sorted input or few distinct keys.
const NETWORK_MAX_SIZE: usize = 16;

/// Sorts `v`, all of whose elements are not smaller than `ancestor` if there is one, falling
/// back to heap sort once `limit` more levels of partitioning have not finished the job.
fn sort<'a, T: Ord>(mut v: &'a mut [T], mut ancestor: Option<&'a T>, mut limit: u32) {
    loop {
        let short = if mem::size_of::<T>() <= NETWORK_MAX_SIZE {
            SHORT
        } else {
            SHORT_INSERTION
        };
        if v.len() <= short {
            sort_short(v);
            return;
        }
        if limit == 0 {
            heap_sort(v);
            return;
        }
        limit -= 1;

        let pivot = choose_pivot(v);
        v.swap(0, pivot);
        if let Some(ancestor) = ancestor {
            // No element is smaller than `ancestor`, so a pivot not greater than it equals it,
            // and so does every element not greater than the pivot: those are done.
            if v[0] <= *ancestor {
                let equal = partition(v, &mut |e: &T, pivot: &T| e <= pivot);
                v = &mut v[equal + 1..];
                continue;
            }
        }
        let smaller = partition(v, &mut |e: &T, pivot: &T| e < pivot);
        v.swap(0, smaller);
        let (left, rest) = v.split_at_mut(smaller);
        let (pivot, right) = rest.split_first_mut().expect("the pivot is in `rest`");
        // The elements of `right` are not smaller than `pivot`, those of `left` not smaller
        // than `ancestor`.
        if left.len() < right.len() {
            sort(left, ancestor, limit);
            v = right;
            ancestor = Some(pivot);
        } else {
            sort(right, Some(pivot), limit);
            v = left;
        }
    }
}

/// Partitions `v[1..]` around the pivot `v[0]`, which stays in place: the elements for which
/// `goes_left(element, pivot)` holds come first. Returns how many there are.
fn partition<T: Ord>(v: &mut [T], goes_left: &mut impl FnMut(&T, &T) -> bool) -> usize {
    let (pivot, rest) = v.split_first_mut().expect("`v` is not empty");
    if mem::size_of::<T>() <= LOMUTO_MAX_SIZE {
        partition_lomuto(rest, pivot, goes_left)
    } else {
        partition_hoare(rest, pivot, goes_left)
    }
}

/// Partitions `v` around `pivot` without branching on the comparisons; see [`partition`].
///
/// The first element is held out, leaving a vacant slot. Each following element `e` is then
/// compared with the pivot, the first element of the larger part moves into the vacant slot,
/// which is at the end of the larger part, `e` moves into the slot that frees up, and `e`'s own
/// slot becomes the vacant one; `e` joins the smaller part if it goes left, and is the new
/// first element of the larger part otherwise. Every step makes the same two moves whatever
/// the comparison says, and the held element is placed the same way at the end.
fn partition_lomuto<T>(
    v: &mut [T],
    pivot: &T,
    goes_left: &mut impl FnMut(&T, &T) -> bool,
) -> usize {
    let len = v.len();
    if len == 0 {
        return 0;
    }
    let base = v.as_mut_ptr();
    // SAFETY: every index is below `len`: before step `i` the vacant slot is `i - 1` and
    // `left <= i - 1`, so all pointers stay inside `v`. Before step `i`, the elements that were
    // at `1..i` occupy `0..i - 1`, the vacant slot is `i - 1`, and `v[i..]` is untouched; the
    // comparison runs before the step's moves and with `v[i]` in place, and nothing else can
    // panic. If it panics, `hole` writes the held element into the vacant slot, so `v` holds
    // every element exactly once whatever `goes_left` answered before. `pivot` is not in `v`.
    unsafe {
        let held = ManuallyDrop::new(ptr::read(base));
        let mut hole = Hole {
            src: &*held,
            dst: base,
        };
        let mut left = 0;
        for i in 1..len {
            let e = base.add(i);
            let goes = goes_left(&*e, pivot);
            let first_larger = base.add(left);
            ptr::copy(first_larger, hole.dst, 1);
            ptr::copy_nonoverlapping(e, first_larger, 1);
            hole.dst = e;
            left += usize::from(goes);
        }
        let goes = goes_left(&*held, pivot);
        let first_larger = base.add(left);
        ptr::copy(first_larger, hole.dst, 1);
        hole.dst = first_larger;
        // Dropping `hole` writes the held element into the slot just freed.
        drop(hole);
        left + usize::from(goes)
    }
}

/// Partitions `v` around `pivot` by moving misplaced pairs; see [`partition`].
///
/// Scans from the left for an element that does not go left and from the right for one that
/// does, and repeats until the scans meet. Instead of swapping each such pair, which takes
/// three moves, the elements are moved in one cycle of two moves per pair: the left element
/// of the first pair is held out and the right one moves into its slot, leaving a vacant slot
/// on the right; each later pair moves its left element into the vacant slot and its right
/// one into the left one's slot, which leaves the right one's slot vacant; and the held
/// element fills the last vacant slot. With 256-byte elements this is 1.1 times faster.
fn partition_hoare<T>(v: &mut [T], pivot: &T, goes_left: &mut impl FnMut(&T, &T) -> bool) -> usize {
    let base = v.as_mut_ptr();
    let (mut left, mut right) = (0, v.len());
    // SAFETY: the scans only read `v[left..right]`, and a pair has `left < right`, so every
    // index stays inside `v` and the two elements of a pair are distinct, whatever
    // `goes_left` answers. The vacant slot is always at `right` or beyond, outside the part
    // the scans read, and every element is compared in place, before it moves. If a
    // comparison panics, `hole` writes the held element into the vacant slot, so `v` holds
    // every element exactly once. `pivot` is not in `v`.
    unsafe {
        // Moves `left` and `right` to the next misplaced pair; returns `false` once the scans
        // have met instead, with `left == right`. The right scan stops short of `left`, so an
        // `Ord` that answers differently for the same element cannot make a pair of one.
        let mut next_pair = |left: &mut usize, right: &mut usize| {
            while *left < *right && goes_left(&*base.add(*left), pivot) {
                *left += 1;
            }
            if *left == *right {
                return false;
            }
            loop {
                *right -= 1;
                if *right == *left {
                    return false;
                }
                if goes_left(&*base.add(*right), pivot) {
                    return true;
                }
            }
        };
        if !next_pair(&mut left, &mut right) {
            return left;
        }
        let held = ManuallyDrop::new(ptr::read(base.add(left)));
        ptr::copy_nonoverlapping(base.add(right), base.add(left), 1);
        let mut hole = Hole {
            src: &*held,
            dst: base.add(right),
        };
        left += 1;
        while next_pair(&mut left, &mut right) {
            ptr::copy_nonoverlapping(base.add(left), hole.dst, 1);
            ptr::copy_nonoverlapping(base.add(right), base.add(left), 1);
            hole.dst = base.add(right);
            left += 1;
        }
        // Dropping `hole` writes the held element into the last vacant slot.
        left
    }
}

/// Returns the index of the median of three samples of `v`, from [`NINTHER`] elements on of
/// the median of the medians of three groups of three, and from [`PSEUDO_MEDIAN`] elements on
/// of a pseudo-median of more samples.
fn choose_pivot<T: Ord>(v: &[T]) -> usize {
    let len = v.len();
    let (a, b, c) = (len / 4, len / 2, len / 4 * 3);
    if len < NINTHER {
        return median_of_3(v, a, b, c);
    }
    if len < PSEUDO_MEDIAN {
        let step = len / 8;
        let a = median_of_3(v, a - step, a, a + step);
        let b = median_of_3(v, b - step, b, b + step);
        let c = median_of_3(v, c - step, c, c + step);
        return median_of_3(v, a, b, c);
    }
    let mut depth = 3;
    let mut reach = PSEUDO_MEDIAN * 16;
    while len >= reach {
        depth += 1;
        reach = reach.saturating_mul(16);
    }
    pseudo_median(v, len / 2, len / 3, depth)
}

/// Returns the index of the median of the pseudo-medians around `mid - step`, `mid` and
/// `mid + step`, each `depth - 1` levels deep with a third of the step, or at depth 1 of the
/// three elements there.
///
/// The `3^depth` samples are evenly spaced, `step / 3^(depth - 1)` apart, and lie less than
/// `1.5·step` from `mid`. Samples placed symmetrically around the middle matter: on nearly
/// sorted input, the standard library's `0`, `n/2` and `7n/8` make the pivot's rank about
/// `0.56n` and cost 1.08 times as many comparisons.
fn pseudo_median<T: Ord>(v: &[T], mid: usize, step: usize, depth: u32) -> usize {
    if depth == 1 {
        return median_of_3(v, mid - step, mid, mid + step);
    }
    let a = pseudo_median(v, mid - step, step / 3, depth - 1);
    let b = pseudo_median(v, mid, step / 3, depth - 1);
    let c = pseudo_median(v, mid + step, step / 3, depth - 1);
    median_of_3(v, a, b, c)
}

/// Returns whichever of `a`, `b` and `c` indexes the median of the three elements, with two or
/// three comparisons.
fn median_of_3<T: Ord>(v: &[T], a: usize, b: usize, c: usize) -> usize {
    let b_below_a = v[b] < v[a];
    let c_below_b = v[c] < v[b];
    if b_below_a == c_below_b {
        // `a <= b <= c` or `a > b > c`.
        return b;
    }
    // `b` is the smallest or the largest: the median is the other extreme of `a` and `c`.
    let c_below_a = v[c] < v[a];
    if b_below_a == c_below_a { c } else { a }
}

/// Sorts a part of at most [`SHORT`] elements by a sorting network, or for elements larger
/// than [`NETWORK_MAX_SIZE`] a part of at most [`SHORT_INSERTION`] elements by insertion sort.
fn sort_short<T: Ord>(v: &mut [T]) {
    if mem::size_of::<T>() <= NETWORK_MAX_SIZE {
        match v.len() {
            0 | 1 => {}
            2 => network::<T, 2>(v),
            3 => network::<T, 3>(v),
            4 => network::<T, 4>(v),
            5 => network::<T, 5>(v),
            6 => network::<T, 6>(v),
            7 => network::<T, 7>(v),
            8 => network::<T, 8>(v),
            _ => unreachable!("parts of more than {SHORT} elements are partitioned"),
        }
    } else {
        for end in 2..=v.len() {
            insert_last(&mut v[..end]);
        }
    }
}

/// Sorts `v`, which has `N` elements, by the sorting network for `N` elements.
///
/// Each length gets its own copy of the network, with constant indices, so the compiler keeps
/// the elements in registers between compare-exchanges. A loop over the table with the length
/// known only at run time reloads both elements and checks both indices for each of them;
/// that made the whole sort of random integers 1.07 times slower from 100,000 elements, 1.2
/// times with 100 and 1.5 times with 20.
#[inline(always)]
fn network<T: Ord, const N: usize>(v: &mut [T]) {
    let v: &mut [T; N] = v.try_into().expect("`v` has `N` elements");
    for &(i, j) in NETWORKS[N] {
        compare_exchange(v, i, j);
    }
}

/// Moves the last element of `v` left past the elements greater than it, `v[..len - 1]`
/// being sorted.
///
/// A plain linear insertion: for parts this short, it is 1.08–1.18 times faster than this
/// crate's [`insertion`](fn@super::insertion) sort, which inserts elements in pairs with a
/// galloping search, for 64-byte elements, and 1.16 times faster for 256-byte ones.
fn insert_last<T: Ord>(v: &mut [T]) {
    let base = v.as_mut_ptr();
    let last = v.len() - 1;
    // SAFETY: `last >= 1`, since `sort_short` passes at least two elements, and the hole only
    // moves left while it is above `base`, so every pointer stays inside `v`. The last element
    // is compared in place, then read out, which leaves its slot vacant; each step moves the
    // element left of the vacant slot into it, so the vacant slot moves one to the left. The
    // held element is compared as the copy that gets written back, and if a comparison
    // panics, `hole` writes it into the vacant slot, so `v` holds every element exactly once.
    unsafe {
        if *base.add(last) >= *base.add(last - 1) {
            return;
        }
        let held = ManuallyDrop::new(ptr::read(base.add(last)));
        let mut hole = Hole {
            src: &*held,
            dst: base.add(last),
        };
        loop {
            ptr::copy_nonoverlapping(hole.dst.sub(1), hole.dst, 1);
            hole.dst = hole.dst.sub(1);
            if hole.dst == base || *held >= *hole.dst.sub(1) {
                break;
            }
        }
        // Dropping `hole` writes the held element into the vacant slot.
    }
}

/// Optimal sorting networks for 0 to 8 elements: applying [`compare_exchange`] to each pair in
/// order sorts any input of that length (1, 3, 5, 9, 12, 16 and 19 comparisons from 2 on).
const NETWORKS: [&[(usize, usize)]; SHORT + 1] = [
    &[],
    &[],
    &[(0, 1)],
    &[(0, 2), (0, 1), (1, 2)],
    &[(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)],
    &[
        (0, 3),
        (1, 4),
        (0, 2),
        (1, 3),
        (0, 1),
        (2, 4),
        (1, 2),
        (3, 4),
        (2, 3),
    ],
    &[
        (0, 5),
        (1, 3),
        (2, 4),
        (1, 2),
        (3, 4),
        (0, 3),
        (2, 5),
        (0, 1),
        (2, 3),
        (4, 5),
        (1, 2),
        (3, 4),
    ],
    &[
        (0, 6),
        (2, 3),
        (4, 5),
        (0, 2),
        (1, 4),
        (3, 6),
        (0, 1),
        (2, 5),
        (3, 4),
        (1, 2),
        (4, 6),
        (2, 3),
        (4, 5),
        (1, 2),
        (3, 4),
        (5, 6),
    ],
    &[
        (0, 2),
        (1, 3),
        (4, 6),
        (5, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (2, 4),
        (3, 5),
        (1, 4),
        (3, 6),
        (1, 2),
        (3, 4),
        (5, 6),
    ],
];

/// Swaps `v[i]` and `v[j]`, `i < j`, if `v[j] < v[i]`, without branching on the comparison.
#[inline(always)]
fn compare_exchange<T: Ord>(v: &mut [T], i: usize, j: usize) {
    assert!(i < j && j < v.len());
    let base = v.as_mut_ptr();
    // SAFETY: `i` and `j` are distinct indices into `v`. The comparison runs with both
    // elements in place, so a panic leaves `v` untouched. Then the element that belongs at `j`
    // is read out, the one that belongs at `i` is copied there (onto itself if no swap is
    // needed, which `ptr::copy` allows), and the element read out is written to `j`. Nothing in
    // between can panic.
    unsafe {
        let (a, b) = (base.add(i), base.add(j));
        let swap = *b < *a;
        let (to_a, to_b) = if swap { (b, a) } else { (a, b) };
        let tmp = ManuallyDrop::new(ptr::read(to_b));
        ptr::copy(to_a, a, 1);
        ptr::copy_nonoverlapping(&*tmp, b, 1);
    }
}

/// Sorts `v` by heap sort: `O(n log n)` comparisons on any input, in place.
fn heap_sort<T: Ord>(v: &mut [T]) {
    /// Moves `v[node]` down the max-heap `v` until neither child is larger.
    fn sift_down<T: Ord>(v: &mut [T], mut node: usize) {
        loop {
            let mut child = 2 * node + 1;
            if child >= v.len() {
                return;
            }
            if child + 1 < v.len() && v[child] < v[child + 1] {
                child += 1;
            }
            if v[child] <= v[node] {
                return;
            }
            v.swap(node, child);
            node = child;
        }
    }
    for node in (0..v.len() / 2).rev() {
        sift_down(v, node);
    }
    for end in (1..v.len()).rev() {
        v.swap(0, end);
        sift_down(&mut v[..end], 0);
    }
}

/// A vacant slot of a slice, plus the element held outside the slice that belongs there.
///
/// Writing the held element on drop is what keeps [`partition_lomuto`], [`partition_hoare`]
/// and [`insert_last`] panic-safe: the slice is whole again even if a comparison unwinds while
/// the element is out.
struct Hole<T> {
    src: *const T,
    dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to the single vacant slot of the slice being partitioned, and
        // `src` to the held element, which is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) }
    }
}

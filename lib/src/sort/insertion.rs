use core::ptr;

/// Sorts a slice in place using pair insertion sort with galloping search.
///
/// A sorted prefix grows from the left two elements at a time. The two new elements are
/// ordered with one comparison, the larger one's place in the prefix is searched for from
/// the right, and the smaller one's place is searched for only to the left of that, since it
/// cannot lie anywhere else. Both then move into place together. This is the pair insertion
/// sort that Java's dual-pivot quicksort uses for short ranges, with a galloping search in
/// place of its linear scan.
///
/// Three refinements keep the algorithm cheap:
///
/// * **Sorted fast path.** The smaller element of a pair is compared with the last element of
///   the prefix before any search. If it is not smaller, neither is the larger one, and the
///   pair is done after two comparisons, one per element: already-sorted input costs exactly
///   `n - 1` comparisons, as with plain insertion sort.
/// * **Galloping search.** An insertion point is found by probing 1, 2, 4, 8, ... slots to the
///   left of where the search starts, then binary searching the last gap. An element that
///   moves `d` slots left costs at most `2⌊log₂ d⌋ + 2` comparisons: logarithmic in how far it
///   moves rather than linear (a plain scan) or logarithmic in the length of the prefix (a
///   binary search), so nearly-sorted input stays nearly linear.
/// * **Pair moves.** Prefix elements greater than both new elements shift right by two slots
///   at once and those between them by one, and each new element is written once into its
///   final slot. A prefix element that both new elements pass therefore moves once instead of
///   twice. Both insertion points are found before anything moves, so no comparison ever
///   runs while an element is out of the slice.
///
/// Against inserting the same elements one at a time with the same galloping search, a pair
/// spends one comparison on ordering itself, and in exchange the smaller element's search
/// starts from the larger one's insertion point instead of from the end of the prefix, and
/// the prefix elements past that point move once instead of twice. On 1000 elements, counting
/// a move each time an element changes slot:
///
/// | Input (`n = 1000`)         | Comparisons, pair | one at a time | Moves, pair | one at a time |
/// |----------------------------|-------------------|---------------|-------------|---------------|
/// | Sorted                     | 999               | 999           | 0           | 0             |
/// | Random, distinct (mean)    | ≈ 14,580          | ≈ 14,920      | ≈ 167,400   | ≈ 250,600     |
/// | Reverse sorted             | 5,986             | 9,966         | 250,998     | 500,499       |
/// | Worst case, per column     | 17,020            | 17,169        | 250,999     | 500,499       |
///
/// On random input the two searches of a pair together cost slightly less than two single
/// ones, and a third of the moves are saved (`≈ n²/6` against `≈ n²/4`). On reverse-sorted
/// input both halve: the larger element goes to the front, which leaves the smaller one
/// nothing to search and every prefix element to pass only once.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order. A pair is reordered only when its
///   second element is strictly smaller, and each element is inserted after every prefix
///   element that is not greater than it.
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: comparisons grow with the logarithm of how far elements move, moves with
///   the distance itself.
///
/// Let `dₓ` be the number of elements before `x` in the input that are greater than `x`, which
/// is how many slots `x` moves left, and `I = Σ dₓ` the number of inversions.
///
/// | Case                     | Comparisons                        | Moves                    |
/// |--------------------------|------------------------------------|--------------------------|
/// | Best (already sorted)    | `n - 1`                            | `0`                      |
/// | Any input                | `≤ n - 1 + Σₓ 2⌈log₂(dₓ + 1)⌉`     | `≤ I + n`                |
/// | Reverse sorted           | `≈ (n/2)·log₂ n + n`               | `≈ n²/4` (the worst)     |
/// | Worst                    | `≤ 2n·log₂ n` (`≈ 2n·log₂ n - 3n`) | `≈ n²/4`                 |
///
/// The adaptive bound is tight: input such as `[1, 0, 3, 2, 5, 4, ...]` comes within two
/// comparisons of it. By concavity it is below `3n + 2n·log₂(1 + I/n)`, so input with few
/// inversions sorts in close to linear time. The worst case needs every element to move a long
/// way, but not to the front, where a search is cut short. On reverse-sorted input, a pair
/// inserted into a prefix of `i` elements costs exactly `⌈log₂ i⌉ + 3` comparisons and
/// `i + 2` moves, the most a pair can move (only the first pair, whose larger element ends in
/// the slot it started in, moves two), and the last element when `n` is even costs
/// `⌈log₂(n - 1)⌉ + 1` comparisons and `n` moves.
///
/// Insertion sort moves `O(n²)` elements on random input and is mainly of educational value
/// beyond short or nearly-sorted slices; prefer [`slice::sort`] for real workloads.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: every comparison is made while all elements are in the slice, and
/// the moves that follow cannot panic.
///
/// # Examples
///
/// ```
/// use a::sort::insertion;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// insertion(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // Equal keys keep their order: sorting by the first field leaves 'a' before 'b'.
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
/// let mut cards = [Card(2, 'a'), Card(1, 'x'), Card(2, 'b'), Card(0, 'y')];
/// insertion(&mut cards);
/// assert_eq!(cards, [Card(0, 'y'), Card(1, 'x'), Card(2, 'a'), Card(2, 'b')]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// insertion(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn insertion<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    // `arr[..i]` is sorted. A lone first element is, so the rest is taken in pairs, with one
    // element left over when `len` is even.
    let mut i = 1;
    while i + 1 < len {
        insert_pair(&mut arr[..i + 2]);
        i += 2;
    }
    if i < len {
        insert_last(arr);
    }
}

/// Inserts the last two elements of `v` into the sorted rest of it, which is not empty.
fn insert_pair<T: Ord>(v: &mut [T]) {
    let i = v.len() - 2;
    // Order the pair. Only a second element strictly smaller than the first counts as out of
    // order, so equal elements keep their order.
    let swapped = v[i + 1] < v[i];
    let (s, l) = if swapped { (i + 1, i) } else { (i, i + 1) };

    // Fast path: the smaller element does not go before the last element of the prefix, so
    // the larger one does not either.
    if v[s] >= v[i - 1] {
        if swapped {
            v.swap(i, i + 1);
        }
        return;
    }

    let (prefix, pair) = v.split_at(i);
    let (small, large) = (&pair[s - i], &pair[l - i]);
    // Each element goes after every prefix element that is not greater than it, so it ends
    // up `d` slots to the left of where its search starts, `d` being the number of prefix
    // elements greater than it there.
    let pl = i - gallop(prefix, large, 0);
    // The smaller element cannot go past the larger one, so only `prefix[..pl]` is searched.
    // If the larger element stays at the end, the fast-path comparison has already shown
    // that the smaller one goes before `prefix[i - 1]`.
    let ps = pl - gallop(&prefix[..pl], small, usize::from(pl == i));

    if !swapped && pl == i {
        // The larger element is already in its final slot; only the smaller one moves. The
        // general path below would give the same result, at the cost of reading the larger
        // element out and writing it back.
        move_last_to_front(&mut v[ps..=i]);
        return;
    }
    let base = v.as_mut_ptr();
    // SAFETY: `ps <= pl <= i` and `i + 2 == v.len()`, so every pointer stays inside `v`.
    // Both pair elements are read out first, which leaves `v[i]` and `v[i + 1]` logically
    // uninitialised. Moving `v[pl..i]` two slots right into `v[pl + 2..i + 2]` fills those
    // two slots and vacates `v[pl]` and `v[pl + 1]`; moving `v[ps..pl]` one slot right into
    // `v[ps + 1..pl + 1]` fills `v[pl]` and vacates `v[ps]` (`ptr::copy` allows the overlap
    // both times). The two elements read out are then written into the two vacant slots,
    // `v[ps]` and `v[pl + 1]`. Every comparison has already been made and nothing in between
    // can panic, so `v` holds every element exactly once again when the block ends.
    unsafe {
        let small = ptr::read(base.add(s));
        let large = ptr::read(base.add(l));
        ptr::copy(base.add(pl), base.add(pl + 2), i - pl);
        ptr::copy(base.add(ps), base.add(ps + 1), pl - ps);
        ptr::write(base.add(ps), small);
        ptr::write(base.add(pl + 1), large);
    }
}

/// Inserts the last element of `v` into the sorted rest of it.
fn insert_last<T: Ord>(v: &mut [T]) {
    let i = v.len() - 1;
    let (prefix, last) = v.split_at(i);
    let d = gallop(prefix, &last[0], 0);
    if d > 0 {
        move_last_to_front(&mut v[i - d..]);
    }
}

/// Returns how many elements at the end of the sorted slice `v` are greater than `x`, given
/// that the last `known` of them are (`known` is at most `v.len()`).
///
/// Probes 1, 2, 4, ... slots from the end, skipping those already known, until it meets an
/// element that is not greater than `x` or reaches the front, then binary searches the gap
/// between the last two probes. An answer `d >= 1` costs at most `2⌊log₂ d⌋ + 2` comparisons,
/// one fewer when `known` is 1, and `d = 0` costs one.
///
/// The result never exceeds `v.len()`, whatever the [`Ord`] implementation answers.
fn gallop<T: Ord>(v: &[T], x: &T, known: usize) -> usize {
    let n = v.len();
    // The answer lies in `lo..hi`.
    let (mut lo, mut hi) = (known, n + 1);
    let mut k = 1;
    while k <= lo {
        k = k.saturating_mul(2);
    }
    loop {
        if k >= n {
            // The probe would reach the front or past it: probe the front element instead.
            if lo < n {
                if *x < v[0] {
                    lo = n;
                } else {
                    hi = n;
                }
            }
            break;
        }
        if *x < v[n - k] {
            lo = k;
            k = k.saturating_mul(2);
        } else {
            hi = k;
            break;
        }
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if *x < v[n - mid] {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Moves the last element of `v` to the front, shifting every other element one slot right.
fn move_last_to_front<T>(v: &mut [T]) {
    let Some(last) = v.len().checked_sub(1) else {
        return;
    };
    let base = v.as_mut_ptr();
    // SAFETY: `last < v.len()`, so every pointer stays inside `v`. Reading out `v[last]`
    // leaves that slot logically uninitialised; moving `v[..last]` one slot right into
    // `v[1..]` (`ptr::copy` allows the overlap) fills it and vacates `v[0]`, where the element
    // read out is written. Nothing in between can panic, so `v` holds every element exactly
    // once again when the block ends.
    unsafe {
        let x = ptr::read(base.add(last));
        ptr::copy(base, base.add(1), last);
        ptr::write(base, x);
    }
}

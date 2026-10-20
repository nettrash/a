use a::sort::merge;
use std::cell::Cell;
use std::cmp::Ordering;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Merge sort sorts runs of at most this many elements as a whole instead of splitting them.
const RUN: usize = 8;

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    merge(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    merge(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    merge(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    merge(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    merge(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    merge(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    merge(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        merge(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    merge(&mut v);
    assert_eq!(
        v,
        vec![
            "apple", "apple", "banana", "cherry", "date", "fig", "kiwi", "lime", "pear", "plum"
        ]
    );
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = (0..50).rev().map(|i| format!("{i:03}")).collect();
    merge(&mut v);
    assert_eq!(v, (0..50).map(|i| format!("{i:03}")).collect::<Vec<_>>());
}

#[test]
fn sorts_large_elements() {
    let mut v: Vec<[u64; 8]> = (0..100u64)
        .rev()
        .map(|k| [k, 1, 2, 3, 4, 5, 6, 7])
        .collect();
    merge(&mut v);
    assert!(v.iter().map(|e| e[0]).eq(0..100));
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        merge(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    let n = size(10_000, 300) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    merge(&mut v);
    assert_eq!(v, (0..n).collect::<Vec<_>>());
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).rev().collect();
    merge(&mut v[5..25]);
    let expected: Vec<u32> = (25..30).rev().chain(5..25).chain((0..5).rev()).collect();
    assert_eq!(v, expected);
}

/// Small deterministic xorshift generator, so the tests need no extra dependencies.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Shuffles `v` in place (Fisher–Yates).
fn shuffle<T>(v: &mut [T], state: &mut u64) {
    for i in (1..v.len()).rev() {
        v.swap(i, (xorshift(state) % (i as u64 + 1)) as usize);
    }
}

/// Calls `f` with every permutation of `0..n` (Heap's algorithm).
fn for_each_permutation(n: usize, f: &mut impl FnMut(&[usize])) {
    fn go(k: usize, v: &mut [usize], f: &mut impl FnMut(&[usize])) {
        if k <= 1 {
            f(v);
            return;
        }
        for i in 0..k - 1 {
            go(k - 1, v, f);
            if k % 2 == 1 {
                v.swap(0, k - 1);
            } else {
                v.swap(i, k - 1);
            }
        }
        go(k - 1, v, f);
    }
    let mut v: Vec<usize> = (0..n).collect();
    go(n, &mut v, f);
}

/// Calls `f` with every sequence of length `len` over the keys `0..keys`.
fn for_each_sequence(len: u32, keys: u8, f: &mut impl FnMut(&[u8])) {
    for code in 0..(keys as usize).pow(len) {
        let mut c = code;
        let v: Vec<u8> = (0..len)
            .map(|_| {
                let d = (c % keys as usize) as u8;
                c /= keys as usize;
                d
            })
            .collect();
        f(&v);
    }
}

/// Calls `f` with every permutation up to length 8, every sequence over three keys up to
/// length 11, and random sequences of distinct and of repeated keys up to length 300, sorted,
/// reversed and shuffled.
fn for_each_test_input(seed: u64, f: &mut impl FnMut(&[usize])) {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, f);
    }
    for len in 0..=size(11, 9) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            let s: Vec<usize> = s.iter().map(|&k| k as usize).collect();
            f(&s);
        });
    }
    let mut state = seed;
    for n in [9, 16, 17, 33, 100, size(300, 40)] {
        for _ in 0..size(20, 2) {
            let mut v: Vec<usize> = (0..n).collect();
            f(&v);
            v.reverse();
            f(&v);
            shuffle(&mut v, &mut state);
            f(&v);
            let v: Vec<usize> = (0..n).map(|_| xorshift(&mut state) as usize % 3).collect();
            f(&v);
        }
    }
}

/// An element compared by `key` only, carrying the index it started at.
#[derive(Debug, Clone, Copy)]
struct ByKey<K> {
    key: K,
    idx: usize,
}
impl<K: Ord> PartialEq for ByKey<K> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord> Eq for ByKey<K> {}
impl<K: Ord> PartialOrd for ByKey<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord> Ord for ByKey<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys`, each tagged with its index, and checks the result against std's stable sort:
/// the same keys in the same order, and equal keys still in their original order.
fn assert_sorts_like_std_stable<K: Ord + Copy + std::fmt::Debug>(keys: &[K]) {
    let mut expected: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|e| e.0);
    let mut actual: Vec<ByKey<K>> = keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| ByKey { key, idx })
        .collect();
    merge(&mut actual);
    let actual: Vec<(K, usize)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert_eq!(actual, expected, "input: {keys:?}");
}

#[test]
fn is_stable_for_equal_keys() {
    // Equal keys met at the front and at the back of a merge, in the odd middle slot, and
    // across halves that are swapped as blocks.
    assert_sorts_like_std_stable(&[2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    assert_sorts_like_std_stable(&[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]);
    assert_sorts_like_std_stable(&[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_sorts_like_std_stable(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]);
}

#[test]
fn sorts_every_permutation_up_to_8_stably() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts_like_std_stable(p));
    }
}

#[test]
fn sorts_every_sequence_over_few_keys_stably() {
    // Few keys put equal elements on both sides of every merge. Lengths from 9 on take the
    // merge path; up to 16 over two keys, every split point of every merge is covered.
    for len in 0..=size(11, 9) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_like_std_stable(s));
    }
    for len in 0..=size(16, 10) as u32 {
        for_each_sequence(len, 2, &mut |s| assert_sorts_like_std_stable(s));
    }
}

#[test]
fn matches_std_stable_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(300, 40) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(4, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts_like_std_stable(&keys);
            }
        }
    }
}

/// A key that counts every comparison made between two of them.
#[derive(Debug, Clone)]
struct Counted<'a, K> {
    key: K,
    comparisons: &'a Cell<usize>,
}
impl<K: Ord> PartialEq for Counted<'_, K> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord> Eq for Counted<'_, K> {}
impl<K: Ord> PartialOrd for Counted<'_, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord> Ord for Counted<'_, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.comparisons.set(self.comparisons.get() + 1);
        self.key.cmp(&other.key)
    }
}

/// Sorts a copy of `keys` with `sort`, checks the result, and returns the number of
/// comparisons.
fn comparisons_of<K: Ord + Clone + std::fmt::Debug>(
    keys: &[K],
    sort: impl FnOnce(&mut [Counted<K>]),
) -> usize {
    let count = Cell::new(0);
    let mut v: Vec<Counted<K>> = keys
        .iter()
        .map(|k| Counted {
            key: k.clone(),
            comparisons: &count,
        })
        .collect();
    sort(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

fn comparisons<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> usize {
    comparisons_of(keys, |v| merge(v))
}

/// The comparisons the documented algorithm makes on `keys`. A run of at most 8 elements is
/// scanned up to its first descent (all of it, `len − 1` comparisons, if it is sorted); an
/// unsorted one then costs 0, 1, 3 or 5 comparisons for each half of 1, 2, 3 or 4 elements,
/// plus `2⌊len/2⌋` for merging them, or just the cost of sorting it whole if it has at most
/// 4 elements. Every merge of `m` elements costs one comparison if its halves are in order,
/// two if they are reversed, and `2 + 2⌊m/2⌋` otherwise.
fn model(keys: &[usize]) -> usize {
    fn go(v: &mut [usize]) -> usize {
        let len = v.len();
        if len <= RUN {
            let small = |n: usize| [0, 0, 1, 3, 5][n];
            let descent = (1..len).find(|&i| v[i] < v[i - 1]);
            let Some(scanned) = descent else {
                return len.saturating_sub(1);
            };
            v.sort();
            return scanned
                + if len <= 4 {
                    small(len)
                } else {
                    small(len / 2) + small(len - len / 2) + 2 * (len / 2)
                };
        }
        let mid = len / 2;
        let c = go(&mut v[..mid]) + go(&mut v[mid..]);
        if v[mid] >= v[mid - 1] {
            return c + 1;
        }
        if v[len - 1] < v[0] {
            v.rotate_left(mid);
            return c + 2;
        }
        v.sort();
        c + 2 + 2 * (len / 2)
    }
    go(&mut keys.to_vec())
}

/// The model counts a merge of `m` elements as `2⌊m/2⌋` comparisons, which holds exactly for
/// elements of at most 8 bytes; `Counted` is 16 bytes, and a merge of larger elements stops
/// comparing as soon as a half has run out, so the model is an upper bound for it.
#[test]
fn comparisons_match_the_model() {
    for_each_test_input(0xC0C7_A11E_5AAE_0001, &mut |v| {
        let (counted, modelled) = (comparisons(v), model(v));
        assert!(counted <= modelled, "{counted} > {modelled}: {v:?}");
        // Halves that run out early save at most one comparison per element per merge level.
        let levels = v.len().max(1).next_power_of_two().trailing_zeros() as usize + 1;
        assert!(
            counted + v.len() * levels >= modelled,
            "{counted} vs {modelled}: {v:?}"
        );
    });
}

#[test]
fn sorted_input_costs_n_minus_1_comparisons() {
    for n in 0..size(1000, 100) {
        let expected = n.saturating_sub(1);
        assert_eq!(
            comparisons(&(0..n).collect::<Vec<_>>()),
            expected,
            "n = {n}"
        );
        assert_eq!(comparisons(&vec![7; n]), expected, "n = {n}");
    }
}

#[test]
fn reverse_sorted_input_costs_about_2_6n_comparisons() {
    let n = size(1000, 100);
    let v: Vec<usize> = (0..n).rev().collect();
    let c = comparisons(&v);
    assert_eq!(c, model(&v));
    if !cfg!(miri) {
        assert_eq!(c, 2590);
    }
    assert!(c <= 3 * n, "{c} comparisons");
}

#[test]
fn random_input_costs_about_n_log2_n_comparisons() {
    // At most `n·log₂ n` on random input in practice: about 9,450 for `n = 1000`.
    let n = size(1000, 100);
    let bound = n as f64 * (n as f64).log2();
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    for _ in 0..size(20, 1) {
        let mut v: Vec<usize> = (0..n).collect();
        shuffle(&mut v, &mut state);
        let c = comparisons(&v);
        assert!((c as f64) < bound, "{c} comparisons, bound {bound}");
    }
}

/// An element that notices when the sort has moved it: every comparison checks its address
/// against the one it was last seen at. Only slots of the slice count, so the count is a lower
/// bound on the moves made; zero means every element sat at its starting address whenever it
/// was compared in the slice, and is still there at the end.
struct Watched<'a, K> {
    key: K,
    seen_at: Cell<usize>,
    moves: &'a Cell<usize>,
    slots: &'a Cell<(usize, usize)>,
}

impl<K> Watched<'_, K> {
    fn check(&self) {
        let here = self as *const Self as usize;
        let (start, end) = self.slots.get();
        if !(start..end).contains(&here) {
            return;
        }
        if self.seen_at.replace(here) != here {
            self.moves.set(self.moves.get() + 1);
        }
    }
}

impl<K: Ord> PartialEq for Watched<'_, K> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord> Eq for Watched<'_, K> {}
impl<K: Ord> PartialOrd for Watched<'_, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord> Ord for Watched<'_, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.check();
        other.check();
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys` and returns the number of element moves observed.
fn observed_moves<K: Ord + Copy + std::fmt::Debug>(keys: &[K]) -> usize {
    let moves = Cell::new(0);
    let slots = Cell::new((0, 0));
    let mut v: Vec<Watched<K>> = keys
        .iter()
        .map(|&key| Watched {
            key,
            seen_at: Cell::new(0),
            moves: &moves,
            slots: &slots,
        })
        .collect();
    let start = v.as_ptr() as usize;
    slots.set((start, start + std::mem::size_of_val(v.as_slice())));
    for w in &v {
        w.seen_at.set(w as *const Watched<K> as usize);
    }
    merge(&mut v);
    for w in &v {
        w.check();
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moves.get()
}

#[test]
fn non_decreasing_input_is_never_moved() {
    // Every non-decreasing sequence over four keys up to length 24 (as cut points between
    // runs), plus long ones: no merge is ever needed.
    let check = |v: &[u8]| assert_eq!(observed_moves(v), 0, "{v:?}");
    for len in 0..=size(24, 12) {
        for a in 0..=len {
            for b in a..=len {
                for c in b..=len {
                    let v: Vec<u8> = (0..len)
                        .map(|i| (i >= a) as u8 + (i >= b) as u8 + (i >= c) as u8)
                        .collect();
                    check(&v);
                }
            }
        }
    }
    check(&(0..=255).collect::<Vec<u8>>());
}

/// Sorts `v` and returns how many allocations that made.
fn allocations_of<T: Ord>(v: &mut [T]) -> usize {
    crate::allocations::count(|| merge(v))
}

#[test]
fn allocates_one_buffer_and_only_beyond_4_kib() {
    // 512 `u64`s fill the 4 KiB stack buffer.
    let mut state = 0x0A11_0C47_E5B0_FFE5;
    let ns = if cfg!(miri) {
        vec![0, 1, 9, 100, 511, 512, 513, 600]
    } else {
        (0..700).collect()
    };
    for n in ns {
        let mut v: Vec<u64> = (0..n).map(|_| xorshift(&mut state)).collect();
        assert_eq!(allocations_of(&mut v), usize::from(n > 512), "n = {n}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}

#[test]
fn short_slices_of_large_elements_do_not_allocate() {
    // Four 1 KiB elements fill the stack buffer; up to 8 are sorted in place without it.
    for n in 0..=RUN + 4 {
        let mut v: Vec<[u64; 128]> = (0..n as u64).rev().map(|k| [k; 128]).collect();
        assert_eq!(allocations_of(&mut v), usize::from(n > RUN), "n = {n}");
        assert!(v.iter().map(|e| e[0]).eq(0..n as u64));
    }
}

/// Sorts `keys` with comparisons that start panicking after `budget` of them, and checks that
/// no element is lost, duplicated, dropped or leaked. Returns whether the sort completed.
fn sort_with_comparison_budget(keys: &[u32], budget: usize) -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    struct Tracked<'a> {
        key: u32,
        id: usize,
        drops: Rc<Vec<Cell<u32>>>,
        comparisons_left: &'a Cell<usize>,
    }
    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            let count = &self.drops[self.id];
            count.set(count.get() + 1);
        }
    }
    impl PartialEq for Tracked<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl Eq for Tracked<'_> {}
    impl PartialOrd for Tracked<'_> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Tracked<'_> {
        fn cmp(&self, other: &Self) -> Ordering {
            let left = self.comparisons_left.get();
            if left == 0 {
                panic!("comparison budget exhausted");
            }
            self.comparisons_left.set(left - 1);
            self.key.cmp(&other.key)
        }
    }

    let len = keys.len();
    let drops = Rc::new((0..len).map(|_| Cell::new(0)).collect::<Vec<_>>());
    let comparisons_left = Cell::new(budget);
    let mut v: Vec<Tracked> = keys
        .iter()
        .enumerate()
        .map(|(id, &key)| Tracked {
            key,
            id,
            drops: Rc::clone(&drops),
            comparisons_left: &comparisons_left,
        })
        .collect();

    let result = catch_unwind(AssertUnwindSafe(|| merge(&mut v)));

    if result.is_ok() {
        // Sorted and stable: equal keys keep their ids in increasing order.
        assert!(
            v.windows(2)
                .all(|w| (w[0].key, w[0].id) < (w[1].key, w[1].id))
        );
    }
    let mut ids: Vec<usize> = v.iter().map(|t| t.id).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..len).collect::<Vec<_>>(),
        "element lost or duplicated"
    );
    assert!(
        v.iter().all(|t| t.key == keys[t.id]),
        "element changed during sort"
    );
    assert!(
        drops.iter().all(|d| d.get() == 0),
        "element dropped during sort"
    );

    drop(v);
    assert!(
        drops.iter().all(|d| d.get() == 1),
        "element leaked or double-dropped"
    );
    result.is_ok()
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    let len = size(45, 19);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    // Panics land in insertion-sorted runs, in both checks of every merge, and at every
    // step of the bidirectional merges: random and repeated keys merge at every level,
    // reversed input swaps halves, and odd lengths have a middle slot.
    let layouts: Vec<Vec<u32>> = vec![
        (0..len as u32).rev().collect(),
        (0..len as u32).map(|id| id % 3).collect(),
        (0..len as u32).map(|id| (len as u32 - id) / 3).collect(),
        random,
        (0..len as u32).map(|id| id ^ 1).collect(),
        (1..len as u32).chain([0]).collect(),
    ];
    for keys in layouts {
        let total = comparisons(&keys);
        for budget in 0..=total {
            let completed = sort_with_comparison_budget(&keys, budget);
            assert_eq!(completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that is not a total order can make the two ends of a merge take the same
    // element; the merge must notice and put the halves back. The order of the result is
    // unspecified, but no element may be lost or duplicated, no index may go out of bounds,
    // and the sort must terminate.
    struct Liar<'a> {
        name: String,
        answer: &'a dyn Fn() -> Ordering,
    }
    impl PartialEq for Liar<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl Eq for Liar<'_> {}
    impl PartialOrd for Liar<'_> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Liar<'_> {
        fn cmp(&self, _: &Self) -> Ordering {
            (self.answer)()
        }
    }

    let rng = Cell::new(0x5EED_5EED_5EED_5EED_u64);
    let random = || {
        let mut state = rng.get();
        let r = xorshift(&mut state);
        rng.set(state);
        [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize]
    };
    let answers: [&dyn Fn() -> Ordering; 4] = [
        &|| Ordering::Less,
        &|| Ordering::Greater,
        &|| Ordering::Equal,
        &random,
    ];
    for answer in answers {
        for len in 0..size(80, 25) {
            for _ in 0..size(10, 2) {
                let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
                let mut v: Vec<Liar> = names
                    .iter()
                    .map(|name| Liar {
                        name: name.clone(),
                        answer,
                    })
                    .collect();

                merge(&mut v);

                let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
                after.sort_unstable();
                let mut names = names;
                names.sort_unstable();
                assert_eq!(after, names);
            }
        }
    }
}

thread_local! {
    /// Comparisons made by the elements below, which are too small to carry a counter.
    static MUTATING_COMPARISONS: Cell<usize> = const { Cell::new(0) };
}

/// Whether the elements below fail the test when the sort compares an element with itself.
/// The pivot selection still does (the ninther's sample groups share elements, and the pivot
/// may be the ancestor it is compared with), so until that is fixed such a comparison only
/// counts as two mutations of the one element.
const FORBID_SELF_COMPARISON: bool = true;

/// An 8-byte element whose comparison mutates both of its arguments through interior
/// mutability and moves heap ownership while doing so: it takes the boxes out of both cells,
/// drops them, and puts fresh ones back with the comparison counted. A bitwise copy of an
/// element made before a later comparison of its twin is stale, and dropping it frees memory
/// twice (the standard library's `Freeze` rule). Small enough for the paths reserved for
/// elements of at most 8 bytes.
struct Swapping(Cell<Option<Box<(u32, u32, u32)>>>);

impl Swapping {
    fn new(key: u32, idx: u32) -> Self {
        Self(Cell::new(Some(Box::new((key, idx, 0)))))
    }
    /// The key, the starting index and the number of comparisons taken part in.
    fn get(&self) -> (u32, u32, u32) {
        let b = self.0.take().expect("box present");
        let t = *b;
        self.0.set(Some(b));
        t
    }
}
impl PartialEq for Swapping {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Swapping {}
impl PartialOrd for Swapping {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Swapping {
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(
            !FORBID_SELF_COMPARISON || !std::ptr::eq(self, other),
            "an element compared with itself"
        );
        MUTATING_COMPARISONS.with(|c| c.set(c.get() + 1));
        let a = self
            .0
            .take()
            .expect("box present: a stale copy was compared");
        if std::ptr::eq(self, other) {
            self.0.set(Some(Box::new((a.0, a.1, a.2 + 2))));
            return Ordering::Equal;
        }
        let b = other
            .0
            .take()
            .expect("box present: a stale copy was compared");
        let r = a.0.cmp(&b.0);
        self.0.set(Some(Box::new((a.0, a.1, a.2 + 1))));
        other.0.set(Some(Box::new((b.0, b.1, b.2 + 1))));
        r
    }
}

/// A 40-byte element: every comparison replaces the log of both arguments by a new, longer
/// allocation, freeing the old one.
struct Reallocating {
    key: u32,
    idx: u32,
    log: std::cell::RefCell<Vec<u8>>,
}

impl Reallocating {
    fn touch(&self) {
        let mut log = self.log.borrow_mut();
        let mut fresh = Vec::with_capacity(log.len() + 1);
        fresh.extend_from_slice(&log);
        fresh.push(1);
        *log = fresh;
    }
}
impl PartialEq for Reallocating {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Reallocating {}
impl PartialOrd for Reallocating {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Reallocating {
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(
            !FORBID_SELF_COMPARISON || !std::ptr::eq(self, other),
            "an element compared with itself"
        );
        MUTATING_COMPARISONS.with(|c| c.set(c.get() + 1));
        self.touch();
        other.touch();
        self.key.cmp(&other.key)
    }
}

#[test]
fn mutating_comparison_keeps_every_mutation() {
    // Every comparison mutates both arguments, so the sum of the elements' mutation counts
    // must be twice the number of comparisons: a bitwise copy kept after its twin was compared
    // again would carry fewer, and would free the twin's old allocation a second time (which
    // Miri reports). Lengths cover the short-slice paths, the eager chunks and quick sort's
    // leaves for both element sizes, and few keys put equal elements in every merge and
    // partition.
    let mut state = 0x5A7E_C0DE_5A7E_C0DE;
    let comparisons = || MUTATING_COMPARISONS.with(|c| c.replace(0));
    for len in (0..=size(70, 40)).chain([size(150, 60), size(1000, 100)]) {
        for modulus in [2u64, 5, 1000] {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % modulus) as u32)
                .collect();

            let mut v: Vec<Swapping> = keys
                .iter()
                .zip(0..)
                .map(|(&k, i)| Swapping::new(k, i))
                .collect();
            comparisons();
            merge(&mut v);
            let c = comparisons();
            let got: Vec<(u32, u32, u32)> = v.iter().map(Swapping::get).collect();
            assert!(
                got.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)),
                "8-byte, len {len}: {keys:?}"
            );
            let mutations: usize = got.iter().map(|t| t.2 as usize).sum();
            assert_eq!(mutations, 2 * c, "8-byte, len {len}: a stale copy was kept");

            let mut v: Vec<Reallocating> = keys
                .iter()
                .zip(0..)
                .map(|(&key, idx)| Reallocating {
                    key,
                    idx,
                    log: std::cell::RefCell::new(Vec::new()),
                })
                .collect();
            comparisons();
            merge(&mut v);
            let c = comparisons();
            assert!(
                v.windows(2)
                    .all(|w| (w[0].key, w[0].idx) < (w[1].key, w[1].idx)),
                "40-byte, len {len}: {keys:?}"
            );
            let mutations: usize = v.iter().map(|e| e.log.borrow().len()).sum();
            assert_eq!(
                mutations,
                2 * c,
                "40-byte, len {len}: a stale copy was kept"
            );
        }
    }
}

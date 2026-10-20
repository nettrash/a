use a::sort::tim;
use std::cell::Cell;
use std::cmp::Ordering;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Tim sort sorts slices shorter than this by insertion alone; longer ones are split into runs
/// of at least half this length, which are merged.
const MIN_MERGE: usize = 32;

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    tim(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    tim(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    tim(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    tim(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    tim(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    tim(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    tim(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..80 {
        let mut v = vec![7; len];
        tim(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    tim(&mut v);
    assert_eq!(
        v,
        vec![
            "apple", "apple", "banana", "cherry", "date", "fig", "kiwi", "lime", "pear", "plum"
        ]
    );
}

#[test]
fn sorts_heap_owning_elements() {
    let mut state = 0x5EED_0F57_4712_6500;
    let mut v: Vec<String> = (0..200)
        .map(|_| format!("{:03}", xorshift(&mut state) % 1000))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    tim(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x1A26_E000_E1E3_E275;
    let mut v: Vec<[u64; 8]> = (0..300)
        .map(|_| [xorshift(&mut state) % 100, 1, 2, 3, 4, 5, 6, 7])
        .collect();
    let mut expected = v.clone();
    expected.sort();
    tim(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..80 {
        let mut v = vec![(); len];
        tim(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_random_input() {
    let n = size(100_000, 2000);
    let mut state = 0x7105_0127_0000_0001;
    let mut v: Vec<u64> = (0..n).map(|_| xorshift(&mut state)).collect();
    let mut expected = v.clone();
    expected.sort();
    tim(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_long_reverse_sorted() {
    let n = size(10_000, 300) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    tim(&mut v);
    assert_eq!(v, (0..n).collect::<Vec<_>>());
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    tim(&mut v[5..95]);
    let expected: Vec<u32> = (95..100).rev().chain(5..95).chain((0..5).rev()).collect();
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

/// Returns `n` keys made of runs: ascending, or descending (strictly or not), each of a random
/// length up to `max_run`, with keys below `keys`. Runs of many lengths make the merge policy
/// merge runs of many different lengths, and few keys put equal elements on both sides of
/// most merges.
fn runs_of_keys(n: usize, max_run: usize, keys: u64, state: &mut u64) -> Vec<u64> {
    let mut v = Vec::with_capacity(n);
    while v.len() < n {
        let len = (1 + xorshift(state) as usize % max_run).min(n - v.len());
        let mut run: Vec<u64> = (0..len).map(|_| xorshift(state) % keys).collect();
        run.sort();
        match xorshift(state) % 3 {
            0 => run.reverse(),
            1 => {
                run.dedup();
                run.reverse();
            }
            _ => {}
        }
        v.extend(run);
    }
    v
}

/// Returns two ascending runs of `n / 2` and `n - n / 2` elements whose keys interleave in
/// blocks of `block`: merging them makes one run win `block` times in a row, over and over,
/// at both ends of the merge, which is what sets off galloping.
fn interleaved_blocks(n: usize, block: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).filter(|k| (k / block).is_multiple_of(2)).collect();
    v.extend((0..n).filter(|k| (k / block) % 2 == 1));
    v
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
    tim(&mut actual);
    let actual: Vec<(K, usize)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert_eq!(actual, expected, "input: {keys:?}");
}

#[test]
fn is_stable_for_equal_keys() {
    // Equal keys in a pair, in a non-strictly descending run (which must not be reversed as a
    // whole), and on both sides of merges.
    assert_sorts_like_std_stable(&[2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    assert_sorts_like_std_stable(&[3, 3, 2, 2, 1, 1, 0, 0]);
    let keys: Vec<u8> = (0..100).map(|i| (i % 7 == 0) as u8).collect();
    assert_sorts_like_std_stable(&keys);
    let keys: Vec<u8> = (0..200).map(|i| (i / 50 % 2) as u8).collect();
    assert_sorts_like_std_stable(&keys);
}

/// An 8-byte element compared by `key` only, carrying the index it started at: small enough
/// for short runs to be built by merge sort's short-run sort instead of by insertion.
#[derive(Debug, Clone, Copy)]
struct Small {
    key: u32,
    idx: u32,
}
impl PartialEq for Small {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}
impl Eq for Small {}
impl PartialOrd for Small {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Small {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Like [`assert_sorts_like_std_stable`], with 8-byte elements.
fn assert_sorts_small_like_std_stable(keys: &[u32]) {
    assert_eq!(std::mem::size_of::<Small>(), 8);
    let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|e| e.0);
    let mut actual: Vec<Small> = keys
        .iter()
        .zip(0..)
        .map(|(&key, idx)| Small { key, idx })
        .collect();
    tim(&mut actual);
    let actual: Vec<(u32, u32)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert_eq!(actual, expected, "input: {keys:?}");
}

#[test]
fn sorts_small_elements_stably() {
    for len in 0..=size(10, 7) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            let keys: Vec<u32> = s.iter().map(|&k| k.into()).collect();
            assert_sorts_small_like_std_stable(&keys);
        });
    }
    let mut state = 0x5A11_E1E3_E275_0008;
    for len in (0..size(300, 80)).chain([size(1000, 200), size(5000, 300)]) {
        for modulus in [2, 3, 8, 64, 1 << 32] {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % modulus) as u32)
                .collect();
            assert_sorts_small_like_std_stable(&keys);
        }
    }
    for _ in 0..size(100, 10) {
        let n = xorshift(&mut state) as usize % size(3000, 300);
        let keys: Vec<u32> = runs_of_keys(n, 1 + n / 8, 40, &mut state)
            .iter()
            .map(|&k| k as u32)
            .collect();
        assert_sorts_small_like_std_stable(&keys);
    }
}

#[test]
fn sorts_every_permutation_up_to_8_stably() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts_like_std_stable(p));
    }
}

#[test]
fn sorts_every_sequence_over_few_keys_stably() {
    // Every way to insert pairs and single elements among equal ones.
    for len in 0..=size(11, 8) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_like_std_stable(s));
    }
    for len in 0..=size(16, 10) as u32 {
        for_each_sequence(len, 2, &mut |s| assert_sorts_like_std_stable(s));
    }
}

#[test]
fn matches_std_stable_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in (0..size(300, 80)).chain([size(1000, 200), size(5000, 300)]) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(3, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts_like_std_stable(&keys);
            }
        }
    }
}

#[test]
fn matches_std_stable_sort_on_runs() {
    let mut state = 0x0F0F_1234_5678_9ABC;
    for _ in 0..size(300, 20) {
        let n = xorshift(&mut state) as usize % size(3000, 300);
        let max_run = 1 + xorshift(&mut state) as usize % 400;
        let keys = 1 + xorshift(&mut state) % 600;
        assert_sorts_like_std_stable(&runs_of_keys(n, max_run, keys, &mut state));
    }
}

#[test]
fn matches_std_stable_sort_on_interleaved_blocks() {
    // Galloping at both ends of a merge, leaving and entering galloping mode, and runs that
    // run out while galloping.
    for n in [MIN_MERGE * 2, 100, 333, size(2000, 400)] {
        for block in [1, 2, 6, 7, 8, 20, 64, 150] {
            let mut keys = interleaved_blocks(n, block);
            assert_sorts_like_std_stable(&keys);
            // Equal keys across the two runs.
            for k in &mut keys {
                *k /= 3;
            }
            assert_sorts_like_std_stable(&keys);
        }
    }
}

#[test]
fn matches_std_stable_sort_on_sorted_input_with_extras() {
    let mut state = 0xA99E_7D00_0000_0001;
    for n in [MIN_MERGE, 100, 1000, size(10_000, 500)] {
        for extra in [1, 2, 5, n / 10] {
            let mut keys: Vec<u64> = (0..(n - extra) as u64).map(|k| k * 10).collect();
            keys.extend((0..extra).map(|_| xorshift(&mut state) % (10 * n as u64)));
            assert_sorts_like_std_stable(&keys);
            keys.rotate_right(extra);
            assert_sorts_like_std_stable(&keys);
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

/// Sorts a copy of `keys`, checks the result, and returns the number of comparisons.
fn comparisons<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> usize {
    let count = Cell::new(0);
    let mut v: Vec<Counted<K>> = keys
        .iter()
        .map(|k| Counted {
            key: k.clone(),
            comparisons: &count,
        })
        .collect();
    tim(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
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
fn strictly_descending_input_costs_n_minus_1_comparisons() {
    for n in 0..size(1000, 100) {
        let v: Vec<usize> = (0..n).rev().collect();
        assert_eq!(comparisons(&v), n.saturating_sub(1), "n = {n}");
    }
}

#[test]
fn random_input_costs_about_n_log2_n_comparisons() {
    // Insertion into runs of 16 to 32 elements costs more than a binary search would, so the
    // total is somewhat above `n·log₂ n`: about 1.17 times for `n = 1000`, 1.06 for 100,000.
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    for n in [size(1000, 100), size(100_000, 1000)] {
        let bound = 1.25 * n as f64 * (n as f64).log2();
        for _ in 0..size(5, 1) {
            let mut v: Vec<usize> = (0..n).collect();
            shuffle(&mut v, &mut state);
            let c = comparisons(&v);
            assert!((c as f64) < bound, "{c} comparisons, bound {bound}");
        }
    }
}

#[test]
fn galloping_merges_interleaved_blocks_in_few_comparisons() {
    // Two runs of 2048 cost 4095 comparisons to find; merging them one element at a time
    // would cost about 4096 more, galloping through the 64 blocks of 64 costs about 13 per
    // block.
    let n = 4096;
    let c = comparisons(&interleaved_blocks(n, 64));
    assert!(c < n + n / 4, "{c} comparisons");
}

#[test]
fn appended_elements_cost_few_comparisons() {
    // A sorted slice with `k` random elements appended: about `n` comparisons to find the
    // runs and `O(k log n)` more to place the extras, against `n·log₂ n` for random input.
    let mut state = 0xAFFE_AFFE_AFFE_AFFE;
    let n = size(10_000, 1000);
    for k in [1, 10, 100] {
        let mut v: Vec<u64> = (0..(n - k) as u64).map(|x| x * 1000).collect();
        v.extend((0..k).map(|_| xorshift(&mut state) % (1000 * n as u64)));
        let c = comparisons(&v);
        let bound = n + 40 * k * (n as f64).log2() as usize;
        assert!(c <= bound, "k = {k}: {c} comparisons, bound {bound}");
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
    tim(&mut v);
    for w in &v {
        w.check();
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moves.get()
}

#[test]
fn non_decreasing_input_is_never_moved() {
    // Every non-decreasing sequence over four keys up to length 40 (as cut points between
    // runs of equal keys), plus a long one made of runs of 20.
    let check = |v: &[u8]| assert_eq!(observed_moves(v), 0, "{v:?}");
    for len in 0..=size(40, 12) {
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
    check(&(0..=255).flat_map(|k| [k; 20]).collect::<Vec<u8>>());
}

/// Sorts `v` and returns how many allocations that made.
fn allocations_of<T: Ord>(v: &mut [T]) -> usize {
    crate::allocations::count(|| tim(v))
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
fn a_single_run_never_allocates() {
    // Sorted and strictly descending input is one run: nothing is merged.
    let n = size(100_000, 2000) as u64;
    let mut v: Vec<u64> = (0..n).collect();
    assert_eq!(allocations_of(&mut v), 0);
    let mut v: Vec<u64> = (0..n).rev().collect();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.iter().copied().eq(0..n));
}

#[test]
fn short_slices_of_large_elements_do_not_allocate() {
    // Slices shorter than `MIN_MERGE` are sorted by insertion, in place, however large their
    // elements; four 1 KiB elements would fill the stack buffer.
    for n in 0..=MIN_MERGE + 4 {
        let mut v: Vec<[u64; 128]> = (0..n as u64).map(|k| [k * 7 % 11; 128]).collect();
        assert_eq!(
            allocations_of(&mut v),
            usize::from(n >= MIN_MERGE),
            "n = {n}"
        );
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
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

    let result = catch_unwind(AssertUnwindSafe(|| tim(&mut v)));

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
    let mut state = 0x0123_4567_89AB_CDEF;
    let short = size(31, 13);
    let long = size(130, 70);
    let random = |len: usize, state: &mut u64| -> Vec<u32> {
        (0..len).map(|_| xorshift(state) as u32 % 7).collect()
    };
    // Panics land in pair and single insertions, while finding and reversing runs, in the
    // searches that trim a merge, at the front and the back of merges, while galloping at
    // either end, and in the search that places a last element.
    let layouts: Vec<Vec<u32>> = vec![
        random(short, &mut state),
        (0..short as u32).rev().collect(),
        random(long, &mut state),
        (0..long as u32).map(|id| id % 3).collect(),
        (0..long as u32).map(|id| (long as u32 - id) / 3).collect(),
        interleaved_blocks(long, 9)
            .iter()
            .map(|&k| k as u32)
            .collect(),
        interleaved_blocks(long, 1)
            .iter()
            .map(|&k| k as u32)
            .collect(),
        (1..long as u32).chain([0]).collect(),
        runs_of_keys(long, 40, 20, &mut state)
            .iter()
            .map(|&k| k as u32)
            .collect(),
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
fn panicking_comparison_keeps_every_element_exactly_once_in_split_merges() {
    // Runs of random content, whose merges fail to gallop: from the second such merge of a
    // sort on, a merge long enough is split in two and merged as four chains of comparisons
    // at once (see `tim.rs`), and a panic there must give back every element too. Natively
    // every budget is tried; under Miri every `stride`-th.
    let mut state = 0x5A17_2E4D_C0DE_F00D;
    let run = size(100, 50);
    for runs in [3, 8] {
        let mut keys: Vec<u32> = Vec::with_capacity(runs * run);
        for _ in 0..runs {
            let mut r: Vec<u32> = (0..run)
                .map(|_| xorshift(&mut state) as u32 % 1000)
                .collect();
            r.sort_unstable();
            keys.extend(r);
        }
        let total = comparisons(&keys);
        let stride = if cfg!(miri) { total / 40 + 1 } else { 1 };
        for budget in (0..total).step_by(stride).chain([total]) {
            let completed = sort_with_comparison_budget(&keys, budget);
            assert_eq!(completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that is not a total order can make a run look empty to a search or make a
    // merge run out of one run early. The order of the result is unspecified, but no element
    // may be lost or duplicated, no index may go out of bounds, and the sort must terminate.
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
    // Mostly one answer, sometimes another: long runs, then merges that gallop on lies.
    let skewed = || {
        let mut state = rng.get();
        let r = xorshift(&mut state);
        rng.set(state);
        if r.is_multiple_of(16) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    };
    let answers: [&dyn Fn() -> Ordering; 5] = [
        &|| Ordering::Less,
        &|| Ordering::Greater,
        &|| Ordering::Equal,
        &random,
        &skewed,
    ];
    for answer in answers {
        for len in (0..size(80, 40)).chain([size(500, 100)]) {
            for _ in 0..size(10, 2) {
                let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
                let mut v: Vec<Liar> = names
                    .iter()
                    .map(|name| Liar {
                        name: name.clone(),
                        answer,
                    })
                    .collect();

                tim(&mut v);

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
            tim(&mut v);
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
            tim(&mut v);
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

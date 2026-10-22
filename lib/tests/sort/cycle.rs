use a::sort::cycle;
use std::cell::Cell;
use std::cmp::Ordering;
use std::mem::size_of;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Slots the sort remembers as written at once (4 KiB of bits); longer slices open a new
/// window once their cycle starts pass the end of the first one.
const WINDOW: usize = 4096 * 8;

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    cycle(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    cycle(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    cycle(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    cycle(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    cycle(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    for n in [99, 100] {
        let mut v: Vec<u32> = (0..n).rev().collect();
        cycle(&mut v);
        assert_eq!(v, (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn sorts_non_strictly_descending() {
    // Not strictly descending, so it is sorted by cycles rather than reversed.
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    cycle(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    cycle(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        cycle(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    cycle(&mut v);
    assert_eq!(
        v,
        vec![
            "apple", "apple", "banana", "cherry", "date", "fig", "kiwi", "lime", "pear", "plum"
        ]
    );
}

#[test]
fn sorts_heap_owning_elements() {
    let mut state = 0x1EAF_5EED_1EAF_5EED;
    let mut v: Vec<String> = (0..size(200, 40))
        .map(|_| format!("{:03}", xorshift(&mut state) % 1000))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    cycle(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [size(30, 10), size(300, 40)] {
        let mut v: Vec<[u64; 32]> = (0..len).map(|_| [xorshift(&mut state) % 50; 32]).collect();
        let mut expected = v.clone();
        expected.sort();
        cycle(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        cycle(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    cycle(&mut v[5..25]);
    let mut middle = before[5..25].to_vec();
    middle.sort_unstable();
    assert_eq!(&v[..5], &before[..5]);
    assert_eq!(&v[5..25], &middle[..]);
    assert_eq!(&v[25..], &before[25..]);
}

#[test]
fn sorts_long_runs_of_equal_small_elements() {
    // Elements of up to 8 bytes skip long runs of equal keys 16 at a time; every type and
    // run length, including runs that end inside, at the end of and past a block of 16.
    let mut state = 0xC0FF_EE00_C0FF_EE00;
    for len in [0, 31, 32, 33, 47, 48, 49, 63, 64, 65, 200, size(2000, 300)] {
        for keys in [1, 2, 3, 4, 7] {
            let v: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % keys).collect();
            let as_u8: Vec<u8> = v.iter().map(|&k| k as u8).collect();
            let as_u32: Vec<u32> = v.iter().map(|&k| k as u32 * 1000).collect();
            let as_i64: Vec<i64> = v.iter().map(|&k| k as i64 - 2).collect();
            assert_sorts_like_std(&v);
            assert_sorts_like_std(&as_u8);
            assert_sorts_like_std(&as_u32);
            assert_sorts_like_std(&as_i64);
        }
    }
    // Long runs with the odd element in each position of a block of 16.
    for odd in 0..40 {
        let mut v = vec![5u16; 80];
        v[odd] = 3;
        v[79 - odd / 2] = 9;
        assert_sorts_like_std(&v);
    }
}

#[test]
fn sorts_slices_longer_than_the_window() {
    // The written slots are remembered in a window of 32,768 slots; past it, cycle starts are
    // counted again. One cycle through the whole slice writes slots on both sides of the
    // window's end. Quadratic, so it only runs with optimisations and outside Miri.
    if cfg!(miri) || cfg!(debug_assertions) {
        return;
    }
    let n = WINDOW + 1000;
    let rotated: Vec<u32> = (1..n as u32).chain([0]).collect();
    assert_sorts_like_std(&rotated);
    let mut state = 0x0DDB_A110_DDBA_1100;
    let random: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32 % 1000).collect();
    assert_sorts_like_std(&random);
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

fn assert_sorts_like_std<T: Ord + Clone + std::fmt::Debug>(input: &[T]) {
    let mut expected = input.to_vec();
    expected.sort();
    let mut actual = input.to_vec();
    cycle(&mut actual);
    assert!(actual == expected, "input: {input:?}");
}

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts_like_std(p));
    }
}

#[test]
fn sorts_every_sequence_over_few_keys() {
    // Duplicates exercise the skip past equal elements already placed.
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_like_std(s));
    }
    for len in 0..=size(12, 8) as u32 {
        for_each_sequence(len, 2, &mut |s| assert_sorts_like_std(s));
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
        self.key == other.key
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

#[test]
fn matches_std_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(150, 40) {
        // Keys from a small range force plenty of duplicates, keys from the full range make
        // them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(3, 1) {
                let input: Vec<ByKey<u64>> = (0..len)
                    .map(|idx| ByKey {
                        key: xorshift(&mut state) % modulus,
                        idx,
                    })
                    .collect();
                let mut expected: Vec<u64> = input.iter().map(|e| e.key).collect();
                expected.sort_unstable();
                let mut actual = input.clone();
                cycle(&mut actual);
                // Cycle sort is not stable, so only the order of keys is fixed...
                let keys: Vec<u64> = actual.iter().map(|e| e.key).collect();
                assert_eq!(keys, expected);
                // ...but every tagged element must still be there exactly once.
                let mut idxs: Vec<usize> = actual.iter().map(|e| e.idx).collect();
                idxs.sort_unstable();
                assert_eq!(idxs, (0..len).collect::<Vec<_>>());
            }
        }
    }
}

/// A key that counts every comparison, `==` included, made between two of them. It is
/// 16 bytes, so the sort skips equal elements one at a time, as the textbook does.
#[derive(Debug, Clone)]
struct Counted<'a> {
    key: u32,
    idx: u32,
    comparisons: &'a Cell<usize>,
}
impl PartialEq for Counted<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.comparisons.set(self.comparisons.get() + 1);
        self.key == other.key
    }
}
impl Eq for Counted<'_> {}
impl PartialOrd for Counted<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Counted<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.comparisons.set(self.comparisons.get() + 1);
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys` and returns the starting index of the element in each slot, and the number
/// of comparisons.
fn sorted_with_count(keys: &[u32]) -> (Vec<usize>, usize) {
    let count = Cell::new(0);
    let mut v: Vec<Counted> = keys
        .iter()
        .zip(0..)
        .map(|(&key, idx)| Counted {
            key,
            idx,
            comparisons: &count,
        })
        .collect();
    cycle(&mut v);
    let mut expected = keys.to_vec();
    expected.sort_unstable();
    assert_eq!(v.iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    (v.iter().map(|c| c.idx as usize).collect(), count.get())
}

/// The outcome of [`textbook`]: the starting index of the element in each slot, the
/// comparisons made, and the number of elements written into the slice.
struct Outcome {
    order: Vec<usize>,
    comparisons: usize,
    writes: usize,
}

/// Cycle sort as Wikipedia states it, with the item in hand, counting comparisons (`<` and
/// `==`) and writes into the slice. With `skip_written`, a cycle start that an earlier cycle
/// has written is passed over without counting.
fn textbook(keys: &[u32], skip_written: bool) -> Outcome {
    let n = keys.len();
    let mut v: Vec<(u32, usize)> = keys.iter().copied().zip(0..).collect();
    let (mut comparisons, mut writes) = (0, 0);
    let mut written = vec![false; n];
    let mut count_less = |v: &[(u32, usize)], start: usize, item: u32| {
        comparisons += n - start - 1;
        v[start + 1..].iter().filter(|e| e.0 < item).count()
    };
    let mut equal_run = Vec::new();
    for start in 0..n.saturating_sub(1) {
        if skip_written && written[start] {
            continue;
        }
        let mut item = v[start];
        let mut pos = start + count_less(&v, start, item.0);
        if pos == start {
            continue;
        }
        loop {
            // Skip past the elements equal to the item; the last comparison finds one that
            // is not.
            let mut skipped = 0;
            while v[pos].0 == item.0 {
                pos += 1;
                skipped += 1;
            }
            equal_run.push(skipped + 1);
            std::mem::swap(&mut v[pos], &mut item);
            written[pos] = true;
            writes += 1;
            pos = start + count_less(&v, start, item.0);
            if pos == start {
                v[start] = item;
                writes += 1;
                break;
            }
        }
    }
    Outcome {
        order: v.iter().map(|e| e.1).collect(),
        comparisons: comparisons + equal_run.iter().sum::<usize>(),
        writes,
    }
}

/// The comparisons spent recognising sorted and strictly descending input: the first scan
/// stops at the first descent, the second at the first pair that is not a strict descent.
fn precheck_comparisons(keys: &[u32]) -> usize {
    let n = keys.len();
    if n < 2 {
        return 0;
    }
    let first = keys.windows(2).position(|w| w[0] > w[1]);
    let Some(first) = first else { return n - 1 };
    let second = keys.windows(2).position(|w| w[1] >= w[0]).unwrap_or(n - 2);
    first + 1 + second + 1
}

#[test]
fn makes_the_textbook_moves_without_counting_written_starts() {
    // Exactly the textbook's result and comparisons, except that cycle starts written by an
    // earlier cycle are passed over, plus the scans for sorted and descending input.
    let check = |keys: &[u32]| {
        let (order, comparisons) = sorted_with_count(keys);
        let mut sorted = keys.to_vec();
        sorted.sort_unstable();
        let strictly_descending = keys.windows(2).all(|w| w[1] < w[0]);
        if keys == sorted || strictly_descending {
            return;
        }
        let model = textbook(keys, true);
        assert_eq!(order, model.order, "{keys:?}");
        assert_eq!(
            comparisons,
            model.comparisons + precheck_comparisons(keys),
            "{keys:?}"
        );
        assert_eq!(textbook(keys, false).order, model.order);
    };
    for n in 0..=size(7, 5) {
        for_each_permutation(n, &mut |p| {
            check(&p.iter().map(|&k| k as u32).collect::<Vec<_>>())
        });
    }
    for len in 0..=size(8, 5) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            check(&s.iter().map(|&k| k as u32).collect::<Vec<_>>())
        });
    }
    let mut state = 0xFEED_FACE_FEED_FACE;
    for len in [10, 33, 100, size(300, 60)] {
        for keys in [2, 5, 50, 1 << 30] {
            let v: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % keys) as u32)
                .collect();
            check(&v);
        }
    }
}

#[test]
fn passing_over_written_starts_saves_a_third_of_the_comparisons() {
    let mut state = 0x5A5A_5A5A_1234_4321;
    let n = size(1000, 100);
    let mut v: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut v, &mut state);
    let (_, ours) = sorted_with_count(&v);
    let full = textbook(&v, false).comparisons;
    // About `1.5·n²` for the textbook, `n²` here.
    assert!(full > n * n * 13 / 10, "{full}");
    assert!(ours * 10 < full * 7, "{ours} against {full}");
}

#[test]
fn sorted_and_strictly_descending_input_costs_about_n_comparisons() {
    for n in 2..size(200, 40) {
        let up: Vec<u32> = (0..n as u32).collect();
        assert_eq!(sorted_with_count(&up).1, n - 1);
        let steps: Vec<u32> = (0..n as u32).map(|i| i / 3).collect();
        assert_eq!(sorted_with_count(&steps).1, n - 1);
        let down: Vec<u32> = (0..n as u32).rev().collect();
        assert_eq!(sorted_with_count(&down).1, n);
    }
}

/// An element that records where the sort has written it. Every comparison checks the
/// element's address: an address inside the slice other than the one it started at counts
/// as a slot it was written to (the element in hand, outside the slice, does not count).
/// Every element in the unsorted part of the slice is compared at every count, so an element
/// written to a slot and moved on again would be seen in both.
struct Watched<'a> {
    key: u32,
    origin: usize,
    slots: Cell<usize>,
    last: Cell<usize>,
    slice: &'a Cell<(usize, usize)>,
}

impl Watched<'_> {
    fn check(&self) {
        let here = self as *const Self as usize;
        let (lo, hi) = self.slice.get();
        if (lo..hi).contains(&here) && here != self.origin && here != self.last.replace(here) {
            self.slots.set(self.slots.get() + 1);
        }
    }
}

impl PartialEq for Watched<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.check();
        other.check();
        self.key == other.key
    }
}
impl Eq for Watched<'_> {}
impl PartialOrd for Watched<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Watched<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.check();
        other.check();
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys`, checks that every element was written at most once, into its final slot,
/// and returns the number of elements that moved.
fn moves(keys: &[u32]) -> usize {
    let slice = Cell::new((0, 0));
    let mut v: Vec<Watched> = keys
        .iter()
        .map(|&key| Watched {
            key,
            origin: 0,
            slots: Cell::new(0),
            last: Cell::new(0),
            slice: &slice,
        })
        .collect();
    let range = v.as_ptr_range();
    slice.set((range.start as usize, range.end as usize));
    for w in &mut v {
        w.origin = w as *const Watched as usize;
    }
    cycle(&mut v);
    let mut moved = 0;
    for w in &v {
        let here = w as *const Watched as usize;
        assert!(w.slots.get() <= 1, "an element was written twice: {keys:?}");
        if w.slots.get() == 1 {
            assert_eq!(w.last.get(), here, "an element moved on: {keys:?}");
        }
        moved += usize::from(here != w.origin);
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moved
}

/// The fewest elements any sort must move: those in a slot whose key is not the one that
/// belongs there.
fn misplaced(keys: &[u32]) -> usize {
    let mut sorted = keys.to_vec();
    sorted.sort_unstable();
    keys.iter().zip(&sorted).filter(|(a, b)| a != b).count()
}

#[test]
fn writes_each_misplaced_element_once_and_no_other() {
    // The textbook writes exactly the misplaced elements too, counting the last element of
    // each cycle, written into the slot its first element left.
    let check = |keys: &[u32]| {
        assert_eq!(moves(keys), misplaced(keys), "{keys:?}");
        assert_eq!(textbook(keys, false).writes, misplaced(keys), "{keys:?}");
    };
    for n in 0..=size(7, 5) {
        for_each_permutation(n, &mut |p| {
            check(&p.iter().map(|&k| k as u32).collect::<Vec<_>>())
        });
    }
    for len in 0..=size(8, 5) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            check(&s.iter().map(|&k| k as u32).collect::<Vec<_>>())
        });
    }
    let mut state = 0x7777_1111_7777_1111;
    for len in [50, size(400, 80)] {
        for keys in [2, 4, 16, 1 << 30] {
            let v: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % keys) as u32)
                .collect();
            check(&v);
        }
        check(&(0..len as u32).rev().collect::<Vec<_>>());
        check(&(0..len as u32).collect::<Vec<_>>());
    }
    // Each element in its final slot already stays where it is.
    assert_eq!(moves(&[0, 2, 1, 3, 4, 6, 5]), 4);
    assert_eq!(moves(&[1, 1, 0, 1, 1]), 2);
}

#[test]
fn does_not_allocate() {
    let mut state = 0xA110_CA7E_A110_CA7E;
    let mut ints: Vec<u64> = (0..size(500, 50))
        .map(|_| xorshift(&mut state) % 100)
        .collect();
    let mut strings: Vec<String> = (0..size(200, 30))
        .map(|_| format!("{}", xorshift(&mut state)))
        .collect();
    assert_eq!(crate::allocations::count(|| cycle(&mut ints)), 0);
    assert_eq!(crate::allocations::count(|| cycle(&mut strings)), 0);
    assert!(ints.windows(2).all(|w| w[0] <= w[1]));
    assert!(strings.windows(2).all(|w| w[0] <= w[1]));
}

/// Sorts elements whose comparisons start panicking after `budget` of them, and checks that
/// no element is lost, duplicated, dropped or leaked. Returns whether the sort completed.
fn sort_with_comparison_budget(keys: &[u32], budget: usize) -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct Shared {
        drops: Vec<Cell<u32>>,
        comparisons_left: Cell<usize>,
    }
    struct Tracked<'a> {
        key: u32,
        id: u32,
        shared: &'a Shared,
    }
    impl Tracked<'_> {
        fn spend(&self) {
            let left = self.shared.comparisons_left.get();
            if left == 0 {
                panic!("comparison budget exhausted");
            }
            self.shared.comparisons_left.set(left - 1);
        }
    }
    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            let count = &self.shared.drops[self.id as usize];
            count.set(count.get() + 1);
        }
    }
    impl PartialEq for Tracked<'_> {
        fn eq(&self, other: &Self) -> bool {
            self.spend();
            self.key == other.key
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
            self.spend();
            self.key.cmp(&other.key)
        }
    }

    let len = keys.len();
    let shared = Shared {
        drops: (0..len).map(|_| Cell::new(0)).collect(),
        comparisons_left: Cell::new(budget),
    };
    let drops = &shared.drops;
    let mut v: Vec<Tracked> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Tracked {
            key,
            id,
            shared: &shared,
        })
        .collect();

    let result = catch_unwind(AssertUnwindSafe(|| cycle(&mut v)));

    if result.is_ok() {
        assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
    }
    let mut ids: Vec<usize> = v.iter().map(|t| t.id as usize).collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..len).collect::<Vec<_>>(),
        "element lost or duplicated"
    );
    assert!(
        v.iter().all(|t| t.key == keys[t.id as usize]),
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
    // Panics land in the scans for sorted input, in the count for a cycle start, with no
    // element out of the slice, and in the counts and skips past equal elements while an
    // element is in hand.
    let len = size(24, 12);
    let mut state = 0x0123_4567_89AB_CDEF;
    let distinct: Vec<u32> = {
        let mut v: Vec<u32> = (0..len as u32).collect();
        shuffle(&mut v, &mut state);
        v
    };
    let layouts: Vec<Vec<u32>> = vec![
        distinct,
        (0..len).map(|_| xorshift(&mut state) as u32 % 4).collect(),
        (0..len as u32).map(|id| id % 3).collect(),
        (0..len as u32).map(|id| id ^ 1).collect(),
        (1..len as u32).chain([0]).collect(),
        (0..len as u32).rev().map(|i| i / 2).collect(),
    ];
    // Every budget below the number of comparisons must panic, and only those: a panic from
    // anything else fails the test.
    for keys in layouts {
        let total = sorted_with_count(&keys).1;
        for budget in (0..=total).step_by(size(1, 2)).chain([total]) {
            let completed = sort_with_comparison_budget(&keys, budget);
            assert_eq!(completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

/// What a lying `<` says, and what a lying `==` says.
type Answers = (fn() -> Ordering, fn() -> bool);

thread_local! {
    /// State of the random answers of [`inconsistent_ord_keeps_every_element_exactly_once`].
    static LIES: Cell<u64> = const { Cell::new(0x5EED_5EED_5EED_5EED) };
    /// The answers the 8-byte `Small` liar gives, which leave it no room for them.
    static SMALL_ANSWERS: Cell<Answers> =
        const { Cell::new((|| Ordering::Equal, || true)) };
}

fn lie() -> u64 {
    LIES.with(|s| {
        let mut state = s.get();
        let r = xorshift(&mut state);
        s.set(state);
        r
    })
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that is not a total order makes the order of the result unspecified, but no
    // element may be lost or duplicated, no slot past the end may be touched, and the sort
    // must terminate: counts may place an element anywhere, skips past "equal" elements may
    // run to the end, and cycles may never close.
    struct Liar<P> {
        id: P,
        less: fn() -> Ordering,
        equal: fn() -> bool,
    }
    impl<P> PartialEq for Liar<P> {
        fn eq(&self, _: &Self) -> bool {
            (self.equal)()
        }
    }
    impl<P> Eq for Liar<P> {}
    impl<P> PartialOrd for Liar<P> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<P> Ord for Liar<P> {
        fn cmp(&self, _: &Self) -> Ordering {
            (self.less)()
        }
    }

    /// An 8-byte liar: elements of up to 8 bytes skip runs of "equal" elements 16 at a time.
    struct Small(u64);
    impl PartialEq for Small {
        fn eq(&self, _: &Self) -> bool {
            (SMALL_ANSWERS.with(Cell::get).1)()
        }
    }
    impl Eq for Small {}
    impl PartialOrd for Small {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Small {
        fn cmp(&self, _: &Self) -> Ordering {
            (SMALL_ANSWERS.with(Cell::get).0)()
        }
    }
    assert_eq!(size_of::<Small>(), 8);

    fn check<P: Clone + Ord + std::fmt::Debug>(
        less: fn() -> Ordering,
        equal: fn() -> bool,
        make: impl Fn(usize) -> P,
    ) {
        for len in 0..size(70, 20) {
            for _ in 0..size(3, 1) {
                let ids: Vec<P> = (0..len).map(&make).collect();
                let mut v: Vec<Liar<P>> = ids
                    .iter()
                    .map(|id| Liar {
                        id: id.clone(),
                        less,
                        equal,
                    })
                    .collect();
                cycle(&mut v);
                let mut after: Vec<P> = v.into_iter().map(|l| l.id).collect();
                after.sort_unstable();
                let mut ids = ids;
                ids.sort_unstable();
                assert_eq!(after, ids);
            }
        }
    }

    let orders: [fn() -> Ordering; 4] = [
        || Ordering::Less,
        || Ordering::Greater,
        || Ordering::Equal,
        || [Ordering::Less, Ordering::Equal, Ordering::Greater][(lie() % 3) as usize],
    ];
    let equals: [fn() -> bool; 4] = [
        || true,
        || false,
        || !lie().is_multiple_of(8),
        || lie().is_multiple_of(2),
    ];
    for less in orders {
        for equal in equals {
            check(less, equal, |i| format!("element {i}"));
            SMALL_ANSWERS.with(|a| a.set((less, equal)));
            for len in (0..size(70, 20)).chain([size(300, 40)]) {
                let mut v: Vec<Small> = (0..len as u64).map(Small).collect();
                cycle(&mut v);
                let mut after: Vec<u64> = v.into_iter().map(|s| s.0).collect();
                after.sort_unstable();
                assert_eq!(after, (0..len as u64).collect::<Vec<_>>());
            }
        }
    }
}

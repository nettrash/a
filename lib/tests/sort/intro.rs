use a::sort::intro;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::mem::size_of;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Intro sort moves every element the same way whatever its size, but the test elements still
/// carry `PAD` bytes of payload: `SMALL` keeps them at 16 bytes or less, `MEDIUM` makes them 17
/// to 64 bytes and `LARGE` more, and every test that depends on how elements are moved runs
/// with all three, so that a size-dependent mistake in the moves cannot go unnoticed.
const SMALL: usize = 0;
const MEDIUM: usize = 48;
const LARGE: usize = 112;

#[test]
fn test_elements_cover_every_path() {
    assert!(size_of::<ByKey<u64, SMALL>>() <= 16);
    assert!((17..=64).contains(&size_of::<ByKey<u64, MEDIUM>>()));
    assert!(size_of::<ByKey<u64, LARGE>>() > 64);
    assert!(size_of::<Counted<usize, SMALL>>() <= 16);
    assert!((17..=64).contains(&size_of::<Counted<usize, MEDIUM>>()));
    assert!(size_of::<Counted<usize, LARGE>>() > 64);
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    intro(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    intro(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    intro(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    intro(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    intro(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    intro(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_non_strictly_descending() {
    // Not a strictly descending run, so it is partitioned rather than reversed.
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    intro(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    intro(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        intro(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    intro(&mut v);
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
    let mut v: Vec<String> = (0..200)
        .map(|_| format!("{:03}", xorshift(&mut state) % 1000))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    intro(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    let mut v: Vec<[u64; 32]> = (0..200).map(|_| [xorshift(&mut state) % 50; 32]).collect();
    let mut expected = v.clone();
    expected.sort();
    intro(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        intro(&mut v);
        assert_eq!(v.len(), len);
    }
}

/// Parts of at most this many elements are sorted by insertion; longer ones are partitioned.
const SHORT: usize = 24;

/// Partitioning compares blocks of this many elements on each side.
const BLOCK: usize = 64;

#[test]
fn sorts_lengths_around_every_threshold() {
    // Insertion sort against partitioning, median of three against the ninther (128), and one
    // or two full blocks per side against the last blocks sharing what is left.
    let mut state = 0x7B10_C450_0000_0001;
    for n in (SHORT - 2..SHORT + 3)
        .chain(2 * BLOCK - 2..2 * BLOCK + 3)
        .chain(4 * BLOCK - 2..4 * BLOCK + 3)
    {
        for modulus in [2, 5, u64::MAX] {
            let keys: Vec<u64> = (0..n).map(|_| xorshift(&mut state) % modulus).collect();
            assert_sorts(&keys);
        }
    }
}

#[test]
fn sorts_long_inputs() {
    let n = size(100_000, 4_500);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let shapes: Vec<Vec<u32>> = vec![
        (0..n as u32).map(|_| xorshift(&mut state) as u32).collect(),
        (0..n as u32).map(|i| i % 7).collect(),
        // Organ pipe: ascending then descending.
        (0..n as u32).map(|i| i.min(n as u32 - 1 - i)).collect(),
        // Sorted with the last element moved to the front.
        std::iter::once(n as u32).chain(0..n as u32 - 1).collect(),
    ];
    for mut v in shapes {
        assert_sorts(&v);
        let mut expected = v.clone();
        expected.sort_unstable();
        intro(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    intro(&mut v[5..25]);
    let mut middle = before[5..25].to_vec();
    middle.sort_unstable();
    assert_eq!(&v[..5], &before[..5]);
    assert_eq!(&v[5..25], &middle[..]);
    assert_eq!(&v[25..], &before[25..]);
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

/// An element compared by `key` only, carrying the index it started at and `PAD` bytes of
/// payload.
#[derive(Debug, Clone, Copy)]
struct ByKey<K, const PAD: usize> {
    key: K,
    idx: usize,
    _pad: [u8; PAD],
}
impl<K: Ord, const PAD: usize> PartialEq for ByKey<K, PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord, const PAD: usize> Eq for ByKey<K, PAD> {}
impl<K: Ord, const PAD: usize> PartialOrd for ByKey<K, PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord, const PAD: usize> Ord for ByKey<K, PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys`, each tagged with its index, and checks that the keys come out in order and
/// that every element is still there exactly once, unchanged.
fn assert_sorts_with<K: Ord + Copy + std::fmt::Debug, const PAD: usize>(keys: &[K]) {
    let mut v: Vec<ByKey<K, PAD>> = keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| ByKey {
            key,
            idx,
            _pad: [0; PAD],
        })
        .collect();
    intro(&mut v);
    assert!(
        v.windows(2).all(|w| w[0].key <= w[1].key),
        "input: {keys:?}"
    );
    let mut idx: Vec<usize> = v.iter().map(|e| e.idx).collect();
    idx.sort_unstable();
    assert!(idx.into_iter().eq(0..keys.len()), "input: {keys:?}");
    assert!(
        v.iter().all(|e| e.key == keys[e.idx]),
        "element changed: {keys:?}"
    );
}

fn assert_sorts<K: Ord + Copy + std::fmt::Debug>(keys: &[K]) {
    assert_sorts_with::<K, SMALL>(keys);
    assert_sorts_with::<K, MEDIUM>(keys);
    assert_sorts_with::<K, LARGE>(keys);
}

#[test]
fn sorts_every_permutation_up_to_8() {
    // Slices of up to 24 elements are sorted by insertion alone, so this checks it on every
    // input of these lengths.
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts(p));
    }
}

#[test]
fn sorts_every_sequence_over_few_keys() {
    // Few keys put elements equal to the pivot on both sides of every partition.
    for len in 0..=size(11, 8) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts(s));
    }
    for len in 0..=size(16, 10) as u32 {
        for_each_sequence(len, 2, &mut |s| assert_sorts(s));
    }
}

#[test]
fn sorts_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(300, 70) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(3, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts(&keys);
            }
        }
    }
}

/// A key that counts every comparison made between two of them, with `PAD` bytes of payload.
#[derive(Debug, Clone)]
struct Counted<'a, K, const PAD: usize> {
    key: K,
    comparisons: &'a Cell<usize>,
    _pad: [u8; PAD],
}
impl<K: Ord, const PAD: usize> PartialEq for Counted<'_, K, PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord, const PAD: usize> Eq for Counted<'_, K, PAD> {}
impl<K: Ord, const PAD: usize> PartialOrd for Counted<'_, K, PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord, const PAD: usize> Ord for Counted<'_, K, PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.comparisons.set(self.comparisons.get() + 1);
        self.key.cmp(&other.key)
    }
}

/// Sorts a copy of `keys`, checks the result, and returns the number of comparisons.
fn comparisons_with<K: Ord + Clone + std::fmt::Debug, const PAD: usize>(keys: &[K]) -> usize {
    let count = Cell::new(0);
    let mut v: Vec<Counted<K, PAD>> = keys
        .iter()
        .map(|k| Counted {
            key: k.clone(),
            comparisons: &count,
            _pad: [0; PAD],
        })
        .collect();
    intro(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

fn comparisons<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> usize {
    comparisons_with::<K, SMALL>(keys)
}

#[test]
fn sorted_reverse_sorted_and_equal_input_costs_n_minus_1_comparisons() {
    for n in 0..size(1000, 100) {
        let expected = n.saturating_sub(1);
        let sorted: Vec<usize> = (0..n).collect();
        let reverse: Vec<usize> = (0..n).rev().collect();
        assert_eq!(comparisons(&sorted), expected, "n = {n}");
        assert_eq!(comparisons(&reverse), expected, "n = {n}");
        assert_eq!(comparisons(&vec![7; n]), expected, "n = {n}");
    }
}

#[test]
fn random_input_costs_about_n_log2_n_comparisons() {
    // About 11,800 for `n = 1000`, insertion into parts of up to 24 elements costing more than
    // partitioning them further would; never 1.25 times `n·log₂ n` in practice.
    let n = size(1000, 200);
    let bound = 1.25 * n as f64 * (n as f64).log2();
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    for _ in 0..size(30, 2) {
        let mut v: Vec<usize> = (0..n).collect();
        shuffle(&mut v, &mut state);
        let small = comparisons_with::<_, SMALL>(&v);
        let large = comparisons_with::<_, LARGE>(&v);
        assert!((small as f64) < bound, "{small} comparisons, bound {bound}");
        assert!((large as f64) < bound, "{large} comparisons, bound {bound}");
    }
}

#[test]
fn equal_keys_split_evenly() {
    // Elements equal to the pivot stop the scans on both sides, so they are spread over both
    // parts: few distinct keys cost `O(n log n)` like any other input, without ever needing
    // the heap sort fallback, instead of degrading to quadratic time.
    let n = size(20_000, 500);
    let bound = 1.5 * n as f64 * (n as f64).log2();
    let mut state = 0xFE11_FE11_FE11_FE11;
    for k in [2usize, 3, 4, 16] {
        let v: Vec<usize> = (0..n).map(|_| xorshift(&mut state) as usize % k).collect();
        let c = comparisons(&v);
        assert!(
            (c as f64) < bound,
            "k = {k}: {c} comparisons, bound {bound}"
        );
    }
    // Two keys alternating, and all equal but one element out of place.
    let alternating: Vec<usize> = (0..n).map(|i| i % 2).collect();
    let mut nearly_equal = vec![1; n];
    nearly_equal[0] = 2;
    for v in [alternating, nearly_equal] {
        let c = comparisons(&v);
        assert!((c as f64) < bound, "{c} comparisons, bound {bound}");
    }
}

/// McIlroy's adversary ("A Killer Adversary for Quicksort", 1999). Elements start out as
/// "gas", greater than every other value; a comparison between two gas elements freezes one of
/// them to the next smallest value, choosing the one that is not the likely pivot (the gas
/// element compared most recently). Values only ever go from gas to solid, so the answers are
/// consistent with the final values, a total order, and the sort must still sort; but any
/// quick sort that relies on its pivots alone is driven to quadratic time.
struct Adversary<'a> {
    id: usize,
    state: &'a RefCell<AdversaryState>,
}

struct AdversaryState {
    values: Vec<usize>,
    solid: usize,
    candidate: usize,
    comparisons: usize,
}

impl AdversaryState {
    const GAS: usize = usize::MAX;

    fn freeze(&mut self, id: usize) {
        self.values[id] = self.solid;
        self.solid += 1;
    }
}

impl PartialEq for Adversary<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Adversary<'_> {}
impl PartialOrd for Adversary<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Adversary<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let mut s = self.state.borrow_mut();
        s.comparisons += 1;
        let (x, y) = (self.id, other.id);
        if s.values[x] == AdversaryState::GAS && s.values[y] == AdversaryState::GAS {
            if x == s.candidate {
                s.freeze(x);
            } else {
                s.freeze(y);
            }
        }
        if s.values[x] == AdversaryState::GAS {
            s.candidate = x;
        } else if s.values[y] == AdversaryState::GAS {
            s.candidate = y;
        }
        s.values[x].cmp(&s.values[y])
    }
}

#[test]
fn adversarial_input_still_costs_o_n_log_n() {
    // Pivots alone cannot defend against the adversary; the heap sort fallback must. Without
    // it, `n = 10,000` costs 9.1 million comparisons; with it, about 470,000.
    //
    // The first two elements start out solid and descending, followed by gas, which is
    // greater: otherwise the adversary would answer the initial scan for a sorted run as if
    // the whole slice were one, and the sort would rightly stop after `n − 1` comparisons.
    for n in [100, 1000, size(10_000, 1000)] {
        let mut values = vec![AdversaryState::GAS; n];
        values[0] = 1;
        values[1] = 0;
        let state = RefCell::new(AdversaryState {
            values,
            solid: 2,
            candidate: 0,
            comparisons: 0,
        });
        let mut v: Vec<Adversary> = (0..n).map(|id| Adversary { id, state: &state }).collect();
        intro(&mut v);
        let s = state.borrow();
        assert!(v.windows(2).all(|w| s.values[w[0].id] <= s.values[w[1].id]));
        let bound = 4.0 * n as f64 * (n as f64).log2();
        assert!(
            (s.comparisons as f64) < bound,
            "n = {n}: {} comparisons, bound {bound}",
            s.comparisons
        );
    }
}

/// An element that notices when the sort has moved it: every comparison checks its address
/// against the one it was last seen at. Zero observed moves means every element sat at its
/// starting address whenever it was compared, and is still there at the end.
struct Watched<'a, K> {
    key: K,
    seen_at: Cell<usize>,
    moves: &'a Cell<usize>,
}

impl<K> Watched<'_, K> {
    fn check(&self) {
        let here = self as *const Self as usize;
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

#[test]
fn non_decreasing_input_is_never_moved() {
    for n in 0..size(300, 60) {
        for keys in [(0..n).collect::<Vec<_>>(), (0..n).map(|i| i / 3).collect()] {
            let moves = Cell::new(0);
            let mut v: Vec<Watched<usize>> = keys
                .iter()
                .map(|&key| Watched {
                    key,
                    seen_at: Cell::new(0),
                    moves: &moves,
                })
                .collect();
            for w in &v {
                w.seen_at.set(w as *const Watched<usize> as usize);
            }
            intro(&mut v);
            for w in &v {
                w.check();
            }
            assert_eq!(moves.get(), 0, "{keys:?}");
        }
    }
}

/// Sorts elements of `PAD` extra bytes whose comparisons start panicking after `budget` of
/// them, and checks that no element is lost, duplicated, dropped or leaked. Returns whether
/// the sort completed.
fn sort_with_comparison_budget<const PAD: usize>(keys: &[u32], budget: usize) -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct Shared {
        drops: Vec<Cell<u32>>,
        comparisons_left: Cell<usize>,
    }
    // 16 bytes plus `PAD`, like the `Counted` elements whose comparisons set the budgets:
    // elements of another size class could take other paths and make a different number.
    struct Tracked<'a, const PAD: usize> {
        key: u32,
        id: u32,
        shared: &'a Shared,
        _pad: [u8; PAD],
    }
    impl<const PAD: usize> Drop for Tracked<'_, PAD> {
        fn drop(&mut self) {
            let count = &self.shared.drops[self.id as usize];
            count.set(count.get() + 1);
        }
    }
    impl<const PAD: usize> PartialEq for Tracked<'_, PAD> {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl<const PAD: usize> Eq for Tracked<'_, PAD> {}
    impl<const PAD: usize> PartialOrd for Tracked<'_, PAD> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<const PAD: usize> Ord for Tracked<'_, PAD> {
        fn cmp(&self, other: &Self) -> Ordering {
            let left = self.shared.comparisons_left.get();
            if left == 0 {
                panic!("comparison budget exhausted");
            }
            self.shared.comparisons_left.set(left - 1);
            self.key.cmp(&other.key)
        }
    }

    let len = keys.len();
    let shared = Shared {
        drops: (0..len).map(|_| Cell::new(0)).collect(),
        comparisons_left: Cell::new(budget),
    };
    let drops = &shared.drops;
    let mut v: Vec<Tracked<PAD>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Tracked {
            key,
            id,
            shared: &shared,
            _pad: [0; PAD],
        })
        .collect();

    let result = catch_unwind(AssertUnwindSafe(|| intro(&mut v)));

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
    // Longer than two blocks, so partitioning runs a full block on each side before the last
    // blocks share what is left.
    let len = size(150, 40);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    let distinct: Vec<u32> = {
        let mut v: Vec<u32> = (0..len as u32).collect();
        shuffle(&mut v, &mut state);
        v
    };
    // Panics land in the run scan, pivot selection, block partitioning on either side, the
    // last blocks and insertion sort.
    let layouts: Vec<Vec<u32>> = vec![
        distinct,
        random,
        (0..len as u32).map(|id| id % 3).collect(),
        (0..len as u32).map(|id| id ^ 1).collect(),
        (0..len as u32).map(|i| i.min(len as u32 - 1 - i)).collect(),
        (1..len as u32).chain([0]).collect(),
    ];
    // Each size class takes its own paths, so each makes its own number of comparisons. Every
    // budget below it must panic, and only those: a panic from anything else fails the test.
    fn check<const PAD: usize>(keys: &[u32]) {
        let total = comparisons_with::<_, PAD>(keys);
        for budget in 0..=total {
            let completed = sort_with_comparison_budget::<PAD>(keys, budget);
            assert_eq!(completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
    for keys in layouts {
        check::<SMALL>(&keys);
        check::<MEDIUM>(&keys);
        check::<LARGE>(&keys);
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that is not a total order makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // terminate.
    struct Liar<'a, const PAD: usize> {
        name: String,
        answer: &'a dyn Fn() -> Ordering,
        _pad: [u8; PAD],
    }
    impl<const PAD: usize> PartialEq for Liar<'_, PAD> {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl<const PAD: usize> Eq for Liar<'_, PAD> {}
    impl<const PAD: usize> PartialOrd for Liar<'_, PAD> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl<const PAD: usize> Ord for Liar<'_, PAD> {
        fn cmp(&self, _: &Self) -> Ordering {
            (self.answer)()
        }
    }

    fn check<const PAD: usize>(answer: &dyn Fn() -> Ordering) {
        for len in 0..size(100, 30) {
            for _ in 0..size(5, 1) {
                let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
                let mut v: Vec<Liar<PAD>> = names
                    .iter()
                    .map(|name| Liar {
                        name: name.clone(),
                        answer,
                        _pad: [0; PAD],
                    })
                    .collect();

                intro(&mut v);

                let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
                after.sort_unstable();
                let mut names = names;
                names.sort_unstable();
                assert_eq!(after, names);
            }
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
        check::<SMALL>(answer);
        check::<MEDIUM>(answer);
        check::<LARGE>(answer);
    }
}

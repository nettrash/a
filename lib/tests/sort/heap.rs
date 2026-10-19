use a::sort::heap;
use std::cell::Cell;
use std::cmp::Ordering;
use std::mem::size_of;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Heap sort uses a heap with two children per node below 64 KiB of elements and four from
/// there on. The test elements carry `PAD` bytes of payload: `SMALL` keeps them at 16 bytes,
/// so the slices of most tests use two children, and `LARGE` makes them 128 bytes, so that
/// [`FOUR_CHILDREN`] of them already use four, cheaply enough for Miri.
const SMALL: usize = 0;
const LARGE: usize = 112;

/// The shortest slice of `LARGE` elements sorted on a heap with four children per node.
const FOUR_CHILDREN: usize = 512;

#[test]
fn test_elements_cover_both_heap_shapes() {
    const QUATERNARY_MIN_BYTES: usize = 64 * 1024;
    assert_eq!(size_of::<ByKey<u64, SMALL>>(), 16);
    assert_eq!(
        size_of::<ByKey<u64, LARGE>>() * FOUR_CHILDREN,
        QUATERNARY_MIN_BYTES
    );
    assert_eq!(size_of::<Counted<usize, SMALL>>(), 16);
    assert_eq!(
        size_of::<Counted<usize, LARGE>>() * FOUR_CHILDREN,
        QUATERNARY_MIN_BYTES
    );
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    heap(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    heap(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    heap(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    heap(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    heap(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    heap(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_non_strictly_descending() {
    // Not a strictly descending run, so it is heap sorted rather than reversed.
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    heap(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    heap(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        heap(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    heap(&mut v);
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
    heap(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [200, 3000] {
        // 3000 elements of 256 bytes take four children per node.
        let mut v: Vec<[u64; 32]> = (0..len).map(|_| [xorshift(&mut state) % 50; 32]).collect();
        let mut expected = v.clone();
        expected.sort();
        heap(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        heap(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_inputs() {
    // 400 KB of `u32`s outside Miri, so four children per node, and 18 KB under it, so two.
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
        let mut expected = v.clone();
        expected.sort_unstable();
        heap(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_with_four_children_per_node() {
    // Every length from the threshold on, until the last node with children has had each
    // possible number of them a few times over, and some longer ones.
    let mut state = 0x4444_4444_4444_4444;
    let lens = (FOUR_CHILDREN..FOUR_CHILDREN + size(24, 8)).chain([size(5000, 1000)]);
    for len in lens {
        for modulus in [3, u64::MAX] {
            let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
            assert_sorts_with::<u64, LARGE>(&keys);
        }
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    heap(&mut v[5..25]);
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
    heap(&mut v);
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
    assert_sorts_with::<K, LARGE>(keys);
}

#[test]
fn sorts_every_permutation_up_to_8() {
    // Every heap shape up to three levels, with every order of its elements.
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts(p));
    }
}

#[test]
fn sorts_every_sequence_over_few_keys() {
    // Few keys make the held element equal to the elements it is compared with, on the way
    // down and on the way up.
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
    heap(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

fn comparisons<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> usize {
    comparisons_with::<K, SMALL>(keys)
}

#[test]
fn sorted_and_reverse_sorted_input_costs_n_minus_1_comparisons() {
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
    // Bottom-up removal makes one comparison per level on the way down and few on the way up:
    // about `n·log₂ n + 0.5n` in all, against `2n·log₂ n` for the textbook heap sort. Four
    // children per node take three comparisons per level of half as many: about
    // `1.5·n·log₂ n`.
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    for _ in 0..size(10, 2) {
        let n = 1000;
        let mut v: Vec<usize> = (0..n).collect();
        shuffle(&mut v, &mut state);
        let lg = n as f64 * (n as f64).log2();
        let two = comparisons_with::<_, SMALL>(&v) as f64;
        assert!(two < lg + n as f64, "{two} comparisons with two children");
        let four = comparisons_with::<_, LARGE>(&v) as f64;
        assert!(four < 1.6 * lg, "{four} comparisons with four children");
    }
}

#[test]
fn every_input_costs_o_n_log_n_comparisons() {
    // Heap sort is not adaptive: no input makes it much cheaper, but none makes it costlier
    // either. Bounded by the textbook heap sort's `2n·log₂ n` here.
    let n = size(4096, 600);
    let mut state = 0x5EED_0F5E_ED0F_5EED;
    let shapes: Vec<Vec<u32>> = vec![
        (0..n as u32).map(|_| xorshift(&mut state) as u32).collect(),
        (0..n as u32)
            .map(|_| xorshift(&mut state) as u32 % 4)
            .collect(),
        (0..n as u32).map(|i| i.min(n as u32 - 1 - i)).collect(),
        (0..n as u32).map(|i| i % 64).collect(),
        (0..n as u32).map(|i| i ^ 1).collect(),
        std::iter::once(n as u32).chain(0..n as u32 - 1).collect(),
    ];
    let bound = 2.0 * n as f64 * (n as f64).log2();
    for keys in &shapes {
        let two = comparisons_with::<_, SMALL>(keys) as f64;
        let four = comparisons_with::<_, LARGE>(keys) as f64;
        assert!(
            two < bound,
            "{two} comparisons with two children, bound {bound}"
        );
        assert!(
            four < bound,
            "{four} comparisons with four children, bound {bound}"
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
            heap(&mut v);
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
    // elements of another size could take the other heap shape and make a different number.
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

    let result = catch_unwind(AssertUnwindSafe(|| heap(&mut v)));

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
    // Panics land in the run scan, building the heap and both halves of a removal. With
    // `LARGE` elements and at least `FOUR_CHILDREN` of them, the heap has four children per
    // node; Miri tries every 61st budget there.
    fn layouts(len: usize, state: &mut u64) -> Vec<Vec<u32>> {
        let random: Vec<u32> = (0..len).map(|_| xorshift(state) as u32 % 7).collect();
        let distinct: Vec<u32> = {
            let mut v: Vec<u32> = (0..len as u32).collect();
            shuffle(&mut v, state);
            v
        };
        vec![
            distinct,
            random,
            (0..len as u32).map(|id| id % 3).collect(),
            (0..len as u32).map(|id| id ^ 1).collect(),
            (0..len as u32).map(|i| i.min(len as u32 - 1 - i)).collect(),
            (1..len as u32).chain([0]).collect(),
        ]
    }
    // Every budget below the number of comparisons must panic, and only those: a panic from
    // anything else fails the test.
    fn check<const PAD: usize>(keys: &[u32], step: usize) {
        let total = comparisons_with::<_, PAD>(keys);
        for budget in (0..=total).step_by(step).chain([total]) {
            let completed = sort_with_comparison_budget::<PAD>(keys, budget);
            assert_eq!(completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
    let mut state = 0x0123_4567_89AB_CDEF;
    for keys in layouts(size(60, 24), &mut state) {
        check::<SMALL>(&keys, 1);
        check::<LARGE>(&keys, 1);
    }
    for keys in layouts(FOUR_CHILDREN + 3, &mut state) {
        check::<LARGE>(&keys, size(1, 61));
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

    fn check<const PAD: usize>(answer: &dyn Fn() -> Ordering, lens: impl Iterator<Item = usize>) {
        for len in lens {
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

                heap(&mut v);

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
    // `Liar` is 40 bytes plus `PAD`: `LARGE` ones take four children per node from 432.
    for answer in answers {
        check::<SMALL>(answer, 0..size(100, 30));
        check::<LARGE>(answer, (0..size(100, 30)).chain(480..480 + size(8, 2)));
    }
}

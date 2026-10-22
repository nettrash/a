use a::sort::tree;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::mem::size_of;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Tree sort copies elements of 5 to 8 bytes into nodes of 16 bytes and elements of 17 to 24
/// bytes into nodes of 32 bytes, and compares all others where they are, in the slice. The
/// test elements carry `PAD` bytes of payload after 8 bytes of data: `P8` keeps them at 8
/// bytes, `P24` makes them 24 and `P128` 128, one of each layout.
const P8: usize = 0;
const P24: usize = 16;
const P128: usize = 120;

#[test]
fn test_elements_cover_every_layout() {
    for (size, pad) in [(8, P8), (24, P24), (128, P128)] {
        assert_eq!(size, 8 + pad);
    }
    assert_eq!(size_of::<ByKey<P8>>(), 8);
    assert_eq!(size_of::<ByKey<P24>>(), 24);
    assert_eq!(size_of::<ByKey<P128>>(), 128);
    assert_eq!(size_of::<Counted<P8>>(), 8);
    assert_eq!(size_of::<Counted<P24>>(), 24);
    assert_eq!(size_of::<Counted<P128>>(), 128);
    assert_eq!(size_of::<Tracked<P8>>(), 8);
    assert_eq!(size_of::<Tracked<P24>>(), 24);
    assert_eq!(size_of::<Tracked<P128>>(), 128);
    assert_eq!(size_of::<Liar<P8>>(), 8);
    assert_eq!(size_of::<Liar<P24>>(), 24);
    assert_eq!(size_of::<Liar<P128>>(), 128);
    assert_eq!(size_of::<Swapping<P8>>(), 8);
    assert_eq!(size_of::<Swapping<P24>>(), 24);
    assert_eq!(size_of::<Reallocating>(), 40);
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    tree(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    tree(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    tree(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    tree(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    tree(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u64> = (0..100).rev().collect();
    tree(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    tree(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        tree(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    tree(&mut v);
    assert_eq!(
        v,
        vec![
            "apple", "apple", "banana", "cherry", "date", "fig", "kiwi", "lime", "pear", "plum"
        ]
    );
}

#[test]
fn sorts_heap_owning_elements() {
    // `String`s are 24 bytes, kept in the nodes; boxes are 8.
    let mut state = 0x1EAF_5EED_1EAF_5EED;
    let mut v: Vec<String> = (0..size(500, 100))
        .map(|_| format!("{:03}", xorshift(&mut state) % 1000))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    tree(&mut v);
    assert_eq!(v, expected);

    let mut v: Vec<Box<u32>> = (0..size(500, 100))
        .map(|_| Box::new(xorshift(&mut state) as u32 % 100))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    tree(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [0, 1, 2, 3, 200, size(3000, 300)] {
        let mut v: Vec<[u64; 32]> = (0..len).map(|_| [xorshift(&mut state) % 50; 32]).collect();
        let mut expected = v.clone();
        expected.sort();
        tree(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_over_aligned_elements() {
    // Too strictly aligned for the room in a node, so compared in the slice.
    #[repr(align(16))]
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    struct Aligned(u64);
    let mut state = 0xA116_0A11_6A11_6A11;
    let mut v: Vec<Aligned> = (0..size(1000, 100))
        .map(|_| Aligned(xorshift(&mut state) % 300))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    tree(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        tree(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_inputs() {
    // From 1024 elements on, four searches run side by side; from 262,144 on, eight.
    let n = size(300_000, 2_000);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let shapes: Vec<Vec<u64>> = vec![
        (0..n).map(|_| xorshift(&mut state)).collect(),
        (0..n as u64).map(|i| i % 7).collect(),
        (0..n).map(|_| xorshift(&mut state) % 16).collect(),
        // Organ pipe: ascending then descending.
        (0..n as u64).map(|i| i.min(n as u64 - 1 - i)).collect(),
        // Sorted with the last element moved to the front.
        std::iter::once(n as u64).chain(0..n as u64 - 1).collect(),
        // Sawtooth.
        (0..n as u64).map(|i| i % 300).collect(),
    ];
    for mut v in shapes {
        let mut expected = v.clone();
        expected.sort_unstable();
        let mut narrow: Vec<u32> = v.iter().map(|&k| k as u32).collect();
        let mut expected_narrow = narrow.clone();
        expected_narrow.sort_unstable();
        tree(&mut v);
        assert_eq!(v, expected);
        tree(&mut narrow);
        assert_eq!(narrow, expected_narrow);
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    tree(&mut v[5..25]);
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

/// Inputs whose runs make insertions go next to the one before, up and down, at the ends of
/// the tree and inside it, so that they start from the last insertion, and inputs that
/// degrade an unbalanced tree.
fn shaped_inputs(n: usize, state: &mut u64) -> Vec<Vec<usize>> {
    let mut nearly: Vec<usize> = (0..n).collect();
    for _ in 0..n / 20 + 1 {
        let a = xorshift(state) as usize % n.max(1);
        let b = xorshift(state) as usize % n.max(1);
        nearly.swap(a, b);
    }
    let period = (n / 7).max(2);
    vec![
        (0..n).collect(),
        (0..n).rev().collect(),
        // Organ pipe and its inverse.
        (0..n).map(|i| i.min(n - 1 - i)).collect(),
        (0..n).map(|i| i.max(n - 1 - i)).collect(),
        // Ascending and descending sawtooth.
        (0..n).map(|i| i % period).collect(),
        (0..n).map(|i| period - i % period).collect(),
        // Alternately from both ends: every insertion goes beside the one two before.
        (0..n)
            .map(|i| if i % 2 == 0 { i / 2 } else { n - i / 2 })
            .collect(),
        // Runs that interleave: every element goes between two of the run before.
        (0..n).map(|i| (i % period) * 7 + i / period).collect(),
        // Sorted with a few elements displaced, some of them to the ends.
        nearly,
        // Sorted with the largest element first, and with random elements appended.
        std::iter::once(n).chain(0..n.saturating_sub(1)).collect(),
        (0..n - n / 10)
            .chain((0..n / 10).map(|_| xorshift(state) as usize % n.max(1)))
            .collect(),
        // Equal keys in runs.
        (0..n).map(|i| i / 5).collect(),
        (0..n).map(|i| (n - i) / 5).collect(),
    ]
}

/// Calls `f` with every permutation up to length 8, every sequence over three keys up to
/// length 11, random sequences of distinct and of repeated keys up to length 300, and the
/// shaped inputs.
fn for_each_test_input(seed: u64, f: &mut impl FnMut(&[usize])) {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, f);
    }
    for len in 0..=size(11, 7) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            let s: Vec<usize> = s.iter().map(|&k| k as usize).collect();
            f(&s);
        });
    }
    let mut state = seed;
    for n in [9, 16, 17, 33, 100, size(300, 40)] {
        for _ in 0..size(10, 1) {
            let mut v: Vec<usize> = (0..n).collect();
            shuffle(&mut v, &mut state);
            f(&v);
            let v: Vec<usize> = (0..n).map(|_| xorshift(&mut state) as usize % 3).collect();
            f(&v);
        }
        for v in shaped_inputs(n, &mut state) {
            f(&v);
        }
    }
}

/// Replaces every key by its rank among the distinct keys: the same order and the same ties
/// in a small integer.
fn ranks<K: Ord + Copy>(keys: &[K]) -> Vec<u32> {
    let mut distinct = keys.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    keys.iter()
        .map(|k| distinct.binary_search(k).unwrap() as u32)
        .collect()
}

/// An element compared by `key` only, carrying the index it started at and `PAD` bytes of
/// payload.
#[derive(Debug, Clone, Copy)]
struct ByKey<const PAD: usize> {
    key: u32,
    idx: u32,
    _pad: [u8; PAD],
}
impl<const PAD: usize> PartialEq for ByKey<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for ByKey<PAD> {}
impl<const PAD: usize> PartialOrd for ByKey<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for ByKey<PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys`, each tagged with its index, in elements of `PAD` extra bytes, and checks the
/// result against std's stable sort: the same keys in the same order, and equal keys still in
/// their original order.
fn assert_sorts_like_std_stable_with<const PAD: usize>(keys: &[u32]) {
    let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|e| e.0);
    let mut actual: Vec<ByKey<PAD>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, idx)| ByKey {
            key,
            idx,
            _pad: [0; PAD],
        })
        .collect();
    tree(&mut actual);
    let actual: Vec<(u32, u32)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert_eq!(actual, expected, "input: {keys:?}");
}

fn assert_sorts_like_std_stable<K: Ord + Copy>(keys: &[K]) {
    let keys = ranks(keys);
    assert_sorts_like_std_stable_with::<P8>(&keys);
    assert_sorts_like_std_stable_with::<P24>(&keys);
    assert_sorts_like_std_stable_with::<P128>(&keys);
}

#[test]
fn is_stable_for_equal_keys() {
    assert_sorts_like_std_stable(&[2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]);
    assert_sorts_like_std_stable(&[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]);
    assert_sorts_like_std_stable(&[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    assert_sorts_like_std_stable(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]);
    assert_sorts_like_std_stable(&[3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_every_permutation_up_to_8_stably() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts_like_std_stable(p));
    }
}

#[test]
fn sorts_every_sequence_over_few_keys_stably() {
    // Few keys put equal elements on both sides of every node, and make searches side by
    // side go to the same place.
    for len in 0..=size(11, 7) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_like_std_stable(s));
    }
    for len in 0..=size(16, 9) as u32 {
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
            for _ in 0..size(3, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts_like_std_stable(&keys);
            }
        }
    }
}

#[test]
fn matches_std_stable_sort_with_searches_side_by_side() {
    // Four searches at once from 1024 elements on, eight from 262,144 on: few keys make
    // them go to the same slots and through the same rotated nodes.
    let mut state = 0x51DE_B151_DEB1_51DE;
    let lens = if cfg!(miri) {
        vec![1024, 1031]
    } else {
        vec![1024, 1027, 5000, 262_144, 262_151]
    };
    for len in lens {
        for modulus in [2, 16, 1000, u64::MAX] {
            let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
            assert_sorts_like_std_stable(&keys);
        }
    }
}

#[test]
fn matches_std_stable_sort_on_shaped_input() {
    let mut state = 0x5A4E_D5A4_ED5A_4ED5;
    for n in (1..size(70, 20)).chain([size(1000, 100), size(5000, 1100)]) {
        for keys in shaped_inputs(n, &mut state) {
            assert_sorts_like_std_stable(&keys);
        }
    }
}

thread_local! {
    /// Comparisons made by `Counted` elements.
    static COMPARISONS: Cell<usize> = const { Cell::new(0) };
}

/// A key that counts every comparison made between two of them, with `PAD` bytes of payload.
#[derive(Debug, Clone)]
struct Counted<const PAD: usize> {
    key: u64,
    _pad: [u8; PAD],
}
impl<const PAD: usize> PartialEq for Counted<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for Counted<PAD> {}
impl<const PAD: usize> PartialOrd for Counted<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for Counted<PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        COMPARISONS.with(|c| c.set(c.get() + 1));
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys` in elements of `PAD` extra bytes, checks the result, and returns the number of
/// comparisons.
fn comparisons_with<const PAD: usize>(keys: &[u32]) -> usize {
    let mut v: Vec<Counted<PAD>> = keys
        .iter()
        .map(|&k| Counted {
            key: u64::from(k),
            _pad: [0; PAD],
        })
        .collect();
    COMPARISONS.with(|c| c.set(0));
    tree(&mut v);
    let count = COMPARISONS.with(|c| c.get());
    let mut expected = keys.to_vec();
    expected.sort();
    assert!(
        v.iter()
            .map(|c| c.key)
            .eq(expected.into_iter().map(u64::from))
    );
    count
}

/// Returns the number of comparisons sorting `keys` takes, the same whichever layout.
fn comparisons<K: Ord + Copy + std::fmt::Debug>(keys: &[K]) -> usize {
    let ranked = ranks(keys);
    let c = comparisons_with::<P8>(&ranked);
    assert_eq!(comparisons_with::<P24>(&ranked), c, "{keys:?}");
    assert_eq!(comparisons_with::<P128>(&ranked), c, "{keys:?}");
    c
}

#[test]
fn sorted_and_reverse_sorted_input_costs_n_minus_1_comparisons() {
    // Every insertion goes right after the maximum or right before the minimum, which the
    // insertion before it was, at one comparison.
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
fn runs_cost_at_most_2_comparisons_per_element() {
    // An element that goes right after or right before the one inserted before it costs two
    // comparisons, with the last insertion and its neighbour; the first of each run is
    // searched for from the root.
    let n = size(2000, 200);
    let runs = 10;
    let len = n / runs;
    let lg = (n as f64).log2().ceil() as usize;
    let mut state = 0x2C0A_5E5E_2C0A_5E5E;
    let mut starts: Vec<usize> = (0..runs).collect();
    shuffle(&mut starts, &mut state);
    // Runs of disjoint ranges in random order, ascending and descending.
    let up: Vec<usize> = starts
        .iter()
        .flat_map(|&s| (0..len).map(move |i| s * len + i))
        .collect();
    let down: Vec<usize> = starts
        .iter()
        .flat_map(|&s| (0..len).rev().map(move |i| s * len + i))
        .collect();
    for keys in [up, down] {
        let c = comparisons(&keys);
        let bound = 2 * n + runs * (2 * lg + 4);
        assert!(c <= bound, "{c} comparisons, bound {bound}");
    }
}

#[test]
fn random_input_costs_about_n_log2_n_minus_n_comparisons() {
    // An AVL tree built from random input is nearly perfectly balanced: about
    // `n·log₂ n − 1.2n` comparisons, against about `1.39·n·log₂ n − 2.8n` for an unbalanced
    // binary search tree.
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    let n = size(1000, 200);
    let bound = n as f64 * (n as f64).log2() - n as f64;
    for _ in 0..size(10, 1) {
        let mut v: Vec<usize> = (0..n).collect();
        shuffle(&mut v, &mut state);
        let c = comparisons(&v) as f64;
        assert!(c < bound, "{c} comparisons, bound {bound}");
    }
}

/// The most comparisons an insertion into a tree of `n` nodes can cost: two that try the last
/// insertion's neighbourhood, a search, and a second search if a search made side by side
/// with others was overtaken by their insertions. An AVL tree of `n` nodes is less than
/// `1.4405·log₂(n + 2)` levels high, whatever the order of insertion.
fn insertion_bound(n: usize) -> usize {
    2 + 2 * (1.4405 * ((n + 2) as f64).log2()) as usize
}

#[test]
fn every_input_stays_within_the_height_bound() {
    for_each_test_input(0xB0B0_1234_5678_9ABC, &mut |v| {
        let c = comparisons(v);
        let bound: usize = (0..v.len()).map(insertion_bound).sum();
        assert!(c <= bound, "{c} comparisons, bound {bound}: {v:?}");
    });
}

#[test]
fn every_input_costs_o_n_log_n_comparisons() {
    // No input makes a search longer than the height of the tree, including those that
    // degrade an unbalanced binary search tree to a list; and searches rarely need to go on
    // again, so about one search per element.
    let n = size(4096, 1100);
    let mut state = 0x5EED_0F5E_ED0F_5EED;
    let mut shapes = shaped_inputs(n, &mut state);
    shapes.push((0..n).map(|_| xorshift(&mut state) as usize % 4).collect());
    shapes.push((0..n).map(|i| i ^ 1).collect());
    shapes.push((0..n).map(|i| i % 64).collect());
    let height = 1.4405 * ((n + 2) as f64).log2();
    let bound = (n as f64 * (height + 2.0)) as usize;
    for keys in &shapes {
        let c = comparisons(keys);
        assert!(c < bound, "{c} comparisons, bound {bound}");
    }
}

/// Sorts `v` and returns how many allocations that made.
fn allocations_of<T: Ord>(v: &mut [T]) -> usize {
    crate::allocations::count(|| tree(v))
}

#[test]
fn allocates_once_and_only_beyond_4_kib() {
    // The tree takes 17 bytes per `u64`, a 16-byte node and a balance factor: 240 of them fit
    // in the 4 KiB kept on the stack.
    let mut state = 0x0A11_0C47_E5B0_FFE5;
    let ns = if cfg!(miri) {
        vec![0, 1, 2, 100, 240, 241, 300]
    } else {
        (0..400).collect()
    };
    for n in ns {
        let mut v: Vec<u64> = (0..n).map(|_| xorshift(&mut state)).collect();
        assert_eq!(allocations_of(&mut v), usize::from(n > 240), "n = {n}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}

#[test]
fn elements_in_the_slice_allocate_beyond_4_kib_too() {
    // Elements that stay in the slice are gathered into a buffer in order: 137 bytes per
    // 128-byte element, with an 8-byte node and a balance factor.
    for n in 0..=40 {
        let mut v: Vec<ByKey<P128>> = (0..n as u32)
            .rev()
            .map(|key| ByKey {
                key,
                idx: 0,
                _pad: [0; P128],
            })
            .collect();
        assert_eq!(allocations_of(&mut v), usize::from(n > 29), "n = {n}");
        assert!(v.iter().map(|e| e.key).eq(0..n as u32));
    }
}

thread_local! {
    /// How often each `Tracked` element was dropped, and how many comparisons are left before
    /// they start panicking.
    static TRACKING: RefCell<(Vec<u32>, usize)> = const { RefCell::new((Vec::new(), 0)) };
}

/// An element that counts its drops and whose comparisons panic once the budget runs out,
/// with `PAD` bytes of payload.
struct Tracked<const PAD: usize> {
    key: u32,
    id: u32,
    _pad: [u8; PAD],
}
impl<const PAD: usize> Drop for Tracked<PAD> {
    fn drop(&mut self) {
        TRACKING.with(|t| t.borrow_mut().0[self.id as usize] += 1);
    }
}
impl<const PAD: usize> PartialEq for Tracked<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for Tracked<PAD> {}
impl<const PAD: usize> PartialOrd for Tracked<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for Tracked<PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        let exhausted = TRACKING.with(|t| {
            let mut t = t.borrow_mut();
            if t.1 == 0 {
                return true;
            }
            t.1 -= 1;
            false
        });
        if exhausted {
            panic!("comparison budget exhausted");
        }
        self.key.cmp(&other.key)
    }
}

/// Sorts keys in elements of `PAD` extra bytes whose comparisons start panicking after
/// `budget` of them, and checks that no element is lost, duplicated, moved, dropped or
/// leaked. Returns whether the sort completed.
fn sort_with_comparison_budget<const PAD: usize>(keys: &[u32], budget: usize) -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let len = keys.len();
    TRACKING.with(|t| *t.borrow_mut() = (vec![0; len], budget));
    let drops = || TRACKING.with(|t| t.borrow().0.clone());
    let mut v: Vec<Tracked<PAD>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Tracked {
            key,
            id,
            _pad: [0; PAD],
        })
        .collect();

    let result = catch_unwind(AssertUnwindSafe(|| tree(&mut v)));

    if result.is_ok() {
        // Sorted and stable: equal keys keep their ids in increasing order.
        assert!(
            v.windows(2)
                .all(|w| (w[0].key, w[0].id) < (w[1].key, w[1].id))
        );
    } else {
        // Nothing has moved yet when a comparison panics.
        assert!(
            v.iter().zip(0..).all(|(t, id)| t.id == id),
            "slice reordered"
        );
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
        drops().iter().all(|&d| d == 0),
        "element dropped during sort"
    );

    drop(v);
    assert!(
        drops().iter().all(|&d| d == 1),
        "element leaked or double-dropped"
    );
    result.is_ok()
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    // Panics land in searches from the root, side by side and alone, in searches that go on
    // after an earlier insertion, and in both comparisons that try the last insertion's
    // neighbourhood.
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
            (0..len as u32).rev().collect(),
            (0..len as u32).map(|id| id % 3).collect(),
            (0..len as u32).map(|id| id ^ 1).collect(),
            (0..len as u32).map(|i| i.min(len as u32 - 1 - i)).collect(),
            (1..len as u32).chain([0]).collect(),
        ]
    }
    // Every budget below the number of comparisons must panic, and only those: a panic from
    // anything else fails the test.
    fn check<const PAD: usize>(keys: &[u32], step: usize) {
        let total = comparisons_with::<PAD>(keys);
        for budget in (0..total).step_by(step).chain([total]) {
            let completed = sort_with_comparison_budget::<PAD>(keys, budget);
            assert_eq!(completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
    let mut state = 0x0123_4567_89AB_CDEF;
    for keys in layouts(size(60, 20), &mut state) {
        check::<P8>(&keys, 1);
        check::<P24>(&keys, 1);
        check::<P128>(&keys, 1);
    }
    // Four searches side by side.
    for keys in layouts(1030, &mut state) {
        check::<P8>(&keys, size(7, 997));
    }
}

thread_local! {
    /// The answer of every comparison between `Liar`s, and the generator of random ones.
    static LIES: Cell<(fn() -> Ordering, u64)> = const { Cell::new((less, 0x5EED_5EED_5EED_5EED)) };
}

fn less() -> Ordering {
    Ordering::Less
}

fn greater() -> Ordering {
    Ordering::Greater
}

fn equal() -> Ordering {
    Ordering::Equal
}

fn random() -> Ordering {
    let (answer, mut state) = LIES.with(Cell::get);
    let r = xorshift(&mut state);
    LIES.with(|l| l.set((answer, state)));
    [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize]
}

/// An element whose comparisons answer whatever `LIES` says, owning a box so that a lost or
/// duplicated element shows up as a leak or a double free, with `PAD` bytes of payload.
struct Liar<const PAD: usize> {
    name: Box<u32>,
    _pad: [u8; PAD],
}
impl<const PAD: usize> PartialEq for Liar<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for Liar<PAD> {}
impl<const PAD: usize> PartialOrd for Liar<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for Liar<PAD> {
    fn cmp(&self, _: &Self) -> Ordering {
        (LIES.with(Cell::get).0)()
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that is not a total order makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // terminate. Lies make searches side by side and from the last insertion end up anywhere.
    fn check<const PAD: usize>(lens: impl Iterator<Item = usize>) {
        for len in lens {
            for _ in 0..size(5, 1) {
                let mut v: Vec<Liar<PAD>> = (0..len as u32)
                    .map(|i| Liar {
                        name: Box::new(i),
                        _pad: [0; PAD],
                    })
                    .collect();

                tree(&mut v);

                let mut after: Vec<u32> = v.into_iter().map(|l| *l.name).collect();
                after.sort_unstable();
                assert!(after.into_iter().eq(0..len as u32));
            }
        }
    }

    let answers: [fn() -> Ordering; 4] = [less, greater, equal, random];
    for answer in answers {
        LIES.with(|l| l.set((answer, 0x5EED_5EED_5EED_5EED)));
        let lens = || (0..size(100, 30)).chain([size(2000, 1100)]);
        check::<P8>(lens());
        check::<P24>(lens());
        check::<P128>(lens());
    }
}

thread_local! {
    /// Comparisons made by the elements below, which are too small to carry a counter.
    static MUTATING_COMPARISONS: Cell<usize> = const { Cell::new(0) };
}

/// An element whose comparison mutates both of its arguments through interior mutability and
/// moves heap ownership while doing so: it takes the boxes out of both cells, drops them, and
/// puts fresh ones back with the comparison counted. A bitwise copy of an element made before
/// a later comparison of its twin is stale, and dropping it frees memory twice. 8 bytes plus
/// `PAD`, kept in the tree's nodes at 8 and 24 bytes.
struct Swapping<const PAD: usize>(Cell<Option<Box<(u32, u32, u32)>>>, [u8; PAD]);

impl<const PAD: usize> Swapping<PAD> {
    fn new(key: u32, idx: u32) -> Self {
        Self(Cell::new(Some(Box::new((key, idx, 0)))), [0; PAD])
    }
    /// The key, the starting index and the number of comparisons taken part in.
    fn get(&self) -> (u32, u32, u32) {
        let b = self.0.take().expect("box present");
        let t = *b;
        self.0.set(Some(b));
        t
    }
}
impl<const PAD: usize> PartialEq for Swapping<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for Swapping<PAD> {}
impl<const PAD: usize> PartialOrd for Swapping<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for Swapping<PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        assert!(
            !std::ptr::eq(self, other),
            "an element compared with itself"
        );
        MUTATING_COMPARISONS.with(|c| c.set(c.get() + 1));
        let a = self
            .0
            .take()
            .expect("box present: a stale copy was compared");
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

/// Sorts `keys` in `Swapping` elements of `PAD` extra bytes and checks that every mutation
/// made by a comparison is still there.
fn check_swapping<const PAD: usize>(keys: &[u32]) {
    let mut v: Vec<Swapping<PAD>> = keys
        .iter()
        .zip(0..)
        .map(|(&k, i)| Swapping::new(k, i))
        .collect();
    MUTATING_COMPARISONS.with(|c| c.set(0));
    tree(&mut v);
    let c = MUTATING_COMPARISONS.with(|c| c.get());
    let got: Vec<(u32, u32, u32)> = v.iter().map(Swapping::get).collect();
    assert!(
        got.windows(2).all(|w| (w[0].0, w[0].1) < (w[1].0, w[1].1)),
        "{}-byte: {keys:?}",
        8 + PAD
    );
    let mutations: usize = got.iter().map(|t| t.2 as usize).sum();
    assert_eq!(mutations, 2 * c, "{}-byte: a stale copy was kept", 8 + PAD);
}

/// A 40-byte element, compared where it is in the slice: every comparison replaces the log of
/// both arguments by a new, longer allocation, freeing the old one.
struct Reallocating {
    key: u32,
    idx: u32,
    log: RefCell<Vec<u8>>,
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
            !std::ptr::eq(self, other),
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
    // Miri reports). Few keys make searches side by side collide and go on again.
    let mut state = 0x5A7E_C0DE_5A7E_C0DE;
    let comparisons = || MUTATING_COMPARISONS.with(|c| c.replace(0));
    for len in (0..=size(70, 30)).chain([size(1100, 1030)]) {
        for modulus in [2u64, 5, 1000] {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % modulus) as u32)
                .collect();

            check_swapping::<P8>(&keys);
            check_swapping::<P24>(&keys);

            let mut v: Vec<Reallocating> = keys
                .iter()
                .zip(0..)
                .map(|(&key, idx)| Reallocating {
                    key,
                    idx,
                    log: RefCell::new(Vec::new()),
                })
                .collect();
            comparisons();
            tree(&mut v);
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

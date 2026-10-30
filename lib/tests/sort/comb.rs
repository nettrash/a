use a::sort::comb;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::mem::size_of;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Comb sort moves elements in one of three ways, chosen by their size: elements of at most
/// 16 bytes are carried in registers at the gaps of [`LANE_GAPS`], elements of at most 32
/// bytes are swapped without branching at the other gaps, and larger ones with a branch. The
/// test elements below carry `PAD` bytes of payload: `SMALL` makes them 9 to 32 bytes (16 for
/// [`ByKey`] and [`Counted`], which are carried; 32 for [`Watched`], which is not), `MEDIUM`
/// 17 to 32 bytes and `LARGE` more, and every test that depends on how elements are moved runs
/// with them, and with [`Tiny`] elements of 4 bytes, which keep their bookkeeping in
/// thread-local state to stay that small.
const SMALL: usize = 0;
const MEDIUM: usize = 16;
const LARGE: usize = 40;

/// The gaps at which elements of at most 16 bytes are carried in registers, when the slice
/// is at least four times as long as the gap.
const LANE_GAPS: [usize; 7] = [2, 3, 4, 6, 8, 11, 14];

#[test]
fn test_elements_cover_every_pass() {
    assert!(size_of::<Tiny>() <= 8);
    assert!(size_of::<ByKey<u64, SMALL>>() <= 16);
    assert!(size_of::<Counted<usize, SMALL>>() <= 16);
    for medium in [
        size_of::<ByKey<u64, MEDIUM>>(),
        size_of::<Counted<usize, MEDIUM>>(),
    ] {
        assert!((17..=32).contains(&medium), "{medium} bytes");
    }
    for small in [
        size_of::<ByKey<u64, SMALL>>(),
        size_of::<Counted<usize, SMALL>>(),
        size_of::<Watched<usize, SMALL>>(),
    ] {
        assert!((9..=32).contains(&small), "{small} bytes");
    }
    assert!(size_of::<ByKey<u64, LARGE>>() > 32);
    assert!(size_of::<Counted<usize, LARGE>>() > 32);
    assert!(size_of::<Watched<usize, LARGE>>() > 32);
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    comb(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    comb(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    comb(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    comb(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    comb(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    comb(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    comb(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..20 {
        let mut v = vec![7; len];
        comb(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    comb(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    comb(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
}

#[test]
fn sorts_large_elements() {
    let mut v: Vec<[u64; 8]> = (0..100u64)
        .rev()
        .map(|k| [k, 1, 2, 3, 4, 5, 6, 7])
        .collect();
    comb(&mut v);
    assert!(v.iter().map(|e| e[0]).eq(0..100));
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..17 {
        let mut v = vec![(); len];
        comb(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    let n = size(1000, 100) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    comb(&mut v);
    assert_eq!(v, (0..n).collect::<Vec<_>>());
}

#[test]
fn sorts_turtles_at_the_end() {
    for n in 2..size(300, 40) {
        let mut v: Vec<usize> = (n / 2..n).chain(0..n / 2).collect();
        comb(&mut v);
        assert_eq!(v, (0..n).collect::<Vec<_>>(), "n = {n}");
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v = vec![9, 8, 7, 3, 1, 2, 0, 6, 5, 4];
    comb(&mut v[3..7]);
    assert_eq!(v, vec![9, 8, 7, 0, 1, 2, 3, 6, 5, 4]);
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
/// length 9, and random sequences of distinct and of repeated keys up to length 257.
fn for_each_test_input(seed: u64, f: &mut impl FnMut(&[usize])) {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, f);
    }
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            let s: Vec<usize> = s.iter().map(|&k| k as usize).collect();
            f(&s);
        });
    }
    let mut state = seed;
    for n in [9, 16, 33, 100, size(257, 40)] {
        for _ in 0..size(20, 2) {
            let mut v: Vec<usize> = (0..n).collect();
            shuffle(&mut v, &mut state);
            f(&v);
            let v: Vec<usize> = (0..n).map(|_| xorshift(&mut state) as usize % 3).collect();
            f(&v);
        }
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
/// that every element is still there exactly once. Equal keys may come out in any order.
fn assert_sorts<K: Ord + Copy + std::fmt::Debug, const PAD: usize>(keys: &[K]) {
    let mut v: Vec<ByKey<K, PAD>> = keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| ByKey {
            key,
            idx,
            _pad: [0; PAD],
        })
        .collect();
    comb(&mut v);
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

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| {
            assert_sorts::<_, SMALL>(p);
            assert_sorts::<_, MEDIUM>(p);
            assert_sorts::<_, LARGE>(p);
        });
    }
}

#[test]
fn sorts_every_sequence_over_three_keys_up_to_9() {
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            assert_sorts::<_, SMALL>(s);
            assert_sorts::<_, MEDIUM>(s);
            assert_sorts::<_, LARGE>(s);
        });
    }
}

#[test]
fn sorts_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(100, 24) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(8, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts::<_, SMALL>(&keys);
                assert_sorts::<_, MEDIUM>(&keys);
                assert_sorts::<_, LARGE>(&keys);
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
fn comparisons<K: Ord + Clone + std::fmt::Debug, const PAD: usize>(keys: &[K]) -> usize {
    let count = Cell::new(0);
    let mut v: Vec<Counted<K, PAD>> = keys
        .iter()
        .map(|k| Counted {
            key: k.clone(),
            comparisons: &count,
            _pad: [0; PAD],
        })
        .collect();
    comb(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

/// The gaps comb sort uses for `n` elements: each the previous one (starting from `n`)
/// times 10/13, rounded down, with 9 and 10 replaced by 11, down to 2.
fn gaps(n: usize) -> Vec<usize> {
    let mut gaps = Vec::new();
    let mut gap = n;
    loop {
        gap = gap * 10 / 13;
        if gap == 9 || gap == 10 {
            gap = 11;
        }
        if gap <= 1 {
            return gaps;
        }
        gaps.push(gap);
    }
}

/// What classic comb sort does on some input.
struct Reference {
    /// The number of comparisons made.
    comparisons: usize,
    /// The number of element moves: two per gapped swap, plus, for the final bubble sort, one
    /// per inversion left by the gapped passes and one per carry (a maximal run of swaps in
    /// one pass that move the same element, which a hole-shifting sort does as one carry).
    moves: usize,
    /// The same, for elements carried in registers at the gaps of [`LANE_GAPS`] (for slices at
    /// least four times as long as the gap): there, a
    /// swap costs one move, plus one per carry along a chain (a maximal run of swaps in one
    /// pass at positions `gap` apart, which move the same element).
    moves_carried: usize,
}

/// Runs classic swap-based comb sort with the rule of 11, finished by bubble sort with early
/// exit and last-swap bound, on a copy of `keys`.
fn reference<K: Ord + Clone>(keys: &[K]) -> Reference {
    let mut v = keys.to_vec();
    let n = v.len();
    let (mut comparisons, mut moves, mut moves_carried) = (0, 0, 0);
    for gap in gaps(n) {
        // Whether the last pair of each chain was swapped.
        let mut carrying = vec![false; gap];
        for i in 0..n - gap {
            comparisons += 1;
            let swap = v[i + gap] < v[i];
            if swap {
                v.swap(i, i + gap);
                moves += 2;
                moves_carried += if LANE_GAPS.contains(&gap) && n >= 4 * gap {
                    1 + usize::from(!carrying[i % gap])
                } else {
                    2
                };
            }
            carrying[i % gap] = swap;
        }
    }
    // The final bubble sort moves elements the same way either way.
    let saved_by_carrying = moves - moves_carried;
    moves += inversions(&v);
    let mut hi = n;
    while hi > 1 {
        let mut last = 0;
        let mut carrying = false;
        for j in 1..hi {
            comparisons += 1;
            let swap = v[j] < v[j - 1];
            if swap {
                v.swap(j - 1, j);
                last = j;
                moves += usize::from(!carrying);
            }
            carrying = swap;
        }
        hi = last;
    }
    Reference {
        comparisons,
        moves,
        moves_carried: moves - saved_by_carrying,
    }
}

/// The number of pairs of elements that are out of order.
fn inversions<K: Ord>(keys: &[K]) -> usize {
    (0..keys.len())
        .map(|j| keys[..j].iter().filter(|&e| *e > keys[j]).count())
        .sum()
}

#[test]
fn gaps_follow_the_rule_of_11() {
    assert_eq!(gaps(100), [76, 58, 44, 33, 25, 19, 14, 11, 8, 6, 4, 3, 2]);
    assert_eq!(gaps(14), [11, 8, 6, 4, 3, 2]);
    assert_eq!(gaps(13), [11, 8, 6, 4, 3, 2]);
    assert_eq!(gaps(12), [11, 8, 6, 4, 3, 2]);
    assert_eq!(gaps(2), []);
    for n in 0..2000 {
        assert!(!gaps(n).contains(&9) && !gaps(n).contains(&10), "n = {n}");
    }
}

#[test]
fn comparisons_match_classic_comb_sort() {
    for_each_test_input(0xC0C7_A11E_5AAE_0001, &mut |v| {
        let expected = reference(v).comparisons;
        assert_eq!(comparisons::<_, SMALL>(v), expected, "{v:?}");
        assert_eq!(comparisons::<_, MEDIUM>(v), expected, "{v:?}");
        assert_eq!(comparisons::<_, LARGE>(v), expected, "{v:?}");
    });
}

#[test]
fn sorted_input_costs_the_gapped_passes_plus_n_minus_1() {
    for n in 0..size(300, 40) {
        let expected: usize = gaps(n).iter().map(|g| n - g).sum::<usize>() + n.saturating_sub(1);
        let v: Vec<usize> = (0..n).collect();
        assert_eq!(comparisons::<_, SMALL>(&v), expected, "n = {n}");
        assert_eq!(comparisons::<_, LARGE>(&v), expected, "n = {n}");
        assert_eq!(comparisons::<_, SMALL>(&vec![7; n]), expected, "n = {n}");
    }
}

#[test]
fn random_input_costs_about_n_log_n_comparisons() {
    // `≈ n·(log₁.₃ n − 4.5)` on average: 21,300 for `n = 1000`, and never far above it.
    let n = size(1000, 100);
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    let bound = n as f64 * ((n as f64).log(1.3) - 3.5);
    for _ in 0..size(20, 1) {
        let mut v: Vec<usize> = (0..n).collect();
        shuffle(&mut v, &mut state);
        let c = comparisons::<_, SMALL>(&v);
        assert!((c as f64) < bound, "{c} comparisons, bound {bound}");
    }
}

/// An element that notices when the sort has moved it: every comparison checks its address
/// against the one it was last seen at. Rust moves are plain copies, so several moves between
/// two comparisons count as one (and a move there and back as none): the count is a lower
/// bound on the element moves the sort made. Only slots of the slice count: an element compared
/// while it is held outside the slice, on its way from one slot to another, is not seen there.
struct Watched<'a, K, const PAD: usize> {
    key: K,
    seen_at: Cell<usize>,
    moves: &'a Cell<usize>,
    slots: &'a Cell<(usize, usize)>,
    _pad: [u8; PAD],
}

impl<K, const PAD: usize> Watched<'_, K, PAD> {
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

impl<K: Ord, const PAD: usize> PartialEq for Watched<'_, K, PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord, const PAD: usize> Eq for Watched<'_, K, PAD> {}
impl<K: Ord, const PAD: usize> PartialOrd for Watched<'_, K, PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord, const PAD: usize> Ord for Watched<'_, K, PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.check();
        other.check();
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys` and returns the number of element moves observed, including the final one
/// of every element that ends up away from where it was last seen.
fn observed_moves<K: Ord + Copy + std::fmt::Debug, const PAD: usize>(keys: &[K]) -> usize {
    let moves = Cell::new(0);
    let slots = Cell::new((0, 0));
    let mut v: Vec<Watched<K, PAD>> = keys
        .iter()
        .map(|&key| Watched {
            key,
            seen_at: Cell::new(0),
            moves: &moves,
            slots: &slots,
            _pad: [0; PAD],
        })
        .collect();
    let start = v.as_ptr() as usize;
    slots.set((start, start + std::mem::size_of_val(v.as_slice())));
    for w in &v {
        w.seen_at.set(w as *const Watched<K, PAD> as usize);
    }
    comb(&mut v);
    for w in &v {
        w.check();
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moves.get()
}

#[test]
fn non_decreasing_input_is_never_moved() {
    // Zero observed moves means every element sat at its starting address whenever it was
    // compared and is still there at the end: the branchless pass rewrites a pair that needs
    // no swap with itself, which leaves both where they were. Every non-decreasing sequence
    // over four keys up to length 12 (as cut points between runs), plus long ones.
    let check = |v: &[u8]| {
        assert_eq!(observed_moves::<_, SMALL>(v), 0, "{v:?}");
        assert_eq!(observed_moves::<_, LARGE>(v), 0, "{v:?}");
    };
    for len in 0..=size(12, 6) {
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
    for n in 0..50 {
        check(&vec![7; n]);
    }
}

#[test]
fn moves_match_classic_comb_sort() {
    // Every element is compared in its slot before it is swapped, shifted or picked up, and
    // is held outside the slice only while the final bubble sort carries it, so the observed
    // count is exact here: two per gapped swap, then one per inversion and one per carry.
    for_each_test_input(0xFACE_FEED_0DD5_EED5, &mut |v| {
        let expected = reference(v).moves;
        assert_eq!(observed_moves::<_, SMALL>(v), expected, "{v:?}");
        assert_eq!(observed_moves::<_, LARGE>(v), expected, "{v:?}");
    });
}

/// Sorts elements of `PAD` extra bytes whose comparisons start panicking after `budget` of
/// them, and checks that no element is lost, duplicated, dropped or leaked. Returns whether
/// the sort completed.
fn sort_with_comparison_budget<const PAD: usize>(keys: &[u32], budget: usize) -> bool {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    struct Tracked<'a, const PAD: usize> {
        key: u32,
        id: usize,
        drops: Rc<Vec<Cell<u32>>>,
        comparisons_left: &'a Cell<usize>,
        _pad: [u8; PAD],
    }
    impl<const PAD: usize> Drop for Tracked<'_, PAD> {
        fn drop(&mut self) {
            let count = &self.drops[self.id];
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
    let mut v: Vec<Tracked<PAD>> = keys
        .iter()
        .enumerate()
        .map(|(id, &key)| Tracked {
            key,
            id,
            drops: Rc::clone(&drops),
            comparisons_left: &comparisons_left,
            _pad: [0; PAD],
        })
        .collect();

    let result = catch_unwind(AssertUnwindSafe(|| comb(&mut v)));

    if result.is_ok() {
        assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
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
    let len = size(32, 12);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    // Panics land in every gapped pass and in the final bubble sort: reverse order and the
    // random keys swap often in both, repeated keys stop bubble carries at equal elements.
    let layouts: Vec<Vec<u32>> = vec![
        (0..len as u32).rev().collect(),
        (0..len as u32).map(|id| id % 3).collect(),
        (0..len as u32).map(|id| (len as u32 - id) / 3).collect(),
        random,
        (0..len as u32).map(|id| id ^ 1).collect(),
        (1..len as u32).chain([0]).collect(),
    ];
    for keys in layouts {
        // Every budget below the full number of comparisons must panic, and only those: a
        // panic from anything else fails the test instead of trying ever larger budgets.
        let total = comparisons::<_, SMALL>(&keys);
        for budget in 0..=total {
            let small = sort_with_comparison_budget::<SMALL>(&keys, budget);
            let large = sort_with_comparison_budget::<LARGE>(&keys, budget);
            assert_eq!(small, budget == total, "budget {budget}: {keys:?}");
            assert_eq!(large, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that answers at random makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // still terminate.
    struct Liar<'a, const PAD: usize> {
        name: String,
        rng: &'a Cell<u64>,
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
            let mut state = self.rng.get();
            let r = xorshift(&mut state);
            self.rng.set(state);
            [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize]
        }
    }

    fn check<const PAD: usize>(rng: &Cell<u64>) {
        for len in 0..size(64, 20) {
            for _ in 0..size(50, 3) {
                let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
                let mut v: Vec<Liar<PAD>> = names
                    .iter()
                    .map(|name| Liar {
                        name: name.clone(),
                        rng,
                        _pad: [0; PAD],
                    })
                    .collect();

                comb(&mut v);

                let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
                after.sort_unstable();
                let mut names = names;
                names.sort_unstable();
                assert_eq!(after, names);
            }
        }
    }

    let rng = Cell::new(0x5EED_5EED_5EED_5EED);
    check::<SMALL>(&rng);
    check::<LARGE>(&rng);
}

/// A 4-byte element, small enough to be carried in registers at small gaps. It is compared by
/// `key` only and carries the index it started at; everything else it needs — comparison
/// count and budget, move and drop tracking, an optional random `Ord` — lives in [`TINY`].
struct Tiny {
    key: u16,
    id: u16,
}

/// The bookkeeping of the [`Tiny`] elements of the sort running on this thread.
#[derive(Default)]
struct TinyState {
    comparisons: Cell<usize>,
    /// Comparisons left before one panics, if limited.
    budget: Cell<Option<usize>>,
    /// The state of a random generator that answers comparisons instead of the keys, if set.
    liar: Cell<Option<u64>>,
    /// Moves observed as in [`Watched`]: the address each element was last seen at, within
    /// the slice `slots`.
    moves: Cell<usize>,
    slots: Cell<(usize, usize)>,
    seen_at: RefCell<Vec<usize>>,
    drops: RefCell<Vec<u32>>,
}

thread_local! {
    static TINY: TinyState = TinyState::default();
}

impl TinyState {
    fn observe(&self, t: &Tiny) {
        let here = t as *const Tiny as usize;
        let (start, end) = self.slots.get();
        if !(start..end).contains(&here) {
            return;
        }
        let mut seen_at = self.seen_at.borrow_mut();
        if seen_at[t.id as usize] != here {
            seen_at[t.id as usize] = here;
            self.moves.set(self.moves.get() + 1);
        }
    }
}

impl Drop for Tiny {
    fn drop(&mut self) {
        TINY.with(|s| s.drops.borrow_mut()[self.id as usize] += 1);
    }
}
impl PartialEq for Tiny {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Tiny {}
impl PartialOrd for Tiny {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Tiny {
    fn cmp(&self, other: &Self) -> Ordering {
        TINY.with(|s| {
            if let Some(left) = s.budget.get() {
                if left == 0 {
                    panic!("comparison budget exhausted");
                }
                s.budget.set(Some(left - 1));
            }
            s.comparisons.set(s.comparisons.get() + 1);
            s.observe(self);
            s.observe(other);
            if let Some(mut state) = s.liar.get() {
                let r = xorshift(&mut state);
                s.liar.set(Some(state));
                return [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize];
            }
            self.key.cmp(&other.key)
        })
    }
}

/// What happened when [`Tiny`] elements were sorted.
struct TinyRun {
    comparisons: usize,
    moves: usize,
    completed: bool,
}

/// Sorts `keys` as [`Tiny`] elements, with comparisons panicking after `budget` of them and
/// answered at random from `liar` if those are set. Checks that every element is still there
/// exactly once, unchanged, and dropped exactly once afterwards, and that the result is sorted
/// if the sort completed with honest comparisons.
fn sort_tiny(keys: &[usize], budget: Option<usize>, liar: Option<u64>) -> TinyRun {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let n = keys.len();
    let mut v: Vec<Tiny> = keys
        .iter()
        .enumerate()
        .map(|(id, &key)| Tiny {
            key: key as u16,
            id: id as u16,
        })
        .collect();
    TINY.with(|s| {
        s.comparisons.set(0);
        s.budget.set(budget);
        s.liar.set(liar);
        s.moves.set(0);
        let start = v.as_ptr() as usize;
        s.slots.set((start, start + size_of::<Tiny>() * n));
        *s.seen_at.borrow_mut() = v.iter().map(|t| t as *const Tiny as usize).collect();
        *s.drops.borrow_mut() = vec![0; n];
    });

    let completed = catch_unwind(AssertUnwindSafe(|| comb(&mut v))).is_ok();

    let run = TINY.with(|s| {
        s.budget.set(None);
        for t in &v {
            s.observe(t);
        }
        TinyRun {
            comparisons: s.comparisons.get(),
            moves: s.moves.get(),
            completed,
        }
    });
    if completed && liar.is_none() {
        assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    }
    let mut ids: Vec<usize> = v.iter().map(|t| t.id as usize).collect();
    ids.sort_unstable();
    assert!(ids.into_iter().eq(0..n), "element lost or duplicated");
    assert!(
        v.iter().all(|t| t.key as usize == keys[t.id as usize]),
        "element changed during sort"
    );
    TINY.with(|s| assert!(s.drops.borrow().iter().all(|&d| d == 0), "dropped"));
    drop(v);
    TINY.with(|s| {
        assert!(
            s.drops.borrow().iter().all(|&d| d == 1),
            "element leaked or double-dropped"
        )
    });
    run
}

#[test]
fn tiny_elements_match_classic_comb_sort() {
    // Carrying along chains makes the same comparisons and one move per swap plus one per
    // carry. Every element is compared in its slot before it is shifted or picked up, the
    // head of a chain included, so the observed count is exact.
    for_each_test_input(0x7111_0E1E_3E17_5EED, &mut |v| {
        let expected = reference(v);
        let run = sort_tiny(v, None, None);
        assert_eq!(run.comparisons, expected.comparisons, "{v:?}");
        assert_eq!(run.moves, expected.moves_carried, "{v:?}");
    });
}

#[test]
fn tiny_non_decreasing_input_is_never_moved() {
    // A carried element that turns out not to be the larger is written back where it was.
    for n in 0..size(300, 40) {
        let sorted: Vec<usize> = (0..n).collect();
        assert_eq!(sort_tiny(&sorted, None, None).moves, 0, "n = {n}");
        let runs: Vec<usize> = (0..n).map(|i| i * 4 / n.max(1)).collect();
        assert_eq!(sort_tiny(&runs, None, None).moves, 0, "{runs:?}");
    }
}

#[test]
fn tiny_panicking_comparison_keeps_every_element_exactly_once() {
    // Panics land at every step of every chain, the first comparison of a chain (made with
    // both elements in place) included.
    let len = size(32, 12);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<usize> = (0..len)
        .map(|_| xorshift(&mut state) as usize % 7)
        .collect();
    let layouts: Vec<Vec<usize>> = vec![
        (0..len).rev().collect(),
        (0..len).map(|id| id % 3).collect(),
        random,
        (0..len).map(|id| id ^ 1).collect(),
        (1..len).chain([0]).collect(),
    ];
    for keys in layouts {
        let total = sort_tiny(&keys, None, None).comparisons;
        for budget in 0..=total {
            let run = sort_tiny(&keys, Some(budget), None);
            assert_eq!(run.completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn tiny_inconsistent_ord_keeps_every_element_exactly_once() {
    let mut seed = 0x5EED_5EED_5EED_5EED;
    for len in 0..size(64, 20) {
        for _ in 0..size(50, 3) {
            let keys: Vec<usize> = (0..len).collect();
            sort_tiny(&keys, None, Some(xorshift(&mut seed)));
        }
    }
}

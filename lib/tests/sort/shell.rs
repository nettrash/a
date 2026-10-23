use a::sort::shell;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::mem::size_of;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Shell sort sorts elements of at most 8 bytes with Ciura's gaps, inserting through a
/// branchless window while the slice is far from sorted, and larger elements with Sedgewick's
/// gaps by plain insertion, which makes exactly the comparisons of classic Shell sort. The
/// test elements below carry `PAD` bytes of payload: `SMALL` makes them 9 to 32 bytes and
/// `LARGE` more, both taking the plain path, and every test runs with both; [`Tiny`] elements
/// of 4 bytes, which keep their bookkeeping in thread-local state to stay that small, take the
/// window.
const SMALL: usize = 0;
const LARGE: usize = 40;

#[test]
fn test_elements_cover_every_pass() {
    assert!(size_of::<Tiny>() <= 8);
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
    shell(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    shell(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    shell(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    shell(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    shell(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    shell(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    shell(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..20 {
        let mut v = vec![7; len];
        shell(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    shell(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    shell(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
}

#[test]
fn sorts_large_elements() {
    let mut v: Vec<[u64; 8]> = (0..100u64)
        .rev()
        .map(|k| [k, 1, 2, 3, 4, 5, 6, 7])
        .collect();
    shell(&mut v);
    assert!(v.iter().map(|e| e[0]).eq(0..100));
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..17 {
        let mut v = vec![(); len];
        shell(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    let n = size(1000, 100) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    shell(&mut v);
    assert_eq!(v, (0..n).collect::<Vec<_>>());
}

#[test]
fn sorts_long_random_input() {
    // Long enough for Ciura's sequence to be extended past 701, and the window to run many
    // passes; `u64` takes the window, `ByKey` the plain path.
    let n = size(100_000, 3000);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let keys: Vec<u64> = (0..n).map(|_| xorshift(&mut state)).collect();
    let mut v = keys.clone();
    let mut expected = keys.clone();
    expected.sort_unstable();
    shell(&mut v);
    assert_eq!(v, expected);
    for modulus in [3, 1000] {
        let keys: Vec<u64> = keys.iter().map(|k| k % modulus).collect();
        let mut v = keys.clone();
        let mut expected = keys.clone();
        expected.sort_unstable();
        shell(&mut v);
        assert_eq!(v, expected);
        assert_sorts::<_, SMALL>(&keys[..size(5000, 300)]);
    }
}

#[test]
fn sorts_turtles_at_the_end() {
    for n in 2..size(300, 40) {
        let mut v: Vec<usize> = (n / 2..n).chain(0..n / 2).collect();
        shell(&mut v);
        assert_eq!(v, (0..n).collect::<Vec<_>>(), "n = {n}");
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v = vec![9, 8, 7, 3, 1, 2, 0, 6, 5, 4];
    shell(&mut v[3..7]);
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
    shell(&mut v);
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
            assert_sorts::<_, LARGE>(p);
        });
    }
}

#[test]
fn sorts_every_sequence_over_three_keys_up_to_9() {
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            assert_sorts::<_, SMALL>(s);
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
    shell(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

/// The gaps Shell sort uses for `n` elements of more than 8 bytes, largest first:
/// Sedgewick's 1982 sequence, `1` and `4^k + 3·2^(k−1) + 1`, below `n`.
fn gaps(n: usize) -> Vec<usize> {
    let mut gaps = vec![1];
    let mut k = 1;
    loop {
        let gap = 4usize.pow(k) + 3 * 2usize.pow(k - 1) + 1;
        if gap >= n {
            break;
        }
        gaps.push(gap);
        k += 1;
    }
    gaps.retain(|&g| g < n);
    gaps.reverse();
    gaps
}

/// The gaps Shell sort uses for `n` elements of at most 8 bytes, largest first: Ciura's
/// sequence, `1, 4, 10, 23, 57, 132, 301, 701`, then times 2.25 rounded down, below `n`.
fn ciura_gaps(n: usize) -> Vec<usize> {
    let mut gaps = vec![1, 4, 10, 23, 57, 132, 301, 701];
    while gaps[gaps.len() - 1] < n {
        let next = gaps[gaps.len() - 1] * 9 / 4;
        gaps.push(next);
    }
    gaps.retain(|&g| g < n);
    gaps.reverse();
    gaps
}

/// The comparisons classic Shell sort makes on `keys` with Sedgewick's gaps, those of
/// [`gaps`].
fn reference(keys: &[usize]) -> usize {
    classic_comparisons(keys, &gaps(keys.len()))
}

/// The comparisons classic Shell sort makes on `keys` with `gaps`, after the scan for one
/// ascending or strictly descending run that covers the whole slice: one h-insertion pass per
/// gap, each element compared with the one `gap` before it and, while that one is greater,
/// carried on past it.
fn classic_comparisons(keys: &[usize], gaps: &[usize]) -> usize {
    let mut v = keys.to_vec();
    let n = v.len();
    if n < 2 {
        return 0;
    }
    let descending = v[1] < v[0];
    let mut comparisons = 1;
    let mut run = 2;
    while run < n {
        comparisons += 1;
        let continues = if descending {
            v[run] < v[run - 1]
        } else {
            v[run] >= v[run - 1]
        };
        if !continues {
            break;
        }
        run += 1;
    }
    if run == n {
        return comparisons;
    }
    for &gap in gaps {
        for i in gap..n {
            let x = v[i];
            let mut j = i;
            while j >= gap {
                comparisons += 1;
                if x >= v[j - gap] {
                    break;
                }
                v[j] = v[j - gap];
                j -= gap;
            }
            v[j] = x;
        }
    }
    comparisons
}

#[test]
fn gaps_follow_sedgewick_1982() {
    assert_eq!(gaps(2), [1]);
    assert_eq!(gaps(9), [8, 1]);
    assert_eq!(gaps(100_000), [65921, 16577, 4193, 1073, 281, 77, 23, 8, 1]);
}

#[test]
fn comparisons_match_classic_shell_sort() {
    for_each_test_input(0xC0C7_A11E_5AAE_0001, &mut |v| {
        let expected = reference(v);
        assert_eq!(comparisons::<_, SMALL>(v), expected, "{v:?}");
        assert_eq!(comparisons::<_, LARGE>(v), expected, "{v:?}");
    });
}

#[test]
fn sorted_reverse_sorted_and_equal_input_costs_n_minus_1_comparisons() {
    for n in 0..size(1000, 100) {
        let expected = n.saturating_sub(1);
        let sorted: Vec<usize> = (0..n).collect();
        let reverse: Vec<usize> = (0..n).rev().collect();
        assert_eq!(comparisons::<_, SMALL>(&sorted), expected, "n = {n}");
        assert_eq!(comparisons::<_, LARGE>(&reverse), expected, "n = {n}");
        assert_eq!(comparisons::<_, SMALL>(&vec![7; n]), expected, "n = {n}");
    }
}

#[test]
fn random_input_costs_about_1_6_n_log2_n_comparisons() {
    // About 15,700 for `n = 1000` with Sedgewick's gaps, and never far above it.
    let n = size(1000, 100);
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    let bound = 1.75 * n as f64 * (n as f64).log2();
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
    shell(&mut v);
    for w in &v {
        w.check();
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moves.get()
}

#[test]
fn non_decreasing_input_is_never_moved() {
    // Zero observed moves means every element sat at its starting address whenever it was
    // compared and is still there at the end. Every non-decreasing sequence over four keys up
    // to length 12 (as cut points between runs), plus long ones.
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

    let result = catch_unwind(AssertUnwindSafe(|| shell(&mut v)));

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
    // Panics land in the run scan and in every pass, at the first comparison of an insertion
    // and further along its chain.
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

                shell(&mut v);

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

/// A 4-byte element, small enough to be inserted through the branchless window. It is compared by
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

    let completed = catch_unwind(AssertUnwindSafe(|| shell(&mut v))).is_ok();

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
fn tiny_elements_sort() {
    for_each_test_input(0x7111_0E1E_3E17_5EED, &mut |v| {
        assert!(sort_tiny(v, None, None).completed);
    });
    let mut state = 0x7111_7111_7111_7111;
    for n in [100, size(1000, 200), size(5000, 300)] {
        for modulus in [2, 16, 1 << 16] {
            let v: Vec<usize> = (0..n)
                .map(|_| (xorshift(&mut state) % modulus) as usize)
                .collect();
            sort_tiny(&v, None, None);
        }
        let reversed: Vec<usize> = (0..n).rev().collect();
        sort_tiny(&reversed, None, None);
    }
}

#[test]
fn tiny_non_decreasing_input_is_never_moved() {
    // The window writes elements that do not move back where they were.
    for n in 0..size(300, 40) {
        let sorted: Vec<usize> = (0..n).collect();
        assert_eq!(sort_tiny(&sorted, None, None).moves, 0, "n = {n}");
        let runs: Vec<usize> = (0..n).map(|i| i * 4 / n.max(1)).collect();
        assert_eq!(sort_tiny(&runs, None, None).moves, 0, "{runs:?}");
    }
}

#[test]
fn tiny_panicking_comparison_keeps_every_element_exactly_once() {
    // Panics land in the window's comparisons, made before it writes anything, and in the
    // insertion that goes on past it: at 64 elements the pass with gap 10 takes the window
    // for its last 24 elements (the gap of 4 too, if those were hard to predict), and the
    // last element of the rotated input goes past it.
    let len = size(64, 24);
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

/// Organ-pipe input, rising to the middle and falling after it.
fn organ_pipe(n: usize) -> Vec<usize> {
    (0..n).map(|i| if i < n / 2 { i } else { n - i }).collect()
}

#[test]
fn tiny_patterned_input_is_inserted_plainly() {
    // Organ-pipe input moves elements in long runs of one step each, which plain insertion
    // predicts well: no block takes the window, so the comparisons are exactly those of
    // classic Shell sort with Ciura's gaps, also in the blocks of the head of each pass and
    // in the last elements of each block, which note their steps.
    for n in [200, size(1000, 300), size(5000, 400)] {
        let keys = organ_pipe(n);
        let expected = classic_comparisons(&keys, &ciura_gaps(n));
        assert_eq!(
            sort_tiny(&keys, None, None).comparisons,
            expected,
            "n = {n}"
        );
    }
    let n = size(2000, 1000);
    let sawtooth: Vec<usize> = (0..n).map(|i| i % (n / 8)).collect();
    let expected = classic_comparisons(&sawtooth, &ciura_gaps(n));
    assert_eq!(sort_tiny(&sawtooth, None, None).comparisons, expected);
}

#[test]
fn tiny_random_input_takes_the_window() {
    // The window compares each element with the four before it on its chain at once: more
    // comparisons than classic Shell sort, which stops at the first that is not greater.
    let n = size(1000, 300);
    let mut state = 0x7A6E_0F0E_11CE_5EED;
    let keys: Vec<usize> = (0..n)
        .map(|_| xorshift(&mut state) as usize % 60_000)
        .collect();
    let classic = classic_comparisons(&keys, &ciura_gaps(n));
    let run = sort_tiny(&keys, None, None);
    assert!(
        run.comparisons > classic * 3 / 2,
        "{} against {classic}",
        run.comparisons
    );
}

#[test]
fn tiny_panicking_comparison_in_long_passes_keeps_every_element_exactly_once() {
    // 200 elements make blocks of more than 64 elements, whose last 32 note their steps:
    // random keys take the window there, organ-pipe keys plain insertion, and sawtooth keys
    // both, with steps that repeat with the period of 3 or 4 in the last elements of a block
    // inserted either way.
    let len = 200;
    let mut state = 0x0FED_CBA9_8765_4321;
    let layouts: Vec<Vec<usize>> = vec![
        (0..len)
            .map(|_| xorshift(&mut state) as usize % 1000)
            .collect(),
        organ_pipe(len),
        (0..len).map(|id| id % 3).collect(),
        (0..len).map(|id| id % 4).collect(),
    ];
    for keys in layouts {
        let total = sort_tiny(&keys, None, None).comparisons;
        for budget in (0..=total).step_by(size(1, 61)) {
            let run = sort_tiny(&keys, Some(budget), None);
            assert_eq!(run.completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn tiny_inconsistent_ord_in_long_passes_keeps_every_element_exactly_once() {
    // Answers at random make the blocks choose either way and the steps look random.
    let mut seed = 0x1DE5_1DE5_1DE5_1DE5;
    for len in [100, 200, size(700, 150), size(3000, 200)] {
        for _ in 0..size(5, 1) {
            let keys: Vec<usize> = (0..len).collect();
            sort_tiny(&keys, None, Some(xorshift(&mut seed)));
        }
    }
}

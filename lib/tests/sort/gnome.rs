use a::sort::gnome;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    gnome(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    gnome(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    gnome(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    gnome(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    gnome(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    gnome(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    gnome(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..20 {
        let mut v = vec![7; len];
        gnome(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    gnome(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    gnome(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..17 {
        let mut v = vec![(); len];
        gnome(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    let n = size(1000, 100) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    gnome(&mut v);
    assert_eq!(v, (0..n).collect::<Vec<_>>());
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
    gnome(&mut actual);
    let actual: Vec<(K, usize)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert_eq!(actual, expected, "input: {keys:?}");
}

#[test]
fn is_stable_for_equal_keys() {
    // Equal keys passed by nothing, met by a carried element, and carried to the front.
    assert_sorts_like_std_stable(&[2, 1, 2, 1, 2]);
    assert_sorts_like_std_stable(&[1, 1, 1, 0, 0]);
    assert_sorts_like_std_stable(&[0, 1, 1, 1, 1, 0, 0]);
    assert_sorts_like_std_stable(&[5, 5, 3, 3, 5, 3, 5, 3]);
}

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts_like_std_stable(p));
    }
}

#[test]
fn sorts_every_sequence_over_three_keys_up_to_9_stably() {
    // Few keys put equal elements next to every carried element, so the index tags check
    // stability exhaustively.
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_like_std_stable(s));
    }
}

#[test]
fn matches_std_stable_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(100, 24) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(8, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts_like_std_stable(&keys);
            }
        }
    }
}

/// A key that records every comparison made between two of them, as the pair of indices the
/// two elements started at, smaller index first.
#[derive(Debug, Clone)]
struct Recorded<'a, K> {
    key: K,
    idx: usize,
    log: &'a RefCell<Vec<(usize, usize)>>,
}
impl<K: Ord> PartialEq for Recorded<'_, K> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<K: Ord> Eq for Recorded<'_, K> {}
impl<K: Ord> PartialOrd for Recorded<'_, K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<K: Ord> Ord for Recorded<'_, K> {
    fn cmp(&self, other: &Self) -> Ordering {
        let pair = (self.idx.min(other.idx), self.idx.max(other.idx));
        self.log.borrow_mut().push(pair);
        self.key.cmp(&other.key)
    }
}

/// Sorts a copy of `keys`, checks the result, and returns the comparisons made, in order.
fn comparison_log<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> Vec<(usize, usize)> {
    let log = RefCell::new(Vec::new());
    let mut v: Vec<Recorded<K>> = keys
        .iter()
        .enumerate()
        .map(|(idx, k)| Recorded {
            key: k.clone(),
            idx,
            log: &log,
        })
        .collect();
    gnome(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    log.into_inner()
}

/// Sorts a copy of `keys`, checks the result, and returns the number of comparisons.
fn comparisons<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> usize {
    comparison_log(keys).len()
}

/// Runs classic gnome sort — one step forward when a pair is in order, a swap and one step
/// back when it is not — on a copy of `keys`, and returns the comparisons it makes, in the
/// same form as [`comparison_log`].
fn reference_log<K: Ord + Clone>(keys: &[K]) -> Vec<(usize, usize)> {
    let mut v: Vec<(K, usize)> = keys.iter().cloned().zip(0..).collect();
    let mut log = Vec::new();
    let mut pos = 0;
    while pos < v.len() {
        if pos == 0 {
            pos += 1;
            continue;
        }
        let (a, b) = (v[pos - 1].1, v[pos].1);
        log.push((a.min(b), a.max(b)));
        if v[pos].0 < v[pos - 1].0 {
            v.swap(pos - 1, pos);
            pos -= 1;
        } else {
            pos += 1;
        }
    }
    log
}

/// The number of pairs of elements that are out of order.
fn inversions<K: Ord>(keys: &[K]) -> usize {
    (0..keys.len())
        .map(|j| keys[..j].iter().filter(|&e| *e > keys[j]).count())
        .sum()
}

/// The number of elements with a greater element somewhere before them: those that move.
fn carried<K: Ord>(keys: &[K]) -> usize {
    (0..keys.len())
        .filter(|&j| keys[..j].iter().any(|e| *e > keys[j]))
        .count()
}

/// The number of elements, other than the first, that are smaller than every element before
/// them: those carried all the way to the front.
fn carried_to_front<K: Ord>(keys: &[K]) -> usize {
    (1..keys.len())
        .filter(|&j| keys[..j].iter().all(|e| *e > keys[j]))
        .count()
}

#[test]
fn comparisons_are_classic_gnome_sort_minus_one_per_inversion() {
    for_each_test_input(0xC0C7_A11E_5AAE_0001, &mut |v| {
        assert_eq!(
            comparisons(v) + inversions(v),
            reference_log(v).len(),
            "{v:?}"
        );
    });
}

#[test]
fn every_comparison_is_one_classic_gnome_sort_makes_in_the_same_order() {
    // Teleporting only skips the comparisons of the walk forward: what is left must be a
    // subsequence of what classic gnome sort compares.
    for_each_test_input(0x6E0B_E5EE_D5A1_7ED0, &mut |v| {
        let reference = reference_log(v);
        let mut rest = reference.iter();
        for pair in comparison_log(v) {
            assert!(rest.any(|r| *r == pair), "{pair:?} out of order for {v:?}");
        }
    });
}

#[test]
fn comparisons_are_n_minus_1_plus_inversions_minus_front_carries() {
    for_each_test_input(0x0BAD_C0FF_EE00_D00D, &mut |v| {
        let n = v.len();
        assert_eq!(
            comparisons(v),
            (n + inversions(v)).saturating_sub(1) - carried_to_front(v),
            "{v:?}"
        );
    });
}

#[test]
fn sorted_input_costs_n_minus_1_comparisons() {
    for n in 0..size(300, 40) {
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
fn reverse_sorted_input_costs_half_of_classic_gnome_sort() {
    // `n(n - 1)/2` comparisons, against the `n(n - 1)` of classic gnome sort.
    for n in 0..size(300, 40) {
        let v: Vec<usize> = (0..n).rev().collect();
        assert_eq!(comparisons(&v), n * n.saturating_sub(1) / 2, "n = {n}");
        assert_eq!(reference_log(&v).len(), n * n.saturating_sub(1), "n = {n}");
    }
}

#[test]
fn turtle_at_the_end_costs_2n_minus_3_comparisons() {
    // `[1, 2, ..., n - 1, 0]`: `n - 2` steps forward, then one walk back of `n - 1` slots
    // that needs no comparison to stop at the front. Classic gnome sort walks forward again
    // over all of them, for `3n - 4`.
    for n in 2..size(300, 40) {
        let v: Vec<usize> = (1..n).chain([0]).collect();
        assert_eq!(comparisons(&v), 2 * n - 3, "n = {n}");
        assert_eq!(reference_log(&v).len(), 3 * n - 4, "n = {n}");
    }
}

/// An element that notices when the sort has moved it: every comparison checks its address
/// against the one it was last seen at. Rust moves are plain copies, so several moves between
/// two comparisons count as one (and a move there and back as none): the count is a lower
/// bound on the element moves the sort made. Only slots of the slice count: an element compared
/// while it is held outside the slice, on its way from one slot to another, is not seen there.
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

/// Sorts `keys` and returns the number of element moves observed, including the final one
/// of every element that ends up away from where it was last seen.
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
    gnome(&mut v);
    for w in &v {
        w.check();
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moves.get()
}

#[test]
fn non_decreasing_input_is_never_moved() {
    // Zero observed moves means every element sat at its starting address whenever it was
    // compared and is still there at the end. Every non-decreasing sequence over four keys
    // up to length 12 (as cut points between runs), plus long ones.
    let check = |v: &[u8]| assert_eq!(observed_moves(v), 0, "{v:?}");
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
fn swapped_neighbours_move_only_their_own_two_elements() {
    // `[1, 0, 3, 2, 5, 4, ...]`: each smaller element is carried one slot left, which moves
    // each pair's two elements once and nothing else.
    for n in 0..size(200, 30) {
        let mut v: Vec<usize> = (0..n).collect();
        for pair in v.as_chunks_mut::<2>().0 {
            pair.reverse();
        }
        assert_eq!(observed_moves(&v), 2 * (n / 2), "{v:?}");
    }
}

#[test]
fn moves_are_exactly_inversions_plus_carried_elements() {
    // Every element is compared in its slot before it is shifted or picked up, and is held
    // outside the slice only while it is carried, so the observed count is exact here: each
    // passed element moves one slot and removes one inversion, and each carried element adds
    // one move for its final write.
    for_each_test_input(0xFACE_FEED_0DD5_EED5, &mut |v| {
        assert_eq!(observed_moves(v), inversions(v) + carried(v), "{v:?}");
    });
}

#[test]
fn reverse_sorted_input_makes_n_n_plus_1_over_2_minus_1_moves() {
    // Every element but the first is carried to the front: `k + 1` moves for the element at
    // index `k`.
    for n in 1..size(300, 40) {
        let v: Vec<usize> = (0..n).rev().collect();
        assert_eq!(observed_moves(&v), n * (n + 1) / 2 - 1, "n = {n}");
    }
}

#[test]
fn turtle_at_the_end_makes_n_moves() {
    // `[1, 2, ..., n - 1, 0]`: the 0 passes the `n - 1` elements before it, plus its own
    // final write.
    for n in 2..size(300, 40) {
        let v: Vec<usize> = (1..n).chain([0]).collect();
        assert_eq!(observed_moves(&v), n, "n = {n}");
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v = vec![9, 8, 7, 3, 1, 2, 0, 6, 5, 4];
    gnome(&mut v[3..7]);
    assert_eq!(v, vec![9, 8, 7, 0, 1, 2, 3, 6, 5, 4]);
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
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

    let len = size(32, 12);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    // Reverse order makes every walk back reach the front, so panics land at every point of
    // the longest carries; repeated keys stop carries at equal elements; the turtle makes one
    // long carry at the very end.
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
        let total = comparisons(&keys);
        for budget in 0..=total {
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

            let result = catch_unwind(AssertUnwindSafe(|| gnome(&mut v)));

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
                drops.iter().all(|d| d.get() == 0),
                "element dropped during sort"
            );

            drop(v);
            assert!(
                drops.iter().all(|d| d.get() == 1),
                "element leaked or double-dropped"
            );

            assert_eq!(result.is_ok(), budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that answers at random makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // still terminate.
    struct Liar<'a> {
        name: String,
        rng: &'a Cell<u64>,
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
            let mut state = self.rng.get();
            let r = xorshift(&mut state);
            self.rng.set(state);
            [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize]
        }
    }

    let rng = Cell::new(0x5EED_5EED_5EED_5EED);
    for len in 0..size(64, 20) {
        for _ in 0..size(50, 3) {
            let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
            let mut v: Vec<Liar> = names
                .iter()
                .map(|name| Liar {
                    name: name.clone(),
                    rng: &rng,
                })
                .collect();

            gnome(&mut v);

            let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
            after.sort_unstable();
            let mut names = names;
            names.sort_unstable();
            assert_eq!(after, names);
        }
    }
}

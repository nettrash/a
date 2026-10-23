use a::sort::stooge;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;

/// Scales an exhaustive or randomised test down when it runs under Miri, where every
/// comparison is slow and stooge sort makes about `n^2.71` of them.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    stooge(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    stooge(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    stooge(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    stooge(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    stooge(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    stooge(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    stooge(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..30 {
        let mut v = vec![7; len];
        stooge(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    stooge(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn sorts_heap_owning_elements() {
    let words = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"];
    let mut v: Vec<String> = words.iter().map(|s| s.to_string()).collect();
    stooge(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);

    // Long enough to recurse above the unrolled parts.
    let n = size(40, 14);
    let mut state = 0xB0B0_CAFE_F00D_0001;
    let keys: Vec<u64> = (0..n).map(|_| xorshift(&mut state) % 50).collect();
    let mut v: Vec<Box<String>> = keys.iter().map(|k| Box::new(format!("{k:03}"))).collect();
    stooge(&mut v);
    let mut expected: Vec<Box<String>> = keys.iter().map(|k| Box::new(format!("{k:03}"))).collect();
    expected.sort();
    assert_eq!(v, expected);
}

#[test]
fn sorts_large_elements() {
    // 64-byte elements that differ only in their last word, so every comparison reads them
    // whole.
    let n = size(60, 14);
    let mut state = 0x1A2B_3C4D_5E6F_7081;
    let mut v: Vec<[u64; 8]> = (0..n)
        .map(|_| {
            let mut e = [7; 8];
            e[7] = xorshift(&mut state) % 10;
            e
        })
        .collect();
    let mut expected = v.clone();
    expected.sort();
    stooge(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..size(40, 14) {
        let mut v = vec![(); len];
        stooge(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    // Fifteen levels of recursion, eleven above the unrolled parts: about 2.15 · 10⁷
    // comparisons.
    let n = size(700, 30) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    stooge(&mut v);
    assert_eq!(v, (0..n).collect::<Vec<_>>());
}

#[test]
fn sorts_every_length_and_shape() {
    // Every length up to 100 covers every unrolled length (2 to 9) and every way of reaching
    // the unrolled parts from above (parts of 7, 8 and 9 elements).
    let mut state = 0x5EED_0000_0000_0001;
    for n in 0..size(100, 22) {
        let shapes: Vec<Vec<u64>> = vec![
            (0..n as u64).collect(),
            (0..n as u64).rev().collect(),
            (0..n).map(|_| xorshift(&mut state)).collect(),
            (0..n).map(|_| xorshift(&mut state) % 4).collect(),
            (0..n as u64).map(|i| i.min(n as u64 - 1 - i)).collect(),
        ];
        for v in shapes {
            let mut expected = v.clone();
            expected.sort();
            let mut actual = v.clone();
            stooge(&mut actual);
            assert_eq!(actual, expected, "input: {v:?}");
        }
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v = vec![9, 8, 7, 3, 1, 2, 0, 6, 5, 4];
    stooge(&mut v[3..7]);
    assert_eq!(v, vec![9, 8, 7, 0, 1, 2, 3, 6, 5, 4]);

    let mut v: Vec<u32> = (0..30).rev().collect();
    stooge(&mut v[5..25]);
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

/// Calls `f` with every permutation up to length 7, every sequence over three keys up to
/// length 8, and random sequences of distinct and of repeated keys of lengths that recurse
/// above the unrolled parts.
fn for_each_test_input(seed: u64, f: &mut impl FnMut(&[usize])) {
    for n in 0..=size(7, 5) {
        for_each_permutation(n, f);
    }
    for len in 0..=size(8, 5) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            let s: Vec<usize> = s.iter().map(|&k| k as usize).collect();
            f(&s);
        });
    }
    let mut state = seed;
    for n in [9, 10, 11, 12, 13, 16, size(33, 14), size(50, 20)] {
        for _ in 0..size(6, 1) {
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

/// Sorts `keys`, each tagged with its index, and returns the indices in their final order,
/// after checking that the keys come out sorted and that every element is there exactly
/// once, unchanged.
fn sorted_indices<K: Ord + Copy + std::fmt::Debug>(keys: &[K]) -> Vec<usize> {
    let mut v: Vec<ByKey<K>> = keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| ByKey { key, idx })
        .collect();
    stooge(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert!(
        v.iter().map(|e| e.key).eq(expected.iter().copied()),
        "input: {keys:?}"
    );
    let mut idx: Vec<usize> = v.iter().map(|e| e.idx).collect();
    let order = idx.clone();
    idx.sort_unstable();
    assert!(idx.into_iter().eq(0..keys.len()), "input: {keys:?}");
    assert!(
        v.iter().all(|e| e.key == keys[e.idx]),
        "element changed: {keys:?}"
    );
    order
}

#[test]
fn is_not_stable() {
    // The first compare-exchange of `[1a, 1b, 0]` swaps `1a` and `0`, which carries `1a` past
    // `1b`; the rest never compares two equal keys out of order again.
    assert_eq!(sorted_indices(&[1, 1, 0]), [2, 1, 0]);
    // Likewise with three equal keys: the first is carried past the other two.
    assert_eq!(sorted_indices(&[1, 1, 1, 0]), [3, 1, 2, 0]);
}

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| {
            sorted_indices(p);
        });
    }
}

#[test]
fn sorts_every_sequence_over_three_keys_up_to_9() {
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            sorted_indices(s);
        });
    }
}

#[test]
fn matches_std_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(64, 18) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(3, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                sorted_indices(&keys);
            }
        }
    }
}

/// How many times `m ↦ m − ⌊m/3⌋` must be applied to `n ≥ 2` to reach 2: the depth of the
/// recursion, every part at the bottom being two elements long.
fn depth(n: usize) -> u32 {
    let mut m = n;
    let mut d = 0;
    while m > 2 {
        m -= m / 3;
        d += 1;
    }
    d
}

/// The number of comparisons stooge sort makes on any input of `n` elements, in closed form:
/// `(3^(D + 1) − 1)/2` with `D` the depth of the recursion. Every part makes one comparison
/// and has three parts below it down to depth `D`, so there are `1 + 3 + … + 3^D` parts.
fn exact_comparisons(n: usize) -> usize {
    if n < 2 {
        return 0;
    }
    (3usize.pow(depth(n) + 1) - 1) / 2
}

#[test]
fn closed_form_matches_the_recurrence() {
    // `C(0) = C(1) = 0`, `C(2) = 1`, `C(n) = 1 + 3·C(n − ⌊n/3⌋)`.
    fn recurrence(n: usize) -> usize {
        match n {
            0 | 1 => 0,
            2 => 1,
            _ => 1 + 3 * recurrence(n - n / 3),
        }
    }
    for n in 0..10_000 {
        assert_eq!(exact_comparisons(n), recurrence(n), "n = {n}");
    }
    let table = [
        (2, 1),
        (3, 4),
        (4, 13),
        (5, 40),
        (8, 121),
        (10, 364),
        (16, 1093),
        (100, 265_720),
        (1000, 64_570_081),
    ];
    for (n, c) in table {
        assert_eq!(exact_comparisons(n), c, "n = {n}");
    }
    // Only the depth counts: 100 and 128 elements cost the same.
    assert_eq!(exact_comparisons(100), exact_comparisons(128));
}

/// An element that counts the comparisons made with it.
#[derive(Debug, Clone)]
struct Counted<'a, K> {
    key: K,
    count: &'a Cell<usize>,
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
        self.count.set(self.count.get() + 1);
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
            count: &count,
        })
        .collect();
    stooge(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert!(v.into_iter().map(|c| c.key).eq(expected), "{keys:?}");
    count.get()
}

#[test]
fn every_input_of_n_elements_costs_exactly_the_same_comparisons() {
    let mut state = 0xC0C7_A11E_5AAE_0001;
    for n in 0..size(100, 20) {
        let expected = exact_comparisons(n);
        let shapes: Vec<Vec<u64>> = vec![
            (0..n as u64).collect(),
            (0..n as u64).rev().collect(),
            vec![7; n],
            (0..n).map(|_| xorshift(&mut state)).collect(),
            (0..n).map(|_| xorshift(&mut state) % 3).collect(),
        ];
        for v in shapes {
            assert_eq!(comparisons(&v), expected, "{v:?}");
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

/// Sorts a copy of `keys` and returns the comparisons made, in order, and the indices the
/// elements started at, in their final order.
fn comparison_log<K: Ord + Clone>(keys: &[K]) -> (Vec<(usize, usize)>, Vec<usize>) {
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
    stooge(&mut v);
    let order = v.iter().map(|r| r.idx).collect();
    drop(v);
    (log.into_inner(), order)
}

/// Runs the textbook stooge sort (Cormen et al., Problem 7-3) on a copy of `keys` and returns
/// the comparisons it makes and the final order, in the same form as [`comparison_log`].
fn textbook_log<K: Ord + Clone>(keys: &[K]) -> (Vec<(usize, usize)>, Vec<usize>) {
    fn go<K: Ord>(v: &mut [(K, usize)], log: &mut Vec<(usize, usize)>) {
        let n = v.len();
        if n < 2 {
            return;
        }
        let (a, b) = (v[0].1, v[n - 1].1);
        log.push((a.min(b), a.max(b)));
        if v[0].0 > v[n - 1].0 {
            v.swap(0, n - 1);
        }
        if n >= 3 {
            let t = n / 3;
            go(&mut v[..n - t], log);
            go(&mut v[t..], log);
            go(&mut v[..n - t], log);
        }
    }
    let mut v: Vec<(K, usize)> = keys.iter().cloned().zip(0..).collect();
    let mut log = Vec::new();
    go(&mut v, &mut log);
    (log, v.into_iter().map(|e| e.1).collect())
}

#[test]
fn makes_the_textbook_comparisons_and_swaps_in_the_same_order() {
    // The unrolled parts and the iteration are only a different way of executing the same
    // algorithm: every comparison is the one the textbook recursion makes at that point, and
    // every swap leaves the elements where the textbook's swap leaves them.
    for_each_test_input(0x6E0B_E5EE_D5A1_7ED0, &mut |v| {
        assert_eq!(comparison_log(v), textbook_log(v), "{v:?}");
    });
}

/// An element that notices when the sort has moved it: every comparison checks its address
/// against the one it was last seen at. Rust moves are plain copies, so several moves between
/// two comparisons count as one (and a move there and back as none): the count is a lower
/// bound on the element moves the sort made.
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
    // Only a strictly smaller later element is swapped forward, so a non-decreasing slice is
    // compared C(n) times and never touched. Every non-decreasing sequence over three keys up
    // to length 12 (as cut points between runs), plus all-equal ones.
    let check = |keys: &[u8]| {
        let moves = Cell::new(0);
        let v: Vec<Watched<u8>> = keys
            .iter()
            .map(|&key| Watched {
                key,
                seen_at: Cell::new(0),
                moves: &moves,
            })
            .collect();
        let mut v = v;
        for w in &v {
            w.seen_at.set(w as *const Watched<u8> as usize);
        }
        stooge(&mut v);
        for w in &v {
            w.check();
        }
        assert_eq!(moves.get(), 0, "{keys:?}");
    };
    for len in 0..=size(12, 6) {
        for a in 0..=len {
            for b in a..=len {
                let v: Vec<u8> = (0..len).map(|i| (i >= a) as u8 + (i >= b) as u8).collect();
                check(&v);
            }
        }
    }
    for n in 0..size(40, 12) {
        check(&vec![7; n]);
    }
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    struct Tracked<'a> {
        key: u32,
        id: usize,
        name: String,
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

    let mut state = 0x0123_4567_89AB_CDEF;
    // Lengths sorted entirely by an unrolled part, and lengths that recurse above one.
    for len in [2, 3, 5, 8, 9, 10, size(16, 11)] {
        let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
        let layouts: Vec<Vec<u32>> = vec![
            (0..len as u32).rev().collect(),
            (0..len as u32).map(|id| id % 3).collect(),
            random,
            (1..len as u32).chain([0]).collect(),
        ];
        for keys in layouts {
            // Every budget below the full number of comparisons must panic, and only those: a
            // panic from anything else fails the test instead of trying ever larger budgets.
            let total = exact_comparisons(len);
            let step = size(1, 1 + total / 40);
            for budget in (0..=total).step_by(step).chain([total]) {
                let drops = Rc::new((0..len).map(|_| Cell::new(0)).collect::<Vec<_>>());
                let comparisons_left = Cell::new(budget);
                let mut v: Vec<Tracked> = keys
                    .iter()
                    .enumerate()
                    .map(|(id, &key)| Tracked {
                        key,
                        id,
                        name: format!("element {id}"),
                        drops: Rc::clone(&drops),
                        comparisons_left: &comparisons_left,
                    })
                    .collect();

                let result = catch_unwind(AssertUnwindSafe(|| stooge(&mut v)));

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
                    v.iter()
                        .all(|t| t.key == keys[t.id] && t.name == format!("element {}", t.id)),
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

                assert_eq!(result.is_ok(), budget == total, "budget {budget}: {keys:?}");
            }
        }
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that answers at random makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // still terminate: here after exactly the comparisons of a consistent order, because
    // which elements stooge sort compares never depends on the answers.
    struct Liar<'a> {
        name: String,
        rng: &'a Cell<u64>,
        count: &'a Cell<usize>,
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
            self.count.set(self.count.get() + 1);
            let mut state = self.rng.get();
            let r = xorshift(&mut state);
            self.rng.set(state);
            [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize]
        }
    }

    let rng = Cell::new(0x5EED_5EED_5EED_5EED);
    let count = Cell::new(0);
    for len in 0..size(64, 16) {
        for _ in 0..size(10, 2) {
            let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
            let mut v: Vec<Liar> = names
                .iter()
                .map(|name| Liar {
                    name: name.clone(),
                    rng: &rng,
                    count: &count,
                })
                .collect();

            count.set(0);
            stooge(&mut v);
            assert_eq!(count.get(), exact_comparisons(len), "len = {len}");

            let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
            after.sort_unstable();
            let mut names = names;
            names.sort_unstable();
            assert_eq!(after, names);
        }
    }
}

#[test]
fn inconsistent_ord_makes_the_textbook_swaps() {
    // Even with answers that contradict each other, the result is the one the textbook
    // recursion produces from the same answers.
    #[derive(Clone)]
    struct Liar<'a> {
        idx: usize,
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
        fn cmp(&self, other: &Self) -> Ordering {
            // Antisymmetric for each comparison, so `a > b` and `b < a` agree, but random
            // from one comparison to the next.
            let mut state = self.rng.get();
            let r = xorshift(&mut state);
            self.rng.set(state);
            let o = [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize];
            if self.idx < other.idx { o } else { o.reverse() }
        }
    }
    let rng = Cell::new(0xD1CE_D1CE_D1CE_D1CE);
    for len in 0..size(40, 14) {
        let seed = rng.get();
        let mut v: Vec<Liar> = (0..len).map(|idx| Liar { idx, rng: &rng }).collect();
        stooge(&mut v);
        let ours: Vec<usize> = v.iter().map(|l| l.idx).collect();
        rng.set(seed);
        let mut t: Vec<(Liar, usize)> =
            (0..len).map(|idx| (Liar { idx, rng: &rng }, idx)).collect();
        fn go(v: &mut [(Liar, usize)]) {
            let n = v.len();
            if n < 2 {
                return;
            }
            if v[0].0 > v[n - 1].0 {
                v.swap(0, n - 1);
            }
            if n >= 3 {
                let t = n / 3;
                go(&mut v[..n - t]);
                go(&mut v[t..]);
                go(&mut v[..n - t]);
            }
        }
        go(&mut t);
        let theirs: Vec<usize> = t.iter().map(|e| e.1).collect();
        assert_eq!(ours, theirs, "len = {len}");
    }
}

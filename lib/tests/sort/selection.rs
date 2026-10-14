use a::sort::selection;
use std::cell::Cell;
use std::cmp::Ordering;

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    selection(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    selection(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    selection(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    selection(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    selection(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    selection(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    selection(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..20 {
        let mut v = vec![7; len];
        selection(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    selection(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    selection(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
}

#[test]
fn sorts_zero_sized_elements() {
    let mut v = vec![(); 16];
    selection(&mut v);
    assert_eq!(v.len(), 16);
}

#[test]
fn sorts_long_reverse_sorted() {
    let mut v: Vec<u32> = (0..1000).rev().collect();
    selection(&mut v);
    assert_eq!(v, (0..1000).collect::<Vec<_>>());
}

#[test]
fn sorts_long_input_with_few_keys() {
    let mut v: Vec<u32> = (0..1000).map(|i| (i * 7919) % 5).collect();
    selection(&mut v);
    let mut expected: Vec<u32> = (0..1000).map(|i| (i * 7919) % 5).collect();
    expected.sort_unstable();
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

fn assert_sorts_like_std<T: Ord + Clone + std::fmt::Debug>(input: &[T]) {
    let mut expected = input.to_vec();
    expected.sort();
    let mut actual = input.to_vec();
    selection(&mut actual);
    assert_eq!(actual, expected, "input: {input:?}");
}

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=8 {
        for_each_permutation(n, &mut |p| assert_sorts_like_std(p));
    }
}

#[test]
fn sorts_every_sequence_over_three_keys_up_to_9() {
    // Many equal keys exercise every way the min and max groups can form, be abandoned
    // and collide, including a minimum equal to the maximum.
    for len in 0..=9 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_like_std(s));
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

#[test]
fn matches_std_unstable_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..100 {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [3, 8, 64, u64::MAX] {
            for _ in 0..8 {
                let input: Vec<ByKey<u64>> = (0..len)
                    .map(|idx| ByKey {
                        key: xorshift(&mut state) % modulus,
                        idx,
                    })
                    .collect();

                let mut expected: Vec<u64> = input.iter().map(|e| e.key).collect();
                expected.sort_unstable();

                let mut actual = input.clone();
                selection(&mut actual);

                // Selection sort is not stable, so only the order of keys is fixed...
                let keys: Vec<u64> = actual.iter().map(|e| e.key).collect();
                assert_eq!(keys, expected);
                // ...but every tagged element must still be there exactly once.
                let mut before: Vec<(u64, usize)> = input.iter().map(|e| (e.key, e.idx)).collect();
                let mut after: Vec<(u64, usize)> = actual.iter().map(|e| (e.key, e.idx)).collect();
                before.sort_unstable();
                after.sort_unstable();
                assert_eq!(after, before);
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
    selection(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

/// Comparisons of one pairwise min-max scan over `m` elements: `⌈3m/2⌉ - 2`.
fn scan_cost(m: usize) -> usize {
    if m < 2 { 0 } else { (3 * m).div_ceil(2) - 2 }
}

/// The documented worst case: `n(3n - 2)/8` for even `n`, `3(n² - 1)/8` for odd `n`.
fn worst_case(n: usize) -> usize {
    if n % 2 == 1 {
        3 * (n * n - 1) / 8
    } else {
        n * (3 * n).saturating_sub(2) / 8
    }
}

#[test]
fn worst_case_formula_sums_the_scans_of_every_pass() {
    for n in 0..500 {
        let passes: usize = (0..=n).rev().step_by(2).map(scan_cost).sum();
        assert_eq!(worst_case(n), passes, "n = {n}");
    }
}

#[test]
fn distinct_keys_cost_exactly_the_worst_case() {
    // Every order of distinct keys costs the same: the pairwise scan never short-circuits.
    for n in 0..=8 {
        for_each_permutation(n, &mut |p| {
            assert_eq!(comparisons(p), worst_case(n), "{p:?}")
        });
    }
    let mut state = 0x2545_F491_4F6C_DD1D;
    for n in [9, 10, 31, 64, 100, 255] {
        let ascending: Vec<usize> = (0..n).collect();
        let descending: Vec<usize> = (0..n).rev().collect();
        let organ_pipe: Vec<usize> = (0..n)
            .map(|i| {
                if i < n / 2 {
                    2 * i
                } else {
                    2 * (n - 1 - i) + 1
                }
            })
            .collect();
        let mut shuffled = ascending.clone();
        shuffle(&mut shuffled, &mut state);
        for input in [ascending, descending, organ_pipe, shuffled] {
            assert_eq!(comparisons(&input), worst_case(n), "n = {n}");
        }
    }
    // 25% fewer than the n(n - 1)/2 of classic selection sort.
    assert_eq!(worst_case(1000), 374_750);
    assert_eq!(1000 * 999 / 2, 499_500);
}

#[test]
fn equal_keys_cost_n_minus_1_comparisons() {
    for n in 1..100 {
        assert_eq!(comparisons(&vec![7; n]), n - 1, "n = {n}");
    }
    assert_eq!(comparisons::<u8>(&[]), 0);
}

#[test]
fn equal_pair_skips_the_comparison_it_can_rule_out() {
    // An equal pair costs one comparison between its two elements and one with the minimum,
    // unless it lies above the minimum of an unequal range: only then is the maximum consulted.
    assert_eq!(comparisons(&[1, 2, 2]), 2); // above an all-equal range: a new maximum
    assert_eq!(comparisons(&[3, 2, 2]), 2); // below the minimum
    assert_eq!(comparisons(&[1, 3, 1, 1]), 3); // equal to the minimum
    assert_eq!(comparisons(&[1, 3, 2, 2]), 1 + 3 + 1); // in between, then a pass over [2, 2]
    assert_eq!(worst_case(3), 3);
    assert_eq!(worst_case(4), 5);
}

#[test]
fn duplicates_never_cost_more_than_the_worst_case() {
    for len in 0..=9 {
        for_each_sequence(len, 3, &mut |s| {
            assert!(comparisons(s) <= worst_case(s.len()), "{s:?}");
        });
    }
    let mut state = 0x1234_5678_9ABC_DEF1;
    for len in 0..200 {
        for keys in [2, 3, 10, 50, 1000] {
            let v: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % keys).collect();
            assert!(comparisons(&v) <= worst_case(len));
        }
    }
}

#[test]
fn k_distinct_keys_finish_in_half_k_passes() {
    // With `k` keys of `r` copies each, every pass retires all copies of its two extremes,
    // so the windows are `n, n - 2r, n - 4r, ...`; an odd `k` ends with a pass over the `r`
    // copies of the middle key, which proves them equal in `r - 1` comparisons and stops.
    let mut state = 0xDEAD_BEEF_CAFE_F00D;
    for k in 1..=12 {
        for r in 1..=8 {
            let n = k * r;
            let bound: usize = (0..k / 2).map(|p| scan_cost(n - 2 * p * r)).sum::<usize>()
                + if k % 2 == 1 { r - 1 } else { 0 };
            for _ in 0..10 {
                let mut v: Vec<usize> = (0..n).map(|i| i % k).collect();
                shuffle(&mut v, &mut state);
                assert!(comparisons(&v) <= bound, "k = {k}, r = {r}: {v:?}");
            }
        }
    }
    // Two keys over 1000 elements: one pass, versus 500 for all-distinct input.
    let v: Vec<u8> = (0..1000).map(|i| (i % 2) as u8).collect();
    assert!(comparisons(&v) <= scan_cost(1000));
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

/// Sorts `keys` and returns the number of element moves observed, including the final one
/// of every element that ends up away from where it was last seen.
fn observed_moves<K: Ord + Copy + std::fmt::Debug>(keys: &[K]) -> usize {
    let moves = Cell::new(0);
    let mut v: Vec<Watched<K>> = keys
        .iter()
        .map(|&key| Watched {
            key,
            seen_at: Cell::new(0),
            moves: &moves,
        })
        .collect();
    for w in &v {
        w.seen_at.set(w as *const Watched<K> as usize);
    }
    selection(&mut v);
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
    for len in 0..=12usize {
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
    check(&(0..1000).map(|i| (i / 125) as u8).collect::<Vec<u8>>());
    for n in 0..50 {
        check(&vec![7; n]);
    }
}

#[test]
fn distinct_keys_cost_at_most_n_minus_1_swaps() {
    // A swap moves two elements, so `n - 1` swaps allow at most `2(n - 1)` observed moves.
    let check = |v: &[usize]| {
        let moves = observed_moves(v);
        assert!(moves <= 2 * v.len().saturating_sub(1), "{v:?}: {moves}");
    };
    for n in 0..=8 {
        for_each_permutation(n, &mut |p| check(p));
    }
    let mut state = 0x2545_F491_4F6C_DD1D;
    for n in [9, 16, 33, 100, 257] {
        for _ in 0..20 {
            let mut v: Vec<usize> = (0..n).collect();
            shuffle(&mut v, &mut state);
            check(&v);
        }
    }
    // Reverse order needs a swap in every pass, so the observed count is far from zero.
    assert!(observed_moves(&(0..100).rev().collect::<Vec<u32>>()) >= 100);
}

#[test]
fn duplicates_cost_at_most_2m_swaps_per_pass() {
    // With `k` keys there are at most `⌈k/2⌉` passes over at most `n` elements each, so at
    // most `2n` swaps, or `4n` observed moves, per pass.
    let check = |v: &[u8], keys: usize| {
        let moves = observed_moves(v);
        assert!(moves <= 4 * v.len() * keys.div_ceil(2), "{v:?}: {moves}");
    };
    for len in 0..=9 {
        for_each_sequence(len, 3, &mut |v| check(v, 3));
    }
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..200 {
        for keys in [2, 5, 16] {
            let v: Vec<u8> = (0..len)
                .map(|_| (xorshift(&mut state) % keys as u64) as u8)
                .collect();
            check(&v, keys);
        }
    }
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

    const LEN: usize = 32;
    // Reverse order keeps the running minimum moving; repeated keys make groups form, get
    // abandoned and collide, so the panic lands in the middle of every kind of swap.
    let layouts: [fn(usize) -> u32; 3] = [
        |id| (LEN - id) as u32,
        |id| (id % 5) as u32,
        |id| ((LEN - id) / 3) as u32,
    ];
    for key_of in layouts {
        for budget in 0..=worst_case(LEN) + 1 {
            let drops = Rc::new((0..LEN).map(|_| Cell::new(0)).collect::<Vec<_>>());
            let comparisons_left = Cell::new(budget);
            let mut v: Vec<Tracked> = (0..LEN)
                .map(|id| Tracked {
                    key: key_of(id),
                    id,
                    drops: Rc::clone(&drops),
                    comparisons_left: &comparisons_left,
                })
                .collect();

            let result = catch_unwind(AssertUnwindSafe(|| selection(&mut v)));

            if result.is_ok() {
                assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
            }
            let mut ids: Vec<usize> = v.iter().map(|t| t.id).collect();
            ids.sort_unstable();
            assert_eq!(
                ids,
                (0..LEN).collect::<Vec<_>>(),
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
        }
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that answers at random breaks every assumption the pass makes about its
    // groups. The order of the result is then unspecified, but no element may be lost or
    // duplicated, no index may go out of bounds, and the sort must still terminate.
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
    for len in 0..64 {
        for _ in 0..50 {
            let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
            let mut v: Vec<Liar> = names
                .iter()
                .map(|name| Liar {
                    name: name.clone(),
                    rng: &rng,
                })
                .collect();

            selection(&mut v);

            let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
            after.sort_unstable();
            let mut names = names;
            names.sort_unstable();
            assert_eq!(after, names);
        }
    }
}

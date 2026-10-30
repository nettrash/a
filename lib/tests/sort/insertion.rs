use a::sort::insertion;
use std::cell::Cell;
use std::cmp::Ordering;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    insertion(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    insertion(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    insertion(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    insertion(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    insertion(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    insertion(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    insertion(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..20 {
        let mut v = vec![7; len];
        insertion(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    insertion(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    insertion(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..17 {
        let mut v = vec![(); len];
        insertion(&mut v);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    let n = size(1000, 100) as u32;
    let mut v: Vec<u32> = (0..n).rev().collect();
    insertion(&mut v);
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
    insertion(&mut actual);
    let actual: Vec<(K, usize)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert_eq!(actual, expected, "input: {keys:?}");
}

#[test]
fn is_stable_for_equal_keys() {
    // Equal keys inside a pair, between a pair and the prefix, and across pairs.
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
    // Few keys put equal elements inside pairs, between a pair and the prefix and at every
    // insertion point, so the index tags check stability exhaustively.
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

/// Inputs that are sorted apart from a little disorder, which an adaptive sort should handle
/// cheaply: a few random swaps, a random tail, ascending runs, swapped neighbours, and small
/// displacements.
fn nearly_sorted_inputs(n: usize, keys: u64, state: &mut u64) -> Vec<Vec<u64>> {
    let sorted: Vec<u64> = (0..n as u128)
        .map(|i| (i * keys as u128 / n.max(1) as u128) as u64)
        .collect();
    let mut shapes = Vec::new();

    for swaps in [1, 2, 5] {
        let mut v = sorted.clone();
        for _ in 0..swaps {
            if n > 1 {
                let i = (xorshift(state) % n as u64) as usize;
                let j = (xorshift(state) % n as u64) as usize;
                v.swap(i, j);
            }
        }
        shapes.push(v);
    }

    for tail in [1, 3, n / 8] {
        let mut v = sorted.clone();
        let tail = tail.min(n);
        for x in &mut v[n - tail..] {
            *x = xorshift(state) % keys.max(1);
        }
        shapes.push(v);
    }

    for runs in [2, 3, 8] {
        let mut v: Vec<u64> = (0..n as u64)
            .map(|_| xorshift(state) % keys.max(1))
            .collect();
        let len = n.div_ceil(runs).max(1);
        for chunk in v.chunks_mut(len) {
            chunk.sort();
        }
        shapes.push(v);
    }

    let mut neighbours = sorted.clone();
    for pair in neighbours.chunks_mut(2) {
        pair.reverse();
    }
    shapes.push(neighbours);

    let mut displaced = sorted.clone();
    for i in 0..n {
        let j = (i + (xorshift(state) % 4) as usize).min(n - 1);
        displaced.swap(i, j);
    }
    shapes.push(displaced);

    shapes
}

#[test]
fn sorts_nearly_sorted_input_stably_and_cheaply() {
    let mut state = 0xC0FF_EE00_D15E_A5E5;
    for n in [0, 1, 2, 3, 10, 33, 64, 100, size(500, 40)] {
        for keys in [3, 50, u64::MAX] {
            for v in nearly_sorted_inputs(n, keys, &mut state) {
                assert_sorts_like_std_stable(&v);
                assert!(comparisons(&v) <= adaptive_bound(&v), "{v:?}");
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
    insertion(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

/// For every element, the number of elements before it that are greater: how far it moves.
fn distances<K: Ord>(keys: &[K]) -> Vec<usize> {
    (0..keys.len())
        .map(|j| keys[..j].iter().filter(|&e| *e > keys[j]).count())
        .collect()
}

/// `2⌈log₂(d + 1)⌉`, which is twice the number of bits in `d`.
fn twice_ceil_log2_succ(d: usize) -> usize {
    2 * (usize::BITS - d.leading_zeros()) as usize
}

/// The documented adaptive bound: `n - 1 + Σₓ 2⌈log₂(dₓ + 1)⌉`.
fn adaptive_bound<K: Ord>(keys: &[K]) -> usize {
    keys.len().saturating_sub(1)
        + distances(keys)
            .into_iter()
            .map(twice_ceil_log2_succ)
            .sum::<usize>()
}

/// `2n·log₂ n`, the documented upper bound on the worst case.
fn n_log_n_bound(n: usize) -> f64 {
    if n < 2 {
        0.0
    } else {
        2.0 * n as f64 * (n as f64).log2()
    }
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
        // Swapped neighbours within each pair only take the fast path's swap.
        let mut v: Vec<usize> = (0..n).collect();
        for pair in v[1.min(n)..].chunks_mut(2) {
            pair.reverse();
        }
        assert_eq!(comparisons(&v), expected, "{v:?}");
    }
    // Every non-decreasing sequence over three keys, as cut points between runs.
    for len in 0..=size(16, 6) {
        for a in 0..=len {
            for b in a..=len {
                let v: Vec<u8> = (0..len).map(|i| (i >= a) as u8 + (i >= b) as u8).collect();
                assert_eq!(comparisons(&v), len.saturating_sub(1), "{v:?}");
            }
        }
    }
}

#[test]
fn adaptive_bound_holds_for_every_input() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| {
            assert!(comparisons(p) <= adaptive_bound(p), "{p:?}")
        });
    }
    for len in 0..=size(9, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            assert!(comparisons(s) <= adaptive_bound(s), "{s:?}")
        });
    }
    let mut state = 0x2545_F491_4F6C_DD1D;
    for len in 0..size(200, 30) {
        for keys in [2, 5, 100, u64::MAX] {
            let v: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % keys).collect();
            let (cost, bound) = (comparisons(&v), adaptive_bound(&v));
            assert!(cost <= bound, "{v:?}");
            // ...and so is the closed form that follows from it by concavity.
            let inversions: usize = distances(&v).into_iter().sum();
            let n = len as f64;
            let closed = 3.0 * n + 2.0 * n * (1.0 + inversions as f64 / n.max(1.0)).log2();
            assert!((bound as f64) <= closed.max(0.0), "{v:?}");
        }
    }
}

#[test]
fn adaptive_bound_is_tight() {
    // `[1, 0, 3, 2, 5, 4, ...]`: the larger element of every pair stays put and the smaller
    // one moves one slot, which costs exactly its share of the bound. Only the first pair,
    // whose search is cut short at the front, and a last lone element come in below it.
    for n in 3..size(300, 40) {
        let v: Vec<usize> = (0..n).map(|i| i ^ 1).collect();
        let slack = adaptive_bound(&v) - comparisons(&v);
        // One for the first pair; one more for the last lone element when `n` is even.
        let expected = 1 + usize::from(n.is_multiple_of(2));
        assert_eq!(
            slack, expected,
            "n = {n}: {v:?} costs {slack} below the bound"
        );
    }
}

/// The documented cost of reverse-sorted input: `⌈log₂ i⌉ + 3` for the pair inserted into a
/// prefix of `i` elements, and `⌈log₂(n - 1)⌉ + 1` for a last lone element.
fn reverse_sorted_cost(n: usize) -> usize {
    let ceil_log2 = |i: usize| (usize::BITS - (i - 1).leading_zeros()) as usize;
    let pairs: usize = (1..n.saturating_sub(1))
        .step_by(2)
        .map(|i| ceil_log2(i) + 3)
        .sum();
    pairs
        + if n >= 2 && n.is_multiple_of(2) {
            ceil_log2(n - 1) + 1
        } else {
            0
        }
}

#[test]
fn reverse_sorted_input_costs_exactly_the_documented_comparisons() {
    for n in 0..size(1001, 40) {
        let v: Vec<usize> = (0..n).rev().collect();
        assert_eq!(comparisons(&v), reverse_sorted_cost(n), "n = {n}");
        // ≈ (n/2)·log₂ n + n
        if n >= 2 {
            let approx = n as f64 / 2.0 * (n as f64).log2() + n as f64;
            let diff = (reverse_sorted_cost(n) as f64 - approx).abs();
            assert!(diff <= n as f64 / 8.0 + 4.0, "n = {n}: {diff}");
        }
    }
    assert_eq!(reverse_sorted_cost(1000), 5986);
}

/// The input of length `d.len()` in which the element at `j` has exactly `d[j]` greater
/// elements before it (`d[j] <= j`): every such table belongs to exactly one permutation.
fn from_distances(d: &[usize]) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::new(); // indices, in ascending order of their keys
    for (j, &dj) in d.iter().enumerate() {
        order.insert(j - dj, j);
    }
    let mut keys = vec![0; d.len()];
    for (rank, &j) in order.iter().enumerate() {
        keys[j] = rank;
    }
    keys
}

/// Keys for a sorted prefix of `i` elements followed by `a`, with `da` greater elements
/// before it, and, if `db` is given, by `b`, with `db` greater elements before it.
fn prefix_then(i: usize, da: usize, db: Option<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = (0..i).map(|j| 8 * j + 8).collect();
    v.push(8 * (i - da) + 4);
    if let Some(db) = db {
        let mut seen = v.clone();
        seen.sort_unstable();
        v.push(if db == 0 {
            seen[i] + 2
        } else {
            seen[i + 1 - db] - 2
        });
    }
    v
}

/// The most comparisons any input of length `n` with distinct keys costs, found black box.
///
/// What a pair costs depends only on where its two elements land in the prefix, and every
/// combination of landing places is possible after any prefix. So the worst case is the sum,
/// over the pairs and a last lone element, of the most each can cost: the cost of sorting an
/// already-sorted prefix (`i - 1` comparisons) followed by that pair, minus the prefix's share.
/// Returns that sum and an input that should attain it.
fn worst_case(n: usize) -> (usize, Vec<usize>) {
    let mut table = vec![0; n];
    let mut total = 0;
    let mut i = 1;
    while i < n {
        let pair = i + 1 < n;
        let mut best = (0, 0, 0);
        for da in 0..=i {
            for db in 0..=if pair { i + 1 } else { 0 } {
                let keys = prefix_then(i, da, pair.then_some(db));
                let cost = comparisons(&keys) - (i - 1);
                if cost > best.0 {
                    best = (cost, da, db);
                }
            }
        }
        total += best.0;
        table[i] = best.1;
        if pair {
            table[i + 1] = best.2;
        }
        i += if pair { 2 } else { 1 };
    }
    (total, from_distances(&table))
}

#[test]
fn worst_case_is_between_2n_log_n_minus_3_5n_and_2n_log_n() {
    // Exhaustively for small n: the black-box decomposition finds the true maximum.
    for n in 0..=size(8, 4) {
        let mut max = 0;
        for_each_permutation(n, &mut |p| max = max.max(comparisons(p)));
        let (worst, input) = worst_case(n);
        assert_eq!(max, worst, "n = {n}");
        assert_eq!(comparisons(&input), worst, "n = {n}");
    }
    for n in (0..=size(16, 10)).chain([31, 32, 33, 63, 64, 100].map(|n| size(n, 12))) {
        let (worst, input) = worst_case(n);
        assert_eq!(comparisons(&input), worst, "n = {n}");
        let bound = n_log_n_bound(n);
        assert!(worst as f64 <= bound, "n = {n}: {worst} > {bound}");
        // Tight: the worst case is `≈ 2n·log₂ n - 3n`.
        assert!(worst as f64 >= bound - 3.5 * n as f64, "n = {n}: {worst}");
    }
    // Random inputs stay below it too.
    let mut state = 0x1234_5678_9ABC_DEF1;
    for n in [10, 100, size(1000, 50)] {
        for _ in 0..size(20, 2) {
            let mut v: Vec<usize> = (0..n).collect();
            shuffle(&mut v, &mut state);
            assert!(comparisons(&v) as f64 <= n_log_n_bound(n));
        }
    }
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
    insertion(&mut v);
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
    check(
        &(0..size(1000, 50))
            .map(|i| (i / 125) as u8)
            .collect::<Vec<u8>>(),
    );
    for n in 0..50 {
        check(&vec![7; n]);
    }
}

#[test]
fn swapped_pairs_move_only_their_own_two_elements() {
    // `[0, 2, 1, 4, 3, ...]`: each pair takes the fast path and swaps, which moves its two
    // elements once each and nothing else.
    for n in 0..size(200, 30) {
        let mut v: Vec<usize> = (0..n).collect();
        for pair in v[1.min(n)..].chunks_mut(2) {
            pair.reverse();
        }
        let swapped = n.saturating_sub(1) / 2;
        assert_eq!(observed_moves(&v), 2 * swapped, "{v:?}");
    }
}

#[test]
fn moves_are_at_most_inversions_plus_n() {
    // The observed count is only a lower bound on the moves made, so this can catch
    // gratuitous moves but cannot prove the bound.
    let check = |v: &[usize]| {
        let inversions: usize = distances(v).into_iter().sum();
        assert!(observed_moves(v) <= inversions + v.len(), "{v:?}");
    };
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| check(p));
    }
    let mut state = 0xDEAD_BEEF_CAFE_F00D;
    for n in [9, 16, 33, 100, size(257, 40)] {
        for _ in 0..size(20, 2) {
            let mut v: Vec<usize> = (0..n).collect();
            shuffle(&mut v, &mut state);
            check(&v);
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

    let len = size(32, 12);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    // Reverse order sends every pair far; repeated keys put equal elements at the insertion
    // points; the random and nearly-sorted shapes mix the fast path with both kinds of move.
    let layouts: Vec<Vec<u32>> = vec![
        (0..len as u32).rev().collect(),
        (0..len as u32).map(|id| id % 3).collect(),
        (0..len as u32).map(|id| (len as u32 - id) / 3).collect(),
        random,
        (0..len as u32).map(|id| id ^ 1).collect(),
        (0..len as u32)
            .map(|id| if id % 5 == 4 { 0 } else { id })
            .collect(),
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

            let result = catch_unwind(AssertUnwindSafe(|| insertion(&mut v)));

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
    // An `Ord` that answers at random breaks the sorted prefix that the searches rely on.
    // The order of the result is then unspecified, but no element may be lost or duplicated,
    // no index may go out of bounds, and the sort must still terminate.
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

            insertion(&mut v);

            let mut after: Vec<String> = v.into_iter().map(|l| l.name).collect();
            after.sort_unstable();
            let mut names = names;
            names.sort_unstable();
            assert_eq!(after, names);
        }
    }
}

#[test]
fn sorts_elements_of_every_size_whether_moved_inline_or_not() {
    // Short moves of elements up to 32 bytes are made inline, longer moves and larger elements
    // by `memmove`: lengths around 32 moves and sizes of 4, 8, 32 and 40 bytes.
    fn check<T: Ord + Clone + std::fmt::Debug>(v: Vec<T>) {
        let mut expected = v.clone();
        expected.sort();
        let mut actual = v;
        insertion(&mut actual);
        assert_eq!(actual, expected);
    }
    let mut state = 0x51F7_0000_0000_0001;
    for len in (0..size(80, 40)).chain([size(300, 70)]) {
        for modulus in [3, u64::MAX] {
            let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
            check(keys.iter().map(|&k| k as u32).collect::<Vec<_>>());
            check(keys.clone());
            check(keys.iter().map(|&k| [k, 1, 2, 3]).collect::<Vec<_>>());
            check(keys.iter().map(|&k| [k, 1, 2, 3, 4]).collect::<Vec<_>>());
            let mut descending = keys;
            descending.sort_unstable_by(|a, b| b.cmp(a));
            check(descending.iter().map(|&k| [k, 1, 2, 3]).collect::<Vec<_>>());
        }
    }
}

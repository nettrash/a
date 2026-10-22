use a::sort::{odd_even, odd_even_parallel};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as Atomic};

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Odd-even sort compare-exchanges elements of at most 8 bytes without branching and larger
/// ones with a branch. The test elements below carry `PAD` bytes of payload: `SMALL` keeps
/// them at 8 bytes and `LARGE` makes them larger, and every test that depends on how elements
/// are moved runs with both, or with [`Tiny`] elements of 4 bytes, which keep their bookkeeping
/// in thread-local state to stay that small.
const SMALL: usize = 0;
const LARGE: usize = 8;

/// The slice length from which `odd_even_parallel` may use threads. Miri, which interprets
/// every step thousands of times slower, sees a lower one, so that it can check the threads
/// at all; run it with `-Zmiri-num-cpus=4` (in `MIRIFLAGS`) to let them start.
const PARALLEL_MIN_LEN: usize = if cfg!(miri) { 64 } else { 4096 };

#[test]
fn test_elements_cover_both_kernels() {
    assert!(size_of::<Tiny>() <= 8);
    assert_eq!(size_of::<ByKey<u32, SMALL>>(), 8);
    assert!(size_of::<ByKey<u32, LARGE>>() > 8);
    assert!(size_of::<Counted<usize>>() > 8);
    assert!(size_of::<Watched<usize>>() > 8);
}

/// Sorts `v` with the sequential or the parallel entry point.
fn sort<T: Ord + Send>(v: &mut [T], parallel: bool) {
    if parallel {
        odd_even_parallel(v);
    } else {
        odd_even(v);
    }
}

#[test]
fn sorts_empty_slice() {
    for parallel in [false, true] {
        let mut v: Vec<i32> = vec![];
        sort(&mut v, parallel);
        assert_eq!(v, vec![]);
    }
}

#[test]
fn sorts_single_element() {
    for parallel in [false, true] {
        let mut v = vec![42];
        sort(&mut v, parallel);
        assert_eq!(v, vec![42]);
    }
}

#[test]
fn sorts_two_elements() {
    for parallel in [false, true] {
        let mut v = vec![2, 1];
        sort(&mut v, parallel);
        assert_eq!(v, vec![1, 2]);

        let mut v = vec![1, 2];
        sort(&mut v, parallel);
        assert_eq!(v, vec![1, 2]);
    }
}

#[test]
fn sorts_already_sorted() {
    for parallel in [false, true] {
        let mut v = vec![1, 2, 3, 4, 5];
        sort(&mut v, parallel);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }
}

#[test]
fn sorts_reverse_sorted() {
    for parallel in [false, true] {
        let mut v = vec![5, 4, 3, 2, 1];
        sort(&mut v, parallel);
        assert_eq!(v, vec![1, 2, 3, 4, 5]);
    }
}

#[test]
fn sorts_with_duplicates() {
    for parallel in [false, true] {
        let mut v = vec![3, 1, 2, 3, 2, 1, 0];
        sort(&mut v, parallel);
        assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
    }
}

#[test]
fn sorts_all_equal() {
    for parallel in [false, true] {
        for len in 0..20 {
            let mut v = vec![7; len];
            sort(&mut v, parallel);
            assert_eq!(v, vec![7; len]);
        }
    }
}

#[test]
fn sorts_strings() {
    for parallel in [false, true] {
        let mut v = vec!["pear", "apple", "banana", "apple"];
        sort(&mut v, parallel);
        assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
    }
}

#[test]
fn sorts_heap_owning_elements() {
    for parallel in [false, true] {
        let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        sort(&mut v, parallel);
        assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
    }
}

#[test]
fn sorts_large_elements() {
    for parallel in [false, true] {
        let mut v: Vec<[u64; 8]> = (0..50u64)
            .rev()
            .map(|k| [k % 7, k, 0, 0, 0, 0, 0, k])
            .collect();
        let mut expected = v.clone();
        expected.sort();
        sort(&mut v, parallel);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_zero_sized_elements() {
    for parallel in [false, true] {
        for len in 0..17 {
            let mut v = vec![(); len];
            sort(&mut v, parallel);
            assert_eq!(v.len(), len);
        }
        // Long enough for the parallel entry point to consider threads.
        let mut v = vec![(); size(10_000, 5000)];
        sort(&mut v, parallel);
    }
}

#[test]
fn sorts_long_reverse_sorted() {
    for parallel in [false, true] {
        let n = size(1000, 100) as u32;
        let mut v: Vec<u32> = (0..n).rev().collect();
        sort(&mut v, parallel);
        assert_eq!(v, (0..n).collect::<Vec<_>>());
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    for parallel in [false, true] {
        let mut v = vec![9, 8, 7, 3, 1, 2, 0, 6, 5, 4];
        sort(&mut v[3..7], parallel);
        assert_eq!(v, vec![9, 8, 7, 0, 1, 2, 3, 6, 5, 4]);
    }
}

#[test]
fn never_allocates_on_one_thread() {
    let mut v: Vec<u64> = (0..2000).rev().collect();
    assert_eq!(crate::allocations::count(|| odd_even(&mut v)), 0);
    let mut v: Vec<String> = (0..300).rev().map(|i| i.to_string()).collect();
    assert_eq!(crate::allocations::count(|| odd_even(&mut v)), 0);
    // Below the parallel threshold, and on sorted input of any length, the parallel entry
    // point does not start threads either.
    let mut v: Vec<u64> = (0..PARALLEL_MIN_LEN as u64 - 1).rev().collect();
    assert_eq!(crate::allocations::count(|| odd_even_parallel(&mut v)), 0);
    let mut v: Vec<u64> = (0..size(100_000, 10_000) as u64).collect();
    assert_eq!(crate::allocations::count(|| odd_even_parallel(&mut v)), 0);
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
    for n in 0..=size(8, 4) {
        for_each_permutation(n, f);
    }
    for len in 0..=size(9, 5) as u32 {
        for_each_sequence(len, 3, &mut |s| {
            let s: Vec<usize> = s.iter().map(|&k| k as usize).collect();
            f(&s);
        });
    }
    let mut state = seed;
    let lens: &[usize] = if cfg!(miri) {
        &[9, 33]
    } else {
        &[9, 16, 33, 100, 257]
    };
    for &n in lens {
        for _ in 0..size(20, 1) {
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
    idx: u32,
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

/// Sorts `keys`, each tagged with its index, and checks the result against std's stable sort:
/// the same keys in the same order, and equal keys still in their original order.
fn assert_sorts_like_std_stable<K: Ord + Copy + Send + std::fmt::Debug, const PAD: usize>(
    keys: &[K],
    parallel: bool,
) {
    let mut expected: Vec<(K, u32)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|e| e.0);
    let mut actual: Vec<ByKey<K, PAD>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, idx)| ByKey {
            key,
            idx,
            _pad: [0; PAD],
        })
        .collect();
    sort(&mut actual, parallel);
    let actual: Vec<(K, u32)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
    assert!(actual == expected, "input: {keys:?}");
}

/// Runs [`assert_sorts_like_std_stable`] with both kernels and both entry points.
fn assert_sorts_stably<K: Ord + Copy + Send + std::fmt::Debug>(keys: &[K]) {
    for parallel in [false, true] {
        assert_sorts_like_std_stable::<K, SMALL>(keys, parallel);
        assert_sorts_like_std_stable::<K, LARGE>(keys, parallel);
    }
}

#[test]
fn is_stable_for_equal_keys() {
    assert_sorts_stably(&[2u32, 1, 2, 1, 2]);
    assert_sorts_stably(&[1u32, 1, 1, 0, 0]);
    assert_sorts_stably(&[0u32, 1, 1, 1, 1, 0, 0]);
    assert_sorts_stably(&[5u32, 5, 3, 3, 5, 3, 5, 3]);
}

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| {
            let p: Vec<u32> = p.iter().map(|&k| k as u32).collect();
            assert_sorts_stably(&p);
        });
    }
}

#[test]
fn sorts_every_sequence_over_three_keys_up_to_9_stably() {
    for len in 0..=size(9, 5) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts_stably(s));
    }
}

#[test]
fn sorts_every_zero_one_sequence_up_to_16() {
    // By the 0-1 principle, a network of compare-exchanges that sorts every sequence of zeros
    // and ones sorts everything: this checks that `⌈n/2⌉` rounds are always enough.
    for len in 0..=size(16, 8) as u32 {
        for code in 0..1u32 << len {
            let v: Vec<u32> = (0..len).map(|i| (code >> i) & 1).collect();
            assert_sorts_like_std_stable::<u32, SMALL>(&v, false);
            assert_sorts_like_std_stable::<u32, LARGE>(&v, false);
        }
    }
}

#[test]
fn matches_std_stable_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(100, 16) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 8, 64, u64::MAX] {
            for _ in 0..size(8, 1) {
                let keys: Vec<u64> = (0..len).map(|_| xorshift(&mut state) % modulus).collect();
                assert_sorts_stably(&keys);
            }
        }
    }
}

/// What a run of odd-even sort did.
#[derive(Debug, PartialEq)]
struct Reference {
    comparisons: usize,
    swaps: usize,
    rounds: usize,
}

/// Compare-exchanges `v[k]` and `v[k + 1]`, counting into `r`.
fn compare_exchange<K: Ord>(v: &mut [K], k: usize, r: &mut Reference) -> bool {
    r.comparisons += 1;
    let swap = v[k + 1] < v[k];
    if swap {
        v.swap(k, k + 1);
        r.swaps += 1;
    }
    swap
}

/// Runs odd-even sort as documented on a copy of `keys`: rounds of an odd phase over the pairs
/// starting at `lo`, `lo + 2`, ..., `lo + 2(m - 1)` and an even phase over the pairs from
/// `lo - 1` to `lo + 2m - 1` that lie within the slice; the next round's odd phase spans one
/// pair beyond the first and the last pair the even phase swapped; it stops after a round whose
/// even phase swapped nothing, or after `⌈n/2⌉` rounds.
fn reference<K: Ord + Clone>(keys: &[K]) -> Reference {
    let mut v = keys.to_vec();
    let n = v.len();
    let mut r = Reference {
        comparisons: 0,
        swaps: 0,
        rounds: 0,
    };
    if n < 2 {
        return r;
    }
    let (mut lo, mut m) = (1, (n - 1) / 2);
    while r.rounds < n.div_ceil(2) {
        r.rounds += 1;
        for i in 0..m {
            compare_exchange(&mut v, lo + 2 * i, &mut r);
        }
        let mut swapped: Option<(usize, usize)> = None;
        let mut k = lo - 1;
        while k + 1 < n && k < lo + 2 * m {
            if compare_exchange(&mut v, k, &mut r) {
                swapped = Some((swapped.map_or(k, |s| s.0), k));
            }
            k += 2;
        }
        let Some((first, last)) = swapped else {
            break;
        };
        lo = first.saturating_sub(1).max(1);
        let hi = if last + 2 < n {
            last + 1
        } else {
            last.wrapping_sub(1)
        };
        if hi == usize::MAX || hi < lo {
            break;
        }
        m = (hi - lo) / 2 + 1;
    }
    r
}

/// Runs the textbook odd-even sort (as on Wikipedia) on a copy of `keys`: an odd phase over
/// all odd pairs and an even phase over all even pairs, until a round swaps nothing.
fn textbook<K: Ord + Clone>(keys: &[K]) -> Reference {
    let mut v = keys.to_vec();
    let n = v.len();
    let mut r = Reference {
        comparisons: 0,
        swaps: 0,
        rounds: 0,
    };
    loop {
        r.rounds += 1;
        let mut sorted = true;
        for start in [1, 0] {
            let mut k = start;
            while k + 1 < n {
                sorted &= !compare_exchange(&mut v, k, &mut r);
                k += 2;
            }
        }
        if sorted {
            return r;
        }
    }
}

/// The number of pairs of elements that are out of order.
fn inversions<K: Ord>(keys: &[K]) -> usize {
    (0..keys.len())
        .map(|j| keys[..j].iter().filter(|&e| *e > keys[j]).count())
        .sum()
}

#[test]
fn reference_never_compares_more_than_the_textbook_and_swaps_the_same() {
    // Every swap removes exactly one inversion, so both make exactly as many swaps as there
    // are inversions; and the bounds only leave out comparisons whose outcome is known.
    for_each_test_input(0x0DD0_E7E4_5EED_0001, &mut |v| {
        let ours = reference(v);
        let theirs = textbook(v);
        assert!(ours.comparisons <= theirs.comparisons, "{v:?}");
        assert_eq!(ours.swaps, theirs.swaps, "{v:?}");
        assert_eq!(ours.swaps, inversions(v), "{v:?}");
        assert!(ours.rounds <= theirs.rounds, "{v:?}");
    });
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

/// Sorts a copy of `keys` as elements larger than 8 bytes, checks the result, and returns
/// the number of comparisons.
fn comparisons<K: Ord + Clone + std::fmt::Debug>(keys: &[K]) -> usize {
    let count = Cell::new(0);
    let mut v: Vec<Counted<K>> = keys
        .iter()
        .map(|k| Counted {
            key: k.clone(),
            comparisons: &count,
        })
        .collect();
    odd_even(&mut v);
    let mut expected = keys.to_vec();
    expected.sort();
    assert_eq!(v.into_iter().map(|c| c.key).collect::<Vec<_>>(), expected);
    count.get()
}

#[test]
fn comparisons_match_the_reference() {
    for_each_test_input(0xC0C7_A11E_5AAE_0001, &mut |v| {
        assert_eq!(comparisons(v), reference(v).comparisons, "{v:?}");
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
fn reverse_sorted_input_costs_n_n_minus_1_over_2_comparisons_if_n_is_even() {
    // Reverse order needs all `n` phases, and every pair compared in them swaps, except for
    // one pair per round at the far end of the odd phases when `n` is odd: then the cost is
    // `(n² - 1)/2`. The cap of `⌈n/2⌉` rounds saves the textbook's final round, which finds
    // nothing left to swap.
    for n in 2..size(300, 40) {
        let v: Vec<usize> = (0..n).rev().collect();
        let r = reference(&v);
        let expected = if n % 2 == 0 {
            n * (n - 1) / 2
        } else {
            (n * n - 1) / 2
        };
        assert_eq!(comparisons(&v), expected, "n = {n}");
        assert_eq!(r.comparisons, expected, "n = {n}");
        assert_eq!(r.rounds, n.div_ceil(2), "n = {n}");
        assert_eq!(textbook(&v).comparisons, expected + n - 1, "n = {n}");
    }
}

#[test]
fn turtle_and_rabbit_cost_a_linear_number_of_comparisons() {
    // `[1, 2, ..., n - 1, 0]` and `[n - 1, 0, 1, ..., n - 2]`: one element travels one slot
    // per phase across the whole slice, which takes the textbook `n - 1` phases over the whole
    // slice, `≈ n²/2` comparisons; here each round compares only around the traveller.
    for n in 3..size(300, 40) {
        let turtle: Vec<usize> = (1..n).chain([0]).collect();
        let rabbit: Vec<usize> = [n - 1].into_iter().chain(0..n - 1).collect();
        // About `3.5n`: after a first round over the whole slice, five comparisons per round
        // around the traveller, which moves two slots per round.
        let expected = if n % 2 == 0 {
            (7 * n - 16) / 2
        } else {
            (7 * n - 13) / 2
        };
        // The textbook takes `⌊n/2⌋ + 1` rounds for the turtle and `⌈n/2⌉ + 1` for the rabbit.
        for (v, rounds) in [(turtle, n / 2 + 1), (rabbit, n.div_ceil(2) + 1)] {
            assert_eq!(comparisons(&v), expected, "{v:?}");
            assert_eq!(reference(&v).comparisons, expected, "{v:?}");
            assert_eq!(textbook(&v).comparisons, (n - 1) * rounds, "{v:?}");
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

/// Sorts `keys` as elements larger than 8 bytes and returns the number of element moves
/// observed, including the final one of every element that ends up away from where it was
/// last seen.
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
    odd_even(&mut v);
    for w in &v {
        w.check();
    }
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key), "{keys:?}");
    moves.get()
}

#[test]
fn moves_are_exactly_twice_the_inversions() {
    // Each swap exchanges two neighbours that are out of order, which moves both and removes
    // one inversion. Every element is compared in its slot before each swap that moves it and
    // again after it (or seen by the final check), so the observed count is exact.
    for_each_test_input(0xFACE_FEED_0DD5_EED5, &mut |v| {
        assert_eq!(observed_moves(v), 2 * inversions(v), "{v:?}");
    });
}

#[test]
fn non_decreasing_input_is_never_moved() {
    let check = |v: &[u8]| assert_eq!(observed_moves(v), 0, "{v:?}");
    for len in 0..=size(12, 6) {
        for a in 0..=len {
            for b in a..=len {
                let v: Vec<u8> = (0..len).map(|i| (i >= a) as u8 + (i >= b) as u8).collect();
                check(&v);
            }
        }
    }
    check(&(0..=255).collect::<Vec<u8>>());
}

/// Sorts elements of `PAD` extra bytes whose comparisons start panicking after `budget` of
/// them, and checks that no element is lost, duplicated, dropped or leaked. Returns whether
/// the sort completed.
fn sort_with_comparison_budget<const PAD: usize>(keys: &[u32], budget: usize) -> bool {
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

    let result = catch_unwind(AssertUnwindSafe(|| odd_even(&mut v)));

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

/// Layouts that make panics land everywhere: reverse order swaps in every pair of every phase,
/// repeated keys mix swaps and non-swaps, and the turtle and rabbit keep the bounds narrow.
fn panic_layouts(len: usize) -> Vec<Vec<u32>> {
    let mut state = 0x0123_4567_89AB_CDEF;
    let n = len as u32;
    vec![
        (0..n).rev().collect(),
        (0..n).map(|id| id % 3).collect(),
        (0..n).map(|id| (n - id) / 3).collect(),
        (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect(),
        (0..n).map(|id| id ^ 1).collect(),
        (1..n).chain([0]).collect(),
        [n - 1].into_iter().chain(0..n - 1).collect(),
    ]
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    for keys in panic_layouts(size(32, 9)) {
        // Every budget below the full number of comparisons must panic, and only those: a
        // panic from anything else fails the test instead of trying ever larger budgets.
        let total = comparisons(&keys);
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
    // still terminate: such answers almost never leave a phase without a swap, so only the
    // cap of `⌈n/2⌉` rounds ends the sort.
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
        for len in 0..size(64, 12) {
            for _ in 0..size(50, 2) {
                let names: Vec<String> = (0..len).map(|i| format!("element {i}")).collect();
                let mut v: Vec<Liar<PAD>> = names
                    .iter()
                    .map(|name| Liar {
                        name: name.clone(),
                        rng,
                        _pad: [0; PAD],
                    })
                    .collect();

                odd_even(&mut v);

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

/// A 4-byte element, small enough to be compare-exchanged without branching. It is compared by
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
    /// Moves observed as in [`Watched`]: the address each element was last seen at.
    moves: Cell<usize>,
    seen_at: RefCell<Vec<usize>>,
    drops: RefCell<Vec<u32>>,
}

thread_local! {
    static TINY: TinyState = TinyState::default();
}

impl TinyState {
    fn observe(&self, t: &Tiny) {
        let here = t as *const Tiny as usize;
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
/// and stable if the sort completed with honest comparisons.
fn sort_tiny(keys: &[usize], budget: Option<usize>, liar: Option<u64>) -> TinyRun {
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
        *s.seen_at.borrow_mut() = v.iter().map(|t| t as *const Tiny as usize).collect();
        *s.drops.borrow_mut() = vec![0; n];
    });

    let completed = catch_unwind(AssertUnwindSafe(|| odd_even(&mut v))).is_ok();

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
        assert!(
            v.windows(2)
                .all(|w| (w[0].key, w[0].id) < (w[1].key, w[1].id)),
            "{keys:?}"
        );
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
fn tiny_elements_match_the_reference() {
    // The branchless kernel makes the same comparisons, and rewriting a pair that is in
    // order in place is not a move: still exactly two moves per swap.
    for_each_test_input(0x7111_0E1E_3E17_5EED, &mut |v| {
        let run = sort_tiny(v, None, None);
        assert_eq!(run.comparisons, reference(v).comparisons, "{v:?}");
        assert_eq!(run.moves, 2 * inversions(v), "{v:?}");
    });
}

#[test]
fn tiny_non_decreasing_input_is_never_moved() {
    for n in 0..size(300, 40) {
        let sorted: Vec<usize> = (0..n).collect();
        assert_eq!(sort_tiny(&sorted, None, None).moves, 0, "n = {n}");
        let runs: Vec<usize> = (0..n).map(|i| i * 4 / n.max(1)).collect();
        assert_eq!(sort_tiny(&runs, None, None).moves, 0, "{runs:?}");
    }
}

#[test]
fn tiny_panicking_comparison_keeps_every_element_exactly_once() {
    for keys in panic_layouts(size(32, 9)) {
        let keys: Vec<usize> = keys.into_iter().map(|k| k as usize).collect();
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
    for len in 0..size(64, 12) {
        for _ in 0..size(50, 2) {
            let keys: Vec<usize> = (0..len).collect();
            sort_tiny(&keys, None, Some(xorshift(&mut seed)));
        }
    }
}

/// Inputs long enough for `odd_even_parallel` to start threads, in shapes that keep it busy
/// for different numbers of phases over different parts of the slice.
fn parallel_inputs(seed: u64) -> Vec<Vec<u32>> {
    let mut state = seed;
    let n = PARALLEL_MIN_LEN * 5 / 4 + xorshift(&mut state) as usize % (PARALLEL_MIN_LEN / 8);
    let mut random: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut random, &mut state);
    let mut nearly: Vec<u32> = (0..n as u32).collect();
    for _ in 0..n / 100 {
        let (i, j) = (
            xorshift(&mut state) as usize % n,
            xorshift(&mut state) as usize % n,
        );
        nearly.swap(i, j);
    }
    // Sorted except for a shuffled middle, so that the active part does not reach the ends.
    let mut middle: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut middle[n / 8..7 * n / 8], &mut state);
    vec![
        random,
        (0..n as u32).rev().collect(),
        nearly,
        middle,
        (0..n).map(|_| xorshift(&mut state) as u32 % 4).collect(),
        (0..n as u32).map(|i| i % 500).collect(),
        // The two ends travel to the far side: the slice is one long active range.
        [n as u32]
            .into_iter()
            .chain(1..n as u32 - 1)
            .chain([0])
            .collect(),
    ]
}

#[test]
fn parallel_sorts_long_inputs_stably() {
    for keys in parallel_inputs(1) {
        assert_sorts_like_std_stable::<u32, SMALL>(&keys, true);
        assert_sorts_like_std_stable::<u32, LARGE>(&keys, true);
    }
    let mut state = 77;
    let mut v: Vec<String> = (0..PARALLEL_MIN_LEN * 3 / 2)
        .map(|_| format!("{:x}", xorshift(&mut state)))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    odd_even_parallel(&mut v);
    assert_eq!(v, expected);
}

/// An element for the parallel entry point, which needs `Send` elements: it keeps its
/// bookkeeping — comparisons, moves and drops, per element so that threads do not contend —
/// in a shared [`SharedState`].
struct Shared {
    key: u32,
    id: u32,
    state: Arc<SharedState>,
}

struct SharedState {
    /// Comparisons each element made as `self`.
    compares: Vec<AtomicUsize>,
    /// Moves observed as in [`Watched`]: the address each element was last seen at.
    seen_at: Vec<AtomicUsize>,
    moves: Vec<AtomicUsize>,
    drops: Vec<AtomicU32>,
    /// The element whose comparison panics, and which of its comparisons that is.
    panic_at: (u32, usize),
    /// Random answers instead of the keys, if non-zero.
    liar: u64,
}

impl Shared {
    fn observe(&self) {
        let here = self as *const Self as usize;
        let id = self.id as usize;
        if self.state.seen_at[id].swap(here, Atomic::Relaxed) != here {
            self.state.moves[id].fetch_add(1, Atomic::Relaxed);
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        self.state.drops[self.id as usize].fetch_add(1, Atomic::Relaxed);
    }
}
impl PartialEq for Shared {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Shared {}
impl PartialOrd for Shared {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Shared {
    fn cmp(&self, other: &Self) -> Ordering {
        let s = &self.state;
        let c = s.compares[self.id as usize].fetch_add(1, Atomic::Relaxed);
        if (self.id, c) == s.panic_at {
            panic!("comparison budget exhausted");
        }
        self.observe();
        other.observe();
        if s.liar != 0 {
            let mut x =
                s.liar ^ (u64::from(self.id) << 40) ^ (u64::from(other.id) << 20) ^ c as u64;
            let r = xorshift(&mut x);
            return [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize];
        }
        self.key.cmp(&other.key)
    }
}

/// What happened when [`Shared`] elements were sorted by `odd_even_parallel`.
struct SharedRun {
    /// Comparisons made by each element as `self`.
    compares: Vec<usize>,
    moves: usize,
    completed: bool,
}

/// Sorts `keys` as [`Shared`] elements with `odd_even_parallel`, with the comparison
/// `panic_at` panicking and comparisons answered at random from a non-zero `liar`. Checks that
/// every element is still there exactly once, unchanged, and dropped exactly once afterwards,
/// that a panic reaches the caller as it was raised, and that the result is sorted and stable
/// if the sort completed with honest comparisons.
fn sort_shared(keys: &[u32], panic_at: (u32, usize), liar: u64) -> SharedRun {
    let n = keys.len();
    let counters = || (0..n).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>();
    let state = Arc::new(SharedState {
        compares: counters(),
        seen_at: counters(),
        moves: counters(),
        drops: (0..n).map(|_| AtomicU32::new(0)).collect(),
        panic_at,
        liar,
    });
    let mut v: Vec<Shared> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Shared {
            key,
            id,
            state: Arc::clone(&state),
        })
        .collect();
    for e in &v {
        state.seen_at[e.id as usize].store(e as *const Shared as usize, Atomic::Relaxed);
    }

    let result = catch_unwind(AssertUnwindSafe(|| odd_even_parallel(&mut v)));

    if let Err(payload) = &result {
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"comparison budget exhausted")
        );
    }
    for e in &v {
        e.observe();
    }
    if result.is_ok() && liar == 0 {
        assert!(
            v.windows(2)
                .all(|w| (w[0].key, w[0].id) < (w[1].key, w[1].id)),
            "not sorted and stable"
        );
    }
    let mut ids: Vec<u32> = v.iter().map(|e| e.id).collect();
    ids.sort_unstable();
    assert!(
        ids.into_iter().eq(0..n as u32),
        "element lost or duplicated"
    );
    assert!(
        v.iter().all(|e| e.key == keys[e.id as usize]),
        "element changed during sort"
    );
    assert!(
        state.drops.iter().all(|d| d.load(Atomic::Relaxed) == 0),
        "element dropped during sort"
    );
    let run = SharedRun {
        compares: state
            .compares
            .iter()
            .map(|c| c.load(Atomic::Relaxed))
            .collect(),
        moves: state.moves.iter().map(|m| m.load(Atomic::Relaxed)).sum(),
        completed: result.is_ok(),
    };
    drop(v);
    assert!(
        state.drops.iter().all(|d| d.load(Atomic::Relaxed) == 1),
        "element leaked or double-dropped"
    );
    run
}

/// No comparison panics.
const NEVER: (u32, usize) = (u32::MAX, 0);

/// The number of inversions of `keys`, by merge sort.
fn inversions_fast(keys: &[u32]) -> usize {
    fn go(v: &mut [u32], buf: &mut Vec<u32>) -> usize {
        if v.len() < 2 {
            return 0;
        }
        let mid = v.len() / 2;
        let mut count = go(&mut v[..mid], buf) + go(&mut v[mid..], buf);
        buf.clear();
        let (mut i, mut j) = (0, mid);
        while i < mid && j < v.len() {
            if v[j] < v[i] {
                count += mid - i;
                buf.push(v[j]);
                j += 1;
            } else {
                buf.push(v[i]);
                i += 1;
            }
        }
        buf.extend_from_slice(&v[i..mid]);
        buf.extend_from_slice(&v[j..]);
        v.copy_from_slice(buf);
        count
    }
    go(&mut keys.to_vec(), &mut Vec::new())
}

#[test]
fn parallel_moves_are_exactly_twice_the_inversions() {
    // The threads make exactly the swaps of the sequential sort, each of two neighbours that
    // are out of order: anything else would show up as extra moves.
    for keys in parallel_inputs(11) {
        let run = sort_shared(&keys, NEVER, 0);
        assert!(run.completed);
        assert_eq!(run.moves, 2 * inversions_fast(&keys));
    }
}

#[test]
fn parallel_panicking_comparison_keeps_every_element_exactly_once() {
    // Panics land on the calling thread before the threads start and in the threads, at
    // elements spread over the slice; each must reach the caller as it was raised.
    let mut state = 0x9A11_0C0D_E5EE_D5ED;
    let mut inputs = parallel_inputs(5);
    inputs.truncate(2);
    for keys in inputs {
        let compares = sort_shared(&keys, NEVER, 0).compares;
        for _ in 0..size(6, 2) {
            let id = xorshift(&mut state) as usize % keys.len();
            let at = xorshift(&mut state) as usize % compares[id].max(1);
            let run = sort_shared(&keys, (id as u32, at), 0);
            assert_eq!(run.completed, compares[id] == 0);
        }
    }
}

#[test]
fn parallel_inconsistent_ord_keeps_every_element_exactly_once() {
    // Random answers keep every phase swapping until the cap of `⌈n/2⌉` rounds.
    let keys = parallel_inputs(3).swap_remove(0);
    assert!(sort_shared(&keys, NEVER, 0x5EED).completed);
}

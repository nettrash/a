use a::sort::{bitonic, bitonic_parallel};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::mem::size_of;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{self, AtomicIsize, AtomicUsize};
use std::sync::{Mutex, MutexGuard, PoisonError};

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Bitonic sort exchanges elements of at most 8 bytes by selecting values, elements of 9 to 24
/// bytes by selecting pointers in its small networks and by branching elsewhere, and larger
/// elements by branching. The test elements carry `PAD` bytes of payload after 8 bytes of key
/// and identity, one size for each way.
const VALUES: usize = 0;
const POINTERS: usize = 8;
const BRANCH: usize = 56;

/// [`bitonic_parallel`] uses a thread per this many elements of at most 8 bytes, at most as
/// many as the system has, so from twice this length on it splits the work across threads.
const PER_THREAD_SMALL: usize = 1 << 14;
/// The same for larger elements.
const PER_THREAD: usize = 1 << 12;

#[test]
fn test_elements_cover_every_exchange() {
    assert_eq!(size_of::<Keyed<VALUES>>(), 8);
    assert_eq!(size_of::<Keyed<POINTERS>>(), 16);
    assert_eq!(size_of::<Keyed<BRANCH>>(), 64);
    assert_eq!(size_of::<Tracked<VALUES>>(), 8);
    assert_eq!(size_of::<Tracked<POINTERS>>(), 16);
    assert_eq!(size_of::<Tracked<BRANCH>>(), 64);
    assert_eq!(size_of::<Probe<VALUES>>(), 4);
    assert_eq!(size_of::<Probe<{ POINTERS + 4 }>>(), 16);
    assert_eq!(size_of::<Probe<{ BRANCH + 4 }>>(), 64);
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    bitonic(&mut v);
    assert_eq!(v, vec![]);
    bitonic_parallel(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    bitonic(&mut v);
    assert_eq!(v, vec![42]);
    bitonic_parallel(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    bitonic(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    bitonic(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    bitonic(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    bitonic(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    bitonic(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        bitonic(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear", "apple", "banana", "apple", "fig", "kiwi", "date", "plum", "lime", "cherry",
    ];
    bitonic(&mut v);
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
    for len in [0, 1, 17, 200, size(1000, 300)] {
        let mut v: Vec<String> = (0..len)
            .map(|_| format!("{:03}", xorshift(&mut state) % 1000))
            .collect();
        let mut expected = v.clone();
        expected.sort();
        bitonic(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [3, 31, 200, size(3000, 300)] {
        let mut v: Vec<[u64; 32]> = (0..len).map(|_| [xorshift(&mut state) % 50; 32]).collect();
        let mut expected = v.clone();
        expected.sort();
        bitonic(&mut v);
        assert_eq!(v, expected);
    }
}

#[test]
fn sorts_zero_sized_elements() {
    for len in 0..size(300, 40) {
        let mut v = vec![(); len];
        bitonic(&mut v);
        assert_eq!(v.len(), len);
    }
    let mut v = vec![(); size(4 * PER_THREAD_SMALL + 1, 2 * PER_THREAD_SMALL + 1)];
    bitonic_parallel(&mut v);
    assert_eq!(
        v.len(),
        size(4 * PER_THREAD_SMALL + 1, 2 * PER_THREAD_SMALL + 1)
    );
}

#[test]
fn sorts_long_inputs() {
    // Lengths around powers of two, where the merges switch between the arbitrary-length
    // and the power-of-two path, and odd ones in between.
    let lens = if cfg!(miri) {
        vec![127, 128, 129, 1000]
    } else {
        vec![1023, 1024, 1025, 3000, 4096, 65_537, 100_000]
    };
    let mut state = 0x7E57_7E57_7E57_7E57;
    for n in lens {
        for mut v in shapes(n, &mut state) {
            let mut expected = v.clone();
            expected.sort_unstable();
            let mut narrow: Vec<Keyed<POINTERS>> = keyed(&v);
            let mut wide: Vec<Keyed<BRANCH>> = keyed(&v);
            bitonic(&mut v);
            bitonic(&mut narrow);
            bitonic(&mut wide);
            assert_eq!(v, expected, "n = {n}");
            assert!(narrow.iter().map(|e| e.key).eq(expected.iter().copied()));
            assert!(wide.iter().map(|e| e.key).eq(expected.iter().copied()));
        }
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..60).map(|i| (i * 17) % 60).collect();
    let before = v.clone();
    bitonic(&mut v[5..45]);
    let mut middle = before[5..45].to_vec();
    middle.sort_unstable();
    assert_eq!(&v[..5], &before[..5]);
    assert_eq!(&v[5..45], &middle[..]);
    assert_eq!(&v[45..], &before[45..]);
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

/// Inputs of length `n` in several shapes: random, few distinct keys, sorted, reverse
/// sorted, organ pipe, sawtooth and nearly sorted.
fn shapes(n: usize, state: &mut u64) -> Vec<Vec<u32>> {
    let m = n as u32;
    let mut nearly: Vec<u32> = (0..m).collect();
    for _ in 0..n / 50 + 1 {
        let i = (xorshift(state) % n.max(1) as u64) as usize;
        let j = (xorshift(state) % n.max(1) as u64) as usize;
        nearly.swap(i, j);
    }
    vec![
        (0..n).map(|_| xorshift(state) as u32).collect(),
        (0..n).map(|_| xorshift(state) as u32 % 4).collect(),
        (0..m).collect(),
        (0..m).rev().collect(),
        (0..m).map(|i| i.min(m - 1 - i)).collect(),
        (0..m).map(|i| i % 97).collect(),
        nearly,
    ]
}

/// Calls `f` with every permutation of `0..n` (Heap's algorithm).
fn for_each_permutation(n: usize, f: &mut impl FnMut(&[u32])) {
    fn go(k: usize, v: &mut [u32], f: &mut impl FnMut(&[u32])) {
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
    let mut v: Vec<u32> = (0..n as u32).collect();
    go(n, &mut v, f);
}

/// Calls `f` with every sequence of length `len` over the keys `0..keys`.
fn for_each_sequence(len: u32, keys: u32, f: &mut impl FnMut(&[u32])) {
    for code in 0..(keys as usize).pow(len) {
        let mut c = code;
        let v: Vec<u32> = (0..len)
            .map(|_| {
                let d = (c % keys as usize) as u32;
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
struct Keyed<const PAD: usize> {
    key: u32,
    idx: u32,
    _pad: [u8; PAD],
}
impl<const PAD: usize> PartialEq for Keyed<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for Keyed<PAD> {}
impl<const PAD: usize> PartialOrd for Keyed<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for Keyed<PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

fn keyed<const PAD: usize>(keys: &[u32]) -> Vec<Keyed<PAD>> {
    keys.iter()
        .enumerate()
        .map(|(idx, &key)| Keyed {
            key,
            idx: idx as u32,
            _pad: [0; PAD],
        })
        .collect()
}

/// Sorts `keys`, each tagged with its index, with `sort`, and checks that the keys come out in
/// order and that every element is still there exactly once, unchanged.
fn assert_sorts_with<const PAD: usize>(keys: &[u32], sort: fn(&mut [Keyed<PAD>])) {
    let mut v = keyed::<PAD>(keys);
    sort(&mut v);
    assert!(
        v.windows(2).all(|w| w[0].key <= w[1].key),
        "input: {keys:?}"
    );
    let mut idx: Vec<u32> = v.iter().map(|e| e.idx).collect();
    idx.sort_unstable();
    assert!(idx.into_iter().eq(0..keys.len() as u32), "input: {keys:?}");
    assert!(
        v.iter().all(|e| e.key == keys[e.idx as usize]),
        "element changed: {keys:?}"
    );
}

fn assert_sorts(keys: &[u32]) {
    assert_sorts_with::<VALUES>(keys, bitonic);
    assert_sorts_with::<POINTERS>(keys, bitonic);
    assert_sorts_with::<BRANCH>(keys, bitonic);
}

#[test]
fn sorts_every_permutation_up_to_8() {
    for n in 0..=size(8, 5) {
        for_each_permutation(n, &mut |p| assert_sorts(p));
    }
}

#[test]
fn sorts_every_binary_sequence() {
    // By the 0-1 principle, a comparator network that sorts every sequence of zeros and ones
    // of a length sorts every input of that length. Up to 16 elements, this is every network
    // the sort applies as a whole in registers.
    for len in 0..=size(16, 9) as u32 {
        for_each_sequence(len, 2, &mut |s| assert_sorts(s));
    }
}

#[test]
fn sorts_every_sequence_over_three_keys() {
    for len in 0..=size(10, 6) as u32 {
        for_each_sequence(len, 3, &mut |s| assert_sorts(s));
    }
}

#[test]
fn sorts_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..size(600, 70) {
        // Keys from a small range force plenty of duplicates, keys from the full range
        // make them almost all distinct.
        for modulus in [2, 3, 64, u64::MAX] {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % modulus) as u32)
                .collect();
            assert_sorts(&keys);
        }
    }
}

/// The comparators of the bitonic sorting network for `n` elements in H. W. Lang's
/// formulation for arbitrary `n`, as `(i, j)` with `i < j`, in the order of the textbook
/// recursion: the first half is sorted descending, the second ascending, and the merge of
/// `n` elements compares `i` with `i + m` for `m` the largest power of two below `n`, then
/// merges `0..m` and `m..n`.
fn lang_network(n: usize) -> Vec<(usize, usize, bool)> {
    fn sort(lo: usize, n: usize, up: bool, out: &mut Vec<(usize, usize, bool)>) {
        if n > 1 {
            let m = n / 2;
            sort(lo, m, !up, out);
            sort(lo + m, n - m, up, out);
            merge(lo, n, up, out);
        }
    }
    fn merge(lo: usize, n: usize, up: bool, out: &mut Vec<(usize, usize, bool)>) {
        if n > 1 {
            let m = 1 << (n - 1).ilog2();
            for i in lo..lo + n - m {
                out.push((i, i + m, up));
            }
            merge(lo, m, up, out);
            merge(lo + m, n - m, up, out);
        }
    }
    let mut out = Vec::new();
    sort(0, n, true, &mut out);
    out
}

#[test]
fn lang_network_sorts_every_binary_sequence() {
    // The reference network itself, checked by the 0-1 principle on bit masks.
    for n in 0..=size(20, 10) {
        let network = lang_network(n);
        for input in 0..1u32 << n {
            let mut bits = input;
            for &(i, j, up) in &network {
                let (bi, bj) = ((bits >> i) & 1, (bits >> j) & 1);
                if (bi > bj) == up && bi != bj {
                    bits ^= (1 << i) | (1 << j);
                }
            }
            let ones = input.count_ones();
            assert_eq!(
                bits,
                ((1u64 << n) - (1u64 << (n as u32 - ones))) as u32,
                "n = {n}"
            );
        }
    }
}

#[test]
fn lang_network_has_n_k_k_plus_1_over_4_comparators_for_n_up_to_2_to_the_k() {
    for k in 0..=size(16, 10) {
        let n = 1usize << k;
        assert_eq!(lang_network(n).len(), n * k * (k + 1) / 4);
    }
    // For `2^(k−1) < n < 2^k`, at most `n·k·(k + 1) / 4` and at least `n·(k − 1)² / 4`.
    for n in 3..size(5000, 300) {
        let k = (n - 1).ilog2() as usize + 1;
        let count = lang_network(n).len();
        assert!(n * (k - 1) * (k - 1) <= 4 * count, "n = {n}");
        assert!(4 * count <= n * k * (k + 1), "n = {n}");
    }
}

/// An element whose comparisons record the positions, within the slice registered in
/// [`PROBE`], of the two elements compared. 4 bytes plus `PAD`.
struct Probe<const PAD: usize> {
    key: u32,
    _pad: [u8; PAD],
}

/// The slice the [`Probe`]s compared on this thread live in, and the positions compared.
struct ProbeLog {
    base: Cell<usize>,
    pairs: RefCell<Vec<(usize, usize)>>,
}

thread_local! {
    static PROBE: ProbeLog = const {
        ProbeLog {
            base: Cell::new(0),
            pairs: RefCell::new(Vec::new()),
        }
    };
}

impl<const PAD: usize> PartialEq for Probe<PAD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize> Eq for Probe<PAD> {}
impl<const PAD: usize> PartialOrd for Probe<PAD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize> Ord for Probe<PAD> {
    fn cmp(&self, other: &Self) -> Ordering {
        PROBE.with(|log| {
            let at = |e: &Self| (e as *const Self as usize - log.base.get()) / size_of::<Self>();
            let (i, j) = (at(self), at(other));
            log.pairs.borrow_mut().push((i.min(j), i.max(j)));
        });
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys` as [`Probe`]s and returns the positions compared, in order.
fn compared_positions<const PAD: usize>(keys: &[u32]) -> Vec<(usize, usize)> {
    let mut v: Vec<Probe<PAD>> = keys
        .iter()
        .map(|&key| Probe {
            key,
            _pad: [0; PAD],
        })
        .collect();
    PROBE.with(|log| {
        log.base.set(v.as_ptr() as usize);
        log.pairs.borrow_mut().clear();
    });
    bitonic(&mut v);
    assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
    PROBE.with(|log| log.pairs.take())
}

fn assert_applies_lang_network<const PAD: usize>(n: usize, state: &mut u64) {
    let mut expected: Vec<(usize, usize)> =
        lang_network(n).iter().map(|&(i, j, _)| (i, j)).collect();
    expected.sort_unstable();
    let random: Vec<u32> = (0..n).map(|_| xorshift(state) as u32).collect();
    let sorted: Vec<u32> = (0..n as u32).collect();
    let first = compared_positions::<PAD>(&random);
    // Data-independent: the same positions are compared in the same order whatever the
    // input.
    assert_eq!(first, compared_positions::<PAD>(&sorted), "n = {n}");
    assert_eq!(first, compared_positions::<PAD>(&vec![1; n]), "n = {n}");
    let mut pairs = first;
    pairs.sort_unstable();
    assert_eq!(pairs, expected, "n = {n}");
}

#[test]
fn compares_exactly_the_pairs_of_lang_network() {
    // Each element size takes its own exchanges and merge structure, but all apply the same
    // comparators, each once.
    let mut state = 0x0BAD_C0FF_EE00_D00D;
    let lens = (0..size(300, 40)).chain(if cfg!(miri) {
        vec![64, 100, 128, 129]
    } else {
        vec![511, 512, 513, 1000, 1024, 2049, 4096, 5000]
    });
    for n in lens {
        assert_applies_lang_network::<VALUES>(n, &mut state);
        assert_applies_lang_network::<{ POINTERS + 4 }>(n, &mut state);
        assert_applies_lang_network::<{ BRANCH + 4 }>(n, &mut state);
    }
}

/// Sorts a copy of `keys` and returns the number of comparisons.
fn comparisons(keys: &[u32]) -> usize {
    compared_positions::<VALUES>(keys).len()
}

#[test]
fn every_input_costs_the_same_number_of_comparisons() {
    // `n·k·(k + 1) / 4` for `n = 2^k`: 2,816 for 256 elements, against about 2,000 for
    // `n·log₂ n`.
    let mut state = 0x5EED_0F5E_ED0F_5EED;
    for k in 0..=size(12, 8) {
        let n = 1usize << k;
        for keys in shapes(n, &mut state) {
            assert_eq!(comparisons(&keys), n * k * (k + 1) / 4, "n = {n}");
        }
    }
    for n in [3, 100, 1000, size(10_000, 300)] {
        let expected = lang_network(n).len();
        for keys in shapes(n, &mut state) {
            assert_eq!(comparisons(&keys), expected, "n = {n}");
        }
    }
}

/// Sorts `v` and returns how many allocations that made.
fn allocations_of<T: Ord + Send>(v: &mut [T], sort: fn(&mut [T])) -> usize {
    crate::allocations::count(|| sort(v))
}

#[test]
fn never_allocates() {
    let mut state = 0xA110_CA7E_A110_CA7E;
    for n in [0, 1, 2, 17, 100, 1000, size(2 * PER_THREAD_SMALL + 1, 1500)] {
        let mut v: Vec<u64> = (0..n).map(|_| xorshift(&mut state)).collect();
        assert_eq!(allocations_of(&mut v, bitonic), 0, "n = {n}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
        let mut v: Vec<String> = (0..n.min(3000))
            .map(|i| format!("{}", i * 7919 % 3001))
            .collect();
        assert_eq!(allocations_of(&mut v, bitonic), 0, "n = {n}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}

#[test]
fn parallel_does_not_allocate_below_its_threshold() {
    // No threads, no task lists: shorter slices are sorted on the calling thread.
    let mut state = 0xA110_CA7E_0000_0001;
    for n in [0, 1, 100, size(2 * PER_THREAD_SMALL - 1, 2000)] {
        let mut v: Vec<u64> = (0..n).map(|_| xorshift(&mut state)).collect();
        assert_eq!(allocations_of(&mut v, bitonic_parallel), 0, "n = {n}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    for n in [0, 1, 100, size(2 * PER_THREAD - 1, 2000)] {
        let mut v: Vec<Keyed<POINTERS>> = keyed(&(0..n as u32).rev().collect::<Vec<_>>());
        assert_eq!(allocations_of(&mut v, bitonic_parallel), 0, "n = {n}");
        assert!(v.iter().map(|e| e.key).eq(0..n as u32));
    }
}

/// The bookkeeping of the [`Tracked`] elements of the sort running on this thread.
#[derive(Default)]
struct TrackState {
    /// How many times each element has been dropped, by id.
    drops: RefCell<Vec<u32>>,
    /// Comparisons left before one panics, if limited.
    budget: Cell<Option<usize>>,
    comparisons: Cell<usize>,
    /// The state of a random generator that answers comparisons instead of the keys, if set.
    liar: Cell<Option<u64>>,
}

thread_local! {
    static TRACK: TrackState = TrackState::default();
}

/// An element compared by `key`, or at random if [`TRACK`] says so, that counts its drops and
/// the comparisons made and panics once the comparison budget is spent. 8 bytes plus `PAD`.
struct Tracked<const PAD: usize> {
    key: u32,
    id: u32,
    _pad: [u8; PAD],
}

impl<const PAD: usize> Drop for Tracked<PAD> {
    fn drop(&mut self) {
        TRACK.with(|s| s.drops.borrow_mut()[self.id as usize] += 1);
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
        TRACK.with(|s| {
            if let Some(left) = s.budget.get() {
                if left == 0 {
                    panic!("comparison budget exhausted");
                }
                s.budget.set(Some(left - 1));
            }
            s.comparisons.set(s.comparisons.get() + 1);
            if let Some(mut state) = s.liar.get() {
                let r = xorshift(&mut state);
                s.liar.set(Some(state));
                return [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize];
            }
            self.key.cmp(&other.key)
        })
    }
}

/// Sorts `keys` as [`Tracked`] elements with `budget` comparisons, all if `None`, answered at
/// random from `liar` if set. Checks that no element is lost, duplicated, changed, dropped
/// during the sort or leaked, and that the result is sorted if the sort completed honestly.
/// Returns whether it completed, and how many comparisons it made.
fn sort_tracked<const PAD: usize>(
    keys: &[u32],
    budget: Option<usize>,
    liar: Option<u64>,
) -> (bool, usize) {
    let len = keys.len();
    TRACK.with(|s| {
        *s.drops.borrow_mut() = vec![0; len];
        s.budget.set(budget);
        s.comparisons.set(0);
        s.liar.set(liar);
    });
    let mut v: Vec<Tracked<PAD>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Tracked {
            key,
            id,
            _pad: [0; PAD],
        })
        .collect();

    let completed = catch_unwind(AssertUnwindSafe(|| bitonic(&mut v))).is_ok();

    if completed && liar.is_none() {
        assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
    }
    let mut ids: Vec<u32> = v.iter().map(|t| t.id).collect();
    ids.sort_unstable();
    assert!(
        ids.into_iter().eq(0..len as u32),
        "element lost or duplicated"
    );
    assert!(
        v.iter().all(|t| t.key == keys[t.id as usize]),
        "element changed during sort"
    );
    let comparisons = TRACK.with(|s| {
        s.budget.set(None);
        s.liar.set(None);
        assert!(
            s.drops.borrow().iter().all(|&d| d == 0),
            "element dropped during sort"
        );
        s.comparisons.get()
    });
    drop(v);
    TRACK.with(|s| {
        assert!(
            s.drops.borrow().iter().all(|&d| d == 1),
            "element leaked or double-dropped"
        );
    });
    (completed, comparisons)
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    // A panic can land in any compare-exchange of the networks in registers, the merges'
    // passes of two, four or eight parts, and between them. Every budget below the number
    // of comparisons must panic, and only those: a panic from anything else fails the test.
    fn check<const PAD: usize>(keys: &[u32], step: usize) {
        let (completed, total) = sort_tracked::<PAD>(keys, None, None);
        assert!(completed);
        for budget in (0..total).step_by(step) {
            let (completed, _) = sort_tracked::<PAD>(keys, Some(budget), None);
            assert!(!completed, "budget {budget}: {keys:?}");
        }
        assert!(sort_tracked::<PAD>(keys, Some(total), None).0);
    }
    let mut state = 0x0123_4567_89AB_CDEF;
    for n in [2, 7, 16, 17, size(40, 24)] {
        for mut keys in shapes(n, &mut state) {
            keys.iter_mut().for_each(|k| *k %= 7);
            check::<VALUES>(&keys, 1);
            check::<POINTERS>(&keys, 1);
            check::<BRANCH>(&keys, 1);
        }
    }
    // Long enough for passes of eight parts.
    for keys in shapes(size(300, 130), &mut state) {
        check::<VALUES>(&keys, size(7, 97));
        check::<POINTERS>(&keys, size(7, 97));
        check::<BRANCH>(&keys, size(7, 97));
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that answers at random makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // terminate. The network does not look at the answers to decide what to compare next,
    // so it makes exactly as many comparisons as with an honest `Ord`.
    let mut state = 0x5EED_5EED_5EED_5EED;
    for len in (0..size(70, 20)).chain([size(300, 130), size(1000, 140)]) {
        for _ in 0..size(4, 1) {
            let keys: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32).collect();
            let liar = Some(xorshift(&mut state) | 1);
            let honest = lang_network(len).len();
            assert_eq!(sort_tracked::<VALUES>(&keys, None, liar), (true, honest));
            assert_eq!(sort_tracked::<POINTERS>(&keys, None, liar), (true, honest));
            assert_eq!(sort_tracked::<BRANCH>(&keys, None, liar), (true, honest));
        }
    }
}

/// Serialises the tests that share [`SHARED`].
static SHARED_LOCK: Mutex<()> = Mutex::new(());

fn lock_shared() -> MutexGuard<'static, ()> {
    SHARED_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The bookkeeping of [`Shared`] elements, which may be compared and moved on any thread.
struct SharedState {
    drops: Mutex<Vec<u32>>,
    /// Comparisons left before they panic; `isize::MAX` for no limit.
    budget: AtomicIsize,
    comparisons: AtomicUsize,
    /// The slice the elements live in, and the positions compared.
    base: AtomicUsize,
    pairs: Mutex<Vec<(usize, usize)>>,
    /// Whether to answer comparisons at random rather than by key.
    lie: atomic::AtomicBool,
}

static SHARED: SharedState = SharedState {
    drops: Mutex::new(Vec::new()),
    budget: AtomicIsize::new(isize::MAX),
    comparisons: AtomicUsize::new(0),
    base: AtomicUsize::new(0),
    pairs: Mutex::new(Vec::new()),
    lie: atomic::AtomicBool::new(false),
};

/// An element for [`bitonic_parallel`] with the bookkeeping of [`Tracked`] in [`SHARED`],
/// recording the positions compared when `RECORD` is set. 16 bytes plus `PAD`.
struct Shared<const PAD: usize, const RECORD: bool> {
    key: u32,
    id: u32,
    /// Distinguishes equal keys for the random answers, which must not depend on the thread.
    salt: u64,
    _pad: [u8; PAD],
}

impl<const PAD: usize, const RECORD: bool> Drop for Shared<PAD, RECORD> {
    fn drop(&mut self) {
        SHARED.drops.lock().unwrap()[self.id as usize] += 1;
    }
}
impl<const PAD: usize, const RECORD: bool> PartialEq for Shared<PAD, RECORD> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl<const PAD: usize, const RECORD: bool> Eq for Shared<PAD, RECORD> {}
impl<const PAD: usize, const RECORD: bool> PartialOrd for Shared<PAD, RECORD> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<const PAD: usize, const RECORD: bool> Ord for Shared<PAD, RECORD> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Once the budget is spent, every comparison on any thread panics.
        if SHARED.budget.fetch_sub(1, atomic::Ordering::Relaxed) <= 0 {
            panic!("comparison budget exhausted");
        }
        SHARED.comparisons.fetch_add(1, atomic::Ordering::Relaxed);
        if RECORD {
            let base = SHARED.base.load(atomic::Ordering::Relaxed);
            let at = |e: &Self| (e as *const Self as usize - base) / size_of::<Self>();
            let (i, j) = (at(self), at(other));
            SHARED.pairs.lock().unwrap().push((i.min(j), i.max(j)));
        }
        if SHARED.lie.load(atomic::Ordering::Relaxed) {
            // Inconsistent, but a function of the two elements only, so the same on every
            // thread and in every order of execution.
            let mut h = self.salt ^ other.salt.rotate_left(17) ^ 0x9E37_79B9_7F4A_7C15;
            return [Ordering::Less, Ordering::Equal, Ordering::Greater]
                [(xorshift(&mut h) % 3) as usize];
        }
        self.key.cmp(&other.key)
    }
}

/// Sorts `keys` as [`Shared`] elements with `sort`, with `budget` comparisons (`isize::MAX`
/// for all), answered at random if `lie`. Checks that no element is lost, duplicated,
/// changed, dropped during the sort or leaked, and that the result is sorted if the sort
/// completed honestly. Returns the panic message if the sort panicked, the ids in their final
/// order, and the number of comparisons.
fn sort_shared<const PAD: usize, const RECORD: bool>(
    keys: &[u32],
    sort: fn(&mut [Shared<PAD, RECORD>]),
    budget: isize,
    lie: bool,
) -> (Option<String>, Vec<u32>, usize) {
    let len = keys.len();
    *SHARED.drops.lock().unwrap() = vec![0; len];
    SHARED.budget.store(budget, atomic::Ordering::Relaxed);
    SHARED.comparisons.store(0, atomic::Ordering::Relaxed);
    SHARED.lie.store(lie, atomic::Ordering::Relaxed);
    SHARED.pairs.lock().unwrap().clear();
    let mut v: Vec<Shared<PAD, RECORD>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Shared {
            key,
            id,
            salt: u64::from(id).wrapping_mul(0x2545_F491_4F6C_DD1D),
            _pad: [0; PAD],
        })
        .collect();
    SHARED
        .base
        .store(v.as_ptr() as usize, atomic::Ordering::Relaxed);

    let result = catch_unwind(AssertUnwindSafe(|| sort(&mut v)));
    let message = result.err().map(|payload| {
        payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_default()
    });

    SHARED.budget.store(isize::MAX, atomic::Ordering::Relaxed);
    SHARED.lie.store(false, atomic::Ordering::Relaxed);
    if message.is_none() && !lie {
        assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
    }
    let order: Vec<u32> = v.iter().map(|t| t.id).collect();
    let mut ids = order.clone();
    ids.sort_unstable();
    assert!(
        ids.into_iter().eq(0..len as u32),
        "element lost or duplicated"
    );
    assert!(
        v.iter().all(|t| t.key == keys[t.id as usize]),
        "element changed during sort"
    );
    assert!(
        SHARED.drops.lock().unwrap().iter().all(|&d| d == 0),
        "element dropped during sort"
    );
    drop(v);
    assert!(
        SHARED.drops.lock().unwrap().iter().all(|&d| d == 1),
        "element leaked or double-dropped"
    );
    let comparisons = SHARED.comparisons.load(atomic::Ordering::Relaxed);
    (message, order, comparisons)
}

/// Lengths of elements of more than 8 bytes that take the parallel path: the shortest, odd
/// lengths whose halves and merges split unevenly, and a power of two.
fn parallel_lens() -> Vec<usize> {
    if cfg!(miri) {
        vec![2 * PER_THREAD + 3]
    } else {
        vec![2 * PER_THREAD, 2 * PER_THREAD + 1, 50_001, 1 << 17, 300_007]
    }
}

#[test]
fn parallel_sorts_long_inputs() {
    let mut state = 0x9A7A_11E1_9A7A_11E1;
    for n in parallel_lens() {
        for keys in shapes(n, &mut state).into_iter().take(size(7, 2)) {
            let mut expected = keys.clone();
            expected.sort_unstable();
            let mut narrow: Vec<Keyed<POINTERS>> = keyed(&keys);
            bitonic_parallel(&mut narrow);
            assert!(narrow.iter().map(|e| e.key).eq(expected.iter().copied()));
            if !cfg!(miri) {
                let mut wide: Vec<Keyed<BRANCH>> = keyed(&keys);
                bitonic_parallel(&mut wide);
                assert!(wide.iter().map(|e| e.key).eq(expected.iter().copied()));
            }
        }
    }
}

#[test]
fn parallel_sorts_long_inputs_of_small_elements() {
    // Elements of at most 8 bytes take the parallel path from twice as many elements per
    // thread; too long for Miri, which checks the same tasks on larger elements.
    let lens = if cfg!(miri) {
        vec![]
    } else {
        vec![
            2 * PER_THREAD_SMALL,
            2 * PER_THREAD_SMALL + 1,
            100_003,
            1 << 18,
        ]
    };
    let mut state = 0x5A11_E1E5_5A11_E1E5;
    for n in lens {
        for mut v in shapes(n, &mut state) {
            let mut expected = v.clone();
            expected.sort_unstable();
            bitonic_parallel(&mut v);
            assert_eq!(v, expected, "n = {n}");
        }
    }
}

#[test]
fn parallel_sorts_strings() {
    let mut state = 0x5781_9C55_5781_9C55;
    let n = size(10 * PER_THREAD + 5, 2 * PER_THREAD + 1);
    let mut v: Vec<String> = (0..n)
        .map(|_| format!("{:x}", xorshift(&mut state) % 100_000))
        .collect();
    let mut expected = v.clone();
    expected.sort();
    bitonic_parallel(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn parallel_applies_the_same_network() {
    // Equal keys come out in an order that depends on nothing but the comparators applied,
    // so the parallel sort must leave them exactly where the sequential one does, and make
    // the same comparisons.
    let _guard = lock_shared();
    let mut state = 0x5A3E_0000_5A3E_0000;
    for n in parallel_lens() {
        let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32 % 50).collect();
        let (message, sequential, count) =
            sort_shared::<0, false>(&keys, bitonic, isize::MAX, false);
        assert_eq!(message, None);
        assert_eq!(count, lang_network(n).len());
        let (message, parallel, count) =
            sort_shared::<0, false>(&keys, bitonic_parallel, isize::MAX, false);
        assert_eq!(message, None);
        assert_eq!(count, lang_network(n).len());
        assert!(parallel == sequential, "n = {n}");
    }
}

#[test]
fn parallel_compares_exactly_the_pairs_of_lang_network() {
    let _guard = lock_shared();
    let mut state = 0x9A12_5EED_9A12_5EED;
    for n in parallel_lens().into_iter().take(size(3, 1)) {
        let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32).collect();
        let (message, _, _) = sort_shared::<0, true>(&keys, bitonic_parallel, isize::MAX, false);
        assert_eq!(message, None);
        // Every comparator of the network spans a power of two, so counting the pairs by
        // their lower position and that span compares the two multisets without sorting
        // them, which Miri would take long over.
        let slot = |(i, j): (usize, usize)| {
            assert!((j - i).is_power_of_two(), "{i} and {j} compared");
            i * usize::BITS as usize + (j - i).ilog2() as usize
        };
        let mut counts = vec![0i32; n * usize::BITS as usize];
        let pairs = std::mem::take(&mut *SHARED.pairs.lock().unwrap());
        for &pair in &pairs {
            counts[slot(pair)] += 1;
        }
        for &(i, j, _) in &lang_network(n) {
            counts[slot((i, j))] -= 1;
        }
        assert!(counts.iter().all(|&c| c == 0), "n = {n}");
    }
}

#[test]
fn parallel_panic_propagates_and_keeps_every_element_exactly_once() {
    // Budgets that run out in the first round, in the middle and near the end: the panic of
    // whichever thread hits it reaches the caller with its message, after every other thread
    // has finished its part, and the slice holds every element exactly once.
    let _guard = lock_shared();
    let mut state = 0x9A1C_0000_9A1C_0000;
    let n = parallel_lens()[0];
    let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32 % 1000).collect();
    let total = lang_network(n).len() as isize;
    let budgets = if cfg!(miri) {
        vec![0, total / 2]
    } else {
        vec![0, 1, 1000, total / 3, total / 2, total - 1]
    };
    for budget in budgets {
        let (message, _, _) = sort_shared::<0, false>(&keys, bitonic_parallel, budget, false);
        assert_eq!(message.as_deref(), Some("comparison budget exhausted"));
    }
    let (message, _, _) = sort_shared::<0, false>(&keys, bitonic_parallel, total, false);
    assert_eq!(message, None);
}

#[test]
fn parallel_inconsistent_ord_keeps_every_element_exactly_once() {
    // The answers depend only on the two elements compared, so the parallel sort must also
    // leave the elements exactly where the sequential one does.
    let _guard = lock_shared();
    let mut state = 0x11A2_0000_11A2_0000;
    for n in parallel_lens().into_iter().take(size(2, 1)) {
        let keys: Vec<u32> = (0..n).map(|_| xorshift(&mut state) as u32).collect();
        let (message, sequential, count) =
            sort_shared::<0, false>(&keys, bitonic, isize::MAX, true);
        assert_eq!((message, count), (None, lang_network(n).len()));
        let (message, parallel, count) =
            sort_shared::<0, false>(&keys, bitonic_parallel, isize::MAX, true);
        assert_eq!((message, count), (None, lang_network(n).len()));
        assert!(parallel == sequential, "n = {n}");
    }
}

#[test]
fn parallel_shuffled_permutations_sort() {
    let mut state = 0x5A0F_F1ED_5A0F_F1ED;
    let n = size(100_000, 2 * PER_THREAD + 7);
    let mut v: Vec<Keyed<POINTERS>> = keyed(&(0..n as u32).collect::<Vec<_>>());
    shuffle(&mut v, &mut state);
    bitonic_parallel(&mut v);
    assert!(v.iter().map(|e| e.key).eq(0..n as u32));
}

#[test]
fn sorts_narrow_elements() {
    // Elements of at most 4 bytes merge stage by stage up to merges of 2048: lengths whose
    // power-of-two merges end below, at and above that, in every width and both directions.
    fn check<T: Ord + Copy + Send + std::fmt::Debug>(v: &[T]) {
        let mut expected = v.to_vec();
        expected.sort_unstable();
        let mut a = v.to_vec();
        bitonic(&mut a);
        assert_eq!(a, expected, "n = {}", v.len());
        let mut b = v.to_vec();
        bitonic_parallel(&mut b);
        assert_eq!(b, expected, "n = {}", v.len());
    }
    let lens = if cfg!(miri) {
        vec![17, 32, 33, 100, 128]
    } else {
        vec![
            17, 31, 32, 33, 64, 100, 255, 256, 1000, 2047, 2048, 2049, 4096, 5000, 70_000,
        ]
    };
    let mut state = 0xB170_0000_0000_0001;
    for n in lens {
        for modulus in [3, 1 << 16, u64::MAX] {
            let r: Vec<u64> = (0..n).map(|_| xorshift(&mut state) % modulus).collect();
            check(&r.iter().map(|&x| x as u8).collect::<Vec<_>>());
            check(&r.iter().map(|&x| x as i8).collect::<Vec<_>>());
            check(&r.iter().map(|&x| x as u16).collect::<Vec<_>>());
            check(&r.iter().map(|&x| x as i16).collect::<Vec<_>>());
            check(&r.iter().map(|&x| x as u32).collect::<Vec<_>>());
            check(&r.iter().map(|&x| x as i32).collect::<Vec<_>>());
            check(&r.iter().map(|&x| x % 2 == 0).collect::<Vec<_>>());
            check(
                &r.iter()
                    .map(|&x| [x as u8, (x >> 8) as u8, (x >> 16) as u8])
                    .collect::<Vec<_>>(),
            );
            check(
                &r.iter()
                    .map(|&x| char::from_u32(x as u32 % 0xD800).unwrap())
                    .collect::<Vec<_>>(),
            );
        }
        check(&(0..n as u32).rev().collect::<Vec<_>>());
        check(&(0..n as u32).collect::<Vec<_>>());
    }
}

/// A 4-byte element, merged stage by stage, that counts its drops and comparisons in
/// [`TRACK`] and panics once the comparison budget is spent.
struct Narrow {
    key: u16,
    id: u16,
}

impl Drop for Narrow {
    fn drop(&mut self) {
        TRACK.with(|s| s.drops.borrow_mut()[self.id as usize] += 1);
    }
}
impl PartialEq for Narrow {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Narrow {}
impl PartialOrd for Narrow {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Narrow {
    fn cmp(&self, other: &Self) -> Ordering {
        TRACK.with(|s| {
            if let Some(left) = s.budget.get() {
                if left == 0 {
                    panic!("comparison budget exhausted");
                }
                s.budget.set(Some(left - 1));
            }
            s.comparisons.set(s.comparisons.get() + 1);
            if let Some(mut state) = s.liar.get() {
                let r = xorshift(&mut state);
                s.liar.set(Some(state));
                return [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize];
            }
            self.key.cmp(&other.key)
        })
    }
}

/// [`sort_tracked`] for [`Narrow`] elements.
fn sort_narrow(keys: &[u16], budget: Option<usize>, liar: Option<u64>) -> (bool, usize) {
    let len = keys.len();
    TRACK.with(|s| {
        *s.drops.borrow_mut() = vec![0; len];
        s.budget.set(budget);
        s.comparisons.set(0);
        s.liar.set(liar);
    });
    let mut v: Vec<Narrow> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Narrow { key, id })
        .collect();
    let completed = catch_unwind(AssertUnwindSafe(|| bitonic(&mut v))).is_ok();
    if completed && liar.is_none() {
        assert!(v.windows(2).all(|w| w[0].key <= w[1].key));
    }
    let mut ids: Vec<u16> = v.iter().map(|t| t.id).collect();
    ids.sort_unstable();
    assert!(
        ids.into_iter().eq(0..len as u16),
        "element lost or duplicated"
    );
    assert!(
        v.iter().all(|t| t.key == keys[t.id as usize]),
        "element changed"
    );
    let comparisons = TRACK.with(|s| {
        s.budget.set(None);
        s.liar.set(None);
        assert!(
            s.drops.borrow().iter().all(|&d| d == 0),
            "dropped during sort"
        );
        s.comparisons.get()
    });
    drop(v);
    TRACK.with(|s| {
        assert!(
            s.drops.borrow().iter().all(|&d| d == 1),
            "leaked or double-dropped"
        )
    });
    (completed, comparisons)
}

#[test]
fn narrow_panicking_or_inconsistent_ord_keeps_every_element_exactly_once() {
    let mut state = 0xB170_0000_0000_0002;
    for n in [32, 33, size(300, 64)] {
        for keys in shapes(n, &mut state) {
            let keys: Vec<u16> = keys.iter().map(|&k| k as u16).collect();
            let (completed, total) = sort_narrow(&keys, None, None);
            assert!(completed);
            assert_eq!(total, lang_network(n).len());
            for budget in (0..total).step_by(size(3, 61)) {
                assert!(!sort_narrow(&keys, Some(budget), None).0, "budget {budget}");
            }
            assert!(sort_narrow(&keys, Some(total), None).0);
            let liar = Some(xorshift(&mut state) | 1);
            assert_eq!(sort_narrow(&keys, None, liar), (true, total));
        }
    }
}

use a::sort::{drift, sort};
use std::cell::Cell;
use std::cmp::Ordering;
use std::panic::{self, AssertUnwindSafe};

/// Scales a randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    sort(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    sort(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    sort(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    sort(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 1, 2, 3, 1, 2];
    sort(&mut v);
    assert_eq!(v, vec![1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_strings() {
    let mut v = vec![
        "pear".to_string(),
        "apple".to_string(),
        "fig".to_string(),
        "apple".to_string(),
    ];
    sort(&mut v);
    assert_eq!(v, vec!["apple", "apple", "fig", "pear"]);
}

#[test]
fn sorts_large_elements() {
    let mut state = 0x1234_5678_9ABC_DEF0_u64;
    let mut v: Vec<[u64; 8]> = (0..size(1000, 100))
        .map(|_| [xorshift(&mut state) % 50, 1, 2, 3, 4, 5, 6, 7])
        .collect();
    let mut expected = v.clone();
    expected.sort();
    sort(&mut v);
    assert_eq!(v, expected);
}

#[test]
fn sorts_zero_sized_elements() {
    let mut v = vec![(); 10];
    sort(&mut v);
    assert_eq!(v.len(), 10);
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v = vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
    sort(&mut v[2..8]);
    assert_eq!(v, vec![9, 8, 2, 3, 4, 5, 6, 7, 1, 0]);
}

/// An element compared by `key` only, carrying the index it started at.
#[derive(Debug, Clone, Copy)]
struct ByKey {
    key: u64,
    idx: usize,
}
impl PartialEq for ByKey {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for ByKey {}
impl PartialOrd for ByKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.key.cmp(&other.key)
    }
}

/// Inputs of the shapes the default sort was chosen on: random, few distinct keys, nearly
/// sorted, sawtooth, runs of `√n` and sorted with elements appended.
fn shapes(n: usize) -> Vec<Vec<u64>> {
    let mut state = 0x5EED_0000_0000_0001_u64 ^ n as u64;
    let s = (n as f64).sqrt().max(2.0) as usize;
    let random: Vec<u64> = (0..n).map(|_| xorshift(&mut state) % 1_000_000).collect();
    let few: Vec<u64> = (0..n).map(|_| xorshift(&mut state) % 16).collect();
    let mut nearly: Vec<u64> = (0..n as u64).collect();
    for _ in 0..n / 100 + 1 {
        let i = xorshift(&mut state) as usize % n;
        let j = xorshift(&mut state) as usize % n;
        nearly.swap(i, j);
    }
    let sawtooth: Vec<u64> = (0..n).map(|i| (i % s) as u64).collect();
    let mut runs = random.clone();
    for run in runs.chunks_mut(s) {
        run.sort_unstable();
    }
    let mut appended: Vec<u64> = (0..(n - n / 100) as u64).collect();
    appended.extend((0..n / 100).map(|_| xorshift(&mut state) % n as u64));
    vec![random, few, nearly, sawtooth, runs, appended]
}

#[test]
fn matches_std_stable_sort() {
    for n in [2, 7, 16, 33, 100, 1000, size(20_000, 2000)] {
        for keys in shapes(n) {
            let mut expected: Vec<(u64, usize)> = keys.iter().copied().zip(0..).collect();
            expected.sort_by_key(|e| e.0);
            let mut actual: Vec<ByKey> = keys
                .iter()
                .enumerate()
                .map(|(idx, &key)| ByKey { key, idx })
                .collect();
            sort(&mut actual);
            let actual: Vec<(u64, usize)> = actual.into_iter().map(|e| (e.key, e.idx)).collect();
            assert_eq!(actual, expected, "n = {n}");
        }
    }
}

#[test]
fn is_drift_sort() {
    // Same result and the same comparisons as drift sort, on every shape.
    thread_local! {
        static COMPARISONS: Cell<usize> = const { Cell::new(0) };
    }
    #[derive(Clone)]
    struct Counted(u64);
    impl PartialEq for Counted {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl Eq for Counted {}
    impl PartialOrd for Counted {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Counted {
        fn cmp(&self, other: &Self) -> Ordering {
            COMPARISONS.with(|c| c.set(c.get() + 1));
            self.0.cmp(&other.0)
        }
    }
    let comparisons = |f: fn(&mut [Counted]), keys: &[u64]| {
        let mut v: Vec<Counted> = keys.iter().map(|&k| Counted(k)).collect();
        COMPARISONS.with(|c| c.set(0));
        f(&mut v);
        COMPARISONS.with(Cell::get)
    };
    for n in [16, 100, size(5000, 500)] {
        for keys in shapes(n) {
            assert_eq!(
                comparisons(sort, &keys),
                comparisons(drift, &keys),
                "n = {n}"
            );
        }
    }
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    thread_local! {
        static BUDGET: Cell<usize> = const { Cell::new(0) };
    }
    #[derive(Debug)]
    struct Fragile(Box<u32>);
    impl PartialEq for Fragile {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl Eq for Fragile {}
    impl PartialOrd for Fragile {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Fragile {
        fn cmp(&self, other: &Self) -> Ordering {
            let left = BUDGET.with(|b| b.replace(b.get().saturating_sub(1)));
            assert!(left > 0, "comparison budget exhausted");
            self.0.cmp(&other.0)
        }
    }
    let mut state = 0xFEED_FACE_CAFE_BEEF_u64;
    let keys: Vec<u32> = (0..size(500, 100))
        .map(|_| xorshift(&mut state) as u32 % 100)
        .collect();
    let mut expected = keys.clone();
    expected.sort_unstable();
    for budget in (0..2000).step_by(size(37, 401)) {
        let mut v: Vec<Fragile> = keys.iter().map(|&k| Fragile(Box::new(k))).collect();
        BUDGET.with(|b| b.set(budget));
        let result = panic::catch_unwind(AssertUnwindSafe(|| sort(&mut v)));
        BUDGET.with(|b| b.set(usize::MAX));
        let mut left: Vec<u32> = v.iter().map(|f| *f.0).collect();
        left.sort_unstable();
        assert_eq!(
            left,
            expected,
            "budget {budget}, panicked: {}",
            result.is_err()
        );
    }
}

#[test]
fn inconsistent_ord_keeps_every_element_exactly_once() {
    #[derive(Debug)]
    struct Liar(u32);
    impl PartialEq for Liar {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }
    impl Eq for Liar {}
    impl PartialOrd for Liar {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Liar {
        fn cmp(&self, _: &Self) -> Ordering {
            thread_local! {
                static STATE: Cell<u64> = const { Cell::new(0xDEAD_BEEF_0BAD_F00D) };
            }
            let r = STATE.with(|s| {
                let mut state = s.get();
                let r = xorshift(&mut state);
                s.set(state);
                r
            });
            [Ordering::Less, Ordering::Equal, Ordering::Greater][(r % 3) as usize]
        }
    }
    for n in [10, 64, 200, size(5000, 500)] {
        let mut v: Vec<Liar> = (0..n as u32).map(Liar).collect();
        sort(&mut v);
        let mut seen: Vec<u32> = v.iter().map(|l| l.0).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..n as u32).collect::<Vec<_>>(), "n = {n}");
    }
}

use a::sort::bubble;

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    bubble(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    bubble(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    bubble(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v = vec![1, 2, 3, 4, 5];
    bubble(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_reverse_sorted() {
    let mut v = vec![5, 4, 3, 2, 1];
    bubble(&mut v);
    assert_eq!(v, vec![1, 2, 3, 4, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0];
    bubble(&mut v);
    assert_eq!(v, vec![0, 1, 1, 2, 2, 3, 3]);
}

#[test]
fn sorts_strings() {
    let mut v = vec!["pear", "apple", "banana", "apple"];
    bubble(&mut v);
    assert_eq!(v, vec!["apple", "apple", "banana", "pear"]);
}

#[test]
fn is_stable_for_equal_keys() {
    #[derive(Debug, Clone, Eq, PartialEq)]
    struct Pair {
        key: i32,
        idx: usize,
    }

    use std::cmp::Ordering;
    impl Ord for Pair {
        fn cmp(&self, other: &Self) -> Ordering {
            self.key.cmp(&other.key)
        }
    }
    impl PartialOrd for Pair {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut v = vec![
        Pair { key: 2, idx: 0 },
        Pair { key: 1, idx: 1 },
        Pair { key: 2, idx: 2 },
        Pair { key: 1, idx: 3 },
        Pair { key: 2, idx: 4 },
    ];

    bubble(&mut v);

    // After sorting by key, pairs with the same key should keep their original relative order.
    let ones: Vec<usize> = v.iter().filter(|p| p.key == 1).map(|p| p.idx).collect();
    let twos: Vec<usize> = v.iter().filter(|p| p.key == 2).map(|p| p.idx).collect();
    assert_eq!(ones, vec![1, 3]);
    assert_eq!(twos, vec![0, 2, 4]);
}

/// Small deterministic xorshift generator, so the tests need no extra dependencies.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn sorts_long_reverse_sorted() {
    let mut v: Vec<u32> = (0..1000).rev().collect();
    bubble(&mut v);
    assert_eq!(v, (0..1000).collect::<Vec<_>>());
}

#[test]
fn sorts_zero_sized_elements() {
    let mut v = vec![(); 16];
    bubble(&mut v);
    assert_eq!(v.len(), 16);
}

#[test]
fn matches_std_stable_sort_on_random_input() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    for len in 0..64 {
        for _ in 0..20 {
            // Keys from a small range force plenty of duplicates; the index tags each
            // element so stability is checked, not just the order of keys.
            let input: Vec<(u8, usize)> = (0..len)
                .map(|idx| ((xorshift(&mut state) % 8) as u8, idx))
                .collect();

            #[derive(Debug, PartialEq, Eq)]
            struct ByKey((u8, usize));
            impl Ord for ByKey {
                fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.0.0.cmp(&other.0.0)
                }
            }
            impl PartialOrd for ByKey {
                fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                    Some(self.cmp(other))
                }
            }

            let mut expected = input.clone();
            expected.sort_by_key(|&(key, _)| key);

            let mut actual: Vec<ByKey> = input.into_iter().map(ByKey).collect();
            bubble(&mut actual);
            let actual: Vec<(u8, usize)> = actual.into_iter().map(|b| b.0).collect();

            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn sorts_heap_owning_elements() {
    let mut v: Vec<String> = ["delta", "alpha", "echo", "charlie", "bravo", "alpha"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    bubble(&mut v);
    assert_eq!(v, ["alpha", "alpha", "bravo", "charlie", "delta", "echo"]);
}

#[test]
fn panicking_comparison_keeps_every_element_exactly_once() {
    use std::cell::Cell;
    use std::cmp::Ordering;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::rc::Rc;

    struct Tracked<'a> {
        key: u32,
        drops: Rc<Vec<Cell<u32>>>,
        comparisons_left: &'a Cell<u32>,
    }
    impl Drop for Tracked<'_> {
        fn drop(&mut self) {
            let count = &self.drops[self.key as usize];
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

    const LEN: u32 = 32;
    // Reverse order keeps elements travelling, so the panic lands in the middle of a shift.
    for budget in 0..LEN * LEN {
        let drops = Rc::new((0..LEN).map(|_| Cell::new(0)).collect::<Vec<_>>());
        let comparisons_left = Cell::new(budget);
        let mut v: Vec<Tracked> = (0..LEN)
            .rev()
            .map(|key| Tracked {
                key,
                drops: Rc::clone(&drops),
                comparisons_left: &comparisons_left,
            })
            .collect();

        let result = catch_unwind(AssertUnwindSafe(|| bubble(&mut v)));

        let mut keys: Vec<u32> = v.iter().map(|t| t.key).collect();
        if result.is_ok() {
            assert!(keys.windows(2).all(|w| w[0] <= w[1]));
        }
        keys.sort_unstable();
        assert_eq!(
            keys,
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

// ---------------------------------------------------------------------------------------------
// Elements of at most 8 bytes without drop glue take the branchless passes. `Small` is such an
// element: it is compared by `key` only, carries the index it started at, and keeps its
// bookkeeping (a comparison count, a comparison budget, a source of random answers) in
// thread-local state to stay that small.

#[derive(Debug, Clone, Copy)]
struct Small {
    key: u32,
    idx: u32,
}

#[derive(Default)]
struct SmallState {
    comparisons: std::cell::Cell<usize>,
    budget: std::cell::Cell<Option<usize>>,
    liar: std::cell::Cell<Option<u64>>,
}

thread_local! {
    static SMALL: SmallState = SmallState::default();
}

impl PartialEq for Small {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Small {}
impl PartialOrd for Small {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Small {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        SMALL.with(|s| {
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
                return [
                    std::cmp::Ordering::Less,
                    std::cmp::Ordering::Equal,
                    std::cmp::Ordering::Greater,
                ][(r % 3) as usize];
            }
            self.key.cmp(&other.key)
        })
    }
}

/// What happened when [`Small`] elements were sorted.
struct SmallRun {
    comparisons: usize,
    completed: bool,
    result: Vec<(u32, u32)>,
}

/// Sorts `keys` as [`Small`] elements, with comparisons panicking after `budget` of them and
/// answered at random from `liar` if those are set, and checks that every element is still
/// there exactly once.
fn sort_small(keys: &[u32], budget: Option<usize>, liar: Option<u64>) -> SmallRun {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut v: Vec<Small> = keys
        .iter()
        .enumerate()
        .map(|(idx, &key)| Small {
            key,
            idx: idx as u32,
        })
        .collect();
    SMALL.with(|s| {
        s.comparisons.set(0);
        s.budget.set(budget);
        s.liar.set(liar);
    });
    let completed = catch_unwind(AssertUnwindSafe(|| bubble(&mut v))).is_ok();
    let comparisons = SMALL.with(|s| {
        s.budget.set(None);
        s.liar.set(None);
        s.comparisons.get()
    });
    let result: Vec<(u32, u32)> = v.iter().map(|e| (e.key, e.idx)).collect();
    let mut idx: Vec<u32> = result.iter().map(|e| e.1).collect();
    idx.sort_unstable();
    assert!(
        idx.into_iter().eq(0..keys.len() as u32),
        "element lost or duplicated: {keys:?}"
    );
    assert!(
        result.iter().all(|&(key, idx)| keys[idx as usize] == key),
        "element changed: {keys:?}"
    );
    SmallRun {
        comparisons,
        completed,
        result,
    }
}

/// Inputs that send the branchless passes down every path: short and long, random with and
/// without duplicates (branchy passes that give up, pairs of passes, pairs that hand back to
/// branchy passes), sorted, reverse sorted, with elements a few slots from their places,
/// sorted with a few far swaps, and with turtles and rabbits.
fn for_each_small_input(seed: u64, f: &mut impl FnMut(&[u32])) {
    let mut state = seed;
    let lens: &[usize] = if cfg!(miri) {
        &[0, 1, 2, 3, 5, 17, 40, 70]
    } else {
        &[
            0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 33, 40, 63, 64, 65, 100, 129, 257, 600,
        ]
    };
    for &n in lens {
        let n32 = n as u32;
        for modulus in [2, 5, 64, u64::MAX] {
            let v: Vec<u32> = (0..n)
                .map(|_| (xorshift(&mut state) % modulus) as u32)
                .collect();
            f(&v);
        }
        f(&(0..n32).collect::<Vec<_>>());
        f(&(0..n32).rev().collect::<Vec<_>>());
        f(&(0..n32)
            .map(|i| i * 8 + (xorshift(&mut state) % 32) as u32)
            .collect::<Vec<_>>());
        let mut v: Vec<u32> = (0..n32).collect();
        for _ in 0..n / 50 + 1 {
            if n > 1 {
                let i = (xorshift(&mut state) % n as u64) as usize;
                let j = (xorshift(&mut state) % n as u64) as usize;
                v.swap(i, j);
            }
        }
        f(&v);
        f(&(1..n32).chain((n > 0).then_some(0)).collect::<Vec<_>>());
        f(&(n > 0)
            .then_some(n32)
            .into_iter()
            .chain(0..n32.saturating_sub(1))
            .collect::<Vec<_>>());
    }
}

#[test]
fn small_test_elements_take_the_branchless_passes() {
    assert!(std::mem::size_of::<Small>() <= 8);
    assert!(!std::mem::needs_drop::<Small>());
}

#[test]
fn small_elements_sort_like_std_stable_sort() {
    for_each_small_input(0x5A11_E1E5_0000_0001, &mut |keys| {
        let run = sort_small(keys, None, None);
        assert!(run.completed);
        let mut expected: Vec<(u32, u32)> = keys.iter().copied().zip(0..).collect();
        expected.sort_by_key(|e| e.0);
        assert_eq!(run.result, expected, "input: {keys:?}");
    });
}

#[test]
fn small_elements_make_exactly_the_classic_comparisons() {
    for_each_small_input(0x5A11_E1E5_0000_0002, &mut |keys| {
        let run = sort_small(keys, None, None);
        assert_eq!(
            run.comparisons,
            classic_comparisons(keys),
            "input: {keys:?}"
        );
    });
}

#[test]
fn small_panicking_comparison_keeps_every_element_exactly_once() {
    let mut state = 0x5A11_E1E5_0000_0003;
    let n = if cfg!(miri) { 40 } else { 90 };
    let random: Vec<u32> = (0..n)
        .map(|_| (xorshift(&mut state) % 1000) as u32)
        .collect();
    let local: Vec<u32> = (0..n as u32)
        .map(|i| i * 8 + (xorshift(&mut state) % 32) as u32)
        .collect();
    let reverse: Vec<u32> = (0..n as u32).rev().collect();
    for keys in [random, local, reverse] {
        let total = sort_small(&keys, None, None).comparisons;
        // Under Miri, every 7th budget only.
        let stride = if cfg!(miri) { 7 } else { 1 };
        for budget in (0..=total).step_by(stride) {
            let run = sort_small(&keys, Some(budget), None);
            assert_eq!(run.completed, budget == total, "budget {budget}: {keys:?}");
        }
    }
}

#[test]
fn small_inconsistent_ord_keeps_every_element_exactly_once() {
    // An `Ord` that answers at random makes the order of the result unspecified, but no
    // element may be lost or duplicated, no index may go out of bounds, and the sort must
    // still terminate.
    let mut seed = 0x5A11_E1E5_0000_0004;
    for_each_small_input(0x5A11_E1E5_0000_0005, &mut |keys| {
        let run = sort_small(keys, None, Some(xorshift(&mut seed) | 1));
        assert!(run.completed);
    });
}

/// The number of comparisons classic swap-based bubble sort with early exit and last-swap
/// bound makes on `keys`.
fn classic_comparisons(keys: &[u32]) -> usize {
    let mut v = keys.to_vec();
    let mut comparisons = 0;
    let mut n = v.len();
    while n > 1 {
        let mut last = 0;
        for j in 1..n {
            comparisons += 1;
            if v[j] < v[j - 1] {
                v.swap(j - 1, j);
                last = j;
            }
        }
        n = last;
    }
    comparisons
}

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

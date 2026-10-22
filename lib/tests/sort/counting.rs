use a::sort::{counting, counting_by_key};
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;

/// Scales an exhaustive or randomised test down when it runs under Miri.
fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// Small deterministic xorshift generator, so the tests need no extra dependencies.
fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Sorts `v` with [`counting`] and checks the result against [`slice::sort_unstable`].
fn assert_counts<T: a::sort::IntegerKey + std::fmt::Debug>(v: &[T]) {
    let mut expected = v.to_vec();
    expected.sort_unstable();
    let mut actual = v.to_vec();
    counting(&mut actual);
    assert_eq!(actual, expected, "input {v:?}");
}

/// Sorts `keys`, each paired with its index, with [`counting_by_key`] and checks the result
/// against the standard library's stable sort.
fn assert_counts_by_key<K: a::sort::IntegerKey + std::fmt::Debug>(keys: &[K]) {
    let mut expected: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|&(k, _)| k);
    let mut actual: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    counting_by_key(&mut actual, |&(k, _)| k);
    assert_eq!(actual, expected, "keys {keys:?}");
}

// ---------------------------------------------------------------------------------------------
// `counting`
// ---------------------------------------------------------------------------------------------

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    counting(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    counting(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    counting(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    counting(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    counting(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    counting(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_non_increasing() {
    // Reversed as a whole: equal values cannot be told apart.
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    counting(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    counting(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        counting(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    counting(&mut v[5..25]);
    let mut middle = before[5..25].to_vec();
    middle.sort_unstable();
    assert_eq!(&v[..5], &before[..5]);
    assert_eq!(&v[5..25], &middle[..]);
    assert_eq!(&v[25..], &before[25..]);
}

#[test]
fn sorts_every_sequence_over_few_values() {
    // Every sequence of up to 7 values out of 4, so every count, run and order is seen.
    let max_len = size(7, 4);
    for len in 0..=max_len {
        let total = 4usize.pow(len as u32);
        for code in 0..total {
            let v: Vec<u8> = (0..len)
                .map(|i| (code / 4usize.pow(i as u32) % 4) as u8)
                .collect();
            assert_counts(&v);
        }
    }
}

/// Checks random slices of every length up to 40 and some longer ones, over `span` values
/// starting at each of `bases`.
macro_rules! check_integers {
    ($t:ty, $state:expr, $span:expr, $bases:expr) => {{
        let lens = (0..40).chain([100, 257, size(3000, 300)]);
        for len in lens {
            for base in $bases {
                let v: Vec<$t> = (0..len)
                    .map(|_| <$t>::wrapping_add(base, (xorshift($state) % $span) as $t))
                    .collect();
                assert_counts(&v);
            }
        }
    }};
}

#[test]
fn sorts_every_integer_type_at_its_extremes() {
    let mut state = 0x1234_5678_9ABC_DEF0;
    let s = &mut state;
    check_integers!(u8, s, 256, [0u8]);
    check_integers!(i8, s, 256, [i8::MIN]);
    check_integers!(u16, s, 1000, [0, u16::MAX - 999]);
    check_integers!(i16, s, 1000, [i16::MIN, -500, i16::MAX - 999]);
    check_integers!(u32, s, 1000, [0, u32::MAX - 999]);
    check_integers!(i32, s, 1000, [i32::MIN, -500, i32::MAX - 999]);
    check_integers!(u64, s, 1000, [0, u64::MAX - 999]);
    check_integers!(i64, s, 1000, [i64::MIN, -500, i64::MAX - 999]);
    check_integers!(u128, s, 1000, [0, u128::MAX - 999]);
    check_integers!(i128, s, 1000, [i128::MIN, -500, i128::MAX - 999]);
    check_integers!(usize, s, 1000, [0, usize::MAX - 999]);
    check_integers!(isize, s, 1000, [isize::MIN, -500, isize::MAX - 999]);
}

#[test]
fn sorts_whole_ranges_of_small_types() {
    let all_u8: Vec<u8> = (0..=255).rev().chain(0..=255).collect();
    assert_counts(&all_u8);
    let all_i8: Vec<i8> = (i8::MIN..=i8::MAX)
        .rev()
        .chain([0, i8::MIN, i8::MAX])
        .collect();
    assert_counts(&all_i8);
    // 65,536 values, allowed for any number of elements.
    assert_counts(&[u16::MAX, 0, 7]);
    assert_counts(&[i16::MAX, i16::MIN, 0]);
}

#[test]
fn sorts_chars_across_the_surrogates() {
    // The range from U+D7FF to U+E000 contains the surrogates, which are not `char`s: the
    // rewrite must never make one.
    let mut state = 0xC4A2_C4A2_C4A2_C4A2;
    let pool = [
        '\u{D7FE}', '\u{D7FF}', '\u{E000}', '\u{E001}', 'a', '\u{FFFF}',
    ];
    for len in 0..size(300, 60) {
        let v: Vec<char> = (0..len)
            .map(|_| pool[(xorshift(&mut state) % pool.len() as u64) as usize])
            .collect();
        assert_counts(&v);
    }
    assert_counts(&['\u{10FFFF}', '\u{10FFFE}', '\u{10FFFF}', '\u{10FF00}']);
    assert_counts(&['\0', '\u{1}', '\0', '\u{FFFF}', '\u{E000}']);
}

#[test]
fn sorts_bools() {
    for len in 0..size(300, 40) {
        let v: Vec<bool> = (0..len).map(|i| (i * 7) % 3 == 0).collect();
        assert_counts(&v);
    }
    assert_counts(&[true, false]);
    assert_counts(&[true, true, true]);
}

#[test]
fn counts_into_eight_tables() {
    // Eight tables are used with at most 2048 values and at least 16 elements per value; the
    // lengths cover every remainder of 8, and the values every table.
    let mut state = 0x7AB1_E5E7_7AB1_E5E7;
    for span in [1, 2, 3, 8, 100, 2048] {
        let base = 16 * span;
        for len in base..base + size(9, 3) {
            let v: Vec<u32> = (0..len as u32)
                .map(|_| 1000 + (xorshift(&mut state) % span as u64) as u32)
                .collect();
            assert_counts(&v);
            // Equal values in runs: every table counts long runs of the same value.
            let mut runs = v.clone();
            runs.sort_unstable();
            runs.rotate_left(len / 3);
            assert_counts(&runs);
        }
    }
}

#[test]
fn rewrites_sparse_ranges() {
    // At least 32 values per element take the loop that skips empty blocks of counts; spans
    // that are not a multiple of 8 leave a partial block.
    let mut state = 0x5BA2_5E5B_A25E_5BA2;
    for len in [2, 3, 7, 8, 9, 100, size(2000, 200)] {
        for span in [65_536u64, 32 * len as u64, 32 * len as u64 + 5] {
            let v: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % span) as u32)
                .collect();
            assert_counts(&v);
        }
        // The smallest and the largest value at the ends of the range.
        let mut v: Vec<u16> = (0..len).map(|i| (i * 31) as u16).collect();
        v[0] = u16::MAX;
        v[1] = 0;
        assert_counts(&v);
    }
}

#[test]
fn writes_ahead_without_losing_values() {
    // Values are written 4 (or 8, for 1- and 2-byte types) at a time ahead of their count: runs
    // of every length from 0 to 10 and slices shorter than that.
    for len in 0..=12 {
        for gap in 1..=3 {
            let v: Vec<u32> = (0..len).map(|i| (i / gap * 5) as u32).rev().collect();
            assert_counts(&v);
            let w: Vec<u8> = (0..len).map(|i| (i / gap * 5) as u8).rev().collect();
            assert_counts(&w);
        }
    }
    for run in 0..=10 {
        let mut v: Vec<u64> = Vec::new();
        for value in 0..12u64 {
            let count = if value % 2 == 0 {
                run
            } else {
                (value as usize) % 3
            };
            v.extend(std::iter::repeat_n(value * 3, count));
        }
        v.reverse();
        v.push(1);
        assert_counts(&v);
    }
}

#[test]
fn sorts_long_inputs() {
    let n = size(200_000, 3_000);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let shapes: Vec<Vec<u32>> = vec![
        (0..n)
            .map(|_| (xorshift(&mut state) % n as u64) as u32)
            .collect(),
        (0..n)
            .map(|_| (xorshift(&mut state) % (4 * n as u64)) as u32)
            .collect(),
        (0..n as u32).map(|i| i % 7).collect(),
        // Organ pipe: ascending then descending.
        (0..n as u32).map(|i| i.min(n as u32 - 1 - i)).collect(),
        // Sorted with the last element moved to the front.
        std::iter::once(n as u32).chain(0..n as u32 - 1).collect(),
    ];
    for v in shapes {
        assert_counts(&v);
    }
}

#[test]
fn allows_a_range_up_to_the_limit() {
    // Up to 65,536 values for any length.
    assert_counts(&[65_535u32, 0]);
    assert_counts(&[-32_768i64, 32_767]);
    // Beyond that, up to 4 values per element.
    let n = size(20_000, 16_400);
    let mut v: Vec<u32> = (0..n as u32).collect();
    v[0] = 4 * n as u32 - 1;
    assert_counts(&v);
}

/// Sorts `v` with [`counting`] and returns the panic message, if any, after checking that a
/// panicking sort left `v` untouched.
fn panic_of<T: a::sort::IntegerKey + std::fmt::Debug>(v: &mut [T]) -> Option<String> {
    let before = v.to_vec();
    let result = catch_unwind(AssertUnwindSafe(|| counting(v)));
    let message = result.err().map(|e| *e.downcast::<String>().unwrap());
    if message.is_some() {
        assert_eq!(v, &before[..], "a panicking sort changed the slice");
    }
    message
}

#[test]
fn panics_on_a_range_over_the_limit_and_leaves_the_slice_alone() {
    let message = panic_of(&mut [65_536u32, 0]).expect("65,537 values for 2 elements");
    assert!(message.contains("65537 possible values"), "{message}");
    assert!(message.contains("at most 65536"), "{message}");
    // Sorted input has nothing to count, but the limit does not depend on the order.
    assert!(panic_of(&mut [0u32, 65_536]).is_some());
    assert!(panic_of(&mut [0u32, 1, 65_536, 1]).is_some());

    let n = size(20_000, 16_400);
    let mut v: Vec<u32> = (0..n as u32).rev().collect();
    v[1] = 4 * n as u32;
    assert!(panic_of(&mut v).is_some());

    // Distances that do not fit in `usize`.
    assert!(panic_of(&mut [u128::MAX, 0, 1]).is_some());
    assert!(panic_of(&mut [i128::MIN, i128::MAX]).is_some());
    assert!(panic_of(&mut [u64::MAX, 0]).is_some());
    assert!(panic_of(&mut ['\u{10FFFF}', '\0']).is_some());
}

/// Sorts `v` with [`counting`] and returns how many allocations that made.
fn allocations_of<T: a::sort::IntegerKey>(v: &mut [T]) -> usize {
    crate::allocations::count(|| counting(v))
}

#[test]
fn allocates_only_for_counts_beyond_4_kib() {
    // 2048 `u16` counters fill 4 KiB, up to 65,535 elements: a range of up to 2048 values, or
    // of up to 256 when eight tables are used, is counted on the stack. Longer slices count in
    // `u32`, 1024 of which fill 4 KiB.
    let mut state = 0x0A11_0C47_E5B0_FFE5;
    for (len, span, allocations) in [
        (100, 2048, 0),
        (100, 2049, 1),
        (16 * 256, 256, 0),
        (16 * 257, 257, 1),
        (size(20_000, 4_000), 2000, 0),
        (size(20_000, 4_000), 3000, 1),
        (70_000, 128, 0),
        (70_000, 129, 1),
    ] {
        let mut v: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % span) as u32)
            .collect();
        // Make sure the whole span is there.
        v[0] = 0;
        v[1] = span as u32 - 1;
        assert_eq!(
            allocations_of(&mut v),
            allocations,
            "len = {len}, span = {span}"
        );
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    // Sorted and reverse-sorted input is only scanned.
    let mut v: Vec<u32> = (0..10_000).map(|i| i * 6).collect();
    assert_eq!(allocations_of(&mut v), 0);
    v.reverse();
    assert_eq!(allocations_of(&mut v), 0);
}

// ---------------------------------------------------------------------------------------------
// `counting_by_key`
// ---------------------------------------------------------------------------------------------

#[test]
fn by_key_sorts_short_slices() {
    assert_counts_by_key::<u32>(&[]);
    assert_counts_by_key(&[1u32]);
    assert_counts_by_key(&[2u32, 1]);
    assert_counts_by_key(&[1u32, 2]);
    assert_counts_by_key(&[1u32, 1]);
}

#[test]
fn by_key_is_stable() {
    let mut state = 0x57AB_1E57_AB1E_57AB;
    for len in 0..size(200, 50) {
        for span in [1, 2, 5, 50] {
            let keys: Vec<u16> = (0..len)
                .map(|_| (xorshift(&mut state) % span) as u16)
                .collect();
            assert_counts_by_key(&keys);
        }
    }
}

#[test]
fn by_key_leaves_sorted_input_and_reverses_only_strictly_descending_input() {
    assert_counts_by_key(&[0u32, 0, 1, 1, 2, 2]);
    assert_counts_by_key(&[5u32, 4, 3, 2, 1, 0]);
    // Not strictly descending: reversing it would swap the equal keys.
    assert_counts_by_key(&[5u32, 4, 4, 3, 2, 1, 0]);
    assert_counts_by_key(&[2u8, 2, 1]);
    let long: Vec<u8> = (0..=255).rev().collect();
    assert_counts_by_key(&long);
    let long: Vec<u8> = (0..size(10_000, 5_000))
        .map(|i| 255 - (i / 40) as u8)
        .collect();
    assert_counts_by_key(&long);
}

#[test]
fn by_key_sorts_every_sequence_over_few_keys() {
    let max_len = size(7, 4);
    for len in 0..=max_len {
        let total = 3usize.pow(len as u32);
        for code in 0..total {
            let keys: Vec<i8> = (0..len)
                .map(|i| (code / 3usize.pow(i as u32) % 3) as i8 - 1)
                .collect();
            assert_counts_by_key(&keys);
        }
    }
}

#[test]
fn by_key_sorts_every_key_type() {
    let mut state = 0xB1CE_B1CE_B1CE_B1CE;
    for len in [0, 1, 2, 9, 100, size(5000, 500)] {
        let mut r = || xorshift(&mut state);
        let u8s: Vec<u8> = (0..len).map(|_| r() as u8).collect();
        assert_counts_by_key(&u8s);
        let i8s: Vec<i8> = (0..len).map(|_| r() as i8).collect();
        assert_counts_by_key(&i8s);
        let bools: Vec<bool> = (0..len).map(|_| r() % 3 == 0).collect();
        assert_counts_by_key(&bools);
        let i32s: Vec<i32> = (0..len).map(|_| (r() % 300) as i32 - 150).collect();
        assert_counts_by_key(&i32s);
        let u64s: Vec<u64> = (0..len).map(|_| u64::MAX - r() % 1000).collect();
        assert_counts_by_key(&u64s);
        let i128s: Vec<i128> = (0..len).map(|_| i128::MIN + (r() % 1000) as i128).collect();
        assert_counts_by_key(&i128s);
        let chars: Vec<char> = (0..len)
            .map(|_| char::from_u32(0xD7F0 + (r() % 0x20) as u32).unwrap_or('\u{E000}'))
            .collect();
        assert_counts_by_key(&chars);
    }
}

#[test]
fn by_key_sorts_small_key_types() {
    let mut state = 0x5A11_5A11_5A11_5A11;
    for len in [4095, 4096, 4097, size(20_000, 4_200)] {
        let u8s: Vec<u8> = (0..len).map(|_| xorshift(&mut state) as u8).collect();
        assert_counts_by_key(&u8s);
        let few: Vec<i8> = (0..len)
            .map(|_| (xorshift(&mut state) % 3) as i8 - 1)
            .collect();
        assert_counts_by_key(&few);
        let bools: Vec<bool> = (0..len)
            .map(|_| xorshift(&mut state).is_multiple_of(2))
            .collect();
        assert_counts_by_key(&bools);
        let mut sorted = u8s.clone();
        sorted.sort_unstable();
        assert_counts_by_key(&sorted);
        sorted.reverse();
        assert_counts_by_key(&sorted);
    }
}

#[test]
fn by_key_places_eight_segments() {
    // Eight segments are used with at most 256 keys and at least 16 elements per key; the
    // lengths cover every remainder of 8, which the last segment takes.
    let mut state = 0x5E65_5E65_5E65_5E65;
    for span in [2u64, 3, 100, 256, 257] {
        let base = 16 * span as usize;
        for len in base..base + size(9, 3) {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % span) as u32)
                .collect();
            assert_counts_by_key(&keys);
        }
    }
}

#[test]
fn by_key_places_runs_of_equal_keys_in_one_segment() {
    // One segment is counted as four quarters side by side and placed four keys at a time:
    // runs of equal keys, short and long, at every length modulo four, in `u16` and `u32`
    // counts.
    let mut state = 0x9E57_9E57_9E57_9E57;
    let lens: Vec<usize> = if cfg!(miri) {
        (1..12).chain([257, 1000]).collect()
    } else {
        (1..12).chain([257, 1000, 65_535, 65_536, 70_001]).collect()
    };
    for len in lens {
        for run in [1, 3, 16, 300] {
            // Runs over more than 256 keys, which take one segment.
            let mut keys: Vec<u32> = (0..len).map(|i| (i / run) as u32 * 7 % 5000).collect();
            for _ in 0..(len / 50).max(1) {
                let i = (xorshift(&mut state) % len as u64) as usize;
                keys[i] = (xorshift(&mut state) % 5000) as u32;
            }
            assert_counts_by_key(&keys);
            assert_counts(&keys);
        }
    }
}

#[test]
fn by_key_sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        counting_by_key(&mut v, |_| 0u8);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn by_key_sorts_owning_elements() {
    let mut state = 0x1EAF_5EED_1EAF_5EED;
    let mut v: Vec<String> = (0..size(500, 100))
        .map(|_| format!("{:03}", xorshift(&mut state) % 1000))
        .collect();
    let mut expected = v.clone();
    expected.sort_by_key(|s| s.len() * 1000 + s.parse::<usize>().unwrap() % 7);
    counting_by_key(&mut v, |s| s.len() * 1000 + s.parse::<usize>().unwrap() % 7);
    assert_eq!(v, expected);
}

#[test]
fn by_key_sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [10, 200, size(3000, 300)] {
        let mut v: Vec<[u64; 32]> = (0..len)
            .map(|i| {
                let mut e = [i as u64; 32];
                e[0] = xorshift(&mut state) % 50;
                e
            })
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|e| e[0]);
        counting_by_key(&mut v, |e| e[0]);
        assert_eq!(v, expected);
    }
}

#[test]
fn by_key_panics_on_a_range_over_the_limit_and_leaves_the_slice_alone() {
    for keys in [[65_536u32, 0, 1], [65_536, 0, 0], [0, 1, 65_536]] {
        let mut v: Vec<(u32, usize)> = keys.iter().copied().zip(0..).collect();
        let before = v.clone();
        let result = catch_unwind(AssertUnwindSafe(|| counting_by_key(&mut v, |e| e.0)));
        assert!(result.is_err(), "{keys:?}");
        assert_eq!(v, before);
    }
}

#[test]
fn by_key_calls_key_once_per_element_and_places_its_latest_state() {
    // A key that counts its calls in the element, through interior mutability: every element
    // is asked once, and the element placed is the one after that call, not a copy taken
    // before it.
    for len in [2, 9, 100, size(3000, 300)] {
        let v: Vec<(u32, Cell<u32>)> = (0..len as u32)
            .map(|i| ((i * 7) % 13, Cell::new(0)))
            .collect();
        let mut v = v;
        counting_by_key(&mut v, |e| {
            e.1.set(e.1.get() + 1);
            e.0
        });
        assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
        assert!(v.iter().all(|e| e.1.get() == 1), "not one call per element");
    }
    for keys in [vec![1u32, 2, 2, 3], vec![3, 2, 1], vec![2, 2, 1]] {
        let mut v: Vec<(u32, Cell<u32>)> = keys.into_iter().map(|k| (k, Cell::new(0))).collect();
        counting_by_key(&mut v, |e| {
            e.1.set(e.1.get() + 1);
            e.0
        });
        assert!(v.iter().all(|e| e.1.get() == 1), "not one call per element");
    }
}

/// An element that counts how many times it is dropped.
struct Tracked {
    key: u32,
    id: usize,
    drops: Rc<Vec<Cell<u32>>>,
}

impl Drop for Tracked {
    fn drop(&mut self) {
        let count = &self.drops[self.id];
        count.set(count.get() + 1);
    }
}

/// Builds tracked elements with `keys`, sorts them with `sort` and checks that every element
/// is in the slice exactly once afterwards and is dropped exactly once with it. Returns the
/// ids in their final order and whether the sort completed.
fn sort_tracked(keys: &[u32], sort: impl FnOnce(&mut [Tracked])) -> (Vec<usize>, bool) {
    let len = keys.len();
    let drops = Rc::new((0..len).map(|_| Cell::new(0)).collect::<Vec<_>>());
    let mut v: Vec<Tracked> = keys
        .iter()
        .enumerate()
        .map(|(id, &key)| Tracked {
            key,
            id,
            drops: Rc::clone(&drops),
        })
        .collect();
    let completed = catch_unwind(AssertUnwindSafe(|| sort(&mut v))).is_ok();
    let order: Vec<usize> = v.iter().map(|t| t.id).collect();
    let mut ids = order.clone();
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0..len).collect::<Vec<_>>(),
        "element lost or duplicated"
    );
    assert!(v.iter().all(|t| t.key == keys[t.id]), "element changed");
    assert!(
        drops.iter().all(|d| d.get() == 0),
        "element dropped during sort"
    );
    drop(v);
    assert!(
        drops.iter().all(|d| d.get() == 1),
        "element leaked or double-dropped"
    );
    (order, completed)
}

#[test]
fn by_key_panicking_key_leaves_the_slice_as_it_was() {
    let len = size(150, 40);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    let layouts: Vec<Vec<u32>> = vec![
        random,
        (0..len as u32).map(|i| i % 3).collect(),
        (0..len as u32).collect(),
        (0..len as u32).rev().collect(),
    ];
    for keys in layouts {
        let calls = Cell::new(0);
        sort_tracked(&keys, |v| {
            counting_by_key(v, |t| {
                calls.set(calls.get() + 1);
                t.key
            })
        });
        let total = calls.get();
        let step = if cfg!(miri) { 7 } else { 1 };
        for budget in (0..total).step_by(step) {
            let calls = Cell::new(0);
            let (order, completed) = sort_tracked(&keys, |v| {
                counting_by_key(v, |t| {
                    if calls.get() == budget {
                        panic!("key budget exhausted");
                    }
                    calls.set(calls.get() + 1);
                    t.key
                })
            });
            assert!(!completed, "budget {budget} of {total}");
            assert_eq!(order, (0..keys.len()).collect::<Vec<_>>(), "slice changed");
        }
    }
}

#[test]
fn by_key_sorts_owning_elements_exactly_once() {
    let mut state = 0x0DD5_0DD5_0DD5_0DD5;
    for len in [2, 3, 9, 40, 300, size(5000, 400)] {
        for span in [2u32, 7, 300] {
            let keys: Vec<u32> = (0..len)
                .map(|_| xorshift(&mut state) as u32 % span)
                .collect();
            let (order, completed) = sort_tracked(&keys, |v| counting_by_key(v, |t| t.key));
            assert!(completed);
            let mut expected: Vec<usize> = (0..len).collect();
            expected.sort_by_key(|&id| keys[id]);
            assert_eq!(order, expected);
        }
    }
}

/// Sorts `v` with [`counting_by_key`] by `e[0]` and returns how many allocations that made.
fn allocations_by_key_of<const N: usize>(v: &mut [[u64; N]]) -> usize {
    crate::allocations::count(|| counting_by_key(v, |e| e[0]))
}

#[test]
fn by_key_allocates_only_beyond_4_kib() {
    // 512 `u64` elements fill the 4 KiB stack buffer, 512 `u64` keys another 4 KiB, and 2048
    // `u16` counters a third: a range of up to 2048 keys, or of up to 256 with eight segments,
    // needs no allocation.
    let mut state = 0xA110_CA7E_A110_CA7E;
    for (len, span, allocations) in [
        (100, 2048, 0),
        (100, 2049, 1),
        (512, 32, 0),
        (513, 32, 2),
        (513, 2100, 3),
    ] {
        let mut v: Vec<[u64; 1]> = (0..len).map(|_| [xorshift(&mut state) % span]).collect();
        v[0] = [0];
        v[1] = [span - 1];
        assert_eq!(
            allocations_by_key_of(&mut v),
            allocations,
            "len = {len}, span = {span}"
        );
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    // Sorted input is only scanned, but its keys are kept as they are read.
    let mut v: Vec<[u64; 1]> = (0..512).map(|i| [i]).collect();
    assert_eq!(allocations_by_key_of(&mut v), 0);
    let mut v: Vec<[u64; 1]> = (0..10_000).map(|i| [i]).collect();
    assert_eq!(allocations_by_key_of(&mut v), 1);
}

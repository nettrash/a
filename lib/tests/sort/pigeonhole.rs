use a::sort::{IntegerKey, pigeonhole, pigeonhole_by_key};
use std::cell::{Cell, RefCell};
use std::fmt::Debug;
use std::mem;
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

/// Shuffles `v` with the Fisher–Yates algorithm.
fn shuffle<T>(v: &mut [T], state: &mut u64) {
    for i in (1..v.len()).rev() {
        let j = (xorshift(state) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// Sorts `v` with [`pigeonhole`] and checks the result against [`slice::sort_unstable`].
fn assert_pigeonhole<T: IntegerKey + Debug>(v: &[T]) {
    let mut expected = v.to_vec();
    expected.sort_unstable();
    let mut actual = v.to_vec();
    pigeonhole(&mut actual);
    assert_eq!(actual, expected, "input {v:?}");
}

/// Sorts `v` with [`pigeonhole_by_key`] by `key` and checks, without assuming any order of equal
/// keys, that the keys end up in order, that the ids `id` of the elements still form a
/// permutation of `0..v.len()`, that every element is `intact`, and that `key` was called once
/// per element (never for fewer than two elements or zero-sized ones).
fn sort_checked<T, K: IntegerKey + Debug>(
    v: &mut [T],
    key: impl Fn(&T) -> K,
    id: impl Fn(&T) -> usize,
    intact: impl Fn(&T) -> bool,
) {
    let len = v.len();
    let mut keys_before: Vec<K> = v.iter().map(&key).collect();
    keys_before.sort_unstable();
    let calls = Cell::new(0);
    pigeonhole_by_key(v, |e| {
        calls.set(calls.get() + 1);
        key(e)
    });
    let keys_after: Vec<K> = v.iter().map(&key).collect();
    assert!(keys_after == keys_before, "keys not sorted: {keys_after:?}");
    let mut ids: Vec<usize> = v.iter().map(&id).collect();
    ids.sort_unstable();
    assert!(ids.iter().copied().eq(0..len), "element lost or duplicated");
    assert!(v.iter().all(intact), "element changed");
    let expected_calls = if len >= 2 && mem::size_of::<T>() != 0 {
        len
    } else {
        0
    };
    assert_eq!(calls.get(), expected_calls, "not one call per element");
}

/// Sorts `keys`, each paired with its index, with [`pigeonhole_by_key`], and checks the result
/// with [`sort_checked`].
fn assert_by_key<K: IntegerKey + Debug>(keys: &[K]) {
    let mut v: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    sort_checked(&mut v, |e| e.0, |e| e.1, |e| e.0 == keys[e.1]);
}

/// Inputs of `n` values in `0..=n` of shapes that the cycles of the permutation, the
/// recognition of sorted input and the counts all treat differently.
fn patterns(n: usize, state: &mut u64) -> Vec<Vec<u32>> {
    let n32 = n as u32;
    let mut permutation: Vec<u32> = (0..n32).collect();
    shuffle(&mut permutation, state);
    let mut nearly_sorted: Vec<u32> = (0..n32).collect();
    for _ in 0..(n / 50).max(1) {
        if n >= 2 {
            let i = (xorshift(state) % n as u64) as usize;
            let j = (xorshift(state) % n as u64) as usize;
            nearly_sorted.swap(i, j);
        }
    }
    vec![
        // Random over as many values as elements.
        (0..n)
            .map(|_| (xorshift(state) % (n as u64 + 1)) as u32)
            .collect(),
        // Many duplicates.
        (0..n).map(|_| (xorshift(state) % 3) as u32).collect(),
        // One element per hole.
        permutation,
        (0..n32).collect(),
        (0..n32).rev().collect(),
        // Non-increasing: reversed as a whole, equal values cannot be told apart.
        (0..n32).map(|i| (n32 - i) / 3).collect(),
        nearly_sorted,
        // Sawtooth.
        (0..n32).map(|i| i % 37).collect(),
        (0..n32).map(|i| 36 - i % 37).collect(),
        // Organ pipe: ascending then descending.
        (0..n32).map(|i| i.min(n32 - 1 - i)).collect(),
        // Sorted with the last element moved to the front, and the first to the back.
        std::iter::once(n32)
            .chain(0..n32.saturating_sub(1))
            .collect(),
        (1..n32).chain(std::iter::once(0)).collect(),
        vec![5; n],
    ]
}

/// Calls `f` for every permutation of `0..n`, by Heap's algorithm.
fn for_each_permutation(n: usize, mut f: impl FnMut(&[u32])) {
    let mut v: Vec<u32> = (0..n as u32).collect();
    let mut c = vec![0; n];
    f(&v);
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i % 2 == 0 {
                v.swap(0, i);
            } else {
                v.swap(c[i], i);
            }
            f(&v);
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
}

/// The message of a caught panic.
fn message_of(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(s) => *s,
        Err(payload) => payload
            .downcast::<&str>()
            .map(|s| s.to_string())
            .unwrap_or_default(),
    }
}

// ---------------------------------------------------------------------------------------------
// `pigeonhole`
// ---------------------------------------------------------------------------------------------

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    pigeonhole(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    pigeonhole(&mut v);
    assert_eq!(v, vec![42]);
    let mut v = vec![u128::MAX];
    pigeonhole(&mut v);
    assert_eq!(v, vec![u128::MAX]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    pigeonhole(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    pigeonhole(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![i64::MIN + 1, i64::MIN];
    pigeonhole(&mut v);
    assert_eq!(v, vec![i64::MIN, i64::MIN + 1]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    pigeonhole(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    pigeonhole(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_non_increasing() {
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    pigeonhole(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    // Non-increasing but for the last pair.
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 1];
    pigeonhole(&mut v);
    assert_eq!(v, vec![0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    pigeonhole(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        pigeonhole(&mut v);
        assert_eq!(v, vec![7; len]);
        let mut v = vec![i128::MIN; len];
        pigeonhole(&mut v);
        assert_eq!(v, vec![i128::MIN; len]);
    }
}

#[test]
fn sorts_two_distinct_values() {
    let mut state = 0x2D15_7146_2D15_7146;
    for len in 2..size(200, 40) {
        let bits: Vec<bool> = (0..len)
            .map(|_| xorshift(&mut state).is_multiple_of(3))
            .collect();
        let near: Vec<i64> = bits
            .iter()
            .map(|&b| if b { i64::MIN + 1 } else { i64::MIN })
            .collect();
        assert_pigeonhole(&near);
        // The two ends of the largest range allowed for any length: 65,536 holes, too many to
        // go through for every length under Miri.
        if !cfg!(miri) || len <= 4 {
            let far: Vec<u32> = bits.iter().map(|&b| if b { 65_535 } else { 0 }).collect();
            assert_pigeonhole(&far);
        }
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    pigeonhole(&mut v[5..25]);
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
            assert_pigeonhole(&v);
            // The same, far apart and at the bottom of a wide type.
            let w: Vec<i64> = v.iter().map(|&x| i64::MIN + 1000 * x as i64).collect();
            assert_pigeonhole(&w);
        }
    }
}

#[test]
fn sorts_every_permutation() {
    // One element per hole: every cycle structure of up to 7 elements.
    for n in 0..=size(7, 5) {
        for_each_permutation(n, |p| {
            assert_pigeonhole(p);
            let shifted: Vec<u128> = p.iter().map(|&x| u128::MAX - x as u128).collect();
            assert_pigeonhole(&shifted);
        });
    }
}

#[test]
fn sorts_random_permutations() {
    // Long cycles, one element per hole.
    let mut state = 0x9E27_9E27_9E27_9E27;
    for n in [10, 100, 1000, size(30_000, 2_000)] {
        let mut v: Vec<i32> = (0..n as i32).map(|i| i - 500).collect();
        shuffle(&mut v, &mut state);
        assert_pigeonhole(&v);
        // A single cycle through all elements.
        let mut v: Vec<u32> = (1..n as u32).chain(std::iter::once(0)).collect();
        pigeonhole(&mut v);
        assert!(v.iter().copied().eq(0..n as u32));
    }
}

#[test]
fn sorts_patterns() {
    let mut state = 0xFA77_E245_FA77_E245;
    for n in [2, 3, 7, 50, 257, size(5000, 400)] {
        for v in patterns(n, &mut state) {
            assert_pigeonhole(&v);
            let w: Vec<i16> = v.iter().map(|&x| x as i16 - 300).collect();
            assert_pigeonhole(&w);
        }
    }
}

/// Checks random slices of every length up to 40 and some longer ones, over `span` values
/// starting at each of `bases`.
macro_rules! check_integers {
    ($t:ty, $state:expr, $span:expr, $bases:expr) => {{
        let lens = (0..size(40, 12)).chain([100, 257, size(3000, 300)]);
        for len in lens {
            for base in $bases {
                let v: Vec<$t> = (0..len)
                    .map(|_| <$t>::wrapping_add(base, (xorshift($state) % $span) as $t))
                    .collect();
                assert_pigeonhole(&v);
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
    check_integers!(i64, s, 1001, [i64::MIN, -500, i64::MAX - 1000]);
    check_integers!(u128, s, 1000, [0, u128::MAX - 999]);
    check_integers!(i128, s, 1000, [i128::MIN, -500, i128::MAX - 999]);
    check_integers!(usize, s, 1000, [0, usize::MAX - 999]);
    check_integers!(isize, s, 1000, [isize::MIN, -500, isize::MAX - 999]);
    // The extremes themselves, with the rest of the range right next to them.
    assert_pigeonhole(&[i64::MIN + 1000, i64::MIN, i64::MIN + 500, i64::MIN]);
    assert_pigeonhole(&[u128::MAX, u128::MAX - 1, u128::MAX, u128::MAX - 60_000]);
    assert_pigeonhole(&[i128::MAX, i128::MAX - 65_535, i128::MAX - 3]);
    assert_pigeonhole(&[isize::MIN, isize::MIN + 2, isize::MIN + 1]);
}

#[test]
fn sorts_whole_ranges_of_small_types() {
    // Any range of a one- or two-byte type is allowed, for any length.
    let all_u8: Vec<u8> = (0..=255).rev().chain(0..=255).collect();
    assert_pigeonhole(&all_u8);
    let all_i8: Vec<i8> = (i8::MIN..=i8::MAX)
        .rev()
        .chain([0, i8::MIN, i8::MAX])
        .collect();
    assert_pigeonhole(&all_i8);
    assert_pigeonhole(&[u8::MAX, 0]);
    assert_pigeonhole(&[i8::MAX, i8::MIN]);
    assert_pigeonhole(&[u16::MAX, 0]);
    assert_pigeonhole(&[u16::MAX, 0, 7]);
    assert_pigeonhole(&[i16::MAX, i16::MIN]);
    assert_pigeonhole(&[i16::MAX, i16::MIN, 0]);
    if !cfg!(miri) {
        let mut all_u16: Vec<u16> = (0..=u16::MAX).collect();
        let mut state = 0x16B1_7516_B175_16B1;
        shuffle(&mut all_u16, &mut state);
        assert_pigeonhole(&all_u16);
    }
}

#[test]
fn sorts_chars_across_the_surrogates() {
    // The range from U+D7FF to U+E000 contains the surrogates, which are not `char`s: their
    // holes stay empty, and no surrogate may ever be made.
    let mut state = 0xC4A2_C4A2_C4A2_C4A2;
    let pool = [
        '\u{D7FE}', '\u{D7FF}', '\u{E000}', '\u{E001}', 'a', '\u{FFFF}',
    ];
    for len in 0..size(300, 60) {
        // Under Miri, mostly the four around the surrogates: the others span 65,438 holes.
        let pool = if cfg!(miri) && len > 6 {
            &pool[..4]
        } else {
            &pool[..]
        };
        let v: Vec<char> = (0..len)
            .map(|_| pool[(xorshift(&mut state) % pool.len() as u64) as usize])
            .collect();
        assert_pigeonhole(&v);
    }
    let around: Vec<char> = (0..size(3000, 200))
        .map(|_| {
            char::from_u32(0xD000 + (xorshift(&mut state) % 0x2000) as u32).unwrap_or('\u{E000}')
        })
        .collect();
    assert_pigeonhole(&around);
    assert_pigeonhole(&['\u{10FFFF}', '\u{10FFFE}', '\u{10FFFF}', '\u{10FF00}']);
    assert_pigeonhole(&['\0', '\u{1}', '\0', '\u{FFFF}', '\u{E000}']);
    assert_pigeonhole(&['\u{E000}', '\u{D7FF}']);
}

#[test]
fn sorts_bools() {
    for len in 0..size(300, 40) {
        let v: Vec<bool> = (0..len).map(|i| (i * 7) % 3 == 0).collect();
        assert_pigeonhole(&v);
    }
    assert_pigeonhole(&[true, false]);
    assert_pigeonhole(&[true, true, true]);
    assert_pigeonhole(&[false, false, true, true, false]);
}

#[test]
fn sorts_long_inputs() {
    let n = size(100_000, 2_000);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let mut shapes = patterns(n, &mut state);
    // Up to 4 values per element: exactly at the limit.
    let mut wide: Vec<u32> = (0..n)
        .map(|_| (xorshift(&mut state) % (4 * n as u64)) as u32)
        .collect();
    wide[0] = 0;
    wide[n / 2] = 4 * n as u32 - 1;
    shapes.push(wide);
    for v in shapes {
        assert_pigeonhole(&v);
    }
    let signed: Vec<i64> = (0..n)
        .map(|_| i64::MIN + (xorshift(&mut state) % 1000) as i64)
        .collect();
    assert_pigeonhole(&signed);
}

/// Sorts `v` with [`pigeonhole`] and returns the panic message, if any, after checking that a
/// panicking sort left `v` untouched.
fn panic_of<T: IntegerKey + Debug>(v: &mut [T]) -> Option<String> {
    let before = v.to_vec();
    let result = catch_unwind(AssertUnwindSafe(|| pigeonhole(v)));
    let message = result.err().map(message_of);
    match &message {
        Some(m) => {
            assert_eq!(v, &before[..], "a panicking sort changed the slice");
            assert!(m.starts_with("pigeonhole sort:"), "{m}");
        }
        None => {
            let mut expected = before;
            expected.sort_unstable();
            assert_eq!(v, &expected[..]);
        }
    }
    message
}

#[test]
fn allows_a_range_up_to_the_limit() {
    // Up to 65,536 values for any length.
    assert!(panic_of(&mut [65_535u32, 0]).is_none());
    assert!(panic_of(&mut [0u32, 65_535]).is_none());
    assert!(panic_of(&mut [65_535u32, 3, 0]).is_none());
    assert!(panic_of(&mut [-32_768i64, 32_767]).is_none());
    assert!(panic_of(&mut [i64::MIN + 65_535, i64::MIN]).is_none());
    assert!(panic_of(&mut [u128::MAX, u128::MAX - 65_535, u128::MAX]).is_none());
    assert!(panic_of(&mut ['\u{FFFF}', '\0']).is_none());
    // 16,384 elements: 4 per element is the same 65,536.
    let n = 16_384;
    let mut v: Vec<u32> = (0..n as u32).rev().collect();
    v[0] = 65_535;
    assert!(panic_of(&mut v).is_none());
    // Beyond that, up to 4 values per element.
    let n = size(20_000, 16_400);
    let mut v: Vec<u32> = (0..n as u32).collect();
    v[0] = 4 * n as u32 - 1;
    assert!(panic_of(&mut v).is_none());
    let mut v: Vec<i32> = (0..n as i32).rev().map(|i| i - 50_000).collect();
    v[1] = 4 * n as i32 - 1 - 50_000;
    assert!(panic_of(&mut v).is_none());
}

#[test]
fn panics_on_a_range_over_the_limit_and_leaves_the_slice_alone() {
    let message = panic_of(&mut [65_536u32, 0]).expect("65,537 values for 2 elements");
    assert!(message.starts_with("pigeonhole sort:"), "{message}");
    // Sorted input has nothing to move, but the limit does not depend on the order.
    assert!(panic_of(&mut [0u32, 65_536]).is_some());
    assert!(panic_of(&mut [65_536u32, 0, 0]).is_some());
    assert!(panic_of(&mut [0u32, 1, 65_536, 1]).is_some());
    assert!(panic_of(&mut [i64::MIN, i64::MIN + 65_536]).is_some());
    assert!(panic_of(&mut ['\u{10000}', '\0']).is_some());

    let n = 16_384;
    let mut v: Vec<u32> = (0..n as u32).collect();
    v[3] = 65_536;
    assert!(panic_of(&mut v).is_some());

    let n = size(20_000, 16_400);
    let mut v: Vec<u32> = (0..n as u32).rev().collect();
    v[1] = 4 * n as u32;
    assert!(panic_of(&mut v).is_some());
    let mut v: Vec<u32> = (0..n as u32).collect();
    v[n - 1] = 4 * n as u32;
    assert!(panic_of(&mut v).is_some());

    // Distances that do not fit in `usize`.
    assert!(panic_of(&mut [u128::MAX, 0, 1]).is_some());
    assert!(panic_of(&mut [i128::MIN, i128::MAX]).is_some());
    assert!(panic_of(&mut [u64::MAX, 0]).is_some());
    assert!(panic_of(&mut [usize::MAX, 0]).is_some());
    assert!(panic_of(&mut [isize::MIN, isize::MAX]).is_some());
    assert!(panic_of(&mut [i32::MIN, i32::MAX]).is_some());
    assert!(panic_of(&mut ['\u{10FFFF}', '\0']).is_some());
}

/// Sorts `v` with [`pigeonhole`] and returns how many allocations that made.
fn allocations_of<T: IntegerKey>(v: &mut [T]) -> usize {
    crate::allocations::count(|| pigeonhole(v))
}

#[test]
fn allocates_only_for_counts_beyond_4_kib() {
    let mut state = 0x0A11_0C47_E5B0_FFE5;
    // Counts that fit 4 KiB on the stack: elements are moved in place, whatever their number.
    // 1024 `u32` values in a range of up to 1024 values need no allocation.
    let mut allocating = Vec::new();
    for (len, span) in [
        (100, 2),
        (100, 512),
        (100, 1024),
        (1024, 1024),
        (1024, 2),
        (size(50_000, 5_000), 64),
    ] {
        let mut v: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % span) as u32)
            .collect();
        // Make sure the whole span is there.
        v[0] = 0;
        v[1] = span as u32 - 1;
        let allocations = allocations_of(&mut v);
        if allocations != 0 {
            allocating.push((len, span, allocations));
        }
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    assert!(
        allocating.is_empty(),
        "(len, span, allocations): {allocating:?}"
    );
    // Counts that do not.
    for (len, span) in [(100, 5000), (size(20_000, 4_000), 60_000)] {
        let mut v: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % span) as u32)
            .collect();
        v[0] = span as u32 - 1;
        v[1] = 0;
        assert!(allocations_of(&mut v) > 0, "len = {len}, span = {span}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    // Sorted and reverse-sorted input is only scanned, whatever its range.
    let mut v: Vec<u32> = (0..10_000).map(|i| i * 6).collect();
    assert_eq!(allocations_of(&mut v), 0);
    v.reverse();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.windows(2).all(|w| w[0] <= w[1]));
}

// ---------------------------------------------------------------------------------------------
// `pigeonhole_by_key`
// ---------------------------------------------------------------------------------------------

#[test]
fn by_key_sorts_short_slices() {
    assert_by_key::<u32>(&[]);
    assert_by_key(&[1u32]);
    assert_by_key(&[2u32, 1]);
    assert_by_key(&[1u32, 2]);
    assert_by_key(&[1u32, 1]);
    assert_by_key(&[i64::MIN + 1, i64::MIN]);
}

#[test]
fn by_key_sorts_random_keys_without_losing_elements() {
    let mut state = 0x57AB_1E57_AB1E_57AB;
    for len in 0..size(200, 50) {
        for span in [1, 2, 5, 50, 1000] {
            let keys: Vec<u16> = (0..len)
                .map(|_| (xorshift(&mut state) % span) as u16)
                .collect();
            assert_by_key(&keys);
        }
    }
}

#[test]
fn by_key_sorts_sorted_and_reverse_input() {
    assert_by_key(&[0u32, 0, 1, 1, 2, 2]);
    assert_by_key(&[5u32, 4, 3, 2, 1, 0]);
    assert_by_key(&[5u32, 4, 4, 3, 2, 1, 0]);
    assert_by_key(&[5u32, 4, 4, 3, 2, 1, 2]);
    assert_by_key(&[2u8, 2, 1]);
    let long: Vec<u8> = (0..=255).rev().collect();
    assert_by_key(&long);
    let long: Vec<u8> = (0..size(10_000, 2_000))
        .map(|i| 255 - (i / 40) as u8)
        .collect();
    assert_by_key(&long);
    let long: Vec<u32> = (0..size(10_000, 2_000) as u32).map(|i| i / 3).collect();
    assert_by_key(&long);
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
            assert_by_key(&keys);
        }
    }
}

#[test]
fn by_key_sorts_every_permutation() {
    for n in 0..=size(7, 5) {
        for_each_permutation(n, assert_by_key::<u32>);
    }
}

#[test]
fn by_key_sorts_patterns() {
    let mut state = 0xB4E7_7E2B_B4E7_7E2B;
    for n in [2, 3, 7, 50, 257, size(5000, 400)] {
        for keys in patterns(n, &mut state) {
            assert_by_key(&keys);
        }
    }
}

#[test]
fn by_key_sorts_every_key_type() {
    let mut state = 0xB1CE_B1CE_B1CE_B1CE;
    for len in [0, 1, 2, 9, 100, size(5000, 500)] {
        let mut r = || xorshift(&mut state);
        let u8s: Vec<u8> = (0..len).map(|_| r() as u8).collect();
        assert_by_key(&u8s);
        let i8s: Vec<i8> = (0..len).map(|_| r() as i8).collect();
        assert_by_key(&i8s);
        // Over their whole range, but for Miri, which takes long over 65,536 holes.
        let mask = if cfg!(miri) { 0x0FFF } else { 0xFFFF };
        let u16s: Vec<u16> = (0..len).map(|_| r() as u16 & mask).collect();
        assert_by_key(&u16s);
        let i16s: Vec<i16> = (0..len).map(|_| (r() as u16 & mask) as i16).collect();
        assert_by_key(&i16s);
        let bools: Vec<bool> = (0..len).map(|_| r() % 3 == 0).collect();
        assert_by_key(&bools);
        let u32s: Vec<u32> = (0..len).map(|_| u32::MAX - (r() % 1000) as u32).collect();
        assert_by_key(&u32s);
        let i32s: Vec<i32> = (0..len).map(|_| (r() % 300) as i32 - 150).collect();
        assert_by_key(&i32s);
        let u64s: Vec<u64> = (0..len).map(|_| u64::MAX - r() % 1000).collect();
        assert_by_key(&u64s);
        let i64s: Vec<i64> = (0..len).map(|_| i64::MIN + (r() % 1001) as i64).collect();
        assert_by_key(&i64s);
        let u128s: Vec<u128> = (0..len).map(|_| u128::MAX - (r() % 1000) as u128).collect();
        assert_by_key(&u128s);
        let i128s: Vec<i128> = (0..len).map(|_| i128::MIN + (r() % 1000) as i128).collect();
        assert_by_key(&i128s);
        let usizes: Vec<usize> = (0..len)
            .map(|_| usize::MAX - (r() % 1000) as usize)
            .collect();
        assert_by_key(&usizes);
        let isizes: Vec<isize> = (0..len)
            .map(|_| isize::MAX - (r() % 1000) as isize)
            .collect();
        assert_by_key(&isizes);
        let chars: Vec<char> = (0..len)
            .map(|_| char::from_u32(0xD7F0 + (r() % 0x820) as u32).unwrap_or('\u{E000}'))
            .collect();
        assert_by_key(&chars);
    }
}

#[test]
fn by_key_sorts_small_key_types_over_their_whole_range() {
    let mut state = 0x5A11_5A11_5A11_5A11;
    // 4096 `u8` keys fill the 4 KiB key buffer.
    let lens: &[usize] = if cfg!(miri) {
        &[2, 3, 4097]
    } else {
        &[2, 3, 4095, 4096, 4097, 20_000]
    };
    for &len in lens {
        // 65,536 holes take long under Miri: there, only for the shortest slices.
        if !cfg!(miri) || len <= 3 {
            let mut u16s: Vec<u16> = (0..len).map(|_| xorshift(&mut state) as u16).collect();
            u16s[0] = u16::MAX;
            u16s[1] = 0;
            assert_by_key(&u16s);
            let mut i16s: Vec<i16> = (0..len).map(|_| xorshift(&mut state) as i16).collect();
            i16s[0] = i16::MIN;
            i16s[len - 1] = i16::MAX;
            assert_by_key(&i16s);
        }
        let mut u8s: Vec<u8> = (0..len).map(|_| xorshift(&mut state) as u8).collect();
        u8s[0] = 255;
        u8s[1] = 0;
        assert_by_key(&u8s);
        let bools: Vec<bool> = (0..len)
            .map(|_| xorshift(&mut state).is_multiple_of(2))
            .collect();
        assert_by_key(&bools);
    }
}

#[test]
fn by_key_sorts_long_inputs() {
    let n = size(100_000, 1_000);
    let mut state = 0x10C6_10C6_10C6_10C6;
    for keys in patterns(n, &mut state) {
        assert_by_key(&keys);
    }
    let mut wide: Vec<u32> = (0..n)
        .map(|_| (xorshift(&mut state) % (4 * n as u64)) as u32)
        .collect();
    wide[0] = 0;
    wide[1] = 4 * n as u32 - 1;
    assert_by_key(&wide);
}

#[test]
fn by_key_sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        let calls = Cell::new(0);
        pigeonhole_by_key(&mut v, |_| {
            calls.set(calls.get() + 1);
            0u8
        });
        assert_eq!(v.len(), len);
        assert_eq!(calls.get(), 0, "key called for zero-sized elements");
    }
}

thread_local! {
    /// How many times a [`Zst`] has been dropped on this thread.
    static ZST_DROPS: Cell<usize> = const { Cell::new(0) };
}

/// A zero-sized element that counts its drops in [`ZST_DROPS`].
struct Zst;

impl Drop for Zst {
    fn drop(&mut self) {
        ZST_DROPS.with(|d| d.set(d.get() + 1));
    }
}

#[test]
fn by_key_neither_drops_nor_duplicates_zero_sized_elements() {
    for len in [0, 1, 2, 3, 40, 1000] {
        ZST_DROPS.with(|d| d.set(0));
        let mut v: Vec<Zst> = (0..len).map(|_| Zst).collect();
        pigeonhole_by_key(&mut v, |_| 3u32);
        assert_eq!(ZST_DROPS.with(Cell::get), 0, "element dropped during sort");
        assert_eq!(v.len(), len);
        drop(v);
        assert_eq!(
            ZST_DROPS.with(Cell::get),
            len,
            "element leaked or double-dropped"
        );
    }
}

#[test]
fn by_key_calls_key_once_per_element_and_places_its_latest_state() {
    // A key that counts its calls in the element, through interior mutability: every element
    // is asked once, and the element placed is the one after that call, not a copy taken
    // before it.
    let mut state = 0xCA11_CA11_CA11_CA11;
    for len in [2, 3, 9, 100, size(3000, 300)] {
        let random: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % 13) as u32)
            .collect();
        let layouts = [
            random,
            (0..len as u32).collect(),
            (0..len as u32).rev().collect(),
            (0..len as u32).map(|i| (len as u32 - i) / 2).collect(),
            vec![4; len],
        ];
        for keys in layouts {
            let mut v: Vec<(u32, Cell<u32>)> = keys.iter().map(|&k| (k, Cell::new(0))).collect();
            pigeonhole_by_key(&mut v, |e| {
                e.1.set(e.1.get() + 1);
                e.0
            });
            assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
            assert!(v.iter().all(|e| e.1.get() == 1), "not one call per element");
        }
    }
    // Fewer than two elements: no call at all.
    for len in [0, 1] {
        let mut v: Vec<(u32, Cell<u32>)> = (0..len).map(|_| (1, Cell::new(0))).collect();
        pigeonhole_by_key(&mut v, |e| {
            e.1.set(e.1.get() + 1);
            e.0
        });
        assert!(
            v.iter().all(|e| e.1.get() == 0),
            "key called for {len} element"
        );
    }
}

#[test]
fn by_key_sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [2, 10, 200, size(3000, 300)] {
        // 72 bytes.
        let mut v: Vec<[u64; 9]> = (0..len)
            .map(|i| {
                let mut e = [(i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15); 9];
                e[0] = xorshift(&mut state) % 50;
                e[1] = i as u64;
                e
            })
            .collect();
        sort_checked(
            &mut v,
            |e| e[0],
            |e| e[1] as usize,
            |e| {
                e[2..]
                    .iter()
                    .all(|&x| x == e[1].wrapping_mul(0x9E37_79B9_7F4A_7C15))
            },
        );
        // 128 bytes, with one key per element.
        let mut v: Vec<[u64; 16]> = (0..len)
            .map(|i| {
                let mut e = [!(i as u64); 16];
                e[0] = i as u64;
                e
            })
            .collect();
        shuffle(&mut v, &mut state);
        sort_checked(
            &mut v,
            |e| e[0] as i32 - 7,
            |e| e[0] as usize,
            |e| e[1..].iter().all(|&x| x == !e[0]),
        );
        assert!(v.iter().map(|e| e[0]).eq(0..len as u64));
    }
}

/// An element over-aligned to 64 bytes.
#[repr(align(64))]
#[derive(Debug)]
struct Aligned64 {
    key: i64,
    id: u32,
    check: [u64; 3],
}

impl Aligned64 {
    fn new(key: i64, id: u32) -> Self {
        Self {
            key,
            id,
            check: [u64::from(id).wrapping_mul(0x9E37_79B9_7F4A_7C15); 3],
        }
    }

    fn intact(&self) -> bool {
        self.check == Self::new(self.key, self.id).check
    }
}

#[test]
fn by_key_sorts_over_aligned_elements() {
    let mut state = 0xA116_A116_A116_A116;
    for len in [2, 3, 17, 300, size(3000, 200)] {
        for span in [2, 50, 4 * len as u64] {
            let mut v: Vec<Aligned64> = (0..len as u32)
                .map(|id| Aligned64::new(i64::MIN + (xorshift(&mut state) % span) as i64, id))
                .collect();
            sort_checked(&mut v, |e| e.key, |e| e.id as usize, Aligned64::intact);
            assert!(v.iter().all(|e| (e as *const Aligned64).is_aligned()));
        }
    }
}

#[test]
fn by_key_sorts_strings() {
    let mut state = 0x1EAF_5EED_1EAF_5EED;
    for len in [2, 3, 50, size(500, 100)] {
        let mut v: Vec<String> = (0..len)
            .map(|id| format!("{:03}-{id}", xorshift(&mut state) % 100))
            .collect();
        let mut before = v.clone();
        let key = |s: &String| s[..3].parse::<u16>().unwrap();
        pigeonhole_by_key(&mut v, key);
        assert!(v.windows(2).all(|w| key(&w[0]) <= key(&w[1])));
        let mut after = v.clone();
        before.sort_unstable();
        after.sort_unstable();
        assert_eq!(after, before, "strings lost, duplicated or changed");
    }
}

#[test]
fn by_key_keeps_the_strong_counts_of_rcs() {
    let mut state = 0x0FC5_0FC5_0FC5_0FC5;
    for len in [2, 3, 40, size(2000, 200)] {
        let keepers: Vec<Rc<usize>> = (0..len).map(Rc::new).collect();
        let keys: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % 9) as u32)
            .collect();
        let mut v: Vec<(u32, Rc<usize>)> = keys
            .iter()
            .zip(&keepers)
            .map(|(&k, rc)| (k, Rc::clone(rc)))
            .collect();
        sort_checked(&mut v, |e| e.0, |e| *e.1, |e| e.0 == keys[*e.1]);
        assert!(
            keepers.iter().all(|rc| Rc::strong_count(rc) == 2),
            "Rc cloned or dropped"
        );
        drop(v);
        assert!(
            keepers.iter().all(|rc| Rc::strong_count(rc) == 1),
            "Rc leaked"
        );
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
fn by_key_sorts_owning_elements_exactly_once() {
    let mut state = 0x0DD5_0DD5_0DD5_0DD5;
    for len in [2, 3, 9, 40, 300, size(5000, 400)] {
        for span in [1u32, 2, 7, 300, 4 * len as u32] {
            let keys: Vec<u32> = (0..len)
                .map(|_| xorshift(&mut state) as u32 % span)
                .collect();
            let (order, completed) = sort_tracked(&keys, |v| pigeonhole_by_key(v, |t| t.key));
            assert!(completed);
            assert!(
                order.windows(2).all(|w| keys[w[0]] <= keys[w[1]]),
                "keys not sorted"
            );
        }
    }
}

#[test]
fn by_key_panicking_key_leaves_the_slice_as_it_was() {
    let len = size(150, 40);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32 % 7).collect();
    let mut permutation: Vec<u32> = (0..len as u32).collect();
    shuffle(&mut permutation, &mut state);
    let layouts: Vec<Vec<u32>> = vec![
        random,
        permutation,
        (0..len as u32).map(|i| i % 3).collect(),
        (0..len as u32).collect(),
        (0..len as u32).rev().collect(),
        vec![1; len],
    ];
    for keys in layouts {
        let calls = Cell::new(0);
        let (_, completed) = sort_tracked(&keys, |v| {
            pigeonhole_by_key(v, |t| {
                calls.set(calls.get() + 1);
                t.key
            })
        });
        assert!(completed);
        let total = calls.get();
        assert_eq!(total, len, "not one call per element");
        let step = if cfg!(miri) { 7 } else { 1 };
        for budget in (0..total).step_by(step) {
            let calls = Cell::new(0);
            let (order, completed) = sort_tracked(&keys, |v| {
                pigeonhole_by_key(v, |t| {
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
fn by_key_panics_on_a_range_over_the_limit_and_leaves_the_slice_alone() {
    for keys in [
        [65_536u32, 0, 1],
        [65_536, 0, 0],
        [0, 1, 65_536],
        [0, 0, 65_536],
    ] {
        let mut v: Vec<(u32, usize)> = keys.iter().copied().zip(0..).collect();
        let before = v.clone();
        let result = catch_unwind(AssertUnwindSafe(|| pigeonhole_by_key(&mut v, |e| e.0)));
        let payload = result
            .err()
            .unwrap_or_else(|| panic!("{keys:?} did not panic"));
        let message = message_of(payload);
        assert!(message.starts_with("pigeonhole sort:"), "{message}");
        assert_eq!(v, before);
        // Owning elements are neither dropped nor duplicated by the panic.
        let (order, completed) = sort_tracked(&keys, |v| pigeonhole_by_key(v, |t| t.key));
        assert!(!completed);
        assert_eq!(order, vec![0, 1, 2]);
    }
    // At the limit, and one past it, beyond 16,384 elements.
    let n = size(20_000, 16_400);
    let mut keys: Vec<u32> = (0..n as u32).rev().collect();
    keys[0] = 4 * n as u32 - 1;
    assert_by_key(&keys);
    keys[0] = 4 * n as u32;
    let mut v: Vec<(u32, usize)> = keys.iter().copied().zip(0..).collect();
    let result = catch_unwind(AssertUnwindSafe(|| pigeonhole_by_key(&mut v, |e| e.0)));
    assert!(result.is_err());
    assert!(v.iter().map(|e| e.1).eq(0..n));
    // Distances that do not fit in `usize`.
    let mut v = [(u128::MAX, 0), (0, 1)];
    assert!(catch_unwind(AssertUnwindSafe(|| pigeonhole_by_key(&mut v, |e| e.0))).is_err());
    assert_eq!(v, [(u128::MAX, 0), (0, 1)]);
    let mut v = [(i64::MIN, 0), (i64::MAX, 1)];
    assert!(catch_unwind(AssertUnwindSafe(|| pigeonhole_by_key(&mut v, |e| e.0))).is_err());
    assert_eq!(v, [(i64::MIN, 0), (i64::MAX, 1)]);
}

/// Sorts `v` with [`pigeonhole_by_key`] by `e[0]` and returns how many allocations that made.
fn allocations_by_key_of<const N: usize>(v: &mut [[u64; N]]) -> usize {
    crate::allocations::count(|| pigeonhole_by_key(v, |e| e[0]))
}

#[test]
fn by_key_allocates_only_beyond_4_kib() {
    let mut state = 0xA110_CA7E_A110_CA7E;
    // Few elements over a small range: counts and holes fit on the stack.
    // 512 `u64` keys fill the 4 KiB key buffer.
    let mut allocating = Vec::new();
    for (len, span) in [(2, 2), (100, 100), (100, 512), (100, 1024), (512, 64)] {
        let mut v: Vec<[u64; 1]> = (0..len).map(|_| [xorshift(&mut state) % span]).collect();
        v[0] = [span - 1];
        v[1] = [0];
        let allocations = allocations_by_key_of(&mut v);
        if allocations != 0 {
            allocating.push((len, span, allocations));
        }
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    assert!(
        allocating.is_empty(),
        "(len, span, allocations): {allocating:?}"
    );
    // Many elements, or a wide range: something must be allocated.
    for (len, span) in [(100, 5000), (size(20_000, 5_000), 50)] {
        let mut v: Vec<[u64; 1]> = (0..len).map(|_| [xorshift(&mut state) % span]).collect();
        v[0] = [span - 1];
        v[1] = [0];
        assert!(
            allocations_by_key_of(&mut v) > 0,
            "len = {len}, span = {span}"
        );
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}

// ---------------------------------------------------------------------------------------------
// Elements of odd sizes and alignments
// ---------------------------------------------------------------------------------------------

thread_local! {
    /// How many times the [`Bytes`] element with each id has been dropped on this thread.
    static DROPS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// An element of `N` bytes, at least 4, aligned as `A`: its id in the first four bytes and the
/// rest filled from it, so that an element pieced together from two others is noticed. It
/// counts its drops in [`DROPS`].
struct Bytes<const N: usize, A> {
    bytes: [u8; N],
    _align: [A; 0],
}

impl<const N: usize, A> Bytes<N, A> {
    fn new(id: u32) -> Self {
        Self {
            bytes: Self::of(id),
            _align: [],
        }
    }

    fn of(id: u32) -> [u8; N] {
        let mut b = [0; N];
        b[..4].copy_from_slice(&id.to_le_bytes());
        for (j, x) in b.iter_mut().enumerate().skip(4) {
            *x = (id.wrapping_mul(2_654_435_761) >> (j % 24)) as u8 ^ j as u8;
        }
        b
    }

    fn id(&self) -> u32 {
        u32::from_le_bytes(self.bytes[..4].try_into().unwrap())
    }
}

impl<const N: usize, A> Drop for Bytes<N, A> {
    fn drop(&mut self) {
        DROPS.with(|d| d.borrow_mut()[self.id() as usize] += 1);
    }
}

/// Zero-sized markers that raise the alignment of a [`Bytes`] above one.
#[repr(align(8))]
struct Align8;
#[repr(align(64))]
struct Align64;

/// Sorts elements of type `Bytes<N, A>` from position `skip` on with [`pigeonhole_by_key`], the
/// element with id `i` by `keys[i]`, with a key that panics after `budget` calls if one is given.
/// Checks that the first `skip` stay where they are, that the rest end up sorted by key (or, if
/// the key panicked, exactly as they were), that none is changed or dropped during the sort, and
/// that each is dropped exactly once afterwards.
fn assert_bytes<const N: usize, A, K: IntegerKey + Debug>(
    keys: &[K],
    skip: usize,
    budget: Option<usize>,
) {
    let len = keys.len();
    DROPS.with(|d| *d.borrow_mut() = vec![0; len]);
    let mut v: Vec<Bytes<N, A>> = (0..len as u32).map(Bytes::new).collect();
    let calls = Cell::new(0);
    let completed = catch_unwind(AssertUnwindSafe(|| {
        pigeonhole_by_key(&mut v[skip..], |e| {
            if Some(calls.get()) == budget {
                panic!("key budget exhausted");
            }
            calls.set(calls.get() + 1);
            keys[e.id() as usize]
        })
    }))
    .is_ok();
    assert!(
        DROPS.with(|d| d.borrow().iter().all(|&n| n == 0)),
        "element dropped during sort"
    );
    assert!(
        v.iter().all(|e| e.bytes == Bytes::<N, A>::of(e.id())),
        "element changed"
    );
    let ids: Vec<u32> = v.iter().map(Bytes::id).collect();
    assert!(ids[..skip].iter().copied().eq(0..skip as u32));
    if completed {
        let mut sorted = ids[skip..].to_vec();
        sorted.sort_unstable();
        assert!(
            sorted.iter().copied().eq(skip as u32..len as u32),
            "{N}-byte elements lost or duplicated"
        );
        assert!(
            ids[skip..]
                .windows(2)
                .all(|w| keys[w[0] as usize] <= keys[w[1] as usize]),
            "{N}-byte elements not sorted, skip {skip}"
        );
    } else {
        assert!(
            ids.iter().copied().eq(0..len as u32),
            "{N}-byte elements moved by a panicking key"
        );
    }
    drop(v);
    assert!(
        DROPS.with(|d| d.borrow().iter().all(|&n| n == 1)),
        "element leaked or double-dropped"
    );
}

#[test]
fn by_key_moves_elements_of_odd_sizes_and_alignments() {
    let mut state = 0xDEAD_BEEF_0BAD_F00D;
    for len in [2, 3, 17, 40, size(300, 60)] {
        let keys: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % 7) as u32 * 1000 + (xorshift(&mut state) % 5) as u32)
            .collect();
        for skip in 0..2 {
            assert_bytes::<5, u8, u32>(&keys, skip, None);
            assert_bytes::<7, u8, u32>(&keys, skip, None);
            assert_bytes::<12, u32, u32>(&keys, skip, None);
            assert_bytes::<20, u8, u32>(&keys, skip, None);
            assert_bytes::<36, u16, u32>(&keys, skip, None);
            assert_bytes::<65, u8, u32>(&keys, skip, None);
            assert_bytes::<72, Align8, u32>(&keys, skip, None);
            assert_bytes::<4, Align64, u32>(&keys, skip, None);
            assert_bytes::<128, Align64, u32>(&keys, skip, None);
        }
        // One element per hole.
        let mut permutation: Vec<i16> = (0..len as i16).collect();
        shuffle(&mut permutation, &mut state);
        assert_bytes::<65, u8, i16>(&permutation, 0, None);
        assert_bytes::<128, Align64, i16>(&permutation, 0, None);
    }
}

#[test]
fn by_key_panicking_key_leaves_odd_elements_as_they_were() {
    let mut state = 0xFA11_FA11_FA11_FA11;
    let len = size(60, 20);
    let keys: Vec<u8> = (0..len).map(|_| (xorshift(&mut state) % 9) as u8).collect();
    let step = if cfg!(miri) { 5 } else { 1 };
    for budget in (0..len).step_by(step) {
        assert_bytes::<65, u8, u8>(&keys, 0, Some(budget));
        assert_bytes::<128, Align64, u8>(&keys, 0, Some(budget));
        assert_bytes::<12, u32, u8>(&keys, 1, Some(budget.min(len - 2)));
    }
}

#[test]
fn places_slices_of_32768_elements_and_more_by_swaps() {
    // From 32,768 elements of up to 40 bytes with their keys, elements out of order are swapped
    // into their holes instead of moved along cycles; this size is kept under Miri too.
    let mut state = 0x53A9_53A9_53A9_53A9;
    let len = 1 << 15;
    let mut v: Vec<u32> = (0..len)
        .map(|_| (xorshift(&mut state) % 1000) as u32)
        .collect();
    let mut expected = v.clone();
    expected.sort_unstable();
    pigeonhole(&mut v);
    assert_eq!(v, expected);
    let keys: Vec<u16> = (0..len)
        .map(|_| (xorshift(&mut state) % 3000) as u16)
        .collect();
    let mut v: Vec<(u16, Box<usize>)> = keys
        .iter()
        .zip(0..)
        .map(|(&k, i)| (k, Box::new(i)))
        .collect();
    pigeonhole_by_key(&mut v, |e| e.0);
    assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
    let mut ids: Vec<usize> = v.iter().map(|e| *e.1).collect();
    ids.sort_unstable();
    assert!(ids.into_iter().eq(0..len));
    assert!(v.iter().all(|e| keys[*e.1] == e.0));
}

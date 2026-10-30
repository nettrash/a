use a::sort::{IntegerKey, sleep};
use std::fmt::Debug;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::{Duration, Instant};

/// A unit long enough that threads on a loaded machine almost always wake in order, and short
/// enough that a few dozen values take a fraction of a second.
const UNIT: Duration = miri_scaled(Duration::from_millis(2));

/// `unit`, or 5000 times `unit` under Miri: Miri's clock is virtual and advances 5 µs for every
/// basic block it interprets, so starting a thread takes it milliseconds of that time, and a
/// unit of milliseconds sorts at random. Sleeping costs Miri nothing, however long.
const fn miri_scaled(unit: Duration) -> Duration {
    if cfg!(miri) {
        unit.saturating_mul(5000)
    } else {
        unit
    }
}

/// How many times a sort that may come out unsorted is tried before the test fails, the unit
/// doubling every time: a real scheduler may wake a thread late, all the more on a machine busy
/// with other tests, so a single unsorted result is not a bug.
const TRIES: u32 = 6;

/// Scales a test down when it runs under Miri.
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

/// Sorts a copy of `v` with [`sleep`] by `unit` once and checks what must always hold, however
/// late the threads wake: the result is a permutation of `v`, and the return value says exactly
/// whether it is sorted. Returns the result and the return value.
fn sleep_once<T: IntegerKey + Send + Debug>(v: &[T], unit: Duration) -> (Vec<T>, bool) {
    let mut actual = v.to_vec();
    let sorted = sleep(&mut actual, unit);
    let mut expected = v.to_vec();
    expected.sort_unstable();
    let mut permuted = actual.clone();
    permuted.sort_unstable();
    assert_eq!(permuted, expected, "not a permutation of {v:?}: {actual:?}");
    assert_eq!(
        sorted,
        actual.is_sorted(),
        "wrong return value for {actual:?}"
    );
    (actual, sorted)
}

/// Sorts `v` with [`sleep`] by `unit`, then by twice `unit`, and so on up to [`TRIES`] times,
/// checking every result with [`sleep_once`], until one comes out sorted; fails if none does.
fn assert_sleep<T: IntegerKey + Send + Debug>(v: &[T], unit: Duration) {
    for t in 0..TRIES {
        if sleep_once(v, unit * (1 << t)).1 {
            return;
        }
    }
    panic!("never sorted in {TRIES} tries: {v:?}");
}

/// Sorts `v` by `unit` with [`sleep`] and returns the panic message, if it panicked, checking
/// that the slice was left as it was.
fn panic_of<T: IntegerKey + Send + Debug>(v: &[T], unit: Duration) -> Option<String> {
    let mut actual = v.to_vec();
    let result = catch_unwind(AssertUnwindSafe(|| sleep(&mut actual, unit)));
    let payload = result.err()?;
    assert_eq!(actual, v, "slice changed by a panicking sort");
    Some(if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        String::new()
    })
}

#[test]
fn sorts_empty_and_single_element_without_sleeping() {
    let mut empty: [u32; 0] = [];
    assert!(sleep(&mut empty, Duration::MAX));
    let mut one = [42u64];
    assert!(sleep(&mut one, Duration::MAX));
    assert_eq!(one, [42]);
    let mut extreme = [i128::MAX];
    assert!(sleep(&mut extreme, Duration::MAX));
    assert_eq!(extreme, [i128::MAX]);
}

#[test]
fn sorts_two_elements() {
    assert_sleep(&[1, 0], UNIT);
    assert_sleep(&[0, 1], UNIT);
    assert_sleep(&[-1i8, -2], UNIT);
}

#[test]
fn sorts_every_permutation_of_a_few_values() {
    // 3! = 6 permutations, each at most 2 units long; one under Miri.
    let perms: &[[u8; 3]] = &[
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    for p in &perms[..size(perms.len(), 1)] {
        assert_sleep(p, UNIT);
    }
}

#[test]
fn sorts_a_random_permutation() {
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let n = size(20, 6);
    let mut v: Vec<u16> = (0..n as u16).collect();
    shuffle(&mut v, &mut state);
    assert_sleep(&v, UNIT);
}

#[test]
fn sorts_with_duplicates() {
    assert_sleep(&[3u32, 1, 3, 0, 1, 2, 0, 3], UNIT);
}

#[test]
fn sorts_negative_values_from_the_smallest() {
    assert_sleep(&[3i32, -2, 0, -5, 1, -1], UNIT);
    assert_sleep(&[-7i64, -9, -8], UNIT);
}

#[test]
fn sorts_values_near_the_extremes_of_every_type() {
    macro_rules! near_extremes {
        ($($t:ty),*) => {$(
            assert_sleep(&[<$t>::MAX, <$t>::MAX - 2, <$t>::MAX - 1], UNIT);
            assert_sleep(&[<$t>::MIN + 2, <$t>::MIN, <$t>::MIN + 1], UNIT);
        )*};
    }
    near_extremes!(
        u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize
    );
}

#[test]
fn sorts_chars() {
    let letters: Vec<char> = "sleep".chars().collect();
    assert_sleep(&letters, UNIT);
    // Across the surrogates: 2049 code points apart, at 1 µs each.
    assert_sleep(
        &['\u{E000}', '\u{D7FF}', 'a'],
        miri_scaled(Duration::from_micros(1)),
    );
}

#[test]
fn sorts_bools() {
    assert_sleep(&[true, false, true, false, false], UNIT);
}

#[test]
fn equal_values_do_not_sleep() {
    // Every value is the smallest, so no thread sleeps past the start, whatever the unit, and
    // equal values come out sorted in any order.
    let n = size(50, 4);
    let start = Instant::now();
    let (v, sorted) = sleep_once(&vec![7u32; n], Duration::from_secs(3600));
    assert!(sorted);
    assert_eq!(v, vec![7u32; n]);
    assert!(start.elapsed() < Duration::from_secs(60));
    let (v, sorted) = sleep_once(&[i128::MIN; 3], Duration::MAX);
    assert!(sorted);
    assert_eq!(v, [i128::MIN; 3]);
}

#[test]
fn takes_at_least_the_range_times_the_unit() {
    // The largest value wakes `(max − min) × unit` after a start that is not before the call.
    let unit = Duration::from_millis(5);
    let start = Instant::now();
    sleep_once(&[9u8, 0, 4], unit);
    assert!(start.elapsed() >= unit * 9);
}

#[test]
fn zero_unit_keeps_a_permutation_and_reports_the_order() {
    let mut state = 12345;
    let n = size(200, 8);
    let mut v: Vec<i32> = (0..n as i32).map(|i| i * 7 - 300).collect();
    shuffle(&mut v, &mut state);
    sleep_once(&v, Duration::ZERO);
    // The whole range of every type at once, with nothing to sleep.
    sleep_once(&[i64::MAX, i64::MIN, 0, -1, 1], Duration::ZERO);
    sleep_once(&[u128::MAX, 0, u128::MAX / 2], Duration::ZERO);
    sleep_once(&[i128::MIN, i128::MAX], Duration::ZERO);
    sleep_once(&[char::MAX, '\0'], Duration::ZERO);
}

#[test]
fn tiny_units_keep_a_permutation() {
    // The whole range of `i16` at a nanosecond per unit: 65.5 µs, too fine to be sure of the
    // order, but never a lost or duplicated value.
    sleep_once(&[i16::MAX, 0, i16::MIN, -1, 1], Duration::from_nanos(1));
    sleep_once(&[u8::MAX, 0, 128, 127], Duration::from_nanos(1));
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v = [9u32, 5, 3, 4, 1, 0];
    for t in 0..TRIES {
        let mut w = v;
        if sleep(&mut w[1..5], UNIT * (1 << t)) {
            v = w;
            break;
        }
        assert_eq!((w[0], w[5]), (9, 0));
    }
    assert_eq!(v, [9, 1, 3, 4, 5, 0]);
}

#[test]
fn panics_on_a_range_the_clock_cannot_represent_and_leaves_the_slice_alone() {
    // 2^128 nanoseconds, saturated to `Duration::MAX`.
    let message = panic_of(&[i128::MAX, 0, i128::MIN], Duration::from_nanos(1)).unwrap();
    assert!(message.starts_with("sleep sort: "), "{message}");
    // 2^64 seconds.
    assert!(panic_of(&[u64::MAX, 0], Duration::from_secs(1)).is_some());
    // The longest unit over a range of one.
    assert!(panic_of(&[true, false], Duration::MAX).is_some());
    assert!(panic_of(&[1u8, 0], Duration::MAX).is_some());
}

#[test]
fn sorts_more_elements_than_cores() {
    // One thread per element, far more than the machine has cores.
    let mut state = 777;
    let n = size(300, 10);
    let mut v: Vec<u32> = (0..n as u32).collect();
    shuffle(&mut v, &mut state);
    assert_sleep(&v, miri_scaled(Duration::from_micros(500)));
}

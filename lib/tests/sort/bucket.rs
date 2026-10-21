use a::sort::{BucketKey, bucket, bucket_by_key};
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
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

/// Shuffles `v` with Fisher–Yates driven by [`xorshift`].
fn shuffle<T>(v: &mut [T], state: &mut u64) {
    for i in (1..v.len()).rev() {
        let j = (xorshift(state) % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
}

/// The input to print in a failure message: the slice itself if it is short, its length
/// otherwise.
fn describe<T: Debug>(v: &[T]) -> String {
    if v.len() <= 64 {
        format!("{v:?}")
    } else {
        format!("{} elements", v.len())
    }
}

/// Sorts `v` with [`bucket`] and checks the result against [`slice::sort_unstable`].
fn assert_bucket<T: BucketKey + Copy + Ord + Debug>(v: &[T]) {
    let mut expected = v.to_vec();
    expected.sort_unstable();
    let mut actual = v.to_vec();
    bucket(&mut actual);
    assert!(actual == expected, "input {}", describe(v));
}

/// The floating-point key types, sorted in `total_cmp` order.
trait Float: BucketKey + Copy + Debug {
    type Raw: Copy + Eq + Debug;
    fn bits(self) -> Self::Raw;
    fn total(a: &Self, b: &Self) -> Ordering;
}

impl Float for f64 {
    type Raw = u64;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn total(a: &Self, b: &Self) -> Ordering {
        a.total_cmp(b)
    }
}

impl Float for f32 {
    type Raw = u32;
    fn bits(self) -> u32 {
        self.to_bits()
    }
    fn total(a: &Self, b: &Self) -> Ordering {
        a.total_cmp(b)
    }
}

fn bits_of<F: Float>(v: &[F]) -> Vec<F::Raw> {
    v.iter().map(|x| x.bits()).collect()
}

/// Sorts `v` with [`bucket`] and checks the result bit for bit against sorting with
/// `total_cmp`: `==` would let NaNs differ and `-0.0` stand for `+0.0`.
fn assert_bucket_floats<F: Float>(v: &[F]) {
    let mut expected = v.to_vec();
    expected.sort_by(F::total);
    let mut actual = v.to_vec();
    bucket(&mut actual);
    assert!(
        bits_of(&actual) == bits_of(&expected),
        "input {}\nexpected {}\nactual {}",
        describe(v),
        describe(&bits_of(&expected)),
        describe(&bits_of(&actual)),
    );
}

/// Sorts `keys`, each paired with its index, with [`bucket_by_key`] and checks the result
/// against the standard library's stable sort.
fn assert_bucket_by_key<K: BucketKey + Copy + Ord + Debug>(keys: &[K]) {
    let mut expected: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|&(k, _)| k);
    let mut actual: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    bucket_by_key(&mut actual, |&(k, _)| k);
    assert!(actual == expected, "keys {}", describe(keys));
}

/// Sorts float `keys`, each paired with its index, with [`bucket_by_key`] and checks the
/// result bit for bit against the standard library's stable sort by `total_cmp`.
fn assert_bucket_by_float_key<F: Float>(keys: &[F]) {
    let mut expected: Vec<(F, usize)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by(|a, b| F::total(&a.0, &b.0));
    let mut actual: Vec<(F, usize)> = keys.iter().copied().zip(0..).collect();
    bucket_by_key(&mut actual, |&(k, _)| k);
    let bits = |v: &[(F, usize)]| v.iter().map(|&(k, i)| (k.bits(), i)).collect::<Vec<_>>();
    assert!(bits(&actual) == bits(&expected), "keys {}", describe(keys));
}

/// A random `f64` with random sign, exponent and mantissa: every finite value, subnormals,
/// infinities and NaNs.
fn any_f64(state: &mut u64) -> f64 {
    f64::from_bits(xorshift(state))
}

/// A random finite `f64` spread over the whole exponent range.
fn finite_f64(state: &mut u64) -> f64 {
    loop {
        let x = any_f64(state);
        if x.is_finite() {
            return x;
        }
    }
}

/// A uniform `f64` in `[0, 1)`.
fn unit_f64(state: &mut u64) -> f64 {
    (xorshift(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Special values: both zeros, both infinities, NaNs of either sign with different payloads,
/// the extremes, the smallest normal and subnormals.
fn special_f64s() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF8_0000_0000_0001),
        f64::from_bits(0xFFF8_0000_0000_0001),
        f64::from_bits(0x7FF0_0000_0000_0001),
        f64::from_bits(0x7FFF_FFFF_FFFF_FFFF),
        f64::from_bits(0xFFFF_FFFF_FFFF_FFFF),
        f64::MIN,
        f64::MAX,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000F_FFFF_FFFF_FFFF),
        -f64::from_bits(0x000F_FFFF_FFFF_FFFF),
        1.0,
        -1.0,
        f64::EPSILON,
    ]
}

/// The `f32` counterparts of [`special_f64s`].
fn special_f32s() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7FC0_0001),
        f32::from_bits(0xFFC0_0001),
        f32::from_bits(0x7F80_0001),
        f32::from_bits(0x7FFF_FFFF),
        f32::from_bits(0xFFFF_FFFF),
        f32::MIN,
        f32::MAX,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x007F_FFFF),
        -f32::from_bits(0x007F_FFFF),
        1.0,
        -1.0,
        f32::EPSILON,
    ]
}

// ---------------------------------------------------------------------------------------------
// `bucket`
// ---------------------------------------------------------------------------------------------

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    bucket(&mut v);
    assert_eq!(v, vec![]);
    let mut v: Vec<f64> = vec![];
    bucket(&mut v);
    assert!(v.is_empty());
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    bucket(&mut v);
    assert_eq!(v, vec![42]);
    assert_bucket_floats(&[f64::NAN]);
    assert_bucket_floats(&[-0.0f32]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    bucket(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    bucket(&mut v);
    assert_eq!(v, vec![1, 2]);

    assert_bucket(&[u128::MAX, 0]);
    assert_bucket(&[i128::MAX, i128::MIN]);
    assert_bucket_floats(&[0.0f64, -0.0]);
    assert_bucket_floats(&[f64::NAN, -f64::NAN]);
    assert_bucket_floats(&[f64::MAX, f64::MIN]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    bucket(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    bucket(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_non_increasing() {
    // Reversed as a whole: equal values cannot be told apart.
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    bucket(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    // Equal floats with different bits are not equal values: `+0.0` before `-0.0` is not
    // non-increasing in `total_cmp` order, nor are NaNs with payloads out of order.
    assert_bucket_floats(&[1.0f64, 0.0, -0.0, -1.0]);
    assert_bucket_floats(&[1.0f64, -0.0, 0.0, -1.0]);
    assert_bucket_floats(&[
        f64::from_bits(0x7FF8_0000_0000_0002),
        f64::from_bits(0x7FF8_0000_0000_0001),
        f64::NAN,
        f64::INFINITY,
    ]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    bucket(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        bucket(&mut v);
        assert_eq!(v, vec![7; len]);
        assert_bucket_floats(&vec![f64::NAN; len]);
        assert_bucket_floats(&vec![-0.0f32; len]);
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    bucket(&mut v[5..25]);
    let mut middle = before[5..25].to_vec();
    middle.sort_unstable();
    assert_eq!(&v[..5], &before[..5]);
    assert_eq!(&v[5..25], &middle[..]);
    assert_eq!(&v[25..], &before[25..]);

    let mut state = 0x5B5B_5B5B_5B5B_5B5B;
    let mut v: Vec<f64> = (0..size(3000, 300)).map(|_| any_f64(&mut state)).collect();
    let before = v.clone();
    let (lo, hi) = (v.len() / 5, v.len() - v.len() / 7);
    bucket(&mut v[lo..hi]);
    let mut middle = before[lo..hi].to_vec();
    middle.sort_by(f64::total_cmp);
    assert_eq!(bits_of(&v[..lo]), bits_of(&before[..lo]));
    assert_eq!(bits_of(&v[lo..hi]), bits_of(&middle));
    assert_eq!(bits_of(&v[hi..]), bits_of(&before[hi..]));
}

#[test]
fn sorts_every_sequence_over_few_values() {
    let max_len = size(7, 4);
    for len in 0..=max_len {
        let total = 4usize.pow(len as u32);
        for code in 0..total {
            let v: Vec<u8> = (0..len)
                .map(|i| (code / 4usize.pow(i as u32) % 4) as u8 * 85)
                .collect();
            assert_bucket(&v);
        }
    }
}

#[test]
fn sorts_every_sequence_over_few_floats() {
    // Every sequence of up to 6 values out of 6 that `==` cannot tell apart or that do not
    // compare at all.
    let values = [
        f64::NEG_INFINITY,
        -0.0,
        0.0,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7FF8_0000_0000_0001),
    ];
    let max_len = size(6, 3);
    let k = values.len();
    for len in 0..=max_len {
        let total = k.pow(len as u32);
        for code in 0..total {
            let v: Vec<f64> = (0..len)
                .map(|i| values[code / k.pow(i as u32) % k])
                .collect();
            assert_bucket_floats(&v);
        }
    }
}

/// Checks random slices of every length up to 70 and some longer ones, over the whole range
/// of the type and over `span` values from each of `bases`.
macro_rules! check_integers {
    ($t:ty, $state:expr, $span:expr, $bases:expr) => {{
        let lens = (0..70).chain([100, 257, 1000, size(5000, 400)]);
        for len in lens {
            let v: Vec<$t> = (0..len)
                .map(|_| {
                    let r = (u128::from(xorshift($state)) << 64) | u128::from(xorshift($state));
                    r as $t
                })
                .collect();
            assert_bucket(&v);
            for base in $bases {
                let v: Vec<$t> = (0..len)
                    .map(|_| <$t>::wrapping_add(base, (xorshift($state) % $span) as $t))
                    .collect();
                assert_bucket(&v);
            }
        }
    }};
}

#[test]
fn sorts_every_integer_type() {
    let mut state = 0x1234_5678_9ABC_DEF0;
    let s = &mut state;
    check_integers!(u8, s, 7, [0u8, 250]);
    check_integers!(i8, s, 7, [i8::MIN, -3]);
    check_integers!(u16, s, 1000, [0, u16::MAX - 999]);
    check_integers!(i16, s, 1000, [i16::MIN, -500]);
    check_integers!(u32, s, 1000, [0, u32::MAX - 999]);
    check_integers!(i32, s, 1000, [i32::MIN, -500, i32::MAX - 999]);
    check_integers!(u64, s, 1 << 40, [0, u64::MAX - (1 << 40)]);
    check_integers!(i64, s, 1000, [i64::MIN, -500, i64::MAX - 999]);
    check_integers!(u128, s, 1000, [0, u128::MAX - 999]);
    check_integers!(i128, s, 1000, [i128::MIN, -500, i128::MAX - 999]);
    check_integers!(usize, s, 1000, [0, usize::MAX - 999]);
    check_integers!(isize, s, 1000, [isize::MIN, -500, isize::MAX - 999]);
}

#[test]
fn sorts_integers_at_both_extremes_together() {
    // The range from `MIN` to `MAX` overflows the type it is measured in.
    let mut state = 0xE47E_E47E_E47E_E47E;
    for len in [2, 3, 5, 17, 100, size(3000, 300)] {
        let mut r = || xorshift(&mut state);
        let i128s: Vec<i128> = (0..len)
            .map(|_| match r() % 4 {
                0 => i128::MIN,
                1 => i128::MAX,
                2 => i128::MIN + (r() % 5) as i128,
                _ => ((u128::from(r()) << 64) | u128::from(r())) as i128,
            })
            .collect();
        assert_bucket(&i128s);
        let u128s: Vec<u128> = (0..len)
            .map(|_| match r() % 4 {
                0 => 0,
                1 => u128::MAX,
                2 => u128::MAX - u128::from(r() % 5),
                _ => u128::from(r() % 5),
            })
            .collect();
        assert_bucket(&u128s);
        let i64s: Vec<i64> = (0..len)
            .map(|_| [i64::MIN, i64::MAX, -1, 0, 1][(r() % 5) as usize])
            .collect();
        assert_bucket(&i64s);
        let u64s: Vec<u64> = (0..len)
            .map(|_| [0, u64::MAX, u64::MAX - 1, 1][(r() % 4) as usize])
            .collect();
        assert_bucket(&u64s);
        let i8s: Vec<i8> = (0..len)
            .map(|_| [i8::MIN, i8::MAX, -1, 0][(r() % 4) as usize])
            .collect();
        assert_bucket(&i8s);
        let isizes: Vec<isize> = (0..len)
            .map(|_| [isize::MIN, isize::MAX, 0][(r() % 3) as usize])
            .collect();
        assert_bucket(&isizes);
    }
}

#[test]
fn sorts_chars_across_the_surrogates() {
    // The range from U+D7FF to U+E000 contains the surrogates, which are not `char`s.
    let mut state = 0xC4A2_C4A2_C4A2_C4A2;
    let pool = [
        '\u{D7FE}',
        '\u{D7FF}',
        '\u{E000}',
        '\u{E001}',
        'a',
        '\u{FFFF}',
        '\u{10FFFF}',
        '\0',
    ];
    for len in 0..size(300, 60) {
        let v: Vec<char> = (0..len)
            .map(|_| pool[(xorshift(&mut state) % pool.len() as u64) as usize])
            .collect();
        assert_bucket(&v);
    }
    for len in [100, size(5000, 500)] {
        let v: Vec<char> = (0..len)
            .map(|_| {
                char::from_u32((xorshift(&mut state) % 0x11_0000) as u32).unwrap_or('\u{E000}')
            })
            .collect();
        assert_bucket(&v);
        // Only the values around the surrogates: one bucket width may straddle them.
        let v: Vec<char> = (0..len)
            .map(|_| {
                char::from_u32(0xD7F0 + (xorshift(&mut state) % 0x830) as u32).unwrap_or('\u{D7FF}')
            })
            .collect();
        assert_bucket(&v);
    }
    assert_bucket(&['\u{10FFFF}', '\u{10FFFE}', '\u{10FFFF}', '\u{10FF00}']);
    assert_bucket(&['\0', '\u{1}', '\0', '\u{FFFF}', '\u{E000}', '\u{D7FF}']);
}

#[test]
fn sorts_bools() {
    for len in 0..size(300, 40) {
        let v: Vec<bool> = (0..len).map(|i| (i * 7) % 3 == 0).collect();
        assert_bucket(&v);
    }
    assert_bucket(&[true, false]);
    assert_bucket(&[true, true, true]);
    assert_bucket(&[false, true, false, true]);
}

#[test]
fn sorts_keys_that_differ_in_few_bits() {
    // Keys that differ only in their lowest bits, only in their highest bits, or in a few bits
    // far apart: the buckets of the first level take all keys in one or two of them.
    let mut state = 0xB175_B175_B175_B175;
    for len in [2, 3, 17, 64, 65, 300, size(5000, 600)] {
        for bits in [
            vec![0u32],
            vec![63],
            vec![0, 63],
            vec![0, 1],
            vec![62, 63],
            vec![3, 50],
            vec![5, 6, 7, 8],
            (0..12).collect(),
            (52..64).collect(),
            (4..40).collect(),
        ] {
            let v: Vec<u64> = (0..len)
                .map(|_| {
                    let r = xorshift(&mut state);
                    bits.iter()
                        .enumerate()
                        .fold(0x5A5A_0000_0000_5A5A, |x, (i, &b)| {
                            x ^ (((r >> i) & 1) << b)
                        })
                })
                .collect();
            assert_bucket(&v);
            let w: Vec<i64> = v.iter().map(|&x| x as i64).collect();
            assert_bucket(&w);
            // The same bit patterns as floats, finite or not.
            let f: Vec<f64> = v.iter().map(|&x| f64::from_bits(x)).collect();
            assert_bucket_floats(&f);
        }
    }
}

#[test]
fn sorts_skewed_integers_that_need_recursion() {
    let mut state = 0x5CE3_5CE3_5CE3_5CE3;
    let n = size(10_000, 1_000);
    let mut r = || xorshift(&mut state);
    // Almost all keys in the first bucket, and one outlier at the end of the range.
    let mut v: Vec<u64> = (0..n).map(|_| r() % 100).collect();
    v.push(u64::MAX);
    assert_bucket(&v);
    v.rotate_right(n / 3);
    assert_bucket(&v);
    // ... and at both ends.
    let mut v: Vec<i64> = (0..n).map(|_| (r() % 100) as i64).collect();
    v.push(i64::MIN);
    v.push(i64::MAX);
    v.rotate_right(n / 2);
    assert_bucket(&v);
    // Powers of two: each level splits off only the largest.
    let v: Vec<u64> = (0..n).map(|_| 1 << (r() % 64)).collect();
    assert_bucket(&v);
    let v: Vec<u128> = (0..n).map(|_| 1 << (r() % 128)).collect();
    assert_bucket(&v);
    let v: Vec<u32> = (0..n).map(|_| (1 << (r() % 32)) - 1).collect();
    assert_bucket(&v);
    // Exponentially distributed: `r` shifted by a random amount.
    let v: Vec<u64> = (0..n).map(|_| r() >> (r() % 64)).collect();
    assert_bucket(&v);
    // Many duplicates of a few values and one far outlier.
    let mut v: Vec<u32> = (0..n).map(|_| [3, 5, 8][(r() % 3) as usize]).collect();
    v[n / 2] = u32::MAX;
    assert_bucket(&v);
    // A dense cluster in a range otherwise empty, plus outliers spread over the rest.
    let v: Vec<u64> = (0..n)
        .map(|i| {
            if i % 97 == 0 {
                r()
            } else {
                (1 << 40) + r() % 4096
            }
        })
        .collect();
    assert_bucket(&v);
    // Many duplicates in each of a few clusters far apart.
    let v: Vec<i32> = (0..n)
        .map(|_| [i32::MIN + (r() % 3) as i32, (r() % 3) as i32, i32::MAX - 2][(r() % 3) as usize])
        .collect();
    assert_bucket(&v);
}

#[test]
fn sorts_patterns() {
    let n = size(20_000, 1_000);
    let mut state = 0x9A77_E295_9A77_E295;
    let mut nearly_sorted: Vec<u32> = (0..n as u32).collect();
    for _ in 0..10 {
        let i = (xorshift(&mut state) % n as u64) as usize;
        let j = (xorshift(&mut state) % n as u64) as usize;
        nearly_sorted.swap(i, j);
    }
    let mut nearly_reversed = nearly_sorted.clone();
    nearly_reversed.reverse();
    let shapes: Vec<Vec<u32>> = vec![
        // Sawtooth of several tooth sizes.
        (0..n as u32).map(|i| i % 17).collect(),
        (0..n as u32).map(|i| i % 1000).collect(),
        (0..n as u32).map(|i| (i % 1000) * 4_000_000).collect(),
        // Descending teeth.
        (0..n as u32).map(|i| 1000 - i % 1000).collect(),
        // Organ pipe: ascending then descending, and the other way round.
        (0..n as u32).map(|i| i.min(n as u32 - 1 - i)).collect(),
        (0..n as u32).map(|i| (n as u32 / 2).abs_diff(i)).collect(),
        nearly_sorted,
        nearly_reversed,
        // Sorted with the last element moved to the front, and the first to the back.
        std::iter::once(n as u32).chain(0..n as u32 - 1).collect(),
        (1..n as u32).chain(std::iter::once(0)).collect(),
        // Two interleaved sorted sequences.
        (0..n as u32)
            .map(|i| if i % 2 == 0 { i } else { u32::MAX - i })
            .collect(),
    ];
    for v in shapes {
        assert_bucket(&v);
        let f: Vec<f32> = v.iter().map(|&x| x as f32 - 500.0).collect();
        assert_bucket_floats(&f);
    }
}

#[test]
fn sorts_long_inputs() {
    let n = size(100_000, 2_000);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let shapes: Vec<Vec<u64>> = vec![
        (0..n).map(|_| xorshift(&mut state)).collect(),
        (0..n).map(|_| xorshift(&mut state) % n as u64).collect(),
        (0..n).map(|_| xorshift(&mut state) % 7).collect(),
        // Organ pipe: ascending then descending.
        (0..n as u64).map(|i| i.min(n as u64 - 1 - i)).collect(),
        // Sorted with the last element moved to the front.
        std::iter::once(n as u64).chain(0..n as u64 - 1).collect(),
        (0..n as u64).collect(),
        (0..n as u64).rev().collect(),
    ];
    for v in shapes {
        assert_bucket(&v);
    }
    let f: Vec<f64> = (0..n).map(|_| any_f64(&mut state)).collect();
    assert_bucket_floats(&f);
    let f: Vec<f64> = (0..n).map(|_| unit_f64(&mut state)).collect();
    assert_bucket_floats(&f);
}

// ---------------------------------------------------------------------------------------------
// `bucket` on floats
// ---------------------------------------------------------------------------------------------

#[test]
fn sorts_random_floats() {
    let mut state = 0xF10A_F10A_F10A_F10A;
    for len in (0..70).chain([100, 257, 1000, size(5000, 400)]) {
        let v: Vec<f64> = (0..len).map(|_| finite_f64(&mut state)).collect();
        assert_bucket_floats(&v);
        let v: Vec<f64> = (0..len).map(|_| any_f64(&mut state)).collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len)
            .map(|_| f32::from_bits(xorshift(&mut state) as u32))
            .collect();
        assert_bucket_floats(&v);
        // Values of an ordinary size, of either sign.
        let v: Vec<f64> = (0..len)
            .map(|_| (unit_f64(&mut state) - 0.5) * 1e6)
            .collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len)
            .map(|_| (unit_f64(&mut state) as f32 - 0.25) * 100.0)
            .collect();
        assert_bucket_floats(&v);
    }
}

#[test]
fn sorts_uniform_floats_in_the_unit_interval() {
    // The textbook case: n values uniform in [0, 1), about one per bucket.
    let mut state = 0xC125_C125_C125_C125;
    for len in [10, 100, 1000, size(50_000, 2_000)] {
        let v: Vec<f64> = (0..len).map(|_| unit_f64(&mut state)).collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len).map(|_| unit_f64(&mut state) as f32).collect();
        assert_bucket_floats(&v);
    }
}

#[test]
fn sorts_special_floats_among_others() {
    let mut state = 0x5BEC_1A15_5BEC_1A15;
    let specials64 = special_f64s();
    let specials32 = special_f32s();
    for len in (0..50).chain([100, 300, size(3000, 300)]) {
        for share in [1, 2, 8] {
            // About `1 / share` of the elements special, the rest random.
            let v: Vec<f64> = (0..len)
                .map(|_| {
                    let r = xorshift(&mut state);
                    if r.is_multiple_of(share) {
                        specials64[(r >> 8) as usize % specials64.len()]
                    } else {
                        (unit_f64(&mut state) - 0.5) * 1000.0
                    }
                })
                .collect();
            assert_bucket_floats(&v);
            let v: Vec<f32> = (0..len)
                .map(|_| {
                    let r = xorshift(&mut state);
                    if r.is_multiple_of(share) {
                        specials32[(r >> 8) as usize % specials32.len()]
                    } else {
                        (unit_f64(&mut state) as f32 - 0.5) * 1000.0
                    }
                })
                .collect();
            assert_bucket_floats(&v);
        }
    }
    let mut all = special_f64s();
    assert_bucket_floats(&all);
    all.reverse();
    assert_bucket_floats(&all);
    let mut all = special_f32s();
    assert_bucket_floats(&all);
    all.reverse();
    assert_bucket_floats(&all);
}

#[test]
fn sorts_signed_zeros() {
    // Every mix of `-0.0` and `+0.0`, alone and with other values: `-0.0` comes first.
    let max_len = size(10, 5);
    for len in 0..=max_len {
        for code in 0..1u32 << len {
            let v: Vec<f64> = (0..len)
                .map(|i| if code >> i & 1 == 0 { 0.0 } else { -0.0 })
                .collect();
            assert_bucket_floats(&v);
            let w: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            assert_bucket_floats(&w);
            let mut x = v.clone();
            x.extend([1.0, -1.0, f64::MIN_POSITIVE]);
            assert_bucket_floats(&x);
        }
    }
    let mut state = 0x2E50_2E50_2E50_2E50;
    let v: Vec<f64> = (0..size(5000, 500))
        .map(|_| {
            if xorshift(&mut state).is_multiple_of(2) {
                0.0
            } else {
                -0.0
            }
        })
        .collect();
    assert_bucket_floats(&v);
}

#[test]
fn sorts_nans_by_sign_and_payload() {
    let mut state = 0x7A7A_7A7A_7A7A_7A7A;
    for len in [1, 2, 5, 33, 300, size(3000, 300)] {
        // Quiet and signalling NaNs of either sign with random payloads, some infinities.
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                match r % 5 {
                    0 => f64::INFINITY,
                    1 => f64::NEG_INFINITY,
                    _ => f64::from_bits((r & 0x800F_FFFF_FFFF_FFFF) | 0x7FF0_0000_0000_0001),
                }
            })
            .collect();
        assert_bucket_floats(&v);
        // NaNs with payloads 0 to 3 only: buckets by value see all of them as one.
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                let sign = (r & 1) << 63;
                f64::from_bits(sign | 0x7FF8_0000_0000_0000 | ((r >> 1) % 4))
            })
            .collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state) as u32;
                f32::from_bits((r & 0x803F_FFFF) | 0x7FC0_0000)
            })
            .collect();
        assert_bucket_floats(&v);
    }
}

#[test]
fn sorts_subnormal_floats() {
    let mut state = 0x50B0_50B0_50B0_50B0;
    for len in [2, 9, 100, size(3000, 300)] {
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                f64::from_bits(r & 0x800F_FFFF_FFFF_FFFF)
            })
            .collect();
        assert_bucket_floats(&v);
        // Subnormals a few ulps apart, and with the smallest normals.
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                let x = f64::from_bits(r % 8);
                match r >> 60 {
                    0 => f64::MIN_POSITIVE,
                    1 => -x,
                    _ => x,
                }
            })
            .collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len)
            .map(|_| f32::from_bits(xorshift(&mut state) as u32 & 0x807F_FFFF))
            .collect();
        assert_bucket_floats(&v);
    }
}

#[test]
fn sorts_floats_whose_range_overflows() {
    // `MAX - MIN` is infinite: the bucket width cannot be measured in the type itself.
    let mut state = 0x0F10_0F10_0F10_0F10;
    for len in [2, 3, 4, 10, 100, size(3000, 300)] {
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                match r % 4 {
                    0 => f64::MAX,
                    1 => f64::MIN,
                    2 => (unit_f64(&mut state) - 0.5) * f64::MAX,
                    _ => unit_f64(&mut state),
                }
            })
            .collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                match r % 4 {
                    0 => f32::MAX,
                    1 => f32::MIN,
                    2 => (unit_f64(&mut state) as f32 - 0.5) * f32::MAX,
                    _ => unit_f64(&mut state) as f32,
                }
            })
            .collect();
        assert_bucket_floats(&v);
    }
    assert_bucket_floats(&[f64::MAX, f64::MIN, 0.0]);
    assert_bucket_floats(&[f64::MAX, f64::MIN, f64::MAX / 2.0, f64::MIN / 2.0, 1.0]);
    assert_bucket_floats(&[f32::MAX, f32::MIN, -0.0]);
}

#[test]
fn sorts_floats_a_few_ulps_apart() {
    // A range so narrow that a bucket width computed by value is below one ulp.
    let mut state = 0x0175_0175_0175_0175;
    for centre in [
        1.0f64,
        -1.0,
        1e300,
        -1e300,
        1e-300,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        for ulps in [1u64, 2, 3, 8, 100] {
            for len in [2, 3, 20, 100, size(3000, 300)] {
                let base = centre.to_bits();
                let v: Vec<f64> = (0..len)
                    .map(|_| {
                        // Towards zero, so never past `MAX` into the infinities.
                        f64::from_bits(base - xorshift(&mut state) % (ulps + 1))
                    })
                    .collect();
                assert_bucket_floats(&v);
            }
        }
    }
    for centre in [1.0f32, -1.0, 1e30, 1e-30] {
        let base = centre.to_bits();
        for len in [2, 7, 100, size(3000, 300)] {
            let v: Vec<f32> = (0..len)
                .map(|_| f32::from_bits(base + (xorshift(&mut state) % 4) as u32))
                .collect();
            assert_bucket_floats(&v);
        }
    }
    // Across a power of two, where the ulp doubles.
    let two = 2.0f64.to_bits();
    let v: Vec<f64> = (0..size(1000, 100) as u64)
        .map(|i| f64::from_bits(two - 8 + (i * 7919) % 16))
        .collect();
    assert_bucket_floats(&v);
}

#[test]
fn sorts_floats_over_a_huge_dynamic_range() {
    // Powers of two from 2^-1000 to 2^1000: by value, all but the largest fall into the
    // first bucket.
    let mut state = 0x9E0B_9E0B_9E0B_9E0B;
    let mut v: Vec<f64> = (-1000..=1000).map(|e| 2f64.powi(e)).collect();
    shuffle(&mut v, &mut state);
    assert_bucket_floats(&v);
    let mut w: Vec<f64> = v.iter().map(|&x| -x).chain(v.iter().copied()).collect();
    shuffle(&mut w, &mut state);
    assert_bucket_floats(&w);
    // Random mantissas with random exponents.
    for len in [10, 300, size(5000, 500)] {
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let e = (xorshift(&mut state) % 2001) as i32 - 1000;
                (1.0 + unit_f64(&mut state)) * 2f64.powi(e)
            })
            .collect();
        assert_bucket_floats(&v);
        let v: Vec<f32> = (0..len)
            .map(|_| {
                let e = (xorshift(&mut state) % 250) as i32 - 125;
                (1.0 + unit_f64(&mut state) as f32) * 2f32.powi(e)
            })
            .collect();
        assert_bucket_floats(&v);
    }
    // Most values tiny, one huge.
    let mut v: Vec<f64> = (0..size(10_000, 1_000))
        .map(|_| unit_f64(&mut state) * 1e-200)
        .collect();
    v.push(1e300);
    assert_bucket_floats(&v);
}

#[test]
fn sorts_integer_valued_and_duplicate_floats() {
    let mut state = 0x1D0B_1D0B_1D0B_1D0B;
    for len in [10, 100, size(5000, 500)] {
        let v: Vec<f64> = (0..len)
            .map(|_| (xorshift(&mut state) % 10) as f64 - 5.0)
            .collect();
        assert_bucket_floats(&v);
        let mut v: Vec<f32> = (0..len)
            .map(|_| [1.5f32, 2.5, -7.0][(xorshift(&mut state) % 3) as usize])
            .collect();
        v[len / 2] = f32::MAX;
        assert_bucket_floats(&v);
    }
}

#[test]
fn sorts_sorted_and_reverse_sorted_floats() {
    let mut state = 0x5027_5027_5027_5027;
    let mut v: Vec<f64> = (0..size(5000, 500)).map(|_| any_f64(&mut state)).collect();
    v.sort_by(f64::total_cmp);
    assert_bucket_floats(&v);
    v.reverse();
    assert_bucket_floats(&v);
    // Sorted by `<` (with `-0.0 == 0.0`) but not by `total_cmp`.
    assert_bucket_floats(&[-1.0, 0.0, -0.0, 0.0, 1.0]);
    assert_bucket_floats(&[1.0, 0.0, -0.0, 0.0, -1.0]);
}

/// Sorts `v` with [`bucket`] and returns how many allocations that made.
fn allocations_of<T: BucketKey>(v: &mut [T]) -> usize {
    crate::allocations::count(|| bucket(v))
}

#[test]
fn allocates_only_beyond_4_kib() {
    // Up to 1024 elements in up to 4096 bytes are sorted with buffers on the stack.
    let mut state = 0x0A11_0C47_E5B0_FFE5;
    let mut v: Vec<u64> = (0..512).map(|_| xorshift(&mut state)).collect();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted());
    let mut v: Vec<u32> = (0..1024).map(|_| xorshift(&mut state) as u32).collect();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted());
    let mut v: Vec<u8> = (0..1024).map(|_| xorshift(&mut state) as u8).collect();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted());
    let mut v: Vec<f64> = (0..512).map(|_| any_f64(&mut state)).collect();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted_by(|a, b| a.total_cmp(b).is_le()));
    // Skewed input that recurses still needs no more than the stack area.
    let mut v: Vec<u64> = (0..511).map(|_| xorshift(&mut state) % 100).collect();
    v.push(u64::MAX);
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted());
    // Beyond 4 KiB or beyond 1024 elements.
    let mut v: Vec<u32> = (0..1025).map(|_| xorshift(&mut state) as u32).collect();
    assert!(allocations_of(&mut v) > 0);
    assert!(v.is_sorted());
    let mut v: Vec<u64> = (0..513).map(|_| xorshift(&mut state)).collect();
    assert!(allocations_of(&mut v) > 0);
    assert!(v.is_sorted());
    let mut v: Vec<u64> = (0..size(100_000, 2_000))
        .map(|_| xorshift(&mut state))
        .collect();
    assert!(allocations_of(&mut v) > 0);
    assert!(v.is_sorted());
}

#[test]
fn sorted_and_reverse_sorted_input_does_not_allocate() {
    // Recognised before any buffer is set up.
    let mut v: Vec<u32> = (0..10_000).collect();
    assert_eq!(allocations_of(&mut v), 0);
    v.reverse();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted());
    let mut v: Vec<u32> = (0..10_000).map(|i| i / 3).rev().collect();
    assert_eq!(allocations_of(&mut v), 0);
    assert!(v.is_sorted());
}

// ---------------------------------------------------------------------------------------------
// `bucket_by_key`
// ---------------------------------------------------------------------------------------------

#[test]
fn by_key_sorts_short_slices() {
    assert_bucket_by_key::<u32>(&[]);
    assert_bucket_by_key(&[1u32]);
    assert_bucket_by_key(&[2u32, 1]);
    assert_bucket_by_key(&[1u32, 2]);
    assert_bucket_by_key(&[1u32, 1]);
    assert_bucket_by_float_key::<f64>(&[]);
    assert_bucket_by_float_key(&[0.0f64, -0.0]);
    assert_bucket_by_float_key(&[f64::NAN, f64::NAN]);
}

#[test]
fn by_key_is_stable() {
    let mut state = 0x57AB_1E57_AB1E_57AB;
    for len in 0..size(200, 50) {
        for span in [1, 2, 5, 50, 1 << 20] {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % span) as u32)
                .collect();
            assert_bucket_by_key(&keys);
        }
    }
}

#[test]
fn by_key_is_stable_on_long_inputs_with_many_equal_keys() {
    // Equal keys spread over every level of recursion: few values, few values plus a far
    // outlier, and clusters far apart.
    let mut state = 0x57AB_57AB_57AB_57AB;
    let n = size(50_000, 2_000);
    let mut r = || xorshift(&mut state);
    for span in [2u64, 3, 17, 1000] {
        let keys: Vec<u64> = (0..n).map(|_| r() % span).collect();
        assert_bucket_by_key(&keys);
        let mut keys = keys;
        keys[n / 3] = u64::MAX;
        assert_bucket_by_key(&keys);
    }
    let keys: Vec<i64> = (0..n)
        .map(|_| [i64::MIN, -1, 0, 1 << 50, i64::MAX][(r() % 5) as usize])
        .collect();
    assert_bucket_by_key(&keys);
    let keys: Vec<u32> = (0..n).map(|_| 1 << (r() % 32)).collect();
    assert_bucket_by_key(&keys);
}

#[test]
fn by_key_is_stable_with_float_keys() {
    // Keys are equal only if their bits are: `-0.0` and `+0.0`, and NaNs with different
    // payloads, are different keys, but the same NaN twice is one key.
    let mut state = 0xF57A_F57A_F57A_F57A;
    let specials = special_f64s();
    for len in (0..60).chain([200, size(5000, 500)]) {
        let keys: Vec<f64> = (0..len)
            .map(|_| specials[(xorshift(&mut state) % specials.len() as u64) as usize])
            .collect();
        assert_bucket_by_float_key(&keys);
        let keys: Vec<f64> = (0..len)
            .map(|_| (xorshift(&mut state) % 4) as f64 * 0.25)
            .collect();
        assert_bucket_by_float_key(&keys);
        let keys: Vec<f32> = (0..len)
            .map(|_| [0.0f32, -0.0, f32::NAN, -f32::NAN, 1.0][(xorshift(&mut state) % 5) as usize])
            .collect();
        assert_bucket_by_float_key(&keys);
        let keys: Vec<f64> = (0..len).map(|_| unit_f64(&mut state)).collect();
        assert_bucket_by_float_key(&keys);
    }
    // Many equal keys and a far outlier, with recursion.
    let mut keys: Vec<f64> = (0..size(30_000, 2_000))
        .map(|_| (xorshift(&mut state) % 5) as f64 * 1e-10)
        .collect();
    keys[7] = 1e300;
    assert_bucket_by_float_key(&keys);
}

#[test]
fn by_key_leaves_sorted_input_and_reverses_only_strictly_descending_input() {
    assert_bucket_by_key(&[0u32, 0, 1, 1, 2, 2]);
    assert_bucket_by_key(&[5u32, 4, 3, 2, 1, 0]);
    // Not strictly descending: reversing it would swap the equal keys.
    assert_bucket_by_key(&[5u32, 4, 4, 3, 2, 1, 0]);
    assert_bucket_by_key(&[2u8, 2, 1]);
    assert_bucket_by_float_key(&[2.0f64, 1.0, 0.0, -0.0, -1.0]);
    assert_bucket_by_float_key(&[2.0f64, 1.0, 1.0, 0.0]);
    let long: Vec<u64> = (0..size(10_000, 2_000) as u64).rev().collect();
    assert_bucket_by_key(&long);
    let long: Vec<u64> = (0..size(10_000, 2_000) as u64)
        .map(|i| i / 2)
        .rev()
        .collect();
    assert_bucket_by_key(&long);
    let long: Vec<u64> = (0..size(10_000, 2_000) as u64).map(|i| i / 2).collect();
    assert_bucket_by_key(&long);
}

#[test]
fn by_key_sorts_every_sequence_over_few_keys() {
    let max_len = size(7, 4);
    for len in 0..=max_len {
        let total = 3usize.pow(len as u32);
        for code in 0..total {
            let keys: Vec<i16> = (0..len)
                .map(|i| (code / 3usize.pow(i as u32) % 3) as i16 * 4000 - 4000)
                .collect();
            assert_bucket_by_key(&keys);
            let floats: Vec<f32> = keys
                .iter()
                .map(|&k| [-0.0, 0.0, f32::NAN][(k / 4000 + 1) as usize])
                .collect();
            assert_bucket_by_float_key(&floats);
        }
    }
}

#[test]
fn by_key_sorts_every_key_type() {
    let mut state = 0xB1CE_B1CE_B1CE_B1CE;
    for len in [0, 1, 2, 9, 63, 64, 100, size(5000, 500)] {
        let mut r = || xorshift(&mut state);
        let u8s: Vec<u8> = (0..len).map(|_| r() as u8).collect();
        assert_bucket_by_key(&u8s);
        let i8s: Vec<i8> = (0..len).map(|_| r() as i8).collect();
        assert_bucket_by_key(&i8s);
        let u16s: Vec<u16> = (0..len).map(|_| r() as u16).collect();
        assert_bucket_by_key(&u16s);
        let i16s: Vec<i16> = (0..len).map(|_| r() as i16).collect();
        assert_bucket_by_key(&i16s);
        let bools: Vec<bool> = (0..len).map(|_| r() % 3 == 0).collect();
        assert_bucket_by_key(&bools);
        let u32s: Vec<u32> = (0..len).map(|_| r() as u32).collect();
        assert_bucket_by_key(&u32s);
        let i32s: Vec<i32> = (0..len).map(|_| r() as i32).collect();
        assert_bucket_by_key(&i32s);
        let u64s: Vec<u64> = (0..len).map(|_| r()).collect();
        assert_bucket_by_key(&u64s);
        let i64s: Vec<i64> = (0..len).map(|_| r() as i64).collect();
        assert_bucket_by_key(&i64s);
        let u128s: Vec<u128> = (0..len)
            .map(|_| (u128::from(r()) << 64) | u128::from(r()))
            .collect();
        assert_bucket_by_key(&u128s);
        let i128s: Vec<i128> = (0..len)
            .map(|_| ((u128::from(r()) << 64) | u128::from(r())) as i128)
            .collect();
        assert_bucket_by_key(&i128s);
        let usizes: Vec<usize> = (0..len).map(|_| r() as usize).collect();
        assert_bucket_by_key(&usizes);
        let isizes: Vec<isize> = (0..len).map(|_| r() as isize % 1000).collect();
        assert_bucket_by_key(&isizes);
        let chars: Vec<char> = (0..len)
            .map(|_| char::from_u32((r() % 0x11_0000) as u32).unwrap_or('\u{E000}'))
            .collect();
        assert_bucket_by_key(&chars);
        let f64s: Vec<f64> = (0..len).map(|_| f64::from_bits(r())).collect();
        assert_bucket_by_float_key(&f64s);
        let f32s: Vec<f32> = (0..len).map(|_| f32::from_bits(r() as u32)).collect();
        assert_bucket_by_float_key(&f32s);
    }
}

#[test]
fn by_key_sorts_zero_sized_elements() {
    for len in (0..40).chain([1000, size(100_000, 2_000)]) {
        let mut v = vec![(); len];
        bucket_by_key(&mut v, |_| 0u8);
        assert_eq!(v.len(), len);
        let calls = Cell::new(0u32);
        bucket_by_key(&mut v, |_| {
            calls.set(calls.get() + 1);
            calls.get()
        });
        assert_eq!(v.len(), len);
        bucket_by_key(&mut v, |_| f64::NAN);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn by_key_sorts_large_elements() {
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [10, 200, size(3000, 300)] {
        let mut v: Vec<[u64; 9]> = (0..len)
            .map(|i| {
                let mut e = [i as u64; 9];
                e[0] = xorshift(&mut state) % 5000;
                e[8] = xorshift(&mut state);
                e
            })
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|e| e[0]);
        bucket_by_key(&mut v, |e| e[0]);
        assert_eq!(v, expected);
        // By a float key made from the payload.
        let mut expected = v.clone();
        expected.sort_by(|a, b| (a[8] as f64).total_cmp(&(b[8] as f64)));
        bucket_by_key(&mut v, |e| e[8] as f64);
        assert_eq!(v, expected);
    }
    for len in [100, size(5000, 200)] {
        let mut v: Vec<[u64; 32]> = (0..len)
            .map(|i| {
                let mut e = [i as u64; 32];
                e[0] = xorshift(&mut state) % 50;
                e
            })
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|e| e[0]);
        bucket_by_key(&mut v, |e| e[0]);
        assert!(v == expected);
    }
}

#[test]
fn by_key_sorts_owning_elements() {
    let mut state = 0x1EAF_5EED_1EAF_5EED;
    for len in [3, 50, size(2000, 200)] {
        let mut v: Vec<String> = (0..len)
            .map(|_| format!("{:0w$}", xorshift(&mut state) % 1000, w = 3))
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|s| s.parse::<u32>().unwrap() % 7);
        bucket_by_key(&mut v, |s| s.parse::<u32>().unwrap() % 7);
        assert_eq!(v, expected);

        let mut v: Vec<Vec<u8>> = (0..len)
            .map(|_| vec![0; (xorshift(&mut state) % 20) as usize])
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|x| x.len());
        bucket_by_key(&mut v, |x| x.len());
        assert_eq!(v, expected);
    }
}

#[test]
fn by_key_moves_rcs_without_cloning_or_dropping_them() {
    let mut state = 0x0BC5_0BC5_0BC5_0BC5;
    for len in [2, 31, 300, size(5000, 400)] {
        let originals: Vec<Rc<f64>> = (0..len)
            .map(|_| Rc::new((xorshift(&mut state) % 100) as f64 - 50.0))
            .collect();
        let mut v: Vec<Rc<f64>> = originals.iter().map(Rc::clone).collect();
        bucket_by_key(&mut v, |x| **x);
        assert!(originals.iter().all(|x| Rc::strong_count(x) == 2));
        let mut expected: Vec<Rc<f64>> = originals.iter().map(Rc::clone).collect();
        expected.sort_by(|a, b| a.total_cmp(b));
        assert!(
            v.iter().zip(&expected).all(|(a, b)| Rc::ptr_eq(a, b)),
            "not stable or not a permutation"
        );
        drop(v);
        drop(expected);
        assert!(originals.iter().all(|x| Rc::strong_count(x) == 1));
    }
}

#[test]
fn by_key_calls_key_once_per_element_and_places_its_latest_state() {
    let mut state = 0xCA11_CA11_CA11_CA11;
    let random: Vec<u32> = (0..size(3000, 300))
        .map(|_| xorshift(&mut state) as u32)
        .collect();
    let mut skewed: Vec<u32> = (0..size(3000, 300) as u32).map(|i| i % 50).collect();
    skewed[5] = u32::MAX;
    let layouts: Vec<Vec<u32>> = vec![
        (0..2).collect(),
        vec![1, 0],
        (0..9).map(|i| (i * 7919) % 1013).collect(),
        (0..100).map(|i| (i * 7919) % 1013).collect(),
        random.clone(),
        skewed,
        (0..size(3000, 300) as u32).collect(),
        (0..size(3000, 300) as u32).rev().collect(),
        (0..size(3000, 300) as u32).map(|i| i / 4).rev().collect(),
        vec![7; size(3000, 300)],
    ];
    for keys in layouts {
        let mut v: Vec<(u32, Cell<u32>)> = keys.iter().map(|&k| (k, Cell::new(0))).collect();
        bucket_by_key(&mut v, |e| {
            e.1.set(e.1.get() + 1);
            e.0
        });
        assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
        assert!(v.iter().all(|e| e.1.get() == 1), "not one call per element");
        // With float keys.
        let mut v: Vec<(u32, Cell<u32>)> = keys.iter().map(|&k| (k, Cell::new(0))).collect();
        bucket_by_key(&mut v, |e| {
            e.1.set(e.1.get() + 1);
            f64::from(e.0) * -0.5
        });
        assert!(v.windows(2).all(|w| w[0].0 >= w[1].0));
        assert!(v.iter().all(|e| e.1.get() == 1), "not one call per element");
    }
}

#[test]
fn by_key_sorts_by_the_keys_a_stateful_key_returned() {
    // A key that returns a different value each time it is asked: the slice is sorted, stably,
    // by the one value each element was given.
    let mut state = 0x57A7_E57A_7E57_A7E5;
    for len in [0, 1, 2, 17, 300, size(5000, 500)] {
        for span in [3u64, 1000, u64::MAX] {
            let rng = Cell::new(xorshift(&mut state));
            let given: Vec<Cell<Option<u64>>> = (0..len).map(|_| Cell::new(None)).collect();
            let mut v: Vec<usize> = (0..len).collect();
            bucket_by_key(&mut v, |&id| {
                let mut s = rng.get();
                let k = xorshift(&mut s) % span;
                rng.set(s);
                assert!(given[id].replace(Some(k)).is_none(), "key asked twice");
                k
            });
            // A single element is not asked for its key: there is nothing to sort.
            let keys: Vec<u64> = given
                .iter()
                .map(|g| {
                    g.get().unwrap_or_else(|| {
                        assert!(len < 2, "key never asked");
                        0
                    })
                })
                .collect();
            let mut expected: Vec<usize> = (0..len).collect();
            expected.sort_by_key(|&id| keys[id]);
            assert_eq!(v, expected);
        }
    }
    // A counter: each element gets a smaller key than the one asked before it.
    let calls = Cell::new(0u32);
    let given: Vec<Cell<Option<u32>>> = (0..size(1000, 100)).map(|_| Cell::new(None)).collect();
    let mut v: Vec<usize> = (0..given.len()).collect();
    bucket_by_key(&mut v, |&id| {
        calls.set(calls.get() + 1);
        let k = u32::MAX - calls.get();
        assert!(given[id].replace(Some(k)).is_none(), "key asked twice");
        k
    });
    let keys: Vec<u32> = given.iter().map(|g| g.get().unwrap()).collect();
    let mut expected: Vec<usize> = (0..given.len()).collect();
    expected.sort_by_key(|&id| keys[id]);
    assert_eq!(v, expected);
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
        for span in [2u32, 300, u32::MAX] {
            let keys: Vec<u32> = (0..len)
                .map(|_| xorshift(&mut state) as u32 % span)
                .collect();
            let mut expected: Vec<usize> = (0..len).collect();
            expected.sort_by_key(|&id| keys[id]);
            let (order, completed) = sort_tracked(&keys, |v| bucket_by_key(v, |t| t.key));
            assert!(completed);
            assert_eq!(order, expected);
            // Spread over 62 bits, with recursion through sparse buckets.
            let (order, completed) = sort_tracked(&keys, |v| {
                bucket_by_key(v, |t| u64::from(t.key) << 30 | u64::from(t.key))
            });
            assert!(completed);
            assert_eq!(order, expected);
            // As floats.
            let (order, completed) =
                sort_tracked(&keys, |v| bucket_by_key(v, |t| f64::from(t.key) * 1e-3));
            assert!(completed);
            assert_eq!(order, expected);
        }
    }
}

#[test]
fn by_key_panicking_key_leaves_the_slice_as_it_was() {
    let len = size(150, 40);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32).collect();
    let mut skewed: Vec<u32> = (0..len as u32).map(|i| i % 5).collect();
    skewed[len / 2] = u32::MAX;
    let layouts: Vec<Vec<u32>> = vec![
        random,
        skewed,
        (0..len as u32).map(|i| i % 3).collect(),
        (0..len as u32).collect(),
        (0..len as u32).rev().collect(),
    ];
    let step = if cfg!(miri) { 7 } else { 1 };
    for keys in layouts {
        for budget in (0..keys.len()).step_by(step) {
            let calls = Cell::new(0);
            let (order, completed) = sort_tracked(&keys, |v| {
                bucket_by_key(v, |t| {
                    if calls.get() == budget {
                        panic!("key budget exhausted");
                    }
                    calls.set(calls.get() + 1);
                    t.key
                })
            });
            assert!(!completed, "budget {budget}");
            assert_eq!(order, (0..keys.len()).collect::<Vec<_>>(), "slice changed");

            let calls = Cell::new(0);
            let (order, completed) = sort_tracked(&keys, |v| {
                bucket_by_key(v, |t| {
                    if calls.get() == budget {
                        panic!("key budget exhausted");
                    }
                    calls.set(calls.get() + 1);
                    f32::from_bits(t.key)
                })
            });
            assert!(!completed, "budget {budget}");
            assert_eq!(order, (0..keys.len()).collect::<Vec<_>>(), "slice changed");
        }
    }
    // Beyond 4 KiB, where the buffers are allocated.
    let keys: Vec<u32> = (0..size(3000, 1100) as u32)
        .map(|i| i.wrapping_mul(2_654_435_761))
        .collect();
    for budget in [0, 1, keys.len() / 2, keys.len() - 1] {
        let calls = Cell::new(0);
        let (order, completed) = sort_tracked(&keys, |v| {
            bucket_by_key(v, |t| {
                if calls.get() == budget {
                    panic!("key budget exhausted");
                }
                calls.set(calls.get() + 1);
                t.key
            })
        });
        assert!(!completed, "budget {budget}");
        assert_eq!(order, (0..keys.len()).collect::<Vec<_>>(), "slice changed");
    }
}

/// Sorts `v` with [`bucket_by_key`] by its first field and returns how many allocations that
/// made.
fn allocations_by_key_of(v: &mut [(u32, u32)]) -> usize {
    crate::allocations::count(|| bucket_by_key(v, |e| e.0))
}

#[test]
fn by_key_allocates_only_beyond_4_kib() {
    // 256 elements of 8 bytes with 4-byte keys fit the stack area.
    let mut state = 0xA110_CA7E_A110_CA7E;
    let mut v: Vec<(u32, u32)> = (0..256).map(|i| (xorshift(&mut state) as u32, i)).collect();
    assert_eq!(allocations_by_key_of(&mut v), 0);
    assert!(v.is_sorted_by_key(|e| e.0));
    let mut v: Vec<(u32, u32)> = (0..256).map(|i| ((i * 7) % 5, i)).collect();
    assert_eq!(allocations_by_key_of(&mut v), 0);
    assert!(v.is_sorted());
    let mut v: Vec<(u32, u32)> = (0..size(100_000, 2_000) as u32)
        .map(|i| (xorshift(&mut state) as u32, i))
        .collect();
    assert!(allocations_by_key_of(&mut v) > 0);
    assert!(v.is_sorted_by_key(|e| e.0));
}

// ---------------------------------------------------------------------------------------------
// Paths chosen by the size of a level and of its elements
// ---------------------------------------------------------------------------------------------

/// Random keys, keys over a range of a third of the length, and three distinct keys.
fn spread_keys(len: usize, state: &mut u64) -> [Vec<u32>; 3] {
    let third = (len as u64 / 3).max(1);
    [
        (0..len).map(|_| xorshift(state) as u32).collect(),
        (0..len).map(|_| (xorshift(state) % third) as u32).collect(),
        (0..len).map(|_| (xorshift(state) % 3) as u32).collect(),
    ]
}

/// Keys in clusters of about ten, far apart, each cluster of eight values in random order: a
/// level of up to 4096 elements gives each cluster a bucket of its own, whose keys the insertion
/// sort then moves one or several places.
fn clustered_keys(len: usize, state: &mut u64) -> Vec<u32> {
    let clusters = (len as u64 / 10).max(1);
    (0..len)
        .map(|_| {
            let r = xorshift(state);
            ((r % clusters) * 10_000 + (r >> 32) % 8) as u32
        })
        .collect()
}

#[test]
fn sorts_across_the_level_size_thresholds() {
    // A level keeps the bucket of each key for up to 2048 keys, counts into four tables from
    // four keys per bucket (16 384 keys for 4096 buckets) and into eight from 16 per bucket.
    let mut state = 0x7B5E_7B5E_7B5E_7B5E;
    let lens: Vec<usize> = if cfg!(miri) {
        vec![2048, 2049]
    } else {
        vec![2047, 2048, 2049, 16_383, 16_384, 65_535, 65_536]
    };
    for len in lens {
        for keys in spread_keys(len, &mut state) {
            assert_bucket(&keys);
            assert_bucket_by_key(&keys);
            let floats: Vec<f32> = keys.iter().map(|&k| k as f32 * 0.25 - 100.0).collect();
            assert_bucket_floats(&floats);
            assert_bucket_by_float_key(&floats);
        }
        let v: Vec<f64> = (0..len).map(|_| unit_f64(&mut state)).collect();
        assert_bucket_floats(&v);
        // Few distinct keys in a range smaller than the buckets: one bucket per key, counted
        // into several tables.
        let keys: Vec<u16> = (0..len)
            .map(|_| (xorshift(&mut state) % 40) as u16)
            .collect();
        assert_bucket(&keys);
        assert_bucket_by_key(&keys);
    }
}

#[test]
fn sorts_floats_crowded_at_zero_with_the_rest_far_from_it() {
    // Most keys zero and the others far from it on both sides: the buckets by value around zero
    // hold the crowd, and the keys are counted again by bits, which leaves the zeros in one
    // bucket all the same, so the cut by value stays and the keys move by it.
    let mut state = 0xC20D_C20D_C20D_C20D;
    for len in [17, 100, 1000, 2048, 2049, size(5000, 500)] {
        for zeros in [[0.0, 0.0], [0.0, -0.0], [-0.0, -0.0]] {
            let v: Vec<f64> = (0..len)
                .map(|_| {
                    let r = xorshift(&mut state);
                    if r % 10 < 6 {
                        zeros[(r >> 8) as usize % 2]
                    } else {
                        let x = 500.0 + unit_f64(&mut state) * 500.0;
                        if r & (1 << 20) == 0 { x } else { -x }
                    }
                })
                .collect();
            assert_bucket_floats(&v);
            assert_bucket_by_float_key(&v);
            let w: Vec<f32> = v.iter().map(|&x| x as f32).collect();
            assert_bucket_floats(&w);
            assert_bucket_by_float_key(&w);
            // On one side of zero only, with the zeros at the end of the range.
            let w: Vec<f64> = v.iter().map(|&x| x.abs()).collect();
            assert_bucket_floats(&w);
            assert_bucket_by_float_key(&w);
        }
        // A crowd of tiny values of either sign rather than zeros, which their bits spread.
        let v: Vec<f64> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                let x = if r % 10 < 6 {
                    unit_f64(&mut state) * 1e-12
                } else {
                    500.0 + unit_f64(&mut state) * 500.0
                };
                if r & (1 << 20) == 0 { x } else { -x }
            })
            .collect();
        assert_bucket_floats(&v);
        assert_bucket_by_float_key(&v);
    }
}

thread_local! {
    /// How many times the [`Owned`] element with each id has been dropped on this thread.
    static DROPS: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/// The padding of an [`Owned`], filled from its id, so that an element pieced together from
/// two others is noticed.
trait Pad: PartialEq {
    fn of(id: u32) -> Self;
}

impl<const N: usize> Pad for [u32; N] {
    fn of(id: u32) -> Self {
        std::array::from_fn(|i| id.wrapping_mul(0x9E37_79B9) ^ i as u32)
    }
}

impl<const N: usize> Pad for [u64; N] {
    fn of(id: u32) -> Self {
        std::array::from_fn(|i| u64::from(id).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64)
    }
}

/// Zero-sized markers that raise the alignment of an [`Owned`] above that of its fields.
#[repr(align(32))]
struct Align32;
#[repr(align(64))]
struct Align64;
#[repr(align(128))]
struct Align128;

/// An element with a key, an id, padding `P` and the alignment of `A`, that counts its drops in
/// [`DROPS`]. Its size and alignment choose how bucket sort moves it: without a branch up to 16
/// bytes, by its position past that, along the cycles of the permutation from 64 bytes.
struct Owned<P, A = ()> {
    key: u32,
    id: u32,
    pad: P,
    _align: [A; 0],
}

impl<P, A> Drop for Owned<P, A> {
    fn drop(&mut self) {
        DROPS.with(|d| d.borrow_mut()[self.id as usize] += 1);
    }
}

/// Sorts elements of type `Owned<P, A>` with `keys`, from position `skip` on, with
/// [`bucket_by_key`] by `key` of their key, and checks that they end up in the order of the
/// standard library's stable sort by `order`, that the first `skip` stay where they are, that
/// none is changed or dropped during the sort, and that each is dropped exactly once afterwards.
fn assert_owned_by<P: Pad, A, K: BucketKey>(
    keys: &[u32],
    skip: usize,
    key: impl Fn(u32) -> K,
    order: impl Fn(&K, &K) -> Ordering,
) {
    let len = keys.len();
    let skip = skip.min(len);
    DROPS.with(|d| *d.borrow_mut() = vec![0; len]);
    let mut v: Vec<Owned<P, A>> = keys
        .iter()
        .zip(0..)
        .map(|(&key, id)| Owned {
            key,
            id,
            pad: P::of(id),
            _align: [],
        })
        .collect();
    let calls = Cell::new(0);
    bucket_by_key(&mut v[skip..], |e| {
        calls.set(calls.get() + 1);
        key(e.key)
    });
    let mut expected: Vec<u32> = (0..len as u32).collect();
    expected[skip..].sort_by(|&a, &b| order(&key(keys[a as usize]), &key(keys[b as usize])));
    let ids: Vec<u32> = v.iter().map(|e| e.id).collect();
    assert!(
        ids == expected,
        "not stable or not a permutation: {} bytes, keys {}",
        mem::size_of::<Owned<P, A>>(),
        describe(keys),
    );
    assert!(
        v.iter()
            .all(|e| e.key == keys[e.id as usize] && e.pad == P::of(e.id)),
        "element changed"
    );
    assert!(
        len - skip < 2 || calls.get() == len - skip,
        "not one call per element"
    );
    assert!(
        DROPS.with(|d| d.borrow().iter().all(|&n| n == 0)),
        "element dropped during sort"
    );
    drop(v);
    assert!(
        DROPS.with(|d| d.borrow().iter().all(|&n| n == 1)),
        "element leaked or double-dropped"
    );
}

/// [`assert_owned_by`] by the key itself.
fn assert_owned<P: Pad, A>(keys: &[u32], skip: usize) {
    assert_owned_by::<P, A, u32>(keys, skip, |k| k, Ord::cmp);
}

/// [`assert_owned_by`] by a float made from the key, in the same order: `-0.0` for 2, between
/// the negative values below it and the positive ones above it.
fn assert_owned_by_float<P: Pad, A>(keys: &[u32], skip: usize) {
    let float = |k: u32| if k == 2 { -0.0 } else { f64::from(k) - 2.5 };
    assert_owned_by::<P, A, f64>(keys, skip, float, f64::total_cmp);
}

/// Key layouts of `len` keys that end the order the keys are read in at different places:
/// each is sorted, strictly descending or non-increasing, or turns out not to be at the first,
/// the second or a later key.
fn order_layouts(len: usize) -> Vec<Vec<u32>> {
    let n = len as u32;
    let mut layouts = vec![
        // Non-increasing, the first three keys equal.
        (0..n).map(|i| (3 * n + 2 - i) / 3).collect::<Vec<_>>(),
        // Non-increasing, the first two keys different.
        (0..n)
            .map(|i| if i == 0 { 2 * n + 9 } else { (n - i) / 2 })
            .collect(),
        // Strictly descending.
        (0..n).rev().collect(),
        // Sorted with duplicates, and all equal.
        (0..n).map(|i| i / 3).collect(),
        vec![7; len],
        // Sorted, but for the last key.
        (1..=n).map(|i| if i == n { 0 } else { i }).collect(),
    ];
    if len >= 3 {
        // Equal keys, then a descent, then larger keys.
        let mut v: Vec<u32> = (0..n).map(|i| (3 * n + 2 - i) / 3).collect();
        v[len - 1] = 3 * n;
        layouts.push(v);
        // A strict descent, then equal keys, then a larger one.
        let mut v: Vec<u32> = (0..n)
            .map(|i| if i == 0 { 2 * n + 9 } else { (n - i) / 2 })
            .collect();
        v[len - 1] = 3 * n;
        layouts.push(v);
        // A strict descent, then an ascent.
        let mut v: Vec<u32> = (0..n).rev().collect();
        v[2] = n + 1;
        layouts.push(v);
    }
    layouts
}

#[test]
fn by_key_recognises_the_order_of_elements_over_16_bytes() {
    // Elements over 16 bytes have their keys read by loops that leave at the first key that
    // ends an order; non-increasing keys are reversed and their runs of equal keys reversed
    // back.
    for len in [2, 3, 4, 5, 16, 17, 40, 300, size(3000, 300)] {
        for keys in order_layouts(len) {
            assert_owned::<[u64; 2], ()>(&keys, 0);
            assert_owned_by_float::<[u64; 2], ()>(&keys, 0);
            assert_owned::<[u64; 7], ()>(&keys, 0);
            // The same layouts of 16-byte elements, read by the loop that tracks both orders.
            assert_owned::<[u32; 2], ()>(&keys, 0);
        }
    }
}

#[test]
fn by_key_gathers_elements_over_16_bytes_by_position() {
    // Elements over 16 bytes out of order often enough are not moved by insertion but their
    // positions are, and each element is then moved once: if the room they are moved to is
    // aligned for positions. A slice of 20-byte elements from its second one is aligned for
    // them only every other element.
    let mut state = 0x6A7E_6A7E_6A7E_6A7E;
    for len in [17, 100, 257, 300, size(3000, 300)] {
        let mut layouts = vec![clustered_keys(len, &mut state)];
        layouts.extend(spread_keys(len, &mut state));
        for keys in &layouts {
            for skip in [0, 1] {
                assert_owned::<[u32; 3], ()>(keys, skip);
                assert_owned::<[u64; 2], ()>(keys, skip);
                assert_owned::<[u32; 2], Align32>(keys, skip);
                assert_owned::<[u64; 5], ()>(keys, skip);
                assert_owned_by_float::<[u64; 2], ()>(keys, skip);
            }
        }
    }
}

#[test]
fn by_key_moves_owning_elements_of_up_to_16_bytes_stably_in_every_insertion_mode() {
    // Elements of up to 16 bytes are moved by insertion with a branch, or one or two places
    // without one, as the keys before them called for: in order, clustered or spread.
    let mut state = 0x1A5E_1A5E_1A5E_1A5E;
    for len in [17, 33, 100, size(3000, 300)] {
        let mut nearly: Vec<u32> = (0..len as u32).map(|i| i * 3).collect();
        for _ in 0..len / 20 + 1 {
            let i = (xorshift(&mut state) % len as u64) as usize;
            nearly[i] = xorshift(&mut state) as u32 % (3 * len as u32);
        }
        // Blocks of clustered keys and of keys in order, one after the other.
        let blocks: Vec<u32> = clustered_keys(len, &mut state)
            .into_iter()
            .enumerate()
            .map(|(i, k)| if i / 64 % 2 == 0 { k } else { i as u32 * 1000 })
            .collect();
        let mut layouts = vec![clustered_keys(len, &mut state), nearly, blocks];
        layouts.extend(spread_keys(len, &mut state));
        for keys in &layouts {
            assert_owned::<[u32; 0], ()>(keys, 0);
            assert_owned::<[u32; 1], ()>(keys, 0);
            assert_owned::<[u32; 2], ()>(keys, 0);
            assert_owned_by_float::<[u32; 0], ()>(keys, 0);
        }
    }
}

#[test]
fn by_key_walks_the_cycles_of_large_elements() {
    // A slice of at least 64 KiB of elements of at least 64 bytes, of at most 8192 elements,
    // is sorted by the indices of its elements, which are then moved along the cycles of that
    // permutation.
    let mut state = 0xC7C1_C7C1_C7C1_C7C1;
    let (lens64, lens128): (Vec<usize>, Vec<usize>) = if cfg!(miri) {
        (vec![1024], vec![512])
    } else {
        (vec![17, 1023, 1024, 3000, 8192, 8193], vec![511, 512, 4000])
    };
    for len in lens64 {
        let mut layouts = vec![clustered_keys(len, &mut state)];
        layouts.extend(spread_keys(len, &mut state));
        for keys in &layouts {
            assert_owned::<[u64; 7], ()>(keys, 0);
            assert_owned::<[u32; 2], Align64>(keys, 0);
            assert_owned_by_float::<[u64; 7], ()>(keys, 0);
        }
        // Float keys mostly `-0.0`, the others far from it on both sides, so that the indices
        // are counted by value and by bits, and moved by value.
        let keys: Vec<u32> = (0..len)
            .map(|_| {
                let r = xorshift(&mut state);
                match r % 10 {
                    0..6 => 2,
                    6 => 0,
                    _ => 1000 + (r >> 8) as u32 % 1000,
                }
            })
            .collect();
        assert_owned_by_float::<[u64; 7], ()>(&keys, 0);
        let keys: Vec<u32> = keys.iter().map(|&k| k.min(1000)).collect();
        assert_owned_by_float::<[u32; 2], Align64>(&keys, 0);
    }
    for len in lens128 {
        for keys in &spread_keys(len, &mut state) {
            assert_owned::<[u32; 2], Align128>(keys, 0);
            assert_owned::<[u32; 2], Align128>(keys, 1);
        }
    }
}

#[test]
fn by_key_walks_the_cycles_of_large_elements_in_a_deeper_level() {
    // More than 8192 elements are moved with their keys at first, but a bucket two levels down
    // of 64 KiB and at most 8192 elements is sorted along cycles again. Four clusters of 50
    // keys lie in the first bucket of the first level, as one key is far above them, and each
    // in a bucket of its own in the second.
    let mut state = 0xDEE9_DEE9_DEE9_DEE9;
    let len = 9000;
    let mut keys: Vec<u32> = (0..len)
        .map(|_| {
            let r = xorshift(&mut state);
            (r % 4) as u32 * 200_000 + (r >> 32) as u32 % 50
        })
        .collect();
    keys[len / 3] = u32::MAX;
    assert_owned::<[u64; 7], ()>(&keys, 0);
    if !cfg!(miri) {
        assert_owned::<[u32; 2], Align64>(&keys, 0);
        assert_owned_by_float::<[u64; 7], ()>(&keys, 0);
    }
}

// ---------------------------------------------------------------------------------------------
// Elements of odd sizes and alignments
// ---------------------------------------------------------------------------------------------

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

/// Sorts elements of type `Bytes<N, A>` from position `skip` on with [`bucket_by_key`], the
/// element with id `i` by `keys[i]`, and checks that they end up in the order of the standard
/// library's stable sort by `order`, unchanged, and that each is dropped exactly once, after the
/// sort.
fn assert_bytes<const N: usize, A, K: BucketKey>(
    keys: &[K],
    skip: usize,
    order: impl Fn(&K, &K) -> Ordering,
) {
    let len = keys.len();
    DROPS.with(|d| *d.borrow_mut() = vec![0; len]);
    let mut v: Vec<Bytes<N, A>> = (0..len as u32).map(Bytes::new).collect();
    let mut expected: Vec<u32> = (0..len as u32).collect();
    expected[skip..].sort_by(|&a, &b| order(&keys[a as usize], &keys[b as usize]));
    bucket_by_key(&mut v[skip..], |e| keys[e.id() as usize]);
    assert!(
        DROPS.with(|d| d.borrow().iter().all(|&n| n == 0)),
        "element dropped during sort"
    );
    let ids: Vec<u32> = v.iter().map(Bytes::id).collect();
    assert_eq!(ids, expected, "{N}-byte elements, skip {skip}");
    assert!(
        v.iter().all(|e| e.bytes == Bytes::<N, A>::of(e.id())),
        "element changed"
    );
    drop(v);
    assert!(
        DROPS.with(|d| d.borrow().iter().all(|&n| n == 1)),
        "element leaked or double-dropped"
    );
}

#[repr(align(8))]
struct Align8;

/// Mostly tiny values or `-0.0`, the others near `far`.
fn crowded_near_zero(len: usize, state: &mut u64, far: f64) -> Vec<f64> {
    (0..len)
        .map(|_| {
            let r = xorshift(state);
            if r % 10 < 6 {
                if r.is_multiple_of(3) {
                    -0.0
                } else {
                    (r >> 20) as f64 * 1e-300
                }
            } else {
                far * (unit_f64(state) + 0.5)
            }
        })
        .collect()
}

#[test]
fn by_key_moves_owning_elements_of_keys_crowded_at_zero() {
    // A level cut by value whose keys crowd at zero is counted again by bits, and moved by
    // whichever cut is better, with its buckets kept (at most 2048 keys) or not.
    let mut state = 0x1234_5678_9ABC_DEF1;
    for len in [300, size(2100, 300)] {
        for far in [1e6, -1e6, 1e300] {
            let keys = crowded_near_zero(len, &mut state, far);
            assert_bytes::<24, Align8, f64>(&keys, 0, f64::total_cmp);
            assert_bytes::<20, u8, f64>(&keys, 1, f64::total_cmp);
            let keys: Vec<f32> = keys.iter().map(|&x| x as f32).collect();
            assert_bytes::<8, Align8, f32>(&keys, 0, f32::total_cmp);
        }
    }
}

#[test]
fn by_key_moves_elements_of_odd_sizes_and_alignments() {
    // Elements over 16 bytes whose room is not aligned for positions are moved by insertion,
    // the others by position, whatever their size.
    let mut state = 0xDEAD_BEEF_0BAD_F00D;
    for len in [17, 40, 300] {
        let keys: Vec<u32> = (0..len)
            .map(|_| (xorshift(&mut state) % 7) as u32 * 1000 + (xorshift(&mut state) % 5) as u32)
            .collect();
        for skip in 0..3 {
            assert_bytes::<20, u8, u32>(&keys, skip, Ord::cmp);
            assert_bytes::<36, u16, u32>(&keys, skip, Ord::cmp);
            assert_bytes::<40, Align8, u32>(&keys, skip, Ord::cmp);
        }
    }
    // 65-byte elements aligned to one byte: 1009 of them take 64 KiB, and are sorted along the
    // cycles of their permutation.
    let keys: Vec<u64> = (0..1009)
        .map(|_| xorshift(&mut state) >> (xorshift(&mut state) % 64))
        .collect();
    assert_bytes::<65, u8, u64>(&keys, 0, Ord::cmp);
    assert_bytes::<65, u8, u64>(&keys, 1, Ord::cmp);
}

#[test]
fn sorts_wide_and_extreme_keys_with_owning_elements() {
    let mut state = 0x0F0F_F0F0_1234_4321;
    let len = 600;
    // `i128` keys of every magnitude and sign: deep levels whose differences need a shift.
    let keys: Vec<i128> = (0..len)
        .map(|_| {
            let r = xorshift(&mut state) as i128 | ((xorshift(&mut state) as i128) << 64);
            r >> (xorshift(&mut state) % 128)
        })
        .collect();
    assert_bytes::<8, Align8, i128>(&keys, 0, Ord::cmp);
    assert_bucket(&keys);
    // Small buckets of several keys each: owning elements of 4 and 8 bytes moved two places.
    let keys: Vec<u16> = (0..len)
        .map(|_| (xorshift(&mut state) % 150) as u16)
        .collect();
    assert_bytes::<4, u8, u16>(&keys, 0, Ord::cmp);
    assert_bytes::<8, Align8, u16>(&keys, 1, Ord::cmp);
    // A range whose width overflows, around subnormals, and a range of subnormals only.
    let keys: Vec<f64> = (0..len)
        .map(|i| match i % 4 {
            0 => f64::MAX,
            1 => f64::MIN,
            2 => f64::from_bits(xorshift(&mut state) % 1000),
            _ => -f64::from_bits(xorshift(&mut state) % 1000),
        })
        .collect();
    assert_bytes::<12, u32, f64>(&keys, 0, f64::total_cmp);
    let keys: Vec<f32> = (0..len)
        .map(|_| f32::from_bits((xorshift(&mut state) % 0x7F_FFFF) as u32))
        .collect();
    assert_bytes::<12, u32, f32>(&keys, 0, f32::total_cmp);
    assert_bucket_floats(&keys);
}

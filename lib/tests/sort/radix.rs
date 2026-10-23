use a::sort::{IntegerKey, radix, radix_by_key};
use std::cell::Cell;
use std::fmt::Debug;
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

/// Sorts `v` with [`radix`] and checks the result against [`slice::sort_unstable`].
fn assert_radix<T: IntegerKey + Debug>(v: &[T]) {
    let mut expected = v.to_vec();
    expected.sort_unstable();
    let mut actual = v.to_vec();
    radix(&mut actual);
    assert_eq!(actual, expected, "input {v:?}");
}

/// Sorts `keys`, each paired with its index, with [`radix_by_key`] and checks the result
/// against the standard library's stable sort.
fn assert_radix_by_key<K: IntegerKey + Debug>(keys: &[K]) {
    let mut expected: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    expected.sort_by_key(|&(k, _)| k);
    let mut actual: Vec<(K, usize)> = keys.iter().copied().zip(0..).collect();
    radix_by_key(&mut actual, |&(k, _)| k);
    assert_eq!(actual, expected, "keys {keys:?}");
}

// ---------------------------------------------------------------------------------------------
// `radix`
// ---------------------------------------------------------------------------------------------

#[test]
fn sorts_empty_slice() {
    let mut v: Vec<i32> = vec![];
    radix(&mut v);
    assert_eq!(v, vec![]);
}

#[test]
fn sorts_single_element() {
    let mut v = vec![42];
    radix(&mut v);
    assert_eq!(v, vec![42]);
}

#[test]
fn sorts_two_elements() {
    let mut v = vec![2, 1];
    radix(&mut v);
    assert_eq!(v, vec![1, 2]);

    let mut v = vec![1, 2];
    radix(&mut v);
    assert_eq!(v, vec![1, 2]);
}

#[test]
fn sorts_already_sorted() {
    let mut v: Vec<u32> = (0..100).collect();
    radix(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_reverse_sorted() {
    let mut v: Vec<u32> = (0..100).rev().collect();
    radix(&mut v);
    assert_eq!(v, (0..100).collect::<Vec<_>>());
}

#[test]
fn sorts_non_increasing() {
    let mut v = vec![5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0];
    radix(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
}

#[test]
fn sorts_with_duplicates() {
    let mut v = vec![3, 1, 2, 3, 2, 1, 0, 3, 0, 2, 1, 1];
    radix(&mut v);
    assert_eq!(v, vec![0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3]);
}

#[test]
fn sorts_all_equal() {
    for len in 0..40 {
        let mut v = vec![7; len];
        radix(&mut v);
        assert_eq!(v, vec![7; len]);
    }
}

#[test]
fn sorts_a_subslice_and_leaves_the_rest_untouched() {
    let mut v: Vec<u32> = (0..30).map(|i| (i * 17) % 30).collect();
    let before = v.clone();
    radix(&mut v[5..25]);
    let mut middle = before[5..25].to_vec();
    middle.sort_unstable();
    assert_eq!(&v[..5], &before[..5]);
    assert_eq!(&v[5..25], &middle[..]);
    assert_eq!(&v[25..], &before[25..]);
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
            assert_radix(&v);
        }
    }
}

/// Checks random slices of every length up to 70, across the change of digit width at 64
/// elements, and some longer ones, over the whole range of the type and over `span` values
/// from each of `bases`.
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
            assert_radix(&v);
            for base in $bases {
                let v: Vec<$t> = (0..len)
                    .map(|_| <$t>::wrapping_add(base, (xorshift($state) % $span) as $t))
                    .collect();
                assert_radix(&v);
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
fn sorts_keys_that_differ_in_few_bits() {
    // Only the bits in which keys differ are sorted: one bit at either end of the type, two bits
    // far apart (the digits between them are the same for all keys and take no pass), and spans
    // that are and are not a multiple of the digit width.
    let mut state = 0xB175_B175_B175_B175;
    for len in [2, 3, 17, 64, 65, 300, size(5000, 600)] {
        for bits in [
            vec![0u32],
            vec![31],
            vec![0, 31],
            vec![3, 27],
            vec![5, 6, 7, 8],
            (0..12).collect(),
            (0..16).collect(),
            (4..28).collect(),
        ] {
            let v: Vec<u32> = (0..len)
                .map(|_| {
                    let r = xorshift(&mut state);
                    bits.iter()
                        .enumerate()
                        .fold(0x5A5A_0000, |x, (i, &b)| x ^ (((r >> i) as u32 & 1) << b))
                })
                .collect();
            assert_radix(&v);
            let w: Vec<i64> = v.iter().map(|&x| i64::from(x as i32) << 20).collect();
            assert_radix(&w);
        }
    }
}

#[test]
fn sorts_short_slices_of_wide_keys() {
    // Fewer than 64 elements use digits of `log₂ n` bits, at least 4: 16 random `u128`s take
    // 32 digits.
    let mut state = 0x5407_5407_5407_5407;
    for len in 2..64 {
        let v: Vec<u128> = (0..len)
            .map(|_| (u128::from(xorshift(&mut state)) << 64) | u128::from(xorshift(&mut state)))
            .collect();
        assert_radix(&v);
        let w: Vec<i8> = (0..len).map(|_| xorshift(&mut state) as i8).collect();
        assert_radix(&w);
        // 16 digits of 4 bits, or 13 of 5, each a count of its own.
        let u: Vec<u64> = (0..len).map(|_| xorshift(&mut state)).collect();
        assert_radix(&u);
        assert_radix_by_key(&u);
    }
}

/// `len` random keys of `bits` bits from bit `low` on, with the byte from bit `shared` on the
/// same in every key, if there is one.
fn short_keys(len: usize, bits: u32, low: u32, shared: Option<u32>, state: &mut u64) -> Vec<u64> {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    };
    let byte = xorshift(state) & 0xFF;
    (0..len)
        .map(|_| {
            let x = (xorshift(state) & mask) << low;
            match shared {
                Some(at) => (x & !(0xFF << at)) | (byte << at),
                None => x,
            }
        })
        .collect()
}

#[test]
fn sorts_short_slices_in_every_digit_plan() {
    // From 16 to 639 elements, keys of up to eight 8-bit digits are counted in rows of eight
    // lanes, one per digit, when those digits save enough passes against digits of 6 bits;
    // from 24 elements on, digits of 6 bits are taken otherwise, below 48 elements per digit of
    // `log₂ n` bits. In lanes, signed values are sorted as their bits, and a digit all keys share
    // takes no pass.
    let lens: &[usize] = if cfg!(miri) {
        &[16, 24, 48, 100, 300]
    } else {
        &[
            16, 17, 23, 24, 31, 32, 47, 48, 63, 64, 95, 96, 100, 127, 128, 143, 144, 200, 256, 319,
            320, 400, 511, 639, 640, 641,
        ]
    };
    let widths: &[u32] = if cfg!(miri) {
        &[9, 24, 48, 64]
    } else {
        &[7, 9, 12, 16, 17, 24, 25, 32, 40, 48, 56, 63, 64]
    };
    let mut state = 0x1A4E_1A4E_1A4E_1A4E;
    for &len in lens {
        for &bits in widths {
            for (low, shared) in [(0, None), (3, None), (0, Some(16))] {
                if bits + low > 64 || (shared.is_some() && bits < 32) {
                    continue;
                }
                let v = short_keys(len, bits, low, shared, &mut state);
                assert_radix(&v);
                assert_radix_by_key(&v);
                // Negative values too: the keys moved down by half their range.
                let half = 1u64 << (bits + low - 1);
                let signed: Vec<i64> = v.iter().map(|&x| x.wrapping_sub(half) as i64).collect();
                assert_radix(&signed);
                assert_radix_by_key(&signed);
                if bits + low <= 32 {
                    let u: Vec<u32> = v.iter().map(|&x| x as u32).collect();
                    assert_radix(&u);
                    let i: Vec<i32> = signed.iter().map(|&x| x as i32).collect();
                    assert_radix(&i);
                    assert_radix_by_key(&i);
                }
            }
        }
        let i16s: Vec<i16> = (0..len).map(|_| xorshift(&mut state) as i16).collect();
        assert_radix(&i16s);
        let isizes: Vec<isize> = (0..len).map(|_| xorshift(&mut state) as isize).collect();
        assert_radix(&isizes);
        let chars: Vec<char> = (0..len)
            .map(|_| char::from_u32((xorshift(&mut state) % 0x11_0000) as u32).unwrap_or('a'))
            .collect();
        assert_radix(&chars);
        // Elements of 72 bytes with eight digits of keys are gathered once, from 64 elements on.
        let mut e: Vec<[u64; 9]> = (0..len)
            .map(|i| {
                let mut e = [i as u64; 9];
                e[0] = xorshift(&mut state);
                e
            })
            .collect();
        let mut expected = e.clone();
        expected.sort_by_key(|e| e[0]);
        radix_by_key(&mut e, |e| e[0]);
        assert_eq!(e, expected);
    }
}

#[test]
fn sorts_across_the_counter_widths_and_runs_of_shared_digits() {
    // Counts are `u16` up to 65,535 elements and `u32` beyond. Nearly sorted keys share their
    // high digits with their neighbours, and four distinct wide keys share every digit with a
    // quarter of the others: the counts and places are handed out four keys at a time.
    let mut state = 0xC0DE_C0DE_C0DE_C0DE;
    let lens: &[usize] = if cfg!(miri) {
        &[300]
    } else {
        &[1000, 65_535, 65_536]
    };
    for &len in lens {
        let four: [u64; 4] = std::array::from_fn(|_| xorshift(&mut state));
        let mut nearly: Vec<u64> = (0..len).map(|_| xorshift(&mut state)).collect();
        nearly.sort_unstable();
        for _ in 0..(len / 100).max(1) {
            let i = (xorshift(&mut state) % len as u64) as usize;
            let j = (xorshift(&mut state) % len as u64) as usize;
            nearly.swap(i, j);
        }
        let shapes: Vec<Vec<u64>> = vec![
            (0..len).map(|_| xorshift(&mut state)).collect(),
            (0..len)
                .map(|_| four[(xorshift(&mut state) % 4) as usize])
                .collect(),
            nearly,
        ];
        for v in shapes {
            assert_radix(&v);
            assert_radix_by_key(&v);
            let narrow: Vec<u32> = v.iter().map(|&x| (x >> 32) as u32).collect();
            assert_radix(&narrow);
            assert_radix_by_key(&narrow);
        }
    }
}

#[test]
fn sorts_chars_and_bools() {
    let mut state = 0xC4A2_C4A2_C4A2_C4A2;
    let pool = ['\u{D7FF}', '\u{E000}', 'a', '\u{FFFF}', '\u{10FFFF}', '\0'];
    for len in 0..size(300, 60) {
        let v: Vec<char> = (0..len)
            .map(|_| pool[(xorshift(&mut state) % pool.len() as u64) as usize])
            .collect();
        assert_radix(&v);
        let b: Vec<bool> = (0..len).map(|i| (i * 7) % 3 == 0).collect();
        assert_radix(&b);
    }
}

#[test]
fn sorts_long_inputs() {
    let n = size(300_000, 3_000);
    let mut state = 0x7E57_7E57_7E57_7E57;
    let shapes: Vec<Vec<u64>> = vec![
        (0..n).map(|_| xorshift(&mut state)).collect(),
        (0..n).map(|_| xorshift(&mut state) % n as u64).collect(),
        (0..n).map(|_| xorshift(&mut state) % 7).collect(),
        // Organ pipe: ascending then descending.
        (0..n as u64).map(|i| i.min(n as u64 - 1 - i)).collect(),
        // Sorted with the last element moved to the front.
        std::iter::once(n as u64).chain(0..n as u64 - 1).collect(),
    ];
    for v in shapes {
        assert_radix(&v);
    }
}

/// Sorts `v` with [`radix`] and returns how many allocations that made.
fn allocations_of<T: IntegerKey>(v: &mut [T]) -> usize {
    crate::allocations::count(|| radix(v))
}

#[test]
fn allocates_only_beyond_4_kib() {
    // 1024 `u32`s fill the 4 KiB stack buffer, and 1024 counters the 4 KiB of counts: 1024 random
    // `u32`s take four digits of 8 bits, 1025 move through an allocated buffer, and 5000 take
    // three digits of 11 bits, too many counts for the stack.
    let mut state = 0x0A11_0C47_E5B0_FFE5;
    for (len, allocations) in [(100, 0), (1024, 0), (1025, 1), (size(5000, 4100), 2)] {
        let mut v: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32).collect();
        assert_eq!(allocations_of(&mut v), allocations, "len = {len}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    let mut v: Vec<u32> = (0..10_000).collect();
    assert_eq!(allocations_of(&mut v), 0);
    v.reverse();
    assert_eq!(allocations_of(&mut v), 0);
    // Values that differ in a few high bits take one digit, but span far more values than it
    // has: they are moved by the digit, not counted by value, which would allocate a count for
    // every value in between.
    for shift in [12, 40, 60] {
        let mut v: Vec<u64> = (0..300)
            .map(|_| (xorshift(&mut state) % 16) << shift)
            .collect();
        assert_eq!(allocations_of(&mut v), 0, "shift = {shift}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
}

// ---------------------------------------------------------------------------------------------
// `radix_by_key`
// ---------------------------------------------------------------------------------------------

#[test]
fn by_key_sorts_short_slices() {
    assert_radix_by_key::<u32>(&[]);
    assert_radix_by_key(&[1u32]);
    assert_radix_by_key(&[2u32, 1]);
    assert_radix_by_key(&[1u32, 2]);
    assert_radix_by_key(&[1u32, 1]);
}

#[test]
fn by_key_is_stable() {
    let mut state = 0x57AB_1E57_AB1E_57AB;
    for len in 0..size(200, 50) {
        for span in [1, 2, 5, 50, 1 << 20] {
            let keys: Vec<u32> = (0..len)
                .map(|_| (xorshift(&mut state) % span) as u32)
                .collect();
            assert_radix_by_key(&keys);
        }
    }
}

#[test]
fn by_key_leaves_sorted_input_and_reverses_only_strictly_descending_input() {
    assert_radix_by_key(&[0u32, 0, 1, 1, 2, 2]);
    assert_radix_by_key(&[5u32, 4, 3, 2, 1, 0]);
    // Not strictly descending: reversing it would swap the equal keys.
    assert_radix_by_key(&[5u32, 4, 4, 3, 2, 1, 0]);
    assert_radix_by_key(&[2u8, 2, 1]);
    let long: Vec<u64> = (0..size(10_000, 2_000) as u64).rev().collect();
    assert_radix_by_key(&long);
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
            assert_radix_by_key(&keys);
        }
    }
}

#[test]
fn by_key_sorts_every_key_type() {
    let mut state = 0xB1CE_B1CE_B1CE_B1CE;
    for len in [0, 1, 2, 9, 63, 64, 100, size(5000, 500)] {
        let mut r = || xorshift(&mut state);
        let u8s: Vec<u8> = (0..len).map(|_| r() as u8).collect();
        assert_radix_by_key(&u8s);
        let i8s: Vec<i8> = (0..len).map(|_| r() as i8).collect();
        assert_radix_by_key(&i8s);
        let bools: Vec<bool> = (0..len).map(|_| r() % 3 == 0).collect();
        assert_radix_by_key(&bools);
        let i32s: Vec<i32> = (0..len).map(|_| r() as i32).collect();
        assert_radix_by_key(&i32s);
        let u64s: Vec<u64> = (0..len).map(|_| r()).collect();
        assert_radix_by_key(&u64s);
        let i128s: Vec<i128> = (0..len)
            .map(|_| ((u128::from(r()) << 64) | u128::from(r())) as i128)
            .collect();
        assert_radix_by_key(&i128s);
        let isizes: Vec<isize> = (0..len).map(|_| r() as isize % 1000).collect();
        assert_radix_by_key(&isizes);
        let chars: Vec<char> = (0..len)
            .map(|_| char::from_u32((r() % 0x11_0000) as u32).unwrap_or('\u{E000}'))
            .collect();
        assert_radix_by_key(&chars);
    }
}

#[test]
fn by_key_sorts_zero_sized_elements() {
    for len in 0..40 {
        let mut v = vec![(); len];
        radix_by_key(&mut v, |_| 0u8);
        assert_eq!(v.len(), len);
    }
}

#[test]
fn by_key_sorts_large_elements_by_moving_and_by_gathering() {
    // Elements are gathered once when the passes would move at least 96 bytes of each and the
    // slice takes at most 16 MiB, and moved in every pass otherwise: 64-byte elements with two
    // digits of keys are gathered, and 256-byte ones beyond 16 MiB moved.
    let mut state = 0x0B16_0B16_0B16_0B16;
    for len in [10, 200, size(3000, 300)] {
        let mut v: Vec<[u64; 8]> = (0..len)
            .map(|i| {
                let mut e = [i as u64; 8];
                e[0] = xorshift(&mut state) % 5000;
                e
            })
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|e| e[0]);
        radix_by_key(&mut v, |e| e[0]);
        assert_eq!(v, expected);
    }
    for len in [100, size(70_000, 200)] {
        let mut v: Vec<[u64; 32]> = (0..len)
            .map(|i| {
                let mut e = [i as u64; 32];
                e[0] = xorshift(&mut state) % 5000;
                e
            })
            .collect();
        let mut expected = v.clone();
        expected.sort_by_key(|e| e[0]);
        radix_by_key(&mut v, |e| e[0]);
        assert!(v == expected);
    }
}

#[test]
fn by_key_calls_key_once_per_element_and_places_its_latest_state() {
    for len in [2, 9, 100, size(3000, 300)] {
        let mut v: Vec<(u32, Cell<u32>)> = (0..len as u32)
            .map(|i| ((i * 7919) % 1013, Cell::new(0)))
            .collect();
        radix_by_key(&mut v, |e| {
            e.1.set(e.1.get() + 1);
            e.0
        });
        assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
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
fn by_key_sorts_owning_elements_exactly_once() {
    // `Tracked` is 24 bytes: gathered with four or more digits of keys.
    let mut state = 0x0DD5_0DD5_0DD5_0DD5;
    for len in [2, 3, 9, 40, 300, size(5000, 400)] {
        for span in [2u32, 300, u32::MAX] {
            let keys: Vec<u32> = (0..len)
                .map(|_| xorshift(&mut state) as u32 % span)
                .collect();
            let (order, completed) = sort_tracked(&keys, |v| radix_by_key(v, |t| t.key));
            assert!(completed);
            let mut expected: Vec<usize> = (0..len).collect();
            expected.sort_by_key(|&id| keys[id]);
            assert_eq!(order, expected);
            // As `u64` keys spread over 62 bits: six digits, gathered from 64 elements on.
            let (order, completed) = sort_tracked(&keys, |v| {
                radix_by_key(v, |t| u64::from(t.key) << 30 | u64::from(t.key))
            });
            assert!(completed);
            let mut expected: Vec<usize> = (0..len).collect();
            expected.sort_by_key(|&id| keys[id]);
            assert_eq!(order, expected);
        }
    }
}

#[test]
fn by_key_panicking_key_leaves_the_slice_as_it_was() {
    let len = size(150, 40);
    let mut state = 0x0123_4567_89AB_CDEF;
    let random: Vec<u32> = (0..len).map(|_| xorshift(&mut state) as u32).collect();
    let layouts: Vec<Vec<u32>> = vec![
        random,
        (0..len as u32).map(|i| i % 3).collect(),
        (0..len as u32).collect(),
        (0..len as u32).rev().collect(),
    ];
    let step = if cfg!(miri) { 7 } else { 1 };
    for keys in layouts {
        for budget in (0..keys.len()).step_by(step) {
            let calls = Cell::new(0);
            let (order, completed) = sort_tracked(&keys, |v| {
                radix_by_key(v, |t| {
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
}

/// Sorts `v` with [`radix_by_key`] by `e[0]` and returns how many allocations that made.
fn allocations_by_key_of<const N: usize>(v: &mut [[u64; N]]) -> usize {
    crate::allocations::count(|| radix_by_key(v, |e| e[0]))
}

#[test]
fn by_key_allocates_only_beyond_4_kib() {
    // 512 `u64` elements fill the 4 KiB stack buffer, and 512 keys another 4 KiB; the keys move
    // with the elements through a second buffer of keys. 300 random 16-bit keys take two digits
    // of 8 bits, whose counts fit on the stack.
    let mut state = 0xA110_CA7E_A110_CA7E;
    for (len, allocations) in [(300, 0), (512, 0), (513, 3)] {
        let mut v: Vec<[u64; 1]> = (0..len).map(|_| [xorshift(&mut state) % 65_536]).collect();
        assert_eq!(allocations_by_key_of(&mut v), allocations, "len = {len}");
        assert!(v.windows(2).all(|w| w[0] <= w[1]));
    }
    // Sorted input is only scanned, but its keys are kept as they are read.
    let mut v: Vec<[u64; 1]> = (0..512).map(|i| [i]).collect();
    assert_eq!(allocations_by_key_of(&mut v), 0);
    let mut v: Vec<[u64; 1]> = (0..10_000).map(|i| [i]).collect();
    assert_eq!(allocations_by_key_of(&mut v), 1);
}

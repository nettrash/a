use super::drift::drift;

/// Sorts a slice with this crate's default sort: [`drift`] sort.
///
/// Use it when there is no reason to prefer a particular algorithm. It is stable, `O(n log n)`
/// on every input, `O(n log k)` with `k` distinct keys and `O(n)` on sorted or strictly
/// descending input, and it takes advantage of runs of `√n` elements or more. It is the same
/// choice the standard library made for [`slice::sort`].
///
/// The choice was measured against this crate's other general-purpose comparison sorts,
/// [`quick`](crate::sort::quick), [`merge`](crate::sort::merge), [`tim`](crate::sort::tim) and
/// [`intro`](crate::sort::intro), on an Apple M4 Max with new input for every sort. The test
/// covered 210 cases: `u64`, `u32`, `u8`, strings and 64-byte elements; random, sorted,
/// reverse-sorted, nearly sorted, 16 distinct keys, sawtooth, runs of `√n` and sorted with 1%
/// appended; 16 to a million elements. Each sort was compared with the fastest of the five in
/// every case. Drift sort is on average (geometric mean) 1.21 times slower than that fastest
/// sort and at worst 2.9 times slower. The others are on average 1.45 (Tim), 1.51 (quick), 1.6
/// (intro) and 1.74 (merge) times slower, and at worst 7.8–13.8 times slower. No sort is the
/// fastest everywhere, but drift sort is never far off:
///
/// * [`quick`](crate::sort::quick) sort is 1.1–1.3 times faster on random integers from 100
///   elements on and 1.1–1.2 times on random strings. Drift sort is 1.3 times faster on 16
///   random integers and on 100,000 or more random bytes, 1.15–1.2 times faster on random
///   64-byte elements from 1000 on, up to 2.1 times on nearly sorted input and 2.8–13 times
///   on sorted input with elements appended. Quick sort is also not stable.
/// * [`intro`](crate::sort::intro) sort is 1.1–1.4 times faster on random 64-byte elements
///   and up to 2.9 times on nearly sorted ones. Drift sort is 1.15–1.3 times faster on random
///   integers from 100 elements on, 2–4.3 times with 16 distinct keys and 3.4–11 times on
///   sorted input with elements appended, from 1000 elements on.
/// * [`tim`](crate::sort::tim) sort is up to 1.8 times faster on nearly sorted input. Drift
///   sort is 1.05–1.6 times faster on random input (5 times on a million bytes) and 2–8 times
///   with 16 distinct keys from 1000 elements on.
/// * [`merge`](crate::sort::merge) sort is up to 1.9 times faster on runs of `√n` elements.
///   Drift sort is 2–8 times faster with 16 distinct keys from 1000 elements on and up to 10
///   times on reverse-sorted 64-byte elements.
///
/// The non-comparison sorts ([`counting`](crate::sort::counting),
/// [`radix`](crate::sort::radix), [`bucket`](crate::sort::bucket),
/// [`pigeonhole`](crate::sort::pigeonhole)) are not candidates because they sort integer or
/// float keys only, not every `T: Ord`.
///
/// # Panics
///
/// Panics only if allocating the buffer fails. If the [`Ord`] implementation of `T` panics,
/// the panic is propagated and `arr` is left in an unspecified order, but it still holds every
/// original element exactly once. An [`Ord`] that is not a total order leaves `arr` in an
/// unspecified order with every element exactly once as well.
///
/// # Examples
///
/// ```
/// use a::sort::sort;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// sort(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// let mut words = vec!["pear", "apple", "fig", "apple"];
/// sort(&mut words);
/// assert_eq!(words, ["apple", "apple", "fig", "pear"]);
/// ```
#[inline]
pub fn sort<T: Ord>(arr: &mut [T]) {
    drift(arr);
}

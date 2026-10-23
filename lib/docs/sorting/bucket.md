[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Bucket

Bucket sort, as CLRS describes it, sorts `n` reals spread uniformly over `[0, 1)` by cutting the range into `n` buckets of equal width, moving every element into the bucket its key falls in, sorting each bucket by insertion and reading the buckets out in order: `O(n)` expected time, as each bucket holds about one element. This implementation cuts the range from the smallest to the largest key of each level into one bucket per element, at most 4096 (65,536 for a level of fewer than 131,072 keys that spread over its range), moves the elements into their buckets with a stable counting pass, sorts runs of buckets of at most 16 elements by one insertion sort, and cuts larger buckets again over their own range. Floats are cut by value at the first level; integers, and floats below it, by their bits, which bounds the depth by the width of the key, so that skewed input, which makes the textbook version quadratic, costs at most a few more levels. Keys are primitive integers, `char`s, `bool`s, `f32`s and `f64`s (the sealed `BucketKey` trait), floats sorted as `f64::total_cmp` orders them. It comes in two forms: `bucket` sorts the keys themselves, and `bucket_by_key` sorts any type by such a key, stably, calling `key` exactly once per element. This implementation uses twelve optimizations:

- Sorted and reverse-sorted input is recognised (non-increasing input too for `bucket_by_key`, whose runs of equal keys are reversed back), and left alone or reversed.
- Keys are read once, as bits that order as they do: signed integers and floats are rewritten in place as unsigned integers in one pass that also finds their range, and back at the end, so every comparison is one of integers.
- Floats crowded at zero are cut by their bits: values over many orders of magnitude crowd near zero when cut by value; when the three buckets there hold more than half the keys, the level is counted again by bits and moved by the cut with the smaller largest bucket (2.6–3 times faster on floats of random exponents).
- Runs of keys in one bucket do not wait on each other: keys are counted and placed four at a time, and a level with at least four keys per bucket counts into four interleaved tables, from 16 into eight (1.8–3 times faster on powers of two, 1.3–2 times with four distinct keys, 1.7 times on nearly sorted input).
- Keys that crowd a few buckets are counted and moved in groups (`bucket`): the tables split the chains of a count, but not those of the moves. A level of at least 512 keys, two or more per bucket, whose eight keys at even distances share buckets in at least four pairs is cut into groups of keys side by side, each counted into tables of its own and moved by places of its own: two groups, or four of two tables each from eight keys per bucket on (1.02–1.15 times faster on powers of two at 10,000 elements, 1.05–1.3 times at 100,000 and 1.3–1.45 times at 1,000,000; 1.12–1.35 times with 4 to 16 distinct keys from 100,000 elements on; 1.12–1.3 times on `f32`s of random exponents from 10,000 on).
- One bucket per key up to 65,536 for keys that spread: cut into 4096 buckets, a level of 5000 to 130,000 keys holds 1 to 32 per bucket, which the insertion sorts or a second level pay for. A level of more than 4096 and fewer than 131,072 keys takes one bucket per key when 32 of its keys at even distances fall in at least 30 different buckets of 4096, unless fewer than 65,536 keys start mostly in order (1.2–2.9 times faster on random and uniform keys and keys below `n` from 10,000 to 130,000 elements; few distinct keys and keys crowding a few buckets keep 4096 buckets).
- Buckets are found once: a level of at most 2048 keys keeps the bucket of each key as it counts it.
- A level of one bucket per key, when there are fewer keys in its range than buckets, is sorted as it is moved (1.5 times faster on keys below `n`).
- Insertion without a branch for keys that move: each chunk of 32 keys is sorted with a branch when few keys moved in the chunk before, and otherwise by writing the larger and the smaller of a key and its neighbour without one (1.2–1.8 times faster on uniform keys).
- Large elements move once: `bucket_by_key` sorts parts of 64 KiB and at most 8192 elements of at least 64 bytes by the indices of their elements, which then move along the cycles of that permutation (1.1–2 times faster), and elements over 16 bytes that keys often pass in a run of small buckets by their positions.
- Keys are read by a loop fit for the size of the elements: a vectorized one up to 16 bytes, and loops that leave at the first key out of order beyond.
- No allocation for small sorts: up to 16 elements are sorted by insertion at once, and buffers and counts (`u16` up to 65,535 elements) of up to 4 KiB each live on the stack; the counts become the starts of the buckets four at a time (1.03–1.07 times faster at 100–1000 elements).

## Examples

Rust usage with integers and floats:

```rust
use a::sort::bucket;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  bucket(&mut v);
  assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

  let mut v = vec![0.42, -1.5, 0.0, f64::INFINITY, -0.0, 0.17];
  bucket(&mut v);
  assert_eq!(v, vec![-1.5, -0.0, 0.0, 0.17, 0.42, f64::INFINITY]);
}
```

Sorting by a key, stably (equal temperatures keep their order):

```rust
use a::sort::bucket_by_key;

fn main() {
  let mut cities = vec![("Oslo", 4.5), ("Cairo", 31.0), ("Lima", 19.25), ("Kyiv", 4.5)];
  bucket_by_key(&mut cities, |&(_, t)| t);
  let names: Vec<&str> = cities.iter().map(|c| c.0).collect();
  assert_eq!(names, vec!["Oslo", "Kyiv", "Lima", "Cairo"]);
}
```

## Efficiency

- Time complexity:
  - Average, for keys spread evenly over their range: O(n), one or two levels of buckets of a few keys
  - Worst: O(n·w) for keys of `w` bits: a level divides the range of its keys by its number of buckets, from 17 to 4096 (65,536 below 131,072 keys), so an element passes at most `⌈w / 4⌉` levels (one more for floats), and while its bucket holds more than 4096 keys, a level takes at least 12 bits off
  - Sorted and reverse-sorted input (also non-increasing for `bucket_by_key`): O(n), one scan
- Comparisons: none to place elements; those of the insertion sorts within buckets of a few keys, and of the scan for a sorted or reverse-sorted run
- Calls to `key` (`bucket_by_key`): exactly n
- Space complexity: O(n) buffers (`bucket`: n keys; `bucket_by_key`: n keys and n elements, or 2n indices for large elements) plus the counts of a level; each on the stack up to 4 KiB, otherwise allocated
- Stability: `bucket_by_key` is stable; `bucket` cannot tell equal keys apart
- Against the textbook bucket sort (`n` buckets as vectors, each sorted by insertion, concatenated): `bucket` is 6.8–11 times faster on random and uniform `u64`s, 4.9–8 times on uniform `f64`s, 3–4.4 times with four distinct keys and 55–180 times on sorted and reverse-sorted input; `bucket_by_key` is 4.5–8 times faster on uniform `u64` keys of 16-byte elements. On skewed input the textbook version is quadratic: `f64`s of random exponents take it 170 times as long at 10,000 elements, powers of two 7–33 times at 1000
- Against the standard library, from 100 elements on: `bucket` is 1.3–3 times faster than `slice::sort_unstable` on random and uniform integers, 1.7–3.7 times on uniform floats (against `sort_unstable_by(f64::total_cmp)`), 2–5.9 times on keys below `n` and 1.25–4.8 times on nearly sorted input; `bucket_by_key` is 1.3–3.1 times faster than `slice::sort_by_key` on random `u64` keys of 16-byte elements, 2.3–4.8 times of 72-byte elements and 1.5–3.4 times on random `f64` keys. Its weakness is keys that equal-width buckets do not spread, where the standard library gains from few distinct values: powers of two take 1.2–4.3 times as long up to 100,000 elements (about as long at 1,000,000), four distinct keys up to 1.5 times from 1000 to 10,000 elements (`f64`s 1.8–2.2 times from 1000 on), and `f32`s of random exponents (mostly zeros and infinities) up to 1.8 times; 16 elements take about as long. For 32-bit integers on large slices, [Radix](radix.md) sort, which does not depend on how the keys are spread, is faster (1.4–1.6 times on random `u32`s from 100,000 elements on), but on random `u64`s it is 1.2–1.35 times slower

use core::mem;

/// Sorts a slice in place using stooge sort.
///
/// Stooge sort compares the first and the last element and swaps them if they are out of
/// order. Then, if the slice has at least three elements, it stooge-sorts its first two thirds
/// (rounded up), its last two thirds, and its first two thirds again. The first call moves
/// every one of the largest third of the elements that starts in the first two thirds into the
/// middle third, so after the second call all of them sit at the end, in order; the third call
/// sorts the rest. The algorithm is Problem 7-3 of Cormen, Leiserson, Rivest and Stein's
/// *Introduction to Algorithms*, where Professors Howard, Fine and Howard, the Three Stooges,
/// propose it as an "elegant" sorting algorithm, and the reader shows that it takes
/// `Θ(n^(log 3 / log 1.5)) ≈ Θ(n^2.71)` time: worse than bubble sort. Which elements are
/// compared depends only on the length of the slice, never on the answers, so stooge sort is a
/// sorting network: one fixed sequence of compare-exchanges for each length.
///
/// The realisation keeps that sequence exactly — every comparison the textbook recursion makes,
/// in the same order, with the same swaps — and executes it as cheaply as it can:
///
/// * **Unrolled bottom levels.** All three parts of a part of `m` elements have
///   `m − ⌊m/3⌋` elements, so all parts at one depth of the recursion have the same length,
///   and the parts four levels above the bottom have 7, 8 or 9 elements throughout one sort.
///   They run as straight-line code: the 121 compare-exchanges of the recursion below them, at
///   constant indices, with no calls and no index checks. The textbook makes one call per
///   compare-exchange, and the calls are its cost: recursing down to two elements without
///   index checks is only 1.05–1.15 times faster than it, while unrolling four levels is 5–8
///   times faster for `u64`, 1.4–1.65 times for strings and 1.4–2.3 times for 64-byte elements
///   than recursing down to two elements. Unrolling two or three levels (parts of at most 4 or
///   6 elements) was 1.0–1.65 times slower, and five levels (at most 13) at most 1.05 times
///   faster, for four times the code.
/// * **A branch, not a branchless select.** Only about 1% of the compare-exchanges swap
///   anything, so a branch is predicted almost always and the compare-exchanges do not wait
///   for one another. A branchless select chains each to the one before: measured 2 times
///   slower than the textbook for `u64`, 4 times for strings and 1.8 times for 64-byte
///   elements.
/// * **The swap is marked cold**, so the compare-exchanges that do not swap are laid out as
///   one straight line: 1.5–1.85 times faster for 64-byte elements, 1.05–1.25 times for
///   strings and 1.1–1.25 times for `u64` from 100 elements on.
///
/// Together, against the textbook: 9–11 times faster for `u64` from 64 elements on (3–10 times
/// below), 1.7–2.2 times for strings, and 4.3–5 times for 64-byte elements from 32 on (2.5–4.5
/// times below; 1.4–2 times with few distinct keys, where every comparison reads all 64
/// bytes). Against a first version that recursed down to two elements
/// with a branchless compare-exchange: 16–22 times faster for `u64` from 64 elements on.
///
/// The code is safe Rust: indices into the unrolled parts are constants, and the parts above
/// them are cut with checked slicing, once per part.
///
/// # Properties
///
/// * **Unstable**: the compare-exchange of the first and the last element jumps over
///   everything between them. `[1a, 1b, 0]` becomes `[0, 1b, 1a]`.
/// * **In place**: no allocation. The recursion nests two calls per level (the compiler turns
///   the third into a loop), `D(n) − 4` levels for `n ≥ 10`, where `D(n) ≈ log₁.₅ n` is the depth
///   defined below: 12 levels for 1000 elements.
/// * **Not adaptive**: every input of `n` elements costs the same comparisons. Already-sorted
///   input is never moved: only a strictly smaller later element is swapped forward.
///
/// Let `D(n)` be the number of times `m ↦ m − ⌊m/3⌋` must be applied to `n` to reach 2 (the
/// depth of the recursion). Every part makes one comparison and has three parts below it, down
/// to depth `D(n)`, so stooge sort of `n ≥ 2` elements makes exactly
/// `C(n) = 1 + 3 + … + 3^D(n) = (3^(D(n) + 1) − 1)/2` comparisons, on every input
/// (equivalently `C(2) = 1` and `C(n) = 1 + 3·C(n − ⌊n/3⌋)`). The number of swaps depends on
/// the input and is at most `C(n)`; on random input it is about 1% of it.
///
/// | `n`     | `D(n)` | Comparisons, every input | Bubble sort, worst |
/// |---------|--------|--------------------------|--------------------|
/// | 2       | 0      | 1                        | 1                  |
/// | 4       | 2      | 13                       | 6                  |
/// | 8       | 4      | 121                      | 28                 |
/// | 16      | 6      | 1,093                    | 120                |
/// | 100     | 11     | 265,720                  | 4,950              |
/// | 1000    | 16     | 64,570,081               | 499,500            |
///
/// `D(n)` grows in steps, so 100 and 128 elements cost the same; on average `C(n)` grows as
/// `n^2.71`.
///
/// Stooge sort is mainly of educational value, as an example of a correct algorithm with a
/// surprising running time; prefer [`slice::sort`] for real workloads. [`slice::sort_unstable`]
/// is 1.2–4 times faster on 8 elements, 50–130 times on 100 random elements, and 900–2400
/// times on 1000 (for `u64`, strings and 64-byte elements).
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: every compare-exchange compares before it swaps. An [`Ord`] that is
/// not a total order leaves `arr` in an unspecified order with every element exactly once,
/// after exactly `C(n)` comparisons, since which elements are compared never depends on the
/// answers.
///
/// # Examples
///
/// ```
/// use a::sort::stooge;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// stooge(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// stooge(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
///
/// // Not stable: sorting by the first field carries 'a' past 'b'.
/// #[derive(Debug, PartialEq, Eq)]
/// struct Card(u8, char);
/// impl Ord for Card {
///     fn cmp(&self, other: &Self) -> std::cmp::Ordering {
///         self.0.cmp(&other.0)
///     }
/// }
/// impl PartialOrd for Card {
///     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
///         Some(self.cmp(other))
///     }
/// }
/// let mut cards = [Card(1, 'a'), Card(1, 'b'), Card(0, 'c')];
/// stooge(&mut cards);
/// assert_eq!(cards, [Card(0, 'c'), Card(1, 'b'), Card(1, 'a')]);
/// ```
pub fn stooge<T: Ord>(arr: &mut [T]) {
    match arr.len() {
        0 | 1 => {}
        2 => sort::<T, 2>(arr),
        3 => sort::<T, 3>(arr),
        4 => sort::<T, 4>(arr),
        5 => sort::<T, 5>(arr),
        6 => sort::<T, 6>(arr),
        7 => sort::<T, 7>(arr),
        8 => sort::<T, 8>(arr),
        9 => sort::<T, 9>(arr),
        n => {
            // Every part at one depth of the recursion has the same length, so the length of
            // the parts that run as straight-line code is known before the sort starts.
            match unrolled_len(n) {
                7 => sort::<T, 7>(arr),
                8 => sort::<T, 8>(arr),
                _ => sort::<T, 9>(arr),
            }
        }
    }
}

/// The longest part that runs as straight-line code: four levels of recursion, 121
/// compare-exchanges on 7 to 9 elements.
///
/// Measured against unrolling fewer or more levels of the same code, relative to this choice
/// (`u64`, strings, 64-byte elements; 16 to 500 elements): down to parts of 2 elements, that
/// is plain recursion, 4.9–8.2, 1.5–1.65 and 1.4–2.3 times slower; of at most 4 elements
/// 1.4–1.65, 1.03–1.09 and 1.03–1.23 times slower; of at most 6 elements 1.0–1.17, 1.01–1.06
/// and 1.0–1.09 times slower. At most 13 elements (364 compare-exchanges, five levels) is
/// within 0.95–1.12 of it, but compiles to four times the code (1917 compare-exchanges
/// instead of 461) for every element type.
const UNROLLED_MAX: usize = 9;

/// Returns the length of the parts of a slice of `n > UNROLLED_MAX` elements at the first
/// depth of the recursion where they are at most [`UNROLLED_MAX`] long: 7, 8 or 9.
///
/// Every part of `m` elements recurses into parts of `m − ⌊m/3⌋`, so all parts at one depth
/// have the same length; `m − ⌊m/3⌋` grows with `m` and is 7 for `m = 10`, so a part longer
/// than 9 never has parts shorter than 7.
fn unrolled_len(n: usize) -> usize {
    let mut m = n;
    while m > UNROLLED_MAX {
        m -= m / 3;
    }
    m
}

/// Stooge-sorts `v`, whose parts at some depth of the recursion (or `v` itself) are `M`
/// elements long; those parts run as straight-line code.
///
/// A slice of any other length — which [`stooge`] never passes — is still left a permutation
/// of itself: all indexing is checked or into arrays of length `M`.
fn sort<T: Ord, const M: usize>(v: &mut [T]) {
    if let Ok(w) = <&mut [T; M]>::try_from(&mut *v) {
        return unrolled::<T, M>(w);
    }
    let n = v.len();
    if n <= M {
        return;
    }
    if v[n - 1] < v[0] {
        v.swap(0, n - 1);
    }
    let t = n / 3;
    sort::<T, M>(&mut v[..n - t]);
    sort::<T, M>(&mut v[t..]);
    // The compiler turns this last call into a jump back to the top: the recursion is two
    // calls deep per level, not three.
    sort::<T, M>(&mut v[..n - t]);
}

/// Stooge-sorts `w` with the recursion fully unrolled: one fixed sequence of compare-exchanges
/// at constant indices, so no index is checked at run time and, for small `Copy` types, the
/// elements stay in registers throughout.
#[inline(always)]
fn unrolled<T: Ord, const M: usize>(w: &mut [T; M]) {
    match M {
        2 => net2(w, 0),
        3 => net3(w, 0),
        4 => net4(w, 0),
        5 => net5(w, 0),
        6 => net6(w, 0),
        7 => net7(w, 0),
        8 => net8(w, 0),
        9 => net9(w, 0),
        _ => unreachable!("no unrolled stooge sort of {M} elements"),
    }
}

/// Swaps `w[i]` and `w[j]` (`i < j`) if `w[j]` is smaller.
///
/// A plain branch, not a branchless select: only about 1% of the compare-exchanges of stooge
/// sort swap anything (4.5% at 16 random elements, 1% from 64 on, 0.2% with few distinct
/// keys), so the branch is predicted almost always, and the compare-exchanges do not wait for
/// one another. A branchless select chains every compare-exchange to the one before: measured
/// 2 times slower than even the textbook recursion for `u64`, 4 times for strings and 1.8
/// times for 64-byte elements.
///
/// The swap is marked as the unlikely side of the branch (see [`unlikely`]). [`mem::swap`] of
/// two disjoint elements, rather than [`slice::swap`], keeps the stack frame small: the latter
/// reserved a temporary element for each of the 121 inlined swaps, 7.7 KiB per level of the
/// recursion for 64-byte elements.
#[inline(always)]
fn cx<T: Ord, const M: usize>(w: &mut [T; M], i: usize, j: usize) {
    if w[j] < w[i] {
        unlikely();
        let (left, right) = w.split_at_mut(j);
        mem::swap(&mut left[i], &mut right[0]);
    }
}

/// Does nothing; a call to it marks the code around it as rarely run.
///
/// Calls to a `#[cold]` function tell LLVM that the branch leading to them is unlikely, so it
/// lays out the compare-exchanges that do not swap as one straight line and moves the swaps out
/// of it. The call itself stays (callers instantiate [`stooge`] in their own crate, where this
/// body is not visible, unless link-time optimisation inlines it) and costs a few cycles per
/// swap. Measured against the same code without it: 1.5–1.85 times faster for 64-byte elements
/// (1.0–1.05 times with few distinct keys, whose comparisons dominate), 1.05–1.25 times for
/// strings, and 1.1–1.25 times for `u64` from 100 elements on, where LLVM now reloads elements
/// instead of carrying them in registers; on 8 to 32 `u64` it is anywhere from 1.35 times
/// faster to 1.3 times slower, as more of the few compare-exchanges swap there. Calling a cold
/// function that does the swap itself did as well for strings and 64-byte elements, but was up
/// to 1.5 times slower on 8 to 32 `u64`. (`core::hint::cold_path` would say this directly, but
/// is not stable in Rust 1.88.)
#[cold]
#[inline(never)]
fn unlikely() {}

/// Defines `$name(w, lo)`: stooge sort of the `$len` elements `w[lo..lo + $len]`, unrolled,
/// with `$sub` the unrolled stooge sort of its parts of `$len − ⌊$len/3⌋` elements.
macro_rules! unrolled_stooge {
    ($name:ident, $len:literal, $sub:ident) => {
        #[doc = concat!("Stooge-sorts `w[lo..lo + ", stringify!($len), "]`, unrolled.")]
        #[inline(always)]
        fn $name<T: Ord, const M: usize>(w: &mut [T; M], lo: usize) {
            cx(w, lo, lo + $len - 1);
            $sub(w, lo);
            $sub(w, lo + $len / 3);
            $sub(w, lo);
        }
    };
}

/// Stooge-sorts `w[lo..lo + 2]`: a single compare-exchange.
#[inline(always)]
fn net2<T: Ord, const M: usize>(w: &mut [T; M], lo: usize) {
    cx(w, lo, lo + 1);
}
unrolled_stooge!(net3, 3, net2);
unrolled_stooge!(net4, 4, net3);
unrolled_stooge!(net5, 5, net4);
unrolled_stooge!(net6, 6, net4);
unrolled_stooge!(net7, 7, net5);
unrolled_stooge!(net8, 8, net6);
unrolled_stooge!(net9, 9, net6);

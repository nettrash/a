use core::mem::{self, MaybeUninit};
use core::ptr;
use core::slice;

use super::counting::STACK_BYTES;

/// Sorts a slice in place using cycle sort.
///
/// Cycle sort, described by B. K. Haddon in 1990, writes to the slice as few times as any sort
/// can. The permutation that sorts the slice splits into cycles (the element in slot `a`
/// belongs in slot `b`, the one in `b` belongs in `c`, and so on back to `a`), and cycle sort
/// rotates each of them once. It takes out the element of the cycle's first slot, finds its
/// final slot by counting the elements smaller than it, writes it there and takes the element
/// it displaces, and so on until the element in hand belongs in the first slot. Equal keys
/// share a block of slots: an element goes to the first slot of its key's block that does not
/// already hold an equal element. Every element is written at most once, straight into its
/// final slot, and an element whose slot already holds its key is not written at all: where
/// writes are slow or wear the memory out (flash, EEPROM), that is the point of the algorithm.
/// The price is the counting: each element is compared with the rest of the slice, `Θ(n²)`
/// comparisons on any input that is not in order already.
///
/// Six refinements keep the algorithm as cheap as cycle sort can be. It writes the same
/// elements to the same slots as the textbook version, and in the same order except on
/// strictly decreasing input:
///
/// * **Written cycle starts are passed over.** After a cycle, the textbook takes the next slot
///   as the next cycle start and counts for its element, even if an earlier cycle wrote it:
///   such an element is final, and the count finds nothing smaller. On random input that is a
///   third of all comparisons, about `0.5n²` of `1.5n²`. A bitmap on the stack (4 KiB, a window
///   of 32,768 slots) remembers the written slots. Measured on random input: 1.23–1.25 times
///   faster for 100 `u64`s and 1.45–1.5 times from 1000 on, 1.45–1.5 times for strings and 64-byte
///   elements (1.3–1.35 times on sawtooth strings); with four distinct keys 1.15–1.25 times
///   from 1000 elements on. On nearly sorted input, where few elements move, it gains nothing,
///   and its bookkeeping (about 40 ns a sort) makes 100 `u64`s 1.08 times slower. Slices
///   longer than the window open a new one at the cycle start that reaches its end.
/// * **Runs of written starts are passed over at once.** A written start is recognised by its
///   bit with a branch, and from there the whole run of written slots is passed over a word at
///   a time: 1.03–1.04 times faster than one bit per start on 100 `u64`s, the same from 1000
///   on. Passing over with no branch at all, by scanning the bits from every start, made each
///   count wait for that scan: up to 1.3 times slower on nearly sorted input of 100 `u64`s.
/// * **Sorted and reverse-sorted input is recognised.** A slice found non-decreasing is left
///   alone, and a strictly decreasing one is reversed, which writes the same elements to the
///   same slots as the cycles would. This costs `n − 1` or `n` comparisons instead of `Θ(n²)`,
///   18–460 times faster from 100 to 3000 elements, and usually two on other input.
/// * **The element in hand.** The textbook holds the element being placed in a variable and
///   swaps it with the slot it belongs in. Generic Rust cannot copy an element out without
///   `Clone`, so its safe form keeps the element in the cycle's first slot and swaps that slot
///   with the target, writing the first slot again at every step. Here the element is moved
///   out once, the first slot stays vacant until the cycle closes, and each step writes only
///   the target slot: exactly one write per element placed. A guard writes the element in hand
///   into the vacant slot if a comparison panics. The speed is the same, within 4% either way.
/// * **Long runs of equal small elements are skipped 16 at a time.** For elements of up to
///   8 bytes, once 32 elements equal to the one being placed have been passed one at a time,
///   they are compared 16 at a time, which integers do in a few vector instructions: 1.06 times
///   faster on 1000 `u64`s with four distinct keys and 1.17 times on 3000, at most 1.06 times
///   slower on 200–300 of them, and the same with 32 distinct keys.
///
/// * **Narrow elements are counted in counters as narrow as they are.** A count compares every
///   element with no branch on the answer, so integers are compared several at a time, as in
///   the textbook's loop. Into one `usize`, the compiler widened the answer of every 1-, 2- and
///   4-byte element to 64 bits and so compared two elements per vector instruction, as for
///   `u64`. Here such elements are counted into counters of their own width, `u8`, `u16` or
///   `u32`, added to the total every 192 elements for `u8` and every 32,768 for the others,
///   before they can overflow: 16, 8 or 4 elements per instruction. On random input this is
///   2.6–3.3 times faster for `u32` from 1000 elements on, 4.1–6.6 times for `u16` and
///   3.9–5 times for `u8` (1.3–1.65 times on 100 elements), and 2.2–5.2 times with four
///   distinct keys; elements of more than 4 bytes are counted as before.
///
/// # Properties
///
/// * **Not stable**: equal elements may be reordered.
/// * **In place**: a bitmap of 4 KiB on the stack, no allocation.
/// * **Minimal writes**: every element whose slot does not already hold its key is written
///   once, into its final slot, and no other element is written.
/// * **Not adaptive**, except to sorted and strictly decreasing input. Elements already in
///   place are not written, but they are still counted for.
///
/// | Case                  | Comparisons                 | Writes                                |
/// |-----------------------|-----------------------------|---------------------------------------|
/// | Non-decreasing        | `n − 1`                     | `0`                                   |
/// | Strictly decreasing   | `n`                         | `2⌊n/2⌋`                              |
/// | Random, distinct keys | `≈ n²` (textbook `≈ 1.5n²`) | `n` minus the elements already placed |
/// | Worst                 | `Θ(n²)`                     | `n`                                   |
///
/// With duplicates, placing an element also compares it with the equal elements already in its
/// key's block (for elements of up to 8 bytes, at most 16 more per run of 32 or more).
///
/// Against the textbook cycle sort as a whole: 1.2–1.3 times faster on 100 random `u64`s,
/// 1.45–1.5 times from 1000 on, and 1.45–1.5 times on random strings and 64-byte elements;
/// 1.15–1.45 times with four distinct keys from 1000 elements on, 11–7000 times on sorted and
/// reverse-sorted input, and as fast on nearly sorted input from 1000 elements on. On up to
/// 200 nearly sorted `u64`s the textbook is 1.08–1.2 times faster, and on 100 `u64`s with four
/// distinct keys 1.04 times.
///
/// Against [`selection`](fn@super::selection) sort, which also moves few elements (at most
/// `n − 1` swaps, up to `2(n − 1)` writes): 2–2.6 times faster on random `u64`s and 4–5.1
/// times on nearly sorted ones, whose counts are vectorised; but selection sort is 1.1–2.3
/// times faster on random strings and 64-byte elements, 1.05–1.2 times on nearly sorted ones
/// and 1.6–2.5 times on sawtooth ones, and, as it gathers all equal keys in each pass,
/// 2.25–700 times faster with four distinct keys. Cycle sort is quadratic, and only worth it
/// where a write costs far more than a comparison: [`slice::sort_unstable`] is 3–6 times
/// faster on 100 random elements, 12–50 times on 1000 and 90–200 times on 10,000 (125 times on
/// 3000 strings, 17–30 times on 5000 random `u32`s, `u16`s or `u8`s), and as fast on sorted
/// and reverse-sorted input.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: nothing is dropped twice or leaked. An [`Ord`] that is not a total
/// order leaves `arr` in an unspecified order with every element exactly once as well, and the
/// sort still ends: a cycle is cut off after as many writes as it has slots.
///
/// # Examples
///
/// ```
/// use a::sort::cycle;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// cycle(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Only the elements in the wrong slot are written: here the 0 and the 2.
/// let mut v = [2, 1, 0, 3];
/// cycle(&mut v);
/// assert_eq!(v, [0, 1, 2, 3]);
///
/// let mut words = vec!["pear", "apple", "fig", "apple"];
/// cycle(&mut words);
/// assert_eq!(words, ["apple", "apple", "fig", "pear"]);
/// ```
pub fn cycle<T: Ord>(arr: &mut [T]) {
    let len = arr.len();
    if len < 2 {
        return;
    }
    // A non-decreasing slice has every element in place already, and a strictly decreasing
    // one is a set of swaps around the middle: reversing it writes each element once, into
    // its final slot, exactly as the cycles would.
    if arr.windows(2).all(|w| w[0] <= w[1]) {
        return;
    }
    if arr.windows(2).all(|w| w[1] < w[0]) {
        arr.reverse();
        return;
    }
    let mut placed = Placed::new();
    let base = arr.as_mut_ptr();
    let mut start = 0;
    while start < len - 1 {
        placed.slide(start, len);
        // A slot written by an earlier cycle holds its final element: counting would find
        // nothing smaller after it. The whole run of written slots is passed over at once.
        if placed.contains(start) {
            start = placed.next_vacant(start);
            continue;
        }
        // SAFETY: `start < len - 1`, and `base` comes from `arr`, of `len` elements, which is
        // not used again while the pointer is.
        unsafe { rotate(base, start, len, &mut placed) }
        start += 1;
    }
}

/// Number of slots [`Placed`] remembers at once: as many bits as fit in [`STACK_BYTES`].
const WINDOW: usize = STACK_BYTES * 8;

/// The slots of a window of the slice that have been written since the window opened,
/// one bit each, on the stack.
///
/// The window covers `lo..end`, at most [`WINDOW`] slots from the cycle start that opened it.
/// Slots past it are never remembered, so their cycle starts are counted as usual.
struct Placed {
    bits: [MaybeUninit<u64>; WINDOW / 64],
    lo: usize,
    end: usize,
}

impl Placed {
    /// An empty window, which the first call to [`slide`](Placed::slide) replaces.
    fn new() -> Self {
        Placed {
            bits: [MaybeUninit::uninit(); WINDOW / 64],
            lo: 0,
            end: 0,
        }
    }

    /// Opens a new window at `start` once `start` has left the current one; `len` is the
    /// length of the slice. Only the words the window uses are cleared, so a short slice does
    /// not pay for clearing all of them.
    #[inline(always)]
    fn slide(&mut self, start: usize, len: usize) {
        if start >= self.end {
            self.lo = start;
            self.end = start + WINDOW.min(len - start);
            let words = (self.end - start).div_ceil(64);
            for word in &mut self.bits[..words] {
                word.write(0);
            }
        }
    }

    /// Whether slot `p` of the current window, `lo <= p < end`, has been written.
    #[inline(always)]
    fn contains(&self, p: usize) -> bool {
        self.word(p - self.lo) & 1 != 0
    }

    /// Returns the bits from that of slot `lo + i` to the end of its word, `i < end - lo`.
    #[inline(always)]
    fn word(&self, i: usize) -> u64 {
        // SAFETY: `i < end - lo <= WINDOW`, so the word is in bounds, and `slide` initialised
        // every word below `(end - lo).div_ceil(64)`.
        let word = unsafe { self.bits.get_unchecked(i / 64).assume_init() };
        word >> (i % 64)
    }

    /// Returns the first slot from `p` on, `lo <= p < end`, that has not been written, or
    /// `end` if there is none: a run of written slots is passed over a word at a time, with no
    /// branch per slot.
    #[inline(always)]
    fn next_vacant(&self, mut p: usize) -> usize {
        loop {
            let i = p - self.lo;
            let word = self.word(i);
            let written = word.trailing_ones() as usize;
            p += written;
            // Bits from `end - lo` on are never set, so `p <= end`.
            if written < 64 - i % 64 || p >= self.end {
                return p;
            }
        }
    }

    /// Remembers that slot `p >= lo` has been written, if it lies in the window.
    #[inline(always)]
    fn insert(&mut self, p: usize) {
        if p < self.end {
            let i = p - self.lo;
            // SAFETY: as in `word`.
            unsafe {
                let word = self.bits.get_unchecked_mut(i / 64);
                word.write(word.assume_init() | 1 << (i % 64));
            }
        }
    }
}

/// Returns how many elements of `rest` are smaller than `item`.
///
/// Every element is compared, with no branch on the answer, so integers are compared several
/// at a time: eight `u64`s per step on Apple M4, and for elements of at most 4 bytes, counted
/// in counters of their width ([`count_in`]), 16 `u32`s or 64 `u8`s.
#[inline(always)]
fn count_less<T: Ord>(rest: &[T], item: &T) -> usize {
    if mem::size_of::<T>() == 1 {
        count_in::<T, u8>(rest, item)
    } else if mem::size_of::<T>() == 2 {
        count_in::<T, u16>(rest, item)
    } else if mem::size_of::<T>() <= 4 {
        count_in::<T, u32>(rest, item)
    } else {
        rest.iter().filter(|x| **x < *item).count()
    }
}

/// A counter as wide as the elements counted with it.
trait Lane: Copy {
    const ZERO: Self;
    /// The most elements one counter counts before it is added to the total: no more than it
    /// can hold.
    const CHUNK: usize;
    /// The counter plus one if `b`, else the counter.
    fn add(self, b: bool) -> Self;
    fn get(self) -> usize;
}

macro_rules! lane {
    ($t:ty, $chunk:expr) => {
        impl Lane for $t {
            const ZERO: Self = 0;
            const CHUNK: usize = {
                assert!($chunk <= <$t>::MAX as usize);
                $chunk
            };
            #[inline(always)]
            fn add(self, b: bool) -> Self {
                self.wrapping_add(b as $t)
            }
            #[inline(always)]
            fn get(self) -> usize {
                self as usize
            }
        }
    };
}
// Measured for `u8`: 192 elements (three times 64, what one step of the loop compares) were
// 1.07–1.25 times faster than 255 or 240 and 1.25–1.3 times faster than counting into `u16`s;
// 128 was as fast on 5000 elements and 1.18 times slower on 1000. For `u16` and `u32` the batch only
// has to be long enough for the sum after it not to matter.
lane!(u8, 192);
lane!(u16, 1 << 15);
lane!(u32, 1 << 15);

/// [`count_less`] with counters `C` as wide as the elements: in batches of `C::CHUNK`, each
/// counted into a `C` that cannot overflow and then added to the total.
#[inline(always)]
fn count_in<T: Ord, C: Lane>(rest: &[T], item: &T) -> usize {
    let mut total = 0;
    for chunk in rest.chunks(C::CHUNK) {
        let mut c = C::ZERO;
        for x in chunk {
            c = c.add(*x < *item);
        }
        total += c.get();
    }
    total
}

/// Returns the first slot from `pos` on, below `len`, that does not hold an element equal to
/// `item`, or `len` if there is none (which only an inconsistent [`Ord`] can cause).
///
/// Elements are compared one at a time, as the textbook does; for elements of up to 8 bytes,
/// a run of more than [`LONG_RUN`] goes on in [`skip_long_run`].
///
/// # Safety
///
/// `pos <= len` and `base` must be valid for `len` elements.
#[inline(always)]
unsafe fn skip_equal<T: Ord>(base: *const T, mut pos: usize, len: usize, item: &T) -> usize {
    // SAFETY: every slot read is below `len`, which is checked before each read.
    unsafe {
        let short = if mem::size_of::<T>() <= 8 {
            len.min(pos + LONG_RUN)
        } else {
            len
        };
        while pos < short {
            if *base.add(pos) != *item {
                return pos;
            }
            pos += 1;
        }
        if pos < len {
            skip_long_run(base, pos, len, item)
        } else {
            pos
        }
    }
}

/// From this many equal elements of up to 8 bytes on, [`skip_long_run`] takes over.
///
/// Measured on `u64`s with four distinct keys, whose runs are long: comparing 16 at a time
/// from the first element on was up to 1.08 times slower on 100–300 elements, where most runs
/// are a few dozen long and the extra exit mispredicts; from 32 on, at most 1.05 times.
const LONG_RUN: usize = 32;

/// Returns what [`skip_equal`] does, for a long run of equal elements of up to 8 bytes: they
/// are compared 16 at a time, which integers do in a few vector instructions, and the 16 that
/// end the run are compared again one by one, to find where. That is at most 16 comparisons
/// more than one at a time.
///
/// It is not inlined: it runs once per long run only, and keeps the code of the counting loop
/// small.
///
/// # Safety
///
/// `pos <= len` and `base` must be valid for `len` elements.
#[inline(never)]
unsafe fn skip_long_run<T: Ord>(base: *const T, mut pos: usize, len: usize, item: &T) -> usize {
    const CHUNK: usize = 16;
    // SAFETY: every slot read is below `len`: `pos + CHUNK <= len` or `pos < len` is checked
    // before each read.
    unsafe {
        while pos + CHUNK <= len {
            let chunk = slice::from_raw_parts(base.add(pos), CHUNK);
            let mut all = true;
            for x in chunk {
                all &= *x == *item;
            }
            if !all {
                break;
            }
            pos += CHUNK;
        }
        while pos < len && *base.add(pos) == *item {
            pos += 1;
        }
        pos
    }
}

/// Rotates the cycle that starts at `v[start]`, `v` being the slice of `len` elements at
/// `base`, of which `v[..start]` holds the `start` smallest elements in order: every element
/// of the cycle is written once, into its final slot.
///
/// # Safety
///
/// `start + 1 < len` and `base` must be valid for reads and writes of `len` elements.
unsafe fn rotate<T: Ord>(base: *mut T, start: usize, len: usize, placed: &mut Placed) {
    // SAFETY: `count_less` returns at most `len - start - 1`, so `start + c < len`, and
    // `skip_equal` returns at most `len`, which is checked: every slot touched is in bounds.
    // The element at `start` is copied into `hand` only once a slot for it has been found,
    // which leaves `v[start]` vacant: it is the only slot the counts and skips never read.
    // Each step then swaps the element in hand with the one in its slot. Until the end of the
    // cycle `_hole` owns the vacant slot, so if a comparison panics, the element in hand is
    // written there and `v` holds every element exactly once.
    unsafe {
        let rest = slice::from_raw_parts(base.add(start + 1), len - start - 1);
        let item = &*base.add(start);
        let c = count_less(rest, item);
        if c == 0 {
            return;
        }
        let pos = skip_equal(base, start + c, len, item);
        if pos == len {
            return;
        }
        let mut hand = MaybeUninit::<T>::uninit();
        let h = hand.as_mut_ptr();
        ptr::copy_nonoverlapping(base.add(start), h, 1);
        let _hole = Hole {
            src: h,
            dst: base.add(start),
        };
        ptr::swap_nonoverlapping(h, base.add(pos), 1);
        placed.insert(pos);
        // A consistent order fills each of `v[start + 1..]` at most once per cycle; an
        // inconsistent one could go around forever, so the cycle is cut off after that many
        // writes.
        let mut writes_left = len - start - 2;
        loop {
            let rest = slice::from_raw_parts(base.add(start + 1), len - start - 1);
            let c = count_less(rest, &*h);
            if c == 0 || writes_left == 0 {
                break;
            }
            let pos = skip_equal(base, start + c, len, &*h);
            if pos == len {
                break;
            }
            ptr::swap_nonoverlapping(h, base.add(pos), 1);
            placed.insert(pos);
            writes_left -= 1;
        }
        // Dropping `_hole` writes the element in hand, which belongs at `start`, there.
    }
}

/// A vacant slot of a slice, plus the element held outside the slice that belongs there.
///
/// Writing the held element on drop is what keeps [`rotate`] panic-safe: the slice is whole
/// again even if a comparison unwinds while the element is out.
struct Hole<T> {
    src: *const T,
    dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to the single vacant slot of the slice being sorted, and `src`
        // to the held element, which is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) }
    }
}

use core::hint::select_unpredictable;
use core::mem::{self, ManuallyDrop};
use core::ptr;

use super::bubble::{GIVE_UP_AFTER, Held, Hole, branchless_pays, forward, scan_apart, step};

/// Sorts a slice in place using cocktail shaker sort.
///
/// Cocktail shaker sort is bubble sort that alternates direction. A forward pass walks the
/// unsorted range from left to right, carrying the largest element seen so far towards the
/// end; a backward pass then walks it from right to left, carrying the smallest element seen
/// so far towards the front. Each pass leaves the elements beyond its last move in their final
/// places, so the unsorted range shrinks from both ends, and a pass that moves nothing ends
/// the sort.
///
/// Bubble sort moves a small element near the end (a "turtle") only one slot left per pass,
/// so a single one costs it `n - 1` passes. Here the next backward pass carries it all the way
/// to its place: `[1, 2, ..., n - 1, 0]` costs `3n - 6` comparisons (for `n ≥ 3`) instead of
/// the `n(n - 1)/2` of bubble sort. On random input it makes about a third fewer comparisons
/// than bubble sort (`≈ n²/3` against `≈ n²/2`); on reverse-sorted input both make
/// `n(n - 1)/2`.
///
/// Three refinements keep the algorithm as cheap as cocktail shaker sort can be:
///
/// * **Last-move tracking at both ends.** A forward pass ends the unsorted range at the last
///   element it moved, and a backward pass starts it just after the last element it moved, so
///   already-sorted input finishes after a single pass of `n - 1` comparisons and sorted runs
///   at either end are never rescanned.
/// * **Hole shifting instead of swapping.** An element travelling `k` slots is read once, the
///   `k` elements it overtakes are each shifted by one slot towards where it came from as
///   they are compared with it, and it is written once into its final slot: `k + 1` moves
///   instead of the `3k` moves of `k` swaps. Shifting while comparing, rather than finding the
///   distance first and moving a block afterwards, touches each element once and avoids a
///   `memmove` call per carry, most of which are only a slot or two long; on random `u64`
///   input that is about twice as fast.
/// * **Branchless passes for small elements.** Whether a comparison starts or ends a carry is
///   hard to predict on random input. For elements of at most 8 bytes with no drop glue, a
///   pass can instead hold the running extreme in a register and, at every comparison, write
///   the other element into the vacant slot behind it, with no branch on the outcome; an
///   element that is not overtaken is merely written back to its own slot. That costs more
///   than predictable branches, so each pass picks its kind from the previous one: a branchy
///   pass that turns out to carry often gives way to a branchless one from where it stands,
///   and a branchless pass that finds carries rare hands back to branchy ones. On random
///   `u64` input this is 1.8–2.6 times faster up to a few hundred elements and 1.2 times at
///   1,000 (at 5,000, where the branches are better predicted, it is as fast as branching),
///   and on input whose elements are each a few slots from their places 1.5–2.6 times
///   faster; already sorted input is still done by one branchy pass.
///
/// The number of comparisons is exactly that of classic cocktail shaker sort with early exit
/// and last-swap bounds at both ends, and so is every outcome. Zero-sized elements, which are
/// all alike, are left as they are without comparing them.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order (only strictly smaller elements are
///   overtaken going right, and only strictly greater ones going left).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the cost grows with how far elements are from their sorted positions, in
///   either direction.
///
/// Let `I` be the number of inversions and `C` the number of carries (elements picked up and
/// moved at least one slot, counted once per pass). Each overtaken element moves one slot and
/// removes one inversion, so every inversion costs exactly one move, and each carry costs one
/// more for the final write.
///
/// | Case                      | Comparisons     | Moves                 |
/// |---------------------------|-----------------|-----------------------|
/// | Best (already sorted)     | `n - 1`         | `0`                   |
/// | Any input                 | `≤ n(n - 1)/2`  | `I + C ≤ 2I`          |
/// | Average                   | `O(n²)`         | `O(n²)`               |
/// | Worst (reverse sorted)    | `n(n - 1)/2`    | `n(n + 1)/2 - 1`      |
///
/// Cocktail shaker sort is quadratic and is mainly of educational value; prefer
/// [`slice::sort`] for real workloads.
///
/// # Panics
///
/// Does not panic by itself. If the [`Ord`] implementation of `T` panics, the panic is
/// propagated and `arr` is left in an unspecified order, but it still holds every original
/// element exactly once: nothing is dropped twice or leaked.
///
/// # Examples
///
/// ```
/// use a::sort::cocktail_shaker;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// cocktail_shaker(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// // The 0 at the end reaches the front in one backward pass.
/// let mut v = [1, 2, 3, 4, 5, 0];
/// cocktail_shaker(&mut v);
/// assert_eq!(v, [0, 1, 2, 3, 4, 5]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// cocktail_shaker(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn cocktail_shaker<T: Ord>(arr: &mut [T]) {
    if mem::size_of::<T>() == 0 {
        // Zero-sized elements are all alike: there is nothing to reorder.
        return;
    }
    let base = arr.as_mut_ptr();
    // Everything before `lo` and from `hi` onwards is already in its final position.
    let (mut lo, mut hi) = (0, arr.len());
    // SAFETY: `lo` only grows and `hi` only shrinks, so every pass works inside `arr`, on a
    // range of at least two elements.
    unsafe {
        if !branchless_pays::<T>() {
            while hi - lo > 1 {
                let s = base.add(lo);
                let Ok(pass) = forward::<T, false>(s, hi - lo, 0, 0) else {
                    unreachable!("a pass that never gives up gave up")
                };
                hi = lo + pass.last;
                if hi - lo <= 1 {
                    break;
                }
                let Ok(pass) = backward::<T, false>(s, hi - lo) else {
                    unreachable!("a pass that never gives up gave up")
                };
                lo += pass.first;
            }
            return;
        }
        let mut branchy = true;
        while hi - lo > 1 {
            let s = base.add(lo);
            let n = hi - lo;
            let resume = if branchy {
                match forward::<T, true>(s, n, 0, CARRY_COST) {
                    Ok(pass) => {
                        branchy = !branchless_wins(pass.carries, n - 1);
                        hi = lo + pass.last;
                        None
                    }
                    Err(resume) => Some(resume),
                }
            } else {
                Some((0, 0))
            };
            if let Some((j0, last)) = resume {
                let (last, moves) = forward_branchless(s, n, j0, last);
                let comparisons = n - 1 - j0;
                // A branchless pass does not see its carries; it makes at most
                // `min(moves, comparisons - moves)` of them.
                branchy = !branchless_wins(moves.min(comparisons - moves), comparisons);
                hi = lo + last;
            }
            if hi - lo <= 1 {
                break;
            }
            let n = hi - lo;
            let resume = if branchy {
                match backward::<T, true>(s, n) {
                    Ok(pass) => {
                        branchy = !branchless_wins(pass.carries, n - 1);
                        lo += pass.first;
                        None
                    }
                    Err(resume) => Some(resume),
                }
            } else {
                Some((n - 1, n))
            };
            if let Some((i0, first)) = resume {
                let (first, moves) = backward_branchless(s, i0, first);
                branchy = !branchless_wins(moves.min(i0 - moves), i0);
                lo += first;
            }
        }
    }
}

/// A branchless pass runs at the latency of a comparison, about twice the cost of a predicted
/// branchy step, while a carry costs a branchy pass up to two mispredictions: branchless
/// passes win once there is a carry per `CARRY_COST` comparisons. Tuned on `u64` from 16 to
/// 1,000 elements.
const CARRY_COST: usize = 16;

/// Passes shorter than this are judged as if they were this long, so that one carry in a
/// short pass does not make the next one branchless.
const JUDGE_MIN: usize = 32;

/// Whether a pass making `carries` carries in `comparisons` comparisons is cheaper branchless.
fn branchless_wins(carries: usize, comparisons: usize) -> bool {
    CARRY_COST * carries >= comparisons.max(JUDGE_MIN)
}

/// What a complete branchy backward pass found.
struct BackPass {
    /// One past the final index of the last element moved (`n` if none was): everything
    /// before there is in its final position.
    first: usize,
    /// The number of carries.
    carries: usize,
}

/// One backward pass over `base[..n]`, `n >= 2`, carrying small elements to the left and
/// branching on each comparison.
///
/// With `GIVE_UP`, the pass stops at the end of a carry once it has scanned at least
/// [`GIVE_UP_AFTER`] elements and makes a carry per [`CARRY_COST`] comparisons. It then
/// returns `Err((i0, first))`: `base[i0]` is the running minimum, `base[i0 + 1..n]` holds what
/// the pass has finished with, and `first` is one past the final index of its last move so
/// far, so that a branchless pass can take over from `i0`.
///
/// # Safety
///
/// `base[..n]` must be valid for reads and writes and hold initialised elements.
#[inline(always)]
unsafe fn backward<T: Ord, const GIVE_UP: bool>(
    base: *mut T,
    n: usize,
) -> Result<BackPass, (usize, usize)> {
    // SAFETY: `p` walks `base[1..n]` downwards, so every pointer stays inside `base[..n]`;
    // `carry_left` keeps the slice whole.
    unsafe {
        let mut first = n;
        let mut carries = 0;
        // Compares `*p.sub(1)` with `*p`.
        let mut p = base.add(n - 1);
        // What a carry ending at `p` leaves to do: record it, stop at the front, and maybe
        // give up.
        macro_rules! after_carry {
            () => {
                let at = p.offset_from_unsigned(base);
                first = at + 1;
                if GIVE_UP {
                    carries += 1;
                }
                if at == 0 {
                    break;
                }
                if GIVE_UP {
                    let seen = n - at;
                    if seen >= GIVE_UP_AFTER && CARRY_COST * carries >= seen {
                        return Err((at - 1, first));
                    }
                }
            };
        }
        if scan_apart::<T, GIVE_UP>() {
            loop {
                while p > base && !(*p < *p.sub(1)) {
                    p = p.sub(1);
                }
                if p == base {
                    break;
                }
                p = carry_left(p, base);
                after_carry!();
                p = p.sub(1);
            }
        } else {
            while p > base {
                if *p < *p.sub(1) {
                    p = carry_left(p, base);
                    after_carry!();
                }
                p = p.sub(1);
            }
        }
        Ok(BackPass { first, carries })
    }
}

/// Carries `*p`, the smallest element seen in a backward pass, left past every strictly
/// greater element before it, shifting each of those right by one slot. Returns the carried
/// element's final slot: `base`, or just after the first element not greater than it.
///
/// # Safety
///
/// `base[..=p]` must be valid for reads and writes and hold initialised elements, `p > base`,
/// and `*p < *p.sub(1)`.
#[inline(always)]
unsafe fn carry_left<T: Ord>(p: *mut T, base: *mut T) -> *mut T {
    // SAFETY: the vacant slot stays above `base` whenever the element before it is read. The
    // element at `p` is moved into `hole`, which leaves exactly one slot logically
    // uninitialised: `hole.dst`. Each step moves the previous element into that slot and moves
    // `hole.dst` back onto the one just vacated, so source and destination never overlap, and
    // the element compared next is never the vacant one. When `hole` is dropped — normally, or
    // while unwinding from a panicking comparison — it writes the carried element back into
    // `hole.dst`, so the slice again holds every element exactly once.
    unsafe {
        let mut hole = Hole {
            value: ManuallyDrop::new(ptr::read(p)),
            dst: p,
        };
        loop {
            let prev = hole.dst.sub(1);
            ptr::copy_nonoverlapping(prev, hole.dst, 1);
            hole.dst = prev;
            if prev == base || *prev.sub(1) <= *hole.value {
                return prev;
            }
        }
    }
}

/// Runs a forward pass over `base[..n]` that is under way — its running maximum at `j0`,
/// everything before `j0` finished, `last` the final index of its last move so far — to the
/// end without branching on comparisons. Returns the final index of the last move and the
/// number of elements overtaken from `j0` on.
///
/// # Safety
///
/// `base[..n]` must be valid for reads and writes and hold initialised elements, and
/// `j0 < n`.
#[inline(always)]
unsafe fn forward_branchless<T: Ord>(
    base: *mut T,
    n: usize,
    j0: usize,
    mut last: usize,
) -> (usize, usize) {
    // SAFETY: every index is `< n`. The running maximum is held in `held`, which leaves one
    // slot logically uninitialised, the guard's `dst`; [`step`] keeps that true. When the
    // guard is dropped — normally, or while unwinding from a panicking comparison — it writes
    // the held element into its slot. `held` is accessed only through `hp` once that exists.
    unsafe {
        let mut held = ManuallyDrop::new(ptr::read(base.add(j0)));
        let hp: *mut ManuallyDrop<T> = &mut held;
        let mut guard = Held {
            src: hp.cast::<T>(),
            dst: base.add(j0),
        };
        let mut moves = 0;
        for j in j0 + 1..n {
            let moved = step(hp, &mut guard.dst, base.add(j));
            last = select_unpredictable(moved, j, last);
            moves += usize::from(moved);
        }
        (last, moves)
    }
}

/// Runs a backward pass that is under way — its running minimum at `i0`, everything after
/// `i0` finished, `first` one past the final index of its last move so far — to the front
/// without branching on comparisons. Returns one past the final index of the last move and
/// the number of elements overtaken from `i0` on.
///
/// # Safety
///
/// `base[..=i0]` must be valid for reads and writes and hold initialised elements.
#[inline(always)]
unsafe fn backward_branchless<T: Ord>(base: *mut T, i0: usize, mut first: usize) -> (usize, usize) {
    // SAFETY: every index is `<= i0`. As in `forward_branchless`, with the held element the
    // running minimum and the vacant slot moving left.
    unsafe {
        let mut held = ManuallyDrop::new(ptr::read(base.add(i0)));
        let hp: *mut ManuallyDrop<T> = &mut held;
        let mut guard = Held {
            src: hp.cast::<T>(),
            dst: base.add(i0),
        };
        let mut moves = 0;
        for i in (1..=i0).rev() {
            // The element at `i - 1` is overtaken if it is strictly greater than the held one.
            let moved = step_back(hp, &mut guard.dst, base.add(i - 1));
            first = select_unpredictable(moved, i, first);
            moves += usize::from(moved);
        }
        (first, moves)
    }
}

/// The mirror image of [`step`]: writes the greater of the element at `x` and the held one
/// into the vacant slot `*hole` (the held one if they are equal), keeps the smaller held, and
/// makes `x` the vacant slot. Returns whether the element at `x` was greater, that is,
/// overtaken.
///
/// # Safety
///
/// As for [`step`].
#[inline(always)]
unsafe fn step_back<T: Ord>(held: *mut ManuallyDrop<T>, hole: &mut *mut T, x: *mut T) -> bool {
    // SAFETY: as for `step`; each of the two elements is written exactly once.
    unsafe {
        let greater = !(*x <= **held);
        let stays = select_unpredictable(greater, ManuallyDrop::new(ptr::read(x)), ptr::read(held));
        let goes_on =
            select_unpredictable(greater, ptr::read(held), ManuallyDrop::new(ptr::read(x)));
        ptr::write(*hole, ManuallyDrop::into_inner(stays));
        ptr::write(held, goes_on);
        *hole = x;
        greater
    }
}

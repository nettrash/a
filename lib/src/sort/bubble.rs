use core::hint::select_unpredictable;
use core::mem::{self, ManuallyDrop};
use core::ptr;

/// Sorts a slice in place using bubble sort.
///
/// Each pass walks the unsorted prefix from left to right, carrying the largest element
/// seen so far towards the end until it meets an element that is not smaller than it.
/// After a pass, everything from the position of the last move onwards is in its final
/// place, so the next pass stops there; a pass that moves nothing ends the sort.
///
/// Four refinements keep the algorithm as cheap as bubble sort can be:
///
/// * **Last-move tracking.** The next pass scans only up to the last element that moved,
///   so already-sorted input finishes after a single pass of `n - 1` comparisons and
///   partially sorted tails are never rescanned.
/// * **Hole shifting instead of swapping.** An element travelling `k` slots to the right is
///   read once, the `k` smaller elements are each shifted left by one, and the element is
///   written once into its final slot: `k + 1` moves instead of the `3k` moves of `k` swaps.
/// * **Branchless passes for small elements.** Whether a comparison starts or ends a carry is
///   hard to predict on most inputs, and each misprediction costs as much as a dozen
///   comparisons. For elements of at most 8 bytes with no drop glue (integers, `char`,
///   `bool`, references), a pass can instead hold the running maximum in a register and, at
///   every comparison, write the smaller of it and the next element into the slot behind,
///   with no branch on the outcome. The comparisons and their outcomes are the same; an
///   element that is not overtaken is merely written back to its own slot.
/// * **Two passes at once.** Each branchless step waits for the one before it, so a single
///   pass runs at the latency of a comparison. Pass `k + 1` only reads slots that pass `k`
///   has finished with, so it can trail pass `k` through the slice in the same loop: two
///   independent chains of comparisons, each made exactly as if the passes ran one after
///   the other; on random `u64` input up to 1.3 times the speed of one branchless pass.
///
/// Branchless passes cost more than predictable branches, so the sort picks the kind of each
/// pass from the previous one: a branchy pass that turns out to carry often gives way to
/// branchless ones from where it stands, and branchless passes that move few elements hand
/// back to branchy ones. Already sorted input is done by one branchy pass, as before.
/// On random `u64` and `u32` input this sorts 1.2–2.9 times faster than branching on every
/// comparison (most around a hundred elements), and input whose elements are each a few slots
/// from their places up to 3.3 times faster.
///
/// The number of comparisons is exactly that of classic bubble sort with early exit, and so
/// is every outcome; only the order in which two consecutive passes make them is interleaved.
/// Zero-sized elements, which are all alike, are left as they are without comparing them.
///
/// # Properties
///
/// * **Stable**: equal elements keep their relative order (only strictly smaller elements
///   are overtaken).
/// * **In place**: `O(1)` auxiliary memory, no allocation.
/// * **Adaptive**: the cost grows with how far elements are from their sorted positions.
///
/// | Case                      | Comparisons | Moves  |
/// |---------------------------|-------------|--------|
/// | Best (already sorted)     | `O(n)`      | `0`    |
/// | Average                   | `O(n²)`     | `O(n²)`|
/// | Worst (reverse sorted)    | `O(n²)`     | `O(n²)`|
///
/// Bubble sort is quadratic and is mainly of educational value; prefer
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
/// use a::sort::bubble;
///
/// let mut v = [5, 1, 4, 2, 8, 0];
/// bubble(&mut v);
/// assert_eq!(v, [0, 1, 2, 4, 5, 8]);
///
/// let mut words = vec!["pear", "apple", "fig"];
/// bubble(&mut words);
/// assert_eq!(words, ["apple", "fig", "pear"]);
/// ```
pub fn bubble<T: Ord>(arr: &mut [T]) {
    if mem::size_of::<T>() == 0 {
        // Zero-sized elements are all alike: there is nothing to reorder.
        return;
    }
    let base = arr.as_mut_ptr();
    // Everything at index `n` and beyond is already in its final position.
    let mut n = arr.len();
    // SAFETY: `n` starts at `arr.len()` and only shrinks, so every pass works inside `arr`.
    unsafe {
        if !branchless_pays::<T>() {
            while n > 1 {
                let Ok(pass) = forward::<T, false>(base, n, 0, 0) else {
                    unreachable!("a pass that never gives up gave up")
                };
                n = pass.last;
            }
            return;
        }
        let mut branchy = true;
        while n > 1 {
            let (j0, last) = if branchy {
                match forward::<T, true>(base, n, BRANCH_MOVE_COST, BRANCH_CARRY_COST) {
                    Ok(pass) => {
                        branchy = BRANCH_MOVE_COST * pass.moves + BRANCH_CARRY_COST * pass.carries
                            < n - 1;
                        n = pass.last;
                        continue;
                    }
                    Err(resume) => resume,
                }
            } else {
                (0, 0)
            };
            let pair = two_passes(base, n, j0, last);
            // A branchless pass does not see its carries; it makes at most `min(moves,
            // comparisons - moves)` of them.
            let carries = pair.moves.min(pair.comparisons - pair.moves);
            branchy =
                BRANCH_MOVE_COST * pair.moves + BRANCH_CARRY_COST * carries < pair.comparisons;
            n = pair.last;
        }
    }
}

/// Whether passes over elements of type `T` may run without branching on comparisons.
///
/// Measured on the whole sort on random input: 1.2–2.9 times faster for `u32` and `u64`;
/// for `(u64, u64)`, `Box<str>` and `String` up to 2.3 times slower (each step waits for a
/// longer comparison), and for `Box<u64>` 1.5–2.6 times slower (each comparison first loads
/// what the held element points to). Owning pointers are kept out by their drop glue; plain
/// references cannot be told apart from integers, take the branchless path, and pay that
/// load in the same way.
pub(super) const fn branchless_pays<T>() -> bool {
    mem::size_of::<T>() <= 8 && !mem::needs_drop::<T>()
}

/// A branchy pass gives way to branchless ones once
/// `BRANCH_MOVE_COST · moves + BRANCH_CARRY_COST · carries` reaches the comparisons made: a
/// carry costs up to two mispredictions, while a move inside a predicted carry costs about
/// as much as a branchless step. Tuned on `u64` and `u32` from 16 to 5,000 elements.
const BRANCH_MOVE_COST: usize = 2;
const BRANCH_CARRY_COST: usize = 8;

/// A branchy pass checks whether to give up only after this many elements, so that a single
/// early carry does not decide for a whole pass.
pub(super) const GIVE_UP_AFTER: usize = 32;

/// Whether a branchy pass looks for its next carry in a loop of its own.
///
/// That loop compiles to one taken branch per element, while the scan and the carry in one
/// loop compiled to two for `u64` in the adaptive passes, which made a scan twice as slow;
/// elements of more than 32 bytes gain 1.1–1.8 times on nearly sorted input. Elements of 9
/// to 32 bytes, and those with drop glue, keep the single loop: with it, random `String`s
/// sort up to 1.2 times faster than with the separate one.
pub(super) const fn scan_apart<T, const GIVE_UP: bool>() -> bool {
    GIVE_UP || mem::size_of::<T>() > 32
}

/// What a complete branchy pass found.
pub(super) struct Pass {
    /// The final index of the last element moved (`0` if none was): everything from there
    /// onwards is in its final position.
    pub(super) last: usize,
    /// The number of carries (elements picked up and moved at least one slot).
    pub(super) carries: usize,
    /// The number of elements overtaken.
    pub(super) moves: usize,
}

/// One forward pass over `base[..n]`, `n >= 2`, branching on each comparison.
///
/// With `GIVE_UP`, the pass stops at the end of a carry once it has scanned at least
/// [`GIVE_UP_AFTER`] elements and `move_cost · moves + carry_cost · carries` has reached the
/// number of elements scanned, that is, once branches look expensive. It
/// then returns `Err((j0, last))`: `base[j0]` is the running maximum, `base[..j0]` holds what
/// the pass has finished with, and `last` is the final index of its last move so far, so that
/// a branchless pass can take over from `j0`.
///
/// # Safety
///
/// `base[..n]` must be valid for reads and writes and hold initialised elements.
#[inline(always)]
pub(super) unsafe fn forward<T: Ord, const GIVE_UP: bool>(
    base: *mut T,
    n: usize,
    move_cost: usize,
    carry_cost: usize,
) -> Result<Pass, (usize, usize)> {
    // SAFETY: `p` walks `base[1..n]`, so every pointer stays inside `base[..n]`;
    // `carry_right` keeps the slice whole.
    unsafe {
        let end = base.add(n);
        let mut last = base;
        let (mut carries, mut moves) = (0, 0);
        let mut p = base.add(1);
        // What a carry ending at `p` leaves to do: record it, stop at the end, and maybe give
        // up.
        macro_rules! after_carry {
            ($from:expr) => {
                last = p.sub(1);
                if GIVE_UP {
                    carries += 1;
                    moves += p.offset_from_unsigned($from);
                }
                if p == end {
                    break;
                }
                if GIVE_UP {
                    let seen = p.offset_from_unsigned(base);
                    if seen >= GIVE_UP_AFTER && move_cost * moves + carry_cost * carries >= seen {
                        return Err((seen, last.offset_from_unsigned(base)));
                    }
                }
            };
        }
        if scan_apart::<T, GIVE_UP>() {
            loop {
                while p < end && !(*p < *p.sub(1)) {
                    p = p.add(1);
                }
                if p == end {
                    break;
                }
                let from = p;
                p = carry_right(p, end);
                after_carry!(from);
                p = p.add(1);
            }
        } else {
            while p < end {
                if *p < *p.sub(1) {
                    let from = p;
                    p = carry_right(p, end);
                    after_carry!(from);
                }
                p = p.add(1);
            }
        }
        Ok(Pass {
            last: last.offset_from_unsigned(base),
            carries,
            moves,
        })
    }
}

/// Carries `*p.sub(1)`, the largest element seen in a pass, right past every strictly
/// smaller element from `p` on, shifting each of those left by one slot. Returns the position
/// after the carried element's final slot: `end`, or the first element not smaller than it.
///
/// # Safety
///
/// `base[..end]` (for some `base < p`) must be valid for reads and writes and hold
/// initialised elements, `p < end`, and `*p < *p.sub(1)`.
#[inline(always)]
unsafe fn carry_right<T: Ord>(mut p: *mut T, end: *mut T) -> *mut T {
    // SAFETY: `p` stays below `end` whenever it is read. The element at `p - 1` is moved into
    // `hole`, which leaves exactly one slot logically uninitialised: `hole.dst`. Each step
    // moves the next element into that slot and advances `hole.dst` onto the one just
    // vacated, so source and destination never overlap. When `hole` is dropped — normally, or
    // while unwinding from a panicking comparison — it writes the carried element back into
    // `hole.dst`, so the slice again holds every element exactly once.
    unsafe {
        let mut hole = Hole {
            value: ManuallyDrop::new(ptr::read(p.sub(1))),
            dst: p.sub(1),
        };
        loop {
            ptr::copy_nonoverlapping(p, hole.dst, 1);
            hole.dst = p;
            p = p.add(1);
            if p == end || *p >= *hole.value {
                return p;
            }
        }
    }
}

/// What a pair of branchless passes found about the second (trailing) one.
struct Pair {
    /// The final index of the trailing pass's last move (`0` if none).
    last: usize,
    /// The number of comparisons the trailing pass made.
    comparisons: usize,
    /// The number of elements the trailing pass overtook.
    moves: usize,
}

/// Runs the pass that is under way over `base[..lim]` — its running maximum at `j0`,
/// everything before `j0` finished, `last_l` the final index of its last move so far — to the
/// end, and the next pass over `base[..last of the first]` trailing it, both branchless.
///
/// The trailing pass reads `base[k]` only once `k` is below the leading pass's last move so
/// far, which is never more than the slot the leading pass holds vacant: it only sees slots
/// the leading pass has finished with, and it would scan them anyway, since the leading pass's
/// last move can only grow. So it makes exactly the comparisons it would make after the
/// leading pass, with the same outcomes.
///
/// # Safety
///
/// `base[..lim]` must be valid for reads and writes and hold initialised elements, `lim >= 2`,
/// `j0 < lim`, and if `j0 > 0`, `last_l < j0`.
#[inline(always)]
unsafe fn two_passes<T: Ord>(base: *mut T, lim: usize, j0: usize, mut last_l: usize) -> Pair {
    // SAFETY: every index is `< lim`. Each pass holds its running maximum outside the slice,
    // in `held_l` or `held_t`, which leaves one slot per pass logically uninitialised: the
    // `dst` of its guard. The leading pass's vacant slot is at the index it has just
    // processed; the trailing pass's is below `k`, which is below `last_l`, which is at most
    // the leading pass's vacant slot, so the two are distinct and neither pass reads the
    // other's. When a guard is dropped — normally, or while unwinding from a panicking
    // comparison — it writes its held element into its vacant slot, so the slice again holds
    // every element exactly once. The held elements are accessed only through `hl` and `ht`
    // once those exist, so the guards' `src` pointers stay valid.
    unsafe {
        let mut held_l = ManuallyDrop::new(ptr::read(base.add(j0)));
        let hl: *mut ManuallyDrop<T> = &mut held_l;
        let mut gl = Held {
            src: hl.cast::<T>(),
            dst: base.add(j0),
        };
        let mut j = j0 + 1;
        if j0 == 0 {
            // Finish slot 0 before the trailing pass picks it up.
            if step(hl, &mut gl.dst, base.add(1)) {
                last_l = 1;
            }
            j = 2;
        }
        let mut held_t = ManuallyDrop::new(ptr::read(base));
        let ht: *mut ManuallyDrop<T> = &mut held_t;
        let mut gt = Held {
            src: ht.cast::<T>(),
            dst: base,
        };
        let (mut last_t, mut moves_t) = (0, 0);
        let mut k = 1;
        while j < lim {
            let moved = step(hl, &mut gl.dst, base.add(j));
            last_l = select_unpredictable(moved, j, last_l);
            j += 1;
            if k < last_l {
                let moved = step(ht, &mut gt.dst, base.add(k));
                last_t = select_unpredictable(moved, k, last_t);
                moves_t += usize::from(moved);
                k += 1;
            }
        }
        // The leading pass is done: its held element goes into its final slot, `lim - 1`, and
        // the trailing pass finishes alone.
        drop(gl);
        while k < last_l {
            let moved = step(ht, &mut gt.dst, base.add(k));
            last_t = select_unpredictable(moved, k, last_t);
            moves_t += usize::from(moved);
            k += 1;
        }
        drop(gt);
        Pair {
            last: last_t,
            comparisons: k - 1,
            moves: moves_t,
        }
    }
}

/// One branchless step of a pass: compares the element at `x` with the held one, writes the
/// smaller into the vacant slot `*hole` (the held one if they are equal, which keeps the sort
/// stable), keeps the larger held, and makes `x` the vacant slot. Returns whether the element
/// at `x` was smaller, that is, overtaken.
///
/// # Safety
///
/// `held` must point to an initialised element that belongs in the vacant slot `*hole`, and
/// `x` to an initialised element of the slice other than `*hole`; afterwards the held element
/// belongs in `x`. The comparison is the only call that can panic, and it runs before
/// anything is moved.
#[inline(always)]
pub(super) unsafe fn step<T: Ord>(
    held: *mut ManuallyDrop<T>,
    hole: &mut *mut T,
    x: *mut T,
) -> bool {
    // SAFETY: as documented above; each of the two elements is written exactly once, one into
    // the vacant slot and one into `held`, from bitwise copies that are never dropped.
    unsafe {
        let smaller = *x < **held;
        let stays = select_unpredictable(smaller, ManuallyDrop::new(ptr::read(x)), ptr::read(held));
        let goes_on =
            select_unpredictable(smaller, ptr::read(held), ManuallyDrop::new(ptr::read(x)));
        ptr::write(*hole, ManuallyDrop::into_inner(stays));
        ptr::write(held, goes_on);
        *hole = x;
        smaller
    }
}

/// An element temporarily moved out of a slice, plus the slot it will be written back to.
///
/// Writing back on drop is what keeps [`bubble`] panic-safe: the slice is whole again even
/// if a comparison unwinds while the element is out.
pub(super) struct Hole<T> {
    pub(super) value: ManuallyDrop<T>,
    pub(super) dst: *mut T,
}

impl<T> Drop for Hole<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to the single uninitialised slot of the slice being sorted,
        // and `value` is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(&*self.value, self.dst, 1) }
    }
}

/// A vacant slot of a slice, plus the element held outside the slice that belongs there.
///
/// Writing the held element back on drop is what keeps the branchless passes panic-safe.
pub(super) struct Held<T> {
    pub(super) src: *const T,
    pub(super) dst: *mut T,
}

impl<T> Drop for Held<T> {
    fn drop(&mut self) {
        // SAFETY: `dst` points to a vacant slot of the slice being sorted, and `src` to the
        // held element, which is never used again after this write.
        unsafe { ptr::copy_nonoverlapping(self.src, self.dst, 1) }
    }
}

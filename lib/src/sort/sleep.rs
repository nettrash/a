use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{PoisonError, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use super::counting::min_max;
use super::key::IntegerKey;

/// Sorts a slice of integers, `char`s or `bool`s by sleeping: one thread per element sleeps for
/// as long as its value says, and the values are written back in the order their threads wake.
/// Returns whether the slice came out sorted, which it may not have.
///
/// Sleep sort was posted anonymously to 4chan's programming board in 2011 as a shell script that
/// started `sleep $n; echo $n` in the background for every argument `n`: the small numbers wake
/// first and are printed first. It is a joke, and a good one: it sorts in "linear time", lets the
/// operating system's timer queue do all the work, and is wrong as soon as the scheduler is late.
/// This is that algorithm with real threads and real sleeping, generalised from non-negative
/// numbers to any [`IntegerKey`]: each element sleeps `(value − min) × unit` past a start all
/// threads share, so negative numbers, `char`s and `bool`s work too, and the smallest values
/// wake right at the start.
///
/// **It is nondeterministic and intended as a curiosity.** Nothing guarantees that a thread
/// meant to wake `unit` after another wakes after it: a thread scheduled late (a loaded machine,
/// a timer that fires late) lets a larger value overtake a smaller one, and the slice is then not
/// sorted. The return value says whether it is: `true` exactly when the slice ends up in order.
/// Whatever happens, the slice ends up a permutation of what it was: every value is written back
/// exactly once, to the next free place.
///
/// Four refinements make the realisation as faithful and as robust as real threads allow; the
/// shares of sorts that came out sorted below are for random keys in `0..n`, on a 16-core Mac
/// loaded by other benchmarks (load average 3–15), so they vary from run to run:
///
/// * **One shared start.** Every thread sleeps until `start + (value − min) × unit`, an instant
///   all threads share, not for a duration from whenever it happened to be started: the order
///   in which threads are started, at 15 µs each for 1000 of them, does not skew when they
///   wake. Sleeping from each thread's own start, 100 elements at 100 µs per unit and 1000
///   elements at 1 ms never came out sorted; from the shared start, almost always.
/// * **The clock starts once every thread is running, and a little later.** All threads are
///   started first and wait at a gate, a [`RwLock`] the caller holds for writing; then the
///   start is set and the gate opened. Waking the waiting threads takes the system 1.5–3 µs per
///   thread, so the start is set 4 µs per thread after the gate opens, by when every thread
///   has read it and gone to sleep: with the start at the gate, the threads of the smallest
///   values, released last, woke after the next ones, and 1000 elements at 100 µs never came
///   out sorted (100 elements: 64%; 1000 elements at 1 ms: 30%).
/// * **Sleeping in halves.** The system may let a sleep run longer than asked, to wake several
///   threads together: here a sleep of 1 ms took 1.5 ms, of 5 ms 7.5 ms, of 50 ms up to 60 ms,
///   so a value of 9 woke after a value of 12. Each thread sleeps for half the time left until
///   its deadline, again and again, and for all of it only once less than 50 µs is left: a
///   sleep that runs late still ends before the deadline, and the last one, being short, runs
///   late by a few dozen microseconds at most. At 1 ms per unit: from 39% to 100% sorted for 10
///   elements, from none to all for 100, and from none to 60–95% for 1000. Spinning through
///   the last microseconds instead of sleeping was no better, and worse with 1000 threads.
/// * **Small stacks.** Each thread gets a stack of 16 KiB instead of the default 2 MiB, which a
///   sleeping thread does not need: starting 5000 threads takes half as long.
///
/// The values are written back lock-free, each waking thread taking the next place from an
/// atomic counter; `(value − min) × unit` is computed in 128 bits and saturates at
/// [`Duration::MAX`], and whether the clock can represent the last deadline is checked before
/// any thread is started. If a thread cannot be started, the threads already waiting at the gate
/// are sent home without writing anything.
///
/// # Properties
///
/// * **Nondeterministic**: whether the result is sorted depends on the scheduler; it is
///   returned. Equal values wake together, in any order, which cannot be seen.
/// * **Time**: `(max − min) × unit`, plus starting `n` threads (about 15 µs each up to 1000
///   threads, more beyond: 0.27 s for 5000, 1 s for 10,000) and 4 µs per thread before the
///   start, whatever the order of the input: sleep sort is "linear" in the range of the values,
///   not in their number.
/// * **Not in place**: one thread per element, each with a 16 KiB stack: `O(n)` threads and
///   memory. A process may only have so many threads (16,384 on macOS).
///
/// The share of sorts that came out sorted on this machine, over three runs of 20–400 sorts
/// each (fewer for the longer sorts):
///
/// | `unit`   | 10 elements | 100 elements | 1000 elements |
/// |----------|-------------|--------------|---------------|
/// | 10 µs    | 12–18%      | 0%           | 0%            |
/// | 100 µs   | 94–100%     | 94–100%      | 89–100%       |
/// | 1 ms     | 99–100%     | 89–100%      | 60–90%        |
///
/// A unit of 10 µs is below what the timers resolve: a thread wakes 10–15 µs after its deadline
/// at best. A longer sort is longer exposed to the other work of the machine, which can hold a
/// thread back by a millisecond or more: 1000 elements at 1 ms per unit take a second, and came
/// out sorted less often than at 100 µs, which take a tenth of that.
///
/// Against [`slice::sort_unstable`] it is, of course, hopeless: the standard library sorts 1000
/// random `u32`s in 3.5 µs, sleep sort at 100 µs per unit in about 115 ms, 30,000 times as long,
/// and 100 in 0.2 µs against 11 ms.
///
/// # Panics
///
/// Panics, leaving the slice as it was, if
///
/// * the clock cannot represent when the largest value would wake, `(max − min) × unit` from
///   now: about 292 billion years on Unix-like platforms, for example the whole range of `i128`
///   at a nanosecond per unit, or of `u64` at a second (a zero `unit` never panics);
/// * the system cannot start a thread for every element (on macOS, a process may have 16,384):
///   the threads already started are sent home and joined first.
///
/// A range the clock can represent is slept through, however long: the whole range of `i64` at a
/// nanosecond per unit takes 584 years.
///
/// # Examples
///
/// ```
/// use a::sort::sleep;
/// use std::time::Duration;
///
/// let mut v = [5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
/// let sorted = sleep(&mut v, Duration::from_millis(2));
/// // Usually true, but the scheduler decides.
/// if sorted {
///     assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
/// }
/// // Always a permutation of the input.
/// v.sort_unstable();
/// assert_eq!(v, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
///
/// // Negative numbers sleep from the smallest one; a zero unit wakes everyone at once.
/// let mut w = [3, -2, i32::MIN, 7];
/// let sorted = sleep(&mut w, Duration::ZERO);
/// assert_eq!(sorted, w.is_sorted());
/// ```
///
/// A range the clock cannot represent panics:
///
/// ```should_panic
/// use a::sort::sleep;
/// use std::time::Duration;
///
/// let mut v = [i128::MIN, i128::MAX];
/// sleep(&mut v, Duration::from_nanos(1));
/// ```
pub fn sleep<T: IntegerKey + Send>(arr: &mut [T], unit: Duration) -> bool {
    let len = arr.len();
    if len < 2 {
        return true;
    }
    let (min, max) = min_max(arr);
    let span = scaled(unit, max.distance(min));
    let lead = LEAD_PER_THREAD.saturating_mul(u32::try_from(len).unwrap_or(u32::MAX));
    // The threads start after this, so this only rejects a range that cannot work at all; a
    // thread whose deadline is still unrepresentable later sleeps for its delay instead.
    if Instant::now()
        .checked_add(lead)
        .and_then(|t| t.checked_add(span))
        .is_none()
    {
        panic!("{SORT}: the clock cannot represent a sleep of {span:?}");
    }
    // `None` until the caller opens the gate with the start, and also if a thread cannot be
    // started, which sends the threads already waiting home without a value written.
    let gate: RwLock<Option<Instant>> = RwLock::new(None);
    let next = AtomicUsize::new(0);
    // From here until the threads are joined, the slice is only touched through `out`: a
    // reference to it held across the scope would alias the threads' writes.
    let out = Out(arr.as_mut_ptr());
    let failed = thread::scope(|s| {
        let mut open = gate.write().unwrap_or_else(PoisonError::into_inner);
        for i in 0..len {
            // SAFETY: `i < len`, and no thread writes before the gate opens, after this loop.
            let x = unsafe { out.read(i) };
            let delay = scaled(unit, x.distance(min));
            let (gate, next, out) = (&gate, &next, &out);
            let started = thread::Builder::new()
                .stack_size(STACK)
                .spawn_scoped(s, move || {
                    let Some(start) = *gate.read().unwrap_or_else(PoisonError::into_inner) else {
                        return;
                    };
                    match start.checked_add(delay) {
                        Some(deadline) => sleep_until(deadline),
                        None => thread::sleep(delay),
                    }
                    let place = next.fetch_add(1, Ordering::Relaxed);
                    // SAFETY: each of the `len` threads takes one place, and the counter hands
                    // out `0..len` once each, so every place of the slice is written by one
                    // thread only and nothing else touches the slice until the scope ends.
                    unsafe { out.write(place, x) };
                });
            if let Err(e) = started {
                // Dropping the guard as `None` releases the threads with nothing written.
                return Some((i, e));
            }
        }
        // The clock starts `lead` from now, by when every thread has read the start: waking
        // them all takes a few microseconds each.
        *open = Some(
            Instant::now()
                .checked_add(lead)
                .unwrap_or_else(Instant::now),
        );
        None
    });
    if let Some((i, e)) = failed {
        panic!("{SORT}: could not start thread {} of {len}: {e}", i + 1);
    }
    debug_assert_eq!(next.load(Ordering::Relaxed), len);
    arr.is_sorted()
}

/// The name the sort panics with.
const SORT: &str = "sleep sort";

/// The stack size of each thread: 16 KiB, the smallest the platforms allow (`std` rounds it up
/// where the minimum is larger). A thread that only waits, sleeps and writes one value needs a
/// few KiB. On macOS, starting threads with the smallest stacks is 2 times faster from about
/// 5000 threads on (0.27 s instead of 0.55 s for 5000, 1.0 s instead of 2.3 s for 10,000, as
/// even 20 KiB costs as much as the default 2 MiB), and as fast below.
const STACK: usize = 16 * 1024;

/// How long after opening the gate the clock starts, per thread. The gate wakes the waiting
/// threads one after the other, which took 1.5–2.9 µs per thread here (up to 2.9 ms for 1000):
/// a thread released after the start whose value is the smallest would wake after the next
/// ones. Starting the clock later by twice that leaves every thread time to read the start
/// and go to sleep: with a start right at the gate, 1000 elements at 1 ms per unit came out
/// sorted in about two of three sorts, almost always broken at the smallest values.
const LEAD_PER_THREAD: Duration = Duration::from_micros(4);

/// Below this much time left, [`sleep_until`] sleeps for all of it; above, for half of it.
const FINE: Duration = Duration::from_micros(50);

/// The slice the threads write their values back into, shared by all of them.
struct Out<T>(*mut T);

// SAFETY: the threads only write through the pointer with `Out::write`, each to a place of its
// own, and a `T` written from another thread is fine for a `T: Send`.
unsafe impl<T: Send> Sync for Out<T> {}

impl<T: Copy> Out<T> {
    /// Reads place `i`.
    ///
    /// # Safety
    ///
    /// `i` must be in bounds of the slice, and no other thread may write place `i` at the same
    /// time.
    #[inline(always)]
    unsafe fn read(&self, i: usize) -> T {
        // SAFETY: guaranteed by the caller.
        unsafe { self.0.add(i).read() }
    }

    /// Writes `x` to place `i`.
    ///
    /// # Safety
    ///
    /// `i` must be in bounds of the slice, and no other thread may access place `i` at the same
    /// time.
    #[inline(always)]
    unsafe fn write(&self, i: usize, x: T) {
        // SAFETY: guaranteed by the caller; the old value is a `Copy` key, nothing to drop.
        unsafe { self.0.add(i).write(x) };
    }
}

/// `unit × times`, or [`Duration::MAX`] if that is longer: the product of a 128-bit distance and
/// a duration of up to `2^64` seconds needs more than 128 bits of nanoseconds.
fn scaled(unit: Duration, times: u128) -> Duration {
    const NANOS_PER_SEC: u128 = 1_000_000_000;
    let nanos = unit.as_nanos().saturating_mul(times);
    match u64::try_from(nanos / NANOS_PER_SEC) {
        Ok(secs) => Duration::new(secs, (nanos % NANOS_PER_SEC) as u32),
        Err(_) => Duration::MAX,
    }
}

/// Sleeps until `deadline`, in halves: while more than [`FINE`] is left, it sleeps for half of
/// what is left, and then for the rest.
///
/// [`thread::sleep`] sleeps at least as long as it is asked, but the system may let it sleep
/// longer to wake several threads together: on macOS, here, a sleep of 1 ms took 1.5 ms, one of
/// 5 ms took 7.5 ms, and one of 50 ms up to 60 ms; a value of 9 and a unit of 1 ms then woke
/// after a value of 12. Asked for half of what is left, a sleep that comes out as late as that
/// still ends before the deadline, and the next one is shorter, and so less late: it ends at
/// most a few dozen microseconds after its deadline, and a thread wakes about `log₂` of its delay
/// in units of [`FINE`] times. From 19% to 100% sorted for 10 elements at 1 ms per unit, from
/// none to all for 100 elements. The loop also covers a sleep clock that runs a little behind
/// [`Instant`]'s: no thread wakes before its deadline.
fn sleep_until(deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        let left = deadline - now;
        thread::sleep(if left > FINE { left / 2 } else { left });
    }
}

[A](../../../README.md) › [Sorting](../../../README.md#sorting)

# Sleep

Sleep sort, posted anonymously to 4chan's programming board in 2011, is a joke: a shell script that started `sleep $n; echo $n` in the background for every argument `n`, so that the small numbers wake, and are printed, first. It sorts in "linear time", lets the operating system's timer queue do all the work, and is wrong as soon as the scheduler is late. `sleep` is that algorithm with real threads and real sleeping: one thread per element sleeps `(value − min) × unit` past a start all threads share, and the values are written back in the order their threads wake. Keys are the same as counting sort's (the sealed `IntegerKey` trait), so negative numbers, `char`s and `bool`s work too, and the smallest values wake right at the start.

**It is nondeterministic and intended as a curiosity.** A thread scheduled late lets a larger value overtake a smaller one, so the slice may come out unsorted; `sleep` returns whether it did, `true` exactly when the slice ends up in order. Whatever happens, the slice ends up a permutation of what it was: every value is written back exactly once. It panics, leaving the slice as it was, if the clock cannot represent when the largest value would wake (about 292 billion years on Unix-like platforms: the whole range of `i128` at a nanosecond per unit, or of `u64` at a second; a zero unit never panics) or if the system cannot start a thread for every element (16,384 per process on macOS), after sending home the threads already started. A range the clock can represent is slept through, however long: the whole range of `i64` at a nanosecond per unit takes 584 years. This implementation uses four optimizations, for faithfulness and robustness rather than speed (the success rates are for random keys in `0..n` on a 16-core Mac loaded by other benchmarks):

- One shared start: every thread sleeps until `start + (value − min) × unit`, not for a duration from whenever it happened to be started, so the order and the cost of starting the threads (15 µs each for 1000 of them) do not skew when they wake. Sleeping from each thread's own start, 100 elements at 100 µs per unit and 1000 elements at 1 ms never came out sorted; from the shared start, almost always.
- The clock starts once every thread is running, and a little later: all threads are started first and wait at a gate, a `RwLock` the caller holds for writing, and the start is set 4 µs per thread after the gate opens, as waking the waiting threads takes the system 1.5–3 µs each. With the start at the gate, the smallest values, released last, woke after the next ones: 1000 elements at 100 µs never came out sorted (100 elements: 64%; 1000 at 1 ms: 30%).
- Sleeping in halves: the system may let a sleep run longer than asked, to wake several threads together (here 1 ms took 1.5 ms, 5 ms took 7.5 ms, 50 ms up to 60 ms). Each thread sleeps for half the time left until its deadline, again and again, and for all of it once less than 50 µs is left, so a late sleep still ends before the deadline and the last, short one is late by a few dozen microseconds at most. At 1 ms per unit: from 39% to 100% sorted for 10 elements, from none to all for 100, and from none to 60–95% for 1000. Spinning through the last microseconds instead was no better.
- Small stacks: each thread gets a 16 KiB stack instead of the default 2 MiB, which halves the time to start 5000 threads or more (0.27 s instead of 0.55 s for 5000).

The values are written back lock-free, each waking thread taking the next place from an atomic counter, and `(value − min) × unit` is computed in 128 bits, saturating at `Duration::MAX`.

## Examples

Rust usage with integers (a unit of a few milliseconds almost always sorts a few values):

```rust
use a::sort::sleep;
use std::time::Duration;

fn main() {
  let mut v = vec![5, 1, 4, 2, 8, 0, 3, 9, 7, 6];
  let sorted = sleep(&mut v, Duration::from_millis(2));
  if sorted {
    assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  } else {
    println!("the scheduler was late: {v:?}");
  }
}
```

Negative numbers sleep from the smallest one, and the return value always tells the truth:

```rust
use a::sort::sleep;
use std::time::Duration;

fn main() {
  let mut v = vec![3, -2, 0, -5, 1];
  let sorted = sleep(&mut v, Duration::from_millis(1));
  assert_eq!(sorted, v.is_sorted());
}
```

## Efficiency

- Time complexity: `(max − min) × unit` of sleeping, plus starting n threads (about 15 µs each up to 1000 threads, more beyond: 0.27 s for 5000, 1 s for 10,000) and 4 µs per thread before the start, whatever the order of the input: "linear" in the range of the values, not in their number
- Comparisons: none between elements (one pass for the smallest and largest value, one to report whether the result is sorted)
- Threads: exactly n, one per element, all alive at once
- Space complexity: O(n): a 16 KiB stack per thread; the values move through no buffer
- Correctness probability (share of sorts that came out sorted, random keys in `0..n`, over three runs of 20–400 sorts on a loaded 16-core Mac):

  | `unit` | 10 elements | 100 elements | 1000 elements |
  |--------|-------------|--------------|---------------|
  | 10 µs  | 12–18%      | 0%           | 0%            |
  | 100 µs | 94–100%     | 94–100%      | 89–100%       |
  | 1 ms   | 99–100%     | 89–100%      | 60–90%        |

  A unit of 10 µs is below what the timers resolve (a thread wakes 10–15 µs after its deadline at best); a longer sort is longer exposed to the other work of the machine, so 1000 elements at 1 ms (a second) came out sorted less often than at 100 µs (a tenth of that)
- Stability: equal values wake together, in any order, which cannot be seen
- Against the standard library: hopeless, of course. `slice::sort_unstable` sorts 1000 random `u32`s in 3.5 µs and 100 in 0.2 µs; `sleep` takes about 115 ms and 11 ms at 100 µs per unit, 30,000–55,000 times as long, and is not always right

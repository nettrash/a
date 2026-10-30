//! A global allocator that counts the allocations made on each thread, shared by the tests
//! that check when an operation allocates (a test binary can only have one global allocator).
//!
//! `realloc` is left to its default, which allocates the new block before freeing the old
//! one: growing a vector counts as an allocation, and it always moves the vector.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: every call is forwarded to the system allocator unchanged.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `try_with` fails only while the thread is being torn down.
        let _ = ALLOCATIONS.try_with(|a| a.set(a.get() + 1));
        // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc` for `layout`.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above, that is from `System`, with this `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` and returns its result with how many allocations it made on this thread.
pub fn count<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let before = ALLOCATIONS.with(Cell::get);
    let result = f();
    (result, ALLOCATIONS.with(Cell::get) - before)
}

//! Working memory: the limbs a kernel reads and writes besides its operands and its result, set
//! aside by its caller after a plan of their number, the `*_itch` function beside each kernel.
//!
//! A plan depends on the lengths of the operands, and not monotonically: a shorter operand can take
//! another method, which needs more. A plan too small for what its kernel does writes past the
//! buffer, and that shows, if at all, as a rare failure of some other test far from it. So in builds
//! with debug assertions a [`Scratch`] carries its length: each kernel checks on entry that it got
//! at least what its own plan says ([`need!`]), and each part it cuts off, or hands on, is checked to
//! fit what is left. A plan that is too small, or a kernel that uses more than its plan, panics
//! there. Without debug assertions a `Scratch` is the pointer alone, and the checks are compiled out.

use core::ptr::NonNull;

/// The limbs of working memory that a kernel may read and write.
#[derive(Clone, Copy)]
#[cfg_attr(not(debug_assertions), repr(transparent))]
pub(super) struct Scratch {
    /// The first limb.
    ptr: *mut u64,
    /// The number of limbs.
    #[cfg(debug_assertions)]
    len: usize,
}

impl Scratch {
    /// The `len` limbs at `ptr`.
    #[inline(always)]
    pub(super) fn new(ptr: *mut u64, len: usize) -> Scratch {
        #[cfg(not(debug_assertions))]
        let _ = len;
        Scratch {
            ptr,
            #[cfg(debug_assertions)]
            len,
        }
    }

    /// No limbs, for a kernel whose plan needs none.
    #[inline(always)]
    pub(super) fn none() -> Scratch {
        Scratch::new(NonNull::dangling().as_ptr(), 0)
    }

    /// The first limb, unchecked.
    #[inline(always)]
    pub(super) fn ptr(self) -> *mut u64 {
        self.ptr
    }

    /// The first `n` limbs, as a buffer of their own: checks that there are as many.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub(super) fn take(self, n: usize) -> *mut u64 {
        self.check(n);
        self.ptr
    }

    /// The first `n` limbs, as working memory of their own: checks that there are as many.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub(super) fn first(self, n: usize) -> Scratch {
        self.check(n);
        Scratch::new(self.ptr, n)
    }

    /// The limbs after the first `n`: checks that there are as many.
    ///
    /// # Safety
    ///
    /// The first `n` limbs must lie in the buffer.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub(super) unsafe fn add(self, n: usize) -> Scratch {
        self.check(n);
        Scratch {
            // SAFETY: as the caller guarantees.
            ptr: unsafe { self.ptr.add(n) },
            #[cfg(debug_assertions)]
            len: self.len - n,
        }
    }

    /// The first `n` limbs, as a buffer of their own, and the working memory after them: checks
    /// that there are as many.
    ///
    /// # Safety
    ///
    /// As for [`Scratch::add`].
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub(super) unsafe fn split(self, n: usize) -> (*mut u64, Scratch) {
        // SAFETY: as the caller guarantees.
        (self.take(n), unsafe { self.add(n) })
    }

    /// Panics, in builds with debug assertions, unless it holds `n` limbs. [`need!`] calls it with a
    /// plan, which it evaluates only in those builds.
    #[inline(always)]
    #[cfg_attr(debug_assertions, track_caller)]
    pub(super) fn check(self, n: usize) {
        #[cfg(debug_assertions)]
        assert!(
            n <= self.len,
            "{n} limbs of working memory needed, {} planned",
            self.len
        );
        #[cfg(not(debug_assertions))]
        let _ = n;
    }
}

/// The `$n` limbs at `$p` as working memory, [`Scratch::new`]: without debug assertions, `$n` is
/// not even named, so that a closure that builds its working memory does not capture the length,
/// which only those builds keep.
macro_rules! scratch {
    ($p:expr, $n:expr) => {{
        #[cfg(debug_assertions)]
        let n = $n;
        #[cfg(not(debug_assertions))]
        let n = 0;
        $crate::math::numbers::scratch::Scratch::new($p, n)
    }};
}
pub(super) use scratch;

/// The first `$n` limbs of the working memory `$s`, [`Scratch::first`]: without debug assertions,
/// `$n` is not evaluated, so that a plan that calls functions the optimizer cannot remove does not
/// cost its call, or the registers its value would hold.
macro_rules! first {
    ($s:expr, $n:expr) => {{
        #[cfg(debug_assertions)]
        let n = $n;
        #[cfg(not(debug_assertions))]
        let n = 0;
        $s.first(n)
    }};
}
pub(super) use first;

/// Checks, in builds with debug assertions, that the working memory `$s` holds the `$n` limbs that
/// a kernel's plan says it needs; without them, `$n` is not evaluated.
macro_rules! need {
    ($s:expr, $n:expr) => {
        if cfg!(debug_assertions) {
            $s.check($n);
        }
    };
}
pub(super) use need;

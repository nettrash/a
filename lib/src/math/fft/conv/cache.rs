//! The plans of the transforms of the convolutions, kept for the next calls of the same lengths.
//!
//! A bounded cache shared by all threads: at most [`ENTRIES`] plans, and at most [`BYTES`]
//! bytes of them by the estimate [`bytes`]: twice `len·size_of::<Complex<T>>()` for a length
//! with no prime factor above 13 (measured: 1.5 times for a real plan, 2 for a complex one), 12
//! times for the others (Bluestein's and Rader's plans, measured at up to 10). The key is the
//! plan's type (which carries the float type and real or complex) and its length. A plan larger
//! than [`BYTES`] alone is made, used and dropped, and evicts nothing. The least recently used
//! plans go first; those still in use elsewhere live on in their `Arc` until dropped there.
//!
//! The lock is held only to look a plan up, to insert one and to unlink those evicted: plans are
//! made, used and dropped outside it, and nothing that can panic runs under it, so a panic in a
//! transform cannot poison it (and a poisoned lock is taken over all the same, as its state is
//! valid between any two of its statements).

use std::any::{Any, TypeId};
use std::sync::{Arc, Mutex, MutexGuard};

/// The most plans kept.
pub(super) const ENTRIES: usize = 16;

/// The most bytes of plans kept, by the estimate [`bytes`].
pub(super) const BYTES: usize = 16 << 20;

/// A kept plan: its type (which carries the float type and real or complex), its length, its
/// estimated bytes and when it was last used.
struct Entry {
    ty: TypeId,
    n: usize,
    bytes: usize,
    used: u64,
    plan: Arc<dyn Any + Send + Sync>,
}

struct Cache {
    entries: Vec<Entry>,
    /// The count of lookups, the clock of `used`.
    clock: u64,
}

static CACHE: Mutex<Cache> = Mutex::new(Cache {
    entries: Vec::new(),
    clock: 0,
});

fn lock() -> MutexGuard<'static, Cache> {
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

/// The estimated bytes of a plan of the length `n` whose complex values take `value` bytes.
pub(super) fn bytes(n: usize, value: usize) -> usize {
    let mut m = n.max(1);
    for p in [2, 3, 5, 7, 11, 13] {
        while m.is_multiple_of(p) {
            m /= p;
        }
    }
    let factor = if m == 1 { 2 } else { 12 };
    n.saturating_mul(value).saturating_mul(factor)
}

/// The plan of the type `P` for the length `n`, kept or made by `make` (and then kept if the
/// bounds allow it), its estimated size `size` bytes.
pub(super) fn plan<P: Any + Send + Sync>(
    n: usize,
    size: usize,
    make: impl FnOnce() -> P,
) -> Arc<P> {
    let ty = TypeId::of::<P>();
    let found = {
        let mut c = lock();
        c.clock += 1;
        let clock = c.clock;
        c.entries
            .iter_mut()
            .find(|e| e.ty == ty && e.n == n)
            .map(|e| {
                e.used = clock;
                e.plan.clone()
            })
    };
    if let Some(p) = found
        && let Ok(p) = p.downcast::<P>()
    {
        return p;
    }
    let p = Arc::new(make());
    if size <= BYTES {
        let mut evicted = Vec::new();
        {
            let mut c = lock();
            // Another thread may have made the same plan meanwhile: the one kept stays.
            if !c.entries.iter().any(|e| e.ty == ty && e.n == n) {
                let mut total: usize = c.entries.iter().map(|e| e.bytes).sum();
                while !c.entries.is_empty() && (c.entries.len() >= ENTRIES || total + size > BYTES)
                {
                    let (i, _) = c
                        .entries
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, e)| e.used)
                        .expect("not empty");
                    let e = c.entries.swap_remove(i);
                    total -= e.bytes;
                    evicted.push(e.plan);
                }
                c.clock += 1;
                let used = c.clock;
                let plan: Arc<dyn Any + Send + Sync> = p.clone();
                c.entries.push(Entry {
                    ty,
                    n,
                    bytes: size,
                    used,
                    plan,
                });
            }
        }
        // The plans evicted are dropped here, outside the lock.
        drop(evicted);
    }
    p
}

//! The limbs of a mantissa: one or two kept in the number itself, more in a vector, so that a
//! number of at most 128 bits is made without an allocation.

use core::ops::{Deref, DerefMut};

/// The limbs of a finite number's mantissa, the least significant first (see
/// [`Kind::Finite`](super::Kind::Finite)): one or two in the value, more on the heap. A mantissa of
/// one or two limbs made by [`Mant::from`] or [`Mant::from_slice`] is always kept in the value;
/// equality is that of the limbs, whatever the storage.
#[derive(Clone)]
pub(crate) enum Mant {
    One(u64),
    Two([u64; 2]),
    Heap(Vec<u64>),
}

impl Mant {
    /// The limbs `m`, copied.
    #[inline]
    pub(crate) fn from_slice(m: &[u64]) -> Mant {
        match *m {
            [a] => Mant::One(a),
            [a, b] => Mant::Two([a, b]),
            _ => Mant::Heap(m.to_vec()),
        }
    }

    /// `n` zero limbs.
    #[inline]
    pub(crate) fn zeroed(n: usize) -> Mant {
        match n {
            1 => Mant::One(0),
            2 => Mant::Two([0; 2]),
            _ => Mant::Heap(vec![0; n]),
        }
    }

    /// The limbs.
    #[inline(always)]
    pub(crate) fn as_slice(&self) -> &[u64] {
        match self {
            Mant::One(a) => core::slice::from_ref(a),
            Mant::Two(a) => a,
            Mant::Heap(v) => v,
        }
    }

    /// The limbs, to change.
    #[inline(always)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u64] {
        match self {
            Mant::One(a) => core::slice::from_mut(a),
            Mant::Two(a) => a,
            Mant::Heap(v) => v,
        }
    }

    /// One or two limbs as a word, aligned at its top (one limb in the high half).
    #[inline(always)]
    pub(crate) fn top(&self) -> u128 {
        match self {
            Mant::One(a) => u128::from(*a) << 64,
            Mant::Two([lo, hi]) => u128::from(*hi) << 64 | u128::from(*lo),
            Mant::Heap(v) => super::small::wide(v),
        }
    }

    /// The vector of the limbs, to reuse for a result: the heap's, or an empty vector (which
    /// allocates nothing) for limbs kept in the value.
    #[inline]
    pub(crate) fn into_spare(self) -> Vec<u64> {
        match self {
            Mant::Heap(v) => v,
            _ => Vec::new(),
        }
    }
}

impl From<Vec<u64>> for Mant {
    /// The limbs of `v`: one or two moved into the value (and `v` freed), more kept in `v`.
    #[inline]
    fn from(v: Vec<u64>) -> Mant {
        match *v {
            [a] => Mant::One(a),
            [a, b] => Mant::Two([a, b]),
            _ => Mant::Heap(v),
        }
    }
}

impl Deref for Mant {
    type Target = [u64];
    #[inline(always)]
    fn deref(&self) -> &[u64] {
        self.as_slice()
    }
}

impl DerefMut for Mant {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [u64] {
        self.as_mut_slice()
    }
}

impl PartialEq for Mant {
    #[inline]
    fn eq(&self, other: &Mant) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Mant {}

impl core::fmt::Debug for Mant {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        core::fmt::Debug::fmt(self.as_slice(), f)
    }
}

// On 64-bit targets a mantissa takes no more room than the vector it replaces, nor a number more
// (checked when the crate is compiled).
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<Mant>() == core::mem::size_of::<Vec<u64>>());
#[cfg(target_pointer_width = "64")]
const _: () = assert!(core::mem::size_of::<super::BigFloat>() <= 48);

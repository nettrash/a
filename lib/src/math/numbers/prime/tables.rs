//! Tables of small primes, computed at compile time: the odd primes below [`LIMIT`], each with what
//! tells divisibility by it with a multiplication, and the products of consecutive ones that fit in
//! a limb, by which a long number is reduced in one pass each.

/// The odd primes in the tables are those below this.
pub(super) const LIMIT: usize = 1 << 13;

/// The number of odd primes below [`LIMIT`].
pub(super) const COUNT: usize = count();

/// The odd primes below `LIMIT`, by a sieve of Eratosthenes over the odd numbers.
const fn sieve() -> [bool; LIMIT / 2] {
    // `composite[i]` for the odd number `2i + 1`.
    let mut composite = [false; LIMIT / 2];
    composite[0] = true;
    let mut i = 1;
    while (2 * i + 1) * (2 * i + 1) < LIMIT {
        if !composite[i] {
            let p = 2 * i + 1;
            let mut j = p * p / 2;
            while j < LIMIT / 2 {
                composite[j] = true;
                j += p;
            }
        }
        i += 1;
    }
    composite
}

/// The number of odd primes below `LIMIT`.
const fn count() -> usize {
    let composite = sieve();
    let mut n = 0;
    let mut i = 0;
    while i < LIMIT / 2 {
        if !composite[i] {
            n += 1;
        }
        i += 1;
    }
    n
}

/// An odd prime and what tells whether it divides a limb `x`: `x inv mod B <= max`, for its inverse
/// `inv` modulo `B = 2^64` and `max = floor((B - 1) / p)`, as the multiples of `p` are exactly the
/// limbs that the multiplication by `inv` maps to `0..=max`.
#[derive(Clone, Copy)]
pub(super) struct Prime {
    /// The prime.
    pub(super) p: u64,
    /// Its inverse modulo `B`.
    pub(super) inv: u64,
    /// `floor((B - 1) / p)`.
    pub(super) max: u64,
}

impl Prime {
    /// Whether the prime divides `x`.
    #[inline(always)]
    pub(super) fn divides(&self, x: u64) -> bool {
        x.wrapping_mul(self.inv) <= self.max
    }
}

/// The inverse of the odd `d` modulo `B`, by Newton's steps, each doubling the bits that are right.
const fn binvert(d: u64) -> u64 {
    let mut v = d.wrapping_mul(3) ^ 2;
    let mut i = 0;
    while i < 4 {
        v = v.wrapping_mul(2u64.wrapping_sub(d.wrapping_mul(v)));
        i += 1;
    }
    v
}

/// The odd primes below `LIMIT`, in order.
pub(super) static PRIMES: [Prime; COUNT] = primes();

/// The odd primes below `LIMIT`, in order, computed.
const fn primes() -> [Prime; COUNT] {
    let composite = sieve();
    let mut primes = [Prime {
        p: 0,
        inv: 0,
        max: 0,
    }; COUNT];
    let (mut i, mut n) = (0, 0);
    while i < LIMIT / 2 {
        if !composite[i] {
            let p = (2 * i + 1) as u64;
            primes[n] = Prime {
                p,
                inv: binvert(p),
                max: u64::MAX / p,
            };
            n += 1;
        }
        i += 1;
    }
    primes
}

/// A run of consecutive primes of [`PRIMES`] whose product fits in a limb: a number is reduced
/// modulo the product once, and the remainder tells divisibility by each of them.
#[derive(Clone, Copy)]
pub(super) struct Group {
    /// The index of the first prime of the run in `PRIMES`.
    pub(super) first: u16,
    /// The index past its last prime.
    pub(super) end: u16,
    /// The product of its primes, shifted left so that its top bit is set: a remainder by it is
    /// one by the product too, up to a multiple of the product.
    pub(super) d: u64,
    /// The reciprocal of `d` for the division of two limbs by it, `floor((B^2 - 1) / d) - B`.
    pub(super) v: u64,
}

/// The number of groups.
pub(super) const GROUPS: usize = {
    let primes = primes();
    let (mut g, mut i) = (0, 0);
    while i < COUNT {
        let mut product = 1u64;
        while i < COUNT {
            match product.checked_mul(primes[i].p) {
                Some(x) => product = x,
                None => break,
            }
            i += 1;
        }
        g += 1;
    }
    g
};

/// The runs of consecutive primes whose products fit in a limb, the longest that do, in order.
pub(super) static GROUP: [Group; GROUPS] = groups();

/// The runs of consecutive primes whose products fit in a limb, computed.
const fn groups() -> [Group; GROUPS] {
    let primes = primes();
    let mut groups = [Group {
        first: 0,
        end: 0,
        d: 0,
        v: 0,
    }; GROUPS];
    let (mut g, mut i) = (0, 0);
    while i < COUNT {
        let first = i;
        let mut product = 1u64;
        while i < COUNT {
            match product.checked_mul(primes[i].p) {
                Some(x) => product = x,
                None => break,
            }
            i += 1;
        }
        let d = product << product.leading_zeros();
        groups[g] = Group {
            first: first as u16,
            end: i as u16,
            d,
            v: (u128::MAX / d as u128) as u64,
        };
        g += 1;
    }
    groups
}

/// The index in [`GROUP`] past the last group whose primes are all below `bound`.
pub(super) fn groups_below(bound: u64) -> usize {
    let mut g = 0;
    while g < GROUPS && PRIMES[GROUP[g].end as usize - 1].p < bound {
        g += 1;
    }
    g
}

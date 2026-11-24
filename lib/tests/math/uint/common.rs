//! What the tests of the fixed-width integers share: a seeded generator, operands biased to the
//! edges, `Number` as the reference, and a quiet way to see whether something panics.

use a::math::numbers::Number;
use a::math::uint::{Int, Uint};
use std::cell::Cell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Once;

/// Scales a randomised test down when it runs under Miri.
pub(super) fn size(normal: usize, miri: usize) -> usize {
    if cfg!(miri) { miri } else { normal }
}

/// SplitMix64: a small seeded generator, so the tests need no dependency.
pub(super) struct Rng(pub(super) u64);

impl Rng {
    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number below `n`.
    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A limb biased to the edges: zero, all ones, one bit, one bit less, small, or random.
    pub(super) fn limb(&mut self) -> u64 {
        match self.below(10) {
            0 => 0,
            1 => u64::MAX,
            2 => 1 << self.below(64),
            3 => (1u64 << self.below(64)).wrapping_sub(1),
            4 => u64::MAX - self.below(3),
            5 => self.below(4),
            6 => !(1u64 << self.below(64)),
            _ => self.next(),
        }
    }

    /// An integer of `N` limbs biased to the edges: a random number of significant limbs, each
    /// limb biased, sometimes a power of two or one next to it, sometimes all ones below a point.
    pub(super) fn uint<const N: usize>(&mut self) -> Uint<N> {
        if N == 0 {
            return Uint::ZERO;
        }
        let bits = Uint::<N>::BITS as u64;
        match self.below(12) {
            0 => {
                // 2^k, 2^k − 1 or 2^k + 1.
                let k = self.below(bits) as u32;
                let p = Uint::<N>::ONE << k;
                match self.below(3) {
                    0 => p,
                    1 => p.wrapping_sub(Uint::ONE),
                    _ => p.wrapping_add(Uint::ONE),
                }
            }
            1 => Uint::MAX.unbounded_shr(self.below(bits) as u32),
            2 => {
                // One nonzero limb, often the top one.
                let mut l = [0u64; N];
                let i = if self.below(2) == 0 {
                    N - 1
                } else {
                    self.below(N as u64) as usize
                };
                l[i] = self.limb().max(1);
                Uint::from_limbs(l)
            }
            _ => {
                let len = 1 + self.below(N as u64) as usize;
                let mut l = [0u64; N];
                for x in l.iter_mut().take(len) {
                    *x = self.limb();
                }
                Uint::from_limbs(l)
            }
        }
    }

    /// A signed integer of `N` limbs biased to the edges, of either sign.
    pub(super) fn int<const N: usize>(&mut self) -> Int<N> {
        let u = self.uint::<N>();
        match self.below(8) {
            0 => Int::MIN,
            1 => Int::MAX,
            2 => u.cast_signed(),
            // The two's complement of a value biased to small magnitudes: negative edges.
            3 | 4 => u.cast_signed().wrapping_neg(),
            _ => {
                u.unbounded_shr(1 + self.below(2) as u32).cast_signed()
                    * if self.below(2) == 0 {
                        Int::ONE
                    } else {
                        Int::MINUS_ONE
                    }
            }
        }
    }
}

/// The values at the edges of `N` limbs: 0 to 3, the top values, powers of two with their
/// neighbours at and around the limb boundaries, all-ones limbs, one nonzero high limb.
pub(super) fn edges<const N: usize>() -> Vec<Uint<N>> {
    let b = Uint::<N>::BITS;
    let mut v = vec![
        Uint::ZERO,
        Uint::ONE,
        Uint::from_u64(2),
        Uint::from_u64(3),
        Uint::from_u64(10),
        Uint::MAX,
        Uint::MAX - Uint::ONE,
        Uint::MAX >> 1u32,
        (Uint::MAX >> 1u32) + Uint::ONE,
    ];
    let mut ks = vec![1, 31, 32, 63, b / 2, b - 2, b - 1];
    if b > 64 {
        ks.extend([64, 65, b - 64, b - 65]);
    }
    for k in ks {
        let p = Uint::<N>::ONE << k;
        v.push(p);
        v.push(p - Uint::ONE);
        v.push(p + Uint::ONE);
    }
    let mut high = [0; N];
    high[N - 1] = 1;
    v.push(Uint::from_limbs(high));
    high[N - 1] = 0x1234_5678_9abc_def1;
    v.push(Uint::from_limbs(high));
    v.push(Uint::from_limbs([u64::MAX - 1; N]));
    v.push(Uint::from_limbs([0x8000_0000_0000_0000; N]));
    v.sort();
    v.dedup();
    v
}

/// The signed values at the edges: the unsigned ones as two's complement, and their negations.
pub(super) fn signed_edges<const N: usize>() -> Vec<Int<N>> {
    let mut v: Vec<Int<N>> = edges::<N>()
        .into_iter()
        .flat_map(|u| [u.cast_signed(), u.cast_signed().wrapping_neg()])
        .collect();
    v.extend([Int::MIN, Int::MAX, Int::MINUS_ONE, Int::MIN + Int::ONE]);
    v.sort();
    v.dedup();
    v
}

/// `2^k` as a number.
pub(super) fn pow2(k: u32) -> Number {
    Number::from(1u64) << k as usize
}

/// `t` as `N` limbs if it is in `[0, 2^(64N))`.
pub(super) fn exact<const N: usize>(t: &Number) -> Option<Uint<N>> {
    Uint::try_from(t).ok()
}

/// `t` modulo `2^(64N)`.
pub(super) fn wrap<const N: usize>(t: &Number) -> Uint<N> {
    Uint::try_from(t.rem_euclid(&pow2(Uint::<N>::BITS))).expect("a remainder fits")
}

/// `t` as a signed integer of `N` limbs if it is in range.
pub(super) fn exact_signed<const N: usize>(t: &Number) -> Option<Int<N>> {
    Int::try_from(t).ok()
}

/// `t` wrapped to `N` limbs in two's complement.
pub(super) fn wrap_signed<const N: usize>(t: &Number) -> Int<N> {
    wrap::<N>(t).cast_signed()
}

/// `t` held to `[0, MAX]`.
pub(super) fn clamp<const N: usize>(t: &Number) -> Uint<N> {
    if t.is_negative() {
        Uint::ZERO
    } else {
        exact(t).unwrap_or(Uint::MAX)
    }
}

/// `t` held to `[MIN, MAX]`.
pub(super) fn clamp_signed<const N: usize>(t: &Number) -> Int<N> {
    exact_signed(t).unwrap_or(if t.is_negative() { Int::MIN } else { Int::MAX })
}

/// Checks the four forms of an operation whose true result is `t`: `overflowing` the value modulo
/// the width and whether `t` is out of range, `checked` and `wrapping` as they derive from it, and
/// `saturating` (when there is one) held to the range.
pub(super) fn forms<const N: usize>(
    what: &str,
    t: &Number,
    overflowing: (Uint<N>, bool),
    checked: Option<Uint<N>>,
    wrapping: Uint<N>,
    saturating: Option<Uint<N>>,
) {
    let e = exact::<N>(t);
    assert_eq!(
        overflowing,
        (wrap::<N>(t), e.is_none()),
        "{what}: overflowing, {t}"
    );
    assert_eq!(checked, e, "{what}: checked, {t}");
    assert_eq!(wrapping, wrap::<N>(t), "{what}: wrapping, {t}");
    if let Some(s) = saturating {
        assert_eq!(s, clamp::<N>(t), "{what}: saturating, {t}");
    }
}

/// [`forms`] for a signed operation.
pub(super) fn forms_signed<const N: usize>(
    what: &str,
    t: &Number,
    overflowing: (Int<N>, bool),
    checked: Option<Int<N>>,
    wrapping: Int<N>,
    saturating: Option<Int<N>>,
) {
    let e = exact_signed::<N>(t);
    assert_eq!(
        overflowing,
        (wrap_signed::<N>(t), e.is_none()),
        "{what}: overflowing, {t}"
    );
    assert_eq!(checked, e, "{what}: checked, {t}");
    assert_eq!(wrapping, wrap_signed::<N>(t), "{what}: wrapping, {t}");
    if let Some(s) = saturating {
        assert_eq!(s, clamp_signed::<N>(t), "{what}: saturating, {t}");
    }
}

thread_local! {
    /// Whether this thread expects a panic, which the hook then keeps quiet about.
    static QUIET: Cell<bool> = const { Cell::new(false) };
}

/// The value of `f`, or `Err` with the panic's message if it panicked; the message of an expected
/// panic is not printed, while the panics of other tests on other threads still are.
pub(super) fn outcome<T>(f: impl FnOnce() -> T) -> Result<T, String> {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            if !QUIET.with(Cell::get) {
                default(info);
            }
        }));
    });
    QUIET.with(|q| q.set(true));
    let r = panic::catch_unwind(AssertUnwindSafe(f));
    QUIET.with(|q| q.set(false));
    r.map_err(|e| {
        e.downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_default()
    })
}

/// Whether `f` panics.
pub(super) fn panics<T>(f: impl FnOnce() -> T) -> bool {
    outcome(f).is_err()
}

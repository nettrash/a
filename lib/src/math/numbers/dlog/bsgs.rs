//! Shanks's baby-step giant-step method in a ring of residues, with its table of baby steps kept
//! for several logarithms to the same base.

use super::super::modular::ring::Ring;

/// The most baby steps a table takes: `2^22` slots of 8 bytes, 32 MiB. Beyond, the giant steps
/// grow instead.
pub(in super::super) const MOST_BABY_STEPS: u64 = 1 << 21;

/// An open-addressing table of the baby steps: each slot holds a tag of the residue's key and the
/// step plus one, 0 for an empty slot, in one word; a probe goes on to the next slot.
struct Table {
    /// The slots, a power of two of them.
    slots: Vec<u64>,
    /// The bits of the index.
    bits: u32,
}

impl Table {
    /// A table for `n` entries, at most half full.
    fn new(n: u64) -> Table {
        let bits = (2 * n.max(8)).next_power_of_two().trailing_zeros();
        Table {
            slots: vec![0; 1 << bits],
            bits,
        }
    }

    /// The slot index and the tag of a key.
    #[inline(always)]
    fn place(&self, key: u64) -> (usize, u64) {
        let h = key.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        ((h >> (64 - self.bits)) as usize, (key ^ (key >> 32)) << 32)
    }

    /// Puts in step `j` for `key`.
    #[inline]
    fn insert(&mut self, key: u64, j: u32) {
        let (mut i, tag) = self.place(key);
        let mask = self.slots.len() - 1;
        while self.slots[i] != 0 {
            i = (i + 1) & mask;
        }
        self.slots[i] = tag | (u64::from(j) + 1);
    }

    /// The first step for `key` that `check` accepts.
    #[inline]
    fn find(&self, key: u64, mut check: impl FnMut(u32) -> bool) -> Option<u32> {
        let (mut i, tag) = self.place(key);
        let mask = self.slots.len() - 1;
        loop {
            let s = self.slots[i];
            if s == 0 {
                return None;
            }
            if s & !0xFFFF_FFFF == tag {
                let j = (s as u32).wrapping_sub(1);
                if check(j) {
                    return Some(j);
                }
            }
            i = (i + 1) & mask;
        }
    }
}

/// The baby steps `g^j`, `0 <= j < s`, of a base `g`, and its giant step `g^(-s)`.
pub(in super::super) struct Bsgs<'r, R: Ring> {
    /// The ring.
    r: &'r R,
    /// The base.
    g: R::V,
    /// The baby steps.
    table: Table,
    /// Their number.
    s: u64,
    /// `g^(-s)`.
    giant: R::V,
    /// The order of `g`, when the baby steps came round to 1.
    order: Option<u64>,
}

impl<'r, R: Ring> Bsgs<'r, R> {
    /// The table of `s >= 1` baby steps of the unit `g`, whose inverse is `g_inv`; fewer when they
    /// come round to 1, which gives the order of `g`.
    pub(in super::super) fn new(r: &'r R, g: &R::V, g_inv: &R::V, s: u64) -> Bsgs<'r, R> {
        let s = s.clamp(1, MOST_BABY_STEPS);
        let mut table = Table::new(s);
        let mut e = r.one();
        let mut order = None;
        let mut j = 0u64;
        while j < s {
            table.insert(r.key(&e), j as u32);
            e = r.mul(&e, g);
            j += 1;
            if r.is_one(&e) {
                order = Some(j);
                break;
            }
        }
        let giant = r.pow_u64(g_inv, j);
        Bsgs {
            r,
            g: g.clone(),
            table,
            s: j,
            giant,
            order,
        }
    }

    /// The smallest `k < n` with `g^k = h`, or `None`.
    pub(in super::super) fn log(&self, h: &R::V, n: u128) -> Option<u128> {
        let r = self.r;
        let check = |y: &R::V, j: u32| r.pow_u64(&self.g, u64::from(j)) == *y;
        if let Some(order) = self.order {
            // The baby steps are every power of `g`.
            let j = self.table.find(r.key(h), |j| check(h, j))?;
            let _ = order;
            return (u128::from(j) < n).then_some(u128::from(j));
        }
        let s = u128::from(self.s);
        let giants = n.div_ceil(s);
        let mut y = h.clone();
        let mut i = 0u128;
        while i < giants {
            if let Some(j) = self.table.find(r.key(&y), |j| check(&y, j)) {
                let k = i * s + u128::from(j);
                return (k < n).then_some(k);
            }
            y = r.mul(&y, &self.giant);
            i += 1;
        }
        None
    }
}

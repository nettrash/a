//! Fields of bits in vectors of limbs, for Kronecker substitution: a coefficient laid out at a bit
//! position, and a field read back.

use crate::math::numbers::{ByteOrder, Number};

/// ORs the limbs `x` into `out` from bit `pos`, the bits that fall past `out` dropped.
#[inline]
pub(crate) fn put(out: &mut [u64], pos: usize, x: &[u64]) {
    let (w, sh) = (pos / 64, (pos % 64) as u32);
    if sh == 0 {
        for (o, &v) in out[w..].iter_mut().zip(x) {
            *o |= v;
        }
        return;
    }
    let mut carry = 0u64;
    let mut i = w;
    for &v in x {
        if i >= out.len() {
            return;
        }
        out[i] |= (v << sh) | carry;
        carry = v >> (64 - sh);
        i += 1;
    }
    if carry != 0 && i < out.len() {
        out[i] |= carry;
    }
}

/// Sets `out` to the `bits` bits of `x` from bit `pos` (zeros past the end of `x`), `out` of
/// `bits.div_ceil(64)` limbs.
#[inline]
pub(crate) fn get(x: &[u64], pos: usize, bits: usize, out: &mut [u64]) {
    let (w, sh) = (pos / 64, (pos % 64) as u32);
    let limb = |i: usize| x.get(i).copied().unwrap_or(0);
    let n = out.len();
    debug_assert!(n == bits.div_ceil(64));
    if sh == 0 {
        for (j, o) in out.iter_mut().enumerate() {
            *o = limb(w + j);
        }
    } else {
        for (j, o) in out.iter_mut().enumerate() {
            *o = (limb(w + j) >> sh) | (limb(w + j + 1) << (64 - sh));
        }
    }
    let top = bits % 64;
    if top != 0 {
        out[n - 1] &= (1u64 << top) - 1;
    }
}

/// The limbs of the magnitude of `x` into `out`, the least significant first, without zeros at
/// the top.
pub(crate) fn limbs(x: &Number, out: &mut Vec<u64>) {
    out.clear();
    if x.byte_order == ByteOrder::LittleEndian {
        out.extend(x.value.chunks(8).map(|c| {
            let mut b = [0; 8];
            b[..c.len()].copy_from_slice(c);
            u64::from_le_bytes(b)
        }));
    } else {
        out.extend(x.value.rchunks(8).map(|c| {
            let mut b = [0; 8];
            b[8 - c.len()..].copy_from_slice(c);
            u64::from_be_bytes(b)
        }));
    }
    while out.last() == Some(&0) {
        out.pop();
    }
}

/// The bits of the magnitude of `x`.
pub(crate) fn bits(x: &Number) -> usize {
    x.bits() as usize
}

/// `x` negated in place modulo `2^bits`, `x` of `bits.div_ceil(64)` limbs below `2^bits`.
#[inline]
pub(crate) fn negate(x: &mut [u64], bits: usize) {
    let mut borrow = true;
    for v in x.iter_mut() {
        let (s, b) = (!*v).overflowing_add(u64::from(borrow));
        *v = s;
        borrow = b;
    }
    let top = bits % 64;
    if top != 0 {
        let n = x.len();
        x[n - 1] &= (1u64 << top) - 1;
    }
}

/// `x += 1`, the carry out returned.
#[inline]
pub(crate) fn increment(x: &mut [u64]) -> bool {
    for v in x.iter_mut() {
        let (s, c) = v.overflowing_add(1);
        *v = s;
        if !c {
            return false;
        }
    }
    true
}

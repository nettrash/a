//! The rings of residues that the orders, roots and logarithms are computed in: modulo an odd
//! number in Montgomery's form, in registers up to seven limbs ([`Odd`], [`Two`], [`Many`]) and
//! in limbs beyond ([`Long`]); modulo a power of two by wrapping products ([`Low`], [`Low2`]);
//! and modulo any number by divisions ([`Plain`]). Every residue is kept fully reduced, so that
//! equal residues are equal values, and [`Ring::key`] gives a word of it to hash.
//!
//! [`with_odd!`] and [`with_ring!`] choose the ring for a modulus.

use super::super::pow::montgomery::{Mont2, MontN, Montgomery, rem, rem_itch};
use super::super::pow::window::{Registers, Ring as LimbRing};
use super::super::scratch::Scratch;
use super::super::{ByteOrder, Divisor, Number};
use super::word::Odd;
use core::cell::RefCell;

/// The arithmetic of a ring of residues.
pub(in super::super) trait Ring {
    /// A residue, fully reduced.
    type V: Clone + PartialEq;

    /// The residue of 1.
    fn one(&self) -> Self::V;

    /// The product of two residues.
    fn mul(&self, a: &Self::V, b: &Self::V) -> Self::V;

    /// The square of a residue.
    fn sqr(&self, a: &Self::V) -> Self::V {
        self.mul(a, a)
    }

    /// The sum of two residues.
    fn add(&self, a: &Self::V, b: &Self::V) -> Self::V;

    /// The difference of two residues.
    fn sub(&self, a: &Self::V, b: &Self::V) -> Self::V;

    /// The residue of the plain number of the limbs `x`, the least significant first, of any
    /// length.
    fn residue(&self, x: &[u64]) -> Self::V;

    /// The plain number of the residue `a`, below the modulus, as limbs without zeros at the top.
    fn plain(&self, a: &Self::V) -> Vec<u64>;

    /// A word of the residue for hashing: the same for equal residues, and different for nearly
    /// all different ones.
    fn key(&self, a: &Self::V) -> u64;

    /// Whether `a` is the residue of 1.
    fn is_one(&self, a: &Self::V) -> bool {
        *a == self.one()
    }

    /// `a` to the power of the limbs `e`, the least significant first: 1 for `e = 0`. By a sliding
    /// window from the top bit, as the powers modulo numbers take it.
    fn pow(&self, a: &Self::V, e: &[u64]) -> Self::V {
        let top = e.iter().rposition(|&l| l != 0);
        let Some(top) = top else {
            return self.one();
        };
        let bits = 64 * top as u64 + 64 - u64::from(e[top].leading_zeros());
        if bits == 1 {
            return a.clone();
        }
        let k = window(bits);
        // The odd powers `a, a^3, ..., a^(2^k - 1)`.
        let mut table = Vec::with_capacity(1 << (k - 1));
        table.push(a.clone());
        if k > 1 {
            let a2 = self.sqr(a);
            for i in 1..1usize << (k - 1) {
                let next = self.mul(&table[i - 1], &a2);
                table.push(next);
            }
        }
        let bit = |i: u64| (e[(i / 64) as usize] >> (i % 64)) & 1 == 1;
        // Bits `lo..=hi` of `e`, at most `k` of them.
        let field = |hi: u64, lo: u64| {
            let mut w = 0u64;
            for i in (lo..=hi).rev() {
                w = (w << 1) | u64::from(bit(i));
            }
            w
        };
        let mut i = bits as i64 - 1;
        let mut r: Option<Self::V> = None;
        while i >= 0 {
            if !bit(i as u64) {
                if let Some(x) = r.as_mut() {
                    *x = self.sqr(x);
                }
                i -= 1;
                continue;
            }
            // The window from bit `i` down, of at most `k` bits, ending on a one.
            let mut lo = (i - i64::from(k) + 1).max(0);
            while !bit(lo as u64) {
                lo += 1;
            }
            let w = field(i as u64, lo as u64);
            let len = i - lo + 1;
            r = Some(match r {
                None => table[(w >> 1) as usize].clone(),
                Some(mut x) => {
                    for _ in 0..len {
                        x = self.sqr(&x);
                    }
                    self.mul(&x, &table[(w >> 1) as usize])
                }
            });
            i = lo - 1;
        }
        r.unwrap_or_else(|| self.one())
    }

    /// `a` and `b` to the one power of the limbs `e`: [`pow`](Self::pow) for each, or side by side
    /// where the ring can overlap them.
    fn pow_pair(&self, a: &Self::V, b: &Self::V, e: &[u64]) -> (Self::V, Self::V) {
        (self.pow(a, e), self.pow(b, e))
    }

    /// `a` to the power of the word `e`.
    #[inline]
    fn pow_u64(&self, a: &Self::V, e: u64) -> Self::V {
        self.pow(a, &[e])
    }

    /// A word of the residue whose low bits look random over the residues, for the walks of
    /// Pollard's rho, which choose their steps and their distinguished points by them, and for the
    /// table of baby steps: by default the key through splitmix64's finaliser, a bijection of the
    /// words each of whose bits depends on every bit of the key, as the low bits of a plain residue
    /// need not look random (modulo `2^k n` those of a group of odd order are all 1 modulo `2^k`,
    /// and a product alone leaves the low `k` bits of its result fixed). The residues in
    /// Montgomery's form are scrambled already, and take their key; those by divisions mix all
    /// their limbs so.
    #[inline(always)]
    fn walk_key(&self, a: &Self::V) -> u64 {
        let z = self.key(a);
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `a` to the power of the limbs `e` by [`Ring::pow`]'s sliding window, its odd powers in an array
/// on the stack, for residues in registers.
#[inline]
fn pow_window<R: Ring>(r: &R, a: &R::V, e: &[u64]) -> R::V
where
    R::V: Copy,
{
    let top = e.iter().rposition(|&l| l != 0);
    let Some(top) = top else {
        return r.one();
    };
    let bits = 64 * top as u64 + 64 - u64::from(e[top].leading_zeros());
    if bits == 1 {
        return *a;
    }
    // The window capped at 5 bits, for 16 odd powers: the exponents of the rings in registers are
    // rarely longer than 673 bits, past which GMP's table would take 6.
    let k = window(bits).min(5);
    let mut table = [*a; 16];
    if k > 1 {
        let a2 = r.sqr(a);
        for i in 1..1usize << (k - 1) {
            table[i] = r.mul(&table[i - 1], &a2);
        }
    }
    let bit = |i: u64| (e[(i / 64) as usize] >> (i % 64)) & 1 == 1;
    let mut i = bits as i64 - 1;
    let mut x: Option<R::V> = None;
    while i >= 0 {
        if !bit(i as u64) {
            if let Some(v) = x.as_mut() {
                *v = r.sqr(v);
            }
            i -= 1;
            continue;
        }
        let mut lo = (i - i64::from(k) + 1).max(0);
        while !bit(lo as u64) {
            lo += 1;
        }
        let mut w = 0usize;
        for j in (lo..=i).rev() {
            w = (w << 1) | usize::from(bit(j as u64));
        }
        x = Some(match x {
            None => table[w >> 1],
            Some(mut v) => {
                for _ in 0..i - lo + 1 {
                    v = r.sqr(&v);
                }
                r.mul(&v, &table[w >> 1])
            }
        });
        i = lo - 1;
    }
    x.unwrap_or_else(|| r.one())
}

/// `a` and `b` to the one power of the limbs `e`, by [`pow_window`]'s steps for both side by side:
/// the same windows, so one walk over the exponent, and two chains of products that the processor
/// overlaps.
#[inline]
fn pow_window_pair<R: Ring>(r: &R, a: &R::V, b: &R::V, e: &[u64]) -> (R::V, R::V)
where
    R::V: Copy,
{
    let top = e.iter().rposition(|&l| l != 0);
    let Some(top) = top else {
        return (r.one(), r.one());
    };
    let bits = 64 * top as u64 + 64 - u64::from(e[top].leading_zeros());
    if bits == 1 {
        return (*a, *b);
    }
    let k = window(bits).min(5);
    let (mut ta, mut tb) = ([*a; 16], [*b; 16]);
    if k > 1 {
        let (a2, b2) = (r.sqr(a), r.sqr(b));
        for i in 1..1usize << (k - 1) {
            ta[i] = r.mul(&ta[i - 1], &a2);
            tb[i] = r.mul(&tb[i - 1], &b2);
        }
    }
    let bit = |i: u64| (e[(i / 64) as usize] >> (i % 64)) & 1 == 1;
    let mut i = bits as i64 - 1;
    let mut x: Option<(R::V, R::V)> = None;
    while i >= 0 {
        if !bit(i as u64) {
            if let Some((u, v)) = x.as_mut() {
                *u = r.sqr(u);
                *v = r.sqr(v);
            }
            i -= 1;
            continue;
        }
        let mut lo = (i - i64::from(k) + 1).max(0);
        while !bit(lo as u64) {
            lo += 1;
        }
        let mut w = 0usize;
        for j in (lo..=i).rev() {
            w = (w << 1) | usize::from(bit(j as u64));
        }
        x = Some(match x {
            None => (ta[w >> 1], tb[w >> 1]),
            Some((mut u, mut v)) => {
                for _ in 0..i - lo + 1 {
                    u = r.sqr(&u);
                    v = r.sqr(&v);
                }
                (r.mul(&u, &ta[w >> 1]), r.mul(&v, &tb[w >> 1]))
            }
        });
        i = lo - 1;
    }
    x.unwrap_or_else(|| (r.one(), r.one()))
}

/// The window size for an exponent of `bits` bits, GMP's table, as the powers take it.
fn window(bits: u64) -> u32 {
    const LIMITS: [u64; 6] = [7, 25, 81, 241, 673, 1793];
    1 + LIMITS.iter().filter(|&&l| bits > l).count() as u32
}

// Odd moduli in registers

impl Ring for Odd {
    type V = u64;

    #[inline(always)]
    fn one(&self) -> u64 {
        Odd::unit(self)
    }

    #[inline(always)]
    fn mul(&self, a: &u64, b: &u64) -> u64 {
        Odd::times(self, *a, *b)
    }

    #[inline(always)]
    fn sqr(&self, a: &u64) -> u64 {
        Odd::square(self, *a)
    }

    #[inline(always)]
    fn add(&self, a: &u64, b: &u64) -> u64 {
        Odd::plus(self, *a, *b)
    }

    #[inline(always)]
    fn sub(&self, a: &u64, b: &u64) -> u64 {
        Odd::minus(self, *a, *b)
    }

    #[inline]
    fn residue(&self, x: &[u64]) -> u64 {
        let m = self.m();
        // Horner's rule from the top limb, by divisions of two limbs.
        let r = x.iter().rev().fold(0u64, |r, &l| {
            (((u128::from(r) << 64) | u128::from(l)) % u128::from(m)) as u64
        });
        self.to(r)
    }

    #[inline]
    fn plain(&self, a: &u64) -> Vec<u64> {
        let x = self.from(*a);
        if x == 0 { Vec::new() } else { vec![x] }
    }

    #[inline(always)]
    fn key(&self, a: &u64) -> u64 {
        *a
    }

    #[inline(always)]
    fn is_one(&self, a: &u64) -> bool {
        *a == Odd::unit(self)
    }

    #[inline]
    fn pow_pair(&self, a: &u64, b: &u64, e: &[u64]) -> (u64, u64) {
        match e {
            [] => (Odd::unit(self), Odd::unit(self)),
            [e] => {
                let [x, y] = Odd::powers(self, [*a, *b], [*e, *e]);
                (x, y)
            }
            _ => (self.pow(a, e), self.pow(b, e)),
        }
    }

    #[inline]
    fn pow(&self, a: &u64, e: &[u64]) -> u64 {
        match e {
            [] => Odd::unit(self),
            [e] => Odd::power(self, *a, *e),
            _ => {
                // Limb by limb from the top, each a power of 2^64 of what came before.
                let mut r = Odd::unit(self);
                for &l in e.iter().rev() {
                    for _ in 0..64 {
                        r = Odd::square(self, r);
                    }
                    r = Odd::times(self, r, Odd::power(self, *a, l));
                }
                r
            }
        }
    }

    #[inline(always)]
    fn walk_key(&self, a: &u64) -> u64 {
        self.key(a)
    }
}

/// The residue `x B^N mod m` of the limbs `x`, any number of them, for the odd `N`-limb `m` whose
/// top limb is not zero: Montgomery's form of `x`.
fn residue_of<const N: usize>(x: &[u64], m: &[u64; N]) -> [u64; N] {
    let mut r = [0u64; N];
    let nx = x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
    if nx == 0 {
        return r;
    }
    let need = rem_itch(nx + N, N);
    // On the stack for the residues of numbers of a few limbs, the common case.
    const STACK: usize = 32;
    if need <= 2 * STACK {
        let mut buf = [0u128; STACK];
        let s = Scratch::new(buf.as_mut_ptr().cast(), 2 * STACK);
        // SAFETY: as below, the scratch on the stack.
        unsafe { rem(r.as_mut_ptr(), x.as_ptr(), nx, N, m.as_ptr(), N, s) };
        return r;
    }
    let mut buf = vec![0u128; need.div_ceil(2) + 1];
    let s = Scratch::new(buf.as_mut_ptr().cast(), 2 * buf.len());
    // SAFETY: the scratch holds the remainder's plan and starts on 16 bytes; `m` has `N` limbs,
    // its top one not zero, and `x` its `nx` limbs.
    unsafe { rem(r.as_mut_ptr(), x.as_ptr(), nx, N, m.as_ptr(), N, s) };
    r
}

/// Sets `r` to `a + b mod m`, for `a, b < m`, all of `m`'s length.
#[inline(always)]
fn add_mod(a: &[u64], b: &[u64], m: &[u64], r: &mut [u64]) {
    let mut carry = false;
    for j in 0..m.len() {
        let (x, c1) = a[j].overflowing_add(b[j]);
        let (x, c2) = x.overflowing_add(u64::from(carry));
        r[j] = x;
        carry = c1 | c2;
    }
    // Less `m` when it carried out or is not below `m`.
    let below = !carry && r.iter().rev().cmp(m.iter().rev()) == core::cmp::Ordering::Less;
    if !below {
        let mut borrow = false;
        for j in 0..m.len() {
            let (x, b1) = r[j].overflowing_sub(m[j]);
            let (x, b2) = x.overflowing_sub(u64::from(borrow));
            r[j] = x;
            borrow = b1 | b2;
        }
    }
}

/// Sets `r` to `a - b mod m`, for `a, b < m`, all of `m`'s length.
#[inline(always)]
fn sub_mod(a: &[u64], b: &[u64], m: &[u64], r: &mut [u64]) {
    let mut borrow = false;
    for j in 0..m.len() {
        let (x, b1) = a[j].overflowing_sub(b[j]);
        let (x, b2) = x.overflowing_sub(u64::from(borrow));
        r[j] = x;
        borrow = b1 | b2;
    }
    if borrow {
        let mut carry = false;
        for j in 0..m.len() {
            let (x, c1) = r[j].overflowing_add(m[j]);
            let (x, c2) = x.overflowing_add(u64::from(carry));
            r[j] = x;
            carry = c1 | c2;
        }
    }
}

/// The limbs without the zeros at the top.
fn trimmed(x: &[u64]) -> Vec<u64> {
    let n = x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
    x[..n].to_vec()
}

/// Montgomery's arithmetic modulo an odd number of two limbs, in registers.
#[derive(Clone, Copy)]
pub(in super::super) struct Two {
    /// The reduction.
    ring: Mont2,
    /// The residue of 1.
    one: [u64; 2],
}

impl Two {
    /// The arithmetic modulo the odd `m`, whose high limb is not zero.
    pub(in super::super) fn new(m: [u64; 2]) -> Two {
        Two {
            ring: Mont2::new(m),
            one: residue_of(&[1], &m),
        }
    }
}

impl Ring for Two {
    type V = [u64; 2];

    #[inline(always)]
    fn one(&self) -> [u64; 2] {
        self.one
    }

    #[inline(always)]
    fn mul(&self, a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
        self.ring.mul(*a, *b)
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; 2]) -> [u64; 2] {
        self.ring.sqr(*a)
    }

    #[inline(always)]
    fn add(&self, a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
        let mut r = [0; 2];
        add_mod(a, b, &self.ring.m, &mut r);
        r
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; 2], b: &[u64; 2]) -> [u64; 2] {
        let mut r = [0; 2];
        sub_mod(a, b, &self.ring.m, &mut r);
        r
    }

    fn residue(&self, x: &[u64]) -> [u64; 2] {
        residue_of(x, &self.ring.m)
    }

    fn plain(&self, a: &[u64; 2]) -> Vec<u64> {
        trimmed(&self.ring.plain(*a))
    }

    #[inline]
    fn pow(&self, a: &[u64; 2], e: &[u64]) -> [u64; 2] {
        pow_window(self, a, e)
    }

    #[inline]
    fn pow_pair(&self, a: &[u64; 2], b: &[u64; 2], e: &[u64]) -> ([u64; 2], [u64; 2]) {
        pow_window_pair(self, a, b, e)
    }

    #[inline(always)]
    fn key(&self, a: &[u64; 2]) -> u64 {
        a[0] ^ a[1].rotate_left(29)
    }

    #[inline(always)]
    fn walk_key(&self, a: &[u64; 2]) -> u64 {
        self.key(a)
    }
}

/// Montgomery's arithmetic modulo an odd number of `N` limbs, 3 to 7, in registers.
#[derive(Clone, Copy)]
pub(in super::super) struct Many<const N: usize> {
    /// The reduction.
    ring: MontN<N>,
    /// The residue of 1.
    one: [u64; N],
}

impl<const N: usize> Many<N> {
    /// The arithmetic modulo the odd `m`, whose top limb is not zero.
    pub(in super::super) fn new(m: [u64; N]) -> Many<N> {
        Many {
            ring: MontN::new(m),
            one: residue_of(&[1], &m),
        }
    }
}

impl<const N: usize> Ring for Many<N> {
    type V = [u64; N];

    #[inline(always)]
    fn one(&self) -> [u64; N] {
        self.one
    }

    #[inline(always)]
    fn mul(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        self.ring.mul(*a, *b)
    }

    #[inline(always)]
    fn sqr(&self, a: &[u64; N]) -> [u64; N] {
        self.ring.sqr(*a)
    }

    #[inline(always)]
    fn add(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut r = [0; N];
        add_mod(a, b, &self.ring.m, &mut r);
        r
    }

    #[inline(always)]
    fn sub(&self, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
        let mut r = [0; N];
        sub_mod(a, b, &self.ring.m, &mut r);
        r
    }

    fn residue(&self, x: &[u64]) -> [u64; N] {
        residue_of(x, &self.ring.m)
    }

    fn plain(&self, a: &[u64; N]) -> Vec<u64> {
        trimmed(&self.ring.plain(*a))
    }

    #[inline]
    fn pow(&self, a: &[u64; N], e: &[u64]) -> [u64; N] {
        pow_window(self, a, e)
    }

    #[inline]
    fn pow_pair(&self, a: &[u64; N], b: &[u64; N], e: &[u64]) -> ([u64; N], [u64; N]) {
        pow_window_pair(self, a, b, e)
    }

    #[inline(always)]
    fn key(&self, a: &[u64; N]) -> u64 {
        a[0] ^ a[1].rotate_left(29)
    }

    #[inline(always)]
    fn walk_key(&self, a: &[u64; N]) -> u64 {
        self.key(a)
    }
}

// Long odd moduli

/// Montgomery's arithmetic modulo an odd number of eight limbs or more, on limbs, with the crate's
/// kernels: each product reduced by REDC and brought below the modulus.
pub(in super::super) struct Long {
    /// The modulus, its top limb not zero; the arithmetic points into it. A vector, not a box:
    /// moving a box would assert it unique, which the arithmetic's pointer into it is not.
    m: Vec<u64>,
    /// The limbs the arithmetic keeps, `1/m mod R` for the longest; it points into them.
    _kept: Vec<u128>,
    /// The arithmetic.
    ring: Montgomery,
    /// The working memory of the products, in pairs of limbs so that it starts on 16 bytes.
    scratch: RefCell<Vec<u128>>,
    /// The residue of 1.
    one: Vec<u64>,
    /// The modulus as a number, for the powers.
    modulus: Number,
}

impl Long {
    /// The arithmetic modulo the odd limbs `m`, at least 8 of them, the top one not zero.
    pub(in super::super) fn new(m: &[u64]) -> Long {
        let n = m.len();
        debug_assert!(n >= 8 && m[0] & 1 == 1 && m[n - 1] != 0);
        let m: Vec<u64> = m.to_vec();
        let kept_len = Montgomery::new_itch(n);
        let mut kept = vec![0u128; kept_len.div_ceil(2) + 1];
        // SAFETY: the modulus is odd with `n >= 8` limbs, and its vector's buffer stays where it is
        // as long as the arithmetic (the vector is moved, never grown); the kept limbs too, and
        // they hold `new_itch(n)`.
        let ring = unsafe {
            Montgomery::new(
                m.as_ptr(),
                n,
                Scratch::new(kept.as_mut_ptr().cast(), 2 * kept.len()),
            )
        };
        let need = Montgomery::itch(n).max(rem_itch(1 + n, n));
        let scratch = RefCell::new(vec![0u128; need.div_ceil(2) + 1]);
        let modulus = from_limbs(&m, ByteOrder::LittleEndian);
        let mut long = Long {
            m,
            _kept: kept,
            ring,
            scratch,
            one: Vec::new(),
            modulus,
        };
        long.one = long.residue(&[1]);
        long
    }

    /// Runs `f` on the working memory.
    fn with_scratch<R>(&self, need: usize, f: impl FnOnce(Scratch) -> R) -> R {
        let mut s = self.scratch.borrow_mut();
        if 2 * s.len() < need {
            s.resize(need.div_ceil(2) + 1, 0);
        }
        f(Scratch::new(s.as_mut_ptr().cast(), 2 * s.len()))
    }
}

impl Ring for Long {
    type V = Vec<u64>;

    fn one(&self) -> Vec<u64> {
        self.one.clone()
    }

    fn mul(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let n = self.m.len();
        let mut r = vec![0u64; n];
        self.with_scratch(Montgomery::itch(n), |t| {
            // SAFETY: the residues have `n` limbs each and are below the modulus, the result
            // too, and the scratch holds the ring's plan.
            unsafe {
                self.ring.mul(r.as_mut_ptr(), a.as_ptr(), b.as_ptr(), t);
                self.ring.reduce(r.as_mut_ptr());
            }
        });
        r
    }

    fn sqr(&self, a: &Vec<u64>) -> Vec<u64> {
        let n = self.m.len();
        let mut r = vec![0u64; n];
        self.with_scratch(Montgomery::itch(n), |t| {
            // SAFETY: as for the product.
            unsafe {
                self.ring.sqr(r.as_mut_ptr(), a.as_ptr(), t);
                self.ring.reduce(r.as_mut_ptr());
            }
        });
        r
    }

    fn add(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let mut r = vec![0; self.m.len()];
        add_mod(a, b, &self.m, &mut r);
        r
    }

    fn sub(&self, a: &Vec<u64>, b: &Vec<u64>) -> Vec<u64> {
        let mut r = vec![0; self.m.len()];
        sub_mod(a, b, &self.m, &mut r);
        r
    }

    fn residue(&self, x: &[u64]) -> Vec<u64> {
        let n = self.m.len();
        let mut r = vec![0u64; n];
        let nx = x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        if nx == 0 {
            return r;
        }
        self.with_scratch(rem_itch(nx + n, n), |s| {
            // SAFETY: the scratch holds the remainder's plan and starts on 16 bytes.
            unsafe { rem(r.as_mut_ptr(), x.as_ptr(), nx, n, self.m.as_ptr(), n, s) };
        });
        r
    }

    fn plain(&self, a: &Vec<u64>) -> Vec<u64> {
        let n = self.m.len();
        let mut r = vec![0u64; n];
        self.with_scratch(Montgomery::itch(n), |t| {
            // SAFETY: the residue has `n` limbs; the scratch holds the ring's plan.
            unsafe { self.ring.plain(r.as_mut_ptr(), a.as_ptr(), t) };
        });
        trimmed(&r)
    }

    #[inline(always)]
    fn key(&self, a: &Vec<u64>) -> u64 {
        a[0] ^ a[1].rotate_left(29)
    }

    fn is_one(&self, a: &Vec<u64>) -> bool {
        *a == self.one
    }

    /// By [`Number::pow_mod`], whose products need no allocation each, out of Montgomery's form
    /// and back.
    fn pow(&self, a: &Vec<u64>, e: &[u64]) -> Vec<u64> {
        let x = from_limbs(&self.plain(a), ByteOrder::LittleEndian);
        let e = from_limbs(e, ByteOrder::LittleEndian);
        self.residue(&limbs_of(&x.pow_mod(&e, &self.modulus)))
    }

    #[inline(always)]
    fn walk_key(&self, a: &Vec<u64>) -> u64 {
        self.key(a)
    }
}

// Powers of two

/// The arithmetic modulo `2^e`, `1 <= e <= 64`, by wrapping products.
#[derive(Clone, Copy)]
pub(in super::super) struct Low {
    /// `2^e - 1`.
    mask: u64,
}

impl Low {
    /// The arithmetic modulo `2^e`.
    pub(in super::super) fn new(e: u32) -> Low {
        debug_assert!((1..=64).contains(&e));
        Low {
            mask: u64::MAX >> (64 - e),
        }
    }
}

impl Ring for Low {
    type V = u64;

    #[inline(always)]
    fn one(&self) -> u64 {
        1 & self.mask
    }

    #[inline(always)]
    fn mul(&self, a: &u64, b: &u64) -> u64 {
        a.wrapping_mul(*b) & self.mask
    }

    #[inline(always)]
    fn add(&self, a: &u64, b: &u64) -> u64 {
        a.wrapping_add(*b) & self.mask
    }

    #[inline(always)]
    fn sub(&self, a: &u64, b: &u64) -> u64 {
        a.wrapping_sub(*b) & self.mask
    }

    fn residue(&self, x: &[u64]) -> u64 {
        x.first().copied().unwrap_or(0) & self.mask
    }

    fn plain(&self, a: &u64) -> Vec<u64> {
        if *a == 0 { Vec::new() } else { vec![*a] }
    }

    #[inline(always)]
    fn key(&self, a: &u64) -> u64 {
        *a
    }
}

/// The arithmetic modulo `2^e`, `65 <= e <= 128`, by wrapping products.
#[derive(Clone, Copy)]
pub(in super::super) struct Low2 {
    /// `2^e - 1`.
    mask: u128,
}

impl Low2 {
    /// The arithmetic modulo `2^e`.
    pub(in super::super) fn new(e: u32) -> Low2 {
        debug_assert!((65..=128).contains(&e));
        Low2 {
            mask: u128::MAX >> (128 - e),
        }
    }
}

impl Ring for Low2 {
    type V = u128;

    #[inline(always)]
    fn one(&self) -> u128 {
        1
    }

    #[inline(always)]
    fn mul(&self, a: &u128, b: &u128) -> u128 {
        a.wrapping_mul(*b) & self.mask
    }

    #[inline(always)]
    fn add(&self, a: &u128, b: &u128) -> u128 {
        a.wrapping_add(*b) & self.mask
    }

    #[inline(always)]
    fn sub(&self, a: &u128, b: &u128) -> u128 {
        a.wrapping_sub(*b) & self.mask
    }

    fn residue(&self, x: &[u64]) -> u128 {
        let lo = x.first().copied().unwrap_or(0);
        let hi = x.get(1).copied().unwrap_or(0);
        (u128::from(lo) | (u128::from(hi) << 64)) & self.mask
    }

    fn plain(&self, a: &u128) -> Vec<u64> {
        trimmed(&[*a as u64, (*a >> 64) as u64])
    }

    #[inline(always)]
    fn key(&self, a: &u128) -> u64 {
        (*a as u64) ^ ((*a >> 64) as u64).rotate_left(29)
    }
}

// Any modulus

/// The arithmetic modulo any number at least 2, by products and remainders of numbers: for the
/// even moduli that are not powers of two, and the powers of two beyond `2^128`.
pub(in super::super) struct Plain {
    /// The modulus, prepared for the remainders.
    m: Divisor,
    /// The modulus.
    modulus: Number,
}

impl Plain {
    /// The arithmetic modulo the limbs `m`, at least 2.
    pub(in super::super) fn new(m: &[u64]) -> Plain {
        let modulus = from_limbs(m, ByteOrder::LittleEndian);
        Plain {
            m: Divisor::new(&modulus),
            modulus,
        }
    }
}

impl Ring for Plain {
    type V = Number;

    fn one(&self) -> Number {
        Number::new_from_u8(1, ByteOrder::LittleEndian)
    }

    fn mul(&self, a: &Number, b: &Number) -> Number {
        self.m.rem_euclid(&(a * b))
    }

    fn add(&self, a: &Number, b: &Number) -> Number {
        self.m.rem_euclid(&(a + b))
    }

    fn sub(&self, a: &Number, b: &Number) -> Number {
        self.m.rem_euclid(&(a - b))
    }

    fn residue(&self, x: &[u64]) -> Number {
        self.m.rem_euclid(&from_limbs(x, ByteOrder::LittleEndian))
    }

    fn plain(&self, a: &Number) -> Vec<u64> {
        limbs_of(a)
    }

    fn key(&self, a: &Number) -> u64 {
        let mut b = [0u8; 8];
        let n = a.value.len().min(8);
        b[..n].copy_from_slice(&a.value[..n]);
        u64::from_le_bytes(b)
    }

    fn is_one(&self, a: &Number) -> bool {
        a.value == [1] && self.modulus != 1u8
    }

    /// Every limb of the residue mixed, each through splitmix64's finaliser, not the low one
    /// alone: modulo `2^k n` the residues of odd order are 1 modulo `2^k`, so that their low `k`
    /// bits are fixed and from `k = 64` their low limbs are all 1, and the walks of rho and the
    /// table of baby steps keyed by them saw a few elements or one (rho never met a distinguished
    /// point from `k = 33` on, and the table was one chain).
    fn walk_key(&self, a: &Number) -> u64 {
        // The residues are little-endian in the shortest form; the top zeros are skipped anyway.
        let n = a.value.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        let mut h = 0u64;
        for c in a.value[..n].chunks(8) {
            let mut b = [0u8; 8];
            b[..c.len()].copy_from_slice(c);
            let z = h ^ u64::from_le_bytes(b);
            let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h = z ^ (z >> 31);
        }
        h
    }
}

// Limbs of numbers

/// The magnitude of `x` when it fits in 128 bits, read in its byte order without a copy.
#[inline]
pub(in super::super) fn magnitude_128(x: &Number) -> Option<u128> {
    let n = x.significant();
    if n > 16 {
        return None;
    }
    let v = &x.value;
    let eight = |i: usize| {
        let mut b = [0; 8];
        b.copy_from_slice(&v[i..i + 8]);
        b
    };
    // From 8 bytes on, the low eight and the eight that end at the top, shifted down to the bytes
    // above the low ones; below, one byte at a time (a copy of a varying length would be a call).
    let shift = 8 * (16 - n as u32);
    let (lo, hi) = if x.byte_order == ByteOrder::BigEndian {
        let end = v.len();
        if n >= 8 {
            let lo = u64::from_be_bytes(eight(end - 8));
            let hi = if n > 8 {
                u64::from_be_bytes(eight(end - n)) >> shift
            } else {
                0
            };
            (lo, hi)
        } else {
            let lo = v[end - n..]
                .iter()
                .fold(0u64, |t, &b| (t << 8) | u64::from(b));
            (lo, 0)
        }
    } else if n >= 8 {
        let lo = u64::from_le_bytes(eight(0));
        let hi = if n > 8 {
            u64::from_le_bytes(eight(n - 8)) >> shift
        } else {
            0
        };
        (lo, hi)
    } else {
        let lo = v[..n]
            .iter()
            .rev()
            .fold(0u64, |t, &b| (t << 8) | u64::from(b));
        (lo, 0)
    };
    Some(u128::from(lo) | (u128::from(hi) << 64))
}

/// The magnitude of `x` when it fits in a word.
#[inline]
pub(in super::super) fn magnitude_64(x: &Number) -> Option<u64> {
    magnitude_128(x).and_then(|v| u64::try_from(v).ok())
}

/// The limbs of the magnitude of `x`, the least significant first, without zeros at the top.
pub(in super::super) fn limbs_of(x: &Number) -> Vec<u64> {
    let mut l: Vec<u64> = if x.byte_order == ByteOrder::LittleEndian {
        x.value
            .chunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(b)
            })
            .collect()
    } else {
        x.value
            .rchunks(8)
            .map(|c| {
                let mut b = [0; 8];
                b[8 - c.len()..].copy_from_slice(c);
                u64::from_be_bytes(b)
            })
            .collect()
    };
    while l.last() == Some(&0) {
        l.pop();
    }
    l
}

/// Reads the magnitude of `x` into `out`, the least significant limb first, and returns the number
/// of its limbs, without zeros at the top: `x.significant().div_ceil(8)`, which `out` must hold,
/// and which are all written.
#[inline]
pub(in super::super) fn read_limbs(x: &Number, out: &mut [core::mem::MaybeUninit<u64>]) -> usize {
    let n = x.significant();
    let (full, rem) = (n / 8, n % 8);
    let v = &x.value;
    // The bytes above the whole limbs: from 8 bytes on, the eight that end at the top, shifted down
    // to them (a copy of a varying length would be a call); below, one at a time.
    let shift = 8 * (8 - rem as u32);
    if x.byte_order == ByteOrder::LittleEndian {
        let limb = |i: usize| {
            let mut b = [0; 8];
            b.copy_from_slice(&v[i..i + 8]);
            u64::from_le_bytes(b)
        };
        for (i, o) in out[..full].iter_mut().enumerate() {
            o.write(limb(8 * i));
        }
        if rem > 0 {
            let top = if n >= 8 {
                limb(n - 8) >> shift
            } else {
                v[..n].iter().rev().fold(0, |t, &b| (t << 8) | u64::from(b))
            };
            out[full].write(top);
        }
    } else {
        let end = v.len();
        let limb = |i: usize| {
            let mut b = [0; 8];
            b.copy_from_slice(&v[i - 8..i]);
            u64::from_be_bytes(b)
        };
        for (i, o) in out[..full].iter_mut().enumerate() {
            o.write(limb(end - 8 * i));
        }
        if rem > 0 {
            let top = if n >= 8 {
                limb(end - n + 8) >> shift
            } else {
                v[end - n..].iter().fold(0, |t, &b| (t << 8) | u64::from(b))
            };
            out[full].write(top);
        }
    }
    n.div_ceil(8)
}

/// The non-negative number of the limbs `x`, the least significant first, in the byte order
/// `order`, in the shortest form.
pub(in super::super) fn from_limbs(x: &[u64], order: ByteOrder) -> Number {
    let n = x.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
    if n == 0 {
        return Number::new_from_u8(0, order);
    }
    let top = x[n - 1];
    let top_bytes = 8 - top.leading_zeros() as usize / 8;
    let len = 8 * (n - 1) + top_bytes;
    // Each limb to its eight bytes, in a vector of exactly the length, not zeroed first; the top
    // limb's bytes as the eight that end at the top, over the limb below (one at a time for a
    // number of one limb), as a copy of a varying length would be a call.
    let mut v: Vec<u8> = Vec::with_capacity(len);
    let dst = v.as_mut_ptr();
    let k = 8 * top_bytes as u32;
    // The eight bytes that end at the top: the top limb's above the limb below's highest.
    let last = if n >= 2 && top_bytes < 8 {
        (top << (64 - k)) | (x[n - 2] >> k)
    } else {
        top
    };
    // SAFETY: the vector holds `len` bytes, and each of them is written before the length is set:
    // the `n - 1` whole limbs at `8 i` (big-endian at `len - 8 i - 8`), then the top limb's bytes,
    // in the eight at `len - 8` (at 0), which agree with those below where they overlap, or one at
    // a time when there is no limb below.
    unsafe {
        if order == ByteOrder::LittleEndian {
            for (i, &l) in x[..n - 1].iter().enumerate() {
                dst.add(8 * i)
                    .cast::<[u8; 8]>()
                    .write_unaligned(l.to_le_bytes());
            }
            if n >= 2 || top_bytes == 8 {
                dst.add(len - 8)
                    .cast::<[u8; 8]>()
                    .write_unaligned(last.to_le_bytes());
            } else {
                for j in 0..top_bytes {
                    *dst.add(j) = (top >> (8 * j)) as u8;
                }
            }
        } else {
            for (i, &l) in x[..n - 1].iter().enumerate() {
                dst.add(len - 8 * i - 8)
                    .cast::<[u8; 8]>()
                    .write_unaligned(l.to_be_bytes());
            }
            if n >= 2 || top_bytes == 8 {
                dst.cast::<[u8; 8]>().write_unaligned(last.to_be_bytes());
            } else {
                for j in 0..top_bytes {
                    *dst.add(top_bytes - 1 - j) = (top >> (8 * j)) as u8;
                }
            }
        }
        v.set_len(len);
    }
    Number {
        value: v,
        negative: false,
        byte_order: order,
    }
}

/// Runs `$body` with `$r` bound to the ring of residues modulo the odd limbs `$m`, without zeros
/// at the top, chosen by their number.
macro_rules! with_odd {
    ($m:expr, |$r:ident| $body:expr) => {
        // A `match` keeps the temporaries of `$m` alive for the whole body.
        match $m {
            m => {
                let m: &[u64] = m;
                match m.len() {
                    1 => {
                        let $r = $crate::math::numbers::modular::word::Odd::new(m[0]);
                        $body
                    }
                    2 => {
                        let $r = $crate::math::numbers::modular::ring::Two::new([m[0], m[1]]);
                        $body
                    }
                    3 => {
                        let $r = $crate::math::numbers::modular::ring::Many::<3>::new([
                            m[0], m[1], m[2],
                        ]);
                        $body
                    }
                    4 => {
                        let $r = $crate::math::numbers::modular::ring::Many::<4>::new([
                            m[0], m[1], m[2], m[3],
                        ]);
                        $body
                    }
                    5 => {
                        let $r = $crate::math::numbers::modular::ring::Many::<5>::new([
                            m[0], m[1], m[2], m[3], m[4],
                        ]);
                        $body
                    }
                    6 => {
                        let $r = $crate::math::numbers::modular::ring::Many::<6>::new([
                            m[0], m[1], m[2], m[3], m[4], m[5],
                        ]);
                        $body
                    }
                    7 => {
                        let $r = $crate::math::numbers::modular::ring::Many::<7>::new([
                            m[0], m[1], m[2], m[3], m[4], m[5], m[6],
                        ]);
                        $body
                    }
                    _ => {
                        let $r = $crate::math::numbers::modular::ring::Long::new(m);
                        $body
                    }
                }
            }
        }
    };
}
pub(in super::super) use with_odd;

/// Runs `$body` with `$r` bound to the ring of residues modulo the limbs `$m`, at least 2, without
/// zeros at the top: [`with_odd!`]'s for an odd modulus, wrapping products for a power of two up to
/// `2^128`, and divisions for the others.
macro_rules! with_ring {
    ($m:expr, |$r:ident| $body:expr) => {
        match $m {
            m => {
                let m: &[u64] = m;
                if m[0] & 1 == 1 {
                    $crate::math::numbers::modular::ring::with_odd!(m, |$r| $body)
                } else {
                    let zeros = m.iter().take_while(|&&l| l == 0).count();
                    let power = m.len() == zeros + 1 && m[zeros].is_power_of_two();
                    let e = 64 * zeros as u64 + u64::from(m[zeros].trailing_zeros());
                    if power && e <= 64 {
                        let $r = $crate::math::numbers::modular::ring::Low::new(e as u32);
                        $body
                    } else if power && e <= 128 {
                        let $r = $crate::math::numbers::modular::ring::Low2::new(e as u32);
                        $body
                    } else {
                        let $r = $crate::math::numbers::modular::ring::Plain::new(m);
                        $body
                    }
                }
            }
        }
    };
}
pub(in super::super) use with_ring;

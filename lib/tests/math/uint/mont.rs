//! `Montgomery<N>` and the modular arithmetic of `Uint<N>` (`pow_mod`, `mul_mod`, `add_mod`,
//! `sub_mod`, `mod_inverse`) against `Number`, Python's `pow` (the data file), and identities.

use a::math::numbers::{ByteOrder, Number};
use a::math::uint::{Montgomery, Uint};

/// Python's `pow(b, e, m)` on seeded random operands, by `data/gen_mont_powm.py`.
const POWM: &str = include_str!("data/mont_powm.txt");

/// A seeded xorshift generator.
fn next(s: &mut u64) -> u64 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    *s
}

/// Runs `$f::<N>($args)` for every width of the sweep: 1 to 9 limbs and 16.
macro_rules! widths {
    ($f:ident($($arg:expr),*)) => {
        $f::<1>($($arg),*);
        $f::<2>($($arg),*);
        $f::<3>($($arg),*);
        $f::<4>($($arg),*);
        $f::<5>($($arg),*);
        $f::<6>($($arg),*);
        $f::<7>($($arg),*);
        $f::<8>($($arg),*);
        $f::<9>($($arg),*);
        $f::<16>($($arg),*);
    };
}

/// `x` as a number.
fn num<const N: usize>(x: &Uint<N>) -> Number {
    Number::from(x)
}

/// The number `x`, which must fit, as a `Uint<N>`.
fn uint<const N: usize>(x: &Number) -> Uint<N> {
    Uint::try_from(x).unwrap()
}

/// `2^k` as a number.
fn pow2(k: usize) -> Number {
    Number::from(1u64) << k
}

/// `R = 2^(64·N)`.
fn r<const N: usize>() -> Number {
    pow2(64 * N)
}

/// Random limbs, at times all ones, zeros or a single bit, which find the carries.
fn limbs<const N: usize>(s: &mut u64) -> [u64; N] {
    let mut l = [0u64; N];
    for x in l.iter_mut() {
        *x = match next(s) % 16 {
            0 => u64::MAX,
            1 => 0,
            2 => 1 << (next(s) % 64),
            _ => next(s),
        };
    }
    l
}

/// A random number of exactly `bits` bits, `1 ≤ bits ≤ 64·N`.
fn with_bits<const N: usize>(s: &mut u64, bits: u32) -> Uint<N> {
    let mut l = limbs::<N>(s);
    let top = ((bits - 1) / 64) as usize;
    for x in l.iter_mut().skip(top + 1) {
        *x = 0;
    }
    let b = (bits - 1) % 64;
    if b < 63 {
        l[top] &= (1 << (b + 1)) - 1;
    }
    l[top] |= 1 << b;
    Uint::from_limbs(l)
}

/// A random odd number of exactly `bits` bits, `bits ≥ 2`.
fn odd_with_bits<const N: usize>(s: &mut u64, bits: u32) -> Uint<N> {
    let mut l = with_bits::<N>(s, bits).into_limbs();
    l[0] |= 1;
    Uint::from_limbs(l)
}

/// A random even number of exactly `bits` bits, `bits ≥ 2`, with a random number of trailing
/// zeros, at times all but the top bit (a power of two).
fn even_with_bits<const N: usize>(s: &mut u64, bits: u32) -> Uint<N> {
    let zeros = 1 + (next(s) % u64::from(bits - 1)) as usize;
    let zeros = if next(s).is_multiple_of(8) {
        bits as usize - 1
    } else {
        zeros
    };
    let high = with_bits::<N>(s, bits - zeros as u32);
    let mut l = high.into_limbs();
    l[0] |= 1;
    uint(&(num(&Uint::<N>::from_limbs(l)) << zeros))
}

/// A random number of `N` limbs: at times 0, 1 or `MAX`.
fn any<const N: usize>(s: &mut u64) -> Uint<N> {
    match next(s) % 16 {
        0 => Uint::ZERO,
        1 => Uint::from_u64(1),
        2 => Uint::MAX,
        _ => Uint::from_limbs(limbs(s)),
    }
}

/// `x − 1`, for a nonzero `x`.
fn dec<const N: usize>(x: &Uint<N>) -> Uint<N> {
    x.overflowing_sub(Uint::from_u64(1)).0
}

/// The number of the hexadecimal digits `x` as a `Uint<N>`.
fn hex<const N: usize>(x: &str) -> Uint<N> {
    uint(&Number::from_str_radix(x, 16, ByteOrder::LittleEndian).unwrap())
}

/// The operands at and above the modulus that `Montgomery` must reduce, and `m − 1` and `MAX`:
/// `m`, `m + 1`, `2·m − 1` (those that fit), `m − 1`, `MAX`.
fn edges<const N: usize>(m: &Uint<N>) -> Vec<Uint<N>> {
    let mut v = vec![*m];
    let (m1, carry) = m.overflowing_add(Uint::from_u64(1));
    if !carry {
        v.push(m1);
    }
    let (twice, carry) = m.overflowing_add(dec(m));
    if !carry {
        v.push(twice);
    }
    v.extend([dec(m), Uint::MAX]);
    v
}

/// The arithmetic modulo `m` and the numbers `Number` needs to check it.
struct Ring<const N: usize> {
    mont: Montgomery<N>,
    mn: Number,
    r: Number,
    r_inv: Number,
}

/// Every operation of `Montgomery` on the operands `a` and `b`, reduced or not, against `Number`;
/// `pow` and `pow_limbs` of `a` by a random exponent of up to two limbs.
fn check_pair<const N: usize>(ring: &Ring<N>, a: &Uint<N>, b: &Uint<N>, s: &mut u64) {
    let Ring { mont, mn, r, r_inv } = ring;
    let m = mont.modulus();
    let (an, bn) = (num(a), num(b));
    let ctx = format!("m = {m:?}, a = {a:?}, b = {b:?}");
    let to = mont.to_montgomery(a);
    assert_eq!(num(&to), (&an * r).rem_euclid(mn), "to, {ctx}");
    assert_eq!(
        num(&mont.from_montgomery(a)),
        (&an * r_inv).rem_euclid(mn),
        "from, {ctx}"
    );
    assert_eq!(
        num(&mont.from_montgomery(&to)),
        an.rem_euclid(mn),
        "round trip, {ctx}"
    );
    let p = (&(&an * &bn) * r_inv).rem_euclid(mn);
    assert_eq!(num(&mont.mul(a, b)), p, "mul, {ctx}");
    assert_eq!(num(&mont.mul(b, a)), p, "mul, {ctx}");
    assert_eq!(
        num(&mont.square(a)),
        (&(&an * &an) * r_inv).rem_euclid(mn),
        "square, {ctx}"
    );
    assert_eq!(
        num(&mont.add(a, b)),
        (&an + &bn).rem_euclid(mn),
        "add, {ctx}"
    );
    assert_eq!(
        num(&mont.sub(a, b)),
        (&an - &bn).rem_euclid(mn),
        "sub, {ctx}"
    );
    assert_eq!(num(&mont.neg(a)), (-&an).rem_euclid(mn), "neg, {ctx}");
    // The inverse of the form a stands for x = a/R: its form is R/x = R²/a.
    let inverse = mont.inverse(a);
    match an.mod_inverse(mn) {
        Some(i) => {
            let form = (&(&i * r) * r).rem_euclid(mn);
            assert_eq!(inverse.map(|x| num(&x)), Some(form), "inverse, {ctx}");
        }
        None => assert_eq!(inverse, None, "inverse, {ctx}"),
    }
    // The power of the form a stands for: the form of (a/R)^e is (a/R)^e·R.
    let e = match next(s) % 3 {
        0 => Uint::<2>::from_u64(next(s) % 40),
        1 => Uint::<2>::from_limbs([next(s), 0]),
        _ => Uint::<2>::from_limbs(limbs(s)),
    };
    let x = (&an * r_inv).rem_euclid(mn);
    let want = (&x.pow_mod(&num(&e), mn) * r).rem_euclid(mn);
    let got = if next(s).is_multiple_of(2) {
        mont.pow(a, &e)
    } else {
        mont.pow_limbs(a, e.as_limbs())
    };
    assert_eq!(num(&got), want, "pow by {e:?}, {ctx}");
}

/// Checks the constants, the conversions and every operation of `Montgomery` modulo `m` against
/// `Number`, on `count` random pairs of operands, reduced and not, and on `edge` of the
/// operands of `edges` (from a random one on; all of them for `usize::MAX`), each with a random
/// partner.
fn check_ring<const N: usize>(m: &Uint<N>, s: &mut u64, count: usize, edge: usize) {
    let mn = num(m);
    let r = r::<N>();
    let ring = Ring {
        mont: Montgomery::new(m),
        r_inv: r.mod_inverse(&mn).unwrap(),
        mn,
        r,
    };
    let mont = &ring.mont;
    assert_eq!(mont.modulus(), m);
    assert_eq!(
        num(&mont.one()),
        ring.r.rem_euclid(&ring.mn),
        "R mod m, m = {m:?}"
    );
    assert_eq!(mont.to_montgomery(&Uint::from_u64(1)), mont.one());
    for _ in 0..count {
        // Reduced operands, and operands of any size, below R.
        let (a, b) = if next(s).is_multiple_of(2) {
            (
                uint(&num(&any::<N>(s)).rem_euclid(&ring.mn)),
                uint(&num(&any::<N>(s)).rem_euclid(&ring.mn)),
            )
        } else {
            (any::<N>(s), any::<N>(s))
        };
        check_pair(&ring, &a, &b, s);
    }
    let e = edges(m);
    let start = next(s) as usize % e.len();
    for k in 0..edge.min(e.len()) {
        let b = any::<N>(s);
        check_pair(&ring, &e[(start + k) % e.len()], &b, s);
    }
}

/// `check_ring` for one odd modulus of each bit length from 2 to `64·N`.
fn ring_at_every_length<const N: usize>(seed: u64) {
    let mut s = seed ^ (N as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let step = if cfg!(miri) { 61 } else { 1 };
    for bits in (2..=64 * N as u32).step_by(step) {
        let m = odd_with_bits::<N>(&mut s, bits);
        check_ring(
            &m,
            &mut s,
            if N <= 4 { 3 } else { 1 },
            if N <= 4 { 2 } else { 1 },
        );
    }
}

#[test]
fn montgomery_agrees_with_number_for_an_odd_modulus_of_every_length() {
    widths!(ring_at_every_length(0x2545_f491_4f6c_dd1d));
}

/// `pow_mod` against `Number::pow_mod` for an odd and an even modulus of each bit length from 2
/// to `64·N`, a random base of `N` limbs, and a random exponent of 1 to 3 limbs.
fn pow_mod_at_every_length<const N: usize>(seed: u64) {
    let mut s = seed ^ (N as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    let step = if cfg!(miri) { 61 } else { 1 };
    for bits in (2..=64 * N as u32).step_by(step) {
        for m in [
            odd_with_bits::<N>(&mut s, bits),
            even_with_bits(&mut s, bits),
        ] {
            let b = any::<N>(&mut s);
            let mn = num(&m);
            let (got, en) = match next(&mut s) % 4 {
                0 => {
                    let e = Uint::<2>::from_limbs(limbs(&mut s));
                    (b.pow_mod(&e, &m), num(&e))
                }
                1 => {
                    let e = Uint::<1>::from_u64(next(&mut s) % 40);
                    (b.pow_mod(&e, &m), num(&e))
                }
                _ => {
                    let e = Uint::<1>::from_limbs(limbs(&mut s));
                    (b.pow_mod(&e, &m), num(&e))
                }
            };
            let want = num(&b).pow_mod(&en, &mn);
            assert_eq!(num(&got), want, "{b:?}^{en:?} mod {m:?}");
        }
    }
}

#[test]
fn pow_mod_agrees_with_number_for_odd_and_even_moduli_of_every_length() {
    widths!(pow_mod_at_every_length(0x6a09_e667_f3bc_c908));
}

/// Moduli of special shape for `N` limbs.
fn special_moduli<const N: usize>(s: &mut u64) -> Vec<Uint<N>> {
    let bits = 64 * N;
    let mut v = vec![
        Uint::MAX,                                      // 2^(64N) − 1
        uint(&(pow2(bits - 1) + Number::from(1u64))),   // 2^(64N−1) + 1
        uint(&pow2(bits - 1)),                          // 2^(64N−1)
        Uint::MAX.overflowing_sub(Uint::from_u64(1)).0, // 2^(64N) − 2
        Uint::from_u64(2),
        Uint::from_u64(3),
        Uint::from_u64(u64::MAX),
        Uint::from_u64(1 << 63),
    ];
    // The top limb u64::MAX, the rest random, odd and even.
    for parity in [1, 0] {
        let mut l = limbs::<N>(s);
        l[N - 1] = u64::MAX;
        l[0] = (l[0] & !1) | parity;
        v.push(Uint::from_limbs(l));
    }
    // 2^k ± 1 and 2^k for a few k.
    for k in [2, 63, 64, 65, bits / 2, bits - 2, bits - 1] {
        if k >= 2 && k < bits {
            let p = pow2(k);
            v.push(uint(&(&p + &Number::from(1u64))));
            v.push(uint(&(&p - &Number::from(1u64))));
            v.push(uint(&p));
        }
    }
    v
}

/// `pow_mod`, `Montgomery::pow`, `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` against
/// `Number` modulo the special moduli, for the bases 0, 1, m − 1, m, m + 1, `MAX` and random
/// ones, and the exponents 0, 1, 2, `MAX` of 1 and `N` limbs, m − 1 and random ones.
fn special<const N: usize>(seed: u64) {
    let mut s = seed ^ (N as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    for m in special_moduli::<N>(&mut s) {
        let mn = num(&m);
        let bases = [
            Uint::ZERO,
            Uint::from_u64(1),
            dec(&m),
            m,
            m.overflowing_add(Uint::from_u64(1)).0,
            Uint::MAX,
            any(&mut s),
        ];
        let exps: [Number; 7] = [
            Number::from(0u64),
            Number::from(1u64),
            Number::from(2u64),
            Number::from(u64::MAX),
            num(&Uint::<N>::MAX),
            num(&dec(&m)),
            num(&any::<N>(&mut s)),
        ];
        for b in &bases {
            let bn = num(b);
            for en in &exps {
                let e = uint::<N>(en);
                let want = bn.pow_mod(en, &mn);
                assert_eq!(num(&b.pow_mod(&e, &m)), want, "{b:?}^{e:?} mod {m:?}");
                if m.as_limbs()[0] & 1 == 1 && m != Uint::from_u64(1) {
                    let mont = Montgomery::new(&m);
                    let x = mont.pow(&mont.to_montgomery(b), &e);
                    assert_eq!(
                        num(&mont.from_montgomery(&x)),
                        want,
                        "pow {b:?}^{e:?} mod {m:?}"
                    );
                }
            }
            for c in &bases {
                let cn = num(c);
                let ctx = format!("{b:?}, {c:?} mod {m:?}");
                assert_eq!(
                    num(&b.mul_mod(c, &m)),
                    (&bn * &cn).rem_euclid(&mn),
                    "mul, {ctx}"
                );
                assert_eq!(
                    num(&b.add_mod(c, &m)),
                    (&bn + &cn).rem_euclid(&mn),
                    "add, {ctx}"
                );
                assert_eq!(
                    num(&b.sub_mod(c, &m)),
                    (&bn - &cn).rem_euclid(&mn),
                    "sub, {ctx}"
                );
            }
            let want = bn.mod_inverse(&mn).map(|x| uint::<N>(&x));
            assert_eq!(b.mod_inverse(&m), want, "inverse of {b:?} mod {m:?}");
        }
        if m.as_limbs()[0] & 1 == 1 {
            check_ring(&m, &mut s, 4, usize::MAX);
        }
    }
}

#[test]
fn moduli_of_special_shape_and_edge_operands_agree_with_number() {
    widths!(special(0x94d0_49bb_1331_11eb));
}

/// `Uint<N>` of a number given in hexadecimal.
fn prime<const N: usize>(x: &str) -> Uint<N> {
    hex(x)
}

/// Fermat's little theorem, the inverse as `x^(p−2)`, and agreement with `Number` for random
/// bases and full-length exponents, modulo the prime `p`.
fn check_prime<const N: usize>(p: &Uint<N>, seed: u64) {
    let mut s = seed;
    let mont = Montgomery::new(p);
    let pn = num(p);
    let p1 = dec(p);
    let p2 = dec(&p1);
    for _ in 0..if cfg!(miri) { 1 } else { 8 } {
        let x = uint::<N>(&num(&any::<N>(&mut s)).rem_euclid(&pn));
        let f = mont.to_montgomery(&x);
        if x.is_zero() {
            assert_eq!(mont.inverse(&f), None);
            continue;
        }
        assert_eq!(mont.pow(&f, &p1), mont.one(), "Fermat, {x:?} mod {p:?}");
        assert_eq!(mont.pow(&f, p), f, "x^p = x, {x:?} mod {p:?}");
        let inv = mont.inverse(&f).unwrap();
        assert_eq!(mont.pow(&f, &p2), inv, "x^(p−2) = 1/x, {x:?} mod {p:?}");
        assert_eq!(mont.mul(&f, &inv), mont.one());
        assert_eq!(x.mod_inverse(p), Some(mont.from_montgomery(&inv)));
        let e = Uint::<N>::from_limbs(limbs(&mut s));
        let want = num(&x).pow_mod(&num(&e), &pn);
        assert_eq!(num(&x.pow_mod(&e, p)), want, "{x:?}^{e:?} mod {p:?}");
    }
    check_ring(p, &mut s, 8, usize::MAX);
}

#[test]
fn the_primes_of_the_named_curves_obey_fermat_and_agree_with_number() {
    // P-256, P-384, secp256k1, BN254's base field, BLS12-381's base field (381 bits in 6 limbs),
    // P-521 (in 9 limbs) and 2^255 − 19.
    let p256 = "ffffffff00000001000000000000000000000000ffffffffffffffffffffffff";
    let p384 = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff";
    let k256 = "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f";
    let bn254 = "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47";
    let bls = "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
    let c25519 = "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed";
    let p521 = num(&Uint::<9>::ZERO) + pow2(521) - Number::from(1u64);
    check_prime::<4>(&prime(p256), 1);
    check_prime::<4>(&prime(k256), 2);
    check_prime::<4>(&prime(bn254), 3);
    check_prime::<4>(&prime(c25519), 4);
    check_prime::<6>(&prime(p384), 5);
    check_prime::<6>(&prime(bls), 6);
    check_prime::<9>(&uint(&p521), 7);
    // The same primes in wider integers, where the top limbs are zero.
    check_prime::<5>(&prime(p256), 8);
    check_prime::<8>(&prime(bls), 9);
    assert_eq!(prime::<6>(bls).bits(), 381);
}

/// A product of two primes and its factors.
fn rsa<const N: usize>(p: &str, q: &str, seed: u64) {
    let (p, q) = (hex::<N>(p), hex::<N>(q));
    let m = p.mul_mod(&q, &Uint::MAX);
    assert_eq!(num(&m), &num(&p) * &num(&q), "the product fits");
    let mn = num(&m);
    let mont = Montgomery::new(&m);
    // Non-units: the factors and their multiples have no inverse.
    for x in [p, q, p.mul_mod(&Uint::from_u64(12345), &m), Uint::ZERO, m] {
        assert_eq!(mont.inverse(&x), None, "{x:?} mod {m:?}");
        assert_eq!(x.mod_inverse(&m), None, "{x:?} mod {m:?}");
    }
    // Euler: x^((p−1)(q−1)) = 1 for a unit x, and RSA's round trip with e = 65537.
    let phi = &(&num(&p) - &Number::from(1u64)) * &(&num(&q) - &Number::from(1u64));
    let e = Number::from(65537u64);
    let d = e.mod_inverse(&phi).unwrap();
    let mut s = seed;
    for _ in 0..if cfg!(miri) { 1 } else { 4 } {
        let x = uint::<N>(&num(&any::<N>(&mut s)).rem_euclid(&mn));
        let c = x.pow_mod(&uint::<N>(&e), &m);
        assert_eq!(num(&c), num(&x).pow_mod(&e, &mn));
        assert_eq!(c.pow_mod(&uint::<N>(&d), &m), x, "RSA round trip of {x:?}");
        if num(&x).mod_inverse(&mn).is_some() {
            let f = mont.to_montgomery(&x);
            assert_eq!(mont.pow(&f, &uint::<N>(&phi)), mont.one());
        }
    }
    check_ring(&m, &mut s, 8, usize::MAX);
}

#[test]
fn rsa_like_products_of_two_primes_round_trip_and_have_no_inverse_for_their_factors() {
    rsa::<2>("c4bb895c60809a11", "d7f20e07ed4202ef", 1);
    rsa::<4>(
        "ee544eeb36cbb40403ed3511d7ec2061",
        "f93bfb39a2ef283a4e0433b7df28435f",
        2,
    );
    rsa::<8>(
        "d092a4d94e4f86d708e369b041747c23c1e3efacf3f5fa17dba8b6150ada364b",
        "fa9e5c47afca1560936e0b4f1fd8218adbcf34d896a8dab3189d51ec6c9084ad",
        3,
    );
    rsa::<16>(
        "c64f96bf782a3ae88384a7f75bd2470b8c7d38462e52011aeb2842b9d326e9c250c4d7db9ffeafc4f1fb2337cb61c8adf6a6c4118327575b32776fead50db7b3",
        "de5f21092932df25ad16dc33307c48265a6c49686a17d2205b3627fc9530d16899bdb0511925535aad952622a2d2d4bbd563fe7ef91d8131e220bb921a9eb569",
        4,
    );
}

/// `pow_mod` and `Montgomery::pow_limbs` with an exponent of `M` limbs modulo `N`-limb moduli,
/// odd and even, the exponent's length at and next to the window's thresholds.
fn long_exponents<const N: usize, const M: usize>(seed: u64) {
    let mut s = seed;
    let lengths = [1, 12, 13, 24, 25, 80, 81, 240, 241, 672, 673, 64 * M as u32];
    for &len in lengths.iter().filter(|&&l| l <= 64 * M as u32) {
        if cfg!(miri) && len > 100 {
            continue;
        }
        let e = with_bits::<M>(&mut s, len);
        let en = num(&e);
        let mbits = 2 + (next(&mut s) % (64 * N as u64 - 1)) as u32;
        for m in [
            odd_with_bits::<N>(&mut s, mbits),
            even_with_bits(&mut s, mbits),
        ] {
            let b = any::<N>(&mut s);
            let want = num(&b).pow_mod(&en, &num(&m));
            assert_eq!(num(&b.pow_mod(&e, &m)), want, "{b:?}^{e:?} mod {m:?}");
            if m.as_limbs()[0] & 1 == 1 {
                let mont = Montgomery::new(&m);
                let x = mont.pow_limbs(&mont.to_montgomery(&b), e.as_limbs());
                assert_eq!(num(&mont.from_montgomery(&x)), want);
                // Zero limbs on top of the exponent change nothing.
                let mut wide = e.as_limbs().to_vec();
                wide.extend([0, 0]);
                assert_eq!(mont.pow_limbs(&mont.to_montgomery(&b), &wide), x);
            }
        }
    }
}

#[test]
fn exponents_of_other_widths_and_every_window_size_agree_with_number() {
    long_exponents::<1, 12>(1);
    long_exponents::<1, 3>(2);
    long_exponents::<4, 1>(3);
    long_exponents::<4, 11>(4);
    long_exponents::<4, 16>(5);
    long_exponents::<6, 2>(6);
    long_exponents::<8, 4>(7);
    long_exponents::<8, 12>(8);
    long_exponents::<9, 1>(9);
    long_exponents::<16, 8>(10);
    // Exponents of no limbs, and of zero limbs only.
    let m = Uint::<4>::from_u64(1_000_003);
    let mont = Montgomery::new(&m);
    let x = mont.to_montgomery(&Uint::from_u64(5));
    assert_eq!(mont.pow(&x, &Uint::<0>::ZERO), mont.one());
    assert_eq!(mont.pow_limbs(&x, &[]), mont.one());
    assert_eq!(mont.pow_limbs(&x, &[0, 0, 0]), mont.one());
    assert_eq!(mont.pow(&Uint::ZERO, &Uint::<2>::ZERO), mont.one());
    assert_eq!(
        Uint::<4>::from_u64(5).pow_mod(&Uint::<0>::ZERO, &m),
        Uint::from_u64(1)
    );
}

/// Widths past 32 and 128 limbs, where the table of odd powers is smaller.
fn wide<const N: usize>(seed: u64) {
    let mut s = seed;
    let lengths: &[u32] = if cfg!(miri) { &[5] } else { &[5, 30, 100, 300] };
    for &len in lengths {
        let e = with_bits::<5>(&mut s, len);
        for m in [
            odd_with_bits::<N>(&mut s, 64 * N as u32),
            odd_with_bits::<N>(&mut s, 70),
            even_with_bits::<N>(&mut s, 64 * N as u32 - 3),
        ] {
            let b = any::<N>(&mut s);
            let want = num(&b).pow_mod(&num(&e), &num(&m));
            assert_eq!(
                num(&b.pow_mod(&e, &m)),
                want,
                "{len} bits mod {:?}",
                m.bits()
            );
        }
    }
}

#[test]
fn wide_integers_use_a_smaller_table_and_agree_with_number() {
    wide::<33>(1);
    wide::<129>(2);
}

/// `mul_mod`, `add_mod`, `sub_mod` and `mod_inverse` against `Number` for random moduli, odd and
/// even, of every length, and operands of any size.
fn plain_mod<const N: usize>(seed: u64) {
    let mut s = seed ^ (N as u64).wrapping_mul(0x94d0_49bb_1331_11eb);
    let step = if cfg!(miri) { 61 } else { 1 };
    for bits in (1..=64 * N as u32).step_by(step) {
        let m = match next(&mut s) % 3 {
            0 if bits >= 2 => even_with_bits::<N>(&mut s, bits),
            _ => with_bits::<N>(&mut s, bits),
        };
        let mn = num(&m);
        let (a, b) = (any::<N>(&mut s), any::<N>(&mut s));
        let (an, bn) = (num(&a), num(&b));
        let ctx = format!("{a:?}, {b:?} mod {m:?}");
        assert_eq!(
            num(&a.mul_mod(&b, &m)),
            (&an * &bn).rem_euclid(&mn),
            "mul, {ctx}"
        );
        assert_eq!(
            num(&a.add_mod(&b, &m)),
            (&an + &bn).rem_euclid(&mn),
            "add, {ctx}"
        );
        assert_eq!(
            num(&a.sub_mod(&b, &m)),
            (&an - &bn).rem_euclid(&mn),
            "sub, {ctx}"
        );
        let want = an.mod_inverse(&mn).map(|x| uint::<N>(&x));
        assert_eq!(a.mod_inverse(&m), want, "inverse, {ctx}");
        // An odd operand modulo an even modulus, which often has an inverse.
        let mut odd = a.into_limbs();
        odd[0] |= 1;
        let odd = Uint::from_limbs(odd);
        let want = num(&odd).mod_inverse(&mn).map(|x| uint::<N>(&x));
        assert_eq!(odd.mod_inverse(&m), want, "inverse of {odd:?} mod {m:?}");
    }
}

#[test]
fn mul_add_sub_mod_and_mod_inverse_agree_with_number_for_any_modulus() {
    widths!(plain_mod(0x5851_f42d_4c95_7f2d));
}

#[test]
fn inverses_exist_exactly_for_units() {
    // Modulo 3·5·7·11·13 = 15015 every residue, against the gcd.
    let m = Uint::<2>::from_u64(15015);
    let mont = Montgomery::new(&m);
    let mn = num(&m);
    let step = if cfg!(miri) { 97 } else { 1 };
    for x in (0..15015u64).step_by(step) {
        let xn = Number::from(x);
        let unit = xn.gcd(&mn) == 1u64;
        let inv = Uint::<2>::from_u64(x).mod_inverse(&m);
        assert_eq!(inv.is_some(), unit, "{x}");
        if let Some(y) = inv {
            assert_eq!(y.mul_mod(&Uint::from_u64(x), &m), Uint::from_u64(1));
        }
        let f = mont.inverse(&mont.to_montgomery(&Uint::from_u64(x)));
        assert_eq!(f.map(|f| mont.from_montgomery(&f)), inv);
    }
    // Modulo 2^64·15015 the odd units are those prime to 15015.
    let m = Uint::<2>::from_limbs([0, 15015]);
    for x in (1..3000u64).step_by(step) {
        let want = Number::from(x).mod_inverse(&num(&m)).map(|y| uint::<2>(&y));
        assert_eq!(Uint::<2>::from_u64(x).mod_inverse(&m), want, "{x}");
    }
}

#[test]
fn powers_of_two_as_moduli_take_the_low_bits() {
    for k in 1..128usize {
        let m = uint::<2>(&pow2(k));
        let mut s = (k as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let b = Uint::<2>::from_limbs(limbs(&mut s));
        let e = Uint::<2>::from_limbs(limbs(&mut s));
        let want = num(&b).pow_mod(&num(&e), &num(&m));
        assert_eq!(num(&b.pow_mod(&e, &m)), want, "k = {k}");
        // An even base vanishes once the exponent is at least k.
        let even = Uint::<2>::from_u64(6);
        assert_eq!(
            even.pow_mod(&Uint::<1>::from_u64(k as u64), &m),
            Uint::ZERO,
            "k = {k}"
        );
    }
}

#[test]
fn modulo_one_every_result_is_zero() {
    let one = Uint::<3>::from_u64(1);
    for b in [Uint::ZERO, Uint::from_u64(1), Uint::from_u64(7), Uint::MAX] {
        for e in [Uint::<1>::ZERO, Uint::from_u64(1), Uint::MAX] {
            assert_eq!(b.pow_mod(&e, &one), Uint::ZERO);
        }
        assert_eq!(b.mul_mod(&b, &one), Uint::ZERO);
        assert_eq!(b.add_mod(&b, &one), Uint::ZERO);
        assert_eq!(b.sub_mod(&Uint::from_u64(5), &one), Uint::ZERO);
        assert_eq!(b.mod_inverse(&one), Some(Uint::ZERO));
    }
    // 0^0 = 1 elsewhere, as for Number.
    let m = Uint::<3>::from_u64(10);
    assert_eq!(Uint::ZERO.pow_mod(&Uint::<1>::ZERO, &m), Uint::from_u64(1));
    assert_eq!(Uint::ZERO.pow_mod(&Uint::<1>::from_u64(3), &m), Uint::ZERO);
}

#[test]
fn the_powers_of_r_go_through_the_add_back_of_the_long_division() {
    // Modulo 2^(64N−1) + 1 the remainder of R = 2^(64N) takes the estimated quotient 2 where the
    // true one is 1, past the test with the second limb when N ≥ 3: Knuth's step D6.
    fn check<const N: usize>() {
        let m = uint::<N>(&(pow2(64 * N - 1) + Number::from(1u64)));
        let mont = Montgomery::new(&m);
        assert_eq!(num(&mont.one()), r::<N>().rem_euclid(&num(&m)));
        let x = Uint::<N>::MAX;
        assert_eq!(
            num(&mont.from_montgomery(&mont.to_montgomery(&x))),
            num(&x).rem_euclid(&num(&m))
        );
        assert_eq!(
            num(&x.mul_mod(&x, &m)),
            (&num(&x) * &num(&x)).rem_euclid(&num(&m))
        );
    }
    check::<3>();
    check::<4>();
    check::<8>();
}

#[test]
fn python_pow_agrees() {
    // Each line: limbs, then the modulus, base, exponent and Python's pow(b, e, m), in hex.
    let mut lines = 0;
    for line in POWM
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let n: usize = f[0].parse().unwrap();
        let v: Vec<Number> = f[1..]
            .iter()
            .map(|x| Number::from_str_radix(x, 16, ByteOrder::LittleEndian).unwrap())
            .collect();
        fn one<const N: usize>(v: &[Number]) {
            let (m, b, e) = (uint::<N>(&v[0]), uint::<N>(&v[1]), uint::<16>(&v[2]));
            assert_eq!(num(&b.pow_mod(&e, &m)), v[3], "{b:?}^{e:?} mod {m:?}");
        }
        match n {
            1 => one::<1>(&v),
            2 => one::<2>(&v),
            3 => one::<3>(&v),
            4 => one::<4>(&v),
            5 => one::<5>(&v),
            6 => one::<6>(&v),
            7 => one::<7>(&v),
            8 => one::<8>(&v),
            9 => one::<9>(&v),
            16 => one::<16>(&v),
            _ => panic!("width {n}"),
        }
        lines += 1;
    }
    assert!(lines >= 40, "{lines} lines");
}

#[test]
#[should_panic(expected = "the modulus must be odd and greater than 1")]
fn an_even_modulus_is_refused() {
    Montgomery::new(&Uint::<4>::from_u64(1 << 40));
}

#[test]
#[should_panic(expected = "the modulus must be odd and greater than 1")]
fn the_modulus_one_is_refused() {
    Montgomery::new(&Uint::<4>::from_u64(1));
}

#[test]
#[should_panic(expected = "the modulus must be odd and greater than 1")]
fn the_modulus_zero_is_refused() {
    Montgomery::new(&Uint::<2>::ZERO);
}

#[test]
#[should_panic(expected = "the modulus must be odd and greater than 1")]
fn no_limbs_hold_no_modulus() {
    Montgomery::new(&Uint::<0>::ZERO);
}

#[test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
fn pow_mod_by_zero_panics() {
    Uint::<4>::from_u64(3).pow_mod(&Uint::<1>::from_u64(2), &Uint::ZERO);
}

#[test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
fn pow_mod_in_no_limbs_panics() {
    Uint::<0>::ZERO.pow_mod(&Uint::<0>::ZERO, &Uint::ZERO);
}

#[test]
fn every_function_in_no_limbs_panics_for_its_only_modulus_zero() {
    let z = Uint::<0>::ZERO;
    let calls: [fn(Uint<0>); 4] = [
        |z| {
            let _ = z.mul_mod(&z, &z);
        },
        |z| {
            let _ = z.add_mod(&z, &z);
        },
        |z| {
            let _ = z.sub_mod(&z, &z);
        },
        |z| {
            let _ = z.mod_inverse(&z);
        },
    ];
    for call in calls {
        let e = std::panic::catch_unwind(|| call(z)).unwrap_err();
        let message = e.downcast_ref::<&str>().copied().unwrap_or_default();
        assert_eq!(
            message,
            "attempt to calculate the remainder with a divisor of zero"
        );
    }
}

#[test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
fn mul_mod_by_zero_panics() {
    Uint::<4>::from_u64(3).mul_mod(&Uint::from_u64(2), &Uint::ZERO);
}

#[test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
fn add_mod_by_zero_panics() {
    Uint::<4>::from_u64(3).add_mod(&Uint::from_u64(2), &Uint::ZERO);
}

#[test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
fn sub_mod_by_zero_panics() {
    Uint::<4>::from_u64(3).sub_mod(&Uint::from_u64(2), &Uint::ZERO);
}

#[test]
#[should_panic(expected = "attempt to calculate the remainder with a divisor of zero")]
fn mod_inverse_by_zero_panics() {
    Uint::<4>::from_u64(3).mod_inverse(&Uint::ZERO);
}

/// A heavier sweep: 16 random moduli, odd and even, of each length from 2 to `64·N` bits, with
/// exponents of `N` limbs (62,304 powers in all). About 6 s in release here and a minute and a
/// half in the debug profile: run it with `cargo test --release --test math -- --ignored
/// uint::mont`.
fn heavy<const N: usize>(seed: u64) {
    let mut s = seed ^ (N as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
    for bits in 2..=64 * N as u32 {
        for _ in 0..16 {
            let m = if next(&mut s).is_multiple_of(2) {
                odd_with_bits::<N>(&mut s, bits)
            } else {
                even_with_bits(&mut s, bits)
            };
            let (b, e) = (any::<N>(&mut s), any::<N>(&mut s));
            let want = num(&b).pow_mod(&num(&e), &num(&m));
            assert_eq!(num(&b.pow_mod(&e, &m)), want, "{b:?}^{e:?} mod {m:?}");
            if m.as_limbs()[0] & 1 == 1 {
                check_ring(&m, &mut s, 2, usize::MAX);
            }
        }
    }
}

#[test]
#[ignore = "a heavier sweep: run it in release"]
fn a_heavier_sweep_agrees_with_number() {
    widths!(heavy(0x1234_5678_9abc_def1));
}

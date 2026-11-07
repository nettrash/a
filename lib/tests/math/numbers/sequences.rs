//! Tests of the integer sequences: Fibonacci and Lucas numbers, Lucas sequences, exact and modulo a
//! number, primorials and partition numbers.
//!
//! The references are independent of the implementation: recurrences on limbs of 64 bits added one
//! by one, products of a limb at a time, matrix powers on `u128`s, and values from PARI/GP 2.19:
//! the number of decimal digits and the residues modulo `2^61 - 1` and `10^9 + 7`.

use a::math::numbers::{ByteOrder, Number};

use super::common::*;

// References on limbs of 64 bits, the least significant first.

/// `x + y`.
fn add(x: &[u64], y: &[u64]) -> Vec<u64> {
    let (a, b) = if x.len() >= y.len() { (x, y) } else { (y, x) };
    let mut r = Vec::with_capacity(a.len() + 1);
    let mut carry = 0u64;
    for (i, &x) in a.iter().enumerate() {
        let s = u128::from(x) + u128::from(b.get(i).copied().unwrap_or(0)) + u128::from(carry);
        r.push(s as u64);
        carry = (s >> 64) as u64;
    }
    if carry != 0 {
        r.push(carry);
    }
    r
}

/// `x - y` for `x >= y`.
fn sub(x: &[u64], y: &[u64]) -> Vec<u64> {
    let mut r = Vec::with_capacity(x.len());
    let mut borrow = 0u64;
    for (i, &v) in x.iter().enumerate() {
        let (d, b1) = v.overflowing_sub(y.get(i).copied().unwrap_or(0));
        let (d, b2) = d.overflowing_sub(borrow);
        r.push(d);
        borrow = u64::from(b1 | b2);
    }
    assert_eq!(borrow, 0, "the reference subtracted a larger number");
    while r.len() > 1 && r.last() == Some(&0) {
        r.pop();
    }
    r
}

/// `x m`, in place.
fn times(x: &mut Vec<u64>, m: u64) {
    let mut carry = 0u64;
    for limb in x.iter_mut() {
        let p = u128::from(*limb) * u128::from(m) + u128::from(carry);
        *limb = p as u64;
        carry = (p >> 64) as u64;
    }
    if carry != 0 {
        x.push(carry);
    }
}

/// The bytes of the limbs `x`, least significant first, in the shortest form.
fn bytes(x: &[u64]) -> Vec<u8> {
    let mut v: Vec<u8> = x.iter().flat_map(|l| l.to_le_bytes()).collect();
    while v.len() > 1 && v.last() == Some(&0) {
        v.pop();
    }
    if v.is_empty() {
        v.push(0);
    }
    v
}

/// Checks that `n` is the non-negative value of the limbs `x`, in byte order `order`, in the
/// shortest form.
fn assert_limbs(n: &Number, x: &[u64], order: ByteOrder, what: &dyn Fn() -> String) {
    assert!(n.byte_order == order, "{}: byte order", what());
    assert!(!n.negative, "{}: sign", what());
    assert_eq!(le_bytes(n), bytes(x), "{}", what());
}

/// The residue of `n` modulo `m`, as a `u64`.
fn residue(n: &Number, m: u64) -> u64 {
    let r = n % &Number::new_from_u64(m, ByteOrder::LittleEndian);
    u64::try_from(&r).expect("a residue below a u64")
}

/// The primes up to `n`, by a sieve of Eratosthenes.
fn primes_to(n: usize) -> Vec<u64> {
    let mut composite = vec![false; n + 1];
    let mut primes = Vec::new();
    for i in 2..=n {
        if !composite[i] {
            primes.push(i as u64);
            let mut j = i * i;
            while j <= n {
                composite[j] = true;
                j += i;
            }
        }
    }
    primes
}

/// PARI/GP 2.19's values: the sequence, the index, the number of decimal digits, and the residues
/// modulo `2^61 - 1` and `10^9 + 7`.
const PARI: [(&str, u64, usize, u64, u64); 38] = [
    ("p", 417, 20, 541220193622112949, 415915619),
    ("p", 418, 20, 1723438945096382051, 614653520),
    ("p", 500, 22, 1239552388271125880, 168879716),
    ("p", 1000, 32, 1678019938100056892, 709496666),
    ("p", 1500, 40, 149782402576588915, 188746905),
    ("p", 2000, 46, 1196672395906640354, 298613045),
    ("p", 3000, 57, 1947475239792742099, 701511400),
    ("p", 10000, 107, 554611280972349736, 17783467),
    ("p", 12345, 119, 903708336282495471, 549711015),
    ("p", 100000, 347, 1031897868136555579, 49037875),
    ("p", 1000000, 1108, 2012848153370158893, 419139981),
    ("p", 2718281, 1830, 2532322902668178, 85178066),
    ("p", 20000001, 4974, 2215371483203106882, 44842207),
    ("F", 187, 39, 133805124077074378, 619822561),
    ("F", 188, 39, 2134925130038214014, 180664866),
    ("F", 372, 78, 357376282709179754, 687587382),
    ("F", 373, 78, 2258442740463199491, 163035358),
    ("F", 1000, 209, 100192510155549716, 517691607),
    ("F", 4096, 856, 1776723183726168900, 124518512),
    ("F", 10001, 2090, 2200516769918533960, 24223428),
    ("F", 100000, 20899, 396479946705568275, 911435502),
    ("F", 1000000, 208988, 1953611743537552514, 918091266),
    ("L", 185, 39, 1257868875378956628, 62704368),
    ("L", 186, 39, 572333251406703071, 678802810),
    ("L", 370, 78, 734137072125491555, 738756567),
    ("L", 371, 78, 1119529116878033722, 899726781),
    ("L", 1000, 209, 1424179987648068153, 697468278),
    ("L", 4096, 857, 1697736134749070728, 296947521),
    ("L", 10001, 2091, 1534381439713500000, 567216148),
    ("L", 100000, 20899, 284203514264748148, 23800955),
    ("L", 1000000, 208988, 569919499037307402, 150710060),
    ("pr", 103, 41, 1006593349269476650, 374554194),
    ("pr", 1000, 416, 31435497341534465, 237762415),
    ("pr", 8191, 3503, 2266994784943692819, 890815867),
    ("pr", 8192, 3503, 2266994784943692819, 890815867),
    ("pr", 8209, 3507, 1607105248264166601, 707401019),
    ("pr", 10000, 4298, 337159011178168082, 561055691),
    ("pr", 100000, 43293, 1972616183870419072, 41232855),
];

/// Checks a value against PARI's: its residues, and its digits from its bits (`10^(d-1) <= x`
/// needs at least `(d - 1) log2(10)` bits, and `x < 10^d` at most `d log2(10)` plus one).
fn assert_pari(x: &Number, name: &str, n: u64, digits: usize, r1: u64, r2: u64) {
    assert_eq!(residue(x, (1 << 61) - 1), r1, "{name}({n}) mod 2^61 - 1");
    assert_eq!(residue(x, 1_000_000_007), r2, "{name}({n}) mod 10^9 + 7");
    let bits = x.bits() as f64;
    let lg = core::f64::consts::LOG2_10;
    assert!(
        bits > (digits - 1) as f64 * lg && bits <= digits as f64 * lg + 1.0,
        "{name}({n}): {bits} bits for {digits} digits"
    );
}

/// Whether the reference value is cheap enough for the build: Miri takes the smallest only.
fn affordable(n: u64) -> bool {
    !cfg!(miri) || n <= 400
}

// Fibonacci and Lucas numbers

/// Whether the test of index `n` runs: every one natively, and under Miri the ends of the tables
/// and of the paths.
fn sampled(n: u64) -> bool {
    !cfg!(miri) || matches!(n, 0..=2 | 92..=94 | 183..=188 | 370..=376 | 400)
}

#[test]
fn fibonacci_and_lucas_numbers_follow_their_recurrences() {
    let top = size(3000, 400) as u64;
    for order in ORDERS {
        let (mut f0, mut f1) = (vec![0u64], vec![1u64]);
        let (mut l0, mut l1) = (vec![2u64], vec![1u64]);
        for n in 0..=top {
            if sampled(n) {
                let f = Number::fibonacci(n, order);
                assert_limbs(&f, &f0, order, &|| format!("F_{n} in {order:?}"));
                let l = Number::lucas(n, order);
                assert_limbs(&l, &l0, order, &|| format!("L_{n} in {order:?}"));
            }
            let f2 = add(&f0, &f1);
            (f0, f1) = (f1, f2);
            let l2 = add(&l0, &l1);
            (l0, l1) = (l1, l2);
        }
    }
}

#[test]
fn fibonacci_and_lucas_numbers_match_pari() {
    for &(name, n, digits, r1, r2) in &PARI {
        if (name == "F" || name == "L") && affordable(n) {
            for order in ORDERS {
                let x = if name == "F" {
                    Number::fibonacci(n, order)
                } else {
                    Number::lucas(n, order)
                };
                assert!(x.byte_order == order && !x.negative);
                assert_pari(&x, name, n, digits, r1, r2);
            }
        }
    }
}

#[test]
fn fibonacci_and_lucas_numbers_satisfy_their_identities() {
    // `F_(2n) = F_n L_n`, `L_n^2 - 5 F_n^2 = 4 (-1)^n` and `F_(m+n) = F_m L_n - (-1)^n F_(m-n)`,
    // across the tables' ends, the doubling's sizes and the transforms' start.
    let mut state = 0x9E37_79B9_7F4A_7C15;
    let count = size(60, 3);
    for i in 0..count {
        let n = match i {
            0..=9 => [184, 185, 186, 187, 371, 372, 373, 374, 4095, 4097][i],
            _ => 1 + xorshift(&mut state) % if i < count / 2 { 2000 } else { 60_000 },
        };
        let n = if cfg!(miri) { n.min(400) } else { n };
        let order = ORDERS[i % 2];
        let f = Number::fibonacci(n, order);
        let l = Number::lucas(n, order);
        let f2 = Number::fibonacci(2 * n, order);
        assert_eq!(f2, &f * &l, "F_(2n) = F_n L_n for n = {n}");
        let four = Number::new_from_i32(if n % 2 == 0 { 4 } else { -4 }, order);
        assert_eq!(
            &(&l * &l) - &(&(&f * &f) * 5u8),
            four,
            "L_n^2 - 5 F_n^2 for n = {n}"
        );
        let m = n + 1 + xorshift(&mut state) % 500;
        let sign = if n % 2 == 0 { 1i8 } else { -1 };
        assert_eq!(
            Number::fibonacci(m + n, order),
            &(&Number::fibonacci(m, order) * &l) - &(&Number::fibonacci(m - n, order) * sign),
            "F_(m+n) for m = {m}, n = {n}"
        );
    }
}

#[test]
fn fibonacci_numbers_agree_with_the_lucas_sequences_modulo_primes() {
    // `F_n mod p` from the long number and from the sequence modulo `p`, for long indices.
    let p = Number::new_from_u64((1 << 61) - 1, ByteOrder::LittleEndian);
    let one = Number::new_from_i32(1, ByteOrder::BigEndian);
    let minus_one = Number::new_from_i32(-1, ByteOrder::BigEndian);
    for n in [size(123_457, 300) as u64, size(1_000_003, 301) as u64] {
        let index = Number::new_from_u64(n, ByteOrder::BigEndian);
        let (u, v) = Number::lucas_sequence_mod(&one, &minus_one, &index, &p);
        assert_eq!(
            u,
            &Number::fibonacci(n, ByteOrder::LittleEndian) % &p,
            "F_{n} mod p"
        );
        assert_eq!(
            v,
            &Number::lucas(n, ByteOrder::BigEndian) % &p,
            "L_{n} mod p"
        );
    }
}

// Lucas sequences

/// `(U_n, V_n)` by the recurrence `X_(k+2) = P X_(k+1) - Q X_k`, on numbers.
fn recurrence(p: &Number, q: &Number, n: u64) -> (Number, Number) {
    let o = ByteOrder::LittleEndian;
    let (mut u0, mut u1) = (Number::new_from_u8(0, o), Number::new_from_u8(1, o));
    let (mut v0, mut v1) = (Number::new_from_u8(2, o), &Number::new_from_u8(0, o) + p);
    for _ in 0..n {
        let u2 = &(p * &u1) - &(q * &u0);
        let v2 = &(p * &v1) - &(q * &v0);
        (u0, u1) = (u1, u2);
        (v0, v1) = (v1, v2);
    }
    (u0, v0)
}

/// Values of `P` and `Q`: small ones of every sign, the special cases, and long ones.
fn parameters() -> Vec<Number> {
    let o = ByteOrder::LittleEndian;
    let mut v: Vec<Number> = [-5i64, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 7, -9, 1 << 40]
        .iter()
        .map(|&x| Number::new_from_i64(x, o))
        .collect();
    v.push(Number::new_from_u128(
        0x1234_5678_9ABC_DEF0_1234_5678_9ABC_DEF1,
        o,
    ));
    v.push(-Number::new_from_u128(
        0xFFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF,
        o,
    ));
    v
}

#[test]
fn lucas_sequences_follow_their_recurrence() {
    let ps = parameters();
    let top = size(70, 12) as u64;
    for (i, p) in ps.iter().enumerate() {
        for (j, q) in ps.iter().enumerate() {
            if cfg!(miri) && (i + j) % 5 != 0 {
                continue;
            }
            let order = ORDERS[(i + j) % 2];
            // `p` in the byte order of the result, non-canonical with zero bytes above its top;
            // `q` in the other one.
            let mut magnitude = le_bytes(p);
            magnitude.extend_from_slice(&[0, 0]);
            let pp = number(&magnitude, p.negative, order);
            for n in (0..=top).step_by(if i + j > 20 { 7 } else { 1 }) {
                let (u, v) = Number::lucas_sequence(&pp, q, n);
                let (eu, ev) = recurrence(p, q, n);
                assert!(u.byte_order == order && v.byte_order == order);
                assert_eq!(u, eu, "U_{n}({p}, {q})");
                assert_eq!(v, ev, "V_{n}({p}, {q})");
                assert_shortest(&u, &|| format!("U_{n}({p}, {q})"));
                assert_shortest(&v, &|| format!("V_{n}({p}, {q})"));
            }
        }
    }
}

#[test]
fn lucas_sequences_with_closed_forms() {
    let o = ByteOrder::BigEndian;
    let int = |x: i64| Number::new_from_i64(x, o);
    // `P^2 = 4Q`: `U_n = n c^(n-1)`, `V_n = 2 c^n` for `P = 2c`.
    for c in [-3i64, -1, 1, 2, 5] {
        for n in [0u64, 1, 2, 3, 10, 33] {
            let (u, v) = Number::lucas_sequence(&int(2 * c), &int(c * c), n);
            let (eu, ev) = recurrence(&int(2 * c), &int(c * c), n);
            assert_eq!((u, v), (eu, ev), "P = {}, Q = {}, n = {n}", 2 * c, c * c);
        }
    }
    // `P = 0` and `Q = 0`, and every zero there is.
    for zero in zeros(o) {
        for n in [0u64, 1, 2, 5, 6, 17] {
            let (u, v) = Number::lucas_sequence(&zero, &int(-3), n);
            assert_eq!((u, v), recurrence(&int(0), &int(-3), n), "P = 0, n = {n}");
            let (u, v) = Number::lucas_sequence(&int(-3), &zero, n);
            assert_eq!((u, v), recurrence(&int(-3), &int(0), n), "Q = 0, n = {n}");
        }
    }
    // `P = 1, Q = -1` and `P = -1, Q = -1`: the Fibonacci and Lucas numbers, signed.
    for n in [0u64, 1, 2, 185, 186, 187, 1000] {
        let (u, v) = Number::lucas_sequence(&int(1), &int(-1), n);
        assert_eq!(u, Number::fibonacci(n, o));
        assert_eq!(v, Number::lucas(n, o));
        let (u, v) = Number::lucas_sequence(&int(-1), &int(-1), n);
        let (eu, ev) = recurrence(&int(-1), &int(-1), n);
        assert_eq!((u, v), (eu, ev), "P = -1, Q = -1, n = {n}");
    }
}

#[test]
fn long_lucas_sequences_satisfy_their_identities() {
    // `V_n^2 - D U_n^2 = 4 Q^n` and `U_(2n) = U_n V_n`, for long `n`.
    let o = ByteOrder::LittleEndian;
    for (p, q) in [(3i64, 1i64), (2, -1), (5, 3), (-7, 11), (1, 5)] {
        let (p, q) = (Number::new_from_i64(p, o), Number::new_from_i64(q, o));
        let d = &(&p * &p) - &(&q * 4u8);
        for n in [size(1000, 40) as u64, size(4321, 41) as u64] {
            let (u, v) = Number::lucas_sequence(&p, &q, n);
            let lhs = &(&v * &v) - &(&d * &(&u * &u));
            assert_eq!(lhs, &q.pow(n as u32) * 4u8, "V^2 - D U^2 for {p}, {q}, {n}");
            let (u2, _) = Number::lucas_sequence(&p, &q, 2 * n);
            assert_eq!(u2, &u * &v, "U_(2n) for {p}, {q}, {n}");
        }
    }
}

/// `(U_n, V_n) mod m` by the power of the matrix `((P, -Q), (1, 0))`, on `u128`s, for `m` below
/// `2^63`.
fn matrix_mod(p: i128, q: i128, n: u128, m: u128) -> (u128, u128) {
    if m == 1 {
        return (0, 0);
    }
    let r = |x: i128| x.rem_euclid(m as i128) as u128;
    let mul = |a: [u128; 4], b: [u128; 4]| {
        [
            (a[0] * b[0] % m + a[1] * b[2] % m) % m,
            (a[0] * b[1] % m + a[1] * b[3] % m) % m,
            (a[2] * b[0] % m + a[3] * b[2] % m) % m,
            (a[2] * b[1] % m + a[3] * b[3] % m) % m,
        ]
    };
    let mut x = [1, 0, 0, 1];
    let mut a = [r(p), r(-q), 1, 0];
    let mut e = n;
    while e > 0 {
        if e & 1 == 1 {
            x = mul(x, a);
        }
        a = mul(a, a);
        e >>= 1;
    }
    // `(U_(n+1), U_n) = M^n (1, 0)`, `V_n = 2 U_(n+1) - P U_n`.
    let (u1, u0) = (x[0], x[2]);
    (u0, (2 * u1 % m + m - r(p) * u0 % m) % m)
}

#[test]
fn lucas_sequences_modulo_small_numbers_match_matrix_powers() {
    let mut state = 0x2545_F491_4F6C_DD1D;
    let count = size(4000, 40);
    for i in 0..count {
        let bits = 1 + xorshift(&mut state) % 63;
        let mut m = u128::from(xorshift(&mut state) >> (64 - bits)).max(1);
        if i % 7 == 0 {
            // Powers of two and their multiples, for the part modulo `2^z`.
            m <<= xorshift(&mut state) % 20;
            m %= 1 << 63;
            m = m.max(1);
        }
        let small = |s: &mut u64| (xorshift(s) % 11) as i128 - 5;
        let p = if i % 3 == 0 {
            xorshift(&mut state) as i64 as i128
        } else {
            small(&mut state)
        };
        let q = if i % 4 == 0 {
            xorshift(&mut state) as i64 as i128
        } else {
            small(&mut state)
        };
        let n = match i % 5 {
            0 => u128::from(xorshift(&mut state) % 8),
            1 => u128::from(xorshift(&mut state)) | u128::from(xorshift(&mut state)) << 64,
            _ => u128::from(xorshift(&mut state) >> (xorshift(&mut state) % 64)),
        };
        let (eu, ev) = matrix_mod(p, q, n, m);
        let order = ORDERS[i % 2];
        let (u, v) = Number::lucas_sequence_mod(
            &Number::new_from_i128(p, ORDERS[(i / 2) % 2]),
            &Number::new_from_i128(q, order),
            &Number::new_from_u128(n, ORDERS[(i / 3) % 2]),
            &Number::new_from_u128(m, order),
        );
        assert!(u.byte_order == order && v.byte_order == order);
        assert_eq!(
            (u128::try_from(&u).unwrap(), u128::try_from(&v).unwrap()),
            (eu, ev),
            "P = {p}, Q = {q}, n = {n}, m = {m}"
        );
    }
}

/// `x mod m`, Euclid's.
fn modulo(x: &Number, m: &Number) -> Number {
    x.rem_euclid(m)
}

#[test]
fn lucas_sequences_modulo_long_numbers_match_the_exact_ones() {
    // Moduli of every shape: odd ones in registers and in limbs (from 8 limbs), even ones whose
    // power of two is in registers or in limbs (beyond 8 limbs), powers of two, and 1 and 2.
    let mut state = 0x0123_4567_89AB_CDEF;
    let o = ByteOrder::LittleEndian;
    let mut moduli = vec![Number::new_from_u8(1, o), Number::new_from_u8(2, o)];
    for limbs in [1usize, 2, 3, 5, 7, 8, 9, 12] {
        let mut m = Number::new(random_magnitude(&mut state, 8 * limbs), o);
        m.value[0] |= 1;
        moduli.push(m.clone());
        moduli.push(&m << 70usize);
        moduli.push(&m << 600usize);
        moduli.push(&Number::new_from_u8(1, o) << (64 * limbs - 1));
    }
    let ps = parameters();
    let n = size(150, 9) as u64;
    for (i, m) in moduli.iter().enumerate() {
        if cfg!(miri) && i % 4 != 1 {
            continue;
        }
        for (j, p) in ps.iter().enumerate().step_by(3) {
            let q = &ps[(j * 7 + i) % ps.len()];
            let (eu, ev) = Number::lucas_sequence(p, q, n);
            let order = ORDERS[(i + j) % 2];
            let mut mm = m.clone();
            if order == ByteOrder::BigEndian {
                mm.value.reverse();
                mm.byte_order = order;
            }
            let (u, v) = Number::lucas_sequence_mod(p, q, &Number::new_from_u64(n, o), &mm);
            assert!(u.byte_order == order && v.byte_order == order);
            assert_eq!(u, modulo(&eu, m), "U_{n}({p}, {q}) mod {m}");
            assert_eq!(v, modulo(&ev, m), "V_{n}({p}, {q}) mod {m}");
        }
    }
}

#[test]
fn lucas_sequences_modulo_long_numbers_with_long_indices() {
    // A long index, against the composition of the sequence: `U_(a+b) = U_a U_(b+1) - Q U_(a-1)
    // U_b`, from pieces of the index taken by the function itself, and `V_(2n) = V_n^2 - 2Q^n`.
    let mut state = 0xDEAD_BEEF_0BAD_F00D;
    let o = ByteOrder::BigEndian;
    let count = size(12, 2);
    for i in 0..count {
        let limbs = [1usize, 2, 4, 7, 8, 10][i % 6];
        let mut m = Number::new(random_magnitude(&mut state, 8 * limbs), o);
        let last = m.value.len() - 1;
        m.value[last] |= 1;
        if i % 3 == 2 {
            m = &m << 130usize;
        }
        let p = Number::new(random_magnitude(&mut state, 8 * limbs + 3), o);
        let q = -Number::new(random_magnitude(&mut state, 5), o);
        let n = Number::new(random_magnitude(&mut state, 24), o);
        let (u, v) = Number::lucas_sequence_mod(&p, &q, &n, &m);
        let (u2, v2) = Number::lucas_sequence_mod(&p, &q, &(&n << 1usize), &m);
        let qn = q.pow_mod(&n, &m);
        assert_eq!(u2, modulo(&(&u * &v), &m), "U_(2n) = U_n V_n");
        assert_eq!(v2, modulo(&(&(&v * &v) - &(&qn << 1usize)), &m), "V_(2n)");
        let (u1, _) = Number::lucas_sequence_mod(&p, &q, &(&n + 1u8), &m);
        let (um, _) = Number::lucas_sequence_mod(&p, &q, &(&n - 1u8), &m);
        let (ua, _) = Number::lucas_sequence_mod(&p, &q, &(&(&n << 1usize) + 1u8), &m);
        // `U_(2n+1) = U_(n+1)^2 - Q U_n^2`, and `U_(n+1) = P U_n - Q U_(n-1)`.
        assert_eq!(
            ua,
            modulo(&(&(&u1 * &u1) - &(&q * &(&u * &u))), &m),
            "U_(2n+1)"
        );
        assert_eq!(
            u1,
            modulo(&(&(&p * &u) - &(&q * &um)), &m),
            "the recurrence"
        );
    }
}

#[test]
fn lucas_sequences_modulo_one_and_at_zero() {
    let o = ByteOrder::LittleEndian;
    let int = |x: i64| Number::new_from_i64(x, o);
    for m in [1i64, 2, 3, 4, 1 << 40, 9] {
        let (u, v) = Number::lucas_sequence_mod(&int(5), &int(7), &int(0), &int(m));
        assert_eq!((u, v), (int(0), int(2 % m)), "n = 0 modulo {m}");
    }
    for zero in zeros(o) {
        let (u, v) = Number::lucas_sequence_mod(&int(5), &int(7), &zero, &int(9));
        assert_eq!((u, v), (int(0), int(2)));
        let (u, v) = Number::lucas_sequence_mod(&zero, &zero, &int(1), &int(9));
        assert_eq!((u, v), (int(1), int(0)));
    }
    let (u, v) = Number::lucas_sequence_mod(&int(3), &int(1), &int(12345), &int(1));
    assert_eq!((u, v), (int(0), int(0)));
}

#[test]
#[should_panic(expected = "modulus")]
fn lucas_sequences_modulo_zero_panic() {
    let o = ByteOrder::LittleEndian;
    let one = Number::new_from_u8(1, o);
    let _ = Number::lucas_sequence_mod(&one, &one, &one, &Number::new(vec![0, 0], o));
}

#[test]
#[should_panic(expected = "modulus")]
fn lucas_sequences_modulo_a_negative_number_panic() {
    let o = ByteOrder::LittleEndian;
    let one = Number::new_from_u8(1, o);
    let _ = Number::lucas_sequence_mod(&one, &one, &one, &Number::new_from_i32(-7, o));
}

#[test]
#[should_panic(expected = "index")]
fn lucas_sequences_with_a_negative_index_panic() {
    let o = ByteOrder::LittleEndian;
    let one = Number::new_from_u8(1, o);
    let _ = Number::lucas_sequence_mod(&one, &one, &Number::new_from_i32(-3, o), &one);
}

// Primorials

#[test]
fn primorials_are_products_of_primes() {
    let top = size(3000, 260);
    let primes = primes_to(top);
    for order in ORDERS {
        let mut x = vec![1u64];
        let mut next = 0;
        for n in 0..=top as u64 {
            if next < primes.len() && primes[next] == n {
                times(&mut x, n);
                next += 1;
            }
            if (n < 200 || n % 97 == 0 || n == top as u64) && (!cfg!(miri) || n % 17 == 0 || n > 98)
            {
                let p = Number::primorial(n, order);
                assert_limbs(&p, &x, order, &|| format!("{n}# in {order:?}"));
            }
        }
    }
}

#[test]
fn primorials_match_pari() {
    for &(name, n, digits, r1, r2) in &PARI {
        if name == "pr" && affordable(n) {
            for order in ORDERS {
                let x = Number::primorial(n, order);
                assert!(x.byte_order == order);
                assert_pari(&x, "primorial", n, digits, r1, r2);
            }
        }
    }
}

#[test]
fn primorials_across_the_tables() {
    // `n# / (n - 1)# = n` exactly when `n` is prime, around the ends of the tables and of the
    // groups of the small primes.
    let ns: &[u64] = if cfg!(miri) {
        &[101, 102, 103]
    } else {
        &[101, 102, 103, 8179, 8191, 8192, 8209, 8210, 65_537, 65_538]
    };
    let o = ByteOrder::LittleEndian;
    for &n in ns {
        let a = Number::primorial(n, o);
        let b = Number::primorial(n - 1, o);
        let prime = Number::new_from_u64(n, o).is_probable_prime();
        let expected = if prime {
            Number::new_from_u64(n, o)
        } else {
            Number::new_from_u8(1, o)
        };
        assert_eq!(&a / &b, expected, "{n}# / {}#", n - 1);
        assert!((&a % &b).is_zero());
    }
}

// Partitions

/// `p(0)` to `p(top)` by Euler's pentagonal recurrence, on limbs.
fn partition_table(top: usize) -> Vec<Vec<u64>> {
    let mut p: Vec<Vec<u64>> = vec![vec![1]];
    for n in 1..=top {
        let (mut plus, mut minus) = (vec![0u64], vec![0u64]);
        let mut k = 1;
        loop {
            let g = k * (3 * k - 1) / 2;
            if g > n {
                break;
            }
            let h = k * (3 * k + 1) / 2;
            let mut t = p[n - g].clone();
            if h <= n {
                t = add(&t, &p[n - h]);
            }
            if k % 2 == 1 {
                plus = add(&plus, &t);
            } else {
                minus = add(&minus, &t);
            }
            k += 1;
        }
        p.push(sub(&plus, &minus));
    }
    p
}

#[test]
fn partitions_follow_the_pentagonal_recurrence() {
    // The table, the series from 417, and the paths of its terms: doubles, limbs on the stack.
    let top = size(2600, 420);
    let table = partition_table(top);
    let step = if cfg!(miri) { 401 } else { 1 };
    for (n, expected) in table.iter().enumerate().step_by(step) {
        let order = ORDERS[n % 2];
        let p = Number::partitions(n as u64, order);
        assert_limbs(&p, expected, order, &|| format!("p({n})"));
    }
    if cfg!(miri) {
        // One value of the series.
        let p = Number::partitions(418, ByteOrder::BigEndian);
        assert_limbs(&p, &table[418], ByteOrder::BigEndian, &|| "p(418)".into());
    }
}

#[test]
fn partitions_match_pari() {
    // Up to `n = 20000001`, where the first terms take the numbers' arithmetic, and `π` takes
    // Chudnovsky's series past its table.
    for &(name, n, digits, r1, r2) in &PARI {
        if name == "p" && (!cfg!(miri) || n == 417) {
            for order in ORDERS {
                let x = Number::partitions(n, order);
                assert!(x.byte_order == order && !x.negative);
                assert_pari(&x, "p", n, digits, r1, r2);
            }
        }
    }
}

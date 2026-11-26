//! Tests of the Chinese remainder theorem, multiplicative orders, primitive roots and discrete
//! logarithms: against brute force for small moduli, against PARI for long ones, and on both byte
//! orders and non-canonical numbers.

use a::math::numbers::{ByteOrder, Factorization, Number, dlog};

fn n(x: i64) -> Number {
    Number::new_from_i64(x, ByteOrder::BigEndian)
}

fn int(x: &Number) -> u128 {
    u128::try_from(x).unwrap()
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// The factorisation of `m >= 1` by trial division, in the byte order `order`: the tests' own,
/// quicker under Miri than [`Number::factor`].
fn factorise(mut m: u64, order: ByteOrder) -> Factorization {
    let mut f = Vec::new();
    let mut p = 2;
    while p * p <= m {
        let mut e = 0;
        while m.is_multiple_of(p) {
            m /= p;
            e += 1;
        }
        if e > 0 {
            f.push((Number::new_from_u64(p, order), e));
        }
        p += 1;
    }
    if m > 1 {
        f.push((Number::new_from_u64(m, order), 1));
    }
    Factorization::new(f, order)
}

/// The same number in the other byte order, with zero bytes above its top.
fn other(x: &Number) -> Number {
    let mut v = x.value.clone();
    v.reverse();
    let order = match x.byte_order {
        ByteOrder::BigEndian => ByteOrder::LittleEndian,
        ByteOrder::LittleEndian => ByteOrder::BigEndian,
    };
    match order {
        ByteOrder::BigEndian => v.insert(0, 0),
        ByteOrder::LittleEndian => v.push(0),
    }
    Number {
        value: v,
        negative: x.negative,
        byte_order: order,
    }
}

fn number(s: &str) -> Number {
    Number::from_str_radix(s, 10, ByteOrder::BigEndian).unwrap()
}

// The Chinese remainder theorem

#[test]
fn crt_against_brute_force() {
    let limit = if cfg!(miri) { 5 } else { 25 };
    for m1 in 1..limit {
        for m2 in 1..limit {
            let l = m1 / gcd(m1, m2) * m2;
            for r1 in 0..m1 {
                for r2 in 0..m2 {
                    let want = (0..l).find(|x| x % m1 == r1 && x % m2 == r2);
                    let got =
                        Number::crt(&[(n(r1 as i64), n(m1 as i64)), (n(r2 as i64), n(m2 as i64))]);
                    assert_eq!(
                        got,
                        want.map(|x| (n(x as i64), n(l as i64))),
                        "{r1} mod {m1}, {r2} mod {m2}"
                    );
                }
            }
        }
    }
    // Three at a time, with residues of any sign and size.
    let ms: &[i64] = if cfg!(miri) {
        &[4, 6, 9]
    } else {
        &[4, 6, 9, 10, 15]
    };
    let rs = if cfg!(miri) { -1..1 } else { -3..3 };
    for &a in ms {
        for &b in ms {
            for &c in ms {
                for r in rs.clone() {
                    let rs = [r * 7 + 1, r * 11 - 5, r * 1_000_000_007 + 2];
                    let l = [a, b, c]
                        .iter()
                        .fold(1i64, |l, &m| l / gcd(l as u64, m as u64) as i64 * m);
                    let want = (0..l).find(|x| {
                        [a, b, c]
                            .iter()
                            .zip(&rs)
                            .all(|(&m, &r)| (x - r).rem_euclid(m) == 0)
                    });
                    let got = Number::crt(&[(n(rs[0]), n(a)), (n(rs[1]), n(b)), (n(rs[2]), n(c))]);
                    assert_eq!(got, want.map(|x| (n(x), n(l))));
                }
            }
        }
    }
    assert_eq!(Number::crt(&[]), Some((n(0), n(1))));
}

#[test]
fn crt_of_long_moduli() {
    let p = (n(1) << 127usize) - 1u8;
    let q = (n(1) << 89usize) - 1u8;
    let r = n(1_000_000_007);
    // Coprime: the solution is the one modulo the product.
    let x = (n(1) << 200usize) + 12345u32;
    let c = [
        (x.rem_euclid(&p), p.clone()),
        (x.rem_euclid(&q), q.clone()),
        (x.rem_euclid(&r), r.clone()),
    ];
    let l = &(&p * &q) * &r;
    assert_eq!(Number::crt(&c), Some((x.rem_euclid(&l), l.clone())));
    // Not coprime: p q and q r share q.
    let c = [
        (x.rem_euclid(&(&p * &q)), &p * &q),
        (x.rem_euclid(&(&q * &r)), &q * &r),
    ];
    assert_eq!(Number::crt(&c), Some((x.rem_euclid(&l), l.clone())));
    // A contradiction modulo q.
    let c = [
        (x.rem_euclid(&(&p * &q)), &p * &q),
        ((&x + 1u8).rem_euclid(&(&q * &r)), &q * &r),
    ];
    assert_eq!(Number::crt(&c), None);
    // In the byte order of the first residue, whatever the moduli's.
    let c = [
        (Number::new_from_u64(3, ByteOrder::LittleEndian), other(&p)),
        (n(5), q.clone()),
    ];
    let (x, l) = Number::crt(&c).unwrap();
    assert_eq!(x.byte_order, ByteOrder::LittleEndian);
    assert_eq!(l.byte_order, ByteOrder::LittleEndian);
    assert_eq!(x.rem_euclid(&p), n(3));
    assert_eq!(x.rem_euclid(&q), n(5));
}

#[test]
fn crt_of_many_words() {
    // More than a group of 64: the groups joined up a tree.
    let count = if cfg!(miri) { 20 } else { 300 };
    let moduli: Vec<u64> = (0..count).map(|i| (1u64 << 62) + 2 * i + 1).collect();
    let x = (n(1) << 5000usize) - 77u8;
    let c: Vec<(Number, Number)> = moduli
        .iter()
        .map(|&m| {
            let m = Number::new_from_u64(m, ByteOrder::BigEndian);
            (x.rem_euclid(&m), m)
        })
        .collect();
    let (y, l) = Number::crt(&c).unwrap();
    let mut want_l = n(1);
    for (_, m) in &c {
        want_l = want_l.lcm(m);
    }
    assert_eq!(l, want_l);
    assert!(y < l);
    for (r, m) in &c {
        assert_eq!(y.rem_euclid(m), *r);
    }
    // A contradiction in the last group.
    let mut c = c;
    let last = c.len() - 1;
    let m = c[last].1.clone();
    c.push(((&c[last].0 + 1u8).rem_euclid(&m), m.clone() * 3u8));
    assert_eq!(Number::crt(&c), None);
}

#[test]
fn crt_of_many_long_moduli() {
    // Eight and more long moduli take the product and remainder trees when they are coprime, and
    // the pairwise joins when they are not.
    let moduli: Vec<Number> = if cfg!(miri) {
        // Seventh powers of primes above 577, each beyond 2^64: no search for primes.
        [601, 607, 613, 617, 619, 631, 641, 643]
            .iter()
            .map(|&p| n(p).pow(7))
            .collect()
    } else {
        let mut moduli = Vec::new();
        let mut p = (n(1) << 100usize) + 1u8;
        for _ in 0..20 {
            p = p.next_prime();
            moduli.push(p.clone());
        }
        moduli
    };
    let x = (n(1) << 1900usize) + 12345u32;
    let c: Vec<(Number, Number)> = moduli
        .iter()
        .map(|m| (x.rem_euclid(m), m.clone()))
        .collect();
    let l: Number = moduli.iter().cloned().product();
    assert_eq!(Number::crt(&c), Some((x.rem_euclid(&l), l.clone())));
    // Not coprime: the first modulus doubled shares it with the second, which is twice it too.
    let mut c2 = c.clone();
    c2[1] = (x.rem_euclid(&(&moduli[0] * 2u8)), &moduli[0] * 2u8);
    let (y, l2) = Number::crt(&c2).unwrap();
    for (r, m) in &c2 {
        assert_eq!(y.rem_euclid(m), *r);
    }
    assert_eq!(&l2 * &moduli[1], &l * 2u8);
    // And contradicting each other.
    c2[1].0 += 1u8;
    assert_eq!(Number::crt(&c2), None);
}

#[test]
#[should_panic(expected = "the modulus must be positive")]
fn crt_refuses_a_zero_modulus() {
    Number::crt(&[(n(1), n(3)), (n(1), n(0))]);
}

// Multiplicative orders

fn order_reference(a: u64, m: u64) -> Option<u64> {
    if m == 1 {
        return Some(1);
    }
    if gcd(a % m, m) != 1 {
        return None;
    }
    let (mut x, mut k) = (a % m, 1);
    while x != 1 {
        x = x * a % m;
        k += 1;
    }
    Some(k)
}

#[test]
fn multiplicative_order_against_brute_force() {
    let limit = if cfg!(miri) { 12 } else { 400 };
    for m in 1..limit {
        for a in 0..m {
            let want = order_reference(a, m);
            let got = n(a as i64).multiplicative_order(&n(m as i64));
            assert_eq!(got.map(|k| int(&k) as u64), want, "order of {a} modulo {m}");
        }
    }
    // Negative and long operands.
    assert_eq!(n(-1).multiplicative_order(&n(7)), Some(n(2)));
    let big = (n(1) << 300usize) + 3u8;
    assert_eq!(
        big.multiplicative_order(&n(101)),
        n(int(&big.rem_euclid(&n(101))) as i64).multiplicative_order(&n(101))
    );
    assert_eq!(n(5).multiplicative_order(&n(1)), Some(n(1)));
}

/// Long moduli against PARI 2.19's `znorder`: `[a, m, order]` for primes `p = c q + 1`, `q`
/// prime, `p 1000003 3^5` and `4 p r^2` (`r` of 22 bits).
#[rustfmt::skip]
const ORDERS: &[(&str, &str, &str)] = &[
    ("7", "86881334562544349363", "86881334562544349362"),
    ("7", "21112227635191172990039685627", "1172900362390381905084432774"),
    ("7", "12460514864762198936102411360881868", "1557564097976772931463375340057586"),
    ("35", "4891180788157036116351430818187", "4891180788157036116351430818186"),
    ("35", "1188560497192954342752726509012507458323", "7336785855777918645635495281571454558"),
    ("35", "429294680083510604418101256555066514892584588", "53661823554660011776417736397253856737816446"),
    ("47", "6885336320492180284431883314047760518011", "3442668160246090142215941657023880259005"),
    ("47", "1673141745289777447916374996156541746694090630019", "30984075410241695709566034800164748760889662090"),
    ("47", "924944899955477457317645519756188589303439927126984364", "115618092543625309301134187495529336486060301061798610"),
    ("53", "11578432003366310894744587053915388263101185281065948198676637", "11578432003366310894744587053915388263101185281065948198676636"),
    ("53", "2813567417494944001463576922905401652251631824063095309354659626268373", "5789227580115158813683188271544748046938855741718255165286516676636"),
    ("53", "892248496828895399136046243839765645516408057437550889453213802880125799892", "111531036693428741472254445676420905241612067236280777353027408610931396036"),
    ("7", "2748905717966100350826927177795519467879436111848843641995081568831931812156381696929221567", "2748905717966100350826927177795519467879436111848843641995081568831931812156381696929221566"),
    ("7", "667986093418030782538099057034223843628395059288194542811819835640623108832291814356057902183522343", "37110301412996739820872989227273313295398020254734409085711935046433437926270081130850308229982282"),
    ("7", "207625148315278815863929364788372061182151876548610709229713454332424892943448984584097428517933056659612", "25953137566858275199467989206355588060677143427723679285563827563376394394439626903123431064635278185698"),
];

#[test]
fn multiplicative_order_of_long_moduli() {
    // The long moduli are factored by `factor`, too slow under Miri.
    let orders = if cfg!(miri) { &ORDERS[..0] } else { ORDERS };
    for &(a, m, o) in orders {
        for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
            let a = Number::from_str_radix(a, 10, order).unwrap();
            let m = number(m);
            let got = a.multiplicative_order(&m).unwrap();
            assert_eq!(got, number(o), "order of {a} modulo {m}");
            assert_eq!(got.byte_order, order);
            assert_eq!(other(&a).multiplicative_order(&other(&m)), Some(number(o)));
        }
    }
    let m = n(3).pow(50) * 4u8;
    assert_eq!(n(3).multiplicative_order(&m), None);
}

// Primitive roots

fn primitive_root_reference(m: u64) -> Option<u64> {
    if m == 1 {
        return Some(0);
    }
    let phi = (1..=m).filter(|&a| gcd(a, m) == 1).count() as u64;
    (1..m).find(|&g| order_reference(g, m) == Some(phi))
}

#[test]
fn primitive_root_against_brute_force() {
    let limit = if cfg!(miri) { 16 } else { 700 };
    for m in 1..limit {
        let want = primitive_root_reference(m);
        let got = Number::primitive_root(&n(m as i64));
        assert_eq!(
            got.map(|g| int(&g) as u64),
            want,
            "primitive root modulo {m}"
        );
    }
}

/// The smallest primitive roots of primes `p = c q + 1`, `q` prime, against PARI's `znprimroot`.
#[rustfmt::skip]
const ROOTS: &[(&str, u64)] = &[
    ("20655169858431258419", 2),
    ("40763686284758741159", 7),
    ("4627868795749960546521970765123", 2),
    ("7164988275408638285791503165735917807191", 6),
    ("141861358291505067031667616981982370300884789912439892809638318630420342296287", 5),
    ("60204008951596638123946536029145751464243703880755527534072370641565715915462057138877544624855199966981338223432895095555688623693217864073318382184724987", 2),
];

#[test]
fn primitive_root_of_long_moduli() {
    let roots = if cfg!(miri) { &ROOTS[..0] } else { ROOTS };
    for &(p, g) in roots {
        for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
            let p = Number::from_str_radix(p, 10, order).unwrap();
            let r = Number::primitive_root(&p).unwrap();
            assert_eq!(r, n(g as i64), "primitive root modulo {p}");
            assert_eq!(r.byte_order, order);
        }
    }
    // Modulo 2 p^k: the smallest odd root.
    if cfg!(miri) {
        return;
    }
    let p = number("1000000000000000003");
    let r = Number::primitive_root(&(&p * 2u8)).unwrap();
    assert_eq!(r, n(5));
    assert_eq!(Number::primitive_root(&(&p * 4u8)), None);
    assert_eq!(Number::primitive_root(&(&p * 3u8)), None);
}

// Discrete logarithms

/// The first `k` with `g^k = h` for each `h < m`: the powers of `g` repeat after at most `m`
/// steps past a start of at most `log2 m`.
fn logs_reference(g: u64, m: u64) -> Vec<Option<u64>> {
    let mut first = vec![None; m as usize];
    let mut x = 1 % m;
    for k in 0..2 * m + 64 {
        first[x as usize].get_or_insert(k);
        x = x * g % m;
    }
    first
}

#[test]
fn discrete_log_against_brute_force() {
    let limit = if cfg!(miri) { 8 } else { 130 };
    for m in 1..limit {
        for g in 0..m {
            let logs = logs_reference(g, m);
            for h in 0..m {
                let want = logs[h as usize];
                let got = n(h as i64).discrete_log(&n(g as i64), &n(m as i64));
                assert_eq!(
                    got.map(|k| int(&k) as u64),
                    want,
                    "log of {h} to {g} modulo {m}"
                );
            }
        }
    }
}

#[test]
fn discrete_log_of_long_moduli() {
    // Every modulus here beyond 2^64 is factored, which takes too long under Miri; the paths are
    // covered there by the other tests.
    if cfg!(miri) {
        return;
    }
    // p - 1 = 2 * 3 * 7 * ... smooth: Pohlig and Hellman with small primes.
    // The largest prime below 2^64: p - 1 = 2^2 11 137 547 5594472617641, the last by rho (too
    // long for Miri).
    if !cfg!(miri) {
        let p = number("18446744073709551557");
        let g = Number::primitive_root(&p).unwrap();
        let k = number("12345678901234567");
        let h = g.pow_mod(&k, &p);
        assert_eq!(h.discrete_log(&g, &p), Some(k.clone()));
    }
    // A prime power and a product with a power of two, beyond 2^64.
    let q = n(1_000_003);
    let m = q.pow(4) * (n(1) << 20usize);
    let g = n(3);
    let k = n(98_765_432_101);
    let h = g.pow_mod(&k, &m);
    // The logarithm by factoring the modulus, beyond Miri's patience.
    let got = if cfg!(miri) {
        k.clone()
    } else {
        h.discrete_log(&g, &m).unwrap()
    };
    assert_eq!(g.pow_mod(&got, &m), h);
    assert!(got <= k);
    // The smallest is below the order.
    let o = g.multiplicative_order(&m).unwrap();
    assert!(got < o);
    assert_eq!(got, k.rem_euclid(&o));
    // In the byte order of `self`.
    let r = other(&h).discrete_log(&other(&g), &other(&m)).unwrap();
    assert_eq!(r, got);
    assert_eq!(r.byte_order, ByteOrder::LittleEndian);
    // A base not prime to a long modulus: 6^k modulo 2^70 3^50 q.
    let m = (n(1) << 70usize) * n(3).pow(50) * &q;
    let g = n(6);
    let ks: &[u32] = if cfg!(miri) {
        &[0, 69, 71]
    } else {
        &[0, 1, 3, 49, 50, 69, 70, 71, 1000]
    };
    for &k in ks {
        let h = g.pow_mod(&n(i64::from(k)), &m);
        let got = h.discrete_log(&g, &m).unwrap();
        assert_eq!(g.pow_mod(&got, &m), h, "6^{k}");
        if k < 70 {
            assert_eq!(got, n(i64::from(k)));
        }
    }
    assert_eq!(n(2).discrete_log(&g, &m), None);
}

#[test]
fn discrete_log_with_a_large_prime_order() {
    // p = 2 q + 1 with q prime of 40 bits: the logarithm of order q by Pollard's rho.
    if cfg!(miri) {
        return;
    }
    let q = number("738539518121");
    let p = number("1477079036243");
    let g = n(4);
    for k in [n(1), n(123_456_789_012), &q - 1u8] {
        let h = g.pow_mod(&k, &p);
        assert_eq!(h.discrete_log(&g, &p), Some(k.clone()));
        assert_eq!(dlog::rho(&h, &g, &p, &q), Some(k.clone()));
        assert_eq!(dlog::bsgs(&h, &g, &p, &q), Some(k.clone()));
        let f = Factorization::new(vec![(q.clone(), 1)], ByteOrder::BigEndian);
        assert_eq!(dlog::pohlig_hellman(&h, &g, &p, &f), Some(k.clone()));
    }
    // p = 3 modulo 4: -1 is not a square, so not a power of 4.
    let minus = &p - 1u8;
    assert_eq!(minus.discrete_log(&g, &p), None);
    assert_eq!(dlog::rho(&minus, &g, &p, &q), None);
    assert_eq!(dlog::bsgs(&minus, &g, &p, &q), None);
}

#[test]
fn dlog_methods_against_brute_force() {
    let limit = if cfg!(miri) { 5 } else { 50 };
    for m in 1..limit {
        for g in 0..m {
            let o = order_reference(g, m);
            let logs = logs_reference(g, m);
            let fs: Vec<Factorization> = o
                .map(|o| {
                    [1u64, 2, 6]
                        .iter()
                        .map(|&k| factorise(o * k, ByteOrder::BigEndian))
                        .collect()
                })
                .unwrap_or_default();
            for h in 0..m {
                // bsgs: the smallest k below the bound.
                for bound in [1u64, 2, 5, m, 2 * m] {
                    let want = logs[h as usize].filter(|&k| k < bound);
                    let got =
                        dlog::bsgs(&n(h as i64), &n(g as i64), &n(m as i64), &n(bound as i64));
                    assert_eq!(
                        got.map(|k| int(&k) as u64),
                        want,
                        "bsgs {h} {g} {m} below {bound}"
                    );
                }
                let Some(o) = o else { continue };
                // Pohlig–Hellman with the order, and with a multiple of it.
                let want = logs[h as usize].filter(|&k| k < o);
                for (f, mult) in fs.iter().zip([1u64, 2, 6]) {
                    let got = dlog::pohlig_hellman(&n(h as i64), &n(g as i64), &n(m as i64), f);
                    assert_eq!(
                        got.map(|k| int(&k) as u64),
                        want,
                        "ph {h} {g} {m} order {o} * {mult}"
                    );
                }
                // rho for prime orders.
                if o > 1 && (2..o).all(|d| o % d != 0) {
                    let got = dlog::rho(&n(h as i64), &n(g as i64), &n(m as i64), &n(o as i64));
                    assert_eq!(
                        got.map(|k| int(&k) as u64),
                        want,
                        "rho {h} {g} {m} order {o}"
                    );
                }
            }
        }
    }
}

#[test]
fn dlog_methods_modulo_even_and_long_numbers() {
    // Modulo 2^100 (2^70 under Miri): the units are ±5^k.
    let e: usize = if cfg!(miri) { 70 } else { 100 };
    let m = n(1) << e;
    let g = n(5);
    let k = (n(1) << (e - 10)) + 12345u32;
    let h = g.pow_mod(&k, &m);
    let order = Factorization::new(vec![(n(2), e as u32 - 2)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&h, &g, &m, &order), Some(k.clone()));
    if !cfg!(miri) {
        assert_eq!(h.discrete_log(&g, &m), Some(k.clone()));
    }
    assert_eq!(dlog::pohlig_hellman(&n(-1), &g, &m, &order), None);
    // Modulo 3 * 2^200 (3 * 2^70 under Miri), by divisions.
    let e2: usize = if cfg!(miri) { 70 } else { 200 };
    let m = n(3) << e2;
    let h = g.pow_mod(&k, &m);
    // The logarithm by factoring the modulus, beyond Miri's patience.
    let got = if cfg!(miri) {
        k.clone()
    } else {
        h.discrete_log(&g, &m).unwrap()
    };
    assert_eq!(g.pow_mod(&got, &m), h);
    let order = Factorization::new(vec![(n(2), e2 as u32 - 1)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&h, &g, &m, &order), Some(got));
    // An order that is not a multiple of the base's.
    let small = Factorization::new(vec![(n(2), 3)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&h, &g, &m, &small), None);
    // Long odd moduli of 2 to 9 limbs, in registers and on limbs.
    if cfg!(miri) {
        return;
    }
    for limbs in [2usize, 3, 5, 7, 8, 9] {
        let p = ((n(1) << (64 * limbs - 3)) + 1u8).next_prime();
        let g = n(3);
        let k = (n(1) << 50usize) + 7u8;
        let h = g.pow_mod(&k, &p);
        // The order of the subgroup: powers of 2 of p - 1, which is 2^k times an odd number.
        let p1 = &p - 1u8;
        let z = p1.trailing_zeros().unwrap();
        let odd = &p1 >> (z as usize);
        let g2 = g.pow_mod(&odd, &p);
        let h2 = h.pow_mod(&odd, &p);
        let order = Factorization::new(vec![(n(2), z as u32)], ByteOrder::BigEndian);
        let x = dlog::pohlig_hellman(&h2, &g2, &p, &order).unwrap();
        assert_eq!(g2.pow_mod(&x, &p), h2, "{limbs} limbs");
        let small = n(1000);
        assert_eq!(
            dlog::bsgs(&g.pow_mod(&small, &p), &g, &p, &n(2000)),
            Some(small.clone())
        );
    }
}

#[test]
#[should_panic(expected = "the order must be positive")]
fn bsgs_refuses_a_zero_order() {
    dlog::bsgs(&n(1), &n(2), &n(7), &n(0));
}

#[test]
#[should_panic(expected = "the modulus must be positive")]
fn discrete_log_refuses_a_negative_modulus() {
    n(1).discrete_log(&n(2), &n(-7));
}

#[test]
fn rings_of_every_length() {
    // The residues in registers (one to seven limbs) and on limbs (eight and more), by the
    // baby steps and giant steps and Pohlig and Hellman's method with small logarithms, which keep
    // this short enough for Miri: odd moduli `2^(64 l) - 3` (not prime: the methods take any).
    for limbs in [1usize, 2, 3, 4, 7, 8, 9] {
        let m = (n(1) << (64 * limbs)) - 3u8;
        for order in [ByteOrder::BigEndian, ByteOrder::LittleEndian] {
            let g = Number::new_from_u64(3, order);
            let k = n(1000);
            let h = g.pow_mod(&k, &m);
            assert_eq!(
                dlog::bsgs(&h, &g, &m, &n(2000)),
                Some(k.clone()),
                "{limbs} limbs"
            );
            assert_eq!(dlog::bsgs(&h, &g, &m, &n(1000)), None, "{limbs} limbs");
            // -1 has order 2: Pohlig and Hellman's method with that order.
            let minus = &m - 1u8;
            let two = Factorization::new(vec![(n(2), 1)], ByteOrder::BigEndian);
            assert_eq!(dlog::pohlig_hellman(&minus, &minus, &m, &two), Some(n(1)));
            assert_eq!(dlog::pohlig_hellman(&n(1), &minus, &m, &two), Some(n(0)));
            assert_eq!(dlog::pohlig_hellman(&g, &minus, &m, &two), None);
        }
    }
}

#[test]
fn rho_modulo_an_even_number() {
    // Modulo 8 p, p = 2q + 1, the elements of odd order q are 1 modulo 8: their residues by
    // divisions are all odd, which once made every walk of rho miss its distinguished points.
    let (q, p) = if cfg!(miri) {
        (n(131_321), n(262_643))
    } else {
        (n(16_777_259), n(33_554_519))
    };
    let m = &p * 8u8;
    // 4 has order q modulo p; the element that is 4 modulo p and 1 modulo 8.
    let (g, _) = Number::crt(&[(n(4), p.clone()), (n(1), n(8))]).unwrap();
    assert_eq!(g.pow_mod(&q, &m), n(1));
    let k = n(123_457);
    let h = g.pow_mod(&k, &m);
    assert_eq!(dlog::rho(&h, &g, &m, &q), Some(k.clone()));
    let f = Factorization::new(vec![(q.clone(), 1)], ByteOrder::BigEndian);
    assert_eq!(dlog::pohlig_hellman(&h, &g, &m, &f), Some(k.clone()));
    assert_eq!(h.discrete_log(&g, &m), Some(k));
    // -1 modulo p is not a power of g.
    let (minus, _) = Number::crt(&[(&p - 1u8, p.clone()), (n(1), n(8))]).unwrap();
    assert_eq!(dlog::rho(&minus, &g, &m, &q), None);
}

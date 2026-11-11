//! Tests of the self-initialising quadratic sieve, `factor::siqs`.

use a::math::numbers::{ByteOrder, Number, factor};

use super::common::*;

// References

/// Balanced semiprimes `p·q` of 20 to 46 digits and their smaller factor, from PARI/GP 2.19: random
/// primes of half the digits each (`setrand(2026)`), the product checked by `factor`.
const SEMIPRIMES: [(&str, &str); 27] = [
    ("14258663670606281711", "1929784301"),
    ("362955568356999055297", "10222407617"),
    ("2093340923969201020391", "42264014413"),
    ("52972402006990907522701", "65398629929"),
    ("561850001925133359385373", "651205793629"),
    ("6638649357291791773524023", "785577583841"),
    ("29339088952930076888055781", "4932861812903"),
    ("121285964984244741748280959", "1376516148269"),
    ("703233672826377723171545039", "25016612102399"),
    ("27013171124355431393576680241", "79387122917407"),
    ("528795441873933006516143714059", "621214074526261"),
    ("2128695321195806714852970469057", "454682486272741"),
    ("36094853308013450696374134718241", "3341478150022373"),
    ("74610044443650128894335858884229", "4758379221444547"),
    ("6279961857511508523266085264440381", "67863024945042737"),
    ("13940532772086746446255318416574943", "32477972296048549"),
    ("369089540099546098577003037135326027", "355456763752979039"),
    (
        "3689756090368133007007803781494130157",
        "562038731897575391",
    ),
    (
        "1460108394706986028246669872859668049",
        "1097043902555184823",
    ),
    (
        "421811032170328413427706571138795403283",
        "5679364274029068053",
    ),
    (
        "11725351076854236155447266274407707059659",
        "107592488772921657781",
    ),
    (
        "6375362044852075763609151869982278584373",
        "43381785455019009751",
    ),
    (
        "71712990957379264115125963013452343876281",
        "214703788793893761709",
    ),
    (
        "7712233453135913368932285510810112425224381",
        "976731713743960381117",
    ),
    (
        "20128265973068341391057768341051314351284287",
        "2456174856045266568161",
    ),
    (
        "784119535391154029098967509928417449822498133",
        "9686263547398340563303",
    ),
    (
        "2803683963527548351796730108619624805269989271",
        "32644549665776976282137",
    ),
];

/// Numbers `p²·q` of 24 to 40 digits and `p`, from the same script.
const SQUARE_TIMES_PRIME: [(&str, &str); 17] = [
    ("11600382321802575116393", "23698651"),
    ("724786193196385483935991", "58368881"),
    ("2261698152765424611771587", "22546409"),
    ("367183506448342490770093073", "894184001"),
    ("274015494928214669180129503", "261112253"),
    ("15141973237350852198889588013", "418363051"),
    ("891795150912042591106706912629", "10009805441"),
    ("127865765020760838138900097301", "1084378879"),
    ("15208585220985526712885660574841", "5391920683"),
    ("9702448990281532427584365240563", "13471829461"),
    ("1525799298649440758544314798766727", "62688285581"),
    ("19164492370907452742408502927925163", "55708764821"),
    ("238887329777462851855669560966716821", "622351264751"),
    ("6700538257746637759168347342682666967", "827218645811"),
    ("12368401948957534170719242749441711911", "644715562861"),
    ("639939765837755711769534195450367445353", "10973516878633"),
    ("1329861175083638842148232074237553272719", "4427355512821"),
];

/// Products of three primes, 24 to 40 digits, and their smallest factor, from the same script.
const THREE_PRIMES: [(&str, &str); 17] = [
    ("525041845174882002732997", "64521139"),
    ("4451984957179157647530683", "53911211"),
    ("1535547965792743510117663", "20773871"),
    ("976737080408826112184375813", "945131393"),
    ("298459346023879774622180527", "110985527"),
    ("67451027332261269831506498267", "813421327"),
    ("101153396658779507722827201209", "3230239847"),
    ("1695188300942426179958049170023", "3977361233"),
    ("4225993642651284344226335448979", "2369133037"),
    ("82636418464572200596103698339967", "10346424619"),
    ("872758966521716638740800056584953", "31401001703"),
    ("17947480065266374834407973912685773", "35996281007"),
    ("369781784541982475869919304078101759", "611011257947"),
    ("806902680576131929927255784508895477", "113768724887"),
    ("73473187928964328908822120444443194519", "688304993213"),
    ("426106450602082452097437879433321893523", "4517572599217"),
    ("463055046795663606716349553160774304163", "1708597656179"),
];

/// Balanced semiprimes of 24 to 44 digits and their smaller factor, from PARI/GP 2.19
/// (`setrand(20261108)`): at least one for each row of the sieve's parameters up to 150 bits, where
/// the matrix is solved by Gaussian elimination.
const TABLE_SEMIPRIMES: [(&str, &str); 17] = [
    ("111613795193090726273351", "148996640213"),
    ("1341974000947881594149903", "131091008407"),
    ("10452520477619479527233843", "1904532067043"),
    ("282164222634889029227637241", "4086141136153"),
    ("1337794258457394739170632417", "26705952827657"),
    ("36264861107851368767096600701", "66698450691527"),
    ("179987916436909885923917286803", "170543852284261"),
    ("1445970105152896764846508099103", "185937127253473"),
    ("36380751143130406270310855448017", "4417114477408279"),
    ("459160229435590713568517303064659", "4336703275418911"),
    ("3243474231660087358132110622056209", "42125966341818193"),
    ("15143582936265471105696149239367897", "14346168686891227"),
    ("726272637885418447246938856992790423", "821112482256945649"),
    (
        "14022179084278879708381335826756212271",
        "1654493585083507187",
    ),
    (
        "6386607454153669457682130258182995137669",
        "60682993056337154101",
    ),
    (
        "394922858841242877528332229574110920321611",
        "438522261324648030191",
    ),
    (
        "28236337517092224097176577602278982999311161",
        "3441763388025965309701",
    ),
];

/// Semiprimes of 50 and 52 digits from the same script, whose matrices of 1500 to 1900 columns go
/// to Gaussian elimination, and of 62 digits (`setrand(20261109)`), whose matrix of more than 5000
/// columns goes to block Lanczos and whose factor base has primes above a block, sieved through
/// buckets.
const LARGE_SEMIPRIMES: [(&str, &str); 3] = [
    (
        "32773262337421171909691810167013181181622908991477",
        "4469410744098811775463937",
    ),
    (
        "1619705415673778220168931712115454058702017942528063",
        "39432550208760643014214763",
    ),
    (
        "54378948913005311650486261776698078969450268801641272586626601",
        "7062248073777160440366694260847",
    ),
];

/// The number of the decimal digits `s`, in byte order `order`.
fn dec(s: &str, order: ByteOrder) -> Number {
    let n: Number = s.parse().expect("decimal digits");
    in_byte_order(n, order)
}

/// `n` with its bytes in `order`.
fn in_byte_order(n: Number, order: ByteOrder) -> Number {
    if n.byte_order == order {
        return n;
    }
    let mut value = n.value;
    value.reverse();
    Number {
        value,
        negative: n.negative,
        byte_order: order,
    }
}

/// The smallest prime factor of `n >= 2`, by trial division.
fn smallest_factor(n: u64) -> u64 {
    let mut d = 2;
    while d * d <= n {
        if n.is_multiple_of(d) {
            return d;
        }
        d += 1;
    }
    n
}

/// Whether `n >= 2` is a power of a prime (the prime itself included).
fn is_prime_power(n: u64) -> bool {
    let p = smallest_factor(n);
    let mut m = n;
    while m.is_multiple_of(p) {
        m /= p;
    }
    m == 1
}

/// Checks that `f` is a factor of `|n|` other than 1 and `|n|`, positive and in the byte order of `n`.
fn assert_factor(n: &Number, f: &Number) {
    let m = n.abs();
    assert_eq!(f.byte_order, n.byte_order, "the byte order of {n}");
    assert!(!f.is_negative(), "a negative factor {f} of {n}");
    assert!(*f > 1u8 && *f < m, "a trivial factor {f} of {n}");
    assert!((&m % f).is_zero(), "{f} does not divide {n}");
}

// Factors

#[test]
fn balanced_semiprimes() {
    // Under Miri only the second, the sieve just above 2^64, in one byte order (Pollard's rho on the
    // first, below, takes some 50,000 steps there; the sieve of one number takes minutes).
    let skip = size(0, 1);
    let count = size(SEMIPRIMES.len(), 1);
    for (n, p) in SEMIPRIMES.iter().skip(skip).take(count) {
        for order in ORDERS.into_iter().take(size(2, 1)) {
            let n = dec(n, order);
            let p = dec(p, order);
            let f = factor::siqs(&n).expect("a factor");
            assert_factor(&n, &f);
            assert!(f == p || f == &n / &p, "{f} is not a prime factor of {n}");
        }
    }
}

#[test]
fn more_than_two_factors() {
    let count = size(SQUARE_TIMES_PRIME.len(), 0);
    for (n, p) in SQUARE_TIMES_PRIME.iter().take(count) {
        let n = dec(n, ByteOrder::BigEndian);
        let f = factor::siqs(&n).expect("a factor");
        assert_factor(&n, &f);
        let p = dec(p, ByteOrder::BigEndian);
        // One of p, p², q, p·q.
        let q = &(&n / &p) / &p;
        assert!(
            [p.clone(), &p * &p, q.clone(), &p * &q].contains(&f),
            "{f} is not made of the factors of {n}"
        );
    }
    for (n, _) in THREE_PRIMES.iter().take(size(THREE_PRIMES.len(), 0)) {
        let n = dec(n, ByteOrder::LittleEndian);
        let f = factor::siqs(&n).expect("a factor");
        assert_factor(&n, &f);
    }
}

#[test]
fn small_numbers() {
    // Every number below the bound: a factor exactly for the composite numbers that are not prime
    // powers, by trial division below 1000 and Pollard's rho beyond.
    let bound = size(20_000, 300) as u64;
    for n in 0..bound {
        for order in ORDERS {
            let x = Number::new_from_u64(n, order);
            let f = factor::siqs(&x);
            if n < 4 || is_prime_power(n) {
                assert!(f.is_none(), "a factor {:?} of {n}", f);
            } else {
                let f = f.unwrap_or_else(|| panic!("no factor of {n}"));
                assert_factor(&x, &f);
            }
        }
    }
}

#[test]
fn words() {
    // Products of two primes above 1000, below 2^64 (Pollard's rho) and just above it (the sieve),
    // and their squares and cubes.
    let mut state = 0x5149_5351u64;
    let count = size(200, 1);
    for k in 0..count {
        let bits = 20 + (k % 13) as u32;
        let mut p = (xorshift(&mut state) >> (64 - bits)) | (1 << (bits - 1)) | 1;
        while smallest_factor(p) != p {
            p += 2;
        }
        let mut q = (xorshift(&mut state) >> (bits + (k % 3) as u32)) | 1001;
        while smallest_factor(q) != q || q == p {
            q += 2;
        }
        let n = Number::new_from_u128(u128::from(p) * u128::from(q), ByteOrder::BigEndian);
        let f = factor::siqs(&n).expect("a factor");
        assert_factor(&n, &f);
        let pp = Number::new_from_u64(p, ByteOrder::LittleEndian);
        assert_eq!(factor::siqs(&pp.pow(2)), None, "{p}^2 is a prime power");
        assert_eq!(factor::siqs(&pp.pow(3)), None, "{p}^3 is a prime power");
        if !cfg!(miri) {
            let square = n.pow(2);
            let f = factor::siqs(&square).expect("a factor of a square");
            assert_factor(&square, &f);
        }
    }
}

#[test]
fn no_factor() {
    for order in ORDERS {
        let none = |n: Number| assert_eq!(factor::siqs(&n), None, "a factor of {n}");
        none(Number::new(vec![], order));
        none(Number::new_from_u8(0, order));
        none(Number::new_from_u8(1, order));
        none(Number::new_from_i32(-1, order));
        none(Number::new_from_u8(2, order));
        none(Number::new_from_i32(-3, order));
        // Primes, of 61, 127 and 521 bits, and powers of primes.
        let two = Number::new_from_u8(2, order);
        for e in [61u32, 127, 521].into_iter().take(size(3, 2)) {
            none(&two.pow(e) - 1u8);
        }
        none(two.pow(200));
        none(Number::new_from_u8(3, order).pow(100));
        none(dec("1000000007", order).pow(7));
        none(dec("170141183460469231731687303715884105727", order).pow(2));
        // Beyond the sieve's range: a factor would take years (its primality test takes long under
        // Miri).
        if !cfg!(miri) {
            let big = &(&two.pow(521) - 1u8) * &(&two.pow(607) - 1u8);
            none(big);
        }
    }
}

#[test]
fn small_factors_and_powers() {
    for order in ORDERS {
        let n = dec(SEMIPRIMES[8].0, order);
        // A factor below 1000 is found first.
        let m = &n * &Number::new_from_u16(997, order);
        let f = factor::siqs(&m).expect("a factor");
        assert_factor(&m, &f);
        let even = &n * &Number::new_from_u8(2, order);
        assert_eq!(factor::siqs(&even), Some(Number::new_from_u8(2, order)));
        // A prime above 1000: the factor base meets 1009, which is the factor; 65537 it does not,
        // and the sieve finds a factor (not under Miri, where it takes an hour).
        for small in [1009u32, 65537].into_iter().take(size(2, 1)) {
            let m = &n * &Number::new_from_u32(small, order);
            let f = factor::siqs(&m).expect("a factor");
            assert_factor(&m, &f);
        }
        // Perfect powers of composite numbers give their root, or a factor of it.
        for e in [2u32, 3, 5] {
            let m = n.pow(e);
            let f = factor::siqs(&m).expect("a factor");
            assert_factor(&m, &f);
        }
    }
}

#[test]
fn sign_and_form_of_the_input() {
    // Under Miri a product of two primes of 7 digits, by Pollard's rho: the sieve is checked by
    // the other tests.
    let (n, p) = if cfg!(miri) {
        ("1000036000099", "1000003")
    } else {
        SEMIPRIMES[10]
    };
    for order in ORDERS {
        let x = dec(n, order);
        let expected = factor::siqs(&x).expect("a factor");
        assert!(expected == dec(p, order) || expected == &x / &dec(p, order));
        // Negative, with zero bytes above the top: the same factor, positive.
        let mut bytes = le_bytes(&x);
        bytes.extend_from_slice(&[0, 0, 0]);
        let y = number(&bytes, true, order);
        let before = y.clone();
        let f = factor::siqs(&y).expect("a factor");
        assert_eq!(f, expected);
        assert!(!f.is_negative());
        assert_eq!(f.byte_order, order);
        // The input is left as it was.
        assert_eq!(y.value, before.value);
        assert_eq!(y.negative, before.negative);
        // A negative zero is zero.
        assert_eq!(factor::siqs(&number(&[0, 0], true, order)), None);
    }
}

#[test]
fn reproducible() {
    let n = dec(
        if cfg!(miri) {
            "1000036000099"
        } else {
            SEMIPRIMES[20].0
        },
        ByteOrder::BigEndian,
    );
    let f = factor::siqs(&n);
    assert!(f.is_some());
    assert_eq!(factor::siqs(&n), f);
}

#[test]
fn semiprimes_across_the_table() {
    // Under Miri the first only: a sieve of 24 digits takes about a minute there.
    for (n, p) in TABLE_SEMIPRIMES
        .iter()
        .take(size(TABLE_SEMIPRIMES.len(), 1))
    {
        for order in ORDERS.into_iter().take(size(2, 1)) {
            let n = dec(n, order);
            let p = dec(p, order);
            let f = factor::siqs(&n).expect("a factor");
            assert_factor(&n, &f);
            assert!(f == p || f == &n / &p, "{f} is not a prime factor of {n}");
        }
    }
}

#[test]
fn lanczos_and_buckets() {
    // Not under Miri, where one of them would take days; in the other byte order the second only.
    for (i, (n, p)) in LARGE_SEMIPRIMES.iter().enumerate().take(size(3, 0)) {
        for order in ORDERS.into_iter().take(if i == 1 { 2 } else { 1 }) {
            let n = dec(n, order);
            let p = dec(p, order);
            let f = factor::siqs(&n).expect("a factor");
            assert_factor(&n, &f);
            assert!(f == p || f == &n / &p, "{f} is not a prime factor of {n}");
        }
    }
}

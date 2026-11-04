//! The exponential, the square root and the logarithm, against mpmath at 50 digits and by the
//! identities they satisfy.

use super::common::{Num, Rng, assert_close, each_type, frobenius, random};
use a::math::complex::{Complex, Float};
use a::math::matrix::Matrix;

/// The real test matrix of mpmath's script: `((7i + 3j + 1) mod 11 - 5) / 4`, plus `shift` on
/// the diagonal.
fn real<T: Num>(n: usize, shift: f64) -> Matrix<T> {
    Matrix::from_fn(n, n, |i, j| {
        let x = ((i * 7 + j * 3 + 1) % 11) as f64 - 5.0;
        T::real(x / 4.0 + if i == j { shift } else { 0.0 })
    })
}

/// The complex test matrix: the real one plus `i·((5i + 2j) mod 7 - 3) / 4`.
fn complex(n: usize, shift: f64) -> Matrix<Complex<f64>> {
    let re = real::<f64>(n, shift);
    Matrix::from_fn(n, n, |i, j| {
        Complex::new(re[(i, j)], (((i * 5 + j * 2) % 7) as f64 - 3.0) / 4.0)
    })
}

/// The reference values as a matrix of the type, column after column.
fn reference<T: Num>(values: &[f64]) -> Matrix<T> {
    let n = (values.len() as f64).sqrt() as usize;
    Matrix::from_column_major(n, n, values.iter().map(|&x| T::real(x)).collect())
}

/// Complex reference values.
fn reference_c(values: &[(f64, f64)]) -> Matrix<Complex<f64>> {
    let n = (values.len() as f64).sqrt() as usize;
    Matrix::from_column_major(
        n,
        n,
        values
            .iter()
            .map(|&(re, im)| Complex::new(re, im))
            .collect(),
    )
}

/// Asserts that `got` is within `tol` of `want`, relatively in the Frobenius norm.
#[track_caller]
fn near<T: Num>(got: &Matrix<T>, want: &Matrix<T>, tol: f64) {
    let d = frobenius(&(got - want)) / frobenius(want).max(f64::MIN_POSITIVE);
    assert!(d <= tol, "relative error {d:e} > {tol:e}");
}

/// The same complex matrix in single precision.
fn narrow(a: &Matrix<Complex<f64>>) -> Matrix<Complex<f32>> {
    a.map(|z| Complex::new(z.re as f32, z.im as f32))
}

#[test]
fn against_mpmath() {
    near(&real::<f64>(5, 0.0).exp(), &reference(&EXP_REAL5), 1e-14);
    near(
        &(real::<f64>(5, 0.0) * 8.0).exp(),
        &reference(&EXP_REAL5_X8),
        1e-12,
    );
    near(&complex(4, 0.0).exp(), &reference_c(&EXP_CPLX4), 1e-14);
    near(
        &(complex(4, 0.0) * Complex::new(8.0, 0.0)).exp(),
        &reference_c(&EXP_CPLX4_X8),
        1e-12,
    );
    near(
        &real::<f64>(5, 3.0).sqrt().unwrap(),
        &reference(&SQRT_REAL5S),
        1e-14,
    );
    near(
        &complex(4, 0.0).sqrt().unwrap(),
        &reference_c(&SQRT_CPLX4),
        1e-14,
    );
    near(
        &real::<f64>(5, 3.0).log().unwrap(),
        &reference(&LOG_REAL5S),
        1e-14,
    );
    near(
        &complex(4, 0.0).log().unwrap(),
        &reference_c(&LOG_CPLX4),
        1e-14,
    );
    near(
        &complex(4, 3.0).log().unwrap(),
        &reference_c(&LOG_CPLX4S),
        1e-14,
    );
    // In single precision.
    near(&real::<f32>(5, 0.0).exp(), &reference(&EXP_REAL5), 1e-5);
    near(
        &(real::<f32>(5, 0.0) * 8.0).exp(),
        &reference(&EXP_REAL5_X8),
        1e-4,
    );
    near(
        &narrow(&complex(4, 0.0)).exp(),
        &narrow(&reference_c(&EXP_CPLX4)),
        1e-5,
    );
    near(
        &real::<f32>(5, 3.0).sqrt().unwrap(),
        &reference(&SQRT_REAL5S),
        1e-5,
    );
    near(
        &narrow(&complex(4, 0.0)).log().unwrap(),
        &narrow(&reference_c(&LOG_CPLX4)),
        1e-5,
    );
    // A real matrix with a negative real eigenvalue has no real square root or logarithm.
    assert!(real::<f64>(5, 0.0).sqrt().is_none());
    assert!(real::<f64>(5, 0.0).log().is_none());
}

fn identities_of<T: Num>() {
    let mut r = Rng(0x5EED_0301);
    for n in [0, 1, 2, 3, 6, 12, 25] {
        let a = random::<T>(&mut r, n, n);
        let e = a.exp();
        let tol = 1e3 * (n + 1) as f64;
        assert_close(&(&e * &(-&a).exp()), &Matrix::identity(n), tol);
        // `det(e^A) = e^tr(A)`.
        let (d, t) = (e.det().c64(), a.trace().c64().exp());
        assert!((d - t).abs() <= tol * T::EPS * t.abs(), "{n}");
        // A matrix whose eigenvalues are near 2 + n has a principal square root and logarithm.
        let mut b = a.clone();
        for i in 0..n {
            b[(i, i)] += T::real(2.0 + n as f64);
        }
        let s = b.sqrt().unwrap();
        assert_close(&(&s * &s), &b, tol);
        let l = b.log().unwrap();
        assert_close(&l.exp(), &b, 4.0 * tol);
        // Scaled up, so that the squaring is exercised: `e^B = (e^(B/2))²`.
        let big = &a * T::real(4.0);
        let half = (&big * T::real(0.5)).exp();
        near(&big.exp(), &(&half * &half), 1e3 * (n + 1) as f64 * T::EPS);
    }
}

#[test]
fn identities() {
    each_type!(identities_of);
}

fn special_of<T: Num>() {
    let z = Matrix::<T>::zeros(3, 3);
    let i = Matrix::<T>::identity(3);
    assert_eq!(z.exp(), i);
    assert_eq!(i.sqrt().unwrap(), i);
    assert_eq!(z.sqrt().unwrap(), z);
    assert!(z.log().is_none());
    assert_close(&i.log().unwrap(), &z, 4.0);
    // A diagonal matrix's functions are its diagonal's.
    let d = Matrix::from_diagonal(&[T::real(0.5), T::real(2.0), T::real(3.0)]);
    let want = Matrix::from_diagonal(&[0.5f64, 2.0, 3.0].map(|x| T::real(x.exp())));
    assert_close(&d.exp(), &want, 32.0);
    let want = Matrix::from_diagonal(&[0.5f64, 2.0, 3.0].map(|x| T::real(x.ln())));
    assert_close(&d.log().unwrap(), &want, 32.0);
    // A nilpotent block has no square root.
    let n: Matrix<T> = Matrix::from_fn(2, 2, |i, j| {
        T::real(if i == 0 && j == 1 { 1.0 } else { 0.0 })
    });
    assert!(n.sqrt().is_none());
    // Its exponential is `I + N`.
    assert_close(&n.exp(), &(&Matrix::identity(2) + &n), 4.0);
    // Not a number.
    let mut nan = i.clone();
    nan[(0, 1)] = T::real(f64::NAN);
    assert!(nan.exp().as_slice().iter().all(|x| x.c64().is_nan()));
    assert!(nan.sqrt().is_none() && nan.log().is_none());
    let _ = <T::Real as Float>::ZERO;
}

#[test]
fn special_matrices() {
    each_type!(special_of);
}

#[test]
fn complex_results_of_complex_matrices() {
    // `-I` has the principal logarithm `iπ·I` and the square root `i·I`.
    let m = Matrix::<Complex<f64>>::identity(2) * Complex::new(-1.0, 0.0);
    let l = m.log().unwrap();
    near(
        &l,
        &(Matrix::identity(2) * Complex::new(0.0, core::f64::consts::PI)),
        1e-15,
    );
    let s = m.sqrt().unwrap();
    near(&s, &(Matrix::identity(2) * Complex::new(0.0, 1.0)), 1e-15);
}

// Generated by mpmath 1.3 at 50 digits (expm, sqrtm, logm), column after column.

const EXP_REAL5: [f64; 25] = [
    -0.40864434110048936,
    0.07409153861270298,
    0.759611586396231,
    -1.465438165588391,
    0.44300515672059304,
    -0.4610621653682522,
    0.24820253399933903,
    1.1254496365586522,
    -0.6863779982863284,
    0.6349008267013937,
    1.4773686802080166,
    -0.7565080847952791,
    2.6373972924452285,
    1.084814467059713,
    -2.0566547520192677,
    0.3540913447357498,
    0.9372094902871732,
    -1.7019830326902705,
    0.9356763039814013,
    1.4802883726160143,
    -1.2900313819662266,
    0.33613324561495866,
    1.2703897059875338,
    -1.509272877545262,
    1.9077274898373084,
];

const EXP_REAL5_X8: [f64; 25] = [
    -843.6038595063792,
    -526.038141015697,
    2076.455379057395,
    -633.898100036784,
    -54.41587440943577,
    -860.1039306428373,
    -320.01098471572135,
    1449.426754662667,
    -679.934901874692,
    238.6959302639472,
    1609.9853002475372,
    -799.8751477834916,
    1514.9199296889763,
    1494.0062990373685,
    -2368.508600913391,
    -1950.2937126329375,
    1237.3443466477381,
    -2673.250490844341,
    -1845.0706485995468,
    3233.743419548897,
    -1944.2327244068822,
    -331.18312684519736,
    2090.85375029058,
    -1599.2562911860668,
    1078.222511954887,
];

const EXP_CPLX4: [(f64, f64); 16] = [
    (-0.46141242768543395, -0.13180022778915032),
    (-0.12437596155949655, -0.26018447305443027),
    (0.926685862836244, 0.08794890319650685),
    (-1.095589195948113, 1.024967655656184),
    (-0.07163644997253843, 0.30325315978584677),
    (-0.09673645004761722, -0.1747872116773369),
    (1.0446356250631441, -0.3627269814027823),
    (0.1301173576710276, 0.6023872472117751),
    (1.0824806915396525, -0.015595961370035728),
    (-0.31187249893051217, 0.5186916680216554),
    (1.663920249370939, -2.6467318436572884),
    (0.9810115477432633, 0.017695731833510996),
    (0.6900147195345175, -0.7846323385132603),
    (0.9743734725799288, -0.19080537462073657),
    (-1.5517810858226964, 1.7152957209822406),
    (-0.026059396506887877, -2.0282874592772604),
];

const EXP_CPLX4_X8: [(f64, f64); 16] = [
    (-1382.16855959852, 2707.950343766726),
    (-2161.424135663659, -422.1506447092399),
    (4879.478260852578, -329.8530824965177),
    (719.6889836017372, 4447.9622847929795),
    (744.0876934652557, 1476.938378628567),
    (-919.6584083097073, 690.6932510566278),
    (1968.5840443348911, -1933.3883067602158),
    (1929.921173336344, 1166.5316174486302),
    (2443.819615409637, -53.82457038259746),
    (102.0346899074381, 1511.5187816318376),
    (33.6028020965853, -4063.8301523857667),
    (2422.102100167491, -1396.982080839599),
    (-2644.7997438238517, -3987.962134031208),
    (2374.5429119546357, -2551.4770796676385),
    (-4056.141481487488, 7044.093953742347),
    (-6455.482330613634, -2682.8608932497846),
];

const SQRT_REAL5S: [f64; 25] = [
    1.4726557017669157,
    0.25869963225054304,
    -0.09695733963423324,
    -0.3375522940041712,
    0.11003937885935161,
    -0.05306358898879797,
    1.287238344333727,
    0.14405976010538635,
    -0.12715784994954754,
    0.4005774480933583,
    0.1238945236586378,
    -0.1416612549084581,
    2.070557416588283,
    0.05552394105539912,
    -0.1775149282554959,
    0.3622556396230759,
    0.04179391263890585,
    -0.17584448143131554,
    2.0326289640180164,
    -0.022304840421976436,
    -0.18806521468875462,
    0.330690324518398,
    -0.028029941422934444,
    -0.25903433472327514,
    1.9045928805296273,
];

const SQRT_CPLX4: [(f64, f64); 16] = [
    (-0.1305950873671088, 0.5002409831307766),
    (0.5687364579965972, 2.1132549179666857),
    (-0.10280809093801707, -0.29857194793352876),
    (-1.2163614076316065, 0.2926712728290518),
    (-0.09706385914292939, -0.47253446037707775),
    (-0.059438751523433565, -1.4369140942553151),
    (0.06454026572827669, 0.3471458286870586),
    (-0.05229320784246422, -0.4398064014265005),
    (0.4205289633395557, -0.19057214794812272),
    (-0.08578810441581991, -0.8670958822472796),
    (1.111851870534163, -0.2282710123556405),
    (0.3675661678192271, 0.061086429402730937),
    (1.243844813555689, 0.3317645285875142),
    (0.9238390907465496, -0.8625483587505157),
    (-0.3650042576031283, -0.07722885629177098),
    (1.7816272707571552, -0.08380081337025719),
];

const LOG_REAL5S: [f64; 25] = [
    0.8312833868435073,
    0.35952712970219924,
    -0.1382131531820212,
    -0.36316908993912417,
    0.07821608385096769,
    -0.0382631762800978,
    0.4544911196313371,
    0.1751564103528259,
    -0.12669487339916827,
    0.5248890878786384,
    0.12373506977331959,
    -0.16534470558189626,
    1.4656313533606755,
    0.0486908462617314,
    -0.16326468166391117,
    0.4145749036392581,
    0.011324702284040112,
    -0.15946537483242978,
    1.4538041805785664,
    -0.04159410892920429,
    -0.18752401025243107,
    0.44412501480070243,
    -0.06190249405993922,
    -0.2670525161843763,
    1.2382259893775003,
];

const LOG_CPLX4: [(f64, f64); 16] = [
    (-1.2351916539007306, 2.3811530609544485),
    (1.1013522710401091, 5.751437486062311),
    (-0.16886490459480136, -0.6350009455521394),
    (-2.7436646713405377, 1.8125147917338595),
    (-0.16995412048423109, -1.12577866391652),
    (-0.5792316538268505, -3.56389671457387),
    (-0.21568718297142192, 0.4721285189603243),
    (0.3331410408969381, -1.1714459533871073),
    (0.7393146490474126, -0.9604020196584055),
    (0.19948488199659853, -2.3136316018839027),
    (0.24879178969959984, -0.3872612461615717),
    (1.0134218889529298, -0.3335882845399491),
    (2.946928526877611, -0.2381245317322044),
    (2.763266668607822, -2.898482098359805),
    (-0.5027866527105621, -0.10753849937985441),
    (2.827599040318983, -0.15175923689983883),
];

const LOG_CPLX4S: [(f64, f64); 16] = [
    (0.7799512274899486, -0.18880625952760624),
    (0.2927472636163324, 0.34760906934962343),
    (-0.09058643529833572, -0.09156940431898979),
    (-0.3673631771492822, -0.1927035138132526),
    (-0.06261256733033056, -0.128982599321903),
    (0.6210560902204794, -0.33636485787824044),
    (0.115788554901526, 0.18604901220250206),
    (-0.15728899455285544, -0.0826112883084996),
    (0.1616718074299953, 0.060634945034871296),
    (-0.15352763914843642, -0.1639427116680186),
    (1.4609964275836624, -0.13699616505197557),
    (0.05383354080708117, 0.13623565594166198),
    (0.37321638171998595, 0.3320896072630335),
    (0.06288512898227337, -0.01766861990230528),
    (-0.1647967458212374, -0.06691084041452824),
    (1.4343901979895786, -0.08839364489784675),
];

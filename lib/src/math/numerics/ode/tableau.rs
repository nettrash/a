//! The coefficients of the explicit Runge–Kutta methods of this module, as their authors
//! published them, and the [`ButcherTableau`] that shows them to a caller.
//!
//! The numbers are facts of the methods, not choices of this crate. The classical method's are in
//! every textbook; the two Dormand–Prince pairs are taken from Hairer, Nørsett and Wanner[^hnw]
//! and from the codes DOPRI5 and DOP853 that go with the book, the first as the fractions the
//! authors give and the second as the decimals of thirty digits they print. Nothing here is
//! trusted on the strength of a transcription: the tests of the crate check every one of them
//! against the order conditions the methods are derived from — all the rooted trees up to the
//! order claimed, for the weights, for the embedded formulas and for the continuous extensions.
//!
//! [^hnw]: E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential Equations I:
//!     Nonstiff Problems*, 2nd edition, Springer, 1993, sections II.1, II.5, II.6 and II.10.

/// The coefficients of an explicit Runge–Kutta method: what [`Rk4::tableau`],
/// [`DormandPrince5::tableau`] and [`DormandPrince853::tableau`] return, and exactly the numbers
/// those methods compute with.
///
/// With stages `kᵢ = f(t + cᵢ·h, y + h·Σⱼ aᵢⱼ·kⱼ)`, a step is `y₁ = y + h·Σᵢ bᵢ·kᵢ`. Each row of
/// [`e`](Self::e) gives an error estimate `h·Σᵢ eᵢ·kᵢ`, the difference of `y₁` and an embedded
/// formula of a lower order; each row of [`d`](Self::d) is one more coefficient of the continuous
/// extension, as the method's page describes it. The stages past the first `b.len()` — one for the
/// 5(4) pair, four for the 8(5,3) method — are evaluated after the step, at its end or for the
/// continuous extension alone, and their rows of `a` are what places them.
///
/// [`Rk4::tableau`]: super::Rk4::tableau
/// [`DormandPrince5::tableau`]: super::DormandPrince5::tableau
/// [`DormandPrince853::tableau`]: super::DormandPrince853::tableau
///
/// # Examples
///
/// ```
/// use a::math::numerics::DormandPrince5;
///
/// let tableau = DormandPrince5::<f64>::new().tableau();
/// // Every row of `a` sums to the abscissa of its stage, and the weights sum to one.
/// for (row, c) in tableau.a.iter().zip(tableau.c) {
///     assert!((row.iter().sum::<f64>() - c).abs() < 1e-15);
/// }
/// assert!((tableau.b.iter().sum::<f64>() - 1.0).abs() < 1e-15);
/// // The pair shares its stages: the seventh is evaluated at the new point, which makes its row
/// // of `a` the weights themselves.
/// assert_eq!(tableau.a[6], &tableau.b[..6]);
/// // And an error estimate is the difference of two formulas for the same step, so its weights
/// // sum to nothing.
/// assert!(tableau.e[0].iter().sum::<f64>().abs() < 1e-15);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ButcherTableau {
    /// The abscissae: stage `i` is evaluated at `t + c[i]·h`.
    pub c: &'static [f64],
    /// The stage coefficients: row `i` holds the `i` numbers `a[i][j]`, `j < i`, that multiply
    /// the stages before it.
    pub a: &'static [&'static [f64]],
    /// The weights of the solution the method propagates.
    pub b: &'static [f64],
    /// The weights of the error estimates, one row for each embedded formula: none for the
    /// classical method, one for the 5(4) pair, and for the 8(5,3) method the fifth-order row
    /// followed by the third-order one.
    pub e: &'static [&'static [f64]],
    /// The rows of the continuous extension that are not fixed by the ends of the step: none for
    /// the classical method, one for the 5(4) pair, four for the 8(5,3) method.
    pub d: &'static [&'static [f64]],
}

// ---------------------------------------------------------------------------------------------
// The classical method (Kutta, 1901)
// ---------------------------------------------------------------------------------------------

const RK4_C: [f64; 4] = [0.0, 0.5, 0.5, 1.0];
const RK4_A: [&[f64]; 4] = [&[], &[0.5], &[0.0, 0.5], &[0.0, 0.0, 1.0]];
const RK4_B: [f64; 4] = [1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];

/// The classical Runge–Kutta method of the fourth order.
pub(super) const RK4: ButcherTableau = ButcherTableau {
    c: &RK4_C,
    a: &RK4_A,
    b: &RK4_B,
    e: &[],
    d: &[],
};

// ---------------------------------------------------------------------------------------------
// Dormand and Prince's 5(4) pair (1980), as in DOPRI5
// ---------------------------------------------------------------------------------------------

const DOPRI5_C: [f64; 7] = [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0];

const DOPRI5_A: [&[f64]; 7] = [
    &[],
    &[1.0 / 5.0],
    &[3.0 / 40.0, 9.0 / 40.0],
    &[44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0],
    &[
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    ],
    &[
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    ],
    &[
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    ],
];

/// The fifth-order weights, which the pair propagates (local extrapolation); the seventh stage has
/// none.
const DOPRI5_B: [f64; 7] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
    0.0,
];

/// The fifth-order weights less the fourth-order ones: DOPRI5's `e1 … e7`.
const DOPRI5_E: [f64; 7] = [
    71.0 / 57600.0,
    0.0,
    -71.0 / 16695.0,
    71.0 / 1920.0,
    -17253.0 / 339200.0,
    22.0 / 525.0,
    -1.0 / 40.0,
];

/// The last coefficient of the continuous extension of order 4: DOPRI5's `d1 … d7`.
const DOPRI5_D: [f64; 7] = [
    -12715105075.0 / 11282082432.0,
    0.0,
    87487479700.0 / 32700410799.0,
    -10690763975.0 / 1880347072.0,
    701980252875.0 / 199316789632.0,
    -1453857185.0 / 822651844.0,
    69997945.0 / 29380423.0,
];

/// Dormand and Prince's pair of orders 5 and 4, with the continuous extension of DOPRI5.
pub(super) const DOPRI5: ButcherTableau = ButcherTableau {
    c: &DOPRI5_C,
    a: &DOPRI5_A,
    b: &DOPRI5_B,
    e: &[&DOPRI5_E],
    d: &[&DOPRI5_D],
};

// ---------------------------------------------------------------------------------------------
// The 8(5,3) method of Dormand and Prince, as in DOP853
// ---------------------------------------------------------------------------------------------
//
// Twelve stages make the step; the thirteenth is the derivative at its end, which the next step
// begins with, and the last three are evaluated only for the continuous extension. The decimals
// are those of DOP853, given to more digits than an `f64` carries so that the values here are the
// ones the code prints.

#[allow(clippy::excessive_precision)]
const DOP853_C: [f64; 16] = [
    0.0,
    0.526001519587677318785587544488e-1,
    0.789002279381515978178381316732e-1,
    0.118350341907227396726757197510,
    0.281649658092772603273242802490,
    0.333333333333333333333333333333,
    0.25,
    0.307692307692307692307692307692,
    0.651282051282051282051282051282,
    0.6,
    0.857142857142857142857142857142,
    1.0,
    1.0,
    0.1,
    0.2,
    0.777777777777777777777777777778,
];

#[allow(clippy::excessive_precision)]
const DOP853_A1: [f64; 0] = [];
#[allow(clippy::excessive_precision)]
const DOP853_A2: [f64; 1] = [5.26001519587677318785587544488e-2];
#[allow(clippy::excessive_precision)]
const DOP853_A3: [f64; 2] = [
    1.97250569845378994544595329183e-2,
    5.91751709536136983633785987549e-2,
];
#[allow(clippy::excessive_precision)]
const DOP853_A4: [f64; 3] = [
    2.95875854768068491816892993775e-2,
    0.0,
    8.87627564304205475450678981324e-2,
];
#[allow(clippy::excessive_precision)]
const DOP853_A5: [f64; 4] = [
    2.41365134159266685502369798665e-1,
    0.0,
    -8.84549479328286085344864962717e-1,
    9.24834003261792003115737966543e-1,
];
#[allow(clippy::excessive_precision)]
const DOP853_A6: [f64; 5] = [
    3.7037037037037037037037037037e-2,
    0.0,
    0.0,
    1.70828608729473871279604482173e-1,
    1.25467687566822425016691814123e-1,
];
#[allow(clippy::excessive_precision)]
const DOP853_A7: [f64; 6] = [
    3.7109375e-2,
    0.0,
    0.0,
    1.70252211019544039314978060272e-1,
    6.02165389804559606850219397283e-2,
    -1.7578125e-2,
];
#[allow(clippy::excessive_precision)]
const DOP853_A8: [f64; 7] = [
    3.70920001185047927108779319836e-2,
    0.0,
    0.0,
    1.70383925712239993810214054705e-1,
    1.07262030446373284651809199168e-1,
    -1.53194377486244017527936158236e-2,
    8.27378916381402288758473766002e-3,
];
#[allow(clippy::excessive_precision)]
const DOP853_A9: [f64; 8] = [
    6.24110958716075717114429577812e-1,
    0.0,
    0.0,
    -3.36089262944694129406857109825,
    -8.68219346841726006818189891453e-1,
    2.75920996994467083049415600797e1,
    2.01540675504778934086186788979e1,
    -4.34898841810699588477366255144e1,
];
#[allow(clippy::excessive_precision)]
const DOP853_A10: [f64; 9] = [
    4.77662536438264365890433908527e-1,
    0.0,
    0.0,
    -2.48811461997166764192642586468,
    -5.90290826836842996371446475743e-1,
    2.12300514481811942347288949897e1,
    1.52792336328824235832596922938e1,
    -3.32882109689848629194453265587e1,
    -2.03312017085086261358222928593e-2,
];
#[allow(clippy::excessive_precision)]
const DOP853_A11: [f64; 10] = [
    -9.3714243008598732571704021658e-1,
    0.0,
    0.0,
    5.18637242884406370830023853209,
    1.09143734899672957818500254654,
    -8.14978701074692612513997267357,
    -1.85200656599969598641566180701e1,
    2.27394870993505042818970056734e1,
    2.49360555267965238987089396762,
    -3.0467644718982195003823669022,
];
#[allow(clippy::excessive_precision)]
const DOP853_A12: [f64; 11] = [
    2.27331014751653820792359768449,
    0.0,
    0.0,
    -1.05344954667372501984066689879e1,
    -2.00087205822486249909675718444,
    -1.79589318631187989172765950534e1,
    2.79488845294199600508499808837e1,
    -2.85899827713502369474065508674,
    -8.87285693353062954433549289258,
    1.23605671757943030647266201528e1,
    6.43392746015763530355970484046e-1,
];
#[allow(clippy::excessive_precision)]
const DOP853_A13: [f64; 12] = [
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2,
];
#[allow(clippy::excessive_precision)]
const DOP853_A14: [f64; 13] = [
    5.61675022830479523392909219681e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.53500210216624811088794765333e-1,
    -2.46239037470802489917441475441e-1,
    -1.24191423263816360469010140626e-1,
    1.5329179827876569731206322685e-1,
    8.20105229563468988491666602057e-3,
    7.56789766054569976138603589584e-3,
    -8.298e-3,
];
#[allow(clippy::excessive_precision)]
const DOP853_A15: [f64; 14] = [
    3.18346481635021405060768473261e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    2.83009096723667755288322961402e-2,
    5.35419883074385676223797384372e-2,
    -5.49237485713909884646569340306e-2,
    0.0,
    0.0,
    -1.08347328697249322858509316994e-4,
    3.82571090835658412954920192323e-4,
    -3.40465008687404560802977114492e-4,
    1.41312443674632500278074618366e-1,
];
#[allow(clippy::excessive_precision)]
const DOP853_A16: [f64; 15] = [
    -4.28896301583791923408573538692e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -4.69762141536116384314449447206,
    7.68342119606259904184240953878,
    4.06898981839711007970213554331,
    3.56727187455281109270669543021e-1,
    0.0,
    0.0,
    0.0,
    -1.39902416515901462129418009734e-3,
    2.9475147891527723389556272149,
    -9.15095847217987001081870187138,
];

const DOP853_A: [&[f64]; 16] = [
    &DOP853_A1,
    &DOP853_A2,
    &DOP853_A3,
    &DOP853_A4,
    &DOP853_A5,
    &DOP853_A6,
    &DOP853_A7,
    &DOP853_A8,
    &DOP853_A9,
    &DOP853_A10,
    &DOP853_A11,
    &DOP853_A12,
    &DOP853_A13,
    &DOP853_A14,
    &DOP853_A15,
    &DOP853_A16,
];

/// The weights of order 8, which are also the row of the thirteenth stage.
#[allow(clippy::excessive_precision)]
const DOP853_B: [f64; 12] = [
    5.42937341165687622380535766363e-2,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    3.1116436695781989440891606237e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    4.47106157277725905176885569043e-2,
];

/// The weights of order 8 less those of the embedded formula of order 5: DOP853's `er1 … er12`.
#[allow(clippy::excessive_precision)]
const DOP853_E5: [f64; 12] = [
    0.1312004499419488073250102996e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.1225156446376204440720569753e1,
    -0.4957589496572501915214079952,
    0.1664377182454986536961530415e1,
    -0.3503288487499736816886487290,
    0.3341791187130174790297318841,
    0.8192320648511571246570742613e-1,
    -0.2235530786388629525884427845e-1,
];

/// The weights of order 8 less those of the embedded formula of order 3, which DOP853 gives as
/// `bhh1`, `bhh2` and `bhh3` on the first, ninth and twelfth stages; the three differences were
/// taken on the published decimals.
#[allow(clippy::excessive_precision)]
const DOP853_E3: [f64; 12] = [
    -1.89800754072407615714702328876e-1,
    0.0,
    0.0,
    0.0,
    0.0,
    4.45031289275240888144113950566,
    1.89151789931450038304281599044,
    -5.8012039600105847814672114227,
    -4.22682321323791962932445679177e-1,
    -1.52160949662516078556178806805e-1,
    2.01365400804030348374776537501e-1,
    2.26517921983608258118062039631e-2,
];

#[allow(clippy::excessive_precision)]
const DOP853_D4: [f64; 16] = [
    -0.84289382761090128651353491142e1,
    0.0,
    0.0,
    0.0,
    0.0,
    0.56671495351937776962531783590,
    -0.30689499459498916912797304727e1,
    0.23846676565120698287728149680e1,
    0.21170345824450282767155149946e1,
    -0.87139158377797299206789907490,
    0.22404374302607882758541771650e1,
    0.63157877876946881815570249290,
    -0.88990336451333310820698117400e-1,
    0.18148505520854727256656404962e2,
    -0.91946323924783554000451984436e1,
    -0.44360363875948939664310572000e1,
];
#[allow(clippy::excessive_precision)]
const DOP853_D5: [f64; 16] = [
    0.10427508642579134603413151009e2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.24228349177525818288430175319e3,
    0.16520045171727028198505394887e3,
    -0.37454675472269020279518312152e3,
    -0.22113666853125306036270938578e2,
    0.77334326684722638389603898808e1,
    -0.30674084731089398182061213626e2,
    -0.93321305264302278729567221706e1,
    0.15697238121770843886131091075e2,
    -0.31139403219565177677282850411e2,
    -0.93529243588444783865713862664e1,
    0.35816841486394083752465898540e2,
];
#[allow(clippy::excessive_precision)]
const DOP853_D6: [f64; 16] = [
    0.19985053242002433820987653617e2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.38703730874935176555105901742e3,
    -0.18917813819516756882830838328e3,
    0.52780815920542364900561016686e3,
    -0.11573902539959630126141871134e2,
    0.68812326946963000169666922661e1,
    -0.10006050966910838403183860980e1,
    0.77771377980534432092869265740,
    -0.27782057523535084065932004339e1,
    -0.60196695231264120758267380846e2,
    0.84320405506677161018159903784e2,
    0.11992291136182789328035130030e2,
];
#[allow(clippy::excessive_precision)]
const DOP853_D7: [f64; 16] = [
    -0.25693933462703749003312586129e2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.15418974869023643374053993627e3,
    -0.23152937917604549567536039109e3,
    0.35763911791061412378285349910e3,
    0.93405324183624310003907691704e2,
    -0.37458323136451633156875139351e2,
    0.10409964950896230045147246184e3,
    0.29840293426660503123344363579e2,
    -0.43533456590011143754432175058e2,
    0.96324553959188282948394950600e2,
    -0.39177261675615439165231486172e2,
    -0.14972683625798562581422125276e3,
];

/// The four coefficients of the continuous extension that the three extra stages make possible:
/// DOP853's `d4 … d7`.
const DOP853_D: [&[f64]; 4] = [&DOP853_D4, &DOP853_D5, &DOP853_D6, &DOP853_D7];

/// The method of order 8 of Dormand and Prince with Hairer's embedded formulas of orders 5 and 3
/// and the continuous extension of order 7 of DOP853.
pub(super) const DOP853: ButcherTableau = ButcherTableau {
    c: &DOP853_C,
    a: &DOP853_A,
    b: &DOP853_B,
    e: &[&DOP853_E5, &DOP853_E3],
    d: &DOP853_D,
};

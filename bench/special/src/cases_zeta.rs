//! ζ (real and complex), the Hurwitz ζ, Li₂, Li_s, Lambert W, the exponential integrals.
use crate::{Case, Rec, b1, bn, bo, g_x, gs};
use a::math::complex::Complex;
use a::math::special as sp;
use statrs::function as st;

type C = Case<f64>;
type G = fn(&mut crate::Rng) -> Rec<f64>;

pub fn add(v: &mut Vec<C>) {
    // ------------------------------------------------ zeta
    for (p, lo, hi) in [
        ("s in [1.1,3]", 1.1, 3.0),
        ("s in [3,56]", 3.0, 56.0),
        ("s in [-0.5,0.9]", -0.5, 0.9),
        ("s in [-20,-0.5]", -20.0, -0.5),
        ("s in [-170,-20]", -170.0, -20.0),
    ] {
        let mut c = C::new("zeta", "zeta", p, 1, g_x(lo, hi), b1(|x| sp::zeta(x[0])));
        gs!(c, r_gsl_zeta);
        bo!(c, r_boost_zeta);
        c.sp("zeta:x0");
        v.push(c);
    }
    let zc: [(&str, G); 4] = [
        ("s in [0.5,5]+i[-50,50]", |r| [r.uni(0.5, 5.0), r.uni(-50.0, 50.0), 0.0, 0.0]),
        ("s in [-10,0.5]+i[-20,20]", |r| [r.uni(-10.0, 0.5), r.uni(-20.0, 20.0), 0.0, 0.0]),
        ("s = 1/2+i[100,1000]", |r| [0.5, r.uni(100.0, 1000.0), 0.0, 0.0]),
        ("s in [1.1,10]+i[-5,5]", |r| [r.uni(1.1, 10.0), r.uni(-5.0, 5.0), 0.0, 0.0]),
    ];
    for (p, g) in zc {
        let mut c = C::new(
            "complex",
            "zeta c64",
            p,
            2,
            g,
            bn(|x, y| {
                let z = sp::zeta(Complex::new(x[0], x[1]));
                y[0] = z.re;
                y[1] = z.im;
            }),
        );
        c.sp("zeta:c01:c");
        v.push(c);
    }

    let mut c = C::new(
        "zeta",
        "hurwitz_zeta",
        "s in [1.5,10], a in [0.1,10]",
        1,
        |r| [r.uni(1.5, 10.0), r.uni(0.1, 10.0), 0.0, 0.0],
        b1(|x| sp::hurwitz_zeta(x[0], x[1])),
    );
    gs!(c, r_gsl_hzeta);
    c.sp("zeta:x0,x1");
    v.push(c);
    let mut c = C::new(
        "zeta",
        "hurwitz_zeta",
        "s in [1.5,10], a in [10,1e4] log-uniform",
        1,
        |r| [r.uni(1.5, 10.0), r.logu(10.0, 1e4), 0.0, 0.0],
        b1(|x| sp::hurwitz_zeta(x[0], x[1])),
    );
    gs!(c, r_gsl_hzeta);
    c.sp("zeta:x0,x1");
    v.push(c);

    // ------------------------------------------------ dilog (slot 1 = 1 - x for SciPy's spence)
    for (p, lo, hi) in [
        ("x in [-1,1]", -1.0, 1.0),
        ("x in [-100,-1]", -100.0, -1.0),
        ("x in [1,20] (real part)", 1.0, 20.0),
    ] {
        let mut c = C::new(
            "zeta",
            "dilog",
            p,
            1,
            move |r| {
                let x = r.uni(lo, hi);
                [x, 1.0 - x, 0.0, 0.0]
            },
            b1(|x| sp::dilog(x[0])),
        );
        gs!(c, r_gsl_dilog);
        if hi <= 1.0 {
            c.sp("spence:x1");
        }
        v.push(c);
    }

    // ------------------------------------------------ polylog; GSL's Fermi–Dirac integrals for
    // x = -e^mu: Li_s(-e^mu) = -F_{s-1}(mu) (mu precomputed in slot 2)
    let fd = |s: f64, lo: f64, hi: f64| {
        move |r: &mut crate::Rng| {
            let mu = r.uni(lo, hi);
            [s, -(mu.exp()), mu, 0.0]
        }
    };
    let pl = |x: &Rec<f64>| sp::polylog(x[0], x[1]);
    let fdn = "GSL Fermi-Dirac integral at mu = ln(-x): Li_s(-e^mu) = -F_(s-1)(mu)";
    let mut c = C::new("polylog", "polylog", "s=0.5, x=-e^mu, mu in [0,10]", 1, fd(0.5, 0.0, 10.0), b1(pl));
    gs!(c, r_gsl_fd_mhalf, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=1.5, x=-e^mu, mu in [0,10]", 1, fd(1.5, 0.0, 10.0), b1(pl));
    gs!(c, r_gsl_fd_half, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=1.5, x=-e^mu, mu in [-10,0]", 1, fd(1.5, -10.0, 0.0), b1(pl));
    gs!(c, r_gsl_fd_half, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=2, x=-e^mu, mu in [0,10]", 1, fd(2.0, 0.0, 10.0), b1(pl));
    gs!(c, r_gsl_fd_1, fdn);
    #[cfg(has_gsl)]
    c.r("gsl-dilog", "gsl_sf_dilog(x)", crate::cf64!(r_gsl_dilog_x1));
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=2.5, x=-e^mu, mu in [0,10]", 1, fd(2.5, 0.0, 10.0), b1(pl));
    gs!(c, r_gsl_fd_3half, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=3, x=-e^mu, mu in [-10,0] (x in [-1,0))", 1, fd(3.0, -10.0, 0.0), b1(pl));
    gs!(c, r_gsl_fd_2, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=3, x=-e^mu, mu in [0,30]", 1, fd(3.0, 0.0, 30.0), b1(pl));
    gs!(c, r_gsl_fd_2, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s=6, x=-e^mu, mu in [0,30]", 1, fd(6.0, 0.0, 30.0), b1(pl));
    gs!(c, r_gsl_fd_int, fdn);
    v.push(c);
    let mut c = C::new("polylog", "polylog", "s integer in [-8,-2], x=-e^mu, mu in [-5,5]", 1, |r| {
        let s = r.int(-8, -2) as f64;
        let mu = r.uni(-5.0, 5.0);
        [s, -(mu.exp()), mu, 0.0]
    }, b1(pl));
    gs!(c, r_gsl_fd_int, fdn);
    v.push(c);
    let nr: [(&str, G); 5] = [
        ("s=3, x in [-1,1]", |r| [3.0, r.uni(-1.0, 1.0), 0.0, 0.0]),
        ("s=2.5, x in [0,1]", |r| [2.5, r.uni(0.0, 1.0), 0.0, 0.0]),
        ("s in [-10,-0.1], x in [-2,0.9]", |r| [r.uni(-10.0, -0.1), r.uni(-2.0, 0.9), 0.0, 0.0]),
        ("s integer in [-8,-2], x in (0,0.9]", |r| [r.int(-8, -2) as f64, r.uni(1e-3, 0.9), 0.0, 0.0]),
        ("s in [1.5,10], x in [-1e10,-2] log-uniform in |x|", |r| [r.uni(1.5, 10.0), -r.logu(2.0, 1e10), 0.0, 0.0]),
    ];
    for (p, g) in nr {
        v.push(C::new("polylog", "polylog", p, 1, g, b1(pl)));
    }

    // ------------------------------------------------ Lambert W
    let lw: [(&str, G); 2] = [
        ("x in [-0.36,3]", |r| [r.uni(-0.36, 3.0), 0.0, 0.0, 0.0]),
        ("x in [3,1e10] log-uniform", |r| [r.logu(3.0, 1e10), 0.0, 0.0, 0.0]),
    ];
    for (p, g) in lw {
        let mut c = C::new("lambert", "lambert_w0", p, 1, g, b1(|x| sp::lambert_w0(x[0])));
        c.r("special", "", b1(|x| <f64 as special::LambertW>::lambert_w0(x[0])));
        c.r("puruspe", "the lambert_w crate (Fukushima)", b1(|x| puruspe::lambert_w0(x[0])));
        gs!(c, r_gsl_lambert_W0);
        bo!(c, r_boost_lambert_w0);
        c.sp("lambertw:z0,0,1e-8:cr");
        v.push(c);
    }
    let lm: [(&str, G); 2] = [
        ("x in [-0.36,-0.01]", |r| [r.uni(-0.36, -0.01), 0.0, 0.0, 0.0]),
        ("x in -[1e-300,0.01] log-uniform", |r| [-r.logu(1e-300, 0.01), 0.0, 0.0, 0.0]),
    ];
    for (p, g) in lm {
        let mut c = C::new("lambert", "lambert_wm1", p, 1, g, b1(|x| sp::lambert_wm1(x[0])));
        c.r("special", "", b1(|x| <f64 as special::LambertW>::lambert_wm1(x[0])));
        c.r("puruspe", "the lambert_w crate (Fukushima)", b1(|x| puruspe::lambert_wm1(x[0])));
        gs!(c, r_gsl_lambert_Wm1);
        bo!(c, r_boost_lambert_wm1);
        c.sp("lambertw:z0,-1,1e-8:cr");
        v.push(c);
    }

    // ------------------------------------------------ exponential integrals
    for (p, lo, hi) in [("x in [0.01,1]", 0.01, 1.0), ("x in [1,50]", 1.0, 50.0), ("x in [50,700]", 50.0, 700.0)] {
        let mut c = C::new("expint", "expint_e1", p, 1, g_x(lo, hi), b1(|x| sp::expint_e1(x[0])));
        c.r("statrs", "exponential::integral(x, 1)", b1(|x| st::exponential::integral(x[0], 1).unwrap_or(f64::NAN)));
        gs!(c, r_gsl_E1);
        bo!(c, r_boost_e1, "expint(1, x)");
        c.sp("exp1:x0");
        v.push(c);
    }
    for (p, lo, hi) in [
        ("x in [0.01,6]", 0.01, 6.0),
        ("x in [6,50]", 6.0, 50.0),
        ("x in [50,700]", 50.0, 700.0),
        ("x in [-50,-0.01]", -50.0, -0.01),
    ] {
        let mut c = C::new("expint", "expint_ei", p, 1, g_x(lo, hi), b1(|x| sp::expint_ei(x[0])));
        gs!(c, r_gsl_Ei);
        bo!(c, r_boost_ei, "expint(x)");
        c.sp("expi:x0");
        v.push(c);
    }
    for (p, n, lo, hi) in [
        ("n=3, x in [0.1,2]", 3.0, 0.1, 2.0),
        ("n=3, x in [2,20]", 3.0, 2.0, 20.0),
        ("n=20, x in [0.1,2]", 20.0, 0.1, 2.0),
        ("n=20, x in [2,20]", 20.0, 2.0, 20.0),
        ("n=100, x in [0.1,50]", 100.0, 0.1, 50.0),
    ] {
        let mut c = C::new(
            "expint",
            "expint_en",
            p,
            1,
            move |r| [n, r.uni(lo, hi), 0.0, 0.0],
            b1(|x| sp::expint_en(x[0] as u32, x[1])),
        );
        c.r("statrs", "exponential::integral(x, n)", b1(|x| st::exponential::integral(x[1], x[0] as u64).unwrap_or(f64::NAN)));
        gs!(c, r_gsl_En);
        bo!(c, r_boost_en, "expint(n, x)");
        c.sp("expn:i0,x1");
        v.push(c);
    }
}

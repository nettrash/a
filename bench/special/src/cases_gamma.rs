//! The gamma family: Γ, ln Γ, ψ, ψ⁽ⁿ⁾, B, ln B, factorial, binomial, Pochhammer, the
//! incomplete gamma and beta functions and their inverses, and the complex Γ and ln Γ.
use crate::{Case, Cmp, Rec, b1, bn, bo, g_x, gs, sys};
use a::math::complex::Complex;
use a::math::special as sp;
use statrs::function as st;

type C = Case<f64>;

fn r2(a: f64, b: f64, c: f64, d: f64) -> Rec<f64> {
    [a, b, c, d]
}

pub fn add(v: &mut Vec<C>) {
    // ---------------------------------------------------------------- gamma
    for (p, lo, hi) in [
        ("x in [0.5,1.5]", 0.5, 1.5),
        ("x in [1.5,8]", 1.5, 8.0),
        ("x in [8,20]", 8.0, 20.0),
        ("x in [20,170]", 20.0, 170.0),
        ("x in [-8,-0.1]", -8.0, -0.1),
        ("x in [-170,-8]", -170.0, -8.0),
    ] {
        let mut c = C::new("gamma", "gamma", p, 1, g_x(lo, hi), b1(|x| sp::gamma(x[0])));
        sys!(c, r_libm_tgamma);
        c.r("libm", "", b1(|x| libm::tgamma(x[0])));
        c.r("statrs", "", b1(|x| st::gamma::gamma(x[0])));
        c.r("special", "", b1(|x| <f64 as special::Gamma>::gamma(x[0])));
        if lo > 0.0 {
            c.r("puruspe", "", b1(|x| puruspe::gamma(x[0])));
        }
        gs!(c, r_gsl_gamma);
        bo!(c, r_boost_tgamma);
        c.sp("gamma:x0");
        v.push(c);
    }

    // ---------------------------------------------------------------- lgamma
    for (p, lo, hi) in [
        ("x in [0.01,0.5]", 0.01, 0.5),
        ("x in [0.5,1.5]", 0.5, 1.5),
        ("x in [1.5,3]", 1.5, 3.0),
        ("x in [3,8]", 3.0, 8.0),
        ("x in [8,100]", 8.0, 100.0),
        ("x in [100,1e6]", 100.0, 1e6),
        ("x in [-10,-0.6]", -10.0, -0.6),
    ] {
        let mut c = C::new("gamma", "lgamma", p, 1, g_x(lo, hi), b1(|x| sp::lgamma(x[0])));
        sys!(c, r_libm_lgamma);
        c.r("libm", "", b1(|x| libm::lgamma(x[0])));
        if lo > 0.0 {
            c.r("statrs", "", b1(|x| st::gamma::ln_gamma(x[0])));
            c.r("puruspe", "", b1(|x| puruspe::ln_gamma(x[0])));
        }
        c.r("special", "", b1(|x| <f64 as special::Gamma>::ln_gamma(x[0]).0));
        gs!(c, r_gsl_lngamma);
        bo!(c, r_boost_lgamma);
        c.sp("gammaln:x0");
        v.push(c);
    }

    // ---------------------------------------------------------------- lgamma_sign
    for (p, lo, hi) in [("x in [0.5,8]", 0.5, 8.0), ("x in [-10,-0.6]", -10.0, -0.6)] {
        let mut c = C::new(
            "gamma",
            "lgamma_sign",
            p,
            2,
            g_x(lo, hi),
            bn(|x, y| {
                let (l, s) = sp::lgamma_sign(x[0]);
                y[0] = l;
                y[1] = s;
            }),
        );
        sys!(c, r_libm_lgamma_r);
        c.r(
            "libm",
            "",
            bn(|x, y| {
                let (l, s) = libm::lgamma_r(x[0]);
                y[0] = l;
                y[1] = s as f64;
            }),
        );
        c.r(
            "special",
            "",
            bn(|x, y| {
                let (l, s) = <f64 as special::Gamma>::ln_gamma(x[0]);
                y[0] = l;
                y[1] = s as f64;
            }),
        );
        gs!(c, r_gsl_lngamma_sgn);
        bo!(c, r_boost_lgamma_sign);
        v.push(c);
    }

    // ---------------------------------------------------------------- digamma
    for (p, lo, hi) in [
        ("x in [0.01,1]", 0.01, 1.0),
        ("x in [1,8]", 1.0, 8.0),
        ("x in [8,1000]", 8.0, 1000.0),
        ("x in [-10,-0.1]", -10.0, -0.1),
    ] {
        let mut c = C::new("gamma", "digamma", p, 1, g_x(lo, hi), b1(|x| sp::digamma(x[0])));
        c.r("statrs", "", b1(|x| st::gamma::digamma(x[0])));
        c.r("special", "", b1(|x| <f64 as special::Gamma>::digamma(x[0])));
        gs!(c, r_gsl_psi);
        bo!(c, r_boost_digamma);
        c.sp("psi:x0");
        v.push(c);
    }

    // ---------------------------------------------------------------- polygamma
    for (p, n, lo, hi) in [
        ("n=1, x in [0.1,8]", 1.0, 0.1, 8.0),
        ("n=1, x in [8,1000]", 1.0, 8.0, 1000.0),
        ("n=1, x in [-5,-0.1]", 1.0, -5.0, -0.1),
        ("n=3, x in [0.1,20]", 3.0, 0.1, 20.0),
        ("n=30, x in [0.5,20]", 30.0, 0.5, 20.0),
        ("n=3, x in [-5,-0.1]", 3.0, -5.0, -0.1),
    ] {
        let mut c = C::new(
            "gamma",
            "polygamma",
            p,
            1,
            move |r| [n, r.uni(lo, hi), 0.0, 0.0],
            b1(|x| sp::polygamma(x[0] as u32, x[1])),
        );
        if n == 1.0 {
            c.r("special", "trigamma", b1(|x| <f64 as special::Gamma>::trigamma(&x[1])));
            gs!(c, r_gsl_psi_1);
            bo!(c, r_boost_trigamma);
        } else {
            gs!(c, r_gsl_psi_n);
            bo!(c, r_boost_polygamma);
        }
        c.sp("polygamma:i0,x1");
        v.push(c);
    }

    // ---------------------------------------------------------------- beta, lbeta
    type G = fn(&mut crate::Rng) -> Rec<f64>;
    let beta_regions: [(&str, G); 4] = [
        ("a,b in [0.1,10]", |r| r2(r.uni(0.1, 10.0), r.uni(0.1, 10.0), 0.0, 0.0)),
        ("a,b in [10,1e4] log-uniform", |r| r2(r.logu(10.0, 1e4), r.logu(10.0, 1e4), 0.0, 0.0)),
        ("a in [0.1,10], b in [10,1e4]", |r| r2(r.uni(0.1, 10.0), r.logu(10.0, 1e4), 0.0, 0.0)),
        ("a in [-10,-0.1], b in [0.1,10]", |r| r2(r.uni(-10.0, -0.1), r.uni(0.1, 10.0), 0.0, 0.0)),
    ];
    for (p, g) in beta_regions {
        let neg = p.starts_with("a in [-");
        let mut c = C::new("gamma", "beta", p, 1, g, b1(|x| sp::beta(x[0], x[1])));
        if !neg {
            c.r("statrs", "", b1(|x| st::beta::beta(x[0], x[1])));
            c.r("puruspe", "", b1(|x| puruspe::beta(x[0], x[1])));
        }
        gs!(c, r_gsl_beta);
        bo!(c, r_boost_beta);
        c.sp("beta:x0,x1");
        v.push(c);

        let mut c = C::new("gamma", "lbeta", p, 1, g, b1(|x| sp::lbeta(x[0], x[1])));
        if !neg {
            c.r("statrs", "", b1(|x| st::beta::ln_beta(x[0], x[1])));
            c.r("special", "", b1(|x| <f64 as special::Beta>::ln_beta(x[0], x[1])));
        }
        gs!(c, r_gsl_lnbeta);
        c.sp("betaln:x0,x1");
        v.push(c);
    }

    // ---------------------------------------------------------------- factorial, binomial, pochhammer
    let mut c = C::new(
        "gamma",
        "factorial",
        "x integer in [0,170]",
        1,
        |r| [r.int(0, 170) as f64, 0.0, 0.0, 0.0],
        b1(|x| sp::factorial(x[0])),
    );
    c.r("statrs", "u64 argument", b1(|x| st::factorial::factorial(x[0] as u64)));
    c.r("puruspe", "usize argument", b1(|x| puruspe::utils::factorial(x[0] as usize)));
    gs!(c, r_gsl_fact, "unsigned argument");
    bo!(c, r_boost_factorial, "unsigned argument");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "factorial",
        "x real in [-0.9,170]",
        1,
        g_x(-0.9, 170.0),
        b1(|x| sp::factorial(x[0])),
    );
    c.note("the references round x+1 first (up to ~300 ulps off near x = -0.9)");
    sys!(c, r_libm_tgamma1, "tgamma(x+1)");
    c.r("libm", "tgamma(x+1)", b1(|x| libm::tgamma(x[0] + 1.0)));
    gs!(c, r_gsl_gamma1, "gamma(x+1)");
    bo!(c, r_boost_tgamma1, "tgamma(x+1)");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "binomial",
        "n,k integers, 0<=k<=n<=60",
        1,
        |r| {
            let n = r.int(0, 60);
            [n as f64, r.int(0, n) as f64, 0.0, 0.0]
        },
        b1(|x| sp::binomial(x[0], x[1])),
    );
    c.r("statrs", "u64 arguments", b1(|x| st::factorial::binomial(x[0] as u64, x[1] as u64)));
    gs!(c, r_gsl_choose, "unsigned arguments");
    bo!(c, r_boost_binomial, "unsigned arguments");
    c.sp("binom:x0,x1");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "binomial",
        "n integer in [1e3,1e6] log-uniform, k integer in [0,n]",
        1,
        |r| {
            let n = r.logu(1e3, 1e6).floor();
            [n, r.int(0, n as i64) as f64, 0.0, 0.0]
        },
        b1(|x| sp::binomial(x[0], x[1])),
    );
    c.r("statrs", "u64 arguments", b1(|x| st::factorial::binomial(x[0] as u64, x[1] as u64)));
    gs!(c, r_gsl_choose, "unsigned arguments");
    bo!(c, r_boost_binomial, "unsigned arguments");
    c.sp("binom:x0,x1");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "binomial",
        "n,k real in [0,60]",
        1,
        |r| [r.uni(0.0, 60.0), r.uni(0.0, 60.0), 0.0, 0.0],
        b1(|x| sp::binomial(x[0], x[1])),
    );
    c.sp("binom:x0,x1");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "pochhammer",
        "x in [0.1,20], n integer in [0,30]",
        1,
        |r| [r.uni(0.1, 20.0), r.int(0, 30) as f64, 0.0, 0.0],
        b1(|x| sp::pochhammer(x[0], x[1])),
    );
    gs!(c, r_gsl_poch);
    bo!(c, r_boost_rising, "rising_factorial(x, int n)");
    c.sp("poch:x0,x1");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "pochhammer",
        "x in [0.1,20], n real in [-0.9,30]",
        1,
        |r| [r.uni(0.1, 20.0), r.uni(-0.9, 30.0), 0.0, 0.0],
        b1(|x| sp::pochhammer(x[0], x[1])),
    );
    gs!(c, r_gsl_poch);
    bo!(c, r_boost_poch_real, "1/tgamma_delta_ratio(x, n)");
    c.sp("poch:x0,x1");
    v.push(c);

    let mut c = C::new(
        "gamma",
        "pochhammer",
        "x in [-20,0], n integer in [0,30]",
        1,
        |r| [r.uni(-20.0, 0.0), r.int(0, 30) as f64, 0.0, 0.0],
        b1(|x| sp::pochhammer(x[0], x[1])),
    );
    gs!(c, r_gsl_poch);
    bo!(c, r_boost_rising, "rising_factorial(x, int n)");
    c.sp("poch:x0,x1");
    v.push(c);

    // ---------------------------------------------------------------- gamma_p, gamma_q
    let ig_regions: [(&str, G); 9] = [
        ("a in [0.01,1], x in [0,2]", |r| r2(r.uni(0.01, 1.0), r.uni(0.0, 2.0), 0.0, 0.0)),
        ("a in [1,20], x in a*[0.5,1.5]", |r| {
            let a = r.uni(1.0, 20.0);
            r2(a, a * r.uni(0.5, 1.5), 0.0, 0.0)
        }),
        ("a in [20,200], x in a+-3sqrt(a)", |r| {
            let a = r.uni(20.0, 200.0);
            r2(a, a + 3.0 * a.sqrt() * r.uni(-1.0, 1.0), 0.0, 0.0)
        }),
        ("a in [200,1e4] log-uniform, x in a+-3sqrt(a)", |r| {
            let a = r.logu(200.0, 1e4);
            r2(a, a + 3.0 * a.sqrt() * r.uni(-1.0, 1.0), 0.0, 0.0)
        }),
        ("a in [1e4,1e6] log-uniform, x in a+-3sqrt(a)", |r| {
            let a = r.logu(1e4, 1e6);
            r2(a, a + 3.0 * a.sqrt() * r.uni(-1.0, 1.0), 0.0, 0.0)
        }),
        ("a integer in [1,20], x in a*[0.5,1.5]", |r| {
            let a = r.int(1, 20) as f64;
            r2(a, a * r.uni(0.5, 1.5), 0.0, 0.0)
        }),
        ("a half-integer in [0.5,19.5], x in a*[0.5,1.5]", |r| {
            let a = r.int(0, 19) as f64 + 0.5;
            r2(a, a * r.uni(0.5, 1.5), 0.0, 0.0)
        }),
        ("a in [1,100], x in a*([0.1,0.5] or [2,5])", |r| {
            let a = r.uni(1.0, 100.0);
            let f = if r.f() < 0.5 { r.uni(0.1, 0.5) } else { r.uni(2.0, 5.0) };
            r2(a, a * f, 0.0, 0.0)
        }),
        ("a in [0.01,1], x in [2,30]", |r| r2(r.uni(0.01, 1.0), r.uni(2.0, 30.0), 0.0, 0.0)),
    ];
    for (p, g) in ig_regions {
        let mut c = C::new("incgamma", "gamma_p", p, 1, g, b1(|x| sp::gamma_p(x[0], x[1])));
        c.r("statrs", "", b1(|x| st::gamma::gamma_lr(x[0], x[1])));
        c.r("special", "", b1(|x| <f64 as special::Gamma>::inc_gamma(x[1], x[0])));
        c.r("puruspe", "", b1(|x| puruspe::gammp(x[0], x[1])));
        gs!(c, r_gsl_gamma_inc_P);
        bo!(c, r_boost_gamma_p);
        c.sp("gammainc:x0,x1");
        v.push(c);

        let mut c = C::new("incgamma", "gamma_q", p, 1, g, b1(|x| sp::gamma_q(x[0], x[1])));
        c.r("statrs", "", b1(|x| st::gamma::gamma_ur(x[0], x[1])));
        c.r("puruspe", "", b1(|x| puruspe::gammq(x[0], x[1])));
        gs!(c, r_gsl_gamma_inc_Q);
        bo!(c, r_boost_gamma_q);
        c.sp("gammaincc:x0,x1");
        v.push(c);
    }

    // ---------------------------------------------------------------- gamma_p_inv, gamma_q_inv
    let igi_regions: [(&str, G); 5] = [
        ("a in [0.01,1], p in (0,1)", |r| r2(r.uni(0.01, 1.0), r.uni(1e-6, 1.0 - 1e-6), 0.0, 0.0)),
        ("a in [1,20], p in (0,1)", |r| r2(r.uni(1.0, 20.0), r.uni(1e-6, 1.0 - 1e-6), 0.0, 0.0)),
        ("a in [20,1e4] log-uniform, p in (0,1)", |r| {
            r2(r.logu(20.0, 1e4), r.uni(1e-6, 1.0 - 1e-6), 0.0, 0.0)
        }),
        ("a in [1,20], p in [1e-300,1e-10] log-uniform", |r| {
            r2(r.uni(1.0, 20.0), r.logu(1e-300, 1e-10), 0.0, 0.0)
        }),
        ("a in [0.01,1], p in [1e-300,1e-10] log-uniform", |r| {
            r2(r.uni(0.01, 1.0), r.logu(1e-300, 1e-10), 0.0, 0.0)
        }),
    ];
    for (p, g) in igi_regions {
        let mut c = C::new("incgamma", "gamma_p_inv", p, 1, g, b1(|x| sp::gamma_p_inv(x[0], x[1])));
        c.r("puruspe", "", b1(|x| puruspe::invgammp(x[1], x[0])));
        gs!(c, r_gsl_gamma_Pinv, "gsl_cdf_gamma_Pinv");
        bo!(c, r_boost_gamma_p_inv);
        c.sp("gammaincinv:x0,x1");
        v.push(c);

        let pq = p.replace("p in", "q in");
        let mut c = C::new("incgamma", "gamma_q_inv", &pq, 1, g, b1(|x| sp::gamma_q_inv(x[0], x[1])));
        gs!(c, r_gsl_gamma_Qinv, "gsl_cdf_gamma_Qinv");
        bo!(c, r_boost_gamma_q_inv);
        c.sp("gammainccinv:x0,x1");
        v.push(c);
    }

    // ---------------------------------------------------------------- beta_inc and inverses
    fn mean_pm3s(r: &mut crate::Rng, a: f64, b: f64) -> f64 {
        let m = a / (a + b);
        let s = (a * b / ((a + b) * (a + b) * (a + b + 1.0))).sqrt();
        loop {
            let x = m + 3.0 * s * r.uni(-1.0, 1.0);
            if x > 0.0 && x < 1.0 {
                return x;
            }
        }
    }
    let ib_regions: [(&str, G); 6] = [
        ("a,b in [0.1,1], x in (0,1)", |r| r2(r.uni(0.1, 1.0), r.uni(0.1, 1.0), r.f(), 0.0)),
        ("a,b in [1,10], x in (0,1)", |r| r2(r.uni(1.0, 10.0), r.uni(1.0, 10.0), r.f(), 0.0)),
        ("a,b in [10,100], x mean+-3sd", |r| {
            let (a, b) = (r.uni(10.0, 100.0), r.uni(10.0, 100.0));
            r2(a, b, mean_pm3s(r, a, b), 0.0)
        }),
        ("a,b in [100,1e4] log-uniform, x mean+-3sd", |r| {
            let (a, b) = (r.logu(100.0, 1e4), r.logu(100.0, 1e4));
            r2(a, b, mean_pm3s(r, a, b), 0.0)
        }),
        ("a,b in [1e5,1e7] log-uniform, x mean+-3sd", |r| {
            let (a, b) = (r.logu(1e5, 1e7), r.logu(1e5, 1e7));
            r2(a, b, mean_pm3s(r, a, b), 0.0)
        }),
        ("a in [20,1e3], b in [0.1,1], x mean+-3sd", |r| {
            let (a, b) = (r.uni(20.0, 1e3), r.uni(0.1, 1.0));
            r2(a, b, mean_pm3s(r, a, b), 0.0)
        }),
    ];
    for (p, g) in ib_regions {
        let mut c = C::new("incbeta", "beta_inc", p, 1, g, b1(|x| sp::beta_inc(x[0], x[1], x[2])));
        c.r("statrs", "beta_reg", b1(|x| st::beta::beta_reg(x[0], x[1], x[2])));
        c.r(
            "special",
            "inc_beta with ln_beta computed per call",
            b1(|x| {
                let lb = <f64 as special::Beta>::ln_beta(x[0], x[1]);
                <f64 as special::Beta>::inc_beta(x[2], x[0], x[1], lb)
            }),
        );
        c.r("puruspe", "betai", b1(|x| puruspe::betai(x[0], x[1], x[2])));
        gs!(c, r_gsl_beta_inc);
        bo!(c, r_boost_ibeta);
        c.sp("betainc:x0,x1,x2");
        v.push(c);

        let mut c = C::new("incbeta", "beta_inc_c", p, 1, g, b1(|x| sp::beta_inc_c(x[0], x[1], x[2])));
        bo!(c, r_boost_ibetac);
        c.sp("betaincc:x0,x1,x2");
        v.push(c);

        // Inverses: the target is I at the region's x, so that the roots lie where x does.
        let pp = p.replace("x in (0,1)", "p = I_x(a,b), x in (0,1)").replace("x mean", "p = I_x(a,b), x mean");
        let gi = move |r: &mut crate::Rng| {
            let s = g(r);
            [s[0], s[1], sp::beta_inc(s[0], s[1], s[2]), 0.0]
        };
        let mut c = C::new("incbeta", "beta_inc_inv", &pp, 1, gi, b1(|x| sp::beta_inc_inv(x[0], x[1], x[2])));
        c.r("statrs", "inv_beta_reg", b1(|x| st::beta::inv_beta_reg(x[0], x[1], x[2])));
        c.r(
            "special",
            "inv_inc_beta with ln_beta computed per call",
            b1(|x| {
                let lb = <f64 as special::Beta>::ln_beta(x[0], x[1]);
                <f64 as special::Beta>::inv_inc_beta(x[2], x[0], x[1], lb)
            }),
        );
        c.r("puruspe", "invbetai", b1(|x| puruspe::invbetai(x[2], x[0], x[1])));
        // GSL's bisection over its beta_inc takes minutes a call for a, b from 1e5
        let huge = p.starts_with("a,b in [1e5");
        if !huge {
            gs!(c, r_gsl_beta_Pinv, "gsl_cdf_beta_Pinv");
        }
        bo!(c, r_boost_ibeta_inv);
        c.sp("betaincinv:x0,x1,x2");
        v.push(c);

        let pq = p.replace("x in (0,1)", "q = 1-I_x(a,b), x in (0,1)").replace("x mean", "q = 1-I_x(a,b), x mean");
        let gq = move |r: &mut crate::Rng| {
            let s = g(r);
            [s[0], s[1], sp::beta_inc_c(s[0], s[1], s[2]), 0.0]
        };
        let mut c = C::new("incbeta", "beta_inc_c_inv", &pq, 1, gq, b1(|x| sp::beta_inc_c_inv(x[0], x[1], x[2])));
        if !huge {
            gs!(c, r_gsl_beta_Qinv, "gsl_cdf_beta_Qinv");
        }
        bo!(c, r_boost_ibetac_inv);
        c.sp("betainccinv:x0,x1,x2");
        v.push(c);
    }

    // ---------------------------------------------------------------- complex gamma, lgamma
    for (p, rad) in [("|z| < 40 (re,im uniform in the disc)", 40.0), ("|z| in [1e3,1e4]", 1e4)] {
        let g = move |r: &mut crate::Rng| loop {
            let (x, y) = if rad > 100.0 {
                let m = r.uni(1e3, 1e4);
                let t = r.uni(-3.0, 3.0);
                (m * t.cos(), m * t.sin())
            } else {
                (r.uni(-rad, rad), r.uni(-rad, rad))
            };
            if rad > 100.0 || x * x + y * y < rad * rad {
                return [x, y, 0.0, 0.0];
            }
        };
        let mut c = C::new(
            "complex",
            "lgamma c64",
            p,
            2,
            g,
            bn(|x, y| {
                let z = sp::lgamma(Complex::new(x[0], x[1]));
                y[0] = z.re;
                y[1] = z.im;
            }),
        );
        #[cfg(has_gsl)]
        c.rc(
            "gsl",
            "gsl_sf_lngamma_complex_e: imaginary part reduced mod 2pi (compared mod 2pi)",
            Cmp::ImMod2Pi,
            crate::cf64!(r_gsl_lngamma_complex),
        );
        c.sp("loggamma:c01:c");
        v.push(c);

        if rad < 100.0 {
            let mut c = C::new(
                "complex",
                "gamma c64",
                p,
                2,
                g,
                bn(|x, y| {
                    let z = sp::gamma(Complex::new(x[0], x[1]));
                    y[0] = z.re;
                    y[1] = z.im;
                }),
            );
            gs!(c, r_gsl_gamma_complex, "exp of gsl_sf_lngamma_complex_e (exp, cos, sin)");
            c.sp("gamma:c01:c");
            v.push(c);
        }
    }
}

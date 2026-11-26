//! The error functions: erf, erfc, erfcx, their inverses, Dawson, Fresnel, Faddeeva, and the
//! complex erf, erfc, erfcx.
use crate::{Case, b1, bn, bo, g_x, gs, sys};
use a::math::complex::Complex;
use a::math::special as sp;
use statrs::function as st;

type C = Case<f64>;

pub fn add(v: &mut Vec<C>) {
    for (p, lo, hi) in [
        ("x in [-0.5,0.5]", -0.5, 0.5),
        ("x in [0.5,3]", 0.5, 3.0),
        ("x in [-3,3]", -3.0, 3.0),
        ("x in [3,6]", 3.0, 6.0),
    ] {
        let mut c = C::new("erf", "erf", p, 1, g_x(lo, hi), b1(|x| sp::erf(x[0])));
        sys!(c, r_libm_erf);
        c.r("libm", "", b1(|x| libm::erf(x[0])));
        c.r("statrs", "", b1(|x| st::erf::erf(x[0])));
        c.r("special", "", b1(|x| <f64 as special::Error>::error(x[0])));
        c.r("puruspe", "", b1(|x| puruspe::erf(x[0])));
        gs!(c, r_gsl_erf);
        bo!(c, r_boost_erf);
        c.sp("erf:x0");
        v.push(c);
    }
    for (p, lo, hi) in [
        ("x in [-0.5,0.5]", -0.5, 0.5),
        ("x in [0.5,3]", 0.5, 3.0),
        ("x in [-3,3]", -3.0, 3.0),
        ("x in [3,8]", 3.0, 8.0),
        ("x in [8,27]", 8.0, 27.0),
    ] {
        let mut c = C::new("erf", "erfc", p, 1, g_x(lo, hi), b1(|x| sp::erfc(x[0])));
        sys!(c, r_libm_erfc);
        c.r("libm", "", b1(|x| libm::erfc(x[0])));
        c.r("statrs", "", b1(|x| st::erf::erfc(x[0])));
        c.r("special", "", b1(|x| <f64 as special::Error>::compl_error(x[0])));
        c.r("puruspe", "", b1(|x| puruspe::erfc(x[0])));
        gs!(c, r_gsl_erfc);
        bo!(c, r_boost_erfc);
        c.sp("erfc:x0");
        v.push(c);
    }
    for (p, lo, hi) in [
        ("x in [-0.5,0.5]", -0.5, 0.5),
        ("x in [0.5,8]", 0.5, 8.0),
        ("x in [8,100]", 8.0, 100.0),
        ("x in [-5,-0.5]", -5.0, -0.5),
    ] {
        let mut c = C::new("erf", "erfcx", p, 1, g_x(lo, hi), b1(|x| sp::erfcx(x[0])));
        c.r("puruspe", "", b1(|x| puruspe::erfcx(x[0])));
        c.sp("erfcx:x0");
        v.push(c);
    }

    type G = fn(&mut crate::Rng) -> crate::Rec<f64>;
    let inv: [(&str, G); 3] = [
        ("x in [-0.5,0.5]", |r| [r.uni(-0.5, 0.5), 0.0, 0.0, 0.0]),
        ("x in (-1,1)", |r| [r.uni(-1.0, 1.0), 0.0, 0.0, 0.0]),
        ("x = 1-t, t in [1e-15,0.5] log-uniform", |r| [1.0 - r.logu(1e-15, 0.5), 0.0, 0.0, 0.0]),
    ];
    for (p, g) in inv {
        let mut c = C::new("erf", "erf_inv", p, 1, g, b1(|x| sp::erf_inv(x[0])));
        c.r("statrs", "", b1(|x| st::erf::erf_inv(x[0])));
        c.r("special", "", b1(|x| <f64 as special::Error>::inv_error(x[0])));
        c.r("puruspe", "", b1(|x| puruspe::inverf(x[0])));
        bo!(c, r_boost_erf_inv);
        c.sp("erfinv:x0");
        v.push(c);
    }
    let cinv: [(&str, G); 2] = [
        ("x in (0,2)", |r| [r.uni(1e-12, 2.0 - 1e-12), 0.0, 0.0, 0.0]),
        ("x in [1e-300,0.5] log-uniform", |r| [r.logu(1e-300, 0.5), 0.0, 0.0, 0.0]),
    ];
    for (p, g) in cinv {
        let mut c = C::new("erf", "erfc_inv", p, 1, g, b1(|x| sp::erfc_inv(x[0])));
        c.r("statrs", "", b1(|x| st::erf::erfc_inv(x[0])));
        c.r("puruspe", "", b1(|x| puruspe::inverfc(x[0])));
        bo!(c, r_boost_erfc_inv);
        c.sp("erfcinv:x0");
        v.push(c);
    }

    for (p, lo, hi) in [
        ("x in [-0.5,0.5]", -0.5, 0.5),
        ("x in [0.5,8]", 0.5, 8.0),
        ("x in [8,1e4]", 8.0, 1e4),
    ] {
        let mut c = C::new("erf", "dawson", p, 1, g_x(lo, hi), b1(|x| sp::dawson(x[0])));
        c.r("puruspe", "", b1(|x| puruspe::dawson(x[0])));
        gs!(c, r_gsl_dawson);
        c.sp("dawsn:x0");
        v.push(c);
    }

    for (p, lo, hi) in [("x in [0,1]", 0.0, 1.0), ("x in [1,8]", 1.0, 8.0), ("x in [8,1e4]", 8.0, 1e4)] {
        let mut c = C::new(
            "erf",
            "fresnel (S and C)",
            p,
            2,
            g_x(lo, hi),
            bn(|x, y| {
                let (s, c) = sp::fresnel(x[0]);
                y[0] = s;
                y[1] = c;
            }),
        );
        c.sp("fresnel:x0:0,1");
        v.push(c);
    }

    type GC = fn(&mut crate::Rng) -> crate::Rec<f64>;
    let fz: [(&str, GC); 5] = [
        ("|z| < 1, Im z >= 0", |r| loop {
            let (x, y) = (r.uni(-1.0, 1.0), r.uni(0.0, 1.0));
            if x * x + y * y < 1.0 {
                return [x, y, 0.0, 0.0];
            }
        }),
        ("z in [-4,4]x[0,4]", |r| [r.uni(-4.0, 4.0), r.uni(0.0, 4.0), 0.0, 0.0]),
        ("z in [-10,10]x[0,10]", |r| [r.uni(-10.0, 10.0), r.uni(0.0, 10.0), 0.0, 0.0]),
        ("z in [-4,4]x[-4,0]", |r| [r.uni(-4.0, 4.0), r.uni(-4.0, 0.0), 0.0, 0.0]),
        ("z in [10,100]x[0,100]", |r| [r.uni(10.0, 100.0), r.uni(0.0, 100.0), 0.0, 0.0]),
    ];
    for (p, g) in fz {
        let mut c = C::new(
            "complex",
            "faddeeva",
            p,
            2,
            g,
            bn(|x, y| {
                let w = sp::faddeeva(Complex::new(x[0], x[1]));
                y[0] = w.re;
                y[1] = w.im;
            }),
        );
        c.r(
            "puruspe",
            "",
            bn(|x, y| {
                let (a, b) = puruspe::faddeeva(x[0], x[1]);
                y[0] = a;
                y[1] = b;
            }),
        );
        c.sp("wofz:c01:c");
        v.push(c);
    }

    let cz: [(&str, GC); 2] = [
        ("|z| < 1", |r| loop {
            let (x, y) = (r.uni(-1.0, 1.0), r.uni(-1.0, 1.0));
            if x * x + y * y < 1.0 {
                return [x, y, 0.0, 0.0];
            }
        }),
        ("z in [-4,4]^2", |r| [r.uni(-4.0, 4.0), r.uni(-4.0, 4.0), 0.0, 0.0]),
    ];
    for (p, g) in cz {
        let mut c = C::new(
            "complex",
            "erf c64",
            p,
            2,
            g,
            bn(|x, y| {
                let w = sp::erf(Complex::new(x[0], x[1]));
                y[0] = w.re;
                y[1] = w.im;
            }),
        );
        c.sp("erf:c01:c");
        v.push(c);
        let mut c = C::new(
            "complex",
            "erfc c64",
            p,
            2,
            g,
            bn(|x, y| {
                let w = sp::erfc(Complex::new(x[0], x[1]));
                y[0] = w.re;
                y[1] = w.im;
            }),
        );
        c.sp("erfc:c01:c");
        v.push(c);
        let mut c = C::new(
            "complex",
            "erfcx c64",
            p,
            2,
            g,
            bn(|x, y| {
                let w = sp::erfcx(Complex::new(x[0], x[1]));
                y[0] = w.re;
                y[1] = w.im;
            }),
        );
        c.sp("erfcx:c01:c");
        v.push(c);
    }
}

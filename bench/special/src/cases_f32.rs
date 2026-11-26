//! `f32` forms, against the references that have an `f32` form (the system libm, the libm
//! crate, Boost with its default float promotion to double, the lambert_w crate).
use crate::{Case, Rec, b1};
use a::math::special as sp;

type C = Case<f32>;
type G = fn(&mut crate::Rng) -> Rec<f32>;

macro_rules! sysf {
    ($c:ident, $f:ident) => {
        $c.r($crate::SYS, "", $crate::cf32!($f));
    };
}
macro_rules! bof {
    ($c:ident, $f:ident) => {
        #[cfg(has_boost)]
        {
            $c.r("boost", "float promoted to double (Boost's default)", $crate::cf32!($f));
        }
    };
}

fn u(r: &mut crate::Rng, a: f64, b: f64) -> f32 {
    r.uni(a, b) as f32
}

pub fn add(v: &mut Vec<C>) {
    for (p, lo, hi) in [("x in [0.5,1.5]", 0.5, 1.5), ("x in [1.5,8]", 1.5, 8.0), ("x in [-8,-0.1]", -8.0, -0.1)] {
        let mut c = C::new("f32", "gamma f32", p, 1, move |r| [u(r, lo, hi), 0.0, 0.0, 0.0], b1(|x| sp::gamma(x[0])));
        sysf!(c, r_libm_f_tgamma);
        c.r("libm", "", b1(|x| libm::tgammaf(x[0])));
        bof!(c, r_boost_f_tgamma);
        v.push(c);
    }
    for (p, lo, hi) in [("x in [0.5,8]", 0.5, 8.0), ("x in [8,1e6]", 8.0, 1e6), ("x in [-10,-0.6]", -10.0, -0.6)] {
        let mut c = C::new("f32", "lgamma f32", p, 1, move |r| [u(r, lo, hi), 0.0, 0.0, 0.0], b1(|x| sp::lgamma(x[0])));
        sysf!(c, r_libm_f_lgamma);
        c.r("libm", "", b1(|x| libm::lgammaf(x[0])));
        bof!(c, r_boost_f_lgamma);
        v.push(c);
    }
    let mut c = C::new("f32", "erf f32", "x in [-3,3]", 1, |r| [u(r, -3.0, 3.0), 0.0, 0.0, 0.0], b1(|x| sp::erf(x[0])));
    sysf!(c, r_libm_f_erf);
    c.r("libm", "", b1(|x| libm::erff(x[0])));
    bof!(c, r_boost_f_erf);
    v.push(c);
    for (p, lo, hi) in [("x in [-3,3]", -3.0, 3.0), ("x in [3,9]", 3.0, 9.0)] {
        let mut c = C::new("f32", "erfc f32", p, 1, move |r| [u(r, lo, hi), 0.0, 0.0, 0.0], b1(|x| sp::erfc(x[0])));
        sysf!(c, r_libm_f_erfc);
        c.r("libm", "", b1(|x| libm::erfcf(x[0])));
        bof!(c, r_boost_f_erfc);
        v.push(c);
    }

    for nu in [0.0f32, 1.0] {
        let g = move |r: &mut crate::Rng| [nu, u(r, 0.0, 25.0), 0.0, 0.0];
        let gy = move |r: &mut crate::Rng| [nu, u(r, 0.01, 25.0), 0.0, 0.0];
        let (oj, oy) = if nu == 0.0 { ("bessel_j nu=0 f32 (j0f)", "bessel_y nu=0 f32 (y0f)") } else { ("bessel_j nu=1 f32 (j1f)", "bessel_y nu=1 f32 (y1f)") };
        let mut c = C::new("f32", oj, "x in [0,25]", 1, g, b1(|x| sp::bessel_j(x[0], x[1])));
        if nu == 0.0 {
            c.r("libm", "", b1(|x| libm::j0f(x[1])));
        } else {
            c.r("libm", "", b1(|x| libm::j1f(x[1])));
        }
        #[cfg(has_boost)]
        c.r("boost", "float promoted to double (Boost's default)", crate::cf32!(r_boost_f_cyl_j_int));
        v.push(c);
        let mut c = C::new("f32", oy, "x in [0.01,25]", 1, gy, b1(|x| sp::bessel_y(x[0], x[1])));
        if nu == 0.0 {
            c.r("libm", "", b1(|x| libm::y0f(x[1])));
        } else {
            c.r("libm", "", b1(|x| libm::y1f(x[1])));
        }
        #[cfg(has_boost)]
        c.r("boost", "float promoted to double (Boost's default)", crate::cf32!(r_boost_f_cyl_y_int));
        v.push(c);
    }
    let gn: G = |r| [r.int(2, 20) as f32, u(r, 0.1, 50.0), 0.0, 0.0];
    let mut c = C::new("f32", "bessel_j integer n f32 (jnf)", "n in [2,20], x in [0.1,50]", 1, gn, b1(|x| sp::bessel_j(x[0], x[1])));
    c.r("libm", "", b1(|x| libm::jnf(x[0] as i32, x[1])));
    bof!(c, r_boost_f_cyl_j_int);
    v.push(c);
    let mut c = C::new("f32", "bessel_y integer n f32 (ynf)", "n in [2,20], x in [0.1,50]", 1, gn, b1(|x| sp::bessel_y(x[0], x[1])));
    c.r("libm", "", b1(|x| libm::ynf(x[0] as i32, x[1])));
    bof!(c, r_boost_f_cyl_y_int);
    v.push(c);

    let mut c = C::new("f32", "bessel_j f32", "nu in [0,10], x in [2,30]", 1, |r| [u(r, 0.0, 10.0), u(r, 2.0, 30.0), 0.0, 0.0], b1(|x| sp::bessel_j(x[0], x[1])));
    bof!(c, r_boost_f_cyl_j);
    v.push(c);
    let mut c = C::new("f32", "bessel_k f32", "nu in [0,10], x in [0.1,2]", 1, |r| [u(r, 0.0, 10.0), u(r, 0.1, 2.0), 0.0, 0.0], b1(|x| sp::bessel_k(x[0], x[1])));
    bof!(c, r_boost_f_cyl_k);
    v.push(c);
    let mut c = C::new("f32", "digamma f32", "x in [1,8]", 1, |r| [u(r, 1.0, 8.0), 0.0, 0.0, 0.0], b1(|x| sp::digamma(x[0])));
    bof!(c, r_boost_f_digamma);
    v.push(c);
    let mut c = C::new("f32", "erf_inv f32", "x in (-1,1)", 1, |r| [u(r, -0.999999, 0.999999), 0.0, 0.0, 0.0], b1(|x| sp::erf_inv(x[0])));
    bof!(c, r_boost_f_erf_inv);
    v.push(c);
    let mut c = C::new(
        "f32",
        "gamma_p f32",
        "a in [1,20], x in a*[0.5,1.5]",
        1,
        |r| {
            let a = r.uni(1.0, 20.0);
            [a as f32, (a * r.uni(0.5, 1.5)) as f32, 0.0, 0.0]
        },
        b1(|x| sp::gamma_p(x[0], x[1])),
    );
    bof!(c, r_boost_f_gamma_p);
    v.push(c);
    let mut c = C::new(
        "f32",
        "ellip_k f32",
        "m in [0,0.8)",
        1,
        |r| {
            let m = u(r, 0.0, 0.8);
            [m, m.sqrt(), 0.0, 0.0]
        },
        b1(|x| sp::ellip_k(x[0])),
    );
    bof!(c, r_boost_f_ellint_1c);
    v.push(c);
    let mut c = C::new("f32", "lambert_w0 f32", "x in [-0.36,3]", 1, |r| [u(r, -0.36, 3.0), 0.0, 0.0, 0.0], b1(|x| sp::lambert_w0(x[0])));
    c.r("lambert_w", "lambert_w0f (Fukushima, f32 throughout)", b1(|x| lambert_w::lambert_w0f(x[0])));
    bof!(c, r_boost_f_lambert_w0);
    v.push(c);
    let mut c = C::new("f32", "expint_ei f32", "x in [0.01,6]", 1, |r| [u(r, 0.01, 6.0), 0.0, 0.0, 0.0], b1(|x| sp::expint_ei(x[0])));
    bof!(c, r_boost_f_ei);
    v.push(c);
    let mut c = C::new("f32", "zeta f32", "s in [1.1,3]", 1, |r| [u(r, 1.1, 3.0), 0.0, 0.0, 0.0], b1(|x| sp::zeta(x[0])));
    bof!(c, r_boost_f_zeta);
    v.push(c);
    let mut c = C::new("f32", "airy_ai f32", "x in [-10,10]", 1, |r| [u(r, -10.0, 10.0), 0.0, 0.0, 0.0], b1(|x| sp::airy_ai(x[0])));
    bof!(c, r_boost_f_airy_ai);
    v.push(c);
}

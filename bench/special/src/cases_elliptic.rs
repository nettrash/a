//! Elliptic integrals (complete, incomplete, Carlson) and the Jacobi elliptic functions.
//! Boost and GSL take the modulus k = sqrt(m): the generator stores it in a spare slot.
use crate::{Case, Rec, b1, bn, bo, gs};
use a::math::special as sp;

type C = Case<f64>;
type G = fn(&mut crate::Rng) -> Rec<f64>;
const KN: &str = "modulus k = sqrt(m) precomputed";

pub fn add(v: &mut Vec<C>) {
    let km: [(&str, G); 3] = [
        ("m in [0,0.8)", |r| {
            let m = r.uni(0.0, 0.8);
            [m, m.sqrt(), 0.0, 0.0]
        }),
        ("m in [0.8,1)", |r| {
            let m = r.uni(0.8, 1.0);
            [m, m.sqrt(), 0.0, 0.0]
        }),
        ("m in [-10,0)", |r| [r.uni(-10.0, 0.0), f64::NAN, 0.0, 0.0]),
    ];
    for (p, g) in km {
        let neg = p.starts_with("m in [-");
        let mut c = C::new("elliptic", "ellip_k", p, 1, g, b1(|x| sp::ellip_k(x[0])));
        if !neg {
            gs!(c, r_gsl_Kcomp, KN);
            bo!(c, r_boost_ellint_1c, KN);
        }
        c.sp("ellipk:x0");
        v.push(c);
        let mut c = C::new("elliptic", "ellip_e", p, 1, g, b1(|x| sp::ellip_e(x[0])));
        if !neg {
            gs!(c, r_gsl_Ecomp, KN);
            bo!(c, r_boost_ellint_2c, KN);
        }
        c.sp("ellipe:x0");
        v.push(c);
    }

    let pm: [(&str, G); 2] = [
        ("n in (-5,0.9), m in [0,0.99)", |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(-5.0, 0.9), m, m.sqrt(), 0.0]
        }),
        ("n in (1.1,10), m in [0,0.99) (principal value)", |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(1.1, 10.0), m, m.sqrt(), 0.0]
        }),
    ];
    for (p, g) in pm {
        let mut c = C::new("elliptic", "ellip_pi", p, 1, g, b1(|x| sp::ellip_pi(x[0], x[1])));
        gs!(c, r_gsl_Pcomp, "modulus k precomputed; GSL's n has the opposite sign");
        bo!(c, r_boost_ellint_3c, KN);
        v.push(c);
    }

    let fm: [(&str, G); 2] = [
        ("phi in [-pi/2,pi/2], m in [0,0.99)", |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2), m, m.sqrt(), 0.0]
        }),
        ("phi in [-20,20], m in [0,0.99)", |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(-20.0, 20.0), m, m.sqrt(), 0.0]
        }),
    ];
    for (p, g) in fm {
        let mut c = C::new("elliptic", "ellip_f", p, 1, g, b1(|x| sp::ellip_f(x[0], x[1])));
        gs!(c, r_gsl_F, KN);
        bo!(c, r_boost_ellint_1i, KN);
        c.sp("ellipkinc:x0,x1");
        v.push(c);
        let mut c = C::new("elliptic", "ellip_e_inc", p, 1, g, b1(|x| sp::ellip_e_inc(x[0], x[1])));
        gs!(c, r_gsl_E, KN);
        bo!(c, r_boost_ellint_2i, KN);
        c.sp("ellipeinc:x0,x1");
        v.push(c);
    }

    let mut c = C::new(
        "elliptic",
        "ellip_pi_inc",
        "n in (-5,0.9), |phi| < 1.5, m in [0,0.99)",
        1,
        |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(-5.0, 0.9), r.uni(-1.5, 1.5), m, m.sqrt()]
        },
        b1(|x| sp::ellip_pi_inc(x[0], x[1], x[2])),
    );
    gs!(c, r_gsl_P, "modulus k precomputed; GSL's n has the opposite sign");
    bo!(c, r_boost_ellint_3i, KN);
    v.push(c);

    let p = "args in [0.01,10]";
    let mut c = C::new("carlson", "carlson_rf", p, 1, |r| [r.uni(0.01, 10.0), r.uni(0.01, 10.0), r.uni(0.01, 10.0), 0.0], b1(|x| sp::carlson_rf(x[0], x[1], x[2])));
    gs!(c, r_gsl_RF);
    bo!(c, r_boost_rf);
    c.sp("elliprf:x0,x1,x2");
    v.push(c);
    let mut c = C::new("carlson", "carlson_rd", p, 1, |r| [r.uni(0.01, 10.0), r.uni(0.01, 10.0), r.uni(0.01, 10.0), 0.0], b1(|x| sp::carlson_rd(x[0], x[1], x[2])));
    gs!(c, r_gsl_RD);
    bo!(c, r_boost_rd);
    c.sp("elliprd:x0,x1,x2");
    v.push(c);
    let mut c = C::new("carlson", "carlson_rj", p, 1, |r| [r.uni(0.01, 10.0), r.uni(0.01, 10.0), r.uni(0.01, 10.0), r.uni(0.01, 10.0)], b1(|x| sp::carlson_rj(x[0], x[1], x[2], x[3])));
    gs!(c, r_gsl_RJ);
    bo!(c, r_boost_rj);
    c.sp("elliprj:x0,x1,x2,x3");
    v.push(c);
    let mut c = C::new("carlson", "carlson_rc", p, 1, |r| [r.uni(0.01, 10.0), r.uni(0.01, 10.0), 0.0, 0.0], b1(|x| sp::carlson_rc(x[0], x[1])));
    gs!(c, r_gsl_RC);
    bo!(c, r_boost_rc);
    c.sp("elliprc:x0,x1");
    v.push(c);

    let jm: [(&str, G); 3] = [
        ("u in [-4,4], m in [0,0.99)", |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(-4.0, 4.0), m, m.sqrt(), 0.0]
        }),
        ("u in [-100,100], m in [0,0.99)", |r| {
            let m = r.uni(0.0, 0.99);
            [r.uni(-100.0, 100.0), m, m.sqrt(), 0.0]
        }),
        ("u in [-4,4], m in [0.99,1)", |r| {
            let m = r.uni(0.99, 1.0);
            [r.uni(-4.0, 4.0), m, m.sqrt(), 0.0]
        }),
    ];
    for (p, g) in jm {
        let mut c = C::new(
            "elliptic",
            "jacobi_elliptic (sn,cn,dn)",
            p,
            3,
            g,
            bn(|x, y| {
                let (s, c, d) = sp::jacobi_elliptic(x[0], x[1]);
                y[0] = s;
                y[1] = c;
                y[2] = d;
            }),
        );
        gs!(c, r_gsl_elljac);
        bo!(c, r_boost_jacobi_elliptic, KN);
        c.sp("ellipj:x0,x1:0,1,2");
        v.push(c);
    }
}

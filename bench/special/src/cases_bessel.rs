//! Bessel functions of integer and real order, spherical Bessel functions, Airy functions.
//! Records: order in slot 0, argument in slot 1.
use crate::{Case, Rec, b1, bo, gs, sys};
use a::math::special as sp;

type C = Case<f64>;
type G = fn(&mut crate::Rng) -> Rec<f64>;

pub fn add(v: &mut Vec<C>) {
    // ------------------------------------------------ orders 0 and 1
    for (p, lo, hi) in [("x in [0,25]", 0.0, 25.0), ("x in [25,1e4]", 25.0, 1e4)] {
        for nu in [0.0, 1.0] {
            let op = if nu == 0.0 { "bessel_j nu=0 (j0)" } else { "bessel_j nu=1 (j1)" };
            let mut c = C::new("bessel01", op, p, 1, move |r| [nu, r.uni(lo, hi), 0.0, 0.0], b1(|x| sp::bessel_j(x[0], x[1])));
            if nu == 0.0 {
                sys!(c, r_libm_j0);
                c.r("libm", "", b1(|x| libm::j0(x[1])));
                gs!(c, r_gsl_J0);
                c.sp("j0:x1");
            } else {
                sys!(c, r_libm_j1);
                c.r("libm", "", b1(|x| libm::j1(x[1])));
                gs!(c, r_gsl_J1);
                c.sp("j1:x1");
            }
            c.r("puruspe", "Jn", b1(|x| puruspe::Jn(x[0] as u32, x[1])));
            bo!(c, r_boost_cyl_j_int, "cyl_bessel_j(int, x)");
            v.push(c);

            let lo_y = lo.max(0.01);
            let op = if nu == 0.0 { "bessel_y nu=0 (y0)" } else { "bessel_y nu=1 (y1)" };
            let py = if lo == 0.0 { "x in [0.01,25]" } else { p };
            let mut c = C::new("bessel01", op, py, 1, move |r| [nu, r.uni(lo_y, hi), 0.0, 0.0], b1(|x| sp::bessel_y(x[0], x[1])));
            if nu == 0.0 {
                sys!(c, r_libm_y0);
                c.r("libm", "", b1(|x| libm::y0(x[1])));
                gs!(c, r_gsl_Y0);
                c.sp("y0:x1");
            } else {
                sys!(c, r_libm_y1);
                c.r("libm", "", b1(|x| libm::y1(x[1])));
                gs!(c, r_gsl_Y1);
                c.sp("y1:x1");
            }
            c.r("puruspe", "Yn", b1(|x| puruspe::Yn(x[0] as u32, x[1])));
            bo!(c, r_boost_cyl_y_int, "cyl_neumann(int, x)");
            v.push(c);
        }
    }
    for nu in [0.0, 1.0] {
        let op = if nu == 0.0 { "bessel_i nu=0 (i0)" } else { "bessel_i nu=1 (i1)" };
        let mut c = C::new("bessel01", op, "x in [0,50]", 1, move |r| [nu, r.uni(0.0, 50.0), 0.0, 0.0], b1(|x| sp::bessel_i(x[0], x[1])));
        c.r("puruspe", "In", b1(|x| puruspe::In(x[0] as u32, x[1])));
        if nu == 0.0 {
            gs!(c, r_gsl_I0);
            c.sp("i0:x1");
        } else {
            gs!(c, r_gsl_I1);
            c.sp("i1:x1");
        }
        bo!(c, r_boost_cyl_i_int, "cyl_bessel_i(int, x)");
        v.push(c);

        let op = if nu == 0.0 { "bessel_k nu=0 (k0)" } else { "bessel_k nu=1 (k1)" };
        let mut c = C::new("bessel01", op, "x in [0.01,50]", 1, move |r| [nu, r.uni(0.01, 50.0), 0.0, 0.0], b1(|x| sp::bessel_k(x[0], x[1])));
        c.r("puruspe", "Kn", b1(|x| puruspe::Kn(x[0] as u32, x[1])));
        if nu == 0.0 {
            gs!(c, r_gsl_K0);
            c.sp("k0:x1");
        } else {
            gs!(c, r_gsl_K1);
            c.sp("k1:x1");
        }
        bo!(c, r_boost_cyl_k_int, "cyl_bessel_k(int, x)");
        v.push(c);
    }

    // ------------------------------------------------ integer orders
    let gint: G = |r| [r.int(2, 20) as f64, r.uni(0.1, 50.0), 0.0, 0.0];
    let p = "n integer in [2,20], x in [0.1,50]";
    let mut c = C::new("bessel_n", "bessel_j integer n (jn)", p, 1, gint, b1(|x| sp::bessel_j(x[0], x[1])));
    sys!(c, r_libm_jn);
    c.r("libm", "", b1(|x| libm::jn(x[0] as i32, x[1])));
    c.r("puruspe", "Jn", b1(|x| puruspe::Jn(x[0] as u32, x[1])));
    gs!(c, r_gsl_Jn);
    bo!(c, r_boost_cyl_j_int, "cyl_bessel_j(int, x)");
    c.sp("jv:x0,x1");
    v.push(c);
    let mut c = C::new("bessel_n", "bessel_y integer n (yn)", p, 1, gint, b1(|x| sp::bessel_y(x[0], x[1])));
    sys!(c, r_libm_yn);
    c.r("libm", "", b1(|x| libm::yn(x[0] as i32, x[1])));
    c.r("puruspe", "Yn", b1(|x| puruspe::Yn(x[0] as u32, x[1])));
    gs!(c, r_gsl_Yn);
    bo!(c, r_boost_cyl_y_int, "cyl_neumann(int, x)");
    c.sp("yn:i0,x1");
    v.push(c);
    let mut c = C::new("bessel_n", "bessel_i integer n (in)", p, 1, gint, b1(|x| sp::bessel_i(x[0], x[1])));
    c.r("puruspe", "In", b1(|x| puruspe::In(x[0] as u32, x[1])));
    gs!(c, r_gsl_In);
    bo!(c, r_boost_cyl_i_int, "cyl_bessel_i(int, x)");
    c.sp("iv:x0,x1");
    v.push(c);
    let mut c = C::new("bessel_n", "bessel_k integer n (kn)", p, 1, gint, b1(|x| sp::bessel_k(x[0], x[1])));
    c.r("puruspe", "Kn", b1(|x| puruspe::Kn(x[0] as u32, x[1])));
    gs!(c, r_gsl_Kn);
    bo!(c, r_boost_cyl_k_int, "cyl_bessel_k(int, x)");
    c.sp("kn:i0,x1");
    v.push(c);

    // ------------------------------------------------ real orders
    let r1: (&str, G) = ("nu in [0,10], x in [0.1,2]", |r| [r.uni(0.0, 10.0), r.uni(0.1, 2.0), 0.0, 0.0]);
    let r2: (&str, G) = ("nu in [0,10], x in [2,30]", |r| [r.uni(0.0, 10.0), r.uni(2.0, 30.0), 0.0, 0.0]);
    let r3: (&str, G) = ("nu in [0,3], x in [30,1e3]", |r| [r.uni(0.0, 3.0), r.uni(30.0, 1e3), 0.0, 0.0]);
    let r4: (&str, G) = ("nu in [10,100], x in [1,200]", |r| [r.uni(10.0, 100.0), r.uni(1.0, 200.0), 0.0, 0.0]);
    let r5: (&str, G) = ("nu in [50,300], x in nu*[0.95,1.05] (turning point)", |r| {
        let n = r.uni(50.0, 300.0);
        [n, n * r.uni(0.95, 1.05), 0.0, 0.0]
    });
    let r6: (&str, G) = ("nu in [-30,0] non-integer, x in [0.1,60]", |r| loop {
        let n = r.uni(-30.0, 0.0);
        if (n - n.round()).abs() > 1e-3 {
            return [n, r.uni(0.1, 60.0), 0.0, 0.0];
        }
    });
    let r7u: (&str, G) = ("nu in [50,300], x in nu*[0.5,2] (Debye)", |r| {
        let n = r.uni(50.0, 300.0);
        [n, n * r.uni(0.5, 2.0), 0.0, 0.0]
    });
    let r7s: (&str, G) = ("nu in [50,1000], x in [1,2000]", |r| [r.uni(50.0, 1000.0), r.uni(1.0, 2000.0), 0.0, 0.0]);

    for (p, g) in [r1, r2, r3, r4, r5, r6] {
        let neg = p.starts_with("nu in [-");
        let mut c = C::new("bessel_v", "bessel_j", p, 1, g, b1(|x| sp::bessel_j(x[0], x[1])));
        if !neg {
            c.r("puruspe", "Jnu_Ynu: computes J and Y together", b1(|x| puruspe::Jnu_Ynu(x[0], x[1]).0));
            gs!(c, r_gsl_Jnu);
        }
        bo!(c, r_boost_cyl_j);
        c.sp("jv:x0,x1");
        v.push(c);
        let mut c = C::new("bessel_v", "bessel_y", p, 1, g, b1(|x| sp::bessel_y(x[0], x[1])));
        if !neg {
            c.r("puruspe", "Jnu_Ynu: computes J and Y together", b1(|x| puruspe::Jnu_Ynu(x[0], x[1]).1));
            gs!(c, r_gsl_Ynu);
        }
        bo!(c, r_boost_cyl_y);
        c.sp("yv:x0,x1");
        v.push(c);
    }
    for (p, g) in [r1, r2, r3, r4, r6, r7u] {
        let neg = p.starts_with("nu in [-");
        let mut c = C::new("bessel_v", "bessel_i", p, 1, g, b1(|x| sp::bessel_i(x[0], x[1])));
        if !neg {
            c.r("puruspe", "Inu_Knu: computes I and K together", b1(|x| puruspe::Inu_Knu(x[0], x[1]).0));
            gs!(c, r_gsl_Inu);
        }
        bo!(c, r_boost_cyl_i);
        c.sp("iv:x0,x1");
        v.push(c);
    }
    for (p, g) in [r1, r2, r3, r4, r7u] {
        let mut c = C::new("bessel_v", "bessel_k", p, 1, g, b1(|x| sp::bessel_k(x[0], x[1])));
        c.r("puruspe", "Inu_Knu: computes I and K together", b1(|x| puruspe::Inu_Knu(x[0], x[1]).1));
        gs!(c, r_gsl_Knu);
        bo!(c, r_boost_cyl_k);
        c.sp("kv:x0,x1");
        v.push(c);
    }
    for (p, g) in [r1, r2, r4, r6, r7s] {
        let neg = p.starts_with("nu in [-");
        let mut c = C::new("bessel_v", "bessel_i_scaled", p, 1, g, b1(|x| sp::bessel_i_scaled(x[0], x[1])));
        if !neg {
            gs!(c, r_gsl_Inu_scaled);
        }
        c.sp("ive:x0,x1");
        v.push(c);
    }
    for (p, g) in [r1, r2, r3, r4, r7s] {
        let mut c = C::new("bessel_v", "bessel_k_scaled", p, 1, g, b1(|x| sp::bessel_k_scaled(x[0], x[1])));
        gs!(c, r_gsl_Knu_scaled);
        c.sp("kve:x0,x1");
        v.push(c);
    }

    // ------------------------------------------------ spherical
    let s1: (&str, G) = ("n in [0,10], x in [0.1,50]", |r| [r.int(0, 10) as f64, r.uni(0.1, 50.0), 0.0, 0.0]);
    let s2: (&str, G) = ("n in [0,60], x in [0.1,200]", |r| [r.int(0, 60) as f64, r.uni(0.1, 200.0), 0.0, 0.0]);
    let s3: (&str, G) = ("n in [10,60], x in [0.1,n/2]", |r| {
        let n = r.int(10, 60) as f64;
        [n, r.uni(0.1, n / 2.0), 0.0, 0.0]
    });
    for (p, g) in [s1, s2, s3] {
        let mut c = C::new("sph_bessel", "sph_bessel_j", p, 1, g, b1(|x| sp::sph_bessel_j(x[0] as u32, x[1])));
        gs!(c, r_gsl_jl);
        bo!(c, r_boost_sph_j);
        c.sp("spherical_jn:i0,x1");
        v.push(c);
        let mut c = C::new("sph_bessel", "sph_bessel_y", p, 1, g, b1(|x| sp::sph_bessel_y(x[0] as u32, x[1])));
        gs!(c, r_gsl_yl);
        bo!(c, r_boost_sph_y);
        c.sp("spherical_yn:i0,x1");
        v.push(c);
    }
    for (p, g) in [s1, s2] {
        let mut c = C::new("sph_bessel", "sph_bessel_i", p, 1, g, b1(|x| sp::sph_bessel_i(x[0] as u32, x[1])));
        gs!(c, r_gsl_il, "gsl_sf_bessel_il_scaled * exp(|x|)");
        c.sp("spherical_in:i0,x1");
        v.push(c);
        let mut c = C::new("sph_bessel", "sph_bessel_k", p, 1, g, b1(|x| sp::sph_bessel_k(x[0] as u32, x[1])));
        gs!(c, r_gsl_kl, "gsl_sf_bessel_kl_scaled * exp(-x)");
        c.sp("spherical_kn:i0,x1");
        v.push(c);
    }

    // ------------------------------------------------ Airy
    for (p, lo, hi) in [("x in [-10,10]", -10.0, 10.0), ("x in [-1e3,-10]", -1e3, -10.0), ("x in [10,100]", 10.0, 100.0)] {
        let g = move |r: &mut crate::Rng| [r.uni(lo, hi), 0.0, 0.0, 0.0];
        let mut c = C::new("airy", "airy_ai", p, 1, g, b1(|x| sp::airy_ai(x[0])));
        gs!(c, r_gsl_airy_Ai);
        bo!(c, r_boost_airy_ai);
        c.sp("airy:x0:0");
        v.push(c);
        let mut c = C::new("airy", "airy_ai_prime", p, 1, g, b1(|x| sp::airy_ai_prime(x[0])));
        gs!(c, r_gsl_airy_Ai_deriv);
        bo!(c, r_boost_airy_ai_prime);
        c.sp("airy:x0:1");
        v.push(c);
        let mut c = C::new("airy", "airy_bi", p, 1, g, b1(|x| sp::airy_bi(x[0])));
        gs!(c, r_gsl_airy_Bi);
        bo!(c, r_boost_airy_bi);
        c.sp("airy:x0:2");
        v.push(c);
        let mut c = C::new("airy", "airy_bi_prime", p, 1, g, b1(|x| sp::airy_bi_prime(x[0])));
        gs!(c, r_gsl_airy_Bi_deriv);
        bo!(c, r_boost_airy_bi_prime);
        c.sp("airy:x0:3");
        v.push(c);
    }
}

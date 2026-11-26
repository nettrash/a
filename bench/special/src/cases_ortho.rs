//! Orthogonal polynomials at fixed degrees (degree in slot 0).
use crate::{Case, b1, bo, gs};
use a::math::special as sp;

type C = Case<f64>;

pub fn add(v: &mut Vec<C>) {
    for n in [4u32, 20, 100, 1000] {
        let nf = n as f64;
        let big = n == 1000;
        let pm = format!("n={n}, x in [-1,1]");

        let mut c = C::new("orthopoly", "legendre_p", &pm, 1, move |r| [nf, r.uni(-1.0, 1.0), 0.0, 0.0], b1(|x| sp::legendre_p(x[0] as u32, x[1])));
        gs!(c, r_gsl_legendre_Pl);
        bo!(c, r_boost_legendre_p);
        c.sp("eval_legendre:i0,x1");
        v.push(c);
        for (op, t) in [("chebyshev_t", true), ("chebyshev_u", false)] {
            let f = if t { b1(|x| sp::chebyshev_t(x[0] as u32, x[1])) } else { b1(|x| sp::chebyshev_u(x[0] as u32, x[1])) };
            let mut c = C::new("orthopoly", op, &pm, 1, move |r| [nf, r.uni(-1.0, 1.0), 0.0, 0.0], f);
            if t {
                bo!(c, r_boost_chebyshev_t);
                c.sp("eval_chebyt:i0,x1");
            } else {
                bo!(c, r_boost_chebyshev_u);
                c.sp("eval_chebyu:i0,x1");
            }
            v.push(c);
        }
        if big {
            continue;
        }
        let mut c = C::new(
            "orthopoly",
            "assoc_legendre_p (m=n/2)",
            &pm,
            1,
            move |r| [nf, (nf / 2.0).floor(), r.uni(-1.0, 1.0), 0.0],
            b1(|x| sp::assoc_legendre_p(x[0] as u32, x[1] as u32, x[2])),
        );
        gs!(c, r_gsl_legendre_Plm);
        bo!(c, r_boost_assoc_legendre_p);
        c.sp("lpmv:x1,x0,x2");
        v.push(c);
        let mut c = C::new(
            "orthopoly",
            "jacobi_p (a=0.5, b=1.5)",
            &pm,
            1,
            move |r| [nf, 0.5, 1.5, r.uni(-1.0, 1.0)],
            b1(|x| sp::jacobi_p(x[0] as u32, x[1], x[2], x[3])),
        );
        bo!(c, r_boost_jacobi_p);
        c.sp("eval_jacobi:i0,x1,x2,x3");
        v.push(c);
        let mut c = C::new(
            "orthopoly",
            "gegenbauer_c (a=1.5)",
            &pm,
            1,
            move |r| [nf, 1.5, r.uni(-1.0, 1.0), 0.0],
            b1(|x| sp::gegenbauer_c(x[0] as u32, x[1], x[2])),
        );
        gs!(c, r_gsl_gegenpoly);
        bo!(c, r_boost_gegenbauer);
        c.sp("eval_gegenbauer:i0,x1,x2");
        v.push(c);

        let h = (2.0 * nf + 1.0).sqrt();
        let ph = format!("n={n}, |x| < sqrt(2n+1)");
        let mut c = C::new("orthopoly", "hermite_h", &ph, 1, move |r| [nf, r.uni(-h, h), 0.0, 0.0], b1(|x| sp::hermite_h(x[0] as u32, x[1])));
        gs!(c, r_gsl_hermite);
        bo!(c, r_boost_hermite);
        c.sp("eval_hermite:i0,x1");
        v.push(c);
        let mut c = C::new("orthopoly", "hermite_he", &ph, 1, move |r| [nf, r.uni(-h, h), 0.0, 0.0], b1(|x| sp::hermite_he(x[0] as u32, x[1])));
        gs!(c, r_gsl_hermite_prob);
        c.sp("eval_hermitenorm:i0,x1");
        v.push(c);

        let l = 4.0 * nf + 2.0;
        let pl = format!("n={n}, x in [0,4n+2]");
        let mut c = C::new("orthopoly", "laguerre_l", &pl, 1, move |r| [nf, r.uni(0.0, l), 0.0, 0.0], b1(|x| sp::laguerre_l(x[0] as u32, x[1])));
        gs!(c, r_gsl_laguerre, "gsl_sf_laguerre_n(n, 0, x)");
        bo!(c, r_boost_laguerre);
        c.sp("eval_laguerre:i0,x1");
        v.push(c);
        let mut c = C::new(
            "orthopoly",
            "assoc_laguerre_l (alpha=2)",
            &pl,
            1,
            move |r| [nf, 2.0, r.uni(0.0, l), 0.0],
            b1(|x| sp::assoc_laguerre_l(x[0] as u32, x[1], x[2])),
        );
        gs!(c, r_gsl_assoc_laguerre);
        bo!(c, r_boost_assoc_laguerre, "laguerre(n, m, x), integer m");
        c.sp("eval_genlaguerre:i0,x1,x2");
        v.push(c);
    }
}

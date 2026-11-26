/* GSL references. Batch loops over the harness's input records: double[4] in, double[4] out.
 * The error handler is switched off (r_gsl_init); the plain forms return NaN on an error. */
#include <math.h>
#include <stddef.h>

#include <gsl/gsl_cdf.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf.h>

typedef double V[4];
#define D1(name, expr) \
  void name(const V *in, V *out, size_t n) { \
    for (size_t i = 0; i < n; ++i) { const double *x = in[i]; out[i][0] = (expr); } }
#define DN(name, ...) \
  void name(const V *in, V *out, size_t n) { \
    for (size_t i = 0; i < n; ++i) { const double *x = in[i]; double *y = out[i]; __VA_ARGS__; } }

#define PD GSL_PREC_DOUBLE

void r_gsl_init(void) { gsl_set_error_handler_off(); }

/* gamma family */
D1(r_gsl_gamma, gsl_sf_gamma(x[0]))
D1(r_gsl_gamma1, gsl_sf_gamma(x[0] + 1.0))
D1(r_gsl_lngamma, gsl_sf_lngamma(x[0]))
DN(r_gsl_lngamma_sgn, { gsl_sf_result r; double s; gsl_sf_lngamma_sgn_e(x[0], &r, &s); y[0] = r.val; y[1] = s; })
D1(r_gsl_psi, gsl_sf_psi(x[0]))
D1(r_gsl_psi_1, gsl_sf_psi_1(x[1]))
D1(r_gsl_psi_n, gsl_sf_psi_n((int)x[0], x[1]))
D1(r_gsl_beta, gsl_sf_beta(x[0], x[1]))
D1(r_gsl_lnbeta, gsl_sf_lnbeta(x[0], x[1]))
D1(r_gsl_fact, gsl_sf_fact((unsigned)x[0]))
D1(r_gsl_choose, gsl_sf_choose((unsigned)x[0], (unsigned)x[1]))
D1(r_gsl_poch, gsl_sf_poch(x[0], x[1]))
D1(r_gsl_gamma_inc_P, gsl_sf_gamma_inc_P(x[0], x[1]))
D1(r_gsl_gamma_inc_Q, gsl_sf_gamma_inc_Q(x[0], x[1]))
D1(r_gsl_gamma_Pinv, gsl_cdf_gamma_Pinv(x[1], x[0], 1.0))
D1(r_gsl_gamma_Qinv, gsl_cdf_gamma_Qinv(x[1], x[0], 1.0))
D1(r_gsl_beta_inc, gsl_sf_beta_inc(x[0], x[1], x[2]))
D1(r_gsl_beta_Pinv, gsl_cdf_beta_Pinv(x[2], x[0], x[1]))
D1(r_gsl_beta_Qinv, gsl_cdf_beta_Qinv(x[2], x[0], x[1]))
DN(r_gsl_lngamma_complex, {
  gsl_sf_result lnr, arg;
  gsl_sf_lngamma_complex_e(x[0], x[1], &lnr, &arg);
  y[0] = lnr.val;
  y[1] = arg.val;
})
DN(r_gsl_gamma_complex, {
  gsl_sf_result lnr, arg;
  gsl_sf_lngamma_complex_e(x[0], x[1], &lnr, &arg);
  double m = exp(lnr.val);
  y[0] = m * cos(arg.val);
  y[1] = m * sin(arg.val);
})

/* error functions */
D1(r_gsl_erf, gsl_sf_erf(x[0]))
D1(r_gsl_erfc, gsl_sf_erfc(x[0]))
D1(r_gsl_dawson, gsl_sf_dawson(x[0]))

/* Bessel: order in x[0], argument in x[1] */
D1(r_gsl_J0, gsl_sf_bessel_J0(x[1]))
D1(r_gsl_J1, gsl_sf_bessel_J1(x[1]))
D1(r_gsl_Y0, gsl_sf_bessel_Y0(x[1]))
D1(r_gsl_Y1, gsl_sf_bessel_Y1(x[1]))
D1(r_gsl_I0, gsl_sf_bessel_I0(x[1]))
D1(r_gsl_I1, gsl_sf_bessel_I1(x[1]))
D1(r_gsl_K0, gsl_sf_bessel_K0(x[1]))
D1(r_gsl_K1, gsl_sf_bessel_K1(x[1]))
D1(r_gsl_Jn, gsl_sf_bessel_Jn((int)x[0], x[1]))
D1(r_gsl_Yn, gsl_sf_bessel_Yn((int)x[0], x[1]))
D1(r_gsl_In, gsl_sf_bessel_In((int)x[0], x[1]))
D1(r_gsl_Kn, gsl_sf_bessel_Kn((int)x[0], x[1]))
D1(r_gsl_Jnu, gsl_sf_bessel_Jnu(x[0], x[1]))
D1(r_gsl_Ynu, gsl_sf_bessel_Ynu(x[0], x[1]))
D1(r_gsl_Inu, gsl_sf_bessel_Inu(x[0], x[1]))
D1(r_gsl_Knu, gsl_sf_bessel_Knu(x[0], x[1]))
D1(r_gsl_Inu_scaled, gsl_sf_bessel_Inu_scaled(x[0], x[1]))
D1(r_gsl_Knu_scaled, gsl_sf_bessel_Knu_scaled(x[0], x[1]))
D1(r_gsl_jl, gsl_sf_bessel_jl((int)x[0], x[1]))
D1(r_gsl_yl, gsl_sf_bessel_yl((int)x[0], x[1]))
D1(r_gsl_il, gsl_sf_bessel_il_scaled((int)x[0], x[1]) * exp(fabs(x[1])))
D1(r_gsl_kl, gsl_sf_bessel_kl_scaled((int)x[0], x[1]) * exp(-x[1]))
D1(r_gsl_airy_Ai, gsl_sf_airy_Ai(x[0], PD))
D1(r_gsl_airy_Bi, gsl_sf_airy_Bi(x[0], PD))
D1(r_gsl_airy_Ai_deriv, gsl_sf_airy_Ai_deriv(x[0], PD))
D1(r_gsl_airy_Bi_deriv, gsl_sf_airy_Bi_deriv(x[0], PD))

/* zeta and the others; Fermi-Dirac: Li_s(-e^mu) = -F_{s-1}(mu), mu in x[2] */
D1(r_gsl_zeta, gsl_sf_zeta(x[0]))
D1(r_gsl_hzeta, gsl_sf_hzeta(x[0], x[1]))
D1(r_gsl_dilog, gsl_sf_dilog(x[0]))
D1(r_gsl_dilog_x1, gsl_sf_dilog(x[1]))   /* polylog records: s, x, mu */
D1(r_gsl_fd_mhalf, -gsl_sf_fermi_dirac_mhalf(x[2]))
D1(r_gsl_fd_half, -gsl_sf_fermi_dirac_half(x[2]))
D1(r_gsl_fd_3half, -gsl_sf_fermi_dirac_3half(x[2]))
D1(r_gsl_fd_1, -gsl_sf_fermi_dirac_1(x[2]))
D1(r_gsl_fd_2, -gsl_sf_fermi_dirac_2(x[2]))
D1(r_gsl_fd_int, -gsl_sf_fermi_dirac_int((int)x[0] - 1, x[2]))
D1(r_gsl_lambert_W0, gsl_sf_lambert_W0(x[0]))
D1(r_gsl_lambert_Wm1, gsl_sf_lambert_Wm1(x[0]))
D1(r_gsl_E1, gsl_sf_expint_E1(x[0]))
D1(r_gsl_Ei, gsl_sf_expint_Ei(x[0]))
D1(r_gsl_En, gsl_sf_expint_En((int)x[0], x[1]))

/* elliptic: GSL takes k = sqrt(m) (precomputed in a spare slot) and Pi's n with the opposite
 * sign (1 + n sin^2) */
D1(r_gsl_Kcomp, gsl_sf_ellint_Kcomp(x[1], PD))                 /* m, k */
D1(r_gsl_Ecomp, gsl_sf_ellint_Ecomp(x[1], PD))                 /* m, k */
D1(r_gsl_Pcomp, gsl_sf_ellint_Pcomp(x[2], -x[0], PD))          /* n, m, k */
D1(r_gsl_F, gsl_sf_ellint_F(x[0], x[2], PD))                   /* phi, m, k */
D1(r_gsl_E, gsl_sf_ellint_E(x[0], x[2], PD))                   /* phi, m, k */
D1(r_gsl_P, gsl_sf_ellint_P(x[1], x[3], -x[0], PD))            /* n, phi, m, k */
D1(r_gsl_RF, gsl_sf_ellint_RF(x[0], x[1], x[2], PD))
D1(r_gsl_RD, gsl_sf_ellint_RD(x[0], x[1], x[2], PD))
D1(r_gsl_RJ, gsl_sf_ellint_RJ(x[0], x[1], x[2], x[3], PD))
D1(r_gsl_RC, gsl_sf_ellint_RC(x[0], x[1], PD))
DN(r_gsl_elljac, { gsl_sf_elljac_e(x[0], x[1], &y[0], &y[1], &y[2]); })

/* orthogonal polynomials: degree in x[0] */
D1(r_gsl_legendre_Pl, gsl_sf_legendre_Pl((int)x[0], x[1]))
D1(r_gsl_legendre_Plm, gsl_sf_legendre_Plm((int)x[0], (int)x[1], x[2]))
D1(r_gsl_gegenpoly, gsl_sf_gegenpoly_n((int)x[0], x[1], x[2]))
D1(r_gsl_hermite, gsl_sf_hermite((int)x[0], x[1]))
D1(r_gsl_hermite_prob, gsl_sf_hermite_prob((int)x[0], x[1]))
D1(r_gsl_laguerre, gsl_sf_laguerre_n((int)x[0], 0.0, x[1]))
D1(r_gsl_assoc_laguerre, gsl_sf_laguerre_n((int)x[0], x[1], x[2]))

// Boost.Math references (header-only). Batch loops over the harness's input records:
// double[4] in, double[4] out (only the used components written).
//
// Policy: no promotion of double to long double (as the area's earlier campaign; on aarch64
// macOS long double is double anyway), float promoted to double (Boost's default, and what
// the crate does for f32), and every error ignored (NaN/inf results, no exceptions or errno).
#include <cmath>
#include <cstddef>

#include <boost/math/special_functions/airy.hpp>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/chebyshev.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/ellint_1.hpp>
#include <boost/math/special_functions/ellint_2.hpp>
#include <boost/math/special_functions/ellint_3.hpp>
#include <boost/math/special_functions/ellint_rc.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <boost/math/special_functions/ellint_rf.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/expint.hpp>
#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/gegenbauer.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/jacobi.hpp>
#include <boost/math/special_functions/jacobi_elliptic.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/lambert_w.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/zeta.hpp>

namespace bm = boost::math;
using namespace boost::math::policies;

typedef policy<promote_double<false>, domain_error<ignore_error>, pole_error<ignore_error>,
               overflow_error<ignore_error>, evaluation_error<ignore_error>,
               rounding_error<ignore_error>, indeterminate_result_error<ignore_error>>
    P64;
typedef policy<promote_double<false>, domain_error<ignore_error>, pole_error<ignore_error>,
               overflow_error<ignore_error>, evaluation_error<ignore_error>,
               rounding_error<ignore_error>, indeterminate_result_error<ignore_error>>
    P32; // float stays promoted to double (the default promote_float<true>)
static const P64 p;
static const P32 pf;

typedef double V[4];
typedef float W[4];
#define D1(name, expr)                                                                       \
  extern "C" void name(const V *in, V *out, size_t n) {                                      \
    for (size_t i = 0; i < n; ++i) {                                                         \
      const double *x = in[i];                                                               \
      out[i][0] = (expr);                                                                    \
    }                                                                                        \
  }
#define DN(name, ...)                                                                       \
  extern "C" void name(const V *in, V *out, size_t n) {                                      \
    for (size_t i = 0; i < n; ++i) {                                                         \
      const double *x = in[i];                                                               \
      double *y = out[i];                                                                    \
      __VA_ARGS__;                                                                                \
    }                                                                                        \
  }
#define F1(name, expr)                                                                       \
  extern "C" void name(const W *in, W *out, size_t n) {                                      \
    for (size_t i = 0; i < n; ++i) {                                                         \
      const float *x = in[i];                                                                \
      out[i][0] = (expr);                                                                    \
    }                                                                                        \
  }

// gamma family
D1(r_boost_tgamma, bm::tgamma(x[0], p))
D1(r_boost_tgamma1, bm::tgamma(x[0] + 1.0, p))
D1(r_boost_lgamma, bm::lgamma(x[0], p))
DN(r_boost_lgamma_sign, {
  int s;
  y[0] = bm::lgamma(x[0], &s, p);
  y[1] = s;
})
D1(r_boost_digamma, bm::digamma(x[0], p))
D1(r_boost_trigamma, bm::trigamma(x[1], p))
D1(r_boost_polygamma, bm::polygamma((int)x[0], x[1], p))
D1(r_boost_beta, bm::beta(x[0], x[1], p))
D1(r_boost_factorial, bm::factorial<double>((unsigned)x[0], p))
D1(r_boost_binomial, bm::binomial_coefficient<double>((unsigned)x[0], (unsigned)x[1], p))
D1(r_boost_rising, bm::rising_factorial(x[0], (int)x[1], p))
D1(r_boost_poch_real, 1.0 / bm::tgamma_delta_ratio(x[0], x[1], p))
D1(r_boost_gamma_p, bm::gamma_p(x[0], x[1], p))
D1(r_boost_gamma_q, bm::gamma_q(x[0], x[1], p))
D1(r_boost_gamma_p_inv, bm::gamma_p_inv(x[0], x[1], p))
D1(r_boost_gamma_q_inv, bm::gamma_q_inv(x[0], x[1], p))
D1(r_boost_ibeta, bm::ibeta(x[0], x[1], x[2], p))
D1(r_boost_ibetac, bm::ibetac(x[0], x[1], x[2], p))
D1(r_boost_ibeta_inv, bm::ibeta_inv(x[0], x[1], x[2], p))
D1(r_boost_ibetac_inv, bm::ibetac_inv(x[0], x[1], x[2], p))

// error functions
D1(r_boost_erf, bm::erf(x[0], p))
D1(r_boost_erfc, bm::erfc(x[0], p))
D1(r_boost_erf_inv, bm::erf_inv(x[0], p))
D1(r_boost_erfc_inv, bm::erfc_inv(x[0], p))

// Bessel (order in x[0], argument in x[1]); the _int forms pass an integer order, which Boost
// dispatches to its integer-order (and order 0/1) code.
D1(r_boost_cyl_j_int, bm::cyl_bessel_j((int)x[0], x[1], p))
D1(r_boost_cyl_y_int, bm::cyl_neumann((int)x[0], x[1], p))
D1(r_boost_cyl_i_int, bm::cyl_bessel_i((int)x[0], x[1], p))
D1(r_boost_cyl_k_int, bm::cyl_bessel_k((int)x[0], x[1], p))
D1(r_boost_cyl_j, bm::cyl_bessel_j(x[0], x[1], p))
D1(r_boost_cyl_y, bm::cyl_neumann(x[0], x[1], p))
D1(r_boost_cyl_i, bm::cyl_bessel_i(x[0], x[1], p))
D1(r_boost_cyl_k, bm::cyl_bessel_k(x[0], x[1], p))
D1(r_boost_sph_j, bm::sph_bessel((unsigned)x[0], x[1], p))
D1(r_boost_sph_y, bm::sph_neumann((unsigned)x[0], x[1], p))
D1(r_boost_airy_ai, bm::airy_ai(x[0], p))
D1(r_boost_airy_bi, bm::airy_bi(x[0], p))
D1(r_boost_airy_ai_prime, bm::airy_ai_prime(x[0], p))
D1(r_boost_airy_bi_prime, bm::airy_bi_prime(x[0], p))

// zeta, Lambert W, exponential integrals
D1(r_boost_zeta, bm::zeta(x[0], p))
D1(r_boost_lambert_w0, bm::lambert_w0(x[0], p))
D1(r_boost_lambert_wm1, bm::lambert_wm1(x[0], p))
D1(r_boost_e1, bm::expint(1u, x[0], p))
D1(r_boost_ei, bm::expint(x[0], p))
D1(r_boost_en, bm::expint((unsigned)x[0], x[1], p))

// elliptic: Boost takes the modulus k = sqrt(m), precomputed by the harness in a spare slot
D1(r_boost_ellint_1c, bm::ellint_1(x[1], p))                 // m, k
D1(r_boost_ellint_2c, bm::ellint_2(x[1], p))                 // m, k
D1(r_boost_ellint_3c, bm::ellint_3(x[2], x[0], p))           // n, m, k
D1(r_boost_ellint_1i, bm::ellint_1(x[2], x[0], p))           // phi, m, k
D1(r_boost_ellint_2i, bm::ellint_2(x[2], x[0], p))           // phi, m, k
D1(r_boost_ellint_3i, bm::ellint_3(x[3], x[0], x[1], p))     // n, phi, m, k
D1(r_boost_rf, bm::ellint_rf(x[0], x[1], x[2], p))
D1(r_boost_rd, bm::ellint_rd(x[0], x[1], x[2], p))
D1(r_boost_rj, bm::ellint_rj(x[0], x[1], x[2], x[3], p))
D1(r_boost_rc, bm::ellint_rc(x[0], x[1], p))
DN(r_boost_jacobi_elliptic, {                                 // u, m, k
  double cn, dn;
  y[0] = bm::jacobi_elliptic(x[2], x[0], &cn, &dn, p);
  y[1] = cn;
  y[2] = dn;
})

// orthogonal polynomials (degree in x[0])
D1(r_boost_legendre_p, bm::legendre_p((int)x[0], x[1], p))
D1(r_boost_assoc_legendre_p, bm::legendre_p((int)x[0], (int)x[1], x[2], p))
D1(r_boost_chebyshev_t, bm::chebyshev_t((unsigned)x[0], x[1], p))
D1(r_boost_chebyshev_u, bm::chebyshev_u((unsigned)x[0], x[1], p))
D1(r_boost_jacobi_p, bm::jacobi((unsigned)x[0], x[1], x[2], x[3]))
D1(r_boost_gegenbauer, bm::gegenbauer((unsigned)x[0], x[1], x[2]))
D1(r_boost_hermite, bm::hermite((unsigned)x[0], x[1], p))
D1(r_boost_laguerre, bm::laguerre((unsigned)x[0], x[1], p))
D1(r_boost_assoc_laguerre, bm::laguerre((unsigned)x[0], (unsigned)x[1], x[2], p))

// f32 (computed in double by Boost's default float promotion, as the crate does)
F1(r_boost_f_tgamma, bm::tgamma(x[0], pf))
F1(r_boost_f_lgamma, bm::lgamma(x[0], pf))
F1(r_boost_f_digamma, bm::digamma(x[0], pf))
F1(r_boost_f_erf, bm::erf(x[0], pf))
F1(r_boost_f_erfc, bm::erfc(x[0], pf))
F1(r_boost_f_erf_inv, bm::erf_inv(x[0], pf))
F1(r_boost_f_gamma_p, bm::gamma_p(x[0], x[1], pf))
F1(r_boost_f_cyl_j_int, bm::cyl_bessel_j((int)x[0], x[1], pf))
F1(r_boost_f_cyl_y_int, bm::cyl_neumann((int)x[0], x[1], pf))
F1(r_boost_f_cyl_j, bm::cyl_bessel_j(x[0], x[1], pf))
F1(r_boost_f_cyl_k, bm::cyl_bessel_k(x[0], x[1], pf))
F1(r_boost_f_ellint_1c, bm::ellint_1(x[1], pf))
F1(r_boost_f_lambert_w0, bm::lambert_w0(x[0], pf))
F1(r_boost_f_ei, bm::expint(x[0], pf))
F1(r_boost_f_zeta, bm::zeta(x[0], pf))
F1(r_boost_f_airy_ai, bm::airy_ai(x[0], pf))

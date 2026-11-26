// Boost reference drivers (Boost.Math tools, quadrature, interpolators, differentiation, odeint)
// for the numerics scoreboard. The user functions are defined here; evaluations are counted.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <vector>
#include <array>
#include <limits>
#include <boost/math/tools/roots.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/math/tools/minima.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/sinh_sinh.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/math/interpolators/makima.hpp>
#include <boost/math/interpolators/pchip.hpp>
#include <boost/math/interpolators/cubic_hermite.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <complex>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
extern "C" {
#include "odeprob.h"
}

#include <boost/version.hpp>
static long b_evals = 0;
extern "C" const char *probe_boost_version(void) {
  static char v[32];
  snprintf(v, sizeof v, "%d.%d", BOOST_VERSION / 100000, BOOST_VERSION / 100 % 1000);
  return v;
}

extern "C" {

// ---------------------------------------------------------------------------------------------
// Roots of one variable: the problems of the GSL driver.
// ---------------------------------------------------------------------------------------------

static inline double rf(int prob, double p, double x) {
  b_evals++;
  switch (prob) {
  case 0: return x * x - p;
  case 1: return std::cos(x) - p * x;
  case 2: { double d = x - p; return d * d * d; }
  default: return std::cbrt(x - p);
  }
}
static inline double rdf(int prob, double p, double x) {
  switch (prob) {
  case 0: return 2.0 * x;
  case 1: return -std::sin(x) - p;
  case 2: { double d = x - p; return 3.0 * d * d; }
  default: { double c = std::cbrt(x - p); return 1.0 / (3.0 * c * c); }
  }
}
static inline double rd2f(int prob, double p, double x) {
  switch (prob) {
  case 0: return 2.0;
  case 1: return -std::cos(x);
  case 2: return 6.0 * (x - p);
  default: { double c = std::cbrt(x - p); return -2.0 / (9.0 * c * c * c * c * c); }
  }
}

// kind 0: bisect, 1: toms748. Tolerance eps_tolerance(bits): |a − b| ≤ max(2^(1−bits), 4ε)·min(|a|, |b|).
double probe_boost_root_bracket(int kind, int prob, double p, double lo, double hi, int bits,
                                long *evals) {
  using namespace boost::math::tools;
  b_evals = 0;
  std::uintmax_t it = 2000;
  auto f = [&](double x) { return rf(prob, p, x); };
  std::pair<double, double> r;
  if (kind == 0)
    r = bisect(f, lo, hi, eps_tolerance<double>(bits), it);
  else
    r = toms748_solve(f, lo, hi, eps_tolerance<double>(bits), it);
  *evals = b_evals;
  return 0.5 * (r.first + r.second);
}

// kind 0: newton_raphson_iterate, 1: halley_iterate, 2: schroder_iterate. Stops when the step is
// below 2^(1−digits) of the iterate.
double probe_boost_root_open(int kind, int prob, double p, double x0, double lo, double hi,
                             int digits, long *evals) {
  using namespace boost::math::tools;
  b_evals = 0;
  std::uintmax_t it = 200;
  double r;
  if (kind == 0) {
    auto f = [&](double x) { return std::make_pair(rf(prob, p, x), rdf(prob, p, x)); };
    r = newton_raphson_iterate(f, x0, lo, hi, digits, it);
  } else {
    auto f = [&](double x) {
      return std::make_tuple(rf(prob, p, x), rdf(prob, p, x), rd2f(prob, p, x));
    };
    r = kind == 1 ? halley_iterate(f, x0, lo, hi, digits, it) : schroder_iterate(f, x0, lo, hi, digits, it);
  }
  *evals = b_evals;
  return r;
}


// One-variable minimisation by brent_find_minima (Brent's method), the problems of the GSL driver.
double probe_min1(int prob, double p, double x);
double probe_boost_min(int prob, double p, double a, double b, int bits, long *evals) {
  b_evals = 0;
  std::uintmax_t it = 2000;
  auto f = [&](double x) { b_evals++; return probe_min1(prob, p, x); };
  auto r = boost::math::tools::brent_find_minima(f, a, b, bits, it);
  *evals = b_evals;
  return r.first;
}

// ---------------------------------------------------------------------------------------------
// Interpolation: makima, pchip, cubic_hermite (moved-in copies of the data).
// ---------------------------------------------------------------------------------------------
typedef boost::math::interpolators::makima<std::vector<double>> BMakima;
typedef boost::math::interpolators::pchip<std::vector<double>> BPchip;
typedef boost::math::interpolators::cubic_hermite<std::vector<double>> BHermite;
// type 0 makima, 1 pchip, 2 cubic_hermite (slopes dy).
void *probe_boost_interp_build(int type, int n, const double *x, const double *y, const double *dy) {
  std::vector<double> xs(x, x + n), ys(y, y + n);
  if (type == 0) return new BMakima(std::move(xs), std::move(ys));
  if (type == 1) return new BPchip(std::move(xs), std::move(ys));
  std::vector<double> ds(dy, dy + n);
  return new BHermite(std::move(xs), std::move(ys), std::move(ds));
}
void probe_boost_interp_free(int type, void *v) {
  if (type == 0) delete (BMakima *)v;
  else if (type == 1) delete (BPchip *)v;
  else delete (BHermite *)v;
}
double probe_boost_interp_eval(int type, void *v, const double *t, int m, int what) {
  double s = 0;
  for (int i = 0; i < m; i++) {
    if (type == 0) s += what ? ((BMakima *)v)->prime(t[i]) : (*(BMakima *)v)(t[i]);
    else if (type == 1) s += what ? ((BPchip *)v)->prime(t[i]) : (*(BPchip *)v)(t[i]);
    else s += what ? ((BHermite *)v)->prime(t[i]) : (*(BHermite *)v)(t[i]);
  }
  return s;
}

// Differentiation: finite_difference_derivative of order 6 and 8 (with its error estimate),
// complex_step_derivative (one complex evaluation).
double probe_dfun(int k, double x);
} // extern "C"
template <class T> static T dfun_t(int k, T x) {
  switch (k) {
  case 0: return sin(x);
  case 1: return exp(x);
  case 2: return log(x);
  case 3: return pow(x, T(2.5));
  default: return atan(x);
  }
}
extern "C" {
double probe_boost_deriv(int kind, int k, double x, double *err, long *evals) {
  using namespace boost::math::differentiation;
  b_evals = 0;
  auto f = [&](double t) { b_evals++; return probe_dfun(k, t); };
  double r;
  if (kind == 0) r = finite_difference_derivative<decltype(f), double, 6>(f, x, err);
  else if (kind == 1) r = finite_difference_derivative<decltype(f), double, 8>(f, x, err);
  else {
    auto g = [&](std::complex<double> z) { b_evals++; return dfun_t(k, z); };
    r = complex_step_derivative(g, x);
    *err = 0;
  }
  *evals = b_evals;
  return r;
}

// ---------------------------------------------------------------------------------------------
// Quadrature: the integrands of the GSL driver (probe_integrand there).
// ---------------------------------------------------------------------------------------------
} // extern "C"
extern "C" {
#include "quadint.h"
}
// The integrand fixed at compile time (K), so that Boost's templates inline it as they would a
// user's lambda.
template <int K>
static double boost_quad_k(int kind, double p, double a, double b, double tol, long *evals,
                           double *err) {
  using namespace boost::math::quadrature;
  static tanh_sinh<double> ts;
  static exp_sinh<double> es;
  static sinh_sinh<double> ss;
  b_evals = 0;
  auto f = [&](double x) { b_evals++; return qint(K, p, x); };
  double e = 0, l1 = 0, r = 0;
  std::size_t levels = 0;
  try {
  switch (kind) {
  case 0: r = gauss_kronrod<double, 15>::integrate(f, a, b, 15, tol, &e, &l1); break;
  case 1: r = gauss_kronrod<double, 21>::integrate(f, a, b, 15, tol, &e, &l1); break;
  case 2: r = ts.integrate(f, a, b, tol, &e, &l1, &levels); break;
  case 3: r = es.integrate(f, a, b, tol, &e, &l1, &levels); break;
  case 4: r = ss.integrate(f, tol, &e, &l1, &levels); break;
  default: r = gauss_kronrod<double, 61>::integrate(f, a, b, 15, tol, &e, &l1); break;
  }
  } catch (...) { r = NAN; e = NAN; }
  *evals = b_evals;
  *err = e;
  return r;
}
extern "C" {

// kind 0 gauss_kronrod<15>, 1 gauss_kronrod<21>, 2 tanh_sinh, 3 exp_sinh ([a, ∞)), 4 sinh_sinh (ℝ),
// 5 gauss_kronrod<61>. tol relative (Boost's criterion: error ≤ tol·L1).
double probe_boost_quad(int kind, int k, double p, double a, double b, double tol, long *evals,
                        double *err) {
  switch (k) {
  case 0: return boost_quad_k<0>(kind, p, a, b, tol, evals, err);
  case 1: return boost_quad_k<1>(kind, p, a, b, tol, evals, err);
  case 2: return boost_quad_k<2>(kind, p, a, b, tol, evals, err);
  case 3: return boost_quad_k<3>(kind, p, a, b, tol, evals, err);
  case 4: return boost_quad_k<4>(kind, p, a, b, tol, evals, err);
  case 5: return boost_quad_k<5>(kind, p, a, b, tol, evals, err);
  case 6: return boost_quad_k<6>(kind, p, a, b, tol, evals, err);
  case 7: return boost_quad_k<7>(kind, p, a, b, tol, evals, err);
  case 8: return boost_quad_k<8>(kind, p, a, b, tol, evals, err);
  default: return boost_quad_k<9>(kind, p, a, b, tol, evals, err);
  }
}

} // extern "C"

namespace odeint = boost::numeric::odeint;

template <class S> struct OSys {
  int prob;
  void operator()(const S &y, S &dy, double t) const {
    b_evals++;
    ode_rhs(prob, t, y.data(), dy.data());
  }
};

// kind 0 dopri5 controlled, 1 fehlberg78 controlled, 2 cash_karp54 controlled, 3 runge_kutta4
// fixed (nsteps), 4 dopri5 dense output at the times ts (integrate_times), 5 bulirsch_stoer.
// Error checker atol + rtol·|y| (a_x = 1, a_dxdt = 0) to match the other codes.
template <class S>
static int boost_ode_t(int kind, int prob, double t0, double t1, double *y0, double rtol,
                       double atol, long nsteps, const double *ts, int nt, long *steps) {
  S y;
  int n = ode_dim(prob);
  if constexpr (!std::is_same<S, std::vector<double>>::value) {
  } else {
    y.resize(n);
  }
  for (int i = 0; i < n; i++) y[i] = y0[i];
  OSys<S> sys{prob};
  size_t st = 0;
  double dt0 = 1e-6;
  try {
  switch (kind) {
  case 0: {
    typedef odeint::runge_kutta_dopri5<S> St;
    typedef odeint::controlled_runge_kutta<St> C;
    typename C::error_checker_type ec(atol, rtol, 1.0, 0.0);
    C c(ec);
    st = odeint::integrate_adaptive(c, sys, y, t0, t1, dt0);
    break;
  }
  case 1: {
    typedef odeint::runge_kutta_fehlberg78<S> St;
    typedef odeint::controlled_runge_kutta<St> C;
    typename C::error_checker_type ec(atol, rtol, 1.0, 0.0);
    C c(ec);
    st = odeint::integrate_adaptive(c, sys, y, t0, t1, dt0);
    break;
  }
  case 2: {
    typedef odeint::runge_kutta_cash_karp54<S> St;
    typedef odeint::controlled_runge_kutta<St> C;
    typename C::error_checker_type ec(atol, rtol, 1.0, 0.0);
    C c(ec);
    st = odeint::integrate_adaptive(c, sys, y, t0, t1, dt0);
    break;
  }
  case 3: {
    odeint::runge_kutta4<S> s;
    st = odeint::integrate_n_steps(s, sys, y, t0, (t1 - t0) / nsteps, (size_t)nsteps);
    break;
  }
  case 4: {
    typedef odeint::runge_kutta_dopri5<S> St;
    typedef odeint::controlled_runge_kutta<St> C;
    typename C::error_checker_type ec(atol, rtol, 1.0, 0.0);
    odeint::dense_output_runge_kutta<C> d{C(ec)};
    std::vector<double> times(ts, ts + nt);
    times.insert(times.begin(), t0);
    double acc = 0;
    st = odeint::integrate_times(d, sys, y, times.begin(), times.end(), dt0,
                                 [&](const S &x, double) { acc += x[0]; });
    break;
  }
  default: {
    odeint::bulirsch_stoer<S> bs(atol, rtol, 1.0, 0.0);
    st = odeint::integrate_adaptive(bs, sys, y, t0, t1, dt0);
    break;
  }
  }
  } catch (...) { st = 0; for (int i = 0; i < n; i++) y[i] = NAN; }
  for (int i = 0; i < n; i++) y0[i] = y[i];
  *steps = (long)st;
  return 0;
}

// Stiff: rosenbrock4 with the analytic Jacobian (ublas).
typedef boost::numeric::ublas::vector<double> uvec;
typedef boost::numeric::ublas::matrix<double> umat;
static long b_jevals = 0;
struct RSys {
  int prob;
  void operator()(const uvec &y, uvec &dy, double t) const {
    b_evals++;
    ode_rhs(prob, t, &y[0], &dy[0]);
  }
};
struct RJac {
  int prob;
  void operator()(const uvec &y, umat &J, const double &t, uvec &dfdt) const {
    b_jevals++;
    int n = ode_dim(prob);
    double Jr[2500];
    ode_jac(prob, t, &y[0], Jr);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) J(i, j) = Jr[i * n + j];
      dfdt[i] = 0.0;
    }
  }
};

extern "C" {

int probe_boost_ode(int kind, int prob, double t0, double t1, double *y0, double rtol,
                    double atol, long nsteps, const double *ts, int nt, long *fe, long *steps) {
  ode_lin_init();
  b_evals = 0;
  int n = ode_dim(prob);
  int r;
  if (n == 2) r = boost_ode_t<std::array<double, 2>>(kind, prob, t0, t1, y0, rtol, atol, nsteps, ts, nt, steps);
  else if (n == 4) r = boost_ode_t<std::array<double, 4>>(kind, prob, t0, t1, y0, rtol, atol, nsteps, ts, nt, steps);
  else r = boost_ode_t<std::vector<double>>(kind, prob, t0, t1, y0, rtol, atol, nsteps, ts, nt, steps);
  *fe = b_evals;
  return r;
}

int probe_boost_rosenbrock(int prob, double t0, double t1, double *y0, double rtol, double atol,
                           long *fe, long *je, long *steps) {
  ode_lin_init();
  b_evals = 0;
  b_jevals = 0;
  int n = ode_dim(prob);
  uvec y(n);
  for (int i = 0; i < n; i++) y[i] = y0[i];
  size_t st = 0;
  try {
    st = odeint::integrate_adaptive(odeint::make_controlled<odeint::rosenbrock4<double>>(atol, rtol),
                                    std::make_pair(RSys{prob}, RJac{prob}), y, t0, t1, 1e-6);
  } catch (...) { for (int i = 0; i < n; i++) y[i] = NAN; }
  for (int i = 0; i < n; i++) y0[i] = y[i];
  *fe = b_evals;
  *je = b_jevals;
  *steps = (long)st;
  return 0;
}

} // extern "C"

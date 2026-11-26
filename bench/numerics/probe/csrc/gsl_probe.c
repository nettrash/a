/* GSL reference drivers for the numerics scoreboard. Every function computes one whole
 * operation on an integrand/system defined here in C (no callback into Rust), and counts the
 * evaluations of the user function. Built at -O3 with the native CPU flags. */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include <gsl/gsl_qrng.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_min.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_chebyshev.h>
#include <gsl/gsl_deriv.h>

static long g_evals = 0;
static long g_devals = 0;

void probe_gsl_init(void) { gsl_set_error_handler_off(); }
#include <gsl/gsl_version.h>
const char *probe_gsl_version(void) { return GSL_VERSION; }

/* ------------------------------------------------------------------------------------------ */
/* Roots of one variable                                                                       */
/* ------------------------------------------------------------------------------------------ */

struct rp { int prob; double p; };

/* 0: x² − p; 1: cos x − p·x; 2: (x − p)³; 3: ∛(x − p). */
static double root_f(double x, void *v) {
  struct rp *r = v;
  g_evals++;
  switch (r->prob) {
  case 0: return x * x - r->p;
  case 1: return cos(x) - r->p * x;
  case 2: { double d = x - r->p; return d * d * d; }
  default: return cbrt(x - r->p);
  }
}
static double root_df(double x, void *v) {
  struct rp *r = v;
  g_devals++;
  switch (r->prob) {
  case 0: return 2.0 * x;
  case 1: return -sin(x) - r->p;
  case 2: { double d = x - r->p; return 3.0 * d * d; }
  default: { double c = cbrt(x - r->p); return 1.0 / (3.0 * c * c); }
  }
}
static void root_fdf(double x, void *v, double *f, double *df) {
  *f = root_f(x, v);
  *df = root_df(x, v);
}

/* kind: 0 bisection, 1 brent, 2 falsepos. Stops on gsl_root_test_interval(lo, hi, 0, epsrel). */
double probe_gsl_root_bracket(int kind, int prob, double p, double lo, double hi, double epsrel,
                              long *evals) {
  const gsl_root_fsolver_type *T =
      kind == 0 ? gsl_root_fsolver_bisection : kind == 1 ? gsl_root_fsolver_brent : gsl_root_fsolver_falsepos;
  static gsl_root_fsolver *s[3] = {0, 0, 0};
  if (!s[kind]) s[kind] = gsl_root_fsolver_alloc(T);
  struct rp r = {prob, p};
  gsl_function F = {root_f, &r};
  g_evals = 0;
  gsl_root_fsolver_set(s[kind], &F, lo, hi);
  double x = 0;
  for (int i = 0; i < 2000; i++) {
    if (gsl_root_fsolver_iterate(s[kind]) != GSL_SUCCESS) break;
    x = gsl_root_fsolver_root(s[kind]);
    double a = gsl_root_fsolver_x_lower(s[kind]), b = gsl_root_fsolver_x_upper(s[kind]);
    if (gsl_root_test_interval(a, b, 0.0, epsrel) == GSL_SUCCESS) break;
  }
  *evals = g_evals;
  return x;
}

/* kind: 0 newton, 1 secant, 2 steffenson. Stops on gsl_root_test_delta(x1, x0, 0, epsrel). */
double probe_gsl_root_fdf(int kind, int prob, double p, double x0, double epsrel, long *evals,
                          long *devals) {
  const gsl_root_fdfsolver_type *T =
      kind == 0 ? gsl_root_fdfsolver_newton : kind == 1 ? gsl_root_fdfsolver_secant : gsl_root_fdfsolver_steffenson;
  static gsl_root_fdfsolver *s[3] = {0, 0, 0};
  if (!s[kind]) s[kind] = gsl_root_fdfsolver_alloc(T);
  struct rp r = {prob, p};
  gsl_function_fdf F = {root_f, root_df, root_fdf, &r};
  g_evals = 0;
  g_devals = 0;
  gsl_root_fdfsolver_set(s[kind], &F, x0);
  double x = x0;
  for (int i = 0; i < 200; i++) {
    if (gsl_root_fdfsolver_iterate(s[kind]) != GSL_SUCCESS) break;
    double xn = gsl_root_fdfsolver_root(s[kind]);
    int done = gsl_root_test_delta(xn, x, 0.0, epsrel) == GSL_SUCCESS;
    x = xn;
    if (done) break;
  }
  *evals = g_evals;
  *devals = g_devals;
  return x;
}

/* ------------------------------------------------------------------------------------------ */
/* Systems of equations: Moré–Garbow–Hillstrom problems                                        */
/* ------------------------------------------------------------------------------------------ */

struct sp { int prob; int n; };

static void sys_f(int prob, int n, const double *x, double *f) {
  switch (prob) {
  case 0: f[0] = 10.0 * (x[1] - x[0] * x[0]); f[1] = 1.0 - x[0]; break;
  case 1: f[0] = 1e4 * x[0] * x[1] - 1.0; f[1] = exp(-x[0]) + exp(-x[1]) - 1.0001; break;
  case 2: {
    double th = atan(x[1] / x[0]) / (2.0 * M_PI);
    if (x[0] < 0) th += 0.5;
    f[0] = 10.0 * (x[2] - 10.0 * th);
    f[1] = 10.0 * (sqrt(x[0] * x[0] + x[1] * x[1]) - 1.0);
    f[2] = x[2];
    break;
  }
  case 3: {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++) {
      double t = (i + 1) * h, xl = i > 0 ? x[i - 1] : 0.0, xr = i + 1 < n ? x[i + 1] : 0.0;
      double u = x[i] + t + 1.0;
      f[i] = 2.0 * x[i] - xl - xr + 0.5 * h * h * u * u * u;
    }
    break;
  }
  case 4:
    for (int i = 0; i < n; i++) {
      double xl = i > 0 ? x[i - 1] : 0.0, xr = i + 1 < n ? x[i + 1] : 0.0;
      f[i] = (3.0 - 2.0 * x[i]) * x[i] - xl - 2.0 * xr + 1.0;
    }
    break;
  default: {
    double s = 0;
    for (int j = 0; j < n; j++) s += cos(x[j]);
    for (int i = 0; i < n; i++) f[i] = n - s + (i + 1) * (1.0 - cos(x[i])) - sin(x[i]);
  }
  }
}

/* Row-major n×n. */
static void sys_j(int prob, int n, const double *x, double *J) {
  memset(J, 0, sizeof(double) * n * n);
  switch (prob) {
  case 0: J[0] = -20.0 * x[0]; J[1] = 10.0; J[2] = -1.0; J[3] = 0.0; break;
  case 1:
    J[0] = 1e4 * x[1]; J[1] = 1e4 * x[0]; J[2] = -exp(-x[0]); J[3] = -exp(-x[1]); break;
  case 2: {
    double r2 = x[0] * x[0] + x[1] * x[1], r = sqrt(r2);
    J[0] = 100.0 * x[1] / (2.0 * M_PI * r2); J[1] = -100.0 * x[0] / (2.0 * M_PI * r2); J[2] = 10.0;
    J[3] = 10.0 * x[0] / r; J[4] = 10.0 * x[1] / r; J[5] = 0.0;
    J[6] = 0.0; J[7] = 0.0; J[8] = 1.0;
    break;
  }
  case 3: {
    double h = 1.0 / (n + 1);
    for (int i = 0; i < n; i++) {
      double t = (i + 1) * h, u = x[i] + t + 1.0;
      J[i * n + i] = 2.0 + 1.5 * h * h * u * u;
      if (i > 0) J[i * n + i - 1] = -1.0;
      if (i + 1 < n) J[i * n + i + 1] = -1.0;
    }
    break;
  }
  case 4:
    for (int i = 0; i < n; i++) {
      J[i * n + i] = 3.0 - 4.0 * x[i];
      if (i > 0) J[i * n + i - 1] = -1.0;
      if (i + 1 < n) J[i * n + i + 1] = -2.0;
    }
    break;
  default:
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        J[i * n + j] = sin(x[j]) + (i == j ? (i + 1) * sin(x[i]) - cos(x[i]) : 0.0);
  }
}

static int gs_f(const gsl_vector *x, void *v, gsl_vector *f) {
  struct sp *s = v;
  g_evals++;
  sys_f(s->prob, s->n, x->data, f->data);
  return GSL_SUCCESS;
}
static int gs_df(const gsl_vector *x, void *v, gsl_matrix *J) {
  struct sp *s = v;
  g_devals++;
  sys_j(s->prob, s->n, x->data, J->data);
  return GSL_SUCCESS;
}
static int gs_fdf(const gsl_vector *x, void *v, gsl_vector *f, gsl_matrix *J) {
  gs_f(x, v, f);
  gs_df(x, v, J);
  return GSL_SUCCESS;
}

/* kind 0 hybridsj, 1 newton, 2 gnewton (with the Jacobian); 3 hybrids, 4 dnewton, 5 broyden
 * (without). Stops on test_delta(dx, x, 1e-15, 4ε) or test_residual(f, 1e-15), 1000
 * iterations at most. Returns the status (0 converged), x overwritten with the answer. */
int probe_gsl_multiroot(int kind, int prob, int n, double *x, long *fevals, long *jevals,
                        int *iters) {
  struct sp s = {prob, n};
  gsl_vector_view xv = gsl_vector_view_array(x, n);
  g_evals = 0;
  g_devals = 0;
  int status = GSL_CONTINUE, it = 0;
  const double epsrel = 4.0 * GSL_DBL_EPSILON;
  if (kind < 3) {
    const gsl_multiroot_fdfsolver_type *T =
        kind == 0 ? gsl_multiroot_fdfsolver_hybridsj : kind == 1 ? gsl_multiroot_fdfsolver_newton : gsl_multiroot_fdfsolver_gnewton;
    gsl_multiroot_fdfsolver *sv = gsl_multiroot_fdfsolver_alloc(T, n);
    gsl_multiroot_function_fdf F = {gs_f, gs_df, gs_fdf, n, &s};
    gsl_multiroot_fdfsolver_set(sv, &F, &xv.vector);
    while (status == GSL_CONTINUE && it < 1000) {
      it++;
      int st = gsl_multiroot_fdfsolver_iterate(sv);
      if (st) { status = st; break; }
      status = gsl_multiroot_test_delta(sv->dx, sv->x, 1e-15, epsrel);
      if (status == GSL_CONTINUE && gsl_multiroot_test_residual(sv->f, 1e-15) == GSL_SUCCESS) status = GSL_SUCCESS;
    }
    memcpy(x, sv->x->data, sizeof(double) * n);
    gsl_multiroot_fdfsolver_free(sv);
  } else {
    const gsl_multiroot_fsolver_type *T =
        kind == 3 ? gsl_multiroot_fsolver_hybrids : kind == 4 ? gsl_multiroot_fsolver_dnewton : gsl_multiroot_fsolver_broyden;
    gsl_multiroot_fsolver *sv = gsl_multiroot_fsolver_alloc(T, n);
    gsl_multiroot_function F = {gs_f, n, &s};
    gsl_multiroot_fsolver_set(sv, &F, &xv.vector);
    while (status == GSL_CONTINUE && it < 1000) {
      it++;
      int st = gsl_multiroot_fsolver_iterate(sv);
      if (st) { status = st; break; }
      status = gsl_multiroot_test_delta(sv->dx, sv->x, 1e-15, epsrel);
      if (status == GSL_CONTINUE && gsl_multiroot_test_residual(sv->f, 1e-15) == GSL_SUCCESS) status = GSL_SUCCESS;
    }
    memcpy(x, sv->x->data, sizeof(double) * n);
    gsl_multiroot_fsolver_free(sv);
  }
  *fevals = g_evals;
  *jevals = g_devals;
  *iters = it;
  return status;
}

/* ------------------------------------------------------------------------------------------ */
/* Quadrature                                                                                  */
/* ------------------------------------------------------------------------------------------ */

#include "quadint.h"
double probe_integrand(int k, double p, double x) { return qint(k, p, x); }
static double qf(double x, void *v) {
  struct rp *r = v;
  g_evals++;
  return probe_integrand(r->prob, r->p, x);
}

/* kind 0 qags, 1 qag GAUSS15, 2 qag GAUSS21, 3 cquad, 4 qng, 5 qagiu (a..∞), 6 qagi (ℝ),
 * 7 romberg. Returns the value; status in *status. */
double probe_gsl_quad(int kind, int k, double p, double a, double b, double epsabs,
                      double epsrel, long *evals, double *err, int *status) {
  static gsl_integration_workspace *w = 0;
  static gsl_integration_cquad_workspace *cw = 0;
  static gsl_integration_romberg_workspace *rw = 0;
  if (!w) {
    w = gsl_integration_workspace_alloc(200);
    cw = gsl_integration_cquad_workspace_alloc(100);
    rw = gsl_integration_romberg_alloc(20);
  }
  struct rp r = {k, p};
  gsl_function F = {qf, &r};
  double res = 0, e = 0;
  size_t ne = 0;
  g_evals = 0;
  int st;
  switch (kind) {
  case 0: st = gsl_integration_qags(&F, a, b, epsabs, epsrel, 200, w, &res, &e); break;
  case 1: st = gsl_integration_qag(&F, a, b, epsabs, epsrel, 200, GSL_INTEG_GAUSS15, w, &res, &e); break;
  case 2: st = gsl_integration_qag(&F, a, b, epsabs, epsrel, 200, GSL_INTEG_GAUSS21, w, &res, &e); break;
  case 3: st = gsl_integration_cquad(&F, a, b, epsabs, epsrel, cw, &res, &e, &ne); break;
  case 4: st = gsl_integration_qng(&F, a, b, epsabs, epsrel, &res, &e, &ne); break;
  case 5: st = gsl_integration_qagiu(&F, a, epsabs, epsrel, 200, w, &res, &e); break;
  case 6: st = gsl_integration_qagi(&F, epsabs, epsrel, 200, w, &res, &e); break;
  default: st = gsl_integration_romberg(&F, a, b, epsabs, epsrel, &res, &ne, rw); e = 0; break;
  }
  *evals = g_evals;
  *err = e;
  *status = st;
  return res;
}

/* Gauss–Legendre fixed rule of n points from gsl_integration_glfixed (table or Newton). */
double probe_gsl_glfixed(int n, int k, double p, double a, double b) {
  gsl_integration_glfixed_table *t = gsl_integration_glfixed_table_alloc(n);
  struct rp r = {k, p};
  gsl_function F = {qf, &r};
  double v = gsl_integration_glfixed(&F, a, b, t);
  gsl_integration_glfixed_table_free(t);
  return v;
}

/* Gaussian rules: type 0 legendre (glfixed table), 1 legendre (fixed, Golub–Welsch), 2 hermite,
 * 3 laguerre, 4 jacobi(α, β). Nodes and weights written out. */
void probe_gsl_gauss(int type, int n, double alpha, double beta, double *x, double *w) {
  if (type == 0) {
    gsl_integration_glfixed_table *t = gsl_integration_glfixed_table_alloc(n);
    for (int i = 0; i < n; i++) gsl_integration_glfixed_point(-1.0, 1.0, i, &x[i], &w[i], t);
    gsl_integration_glfixed_table_free(t);
    return;
  }
  const gsl_integration_fixed_type *T =
      type == 1 ? gsl_integration_fixed_legendre : type == 2 ? gsl_integration_fixed_hermite
      : type == 3 ? gsl_integration_fixed_laguerre : gsl_integration_fixed_jacobi;
  double a = type == 1 ? -1.0 : type == 4 ? -1.0 : 0.0;
  double b = type == 1 ? 1.0 : 1.0;
  gsl_integration_fixed_workspace *ws = gsl_integration_fixed_alloc(T, n, a, b, alpha, beta);
  if (!ws) { for (int i = 0; i < n; i++) x[i] = w[i] = NAN; return; }
  memcpy(x, gsl_integration_fixed_nodes(ws), sizeof(double) * n);
  memcpy(w, gsl_integration_fixed_weights(ws), sizeof(double) * n);
  gsl_integration_fixed_free(ws);
}

/* Monte Carlo: ∫ over the unit cube of dimension d of the integrand family m. */
static double mc_f(double *x, size_t d, void *v) {
  int m = *(int *)v;
  g_evals++;
  switch (m) {
  case 0: { double s = 1; for (size_t j = 0; j < d; j++) s *= x[j] + 0.5; return s; }
  case 1: { double s = 0; for (size_t j = 0; j < d; j++) s += x[j] * x[j]; return exp(-s); }
  default: { double s = 0; for (size_t j = 0; j < d; j++) s += x[j]; return cos(s); }
  }
}
double probe_mc_family(int m, int d, const double *x) { return mc_f((double *)x, d, &m); }

/* kind 0 plain, 1 miser, 2 vegas; n calls; mt19937 seeded 1. */
double probe_gsl_monte(int kind, int m, int d, size_t n, double *err) {
  double lo[64], hi[64];
  for (int j = 0; j < d; j++) { lo[j] = 0; hi[j] = 1; }
  gsl_monte_function F = {mc_f, (size_t)d, &m};
  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, 1);
  double res = 0, e = 0;
  if (kind == 0) {
    gsl_monte_plain_state *s = gsl_monte_plain_alloc(d);
    gsl_monte_plain_integrate(&F, lo, hi, d, n, r, s, &res, &e);
    gsl_monte_plain_free(s);
  } else if (kind == 1) {
    gsl_monte_miser_state *s = gsl_monte_miser_alloc(d);
    gsl_monte_miser_integrate(&F, lo, hi, d, n, r, s, &res, &e);
    gsl_monte_miser_free(s);
  } else {
    gsl_monte_vegas_state *s = gsl_monte_vegas_alloc(d);
    gsl_monte_vegas_integrate(&F, lo, hi, d, n, r, s, &res, &e);
    gsl_monte_vegas_free(s);
  }
  gsl_rng_free(r);
  *err = e;
  return res;
}

/* n points of GSL's Sobol' sequence (d ≤ 40) summed coordinate-wise into acc. */
void probe_gsl_sobol(int d, size_t n, double *acc) {
  gsl_qrng *q = gsl_qrng_alloc(gsl_qrng_sobol, d);
  double x[64];
  for (int j = 0; j < d; j++) acc[j] = 0;
  for (size_t i = 0; i < n; i++) {
    gsl_qrng_get(q, x);
    for (int j = 0; j < d; j++) acc[j] += x[j];
  }
  gsl_qrng_free(q);
}

/* Randomly shifted QMC: r replications of n Sobol' points (GSL's sequence), Cranley–Patterson
 * shifts from mt19937; the same estimator as ours. */
double probe_gsl_rqmc(int m, int d, size_t n, int r, double *err) {
  gsl_rng *g = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(g, 1);
  double est[64], x[64], u[64], sh[64];
  for (int k = 0; k < r; k++) {
    for (int j = 0; j < d; j++) sh[j] = gsl_rng_uniform(g);
    gsl_qrng *q = gsl_qrng_alloc(gsl_qrng_sobol, d);
    double s = 0;
    for (size_t i = 0; i < n; i++) {
      gsl_qrng_get(q, u);
      for (int j = 0; j < d; j++) { double y = u[j] + sh[j]; x[j] = y >= 1.0 ? y - 1.0 : y; }
      s += mc_f(x, d, &m);
    }
    gsl_qrng_free(q);
    est[k] = s / n;
  }
  gsl_rng_free(g);
  double mean = 0;
  for (int k = 0; k < r; k++) mean += est[k];
  mean /= r;
  double sp = 0;
  for (int k = 0; k < r; k++) sp += (est[k] - mean) * (est[k] - mean);
  *err = sqrt(sp / (r * (r - 1.0)));
  return mean;
}

/* ------------------------------------------------------------------------------------------ */
/* Ordinary differential equations                                                             */
/* ------------------------------------------------------------------------------------------ */
#include "odeprob.h"

void probe_ode_init(void) { ode_lin_init(); }
void probe_ode_rhs(int prob, double t, const double *y, double *f) { ode_rhs(prob, t, y, f); }
void probe_ode_jac(int prob, double t, const double *y, double *J) { ode_jac(prob, t, y, J); }

static int go_f(double t, const double y[], double f[], void *v) {
  g_evals++;
  ode_rhs(*(int *)v, t, y, f);
  return GSL_SUCCESS;
}
static int go_j(double t, const double y[], double *J, double dfdt[], void *v) {
  int prob = *(int *)v, n = ode_dim(prob);
  g_devals++;
  ode_jac(prob, t, y, J);
  for (int i = 0; i < n; i++) dfdt[i] = 0.0;
  return GSL_SUCCESS;
}

/* kind 0 rkf45, 1 rkck, 2 rk8pd, 3 msadams, 4 msbdf, 5 bsimp, 6 rk4imp, 7 rk2, 8 rk4 (fixed,
 * nsteps steps). Error per component atol + rtol·|y| (gsl_odeiv2_driver_alloc_y_new). Returns
 * the GSL status; y is overwritten. */
int probe_gsl_ode(int kind, int prob, double t0, double t1, double *y, double rtol, double atol,
                  long nsteps, long *fe, long *je, long *steps) {
  const gsl_odeiv2_step_type *T[] = {gsl_odeiv2_step_rkf45, gsl_odeiv2_step_rkck, gsl_odeiv2_step_rk8pd,
                                     gsl_odeiv2_step_msadams, gsl_odeiv2_step_msbdf, gsl_odeiv2_step_bsimp,
                                     gsl_odeiv2_step_rk4imp, gsl_odeiv2_step_rk2, gsl_odeiv2_step_rk4};
  int p = prob;
  gsl_odeiv2_system sys = {go_f, go_j, (size_t)ode_dim(prob), &p};
  g_evals = 0;
  g_devals = 0;
  double h0 = (kind == 8 && nsteps > 0) ? (t1 - t0) / nsteps : 1e-6;
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, T[kind], h0, atol, rtol);
  gsl_odeiv2_driver_set_nmax(d, 1000000);
  double t = t0;
  int st;
  if (kind == 8 && nsteps > 0)
    st = gsl_odeiv2_driver_apply_fixed_step(d, &t, h0, nsteps, y);
  else
    st = gsl_odeiv2_driver_apply(d, &t, t1, y);
  *steps = (long)d->n;
  gsl_odeiv2_driver_free(d);
  *fe = g_evals;
  *je = g_devals;
  return st;
}

/* Dense output in GSL is not offered; solve_at is the driver applied to each output time. */
int probe_gsl_ode_at(int kind, int prob, double t0, const double *ts, int nt, double *y,
                     double rtol, double atol, long *fe) {
  int p = prob;
  const gsl_odeiv2_step_type *T = kind == 2 ? gsl_odeiv2_step_rk8pd : gsl_odeiv2_step_rkf45;
  gsl_odeiv2_system sys = {go_f, go_j, (size_t)ode_dim(prob), &p};
  g_evals = 0;
  gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, T, 1e-6, atol, rtol);
  double t = t0;
  int st = 0;
  for (int i = 0; i < nt && st == 0; i++) st = gsl_odeiv2_driver_apply(d, &t, ts[i], y);
  gsl_odeiv2_driver_free(d);
  *fe = g_evals;
  return st;
}
void probe_ode_lin_matrix(double *A) { ode_lin_init(); memcpy(A, ode_lin_A, sizeof(double) * 2500); }

/* ------------------------------------------------------------------------------------------ */
/* Optimisation                                                                                */
/* ------------------------------------------------------------------------------------------ */

/* One variable: 0 (x − p)² + 1, 1 x ln x − p x, 2 (x − p)⁴. */
double probe_min1(int prob, double p, double x) {
  switch (prob) {
  case 0: { double d = x - p; return d * d + 1.0; }
  case 1: return x * log(x) - p * x;
  default: { double d = x - p; d *= d; return d * d; }
  }
}
static double m1f(double x, void *v) {
  struct rp *r = v;
  g_evals++;
  return probe_min1(r->prob, r->p, x);
}
/* kind 0 goldensection, 1 brent, 2 quad_golden; stops on test_interval(a, b, 0, epsrel). */
double probe_gsl_min(int kind, int prob, double p, double a, double b, double guess, double epsrel,
                     long *evals) {
  const gsl_min_fminimizer_type *T =
      kind == 0 ? gsl_min_fminimizer_goldensection : kind == 1 ? gsl_min_fminimizer_brent : gsl_min_fminimizer_quad_golden;
  static gsl_min_fminimizer *s[3] = {0, 0, 0};
  if (!s[kind]) s[kind] = gsl_min_fminimizer_alloc(T);
  struct rp r = {prob, p};
  gsl_function F = {m1f, &r};
  g_evals = 0;
  if (gsl_min_fminimizer_set(s[kind], &F, guess, a, b) != GSL_SUCCESS) { *evals = g_evals; return NAN; }
  double x = guess;
  for (int i = 0; i < 2000; i++) {
    if (gsl_min_fminimizer_iterate(s[kind]) != GSL_SUCCESS) break;
    x = gsl_min_fminimizer_x_minimum(s[kind]);
    double lo = gsl_min_fminimizer_x_lower(s[kind]), hi = gsl_min_fminimizer_x_upper(s[kind]);
    if (gsl_min_test_interval(lo, hi, 0.0, epsrel) == GSL_SUCCESS) break;
  }
  *evals = g_evals;
  return x;
}

/* Several variables: 0 chained Rosenbrock (n), 1 Powell singular (4), 2 Wood (4), 3 Beale (2),
 * 4 quadratic Σ λᵢxᵢ²/2, λᵢ = 10^(4i/(n−1)) (n). */
static const double *mn_lambda(int n) {
  static double l[4096];
  static int ln = 0;
  if (ln != n) {
    for (int i = 0; i < n; i++) l[i] = pow(10.0, 4.0 * i / (n - 1));
    ln = n;
  }
  return l;
}
double probe_mn_f(int prob, int n, const double *x) {
  switch (prob) {
  case 0: {
    double s = 0;
    for (int i = 0; i + 1 < n; i++) {
      double a = x[i + 1] - x[i] * x[i], b = 1.0 - x[i];
      s += 100.0 * a * a + b * b;
    }
    return s;
  }
  case 1: {
    double a = x[0] + 10 * x[1], b = x[2] - x[3], c = x[1] - 2 * x[2], d = x[0] - x[3];
    return a * a + 5 * b * b + c * c * c * c + 10 * d * d * d * d;
  }
  case 2: {
    double a = x[1] - x[0] * x[0], b = 1 - x[0], c = x[3] - x[2] * x[2], d = 1 - x[2], e = x[1] - 1, f = x[3] - 1;
    return 100 * a * a + b * b + 90 * c * c + d * d + 10.1 * (e * e + f * f) + 19.8 * e * f;
  }
  case 3: {
    double a = 1.5 - x[0] + x[0] * x[1], b = 2.25 - x[0] + x[0] * x[1] * x[1], c = 2.625 - x[0] + x[0] * x[1] * x[1] * x[1];
    return a * a + b * b + c * c;
  }
  default: {
    const double *l = mn_lambda(n);
    double s = 0;
    for (int i = 0; i < n; i++) s += 0.5 * l[i] * x[i] * x[i];
    return s;
  }
  }
}
void probe_mn_g(int prob, int n, const double *x, double *g) {
  switch (prob) {
  case 0:
    for (int i = 0; i < n; i++) g[i] = 0;
    for (int i = 0; i + 1 < n; i++) {
      double a = x[i + 1] - x[i] * x[i];
      g[i] += -400.0 * x[i] * a - 2.0 * (1.0 - x[i]);
      g[i + 1] += 200.0 * a;
    }
    break;
  case 1: {
    double a = x[0] + 10 * x[1], b = x[2] - x[3], c = x[1] - 2 * x[2], d = x[0] - x[3];
    g[0] = 2 * a + 40 * d * d * d;
    g[1] = 20 * a + 4 * c * c * c;
    g[2] = 10 * b - 8 * c * c * c;
    g[3] = -10 * b - 40 * d * d * d;
    break;
  }
  case 2: {
    double a = x[1] - x[0] * x[0], c = x[3] - x[2] * x[2], e = x[1] - 1, f = x[3] - 1;
    g[0] = -400 * x[0] * a - 2 * (1 - x[0]);
    g[1] = 200 * a + 20.2 * e + 19.8 * f;
    g[2] = -360 * x[2] * c - 2 * (1 - x[2]);
    g[3] = 180 * c + 20.2 * f + 19.8 * e;
    break;
  }
  case 3: {
    double y = x[1], a = 1.5 - x[0] + x[0] * y, b = 2.25 - x[0] + x[0] * y * y, c = 2.625 - x[0] + x[0] * y * y * y;
    g[0] = 2 * a * (y - 1) + 2 * b * (y * y - 1) + 2 * c * (y * y * y - 1);
    g[1] = 2 * a * x[0] + 4 * b * x[0] * y + 6 * c * x[0] * y * y;
    break;
  }
  default: {
    const double *l = mn_lambda(n);
    for (int i = 0; i < n; i++) g[i] = l[i] * x[i];
  }
  }
}

struct mp { int prob; int n; };
static double mm_f(const gsl_vector *x, void *v) {
  struct mp *m = v;
  g_evals++;
  return probe_mn_f(m->prob, m->n, x->data);
}
static void mm_df(const gsl_vector *x, void *v, gsl_vector *g) {
  struct mp *m = v;
  g_devals++;
  probe_mn_g(m->prob, m->n, x->data, g->data);
}
static void mm_fdf(const gsl_vector *x, void *v, double *f, gsl_vector *g) {
  *f = mm_f(x, v);
  mm_df(x, v, g);
}

/* kind 0 vector_bfgs2, 1 conjugate_pr, 2 conjugate_fr; stop when ‖g‖₂ < gtol. kind 3:
 * nmsimplex2 with initial steps `step`, stop when the simplex size < gtol. */
int probe_gsl_multimin(int kind, int prob, int n, double *x, double gtol, double step,
                       int maxiter, long *fe, long *ge, int *iters) {
  struct mp m = {prob, n};
  gsl_vector_view xv = gsl_vector_view_array(x, n);
  g_evals = 0;
  g_devals = 0;
  int it = 0, status = GSL_CONTINUE;
  if (kind < 3) {
    const gsl_multimin_fdfminimizer_type *T =
        kind == 0 ? gsl_multimin_fdfminimizer_vector_bfgs2 : kind == 1 ? gsl_multimin_fdfminimizer_conjugate_pr : gsl_multimin_fdfminimizer_conjugate_fr;
    gsl_multimin_fdfminimizer *s = gsl_multimin_fdfminimizer_alloc(T, n);
    gsl_multimin_function_fdf F = {mm_f, mm_df, mm_fdf, n, &m};
    gsl_multimin_fdfminimizer_set(s, &F, &xv.vector, 0.01, 0.1);
    while (status == GSL_CONTINUE && it < maxiter) {
      it++;
      if (gsl_multimin_fdfminimizer_iterate(s)) { status = GSL_ENOPROG; break; }
      status = gsl_multimin_test_gradient(s->gradient, gtol);
    }
    memcpy(x, s->x->data, sizeof(double) * n);
    gsl_multimin_fdfminimizer_free(s);
  } else {
    gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(gsl_multimin_fminimizer_nmsimplex2, n);
    gsl_multimin_function F = {mm_f, n, &m};
    gsl_vector *ss = gsl_vector_alloc(n);
    gsl_vector_set_all(ss, step);
    gsl_multimin_fminimizer_set(s, &F, &xv.vector, ss);
    while (status == GSL_CONTINUE && it < maxiter) {
      it++;
      if (gsl_multimin_fminimizer_iterate(s)) { status = GSL_ENOPROG; break; }
      status = gsl_multimin_test_size(gsl_multimin_fminimizer_size(s), gtol);
    }
    memcpy(x, s->x->data, sizeof(double) * n);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);
  }
  *fe = g_evals;
  *ge = g_devals;
  *iters = it;
  return status;
}

/* Least squares: model id 0 b₁(1 − e^(−b₂x)) (Misra1a, BoxBOD), 1 Thurber, 2 MGH09, 3 MGH17,
 * 4 a·e^(−λt) + c (synthetic). The data are passed in. */
struct fp { int model; int m; int p; const double *x; const double *y; };
static void model_eval(int model, const double *b, double x, double *v, double *d) {
  switch (model) {
  case 0: { double e = exp(-b[1] * x); *v = b[0] * (1 - e); d[0] = 1 - e; d[1] = b[0] * x * e; break; }
  case 1: {
    double x2 = x * x, x3 = x2 * x;
    double num = b[0] + b[1] * x + b[2] * x2 + b[3] * x3, den = 1 + b[4] * x + b[5] * x2 + b[6] * x3, d2 = den * den;
    *v = num / den;
    d[0] = 1 / den; d[1] = x / den; d[2] = x2 / den; d[3] = x3 / den;
    d[4] = -num * x / d2; d[5] = -num * x2 / d2; d[6] = -num * x3 / d2;
    break;
  }
  case 2: {
    double num = x * x + x * b[1], den = x * x + x * b[2] + b[3];
    *v = b[0] * num / den;
    d[0] = num / den; d[1] = b[0] * x / den; d[2] = -b[0] * num * x / (den * den); d[3] = -b[0] * num / (den * den);
    break;
  }
  case 3: {
    double e4 = exp(-x * b[3]), e5 = exp(-x * b[4]);
    *v = b[0] + b[1] * e4 + b[2] * e5;
    d[0] = 1; d[1] = e4; d[2] = e5; d[3] = -b[1] * x * e4; d[4] = -b[2] * x * e5;
    break;
  }
  default: {
    double e = exp(-b[1] * x);
    *v = b[0] * e + b[2];
    d[0] = e; d[1] = -b[0] * x * e; d[2] = 1;
  }
  }
}
static int fit_f(const gsl_vector *b, void *v, gsl_vector *f) {
  struct fp *q = v;
  double d[8];
  g_evals++;
  for (int i = 0; i < q->m; i++) {
    double val;
    model_eval(q->model, b->data, q->x[i], &val, d);
    f->data[i] = val - q->y[i];
  }
  return GSL_SUCCESS;
}
static int fit_df(const gsl_vector *b, void *v, gsl_matrix *J) {
  struct fp *q = v;
  double d[8], val;
  g_devals++;
  for (int i = 0; i < q->m; i++) {
    model_eval(q->model, b->data, q->x[i], &val, d);
    for (int k = 0; k < q->p; k++) J->data[i * J->tda + k] = d[k];
  }
  return GSL_SUCCESS;
}
/* trs 0 lm, 1 lmaccel, 2 dogleg; xtol/gtol as gsl_multifit_nlinear_driver takes them. */
int probe_gsl_fit(int trs, int model, int m, int p, const double *x, const double *y, double *b,
                  double xtol, double gtol, int maxiter, long *fe, long *je, int *iters) {
  struct fp q = {model, m, p, x, y};
  gsl_multifit_nlinear_parameters params = gsl_multifit_nlinear_default_parameters();
  params.trs = trs == 0 ? gsl_multifit_nlinear_trs_lm : trs == 1 ? gsl_multifit_nlinear_trs_lmaccel : gsl_multifit_nlinear_trs_dogleg;
  gsl_multifit_nlinear_workspace *w = gsl_multifit_nlinear_alloc(gsl_multifit_nlinear_trust, &params, m, p);
  gsl_multifit_nlinear_fdf fdf;
  fdf.f = fit_f;
  fdf.df = fit_df;
  fdf.fvv = NULL;
  fdf.n = m;
  fdf.p = p;
  fdf.params = &q;
  gsl_vector_view bv = gsl_vector_view_array(b, p);
  g_evals = 0;
  g_devals = 0;
  gsl_multifit_nlinear_init(&bv.vector, &fdf, w);
  int info = 0;
  int st = gsl_multifit_nlinear_driver(maxiter, xtol, gtol, 0.0, NULL, NULL, &info, w);
  memcpy(b, gsl_multifit_nlinear_position(w)->data, sizeof(double) * p);
  *iters = (int)gsl_multifit_nlinear_niter(w);
  gsl_multifit_nlinear_free(w);
  *fe = g_evals;
  *je = g_devals;
  return st;
}

/* ------------------------------------------------------------------------------------------ */
/* Interpolation, Chebyshev, differentiation                                                   */
/* ------------------------------------------------------------------------------------------ */

struct sh { gsl_spline *s; gsl_interp_accel *a; };
/* type 0 cspline (natural), 1 akima, 2 steffen. */
void *probe_gsl_spline_build(int type, int n, const double *x, const double *y) {
  const gsl_interp_type *T = type == 0 ? gsl_interp_cspline : type == 1 ? gsl_interp_akima : gsl_interp_steffen;
  struct sh *h = malloc(sizeof *h);
  h->s = gsl_spline_alloc(T, n);
  h->a = gsl_interp_accel_alloc();
  gsl_spline_init(h->s, x, y, n);
  return h;
}
void probe_gsl_spline_free(void *v) {
  struct sh *h = v;
  gsl_spline_free(h->s);
  gsl_interp_accel_free(h->a);
  free(h);
}
/* what 0 value, 1 derivative; accel 1 with the accelerator, 0 without (binary search). */
double probe_gsl_spline_eval(void *v, const double *t, int m, int what, int accel) {
  struct sh *h = v;
  gsl_interp_accel *a = accel ? h->a : NULL;
  double s = 0;
  if (what == 0)
    for (int i = 0; i < m; i++) s += gsl_spline_eval(h->s, t[i], a);
  else
    for (int i = 0; i < m; i++) s += gsl_spline_eval_deriv(h->s, t[i], a);
  return s;
}
double probe_gsl_spline_integ(void *v, double a, double b) {
  struct sh *h = v;
  return gsl_spline_eval_integ(h->s, a, b, h->a);
}

double probe_cheb_f(double x) { return exp(x) * cos(3.0 * x); }
static double cheb_gf(double x, void *v) { (void)v; g_evals++; return probe_cheb_f(x); }
void *probe_gsl_cheb_build(int order, double a, double b) {
  gsl_cheb_series *cs = gsl_cheb_alloc(order);
  gsl_function F = {cheb_gf, NULL};
  gsl_cheb_init(cs, &F, a, b);
  return cs;
}
void probe_gsl_cheb_free(void *v) { gsl_cheb_free(v); }
double probe_gsl_cheb_eval(void *v, const double *t, int m) {
  double s = 0;
  for (int i = 0; i < m; i++) s += gsl_cheb_eval(v, t[i]);
  return s;
}
double probe_gsl_cheb_coef(void *v, int k) { return ((gsl_cheb_series *)v)->c[k]; }
/* The derivative series, made once and evaluated at the points. */
double probe_gsl_cheb_deriv_eval(void *v, const double *t, int m) {
  gsl_cheb_series *cs = v;
  gsl_cheb_series *d = gsl_cheb_alloc(cs->order);
  gsl_cheb_calc_deriv(d, cs);
  double s = 0;
  for (int i = 0; i < m; i++) s += gsl_cheb_eval(d, t[i]);
  gsl_cheb_free(d);
  return s;
}

/* Differentiation: 0 sin, 1 exp, 2 log, 3 x^2.5, 4 atan. */
double probe_dfun(int k, double x) {
  switch (k) {
  case 0: return sin(x);
  case 1: return exp(x);
  case 2: return log(x);
  case 3: return pow(x, 2.5);
  default: return atan(x);
  }
}
static double dgf(double x, void *v) { g_evals++; return probe_dfun(*(int *)v, x); }
/* kind 0 central, 1 forward. */
double probe_gsl_deriv(int kind, int k, double x, double h, double *abserr, long *evals) {
  gsl_function F = {dgf, &k};
  double r = 0;
  g_evals = 0;
  if (kind == 0) gsl_deriv_central(&F, x, h, &r, abserr);
  else gsl_deriv_forward(&F, x, h, &r, abserr);
  *evals = g_evals;
  return r;
}

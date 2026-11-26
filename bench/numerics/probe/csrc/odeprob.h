/* The ODE test problems, shared by the GSL and the Boost drivers (inlined into both), written
 * exactly as the Rust side writes them (src/ode.rs).
 *  0 Arenstorf orbit (4)          1 Kepler, e = 0.9 (4)        2 Brusselator (2)
 *  3 chain of 50 oscillators (100)
 * 10 Robertson (3)               11 Van der Pol, μ = 1000 (2)  12 HIRES (8)
 * 13 OREGO (3)                   14 linear y' = Q·diag(λ)·Q·y, n = 50 */
#ifndef ODEPROB_H
#define ODEPROB_H
#include <math.h>
#include <string.h>

static double ode_lin_A[50 * 50];
static int ode_lin_ready = 0;

static void ode_lin_init(void) {
  if (ode_lin_ready) return;
  double v[50], vv = 0, lam[50];
  for (int i = 0; i < 50; i++) { v[i] = sin((double)(i + 1)); vv += v[i] * v[i]; }
  for (int k = 0; k < 50; k++) lam[k] = -pow(10.0, 6.0 * k / 49.0);
  /* Q = I − 2 v vᵀ / (vᵀv); A = Q diag(λ) Q, row-major. */
  double Q[50 * 50];
  for (int i = 0; i < 50; i++)
    for (int j = 0; j < 50; j++) Q[i * 50 + j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / vv;
  for (int i = 0; i < 50; i++)
    for (int j = 0; j < 50; j++) {
      double s = 0;
      for (int k = 0; k < 50; k++) s += Q[i * 50 + k] * lam[k] * Q[k * 50 + j];
      ode_lin_A[i * 50 + j] = s;
    }
  ode_lin_ready = 1;
}

static inline void ode_rhs(int prob, double t, const double *y, double *f) {
  (void)t;
  switch (prob) {
  case 0: {
    const double mu = 0.012277471, mp = 1.0 - mu;
    double a = y[0] + mu, b = y[0] - mp;
    double d1 = a * a + y[1] * y[1], d2 = b * b + y[1] * y[1];
    d1 = d1 * sqrt(d1);
    d2 = d2 * sqrt(d2);
    f[0] = y[2];
    f[1] = y[3];
    f[2] = y[0] + 2.0 * y[3] - mp * a / d1 - mu * b / d2;
    f[3] = y[1] - 2.0 * y[2] - mp * y[1] / d1 - mu * y[1] / d2;
    break;
  }
  case 1: {
    double r2 = y[0] * y[0] + y[1] * y[1], r3 = r2 * sqrt(r2);
    f[0] = y[2];
    f[1] = y[3];
    f[2] = -y[0] / r3;
    f[3] = -y[1] / r3;
    break;
  }
  case 2: {
    double u = y[0] * y[0] * y[1];
    f[0] = 1.0 + u - 4.0 * y[0];
    f[1] = 3.0 * y[0] - u;
    break;
  }
  case 3: {
    const int m = 50;
    for (int i = 0; i < m; i++) {
      double l = i > 0 ? y[i - 1] : 0.0, r = i + 1 < m ? y[i + 1] : 0.0;
      f[i] = y[m + i];
      f[m + i] = l - 2.0 * y[i] + r;
    }
    break;
  }
  case 10:
    f[0] = -0.04 * y[0] + 1e4 * y[1] * y[2];
    f[2] = 3e7 * y[1] * y[1];
    f[1] = -f[0] - f[2];
    break;
  case 11:
    f[0] = y[1];
    f[1] = 1000.0 * (1.0 - y[0] * y[0]) * y[1] - y[0];
    break;
  case 12:
    f[0] = -1.71 * y[0] + 0.43 * y[1] + 8.32 * y[2] + 0.0007;
    f[1] = 1.71 * y[0] - 8.75 * y[1];
    f[2] = -10.03 * y[2] + 0.43 * y[3] + 0.035 * y[4];
    f[3] = 8.32 * y[1] + 1.71 * y[2] - 1.12 * y[3];
    f[4] = -1.745 * y[4] + 0.43 * y[5] + 0.43 * y[6];
    f[5] = -280.0 * y[5] * y[7] + 0.69 * y[3] + 1.71 * y[4] - 0.43 * y[5] + 0.69 * y[6];
    f[6] = 280.0 * y[5] * y[7] - 1.81 * y[6];
    f[7] = -f[6];
    break;
  case 13: {
    const double s = 77.27, w = 0.161, q = 8.375e-6;
    f[0] = s * (y[1] + y[0] * (1.0 - q * y[0] - y[1]));
    f[1] = (y[2] - (1.0 + y[0]) * y[1]) / s;
    f[2] = w * (y[0] - y[2]);
    break;
  }
  default:
    for (int i = 0; i < 50; i++) {
      double s = 0;
      for (int j = 0; j < 50; j++) s += ode_lin_A[i * 50 + j] * y[j];
      f[i] = s;
    }
  }
}

/* Row-major n×n Jacobian of the stiff problems. */
static inline void ode_jac(int prob, double t, const double *y, double *J) {
  (void)t;
  switch (prob) {
  case 10:
    J[0] = -0.04; J[1] = 1e4 * y[2]; J[2] = 1e4 * y[1];
    J[6] = 0.0; J[7] = 6e7 * y[1]; J[8] = 0.0;
    J[3] = 0.04; J[4] = -1e4 * y[2] - 6e7 * y[1]; J[5] = -1e4 * y[1];
    break;
  case 11:
    J[0] = 0.0; J[1] = 1.0;
    J[2] = -2000.0 * y[0] * y[1] - 1.0; J[3] = 1000.0 * (1.0 - y[0] * y[0]);
    break;
  case 12:
    memset(J, 0, sizeof(double) * 64);
    J[0 * 8 + 0] = -1.71; J[0 * 8 + 1] = 0.43; J[0 * 8 + 2] = 8.32;
    J[1 * 8 + 0] = 1.71; J[1 * 8 + 1] = -8.75;
    J[2 * 8 + 2] = -10.03; J[2 * 8 + 3] = 0.43; J[2 * 8 + 4] = 0.035;
    J[3 * 8 + 1] = 8.32; J[3 * 8 + 2] = 1.71; J[3 * 8 + 3] = -1.12;
    J[4 * 8 + 4] = -1.745; J[4 * 8 + 5] = 0.43; J[4 * 8 + 6] = 0.43;
    J[5 * 8 + 3] = 0.69; J[5 * 8 + 4] = 1.71; J[5 * 8 + 5] = -0.43 - 280.0 * y[7]; J[5 * 8 + 6] = 0.69; J[5 * 8 + 7] = -280.0 * y[5];
    J[6 * 8 + 5] = 280.0 * y[7]; J[6 * 8 + 6] = -1.81; J[6 * 8 + 7] = 280.0 * y[5];
    J[7 * 8 + 5] = -280.0 * y[7]; J[7 * 8 + 6] = 1.81; J[7 * 8 + 7] = -280.0 * y[5];
    break;
  case 13: {
    const double s = 77.27, w = 0.161, q = 8.375e-6;
    J[0] = s * (1.0 - 2.0 * q * y[0] - y[1]); J[1] = s * (1.0 - y[0]); J[2] = 0.0;
    J[3] = -y[1] / s; J[4] = -(1.0 + y[0]) / s; J[5] = 1.0 / s;
    J[6] = w; J[7] = 0.0; J[8] = -w;
    break;
  }
  default:
    memcpy(J, ode_lin_A, sizeof(double) * 2500);
  }
}

static inline int ode_dim(int prob) {
  switch (prob) {
  case 0: case 1: return 4;
  case 2: return 2;
  case 3: return 100;
  case 10: return 3;
  case 11: return 2;
  case 12: return 8;
  case 13: return 3;
  default: return 50;
  }
}
#endif

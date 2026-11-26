/* The quadrature integrands, inlined into the GSL driver (behind its function pointer, as GSL
 * calls any integrand) and into the Boost driver's lambdas (with the integrand fixed at compile
 * time, as a user's lambda is), written as the Rust side writes them. */
#ifndef QUADINT_H
#define QUADINT_H
#include <math.h>
static inline double qint(int k, double p, double x) {
  switch (k) {
  case 0: { double u = p * x; return 4.0 / (1.0 + u * u); }
  case 1: return log(p * x);
  case 2: return 1.0 / sqrt(p * x);
  case 3: return pow(p * x, -0.9);
  case 4: return cos(100.0 * p * x);
  case 5: return exp(-p * x);
  case 6: return exp(-p * x * x);
  case 7: return pow(p * x, -1.1);
  case 8: { double u = p * x; return 1.0 / (1.0 + u * u); }
  default: { double d = x - 0.3; return 1.0 / (d * d + 1e-4 * p); }
  }
}
#endif

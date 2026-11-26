/* System libm references (Apple libm on macOS, glibc's on Linux). Batch loops over the
 * harness's input records: double[4] in, double[4] out (only the used components). */
#define _REENTRANT 1
#define _GNU_SOURCE 1
#include <math.h>
#include <stddef.h>

double lgamma_r(double, int *);

typedef double V[4];
typedef float W[4];
#define D1(name, expr) \
  void name(const V *in, V *out, size_t n) { \
    for (size_t i = 0; i < n; ++i) { const double *x = in[i]; out[i][0] = (expr); } }
#define DN(name, ...) \
  void name(const V *in, V *out, size_t n) { \
    for (size_t i = 0; i < n; ++i) { const double *x = in[i]; double *y = out[i]; __VA_ARGS__; } }
#define F1(name, expr) \
  void name(const W *in, W *out, size_t n) { \
    for (size_t i = 0; i < n; ++i) { const float *x = in[i]; out[i][0] = (expr); } }

D1(r_libm_tgamma, tgamma(x[0]))
D1(r_libm_tgamma1, tgamma(x[0] + 1.0))
D1(r_libm_lgamma, lgamma(x[0]))
DN(r_libm_lgamma_r, { int s; y[0] = lgamma_r(x[0], &s); y[1] = s; })
D1(r_libm_erf, erf(x[0]))
D1(r_libm_erfc, erfc(x[0]))
/* Bessel rows keep the order in x[0] and the argument in x[1]. */
D1(r_libm_j0, j0(x[1]))
D1(r_libm_j1, j1(x[1]))
D1(r_libm_y0, y0(x[1]))
D1(r_libm_y1, y1(x[1]))
D1(r_libm_jn, jn((int)x[0], x[1]))
D1(r_libm_yn, yn((int)x[0], x[1]))

F1(r_libm_f_tgamma, tgammaf(x[0]))
F1(r_libm_f_lgamma, lgammaf(x[0]))
F1(r_libm_f_erf, erff(x[0]))
F1(r_libm_f_erfc, erfcf(x[0]))

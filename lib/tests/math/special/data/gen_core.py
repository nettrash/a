"""The core's plumbing data: a few hundred cases of gamma, lgamma and the complex gamma, to test
the dispatch of every type and the harness (the gamma area's own data replaces it)."""

import random

import mpmath

from gen_common import log_uniform, mp_arg, to_f32, uniform, write

rng = random.Random(20)


def gamma_args(n, single):
    xs = log_uniform(rng, 1e-3, 170.0, n, single) + uniform(rng, -170.0, 0.0, n // 2, single)
    return [x for x in xs if x != int(x) or x > 0]


rows = [((x,), [mpmath.gamma(mp_arg(x))]) for x in gamma_args(300, False)]
write("core_gamma_f64.txt", "gamma(x), f64: Lanczos baseline of the core", rows)
rows = [((x,), [mpmath.gamma(mp_arg(x))]) for x in gamma_args(200, True) if abs(x) < 35]
write("core_gamma_f32.txt", "gamma(x), f32", rows)
xs = log_uniform(rng, 1e-3, 1e300, 300) + uniform(rng, -170.0, -0.001, 100)
rows = [((x,), [mpmath.log(abs(mpmath.gamma(mp_arg(x))))]) for x in xs if x != int(x) or x > 0]
write("core_lgamma_f64.txt", "ln|gamma(x)|, f64", rows)
zs = [(rng.uniform(-20, 40), rng.uniform(-30, 30)) for _ in range(150)]
rows = [((a, b), [mpmath.gamma(mpmath.mpc(a, b))]) for a, b in zs]
write("core_gamma_c64.txt", "gamma(z), Complex<f64>", rows)

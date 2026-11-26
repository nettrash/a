"""SciPy rows of the special-functions scoreboard (the v0.2 speed pass).

SciPy's compiled special functions are reached through its ufuncs on arrays, so they are timed
per element on large arrays. Ours run in the same process through the harness's cdylib
(libsbench: the same records, the same batch functions as the Rust harness), alternating with
SciPy sample by sample. Each sample is one call on a contiguous slice of >= 1 ms (the faster
side repeats it), 21 samples after 2 warm-up rounds, medians in ns per element.

usage: python3 scipy_bench.py LIBSBENCH [--group G1,G2] [--filter TEXT] [--samples N] > rows.tsv

A case's SciPy spec (set in the Rust case list) is  name:args[:outs]
  name  a ufunc of scipy.special (or of its _ufuncs module, for the private ones)
  args  comma list: x0..x3 float64 column, i0..i3 int64 column, c01 complex from columns 0
        and 1, z0 complex from column 0, an int or float literal
  outs  which ufunc outputs to compare with our components, in order (default 0); c = output 0
        complex against our (re, im); cr = output 0 complex, its real part against ours
"""
import ctypes
import math
import sys
import time

import numpy as np
import scipy
import scipy.special as S
import scipy.special._ufuncs as U

PRIVATE = {
    "spherical_jn": U._spherical_jn,
    "spherical_yn": U._spherical_yn,
    "spherical_in": U._spherical_in,
    "spherical_kn": U._spherical_kn,
    "lambertw": U._lambertw,
}
NOTES = {
    "polygamma": "Python wrapper: (-1)^(n+1) n! zeta(n+1, x) with NumPy array ops, allocates its result",
    "airy": "computes Ai, Ai', Bi, Bi' together",
    "ellipj": "computes sn, cn, dn and the amplitude together",
    "fresnel": "computes S and C together",
    "lambertw": "complex result (real part compared), tol 1e-8 (lambertw's default)",
    "spence": "spence(1-x) = Li2(x), 1-x precomputed",
    "lpmv": "lpmv(m, n, x), real order and degree",
    "gammaincinv": "",
}


def fn_for(name, nargs):
    if name == "zeta":
        return U._riemann_zeta if nargs == 1 else U._zeta
    if name in PRIVATE:
        return PRIVATE[name]
    return getattr(S, name)


def sig3(x):
    if not math.isfinite(x):
        return "-"
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    d = max(2 - e, 0)
    return f"{round(x, 2 - e):.{d}f}"


def fmt_u(x):
    if not math.isfinite(x):
        return "-"
    if x < 0.05:
        return "0"
    if x < 100:
        return f"{x:.1f}"
    return f"{x:.1e}"


def main():
    lib = ctypes.CDLL(sys.argv[1])
    flt = None
    grp = None
    samples, warm = 21, 2
    a = sys.argv[2:]
    while a:
        if a[0] == "--group":
            grp = set(a[1].split(","))
            a = a[2:]
        elif a[0] == "--filter":
            flt = a[1]
            a = a[2:]
        elif a[0] == "--samples":
            samples = int(a[1])
            a = a[2:]
        else:
            raise SystemExit(f"unknown argument {a[0]}")
    lib.sb_ncases.restype = ctypes.c_size_t
    lib.sb_case_info.restype = ctypes.c_size_t
    lib.sb_case_info.argtypes = [ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
    lib.sb_case_inputs.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
    lib.sb_case_ours.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    np.seterr(all="ignore")
    buf = ctypes.create_string_buffer(4096)
    out = sys.stdout
    for ci in range(lib.sb_ncases()):
        n = lib.sb_case_info(ci, buf, 4096)
        group, op, params, nout, spec, metric = buf.raw[:n].decode().split("\t")
        if not spec:
            continue
        key = f"{group}|{op}|{params}"
        if (flt and flt not in key) or (grp and group not in grp):
            continue
        t0 = time.time()
        try:
            row = run(lib, ci, int(nout), spec, metric, samples, warm)
        except Exception as e:  # a bad spec or a SciPy failure: report it, go on
            print(f"ERROR {key}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if row is None:
            continue
        ours, ref, notes = row
        ratio = sig3(ours / ref) if math.isfinite(ref) else "-"
        out.write(
            "\t".join(["special", group, op, params, sig3(ours), "scipy", sig3(ref), ratio, notes])
            + "\n"
        )
        out.flush()
        print(f"{time.time() - t0:6.2f}s {key}", file=sys.stderr)


LABEL = {
    "rel": "diff vs ours in ulps",
    "scaled": "diff vs ours in ulps of max(|v|, median |v|)",
    "complex": "diff vs ours in eps of |w|",
}


def run(lib, ci, nout, spec, metric, samples, warm):
    parts = spec.split(":")
    name, args = parts[0], parts[1].split(",")
    outs = parts[2] if len(parts) > 2 else "0"
    f = fn_for(name, len(args))
    is_ufunc = isinstance(f, np.ufunc)

    P = 65536
    rec = np.zeros((P, 4))
    lib.sb_case_inputs(ci, rec.ctypes.data, P)

    def col(tok):
        if tok[0] == "x":
            return np.ascontiguousarray(rec[:, int(tok[1])])
        if tok[0] == "i":
            return np.ascontiguousarray(rec[:, int(tok[1])]).astype(np.int64)
        if tok == "c01":
            return rec[:, 0] + 1j * rec[:, 1]
        if tok[0] == "z":
            return rec[:, int(tok[1])].astype(np.complex128)
        try:
            return np.int64(int(tok))
        except ValueError:
            return np.float64(float(tok))

    cols = [col(t) for t in args]
    ours_out = np.zeros((P, 4))
    if is_ufunc:
        types = [np.result_type(c) for c in cols]
        sp_out = None
        res = f(*[c[:8] if isinstance(c, np.ndarray) else c for c in cols])
        res = res if isinstance(res, tuple) else (res,)
        sp_out = tuple(np.empty(P, dtype=r.dtype) for r in res)

    def ours_run(o, m):
        lib.sb_case_ours(ci, rec.ctypes.data + 32 * o, ours_out.ctypes.data + 32 * o, m)

    def sp_run(o, m):
        a = [c[o : o + m] if isinstance(c, np.ndarray) else c for c in cols]
        if is_ufunc:
            f(*a, out=tuple(x[o : o + m] for x in sp_out))
            return sp_out
        return f(*a)

    def tmeas(fn, m):
        t = time.perf_counter_ns()
        fn(0, m)
        return (time.perf_counter_ns() - t) / m

    # calibrate
    ours_run(0, 256)
    sp_run(0, 256)
    to = tmeas(ours_run, 2048)
    ts = tmeas(sp_run, 2048)
    # check
    nck = int(min(P, max(512, 60e6 / max(to, ts))))
    ours_run(0, nck)
    r = sp_run(0, nck)
    r = r if isinstance(r, tuple) else (r,)
    if outs == "c":
        z = np.asarray(r[0][:nck])
        comps = [z.real, z.imag]
    elif outs == "cr":
        comps = [np.asarray(r[0][:nck]).real]
    else:
        comps = [np.asarray(r[int(k)][:nck]) for k in outs.split(",")]
    worst = np.zeros(nck)
    bad = np.zeros(nck, dtype=bool)
    EPS, TINY = 2.220446049250313e-16, 2.2250738585072014e-308
    if metric == "complex":
        a = ours_out[:nck, 0] + 1j * ours_out[:nck, 1]
        b = comps[0] + 1j * comps[1]
        fin = np.isfinite(a) & np.isfinite(b)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        worst = np.where(fin & ~same, np.abs(a - b) / (np.maximum(np.abs(a), TINY) * EPS), 0.0)
        bad = ~fin & ~same
    else:
        for j, rc in enumerate(comps):
            a = ours_out[:nck, j]
            b = rc.astype(np.float64)
            fin = np.isfinite(a) & np.isfinite(b)
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            fl = TINY
            if metric == "scaled":
                aa = np.abs(a[np.isfinite(a)])
                if len(aa):
                    fl = max(float(np.sort(aa)[len(aa) // 2]), TINY)
            u = np.where(fin & ~same, np.abs(a - b) / (np.maximum(np.abs(a), fl) * EPS), 0.0)
            worst = np.maximum(worst, u)
            bad |= ~fin & ~same
    good = np.sort(worst[~bad])
    mism = int(bad.sum())

    def q(p):
        return good[int((len(good) - 1) * p)] if len(good) else float("nan")

    notes = []
    if NOTES.get(name):
        notes.append(NOTES[name])
    notes.append(f"SciPy {scipy.__version__} ufunc on arrays (NumPy's loop included)")
    notes.append(f"{LABEL[metric]}: max {fmt_u(q(1.0))} p99 {fmt_u(q(0.99))} median {fmt_u(q(0.5))} (n={nck})")
    if mism:
        notes.append(f"NaN/inf mismatches {mism}/{nck}")
    if mism * 5 > nck:
        notes.append(f"not timed: NaN/inf where ours is finite (or the reverse) on {mism}/{nck}")
        return (float("nan"), float("nan"), "; ".join(notes))

    tf, tsl = min(to, ts), max(to, ts)
    m = int(min(P, max(256, math.ceil(1.2e6 / tf))))
    if m * tsl > 30e6:
        m = int(max(256, math.ceil(30e6 / tsl)))
    m = min(m, P)
    ko = max(1, math.ceil(1.0e6 / (m * to)))
    ks = max(1, math.ceil(1.0e6 / (m * ts)))
    so, ss = [], []
    for rnd in range(warm + samples):
        o = (rnd * m) % (P - m + 1)
        order = [0, 1] if rnd % 2 == 0 else [1, 0]
        for which in order:
            if which == 0:
                t = time.perf_counter_ns()
                for _ in range(ko):
                    ours_run(o, m)
                dt = (time.perf_counter_ns() - t) / (m * ko)
                if rnd >= warm:
                    so.append(dt)
            else:
                t = time.perf_counter_ns()
                for _ in range(ks):
                    sp_run(o, m)
                dt = (time.perf_counter_ns() - t) / (m * ks)
                if rnd >= warm:
                    ss.append(dt)
    so, ss = np.array(so), np.array(ss)
    mo, ms = float(np.median(so)), float(np.median(ss))
    sprd = lambda v, med: (np.percentile(v, 75) - np.percentile(v, 25)) / med
    if max(sprd(so, mo), sprd(ss, ms)) > 0.10:
        notes.append(f"spread (IQR/median) ours {100 * sprd(so, mo):.0f}% ref {100 * sprd(ss, ms):.0f}%")
    return (mo, ms, "; ".join(notes))


if __name__ == "__main__":
    main()

"""Reference data of the AAA tests (`../aaa.rs`): scipy.interpolate.AAA (SciPy 1.18) on the
problems of the paper and of SciPy's documentation, written to `aaa_<name>.txt`.

Run from this directory with `python3 gen_aaa.py`; the data are deterministic.

Every number is the 16 hexadecimal digits of the bits of an f64. A file holds, one item a line:
    samples M        then M lines `x.re x.im f.re f.im` (sorted, distinct, as SciPy keeps them)
    terms m          then m lines: the index among the samples of each support point, in order
    weights m        then m lines `w.re w.im`, the weights divided by the one largest in size
    errors k         then k lines: SciPy's error on the samples after each step
    points P         then P lines `t.re t.im r.re r.im`: SciPy's approximation at test points
    poles q          then q lines `p.re p.im`: SciPy's poles (QZ), for information
All runs are with clean_up=False, since the clean-up's decisions rest on residues of spurious
poles, which neither SciPy's QZ nor any other method computes to more than a few digits.
"""

import struct
import warnings

import numpy as np
from scipy.interpolate import AAA
from scipy.special import gamma


def bits(v):
    return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def write(name, z, f, t, **kw):
    z = np.asarray(z)
    f = np.asarray(f)
    # SciPy's own preparation: finite values, sorted distinct points
    keep = np.isfinite(f)
    z, f = z[keep], f[keep]
    z, uni = np.unique(z, return_index=True)
    f = f[uni]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = AAA(z, f, clean_up=False, **kw)
    index = {complex(v): i for i, v in enumerate(z)}
    sp = [index[complex(v)] for v in r.support_points]
    w = r.weights / r.weights[np.argmax(np.abs(r.weights))]
    rt = r(np.asarray(t))
    lines = [f"samples {len(z)}"]
    for a, b in zip(z, f):
        a, b = complex(a), complex(b)
        lines.append(f"{bits(a.real)} {bits(a.imag)} {bits(b.real)} {bits(b.imag)}")
    lines.append(f"terms {len(sp)}")
    lines += [str(i) for i in sp]
    lines.append(f"weights {len(w)}")
    lines += [f"{bits(complex(v).real)} {bits(complex(v).imag)}" for v in w]
    lines.append(f"errors {len(r.errors)}")
    lines += [bits(e) for e in r.errors]
    lines.append(f"points {len(t)}")
    for a, b in zip(t, rt):
        a, b = complex(a), complex(b)
        lines.append(f"{bits(a.real)} {bits(a.imag)} {bits(b.real)} {bits(b.imag)}")
    poles = r.poles()
    lines.append(f"poles {len(poles)}")
    lines += [f"{bits(complex(p).real)} {bits(complex(p).imag)}" for p in poles]
    with open(f"aaa_{name}.txt", "w") as out:
        out.write("\n".join(lines) + "\n")
    print(f"{name}: {len(z)} samples, {len(sp)} terms, error {r.errors[-1]:.2e}")


def main():
    rng = np.random.default_rng(56)
    x = np.linspace(-1, 1, 1000)
    t = np.sort(rng.uniform(-0.999, 0.999, 50))
    write("exp", x, np.exp(x), t)
    xi = -1 + (2 * np.arange(1000) + 1) / 1000
    write("tan", xi, np.tan(np.pi * xi / 2), t)
    write("abs", x, np.abs(x), t)
    xs = np.linspace(0, 1, 1000)
    write("sqrt", xs, np.sqrt(xs), np.sort(rng.uniform(0.001, 0.999, 50)))
    write("runge", x, 1 / (1 + 25 * x * x), t)
    xg = np.linspace(-1.5, 1.5, 100)
    write("gamma", xg, gamma(xg), np.sort(rng.uniform(-1.49, 1.49, 50)))
    s = np.linspace(0, 1, 1000)
    zs = np.exp(-0.5 + s + 15j * np.pi * s)
    ts = np.sort(rng.uniform(0.001, 0.999, 50))
    write("spiral", zs, np.tan(np.pi * zs / 2), np.exp(-0.5 + ts + 15j * np.pi * ts))
    zu = np.exp(2j * np.pi * np.arange(1000) / 1000)
    tu = np.exp(2j * np.pi * rng.uniform(0, 1, 50)) * rng.uniform(0.9, 1.1, 50)
    write("circle", zu, np.log(2 + zu**4) / (1 - 16 * zu**4), tu)


if __name__ == "__main__":
    main()

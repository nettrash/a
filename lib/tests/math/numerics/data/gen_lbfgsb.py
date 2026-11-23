#!/usr/bin/env python3
"""Reference results for lib/tests/math/numerics/lbfgsb.rs.

Every problem of the test file is solved here by scipy.optimize.minimize(method="L-BFGS-B")
with the same memory size and the same two tolerances the test passes (gtol = pgtol,
ftol = the relative reduction tolerance, maxls = 20, maxiter = maxfun = 15000), and the
iterations, the evaluations, the value, the projected gradient, the test that stopped it, the
active set and (for up to 25 variables) the minimiser are printed as the Rust table SCIPY of the
test file. The KKT points of the boxed quadratics are computed independently by the bounded
least squares solver BVLS (scipy.optimize.lsq_linear) and printed as the table KKT, and what
SciPy does on the large box quadratic of the first review as the table LEVEL, the first
iterates of SciPy on some of the problems as the table EARLY, and SciPy's results on three
problems of the second review where the rounding of f decides the stop as ROUNDING and
LSE_SCIPY.

    python3 gen_lbfgsb.py > out.txt     # then paste the six tables into lbfgsb.rs

Written against SciPy 1.18 and NumPy 2.3; SciPy's L-BFGS-B is the reference code of Zhu, Byrd,
Lu and Nocedal, version 3.0, which is what the test compares with.
"""
import math
import sys

import numpy as np
from scipy.optimize import lsq_linear, minimize

INF = math.inf


def rosen(x):
    x = np.asarray(x)
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosen_grad(x):
    x = np.asarray(x)
    g = np.zeros_like(x)
    t = x[1:] - x[:-1] ** 2
    g[:-1] += -400.0 * x[:-1] * t - 2.0 * (1.0 - x[:-1])
    g[1:] += 200.0 * t
    return g


def rosen_start(n):
    return np.array([-1.2 if i % 2 == 0 else 1.0 for i in range(n)])


def driver1(x):
    """The problem of the sample driver of the reference code (Zhu et al. 1997)."""
    f = 0.25 * (x[0] - 1.0) ** 2
    f += np.sum((x[1:] - x[:-1] ** 2) ** 2)
    return 4.0 * float(f)


def driver1_grad(x):
    n = len(x)
    g = np.zeros(n)
    t1 = x[1] - x[0] ** 2
    g[0] = 2.0 * (x[0] - 1.0) - 16.0 * x[0] * t1
    for i in range(1, n - 1):
        t2 = t1
        t1 = x[i + 1] - x[i] ** 2
        g[i] = 8.0 * t2 - 16.0 * x[i] * t1
    g[n - 1] = 8.0 * t1
    return g


def driver1_bounds(n):
    # Odd-numbered variables (1-based) in [1, 100], even-numbered ones in [-100, 100].
    return [(1.0, 100.0) if i % 2 == 0 else (-100.0, 100.0) for i in range(n)]


def tridiagonal(n):
    a = 2.0 * np.eye(n)
    for i in range(n - 1):
        a[i, i + 1] = a[i + 1, i] = -1.0
    return a


def quad_b(n):
    # The right-hand side that puts the unconstrained minimiser partly outside [0, 1].
    return np.array([math.sin(1.7 * (i + 1)) for i in range(n)])


def scaled_quad(n, cond):
    d = np.array([cond ** (i / (n - 1)) for i in range(n)])
    c = np.array([math.sin(i + 1.0) for i in range(n)])
    return d, c


def hs5(x):
    return math.sin(x[0] + x[1]) + (x[0] - x[1]) ** 2 - 1.5 * x[0] + 2.5 * x[1] + 1.0


def hs5_grad(x):
    c = math.cos(x[0] + x[1])
    return np.array([c + 2.0 * (x[0] - x[1]) - 1.5, c - 2.0 * (x[0] - x[1]) + 2.5])


def hs45(x):
    return 2.0 - float(np.prod(x)) / 120.0


def hs45_grad(x):
    return np.array([-float(np.prod(np.delete(x, i))) / 120.0 for i in range(5)])


def hs110(x):
    return float(np.sum(np.log(x - 2.0) ** 2 + np.log(10.0 - x) ** 2) - np.prod(x) ** 0.2)


def hs110_grad(x):
    p = float(np.prod(x)) ** 0.2
    return 2.0 * np.log(x - 2.0) / (x - 2.0) - 2.0 * np.log(10.0 - x) / (10.0 - x) - 0.2 * p / x


def brown(x):
    return (x[0] - 1e6) ** 2 + (x[1] - 2e-6) ** 2 + (x[0] * x[1] - 2.0) ** 2


def brown_grad(x):
    t = x[0] * x[1] - 2.0
    return np.array([2.0 * (x[0] - 1e6) + 2.0 * t * x[1], 2.0 * (x[1] - 2e-6) + 2.0 * t * x[0]])


def problems():
    out = []

    def add(name, f, g, x0, bounds, m=10, pgtol=1e-5, ftol=2.220446049250313e-9, note=""):
        out.append(dict(name=name, f=f, g=g, x0=np.array(x0, float), bounds=bounds, m=m,
                        pgtol=pgtol, ftol=ftol, note=note))

    for n in (2, 10, 50, 100):
        add(f"rosen{n}", rosen, rosen_grad, rosen_start(n), [(-INF, INF)] * n)
    # Tight tolerances: the run goes on to the minimiser itself.
    for n in (2, 10, 100):
        add(f"rosen{n}_tight", rosen, rosen_grad, rosen_start(n), [(-INF, INF)] * n,
            pgtol=1e-10, ftol=0.0)
    # Bounded Rosenbrock: every other variable may not exceed 0.5.
    for n in (2, 10, 100):
        b = [(-2.0, 0.5) if i % 2 == 0 else (-2.0, 2.0) for i in range(n)]
        add(f"rosen{n}_box", rosen, rosen_grad, rosen_start(n), b)
    for n in (2, 10, 100):
        b = [(-2.0, 0.5) if i % 2 == 0 else (-2.0, 2.0) for i in range(n)]
        add(f"rosen{n}_box_tight", rosen, rosen_grad, rosen_start(n), b, pgtol=1e-10, ftol=0.0)
    # Memory sizes other than 10.
    for m in (1, 3, 5, 17):
        b = [(-2.0, 0.5) if i % 2 == 0 else (-2.0, 2.0) for i in range(10)]
        add(f"rosen10_box_m{m}", rosen, rosen_grad, rosen_start(10), b, m=m)
        add(f"rosen10_m{m}", rosen, rosen_grad, rosen_start(10), [(-INF, INF)] * 10, m=m)
    add("driver1", driver1, driver1_grad, np.full(25, 3.0), driver1_bounds(25), m=5)
    add("driver1_tight", driver1, driver1_grad, np.full(25, 3.0), driver1_bounds(25), m=5,
        pgtol=1e-10, ftol=0.0)
    # The convex quadratic x'Ax/2 - b'x in a box.
    for n in (5, 20, 100):
        a, bb = tridiagonal(n), quad_b(n)
        add(f"quad{n}_box", lambda x, a=a, bb=bb: float(0.5 * x @ a @ x - bb @ x),
            lambda x, a=a, bb=bb: a @ x - bb, np.full(n, 0.5), [(0.0, 1.0)] * n,
            pgtol=1e-10, ftol=0.0)
    # The same with only lower bounds, only upper bounds, and none.
    a, bb = tridiagonal(20), quad_b(20)
    fq = lambda x, a=a, bb=bb: float(0.5 * x @ a @ x - bb @ x)
    gq = lambda x, a=a, bb=bb: a @ x - bb
    add("quad20_lower", fq, gq, np.full(20, 0.5), [(0.0, INF)] * 20, pgtol=1e-10, ftol=0.0)
    add("quad20_upper", fq, gq, np.full(20, 0.5), [(-INF, 0.25)] * 20, pgtol=1e-10, ftol=0.0)
    add("quad20_free", fq, gq, np.full(20, 0.5), [(-INF, INF)] * 20, pgtol=1e-10, ftol=0.0)
    add("quad20_interior", fq, gq, np.full(20, 0.5), [(-50.0, 50.0)] * 20, pgtol=1e-10, ftol=0.0)
    add("quad20_outside", fq, gq, np.full(20, 7.5), [(0.0, 1.0)] * 20, pgtol=1e-10, ftol=0.0,
        note="the start is outside the box")
    # A vertex: a linear function with a little curvature, every variable ends on a bound.
    cv = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    add("vertex6", lambda x: float(cv @ x + 0.05 * x @ x), lambda x: cv + 0.1 * x,
        np.zeros(6), [(-1.0, 2.0)] * 6, pgtol=1e-10, ftol=0.0)
    # Badly scaled: the curvatures span `cond`.
    for cond, label in ((1e6, "1e6"), (1e12, "1e12")):
        d, c = scaled_quad(10, cond)
        add(f"scaled10_{label}", lambda x, d=d, c=c: float(0.5 * np.sum(d * (x - c) ** 2)),
            lambda x, d=d, c=c: d * (x - c), np.zeros(10), [(-0.5, 0.5)] * 10,
            pgtol=1e-10, ftol=0.0)
    add("brown_box", brown, brown_grad, [1.0, 1.0], [(0.0, 1e7), (0.0, 1.0)], pgtol=1e-5,
        ftol=0.0)
    add("brown_box_cut", brown, brown_grad, [1.0, 1.0], [(0.0, 5e5), (0.0, 1.0)], pgtol=1e-5,
        ftol=0.0)
    # Hock and Schittkowski's problems with bounds only.
    add("hs5", hs5, hs5_grad, [0.0, 0.0], [(-1.5, 4.0), (-3.0, 3.0)], pgtol=1e-10, ftol=0.0)
    add("hs45", hs45, hs45_grad, [2.0] * 5, [(0.0, float(i + 1)) for i in range(5)],
        pgtol=1e-10, ftol=0.0)
    add("hs110", hs110, hs110_grad, [9.0] * 10, [(2.001, 9.999)] * 10, pgtol=1e-8, ftol=0.0)
    # Nearly linear: c'x + 1e-16 x'x below 2. Along every step s'y is some 1e-16 of -g's, which the
    # curvature test leaves out of the memory.
    cl = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    add("nearly_linear6", lambda x: float(cl @ x + 1e-16 * x @ x), lambda x: cl + 2e-16 * x,
        np.full(6, 0.3), [(-INF, 2.0)] * 6, pgtol=1e-10, ftol=0.0)
    # Three random quadratics x'Ax/2 - b'x of the first review's set (its gen_problems.py, seed
    # 20261121, problems quad256, quad160 and quad37), whose first iterates take the cut back to
    # the box of Morales and Nocedal and the update of the curvature at a breakpoint of the
    # Cauchy search.
    for name, x0, lo, hi, b, a, m, pgtol, ftol in REVIEW_QUADS:
        a, b = np.array(a, float).reshape(len(x0), len(x0)), np.array(b, float)
        add(name, lambda x, a=a, b=b: float(0.5 * x @ a @ x - b @ x), lambda x, a=a, b=b: a @ x - b,
            x0, list(zip(lo, hi)), m=m, pgtol=pgtol, ftol=ftol)
    return out


REVIEW_QUADS = [
    ("review_quad256", [2.245440957823526, -1.517897429612824], [1.0024952564033676, -INF], [INF, INF],
     [175.5725895245648, -47.77808725557092],
     [265.78715457508673, -71.57180612296526, -71.57180612296526, 20.40588688220658], 5, 1e-10, 0.0),
    ("review_quad160",
     [2.8120771454314917, -3.2787311963547703, 1.5244278159586215, -0.768047313909714, -1.8048064076785615],
     [-0.6643240095924081, -INF, -1.543500169733765, 0.2159848323577039, -0.7822393795124747],
     [0.09394305744467335, INF, -0.6316374161751908, INF, 0.5726591865954651],
     [117047.6042277047, 128873.48182632017, 127858.81160176164, -68163.40577399344, 130754.54907105409],
     [112442.77769307228, 118645.44456407262, 123016.19255147915, -65665.11998027004, 124929.48976177708,
      118645.44456407262, 155688.89964608647, 126261.08099195866, -67248.4784480781, 133169.29077524738,
      123016.19255147915, 126261.08099195866, 135637.82802245175, -72407.34083923159, 137007.3726871055,
      -65665.11998027004, -67248.4784480781, -72407.34083923159, 38692.05932680552, -73019.97675276316,
      124929.48976177708, 133169.29077524738, 137007.3726871055, -73019.97675276316, 139506.73847073436],
     10, 1e-10, 1e-12),
    ("review_quad37",
     [-1.5841177879764983, -2.259907726668094, -1.3150547247781654, -3.290999620879834, 1.1837683631834728],
     [-2.149658801293996, -1.0924262194088137, -1.6054073040130068, -1.9437434483263005, -INF],
     [INF, INF, INF, -0.309230955418588, -1.1091653069354457],
     [-119003.02748951523, -324465.3783441008, 76317.2196359906, -40912.94375681026, 24587.555040962456],
     [34533.08082564689, 86238.92344947511, -16214.781863674481, 7417.870998925055, -4865.3327226419315,
      86238.92344947511, 240313.24869211408, -57689.133329504584, 30612.795825426485, -18864.072879603387,
      -16214.781863674481, -57689.133329504584, 26150.267360797818, -16774.067897761925, 8091.742285898617,
      7417.870998925055, 30612.795825426485, -16774.067897761925, 12956.515494991674, -6138.8625890455605,
      -4865.3327226419315, -18864.072879603387, 8091.742285898617, -6138.8625890455605, 3494.4918760503956],
     5, 1e-8, 0.0),
]


def projected_gradient(x, g, bounds):
    pg = np.zeros(len(x))
    for i, (lo, hi) in enumerate(bounds):
        if g[i] < 0:
            pg[i] = max(x[i] - hi, g[i])
        else:
            pg[i] = min(x[i] - lo, g[i])
    return float(np.max(np.abs(pg))) if len(x) else 0.0


def active(x, bounds):
    s = ""
    for xi, (lo, hi) in zip(x, bounds):
        if lo == hi:
            s += "F"
        elif xi == lo:
            s += "L"
        elif xi == hi:
            s += "U"
        else:
            s += "."
    return s


# The problems whose first iterates the test compares one by one with SciPy's, and how many.
EARLY = ("rosen10", "rosen10_box", "rosen10_box_m3", "driver1", "quad20_box", "quad20_lower",
         "quad20_outside", "vertex6", "scaled10_1e6", "brown_box", "hs45", "hs110", "review_quad256",
         "review_quad160", "review_quad37")
EARLY_ITERATIONS = 5


def early():
    """SciPy's first iterates: after each iteration the number of evaluations so far, the
    starting one included, and the point (the callback is called once an iteration)."""
    print("// Made by data/gen_lbfgsb.py: SciPy's first iterates, each the evaluations made so far")
    print("// (the first included) and the point.")
    print("const EARLY: &[(&str, Iterates)] = &[")
    for p in problems():
        if p["name"] not in EARLY:
            continue
        calls = [0]
        steps = []

        def counted(x, f=p["f"]):
            calls[0] += 1
            return f(x)

        def record(xk):
            if len(steps) < EARLY_ITERATIONS:
                steps.append((calls[0], np.array(xk, dtype=float)))

        minimize(counted, p["x0"], jac=p["g"], method="L-BFGS-B", bounds=p["bounds"], callback=record,
                 options=dict(maxcor=p["m"], gtol=p["pgtol"], ftol=p["ftol"], maxiter=15000,
                              maxfun=15000, maxls=20))
        rows = ", ".join(f"({c}, &[{', '.join(repr(float(v)) for v in x)}])" for c, x in steps)
        print(f'    ("{p["name"]}", &[{rows}]),')
    print("];")


# Three of the second review's 800 random problems (its generator lbstress/gen_lb.py; the data are
# copied from its output, ids 625, 635 and 149), where the stop depends on the rounding of f:
# two quadratics on which SciPy ends on the "reduction" of a step that leaves f level, and a
# log-sum-exp with a ridge whose f32 run the test compares with SciPy's f64 value. f and g are
# computed as the review's generator computes them.
def rounding():
    # id 625: quad, n = 2, memory 17, pgtol 1e-09, ftol 2.220446049250313e-13
    x0_625 = [
        0.03369087526957654, -3.2963086401551287,
    ]
    lo_625 = [
        -INF, -4.040565963435006,
    ]
    hi_625 = [
        INF, INF,
    ]
    data_625 = [
        35512.66208772069, -53645.76304781647, -53645.76304781647, 81041.07877394135,
        127414.874779647, -192480.90951089462,
    ]
    # id 635: quad, n = 3, memory 17, pgtol 1e-09, ftol 2.220446049250313e-09
    x0_635 = [
        -2.975667530398329, -5.281298660514384, -0.8181495441580848,
    ]
    lo_635 = [
        -1.1415454326625887, 1.2538402561983948, -INF,
    ]
    hi_635 = [
        0.3701705394272141, 1.2538402561983948, 2.1856618964189924,
    ]
    data_635 = [
        4800157.1113577755, -18600428.428781837, -4023175.119721802, -18600428.428781837,
        74082911.39016782, 16037644.352393784, -4023175.119721802, 16037644.352393784,
        3471961.9856460625, -22442422.024965048, 88695223.16868874, 19196304.888220012,
    ]
    # id 149: lse, n = 40, memory 1, pgtol 1e-05, ftol 2.220446049250313e-09
    x0_149 = [
        0.5553611930039751, 1.60312191376013, 1.3267902489963255, -2.984052653201408,
        -2.8504699391751545, -1.1133908289422794, 1.8650540943563643, 1.4745936249000124,
        -0.07501941011007457, -0.15582157740046437, 1.1067522517672748, 1.339386881819725,
        -1.036957491121891, -0.4581339072150094, 3.913653959514196, -2.1791160072528273,
        -0.0949103842908305, -2.6566156448335865, 0.2726935855420944, 0.46786004927814545,
        0.42113464163279163, -0.5505737246063112, -1.7139866620647644, -2.3560432363054327,
        1.0180821473958996, -0.7427089301652298, 1.6860446938347051, -2.171111883303836,
        0.020824380919556698, -0.4747592833342588, 0.0795045405670847, -3.5977175813156457,
        -1.5938373309818572, -0.7139869523883862, -0.3334512379439324, 0.3959199490403062,
        1.0626166099134302, 0.2810315389856055, -1.2832810459782473, -0.7442004912184835,
    ]
    lo_149 = [
        0.059677116552124126, 0.23565116258179772, -1.3641154872203252, -0.6387028342061859,
        -0.4794572042020986, -INF, -0.3579274380663337, -1.3639675042577322, -0.8369798682407514,
        -0.49139218974843724, -INF, -0.6254662387050021, -0.49396085606115075,
        -2.1329425914166533, -1.6075645253268447, -INF, -INF, -1.9929783042035296, -INF, -INF,
        -0.012148884536409324, -0.7778861765069175, -0.6172423489158643, 0.5433849758311058, -INF,
        0.24854867270786293, -INF, 0.1757353550506198, -0.8098681358950454, -INF,
        -1.5543869389675153, -0.3105598208148608, -0.27229162978613947, 0.016450464777214124,
        0.35986662800643077, -INF, -0.7416826139691779, -2.0400774319065773, -INF, -INF,
    ]
    hi_149 = [
        INF, 0.6990189380785382, -0.8479239962551122, INF, 0.05696159162346342, INF, INF, INF,
        -0.8369798682407514, 0.48732423283104676, INF, 0.16612467325857525, INF, INF,
        -0.5458841061960351, INF, INF, 0.9494791428927555, 0.2571813905235761, INF,
        0.6356119096061017, 0.6858249388279185, 1.787174794956803, INF, 1.343626732171001,
        0.24854867270786293, INF, 0.22150773875047644, 0.9977249607873304, 1.2412714639640303,
        0.4793218645119126, 2.4289077510075754, INF, 0.560401904354155, INF, 0.8562913258724972,
        1.0856139943546463, -0.03549897291448347, INF, INF,
    ]
    data_149 = [
        2.0, 0.0017604017596437808, -0.9258142727967451, -0.6716162597490568,
        -0.24880784789440263, 1.0012382093670922, 0.13788221510307164, -0.46642306118893245,
        -0.1083823120115684, -0.2777632818874784, 1.4354922263667869, 1.1635938638673375,
        2.4359001354071013, 0.8064173610655561, 0.22085526491009572, -0.06501945665863153,
        -0.7317076599452548, 0.5017279872295921, 0.4091004145940108, -0.8803862339201066,
        -0.8379718038820976, 0.2746411289086623, -0.3366024310414633, -0.4997195086393795,
        0.7191710559742404, -0.07499268134951025, 1.0641580791827587, -0.48501219832580855,
        -0.38469039580265735, -0.7990226767338368, -1.3977163985381598, -0.44628823116115357,
        -0.8471490865343269, 1.2046382654913832, 1.193002743403995, -0.1715876011799664,
        -0.5549268755425227, 1.1101737805094145, -2.2881811352683705, 0.362848087448214,
        0.7268365894830385, -0.6458140332580248, -0.6730067856762144, 1.6967328779956101,
        -0.3454859523636702, 1.179313944214341, 1.6493494325276408, 0.46843693633283895,
        -0.5075659869255497, 0.34204162490517054, -0.10458589081661161, -1.5016163578724906,
        2.2452579128671477, 0.33538787570848433, -1.167114372160207, 0.27599062180135964,
        -0.8593622820032486, -1.4665627964283956, -1.5105493032467734, -1.1783502859427715,
        -0.6872838105836985, 0.5877896496574923, -1.167547307900751, -0.12364096707313929,
        1.1324073083246287, 0.7530579041348795, -0.050919084390707026, 0.5881562987109881,
        -1.529176158928944, 0.7366533556679287, 1.1069181838799935, -0.8309800777480725,
        1.11765529125593, -0.6458049548441359, 0.5282070842141305, -0.48894767137376005,
        -0.8493751207772174, 0.8386190764607233, -0.8363796224133611, 1.4847394083801162,
        0.4299031060072906, -0.5667138614126938,
    ]

    def quadratic(data, n):
        a = np.array(data[: n * n]).reshape(n, n)
        b = np.array(data[n * n:])
        return a, b, (lambda x: 0.5 * x @ (a @ x) - b @ x), (lambda x: a @ x - b)

    def lse(data, n):
        kk, mu = int(data[0]), data[1]
        w = np.array(data[2:]).reshape(kk, n)

        def f(x):
            z = w @ x
            zm = z.max()
            return zm + np.log(np.sum(np.exp(z - zm))) + 0.5 * mu * x @ x

        def g(x):
            z = w @ x
            p = np.exp(z - z.max())
            p /= p.sum()
            return w.T @ p + mu * x

        return f, g

    def run(f, g, x0, lo, hi, m, pgtol, ftol):
        bounds = [(None if not math.isfinite(l) else l, None if not math.isfinite(u) else u)
                  for l, u in zip(lo, hi)]
        r = minimize(f, np.array(x0), jac=g, method="L-BFGS-B", bounds=bounds,
                     options={"maxcor": m, "ftol": ftol, "gtol": pgtol, "maxiter": 15000, "maxfun": 15000})
        pg = projected_gradient(r.x, g(r.x), list(zip(lo, hi)))
        return r, pg

    print("// Made by data/gen_lbfgsb.py: SciPy's L-BFGS-B on the second review's quadratics 625 and")
    print("// 635: whether it stopped on the projected gradient, its value and projected gradient there;")
    print("// and the KKT point by BVLS (the fixed variable eliminated first).")
    print("const ROUNDING: &[(usize, bool, f64, f64, &[f64])] = &[")
    for pid, x0, lo, hi, data, n, m, pgtol, ftol in (
            (625, x0_625, lo_625, hi_625, data_625, 2, 17, 1e-09, 2.220446049250313e-13),
            (635, x0_635, lo_635, hi_635, data_635, 3, 17, 1e-09, 2.220446049250313e-09)):
        a, b, f, g = quadratic(data, n)
        r, pg = run(f, g, x0, lo, hi, m, pgtol, ftol)
        on_pg = "PROJECTED GRADIENT" in str(r.message).upper()
        lo_, hi_ = np.array(lo), np.array(hi)
        fixed = lo_ == hi_
        free = ~fixed
        kkt = np.where(fixed, lo_, 0.0)
        af = a[np.ix_(free, free)]
        bf = b[free] - a[np.ix_(free, fixed)] @ lo_[fixed] if fixed.any() else b[free]
        chol = np.linalg.cholesky(af)
        s = lsq_linear(chol.T, np.linalg.solve(chol, bf), bounds=(lo_[free], hi_[free]), method="bvls",
                       tol=1e-15, max_iter=10000)
        kkt[free] = s.x
        print(f"    ({pid}, {'true' if on_pg else 'false'}, {float(r.fun)!r}, {pg!r}, "
              f"&[{', '.join(repr(float(v)) for v in kkt)}]),")
    print("];")
    f, g = lse(data_149, 40)
    r, pg = run(f, g, x0_149, lo_149, hi_149, 1, 1e-05, 2.220446049250313e-09)
    print("// Made by data/gen_lbfgsb.py: SciPy's L-BFGS-B in f64 on the second review's log-sum-exp")
    print("// 149: whether it stopped on the reduction test, its value and its projected gradient.")
    print(f"const LSE_SCIPY: (bool, f64, f64) = ({'true' if 'REDUCTION' in str(r.message).upper() else 'false'}, "
          f"{float(r.fun)!r}, {pg!r});")


def main():
    print("// Made by data/gen_lbfgsb.py (SciPy 1.18): name, iterations, evaluations, value,")
    print("// projected gradient, whether it stopped on the projected gradient, active set, x.")
    print("const SCIPY: &[Reference] = &[")
    for p in problems():
        r = minimize(p["f"], p["x0"], jac=p["g"], method="L-BFGS-B", bounds=p["bounds"],
                     options=dict(maxcor=p["m"], gtol=p["pgtol"], ftol=p["ftol"], maxiter=15000,
                                  maxfun=15000, maxls=20))
        pg = projected_gradient(r.x, p["g"](r.x), p["bounds"])
        on_pg = "PROJECTED GRADIENT" in r.message
        xs = ", ".join(repr(float(v)) for v in r.x) if len(r.x) <= 25 else ""
        print(f'    Reference {{ name: "{p["name"]}", iterations: {r.nit}, evaluations: {r.nfev}, '
              f'value: {float(r.fun)!r}, projected: {pg!r}, on_pg: {"true" if on_pg else "false"}, '
              f'active: "{active(r.x, p["bounds"])}", x: &[{xs}] }},')
    print("];")
    # The KKT point of the boxed quadratics, independently: the bounded least squares
    # min |Rx - R^-T b| with A = R'R, by the active set method BVLS.
    print("// Made by data/gen_lbfgsb.py: the KKT points of the boxed quadratics by BVLS.")
    print("const KKT: &[(usize, &[f64])] = &[")
    for n in (5, 20, 100):
        a, bb = tridiagonal(n), quad_b(n)
        r_ = np.linalg.cholesky(a).T
        rhs = np.linalg.solve(r_.T, bb)
        s = lsq_linear(r_, rhs, bounds=(0.0, 1.0), method="bvls", tol=1e-15)
        g = a @ s.x - bb
        assert projected_gradient(s.x, g, [(0.0, 1.0)] * n) < 1e-13
        print(f"    ({n}, &[{', '.join(repr(float(v)) for v in s.x)}]),")
    print("];")
    # The box quadratic of the first review: A = tridiag(-1, 3, -1), b_i = 3 sin(1.7 i) + 1,
    # x in [0, 1]^n from 0.5, ftol 0. Near the end f stops changing within its rounding while
    # the projected gradient still falls; the reference code takes such steps.
    print("// Made by data/gen_lbfgsb.py: the box quadratic tridiag(-1, 3, -1) of the review at")
    print("// n, pgtol and ftol 0: iterations, whether it stopped on the projected gradient, and")
    print("// the projected gradient there.")
    print("const LEVEL: &[(usize, f64, usize, bool, f64)] = &[")
    for n, pgtol in ((1000, 1e-9), (10000, 1e-7)):
        b = 3.0 * np.sin(1.7 * np.arange(1, n + 1)) + 1.0

        def ax(x):
            y = 3.0 * x
            y[1:] -= x[:-1]
            y[:-1] -= x[1:]
            return y

        r = minimize(lambda x: float(0.5 * x @ ax(x) - b @ x), np.full(n, 0.5), jac=lambda x: ax(x) - b,
                     method="L-BFGS-B", bounds=[(0.0, 1.0)] * n,
                     options=dict(maxcor=10, gtol=pgtol, ftol=0.0, maxiter=15000, maxfun=15000, maxls=20))
        pg = projected_gradient(r.x, ax(r.x) - b, [(0.0, 1.0)] * n)
        on_pg = "PROJECTED GRADIENT" in r.message
        print(f"    ({n}, {pgtol!r}, {r.nit}, {'true' if on_pg else 'false'}, {pg!r}),")
    print("];")
    early()
    rounding()


if __name__ == "__main__":
    main()

"""Generates tables.rs, the coefficients of the erf area: minimax fits by a discrete Remez exchange
in mpmath at 160 bits. Run from this directory: `python3 gen_tables.py` (a few minutes; the fits
are cached as erf_tables_cache.json in the temporary directory, which can be deleted)."""
import mpmath as m, sys, json, os, tempfile, types

# ---- the discrete Remez exchange ----

def cheb_grid(a, b, n):
    a=m.mpf(a); b=m.mpf(b)
    return [ (a+b)/2 - (b-a)/2*m.cos(m.pi*i/(n-1)) for i in range(n)]

def peval(c, r):
    s = m.mpf(0)
    for cj in reversed(c): s = s*r + cj
    return s

def minimax(rs, fs, deg, rel=True, iters=12, powers=None, basis=None, weights=None):
    """rs: grid points (local variable), fs: target values. Returns (coefs, maxerr).
    basis: optional list of functions phi_j(r) (default r^j)."""
    N = len(rs)
    if basis is None:
        basis = [ (lambda j: (lambda r: r**j))(j) for j in range(deg+1)]
    nb = len(basis)
    B = [[phi(r) for phi in basis] for r in rs]
    W = weights if weights is not None else [ (abs(f) if rel else m.mpf(1)) for f in fs]
    # initial reference: chebyshev-like spread indices
    idx = [ int(round((N-1)*(1-m.cos(m.pi*i/(nb)))/2)) for i in range(nb+1)]
    idx = sorted(set(idx))
    while len(idx) < nb+1:
        for k in range(N):
            if k not in idx: idx.append(k); idx.sort(); break
    best = None
    for it in range(iters):
        A = m.matrix(nb+1, nb+1); y = m.matrix(nb+1, 1)
        for i,k in enumerate(idx):
            for j in range(nb): A[i,j] = B[k][j]
            A[i,nb] = (-1)**i * W[k]
            y[i] = fs[k]
        try:
            sol = m.lu_solve(A, y)
        except ZeroDivisionError:
            break
        c = [sol[j] for j in range(nb)]
        err = [ (sum(c[j]*B[k][j] for j in range(nb)) - fs[k])/W[k] for k in range(N)]
        mx = max(abs(e) for e in err)
        if best is None or mx < best[1]: best = (c, mx)
        E = abs(sol[nb])
        if mx <= E*(1+m.mpf(10)**-6): break
        # find local extrema
        ext = []
        for k in range(N):
            l = err[k-1] if k>0 else None; r_ = err[k+1] if k<N-1 else None
            e = err[k]
            if (l is None or abs(e)>=abs(l) or m.sign(e)!=m.sign(l)) and (r_ is None or abs(e)>=abs(r_) or m.sign(e)!=m.sign(r_)):
                if (l is None or (abs(e)>=abs(l) if m.sign(e)==m.sign(l) else True)) and (r_ is None or (abs(e)>=abs(r_) if m.sign(e)==m.sign(r_) else True)):
                    ext.append(k)
        # merge same-sign runs keeping max
        merged = []
        for k in ext:
            if merged and m.sign(err[merged[-1]]) == m.sign(err[k]):
                if abs(err[k]) > abs(err[merged[-1]]): merged[-1] = k
            else:
                merged.append(k)
        # reduce to nb+1 points: drop smallest at ends
        while len(merged) > nb+1:
            if abs(err[merged[0]]) < abs(err[merged[-1]]): merged.pop(0)
            else: merged.pop()
        if len(merged) < nb+1:
            break
        idx = merged
    return best

def round_coefs(c):
    """doubles (python floats) of the coefficients, the first as (hi, lo)."""
    hi = float(c[0]); lo = float(c[0]-hi)
    return [(hi, lo)] + [float(x) for x in c[1:]]

def check_rounded(rs, fs, cr, rel=True, basis=None):
    hi, lo = cr[0]
    c = [m.mpf(hi)+m.mpf(lo)] + [m.mpf(x) for x in cr[1:]]
    mx = 0
    for r,f in zip(rs,fs):
        if basis is None: p = peval(c, r)
        else: p = sum(cj*phi(r) for cj,phi in zip(c,basis))
        e = abs((p-f)/(f if rel else 1))
        if e>mx: mx=e
    return mx

remez = types.SimpleNamespace(cheb_grid=cheb_grid, peval=peval, minimax=minimax, round_coefs=round_coefs, check_rounded=check_rounded)
# ---- the tables ----

m.mp.prec = 160
CACHE = os.path.join(tempfile.gettempdir(), "erf_tables_cache.json")
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save(): json.dump(cache, open(CACHE, "w"), indent=0)

def erfcx(x): return m.erfc(x)*m.exp(x*x)
def daw(x): return m.sqrt(m.pi)/2*m.exp(-x*x)*m.erfi(x)
def fg(x):
    S,C=m.fresnels(x),m.fresnelc(x)
    th=m.pi*x*x/2; s,c=m.sin(th),m.cos(th)
    a=C-m.mpf(1)/2; b=m.mpf(1)/2-S
    return a*s+b*c, b*s-a*c
def lg(x): return float(m.log(x,2)) if x>0 else -999

def fit(key, f, a, b, d, center=None, rel=True, n=300, wfun=None, target=None, dmax=None, single0=False):
    """minimax of f(center + r), r in [a-center, b-center]; returns (coefs[(hi,lo),..], log2 err, deg).
    The degree grows until the unrounded error is below `target`; c0 is kept as a double-double
    (as a double if `single0`), the rest rounded to doubles."""
    if key in cache: return cache[key]
    c = (a+b)/2 if center is None else center
    rs = remez.cheb_grid(a-c, b-c, n); fs = [f(c+r) for r in rs]
    W = [wfun(c+r, fv) for r, fv in zip(rs, fs)] if wfun is not None else [ (abs(fv) if rel else m.mpf(1)) for fv in fs]
    while True:
        co, e = remez.minimax(rs, fs, d, weights=W)
        if target is None or lg(e) <= target or d >= (dmax or 24): break
        d += 1
    hi = float(co[0]); lo = 0.0 if single0 else float(co[0]-hi)
    cr = [(hi, lo)] + [float(v) for v in co[1:]]
    cc = [m.mpf(hi) + lo] + [m.mpf(v) for v in cr[1:]]
    er = max(abs(remez.peval(cc, r) - fv)/w for r, fv, w in zip(rs, fs, W))
    cache[key] = (cr, lg(er), d); save()
    return cache[key]

out = []
def emit(name, rows, doc):
    out.append("/// " + doc)
    if isinstance(rows[0], list):
        w = len(rows[0])
        out.append(f"pub(super) static {name}: [[f64; {w}]; {len(rows)}] = [")
        for r in rows: out.append("    [" + ", ".join(repr(float(v)) for v in r) + "],")
    else:
        out.append(f"pub(super) static {name}: [f64; {len(rows)}] = [")
        for i in range(0, len(rows), 4): out.append("    " + ", ".join(repr(float(v)) for v in rows[i:i+4]) + ",")
    out.append("];\n")

def flat(cr): return [cr[0][0], cr[0][1]] + list(cr[1:])

def pieces(name, f, binades, d, doc, target=None):
    rows=[]; worst=-999
    for e in binades:
        lo = m.mpf(2)**e
        for j in range(8):
            a = lo*(1+m.mpf(j)/8); b = lo*(1+m.mpf(j+1)/8)
            cr, er, dd = fit(f"{name}:{e}:{j}:{d}", f, a, b, d)
            rows.append(flat(cr)); worst=max(worst,er)
    print(name, "worst", worst, file=sys.stderr)
    emit(name, rows, doc + f" Worst relative error after rounding 2^{worst:.1f}.")

parts = sys.argv[1:] or ["all"]
A = "all" in parts

# exp table 2^(j/64)
rows=[]
for j in range(64):
    v = m.mpf(2)**(m.mpf(j)/64); hi=float(v); lo=float(v-hi); rows.append([hi,lo])
emit("EXP2_64", rows, "`2^(j/64)` as `[hi, lo]`, `j = 0..64`.")

# erfcx on [0.5, 8): 4 binades x 8 pieces
pieces("ERFCX_T", erfcx, [-1,0,1,2], 10, "erfcx on `[0.5, 8)`: 32 pieces `[a(1+j/8), a(1+(j+1)/8))`, polynomial in `x − centre`, `[c0 hi, c0 lo, c1, …]`.")
pieces("DAWSON_T", daw, [-1,0,1,2], 11, "Dawson's integral on `[0.5, 8)`, pieces as `ERFCX_T`.")
pieces("FRESNEL_F_T", lambda x: fg(x)[0], [0,1,2], 11, "The Fresnel auxiliary `f` on `[1, 8)`, pieces as `ERFCX_T`.")
pieces("FRESNEL_G_T", lambda x: fg(x)[1], [0,1,2], 11, "The Fresnel auxiliary `g` on `[1, 8)`, pieces as `ERFCX_T`.")

# erfcx on [-0.5, 0.5): 8 uniform pieces of width 1/8
rows=[]; worst=-999; degs=[]
for j in range(8):
    a = m.mpf(-0.5) + m.mpf(j)/8; b = a + m.mpf(1)/8
    cr, er, dd = fit(f"ERFCX_SMALL:{j}", erfcx, a, b, 8, target=-60)
    degs.append(dd); worst=max(worst,er)
    rows.append((cr, dd))
D = max(degs)
emit("ERFCX_S", [flat(cr) + [0.0]*(D-dd) for cr, dd in rows], f"erfcx on `[−1/2, 1/2)`: 8 pieces `[−1/2 + j/8, −1/2 + (j+1)/8)`, polynomials of degree {D} in `x − centre`, `[c0 hi, c0 lo, c1, …]`. Worst 2^{worst:.1f}.")
print("ERFCX_S", degs, worst, file=sys.stderr)

# asymptotic, u = 1/x^2 in [0, 1/64]
def asym(name, g, U, d, doc, target=-58):
    def F(u):
        if u == 0: return g(None)
        x = 1/m.sqrt(u); return g(x)
    cr, er, dd = fit(name, F, m.mpf(0), U, d, center=m.mpf(0), target=target)
    print(name, "deg", dd, "err", er, file=sys.stderr)
    emit(name, flat(cr), doc + f" Degree {dd}, relative error 2^{er:.1f}.")
asym("ERFCX_ASYM", lambda x: 1/m.sqrt(m.pi) if x is None else x*erfcx(x), 1/m.mpf(64), 8,
     "`x·erfcx(x)` for `x ≥ 8` as a polynomial in `u = 1/x²`.")
asym("DAWSON_ASYM", lambda x: m.mpf(1)/2 if x is None else x*daw(x), 1/m.mpf(64), 8,
     "`x·F(x)` (Dawson) for `x ≥ 8` as a polynomial in `u = 1/x²`.")
def F4(g):
    def h(v):
        if v == 0: return g(None)
        return g(1/m.root(v,4))
    return h
cr, er, dd = fit("FRESNEL_F_ASYM", F4(lambda x: 1/m.pi if x is None else x*fg(x)[0]), m.mpf(0), 1/m.mpf(4096), 4, center=m.mpf(0), target=-56)
print("FRESNEL_F_ASYM", dd, er, file=sys.stderr)
emit("FRESNEL_F_ASYM", flat(cr), f"`x·f(x)` for `x ≥ 8` as a polynomial in `v = 1/x⁴`. Degree {dd}, 2^{er:.1f}.")
cr, er, dd = fit("FRESNEL_G_ASYM", F4(lambda x: 1/m.pi**2 if x is None else x**3*fg(x)[1]), m.mpf(0), 1/m.mpf(4096), 4, center=m.mpf(0), target=-54)
print("FRESNEL_G_ASYM", dd, er, file=sys.stderr)
emit("FRESNEL_G_ASYM", flat(cr), f"`x³·g(x)` for `x ≥ 8` as a polynomial in `v = 1/x⁴`. Degree {dd}, 2^{er:.1f}.")

# small arguments, polynomials in t = x^2 (erf, dawson) on [0, 1/4]
def tf(g):
    def h(t):
        if t == 0: return g(None)
        x = m.sqrt(t); return g(x)
    return h
cr, er, dd = fit("ERF_SMALL", tf(lambda x: 2/m.sqrt(m.pi) if x is None else m.erf(x)/x), m.mpf(0), m.mpf(1)/4, 8, center=m.mpf(0), target=-60)
print("ERF_SMALL", dd, er, file=sys.stderr)
c0 = m.mpf(cr[0][0]) + cr[0][1] - 1
emit("ERF_SMALL", [float(c0)] + list(cr[1:]), f"`erf(x)/x − 1` for `|x| < 1/2` as a polynomial in `t = x²` (`c0` = 2/√π − 1). Degree {dd}, relative error of `erf/x` 2^{er:.1f}.")
# dawson small: D/x = 1 + t Q(t); fit Q with weight t/(D/x)
def dq(t):
    if t == 0: return m.mpf(-2)/3
    x=m.sqrt(t); return (daw(x)/x-1)/t
def dqw(t, fv):
    x=m.sqrt(t) if t>0 else m.mpf(0)
    return (1+t*fv)/max(t, m.mpf(10)**-40)
cr, er, dd = fit("DAWSON_SMALL", dq, m.mpf(0), m.mpf(1)/4, 8, center=m.mpf(0), wfun=dqw, target=-60, single0=True)
print("DAWSON_SMALL", dd, er, file=sys.stderr)
emit("DAWSON_SMALL", [cr[0][0]] + list(cr[1:]), f"Dawson: `(F(x)/x − 1)/x²` for `|x| < 1/2` as a polynomial in `t = x²`. Degree {dd}, 2^{er:.1f} relative to `F/x`.")

# fresnel small, w = x^4 in [0, 1]
def cq(w):
    if w == 0: return -m.pi**2/40
    x = m.root(w,4); return (m.fresnelc(x)/x-1)/w
def cqw(w, fv): return (1+w*fv)/max(w, m.mpf(10)**-40)
cr, er, dd = fit("FRESNEL_C_SMALL", cq, m.mpf(0), m.mpf(1), 8, center=m.mpf(0), wfun=cqw, target=-60, single0=True)
print("FRESNEL_C_SMALL", dd, er, file=sys.stderr)
emit("FRESNEL_C_SMALL", [cr[0][0]] + list(cr[1:]), f"Fresnel: `(C(x)/x − 1)/x⁴` for `x < 1` as a polynomial in `w = x⁴`. Degree {dd}, 2^{er:.1f} relative to `C/x`.")
def sfun(w):
    if w == 0: return m.pi/6
    x = m.root(w,4); return m.fresnels(x)/x**3
cr, er, dd = fit("FRESNEL_S_SMALL", sfun, m.mpf(0), m.mpf(1), 8, center=m.mpf(0), target=-60)
print("FRESNEL_S_SMALL", dd, er, file=sys.stderr)
emit("FRESNEL_S_SMALL", flat(cr), f"Fresnel: `S(x)/x³` for `x < 1` as a polynomial in `w = x⁴`, `[c0 hi, c0 lo, c1, …]`. Degree {dd}, 2^{er:.1f}.")

# sin(pi r/2)/r and cos(pi r/2) in s = r^2, |r| <= 0.5 (+ margin)
R2 = m.mpf(0.2501)
cr, er, dd = fit("SINQ", lambda s: m.pi/2 if s == 0 else m.sin(m.pi*m.sqrt(s)/2)/m.sqrt(s), m.mpf(0), R2, 5, center=m.mpf(0), target=-60)
print("SINQ", dd, er, file=sys.stderr)
emit("SINQ", flat(cr), f"`sin(πr/2)/r` in `s = r²`, `|r| ≤ 1/2`, `[c0 hi, c0 lo, c1, …]`. Degree {dd}, 2^{er:.1f}.")
cr, er, dd = fit("COSQ", lambda s: -m.pi**2/8 if s == 0 else (m.cos(m.pi*m.sqrt(s)/2)-1)/s, m.mpf(0), R2, 5, center=m.mpf(0), target=-60, single0=True, wfun=lambda s, fv: (1+s*fv)/max(s, m.mpf(10)**-40))
print("COSQ", dd, er, file=sys.stderr)
emit("COSQ", [cr[0][0]] + list(cr[1:]), f"`(cos(πr/2) − 1)/r²` in `s = r²`, `|r| ≤ 1/2`. Degree {dd}, 2^{er:.1f}.")

# erf_inv initial approximation: y/x in t=x^2 on [0, 1/4]
cr, er, dd = fit("ERFINV_SMALL", lambda t: m.sqrt(m.pi)/2 if t == 0 else m.erfinv(m.sqrt(t))/m.sqrt(t), m.mpf(0), m.mpf(1)/4, 5, center=m.mpf(0), target=-31, single0=True)
print("ERFINV_SMALL", dd, er, file=sys.stderr)
emit("ERFINV_SMALL", [cr[0][0]] + list(cr[1:]), f"Initial `erf_inv(x)/x` for `|x| ≤ 1/2` in `t = x²`. Degree {dd}, 2^{er:.1f}.")
# erfc_inv initial: y as function of s = sqrt(-ln c), c in (0, 1/2]; pieces by binade of s, 2 per binade
def yinv(s):
    c = m.exp(-s*s)
    return m.erfinv(1-c) if c > m.mpf(10)**-30 else erfcinv_big(c)
def erfcinv_big(c):
    y = m.sqrt(-m.log(c))
    return m.findroot(lambda t: m.log(m.erfc(t)) - m.log(c), y)
s0 = m.sqrt(-m.log(m.mpf(1)/2))
rows=[]; worst=-999
for e in range(-1, 5):
    lo = m.mpf(2)**e
    for j in range(2):
        a = lo*(1+m.mpf(j)/2); b = lo*(1+m.mpf(j+1)/2)
        aa = max(a, s0*m.mpf(0.999)); bb = min(b, m.mpf(28))
        cen = (a+b)/2
        cr, er, dd = fit(f"ERFCINV:{e}:{j}", yinv, aa, bb, 6, center=cen, target=-31, dmax=9, single0=True)
        worst=max(worst,er); print("erfcinv piece", e, j, dd, er, file=sys.stderr)
        rows.append([cr[0][0]] + list(cr[1:]) + [0.0]*(9-len(cr)))
emit("ERFCINV_T", rows, f"Initial `erfc_inv(c)` for `c ≤ 1/2` as polynomials in `s − centre`, `s = √(−ln c)`, 2 pieces a binade of `s` from `[0.5, 1)` to `[16, 32)` (zero padded). Worst 2^{worst:.1f}.")

hdr = """//! Coefficient tables of the erf area, made by `gen_tables.py` (mpmath at 160 bits, a discrete
//! Remez exchange; minimax in relative error unless noted). Do not edit by hand.

#![allow(clippy::approx_constant, clippy::excessive_precision, clippy::unreadable_literal)]

"""
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables.rs"), "w").write(hdr + "\n".join(out))
print("ok", file=sys.stderr)

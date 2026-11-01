#!/usr/bin/env python3
"""Generates aarch64.rs, the inline assembly of the curves' fields: python3 kernels.py > aarch64.rs."""
import sys

def rows_first(a, bi, T, p, L):
    """t = a * bi in T[0..n+1)."""
    n = len(a)
    L.append(f"mul {T[0]}, {a[0]}, {bi}")
    L.append(f"umulh {T[1]}, {a[0]}, {bi}")
    for j in range(1, n):
        L.append(f"mul {p}, {a[j]}, {bi}")
        L.append(f"umulh {T[j+1]}, {a[j]}, {bi}")
        L.append(f"{'adds' if j == 1 else 'adcs'} {T[j]}, {T[j]}, {p}")
    L.append(f"adc {T[n]}, {T[n]}, xzr")

def row_add(a, bi, T, p, L, top_extra=None):
    """T[0..n] += a * bi, the carry out of T[n] into T[n+1] (fresh)."""
    n = len(a)
    for j in range(n):
        L.append(f"mul {p}, {a[j]}, {bi}")
        L.append(f"{'adds' if j == 0 else 'adcs'} {T[j]}, {T[j]}, {p}")
    if top_extra:
        L.append(f"adcs {T[n]}, {T[n]}, xzr")
        L.append(f"cset {T[n+1]}, cs")
    else:
        L.append(f"cset {T[n]}, cs")
    for j in range(n):
        L.append(f"umulh {p}, {a[j]}, {bi}")
        op = 'adds' if j == 0 else ('adcs' if (j < n - 1 or top_extra) else 'adc')
        L.append(f"{op} {T[j+1]}, {T[j+1]}, {p}")
    if top_extra:
        L.append(f"adc {T[n+1]}, {T[n+1]}, xzr")

def triangle4(a, T, p, L):
    """T[1..7) = the products of different limbs of the 4 limbs a."""
    L += [f"mul {T[1]}, {a[0]}, {a[1]}", f"umulh {T[2]}, {a[0]}, {a[1]}",
          f"mul {p}, {a[0]}, {a[2]}", f"umulh {T[3]}, {a[0]}, {a[2]}", f"adds {T[2]}, {T[2]}, {p}",
          f"mul {p}, {a[0]}, {a[3]}", f"umulh {T[4]}, {a[0]}, {a[3]}", f"adcs {T[3]}, {T[3]}, {p}",
          f"adc {T[4]}, {T[4]}, xzr",
          f"mul {p}, {a[1]}, {a[2]}", f"adds {T[3]}, {T[3]}, {p}",
          f"mul {p}, {a[1]}, {a[3]}", f"adcs {T[4]}, {T[4]}, {p}",
          f"cset {T[5]}, cs",
          f"umulh {p}, {a[1]}, {a[2]}", f"adds {T[4]}, {T[4]}, {p}",
          f"umulh {p}, {a[1]}, {a[3]}", f"adc {T[5]}, {T[5]}, {p}",
          f"mul {p}, {a[2]}, {a[3]}", f"umulh {T[6]}, {a[2]}, {a[3]}",
          f"adds {T[5]}, {T[5]}, {p}", f"adc {T[6]}, {T[6]}, xzr"]

def triangle(a, T, p, L):
    """T[1..2n-1) = the products of different limbs of a, by rows."""
    n = len(a)
    L.append(f"mul {T[1]}, {a[0]}, {a[1]}")
    L.append(f"umulh {T[2]}, {a[0]}, {a[1]}")
    for j in range(2, n):
        L.append(f"mul {p}, {a[0]}, {a[j]}")
        L.append(f"umulh {T[j+1]}, {a[0]}, {a[j]}")
        L.append(f"{'adds' if j == 2 else 'adcs'} {T[j]}, {T[j]}, {p}")
    L.append(f"adc {T[n]}, {T[n]}, xzr")
    for i in range(1, n - 1):
        m = n - i - 1
        for j in range(m):
            L.append(f"mul {p}, {a[i]}, {a[i+1+j]}")
            L.append(f"{'adds' if j == 0 else 'adcs'} {T[2*i+1+j]}, {T[2*i+1+j]}, {p}")
        L.append(f"cset {T[n+i]}, cs")
        for j in range(m):
            L.append(f"umulh {p}, {a[i]}, {a[i+1+j]}")
            op = ('add' if m == 1 else 'adc') if j == m - 1 else ('adds' if j == 0 else 'adcs')
            L.append(f"{op} {T[2*i+2+j]}, {T[2*i+2+j]}, {p}")

def double_add_squares(a, T, p, L):
    n = len(a)
    w = 2 * n
    L.append(f"adds {T[1]}, {T[1]}, {T[1]}")
    for j in range(2, w - 1):
        L.append(f"adcs {T[j]}, {T[j]}, {T[j]}")
    L.append(f"cset {T[w-1]}, cs")
    L.append(f"mul {T[0]}, {a[0]}, {a[0]}")
    L.append(f"umulh {p}, {a[0]}, {a[0]}")
    L.append(f"adds {T[1]}, {T[1]}, {p}")
    for i in range(1, n):
        L.append(f"mul {p}, {a[i]}, {a[i]}")
        L.append(f"adcs {T[2*i]}, {T[2*i]}, {p}")
        L.append(f"umulh {p}, {a[i]}, {a[i]}")
        L.append(f"{'adc' if i == n - 1 else 'adcs'} {T[2*i+1]}, {T[2*i+1]}, {p}")

def fold4(T, f, p, q, c, L):
    low, high = T[:4], T[4:]
    for j in range(4):
        L.append(f"mul {p}, {high[j]}, {f}")
        L.append(f"{'adds' if j == 0 else 'adcs'} {low[j]}, {low[j]}, {p}")
    L.append(f"cset {c}, cs")
    for j in range(4):
        L.append(f"umulh {p}, {high[j]}, {f}")
        if j < 3:
            L.append(f"{'adds' if j == 0 else 'adcs'} {low[j+1]}, {low[j+1]}, {p}")
        else:
            L.append(f"adc {c}, {c}, {p}")
    L += [f"mul {p}, {c}, {f}", f"umulh {q}, {c}, {f}",
          f"adds {low[0]}, {low[0]}, {p}", f"adcs {low[1]}, {low[1]}, {q}",
          f"adcs {low[2]}, {low[2]}, xzr", f"adcs {low[3]}, {low[3]}, xzr",
          f"csel {p}, {f}, xzr, cs", f"adds {low[0]}, {low[0]}, {p}", f"adc {low[1]}, {low[1]}, xzr",
          f"stp {low[0]}, {low[1]}, [x0]", f"stp {low[2]}, {low[3]}, [x0, #16]"]

def pseudo_mul_4():
    a = ['x4', 'x5', 'x6', 'x7']; bi = 'x16'; p = 'x17'
    T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
    L = ["ldp x4, x5, [x1]", "ldp x6, x7, [x1, #16]", f"ldr {bi}, [x2]"]
    rows_first(a, bi, T[:5], p, L)
    for i in range(1, 4):
        L.append(f"ldr {bi}, [x2, #{8*i}]")
        row_add(a, bi, T[i:i+5], p, L)
    fold4(T, 'x3', p, bi, 'x1', L)
    return L

def pseudo_sqr_4():
    a = ['x4', 'x5', 'x6', 'x7']; p = 'x16'
    T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
    L = ["ldp x4, x5, [x1]", "ldp x6, x7, [x1, #16]"]
    triangle4(a, T, p, L)
    double_add_squares(a, T, p, L)
    fold4(T, 'x3', p, 'x17', 'x1', L)
    return L

def p256_round(t, l, h, m0, m1, L):
    q = t[0]
    L += [f"lsl {l}, {q}, #32", f"lsr {h}, {q}, #32", f"subs {m0}, {q}, {l}", f"sbc {m1}, {q}, {h}"]

def p256_mul():
    a = ['x4', 'x5', 'x6', 'x7']; bi = 'x3'; p = 'x1'
    t = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13']; l, h, m0, m1 = 'x14', 'x15', 'x16', 'x17'
    L = ["ldp x4, x5, [x1]", "ldp x6, x7, [x1, #16]", f"ldr {bi}, [x2]"]
    rows_first(a, bi, t[:5], p, L)
    L.append(f"mov {t[5]}, xzr")
    for i in range(4):
        if i > 0:
            L.append(f"ldr {bi}, [x2, #{8*i}]")
            row_add(a, bi, t[:5] + [t[5]], p, L, top_extra=True)
        p256_round(t, l, h, m0, m1, L)
        L += [f"adds {t[1]}, {t[1]}, {l}", f"adcs {t[2]}, {t[2]}, {h}", f"adcs {t[3]}, {t[3]}, {m0}",
              f"adcs {t[4]}, {t[4]}, {m1}", f"adc {t[5]}, {t[5]}, xzr"]
        t = t[1:] + t[:1]
    # Below 2 p, in t[0..4] and t[4]: subtract p.
    L += ["mov x14, #-1", "mov x15, #0xffffffff", "mov x17, #0xffffffff00000001"]
    d = ['x16', 'x1', 'x3', t[5]]
    L += [f"subs {d[0]}, {t[0]}, x14", f"sbcs {d[1]}, {t[1]}, x15", f"sbcs {d[2]}, {t[2]}, xzr",
          f"sbcs {d[3]}, {t[3]}, x17", f"sbcs xzr, {t[4]}, xzr"]
    for j in range(4):
        L.append(f"csel {t[j]}, {t[j]}, {d[j]}, lo")
    L += [f"stp {t[0]}, {t[1]}, [x0]", f"stp {t[2]}, {t[3]}, [x0, #16]"]
    return L

def p256_sqr():
    a = ['x4', 'x5', 'x6', 'x7']; p = 'x3'
    T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']; l, h, m0, m1 = 'x16', 'x17', 'x1', 'x2'
    L = ["ldp x4, x5, [x1]", "ldp x6, x7, [x1, #16]"]
    triangle4(a, T, p, L)
    double_add_squares(a, T, p, L)
    t = T[:4]
    for i in range(4):
        p256_round(t, l, h, m0, m1, L)
        q = t[0]
        L += [f"adds {t[0]}, {t[1]}, {l}", f"adcs {t[1]}, {t[2]}, {h}", f"adcs {t[2]}, {t[3]}, {m0}",
              f"adc {t[3]}, {m1}, xzr"]
    # The low half onto the high half.
    hi = T[4:]
    L += [f"adds {t[0]}, {t[0]}, {hi[0]}", f"adcs {t[1]}, {t[1]}, {hi[1]}", f"adcs {t[2]}, {t[2]}, {hi[2]}",
          f"adcs {t[3]}, {t[3]}, {hi[3]}", f"cset {hi[0]}, cs"]
    L += ["mov x16, #-1", "mov x17, #0xffffffff", "mov x2, #0xffffffff00000001"]
    d = ['x1', 'x3', hi[1], hi[2]]
    L += [f"subs {d[0]}, {t[0]}, x16", f"sbcs {d[1]}, {t[1]}, x17", f"sbcs {d[2]}, {t[2]}, xzr",
          f"sbcs {d[3]}, {t[3]}, x2", f"sbcs xzr, {hi[0]}, xzr"]
    for j in range(4):
        L.append(f"csel {t[j]}, {t[j]}, {d[j]}, lo")
    L += [f"stp {t[0]}, {t[1]}, [x0]", f"stp {t[2]}, {t[3]}, [x0, #16]"]
    return L

def p384_round(t, regs, L):
    """t += q p, zero at t[0], with q = t[0] + (t[0] << 32) and the shifts of p's sparse limbs."""
    l, h, x0, q, x1v = regs
    L += [f"add {q}, {t[0]}, {t[0]}, lsl #32", f"lsl {l}, {q}, #32", f"lsr {h}, {q}, #32"]
    # q 2^32 - q = (x1' : x0).
    L += [f"subs {x0}, {l}, {q}", f"sbc {x1v}, {h}, xzr"]
    # + (x1' : x0) at limb 0, - (h : l) at limb 1, - q at limb 2, + q at limb 6.
    L += [f"cmn {t[0]}, {x0}", f"adcs {t[1]}, {t[1]}, {x1v}", f"adcs {t[2]}, {t[2]}, xzr",
          f"adcs {t[3]}, {t[3]}, xzr", f"adcs {t[4]}, {t[4]}, xzr", f"adcs {t[5]}, {t[5]}, xzr",
          f"adcs {t[6]}, {t[6]}, {q}", f"adc {t[7]}, {t[7]}, xzr"]
    # h + q + borrow can reach 2^64: its top goes to limb 3 with the borrow chain.
    L += [f"adds {x0}, {h}, {q}", f"cset {h}, cs",
          f"subs {t[1]}, {t[1]}, {l}", f"sbcs {t[2]}, {t[2]}, {x0}", f"sbcs {t[3]}, {t[3]}, {h}",
          f"sbcs {t[4]}, {t[4]}, xzr", f"sbcs {t[5]}, {t[5]}, xzr", f"sbcs {t[6]}, {t[6]}, xzr",
          f"sbc {t[7]}, {t[7]}, xzr"]

def p384_mul():
    a = ['x4', 'x5', 'x6', 'x7', 'x8', 'x9']; bi = 'x3'; p = 'x1'
    t = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']
    regs = ['x20', 'x21', 'x22', 'x23', 'x24']
    L = ["ldp x4, x5, [x1]", "ldp x6, x7, [x1, #16]", "ldp x8, x9, [x1, #32]", f"ldr {bi}, [x2]"]
    rows_first(a, bi, t[:7], p, L)
    L.append(f"mov {t[7]}, xzr")
    for i in range(6):
        if i > 0:
            L.append(f"ldr {bi}, [x2, #{8*i}]")
            row_add(a, bi, t[:7] + [t[7]], p, L, top_extra=True)
        p384_round(t, regs, L)
        t = t[1:] + t[:1]
        L.append(f"mov {t[7]}, xzr")
    return L, t, regs

def p384_finish(t, L):
    # Below 2 p, in t[0..6] and t[6]: subtract p, limbs x25.. hold the constants.
    regs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    L += ["mov x7, #0xffffffff", "mov x8, #0xffffffff00000000", "mov x9, #-2", "mov x20, #-1"]
    L += [f"subs {regs[0]}, {t[0]}, x7", f"sbcs {regs[1]}, {t[1]}, x8", f"sbcs {regs[2]}, {t[2]}, x9",
          f"sbcs {regs[3]}, {t[3]}, x20", f"sbcs {regs[4]}, {t[4]}, x20", f"sbcs {regs[5]}, {t[5]}, x20",
          f"sbcs xzr, {t[6]}, xzr"]
    for j in range(6):
        L.append(f"csel {t[j]}, {t[j]}, {regs[j]}, lo")
    L += [f"stp {t[0]}, {t[1]}, [x0]", f"stp {t[2]}, {t[3]}, [x0, #16]", f"stp {t[4]}, {t[5]}, [x0, #32]"]

def p384_sqr():
    a = ['x4', 'x5', 'x6', 'x7', 'x8', 'x9']; p = 'x3'
    T = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x20', 'x21', 'x22', 'x23']
    L = ["ldp x4, x5, [x1]", "ldp x6, x7, [x1, #16]", "ldp x8, x9, [x1, #32]"]
    triangle(a, T, p, L)
    double_add_squares(a, T, p, L)
    t = T[:6]
    l, h, x0, q, x1v, m = 'x24', 'x25', 'x26', 'x27', 'x28', 'x1'
    for _ in range(6):
        L += [f"add {q}, {t[0]}, {t[0]}, lsl #32", f"lsl {l}, {q}, #32", f"lsr {h}, {q}, #32",
              f"subs {x0}, {l}, {q}", f"sbc {x1v}, {h}, xzr"]
        L += [f"cmn {t[0]}, {x0}", f"adcs {t[1]}, {t[1]}, {x1v}", f"adcs {t[2]}, {t[2]}, xzr",
              f"adcs {t[3]}, {t[3]}, xzr", f"adcs {t[4]}, {t[4]}, xzr", f"adcs {t[5]}, {t[5]}, xzr",
              f"adc {t[0]}, {q}, xzr"]
        L += [f"adds {m}, {h}, {q}", f"cset {h}, cs",
              f"subs {t[1]}, {t[1]}, {l}", f"sbcs {t[2]}, {t[2]}, {m}", f"sbcs {t[3]}, {t[3]}, {h}",
              f"sbcs {t[4]}, {t[4]}, xzr", f"sbcs {t[5]}, {t[5]}, xzr", f"sbc {t[0]}, {t[0]}, xzr"]
        t = t[1:] + t[:1]
    hi = T[6:]
    L += [f"adds {t[0]}, {t[0]}, {hi[0]}"]
    for j in range(1, 6):
        L.append(f"adcs {t[j]}, {t[j]}, {hi[j]}")
    L.append(f"cset {hi[0]}, cs")
    # x4..x9, x24.. are free now.
    L += ["mov x4, #0xffffffff", "mov x5, #0xffffffff00000000", "mov x6, #-2", "mov x7, #-1"]
    d = ['x8', 'x9', 'x24', 'x25', 'x26', 'x27']
    L += [f"subs {d[0]}, {t[0]}, x4", f"sbcs {d[1]}, {t[1]}, x5", f"sbcs {d[2]}, {t[2]}, x6",
          f"sbcs {d[3]}, {t[3]}, x7", f"sbcs {d[4]}, {t[4]}, x7", f"sbcs {d[5]}, {t[5]}, x7",
          f"sbcs xzr, {hi[0]}, xzr"]
    for j in range(6):
        L.append(f"csel {t[j]}, {t[j]}, {d[j]}, lo")
    L += [f"stp {t[0]}, {t[1]}, [x0]", f"stp {t[2]}, {t[3]}, [x0, #16]", f"stp {t[4]}, {t[5]}, [x0, #32]"]
    return L

def chain(first, n, base, r, op, L):
    """r[j] = r[j] op limbs at base, two limbs loaded at a time, the flag through all of them."""
    t = ('x16', 'x17')
    for j in range(0, n, 2):
        if j + 1 < n:
            L.append(f"ldp {t[0]}, {t[1]}, [{base}, #{8*j}]" if j else f"ldp {t[0]}, {t[1]}, [{base}]")
        else:
            L.append(f"ldr {t[0]}, [{base}, #{8*j}]")
        for k in range(2):
            if j + k < n:
                o = first if j + k == 0 else op
                L.append(f"{o} {r[j+k]}, {r[j+k]}, {t[k]}")

def load_regs(r, base, L):
    n = len(r)
    for j in range(0, n, 2):
        if j + 1 < n:
            L.append(f"ldp {r[j]}, {r[j+1]}, [{base}, #{8*j}]" if j else f"ldp {r[j]}, {r[j+1]}, [{base}]")
        else:
            L.append(f"ldr {r[j]}, [{base}, #{8*j}]")

def store_regs(r, L):
    n = len(r)
    for j in range(0, n, 2):
        if j + 1 < n:
            L.append(f"stp {r[j]}, {r[j+1]}, [x0, #{8*j}]" if j else f"stp {r[j]}, {r[j+1]}, [x0]")
        else:
            L.append(f"str {r[j]}, [x0, #{8*j}]")

REGS = ['x4','x5','x6','x7','x8','x9','x10','x11','x12']

def pseudo_add(n):
    r = REGS[:n]; L = []
    load_regs(r, 'x1', L)
    chain('adds', n, 'x2', r, 'adcs', L)
    L.append("csel x16, x3, xzr, cs")
    L.append(f"adds {r[0]}, {r[0]}, x16")
    for j in range(1, n):
        L.append(f"adcs {r[j]}, {r[j]}, xzr")
    L.append("csel x16, x3, xzr, cs")
    L.append(f"add {r[0]}, {r[0]}, x16")
    store_regs(r, L)
    return L

def pseudo_sub(n):
    r = REGS[:n]; L = []
    load_regs(r, 'x1', L)
    chain('subs', n, 'x2', r, 'sbcs', L)
    for _ in range(2):
        L.append("csel x16, x3, xzr, cc")
        L.append(f"subs {r[0]}, {r[0]}, x16")
        for j in range(1, n):
            L.append(f"sbcs {r[j]}, {r[j]}, xzr")
    store_regs(r, L)
    return L

def mont_add(n):
    r = REGS[:n]; d = ['x10','x11','x12','x13','x14','x15'][:n] if n == 6 else ['x8','x9','x10','x11']
    L = []
    load_regs(r, 'x1', L)
    chain('adds', n, 'x2', r, 'adcs', L)
    L.append("cset x1, cs")
    # The sum less m, kept when the carry or the subtraction says it is at least m.
    t = ('x16', 'x17')
    for j in range(0, n, 2):
        L.append(f"ldp {t[0]}, {t[1]}, [x3, #{8*j}]" if j else f"ldp {t[0]}, {t[1]}, [x3]")
        L.append(f"{'subs' if j == 0 else 'sbcs'} {d[j]}, {r[j]}, {t[0]}")
        L.append(f"sbcs {d[j+1]}, {r[j+1]}, {t[1]}")
    L.append("sbcs xzr, x1, xzr")
    for j in range(n):
        L.append(f"csel {r[j]}, {r[j]}, {d[j]}, lo")
    store_regs(r, L)
    return L

def mont_sub(n):
    r = REGS[:n]; L = []
    load_regs(r, 'x1', L)
    chain('subs', n, 'x2', r, 'sbcs', L)
    L.append("csetm x1, cc")
    t = ('x16', 'x17')
    for j in range(0, n, 2):
        L.append(f"ldp {t[0]}, {t[1]}, [x3, #{8*j}]" if j else f"ldp {t[0]}, {t[1]}, [x3]")
        L.append(f"and {t[0]}, {t[0]}, x1")
        L.append(f"and {t[1]}, {t[1]}, x1")
        L.append(f"{'adds' if j == 0 else 'adcs'} {r[j]}, {r[j]}, {t[0]}")
        L.append(f"adcs {r[j+1]}, {r[j+1]}, {t[1]}")
    store_regs(r, L)
    return L

def p521_mul():
    a = ['x3','x4','x5','x6','x7','x8','x9','x10','x11']; bi = 'x12'; p = 'x1'
    T = ['x13','x14','x15','x16','x17','x20','x21','x22','x23','x24']
    L = []
    load_regs(a, 'x1', L)
    L.append(f"ldr {bi}, [x2]")
    rows_first(a, bi, T, p, L)
    L.append(f"str {T[0]}, [x0]")
    T = T[1:] + T[:1]
    for i in range(1, 9):
        L.append(f"ldr {bi}, [x2, #{8*i}]")
        row_add(a, bi, T, p, L)
        L.append(f"str {T[0]}, [x0, #{8*i}]")
        if i < 8:
            T = T[1:] + T[:1]
    h = T[1:]
    # The low limbs back, in the registers of a: bit 576 of the product is 2^55 modulo the prime.
    low = a
    x, y = 'x12', 'x1'
    load_regs(low, 'x0', L)
    L.append(f"lsl {x}, {h[0]}, #55")
    L.append(f"adds {low[0]}, {low[0]}, {x}")
    for j in range(1, 9):
        L.append(f"extr {x}, {h[j]}, {h[j-1]}, #9")
        L.append(f"adcs {low[j]}, {low[j]}, {x}")
    L.append(f"lsr {x}, {h[8]}, #9")
    L.append(f"adc {y}, {x}, xzr")
    L.append(f"lsl {x}, {y}, #55")
    L.append(f"lsr {y}, {y}, #9")
    L.append(f"adds {low[0]}, {low[0]}, {x}")
    L.append(f"adcs {low[1]}, {low[1]}, {y}")
    for j in range(2, 9):
        L.append(f"adcs {low[j]}, {low[j]}, xzr")
    L.append("mov x2, #0x80000000000000")
    L.append(f"csel {x}, x2, xzr, cs")
    L.append(f"adds {low[0]}, {low[0]}, {x}")
    L.append(f"adc {low[1]}, {low[1]}, xzr")
    store_regs(low, L)
    return L

def inline_rust(name, doc, args, lines, inputs, clobbers):
    return rust(name, doc, args, lines, inputs, clobbers).replace("#[inline(never)]", "#[inline(always)]")

def rust(name, doc, args, lines, inputs, clobbers):
    body = ",\n".join(f'            "{l}"' for l in lines)
    ins = "\n".join(f"            {i}," for i in inputs)
    cl = "\n".join(f'            out("{c}") _,' for c in clobbers)
    docs = "\n".join(f"/// {d}" if d else "///" for d in doc)
    return f'''{docs}
#[inline(never)]
pub(super) unsafe fn {name}({args}) {{
    // SAFETY: as the caller guarantees; the block reads the operands' limbs and writes `r`'s last.
    unsafe {{
        asm!(
{body},
{ins}
{cl}
            options(nostack),
        );
    }}
}}
'''

HEAD = '''//! The aarch64 kernels of the curves' fields: a product and its reduction in one run of inline
//! assembly, with the operands read once and the result written once, which leaves the core free to
//! overlap the independent products of a curve's formulas: measured on an Apple M4 Max, OpenSSL's
//! product of P-256 takes about 36 cycles after the one before, but its doubling, eight products and
//! their additions, 233 in all. **This file is generated by `kernels.py`, next to it: change that,
//! not this.**
//!
//! A row of a product is two carry chains, the low limbs of the row's products and then the high
//! ones one position up, each `mul` or `umulh` next to the addition that takes it, so that one
//! register holds them all and the carry flag goes through the row, as `mul` and `umulh` leave
//! it alone.
//!
//! The additions and subtractions come in the same way, inline, as a curve's formulas have about
//! twice as many as products, and the compiler's carry chains took a doubling on secp256k1 from 282
//! cycles to 318: `pseudo_add_4` and its like fold a carry back as `f`, `mont_add_4` and its like
//! compare with the prime. At nine limbs the compiled additions were the faster, and stay.
//!
//! - [`pseudo_mul_4`], [`pseudo_sqr_4`]: the eight limbs of the product, then the high half folded
//!   onto the low one by a product by `f`, for a prime with `2^256 = f`.
//! - [`p256_mul`], [`p256_sqr`]: Montgomery's reduction modulo P-256's prime `p`, where
//!   `-1 / p = 1` modulo `2^64`: the quotient of a round is the low limb `q` itself and `q p` is
//!   made of shifts of it, so that a round takes no product. With `l`, `h` the two limbs of
//!   `q 2^32`, a round adds `l` at the limb above and `h` above that, and the two limbs of
//!   `q (2^64 + 1) - q 2^32` above those.
//! - [`p384_mul`], [`p384_sqr`]: the same for P-384's prime, where `-1 / p = 2^32 + 1`, so that a
//!   round's `q` is a shift and an addition, and `q p = q 2^384 - q 2^128 - q 2^96 + q 2^32 - q`.
//! - [`p521_mul`]: the rows of a product of nine limbs, with the high half shifted onto the low one
//!   by `extr`, for the prime `2^521 - 1`, for squares too: 5 of the 8 products of a doubling are
//!   squares, and a doubling took 1,050 cycles so against 1,330 with the multiplication's kernels.

use core::arch::asm;
'''

SAFETY = ["# Safety", "", "The limbs of the operands must be valid for reads and those of `r` for writes; `r` may be", "an operand."]

def main():
    out = [HEAD]
    out.append(rust("pseudo_mul_4", ["Sets the 4 limbs at `r` to `a b` modulo the prime with `2^256 = f`, `f < 2^62`, not reduced below", "it.", ""] + SAFETY,
        "r: *mut u64, a: *const u64, b: *const u64, f: u64", pseudo_mul_4(),
        ['in("x0") r', 'inout("x1") a => _', 'in("x2") b', 'in("x3") f'],
        ['x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17']))
    out.append(rust("pseudo_sqr_4", ["Sets the 4 limbs at `r` to `a^2` modulo the prime with `2^256 = f`, `f < 2^62`, not reduced below", "it.", ""] + SAFETY,
        "r: *mut u64, a: *const u64, f: u64", pseudo_sqr_4(),
        ['in("x0") r', 'inout("x1") a => _', 'in("x3") f'],
        ['x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17']))
    out.append(rust("p256_mul", ["Sets the 4 limbs at `r` to `a b / 2^256` modulo P-256's prime, below it, for `a` and `b` below it.", ""] + SAFETY,
        "r: *mut u64, a: *const u64, b: *const u64", p256_mul(),
        ['in("x0") r', 'inout("x1") a => _', 'in("x2") b'],
        ['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17']))
    out.append(rust("p256_sqr", ["Sets the 4 limbs at `r` to `a^2 / 2^256` modulo P-256's prime, below it, for `a` below it.", ""] + SAFETY,
        "r: *mut u64, a: *const u64", p256_sqr(),
        ['in("x0") r', 'inout("x1") a => _'],
        ['x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17']))
    L, t, regs = p384_mul()
    p384_finish(t, L)
    out.append(rust("p384_mul", ["Sets the 6 limbs at `r` to `a b / 2^384` modulo P-384's prime, below it, for `a` and `b` below it.", ""] + SAFETY,
        "r: *mut u64, a: *const u64, b: *const u64", L,
        ['in("x0") r', 'inout("x1") a => _', 'inout("x2") b => _'],
        ['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x20','x21','x22','x23','x24']))
    out.append(rust("p384_sqr", ["Sets the 6 limbs at `r` to `a^2 / 2^384` modulo P-384's prime, below it, for `a` below it.", ""] + SAFETY,
        "r: *mut u64, a: *const u64", p384_sqr(),
        ['in("x0") r', 'inout("x1") a => _'],
        ['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x20','x21','x22','x23','x24','x25','x26','x27','x28']))
    out.append(rust("p521_mul", ["Sets the 9 limbs at `r` to `a b` modulo `2^521 - 1`, not reduced below it: the rows of the product,", "with each finished low limb written to `r`, whose limb `i` no later row reads, and the high", "half shifted onto the low one, as `2^576 = 2^55`.", ""] + SAFETY,
        "r: *mut u64, a: *const u64, b: *const u64", p521_mul(),
        ['in("x0") r', 'inout("x1") a => _', 'inout("x2") b => _'],
        ['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x20','x21','x22','x23','x24']))
    NOTE = ["# Safety", "", "The `N` limbs at each pointer must be valid, for writes at `r`, which may be an operand."]
    for n in (4, 6):
        out.append(inline_rust(f"pseudo_add_{n}", [f"Sets the {n} limbs at `r` to `a + b` modulo the prime with `2^{64*n} = f`, `f < 2^62`.", ""] + [x.replace('`N`', str(n)) for x in NOTE],
            "r: *mut u64, a: *const u64, b: *const u64, f: u64", pseudo_add(n),
            ['in("x0") r', 'in("x1") a', 'in("x2") b', 'in("x3") f'], REGS[:n] + ['x16', 'x17']))
        out.append(inline_rust(f"pseudo_sub_{n}", [f"Sets the {n} limbs at `r` to `a - b` modulo the prime with `2^{64*n} = f`, `f < 2^62`.", ""] + [x.replace('`N`', str(n)) for x in NOTE],
            "r: *mut u64, a: *const u64, b: *const u64, f: u64", pseudo_sub(n),
            ['in("x0") r', 'in("x1") a', 'in("x2") b', 'in("x3") f'], REGS[:n] + ['x16', 'x17']))
    for n in (4, 6):
        cl_add = sorted(set(REGS[:n] + (['x10','x11','x12','x13','x14','x15'] if n == 6 else ['x8','x9','x10','x11']) + ['x16','x17']), key=lambda x:int(x[1:]))
        out.append(inline_rust(f"mont_add_{n}", [f"Sets the {n} limbs at `r` to `a + b` modulo the {n} limbs at `m`, for `a` and `b` below `m`.", ""] + [x.replace('`N`', str(n)) for x in NOTE],
            "r: *mut u64, a: *const u64, b: *const u64, m: *const u64", mont_add(n),
            ['in("x0") r', 'inout("x1") a => _', 'in("x2") b', 'in("x3") m'], cl_add))
        out.append(inline_rust(f"mont_sub_{n}", [f"Sets the {n} limbs at `r` to `a - b` modulo the {n} limbs at `m`, for `a` and `b` below `m`.", ""] + [x.replace('`N`', str(n)) for x in NOTE],
            "r: *mut u64, a: *const u64, b: *const u64, m: *const u64", mont_sub(n),
            ['in("x0") r', 'inout("x1") a => _', 'in("x2") b', 'in("x3") m'], REGS[:n] + ['x16', 'x17']))
    sys.stdout.write("\n".join(out))
main()

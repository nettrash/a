#!/usr/bin/env python3
"""Generates aarch64.rs, the inline assembly of the curves' fields: python3 kernels.py > aarch64.rs."""
import re
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
    # The limb above, times `f`, onto the two low limbs: a carry out of them, about once in 2^64 / c,
    # goes to a branch around the end, as in `pseudo_add`.
    L += [f"mul {p}, {c}, {f}", f"umulh {q}, {c}, {f}",
          f"adds {low[0]}, {low[0]}, {p}", f"adcs {low[1]}, {low[1]}, {q}", "b.cc 2f",
          f"adcs {low[2]}, {low[2]}, xzr", f"adcs {low[3]}, {low[3]}, xzr",
          f"csel {p}, {f}, xzr, cs", f"adds {low[0]}, {low[0]}, {p}", f"adc {low[1]}, {low[1]}, xzr", "2:"]

def operands(name, regs):
    """The registers `regs` taking the limbs of the operand `name`."""
    return {r: f"{name}[{i}]" for i, r in enumerate(regs)}

def pseudo_mul_4():
    a = ['x4', 'x5', 'x6', 'x7']; b = ['x0', 'x1', 'x2', 'x20']; p = 'x17'
    T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
    L = []
    rows_first(a, b[0], T[:5], p, L)
    for i in range(1, 4):
        row_add(a, b[i], T[i:i+5], p, L)
    fold4(T, 'x3', p, 'x16', 'x1', L)
    return L, operands('a', a) | operands('b', b) | {'x3': 'f'}, T[:4]

def pseudo_sqr_4():
    a = ['x4', 'x5', 'x6', 'x7']; p = 'x16'
    T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
    L = []
    triangle4(a, T, p, L)
    double_add_squares(a, T, p, L)
    fold4(T, 'x3', p, 'x17', 'x1', L)
    return L, operands('a', a) | {'x3': 'f'}, T[:4]

def p256_round(t, l, h, m0, m1, L):
    q = t[0]
    L += [f"lsl {l}, {q}, #32", f"lsr {h}, {q}, #32", f"subs {m0}, {q}, {l}", f"sbc {m1}, {q}, {h}"]

def p256_mul():
    a = ['x4', 'x5', 'x6', 'x7']; b = ['x0', 'x2', 'x3', 'x20']; p = 'x1'
    t = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13']; l, h, m0, m1 = 'x14', 'x15', 'x16', 'x17'
    L = []
    rows_first(a, b[0], t[:5], p, L)
    L.append(f"mov {t[5]}, xzr")
    for i in range(4):
        if i > 0:
            row_add(a, b[i], t[:5] + [t[5]], p, L, top_extra=True)
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
    return L, operands('a', a) | operands('b', b), t[:4]

def p256_sqr():
    a = ['x4', 'x5', 'x6', 'x7']; p = 'x3'
    T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']; l, h, m0, m1 = 'x16', 'x17', 'x1', 'x2'
    L = []
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
    return L, operands('a', a), t[:4]

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
    a = ['x4', 'x5', 'x6', 'x7', 'x8', 'x9']; b = ['x0', 'x2', 'x25', 'x26', 'x27', 'x28']; p = 'x1'
    t = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17']
    regs = ['x20', 'x21', 'x22', 'x23', 'x24']
    L = []
    rows_first(a, b[0], t[:7], p, L)
    L.append(f"mov {t[7]}, xzr")
    for i in range(6):
        if i > 0:
            row_add(a, b[i], t[:7] + [t[7]], p, L, top_extra=True)
        p384_round(t, regs, L)
        t = t[1:] + t[:1]
        L.append(f"mov {t[7]}, xzr")
    p384_finish(t, L)
    return L, operands('a', a) | operands('b', b), t[:6]

def p384_finish(t, L):
    # Below 2 p, in t[0..6] and t[6]: subtract p, limbs x25.. hold the constants.
    regs = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
    L += ["mov x7, #0xffffffff", "mov x8, #0xffffffff00000000", "mov x9, #-2", "mov x20, #-1"]
    L += [f"subs {regs[0]}, {t[0]}, x7", f"sbcs {regs[1]}, {t[1]}, x8", f"sbcs {regs[2]}, {t[2]}, x9",
          f"sbcs {regs[3]}, {t[3]}, x20", f"sbcs {regs[4]}, {t[4]}, x20", f"sbcs {regs[5]}, {t[5]}, x20",
          f"sbcs xzr, {t[6]}, xzr"]
    for j in range(6):
        L.append(f"csel {t[j]}, {t[j]}, {regs[j]}, lo")

def p384_sqr():
    a = ['x4', 'x5', 'x6', 'x7', 'x8', 'x9']; p = 'x3'
    T = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x20', 'x21', 'x22', 'x23']
    L = []
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
    return L, operands('a', a), t[:6]

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
B = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15']
B9 = ['x13', 'x14', 'x15', 'x17', 'x20', 'x21', 'x22', 'x23', 'x24']
M = ['x16', 'x17', 'x20', 'x21', 'x22', 'x23']

def chain(first, op, r, b, L):
    """r[j] = r[j] op b[j], the flag through all of them."""
    for j in range(len(r)):
        L.append(f"{first if j == 0 else op} {r[j]}, {r[j]}, {b[j]}")

def pseudo_add(n):
    """The sum, and `f` for a carry out of it: a second carry out comes only when that `f` carries out
    of the low limb, about once in 2^64 / f, so it goes to a branch around the end, which leaves the
    other limbs as the first chain set them."""
    r, b = REGS[:n], (B if n <= 6 else B9)[:n]; L = []
    chain('adds', 'adcs', r, b, L)
    L.append("csel x16, x3, xzr, cs")
    L.append(f"adds {r[0]}, {r[0]}, x16")
    L.append("b.cc 2f")
    for j in range(1, n):
        L.append(f"adcs {r[j]}, {r[j]}, xzr")
    L.append("csel x16, x3, xzr, cs")
    L.append(f"add {r[0]}, {r[0]}, x16")
    L.append("2:")
    return L, operands('a', r) | operands('b', b) | {'x3': 'f'}, r

def pseudo_sub(n):
    """The difference, less `f` for a borrow out of it, and the same once more after a branch, which
    is taken but about once in 2^64 / f."""
    r, b = REGS[:n], (B if n <= 6 else B9)[:n]; L = []
    chain('subs', 'sbcs', r, b, L)
    L.append("csel x16, x3, xzr, cc")
    L.append(f"subs {r[0]}, {r[0]}, x16")
    L.append("b.cs 2f")
    for j in range(1, n):
        L.append(f"sbcs {r[j]}, {r[j]}, xzr")
    L.append("csel x16, x3, xzr, cc")
    L.append(f"subs {r[0]}, {r[0]}, x16")
    for j in range(1, n):
        L.append(f"sbcs {r[j]}, {r[j]}, xzr")
    L.append("2:")
    return L, operands('a', r) | operands('b', b) | {'x3': 'f'}, r

def mont_add(n):
    r, b, m = REGS[:n], B[:n], M[:n]; L = []
    chain('adds', 'adcs', r, b, L)
    L.append("cset x1, cs")
    # The sum less m, in the registers of b, kept when the carry or the subtraction says it is at
    # least m.
    for j in range(n):
        L.append(f"{'subs' if j == 0 else 'sbcs'} {b[j]}, {r[j]}, {m[j]}")
    L.append("sbcs xzr, x1, xzr")
    for j in range(n):
        L.append(f"csel {r[j]}, {r[j]}, {b[j]}, lo")
    return L, operands('a', r) | operands('b', b) | operands('m', m), r

def mont_sub(n):
    """The difference and the difference plus m, the second chain a step behind the first, then the
    one the borrow says."""
    r, b, m = REGS[:n], B[:n], M[:n]; L = []
    chain('subs', 'sbcs', r, b, L)
    L.append("csetm x1, cc")
    for j in range(n):
        L.append(f"{'adds' if j == 0 else 'adcs'} {b[j]}, {r[j]}, {m[j]}")
    L.append("cmp x1, xzr")
    for j in range(n):
        L.append(f"csel {r[j]}, {b[j]}, {r[j]}, ne")
    return L, operands('a', r) | operands('b', b) | operands('m', m), r

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
    p521_fold(T[1:], a, 'x12', 'x1', 'x2', L)
    return L

def p521_fold(h, low, x, y, z, L):
    """The low limbs back from `r` into `low`, and the high ones `h` shifted onto them: bit 576 of
    the product is 2^55 modulo the prime."""
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
    L.append(f"mov {z}, #0x80000000000000")
    L.append(f"csel {x}, {z}, xzr, cs")
    L.append(f"adds {low[0]}, {low[0]}, {x}")
    L.append(f"adc {low[1]}, {low[1]}, xzr")
    store_regs(low, L)

def p521_sqr():
    """The square of nine limbs: the products of different limbs once, as a triangle of rows whose
    finished low limbs wait in `r`, then each limb doubled by `extr` and the squares of the limbs
    added in one carry chain, then the fold of `p521_mul`."""
    a = ['x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']
    L = []
    load_regs(a, 'x1', L)
    free = ['x1', 'x2', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17',
            'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27']
    p = 'x28'
    pos = {}
    take = lambda: free.pop(0)
    def spill(k):
        L.append(f"str {pos[k]}, [x0, #{8*k}]")
        free.append(pos.pop(k))
    # Row 0 gives positions 1 to 9, row i adds a[i] a[i+1..] from position 2 i + 1 and a fresh top
    # limb i + 9; positions 2 i + 1 and 2 i + 2 are final after it.
    for k in range(1, 10):
        pos[k] = take()
    rows_first(a[1:], a[0], [pos[k] for k in range(1, 10)], p, L)
    spill(1); spill(2)
    for i in range(1, 8):
        pos[i + 9] = take()
        row_add(a[i+1:], a[i], [pos[k] for k in range(2*i + 1, i + 10)], p, L)
        for k in (2*i + 1, 2*i + 2):
            if k <= 8:
                spill(k)
    # Twice the high limbs in place, from the top down, each with the top bit of the one below.
    top = take()
    L.append(f"lsr {top}, {pos[16]}, #63")
    for k in range(16, 9, -1):
        L.append(f"extr {pos[k]}, {pos[k]}, {pos[k-1]}, #63")
    prev = take()
    L.append(f"ldr {prev}, [x0, #64]")
    L.append(f"extr {pos[9]}, {pos[9]}, {prev}, #63")
    t8 = prev
    # The low limbs from the bottom up, doubled and with the squares added, back to r; then the
    # high ones, in the same carry chain.
    d = take()
    L.append(f"mul {d}, {a[0]}, {a[0]}")
    L.append(f"str {d}, [x0]")
    free.append(d)
    prev = None
    for k in range(1, 9):
        if k == 8:
            cur = t8
        else:
            cur = take()
            L.append(f"ldr {cur}, [x0, #{8*k}]")
        d = take()
        if k == 1:
            L.append(f"lsl {d}, {cur}, #1")
        else:
            L.append(f"extr {d}, {cur}, {prev}, #63")
            free.append(prev)
        prev = cur
        s = take()
        L.append(f"{'umulh' if k % 2 else 'mul'} {s}, {a[k // 2]}, {a[k // 2]}")
        L.append(f"{'adds' if k == 1 else 'adcs'} {d}, {d}, {s}")
        L.append(f"str {d}, [x0, #{8*k}]")
        free += [s, d]
    free.append(prev)
    high = [pos[k] for k in range(9, 17)] + [top]
    for k in range(9, 18):
        s = take()
        L.append(f"{'umulh' if k % 2 else 'mul'} {s}, {a[k // 2]}, {a[k // 2]}")
        L.append(f"{'adc' if k == 17 else 'adcs'} {high[k - 9]}, {high[k - 9]}, {s}")
        free.append(s)
    used = set(high)
    rest = [r for r in a + free if r not in used]
    p521_fold(high, rest[:9], rest[9], rest[10], rest[11], L)
    return L

def registers(name, doc, args, kernel):
    """A kernel on registers: `kernel` gives the lines, the registers that take the operands' limbs
    and those of the result, the low limb first; every other register it names is a scratch one.
    Each register becomes an operand of its own for the compiler to place, as fixed registers left
    it no room for the kernels of numbers' products once both were inlined into one function."""
    lines, inputs, outputs = kernel
    used = set(re.findall(r"\bx\d+\b", "\n".join(lines))) | set(inputs) | set(outputs)
    written = {l.split()[1].rstrip(",") for l in lines
               if " " in l and l.split()[0] not in ("cmn", "cmp", "tst") and not l.startswith("b.")}
    ops = []
    for r in sorted(used, key=lambda x: int(x[1:])):
        if r in inputs and r in outputs:
            ops.append(f'{r} = inout(reg) {inputs[r]} => r[{outputs.index(r)}]')
        elif r in inputs and r in written:
            ops.append(f'{r} = inout(reg) {inputs[r]} => _')
        elif r in inputs:
            ops.append(f'{r} = in(reg) {inputs[r]}')
        elif r in outputs:
            ops.append(f'{r} = out(reg) r[{outputs.index(r)}]')
        else:
            ops.append(f'{r} = out(reg) _')
    n = len(outputs)
    body = ",\n".join(f'            "{re.sub(r"\b(x\d+)\b", r"{\1}", l)}"' for l in lines)
    ops = "\n".join(f"            {o}," for o in ops)
    docs = "\n".join(f"/// {d}" if d else "///" for d in doc)
    return f'''{docs}
#[inline(always)]
pub(super) fn {name}({args}) -> [u64; {n}] {{
    let mut r = [0; {n}];
    // SAFETY: the block reads and writes registers only.
    unsafe {{
        asm!(
{body},
{ops}
            options(pure, nomem, nostack),
        );
    }}
    r
}}
'''

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

HEAD = '''//! The aarch64 kernels of the curves' fields, in inline assembly. **This file is generated by
//! `kernels.py`, next to it: change that, not this.**
//!
//! Up to six limbs a kernel takes its operands and gives its result in registers, and is inlined,
//! so that a curve's formulas keep their elements in registers from one operation to the next:
//! measured on an Apple M4 Max, a product of P-256 took 33 cycles after the one before, an addition
//! 9 and a doubling 272, against 48, 22 and 357 with kernels that read their operands and wrote
//! their result in memory; OpenSSL's assembly takes 38 for the product and 257 for the doubling. The
//! compiler chooses the registers, as fixed ones left it none for the kernels of numbers' products
//! once both were inlined into one function.
//!
//! A row of a product is two carry chains, the low limbs of the row's products and then the high
//! ones one position up, each `mul` or `umulh` next to the addition that takes it, so that one
//! register holds them all and the carry flag goes through the row, as `mul` and `umulh` leave
//! it alone.
//!
//! A prime `2^k - c` folds a carry out of a sum, or the limb above a product's fold, back as a
//! product by `c`, `f = 2^(64 n)` modulo the prime, which carries out of the low limbs about once in
//! `2^64 / f`: that case goes to a branch around the end of the kernel, so that the other limbs are
//! done after one chain. An addition on secp256k1 so took 7 cycles after the one before against 12,
//! P-521's subtraction 13 against 42, and a product and a square of Curve25519's field 31 and 29
//! against 37 and 33.
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
//! - [`p521_mul`], [`p521_sqr`]: nine limbs, too many for the registers, so the operands are read
//!   and the result written in memory, and the finished low limbs of the product wait in `r`; the
//!   high half is shifted onto the low one by `extr`, for the prime `2^521 - 1`. The square takes
//!   the products of different limbs once, doubles them by `extr` and adds the squares of the limbs:
//!   90 cycles after the one before, against 103 by the product's kernel.
//! - [`pseudo_add_4`] and its like fold a carry back as `f`; [`mont_add_4`] and its like compare
//!   with the prime, and their subtractions take the difference or the difference plus the prime,
//!   the second chain a step behind the first, by the borrow.

use core::arch::asm;
'''

SAFETY = ["# Safety", "", "The limbs of the operands must be valid for reads and those of `r` for writes; `r` may be", "an operand."]

def main():
    out = [HEAD]
    A4, B4 = "a: &[u64; 4]", "b: &[u64; 4]"
    A6, B6 = "a: &[u64; 6]", "b: &[u64; 6]"
    out.append(registers("pseudo_mul_4", ["`a b` modulo the prime with `2^256 = f`, `f < 2^62`, not reduced below it."],
        f"{A4}, {B4}, f: u64", pseudo_mul_4()))
    out.append(registers("pseudo_sqr_4", ["`a^2` modulo the prime with `2^256 = f`, `f < 2^62`, not reduced below it."],
        f"{A4}, f: u64", pseudo_sqr_4()))
    out.append(registers("p256_mul", ["`a b / 2^256` modulo P-256's prime, below it, for `a` and `b` below it."],
        f"{A4}, {B4}", p256_mul()))
    out.append(registers("p256_sqr", ["`a^2 / 2^256` modulo P-256's prime, below it, for `a` below it."],
        A4, p256_sqr()))
    out.append(registers("p384_mul", ["`a b / 2^384` modulo P-384's prime, below it, for `a` and `b` below it."],
        f"{A6}, {B6}", p384_mul()))
    out.append(registers("p384_sqr", ["`a^2 / 2^384` modulo P-384's prime, below it, for `a` below it."],
        A6, p384_sqr()))
    out.append(rust("p521_mul", ["Sets the 9 limbs at `r` to `a b` modulo `2^521 - 1`, not reduced below it: the rows of the product,", "with each finished low limb written to `r`, whose limb `i` no later row reads, and the high", "half shifted onto the low one, as `2^576 = 2^55`.", ""] + SAFETY,
        "r: *mut u64, a: *const u64, b: *const u64", p521_mul(),
        ['in("x0") r', 'inout("x1") a => _', 'inout("x2") b => _'],
        ['x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x20','x21','x22','x23','x24']))
    out.append(rust("p521_sqr", ["Sets the 9 limbs at `r` to `a^2` modulo `2^521 - 1`, not reduced below it: the products of", "different limbs once, with the finished low limbs waiting in `r`, then all of them doubled and", "the squares of the limbs added, and the high half shifted onto the low one.", ""] + SAFETY,
        "r: *mut u64, a: *const u64", p521_sqr(),
        ['in("x0") r', 'inout("x1") a => _'],
        ['x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x20','x21','x22','x23','x24','x25','x26','x27','x28']))
    for n in (4, 6, 9):
        a, b = {4: (A4, B4), 6: (A6, B6), 9: ("a: &[u64; 9]", "b: &[u64; 9]")}[n]
        if n == 9:
            out.append(registers(f"pseudo_add_{n}", [f"`a + b` modulo the prime with `2^{64*n} = f`, `f < 2^62`."],
                f"{a}, {b}, f: u64", pseudo_add(n)))
            out.append(registers(f"pseudo_sub_{n}", [f"`a - b` modulo the prime with `2^{64*n} = f`, `f < 2^62`."],
                f"{a}, {b}, f: u64", pseudo_sub(n)))
            continue
        out.append(registers(f"pseudo_add_{n}", [f"`a + b` modulo the prime with `2^{64*n} = f`, `f < 2^62`."],
            f"{a}, {b}, f: u64", pseudo_add(n)))
        out.append(registers(f"pseudo_sub_{n}", [f"`a - b` modulo the prime with `2^{64*n} = f`, `f < 2^62`."],
            f"{a}, {b}, f: u64", pseudo_sub(n)))
        out.append(registers(f"mont_add_{n}", [f"`a + b` modulo `m`, for `a` and `b` below `m`."],
            f"{a}, {b}, m: &[u64; {n}]", mont_add(n)))
        out.append(registers(f"mont_sub_{n}", [f"`a - b` modulo `m`, for `a` and `b` below `m`."],
            f"{a}, {b}, m: &[u64; {n}]", mont_sub(n)))
    sys.stdout.write("\n".join(out))
main()

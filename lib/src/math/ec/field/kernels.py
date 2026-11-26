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

def redc_row_nc(m, q, T, p, L):
    """T[0..n] += q m, for T of n + 1 words that the sum does not overflow (a spare top bit of m),
    which leaves T[0] zero."""
    n = len(m)
    for j in range(n):
        L.append(f"mul {p}, {m[j]}, {q}")
        L.append(f"{'adds' if j == 0 else 'adcs'} {T[j]}, {T[j]}, {p}")
    L.append(f"adc {T[n]}, {T[n]}, xzr")
    for j in range(n):
        L.append(f"umulh {p}, {m[j]}, {q}")
        L.append(f"{'adds' if j == 0 else ('adc' if j == n - 1 else 'adcs')} {T[j+1]}, {T[j+1]}, {p}")

def mont_mul_nc(n):
    """Montgomery's product by rows (CIOS) modulo an m with a spare top bit, m < 2^(64 n - 1): then
    the sum of a round, t + a b_i + q m, is at most (2 m - 1) 2^64, n + 1 limbs, and t stays below
    2 m in n limbs, so that no limb above is carried and one subtraction ends it."""
    if n == 4:
        a = ['x4', 'x5', 'x6', 'x7']; b = ['x0', 'x1', 'x2', 'x3']; m = ['x8', 'x9', 'x10', 'x11']
        t = ['x12', 'x13', 'x14', 'x15', 'x16']; p, q, minv = 'x17', 'x20', 'x21'
        bi = lambda i, L: b[i]
        inputs = operands('a', a) | operands('b', b) | operands('m', m) | {minv: 'minv'}
    else:
        a = ['x4', 'x5', 'x6', 'x7', 'x8', 'x9']; m = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15']
        t = ['x16', 'x17', 'x20', 'x21', 'x22', 'x23', 'x24']; p, q, minv, bp, br = 'x25', 'x26', 'x27', 'x0', 'x1'
        def bi(i, L):
            L.append(f"ldr {br}, [{bp}, #{8 * i}]")
            return br
        inputs = operands('a', a) | operands('m', m) | {minv: 'minv', bp: 'b.as_ptr()'}
    L = []
    rows_first(a, bi(0, L), t[:n+1], p, L)
    for i in range(n):
        if i > 0:
            # t, n limbs, gets a b_i: the carry out of the low chain makes the new top limb.
            row_add(a, bi(i, L), t[:n+1], p, L)
        L.append(f"mul {q}, {t[0]}, {minv}")
        redc_row_nc(m, q, t[:n+1], p, L)
        t = t[1:] + t[:1]
    # t < 2 m in t[0..n]: subtract m into the registers of a, keep t on a borrow.
    d = a
    for j in range(n):
        L.append(f"{'subs' if j == 0 else 'sbcs'} {d[j]}, {t[j]}, {m[j]}")
    for j in range(n):
        L.append(f"csel {t[j]}, {t[j]}, {d[j]}, lo")
    return L, inputs, t[:n]

def mont_sqr_nc(n):
    """Not generated: measured on an Apple M4 Max it took longer than the product of a by a (19.3 ns
    against 15.1 for BN254's prime in a chain of squares), its n rounds of reduction a chain of
    dependent quotients with nothing to overlap them, where the product's rows do.

    Montgomery's square modulo an m with a spare top bit: the 2 n limbs of the square (the
    triangle doubled, the squares added), n rounds of the reduction on the low half, each
    (t + q m) / 2^64 in n limbs as t < 2^(64 n) and m < 2^(64 n - 1), then the high half added:
    below m + a^2 / 2^(64 n) < 2 m, one subtraction."""
    if n == 4:
        a = ['x4', 'x5', 'x6', 'x7']; T = ['x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15']
        m = ['x16', 'x17', 'x20', 'x21']; p, q, minv, extra = 'x22', 'x23', 'x24', 'x25'
        inputs = operands('a', a) | operands('m', m) | {minv: 'minv'}
        L = []
        triangle4(a, T, p, L)
        load_m = []
    else:
        a = ['x4', 'x5', 'x6', 'x7', 'x8', 'x9']
        T = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x20', 'x21', 'x22', 'x23']
        m = a  # loaded once the square is made
        p, q, minv, extra, mp = 'x24', 'x25', 'x26', 'x27', 'x0'
        inputs = operands('a', a) | {minv: 'minv', mp: 'm.as_ptr()'}
        L = []
        triangle(a, T, p, L)
    double_add_squares(a, T, p, L)
    if n == 6:
        for j in range(0, 6, 2):
            L.append(f"ldp {m[j]}, {m[j+1]}, [{mp}, #{8 * j}]")
    t = T[:n] + [extra]
    for _ in range(n):
        L.append(f"mul {q}, {t[0]}, {minv}")
        # t + q m into n + 1 limbs, the top one fresh; t[0] becomes zero and is the next fresh one.
        for j in range(n):
            L.append(f"mul {p}, {m[j]}, {q}")
            L.append(f"{'adds' if j == 0 else 'adcs'} {t[j]}, {t[j]}, {p}")
        L.append(f"cset {t[n]}, cs")
        for j in range(n):
            L.append(f"umulh {p}, {m[j]}, {q}")
            L.append(f"{'adds' if j == 0 else ('adc' if j == n - 1 else 'adcs')} {t[j+1]}, {t[j+1]}, {p}")
        t = t[1:] + t[:1]
    hi = T[n:]
    for j in range(n):
        L.append(f"{'adds' if j == 0 else 'adcs'} {t[j]}, {t[j]}, {hi[j]}")
    d = hi
    for j in range(n):
        L.append(f"{'subs' if j == 0 else 'sbcs'} {d[j]}, {t[j]}, {m[j]}")
    for j in range(n):
        L.append(f"csel {t[j]}, {t[j]}, {d[j]}, lo")
    return L, inputs, t[:n]

REGS = ['x4','x5','x6','x7','x8','x9','x10','x11','x12']
B = ['x10', 'x11', 'x12', 'x13', 'x14', 'x15']
# The double-width arithmetic of the lazy reduction (`Mont::mul_complex`): products of 2 n limbs,
# sums and differences of them, and Montgomery's reduction of one, for a prime with spare bits.
W = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
     'x14', 'x15', 'x16', 'x17', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28']

def mul_wide(n):
    """The 2 n limbs of `a b`, by rows: the first row's products, then each row's added one limb up,
    its carry out the row's new top limb."""
    a, out = W[:n], W[n:3 * n]
    if n == 4:
        b = W[3 * n:4 * n]; p = W[4 * n]
        bi = lambda i, L: b[i]
        inputs = operands('a', a) | operands('b', b)
    else:
        p, bp, br = W[3 * n], W[3 * n + 1], W[3 * n + 2]
        def bi(i, L):
            L.append(f"ldr {br}, [{bp}, #{8 * i}]")
            return br
        inputs = operands('a', a) | {bp: 'b.as_ptr()'}
    L = []
    rows_first(a, bi(0, L), out[:n + 1], p, L)
    for i in range(1, n):
        row_add(a, bi(i, L), out[i:i + n + 1], p, L)
    return L, inputs, out

def redc_wide(n):
    """Montgomery's reduction of `t = l + h R`, below `m R`, for an `m` with a spare top bit:
    `(l + Q m) / R` by the n rounds of the product's reduction on n + 1 limbs, which make it at
    most `m`, then `+ h`, below `2 m` in n limbs, and one subtraction."""
    T, h, m = W[:n + 1], W[n + 1:2 * n + 1], W[2 * n + 1:3 * n + 1]
    q, p, minv = W[3 * n + 1], W[3 * n + 2], W[3 * n + 3]
    inputs = operands('t', T[:n]) | {r: f"t[{n + i}]" for i, r in enumerate(h)} | operands('m', m) \
        | {minv: 'minv'}
    L = [f"mov {T[n]}, xzr"]
    for i in range(n):
        L.append(f"mul {q}, {T[0]}, {minv}")
        redc_row_nc(m, q, T, p, L)
        # The low limb is zero now: it becomes the top one of the next round.
        T = T[1:] + T[:1]
    chain('adds', 'adcs', T[:n], h, L)
    for j in range(n):
        L.append(f"{'subs' if j == 0 else 'sbcs'} {h[j]}, {T[j]}, {m[j]}")
    for j in range(n):
        L.append(f"csel {T[j]}, {T[j]}, {h[j]}, lo")
    return L, inputs, T[:n]

def sub_wide(n, modular):
    """`a - b` in 2 n limbs, `b` read in pairs; `modular`: plus `m R` on a borrow, so that the result
    is `a - b` modulo `m R` for `a`, `b` below `m R`."""
    a, bp, t0, t1 = W[:2 * n], W[2 * n], W[2 * n + 1], W[2 * n + 2]
    inputs = operands('a', a) | {bp: 'b.as_ptr()'}
    L = []
    for j in range(0, 2 * n, 2):
        L.append(f"ldp {t0}, {t1}, [{bp}, #{8 * j}]")
        L.append(f"{'subs' if j == 0 else 'sbcs'} {a[j]}, {a[j]}, {t0}")
        L.append(f"sbcs {a[j + 1]}, {a[j + 1]}, {t1}")
    if modular:
        mp, mask = W[2 * n + 3], W[2 * n + 4]
        inputs |= {mp: 'm.as_ptr()'}
        L.append(f"csetm {mask}, cc")
        for j in range(0, n, 2):
            L.append(f"ldp {t0}, {t1}, [{mp}, #{8 * j}]")
            L.append(f"and {t0}, {t0}, {mask}")
            L.append(f"and {t1}, {t1}, {mask}")
            L.append(f"{'adds' if j == 0 else 'adcs'} {a[n + j]}, {a[n + j]}, {t0}")
            L.append(f"adcs {a[n + j + 1]}, {a[n + j + 1]}, {t1}")
    return L, inputs, a

def add_plain(n):
    """`a + b` in n limbs, not reduced: for `a`, `b` below `m` and `2 m < R`."""
    a, b = W[:n], W[n:2 * n]
    L = []
    chain('adds', 'adcs', a, b, L)
    return L, operands('a', a) | operands('b', b), a

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

def registers(name, doc, args, kernel, readonly=False):
    """A kernel on registers: `kernel` gives the lines, the registers that take the operands' limbs
    and those of the result, the low limb first; every other register it names is a scratch one.
    Each register becomes an operand of its own for the compiler to place, as fixed registers left
    it no room for the kernels of numbers' products once both were inlined into one function."""
    lines, inputs, outputs = kernel
    used = set(re.findall(r"\bx\d+\b", "\n".join(lines))) | set(inputs) | set(outputs)
    written = {l.split()[1].rstrip(",") for l in lines
               if " " in l and l.split()[0] not in ("cmn", "cmp", "tst") and not l.startswith("b.")}
    # `ldp` writes its first two registers.
    written |= {l.split()[2].rstrip(",") for l in lines if l.startswith("ldp ")}
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
    safety = ("the block reads the limbs of the arrays whose pointers it takes, which their references\n"
              "    // hold for it, and writes registers only." if readonly else
              "the block reads and writes registers only.")
    return f'''{docs}
#[inline(always)]
pub(super) fn {name}({args}) -> [u64; {n}] {{
    let mut r = [0; {n}];
    // SAFETY: {safety}
    unsafe {{
        asm!(
{body},
{ops}
            options(pure, {'readonly' if readonly else 'nomem'}, nostack),
        );
    }}
    r
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
//! a subtraction of nine limbs 13 against 42, and a product and a square of Curve25519's field 31
//! and 29 against 37 and 33.
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
//! - [`pseudo_add_4`] and its like fold a carry back as `f`; [`mont_add_4`] and its like compare
//!   with the prime, and their subtractions take the difference or the difference plus the prime,
//!   the second chain a step behind the first, by the borrow.
//! - [`mont_mul_nc_4`], [`mont_mul_nc_6`]: Montgomery's product by rows for a prime with a spare
//!   top bit, which drops the carries above the rows.
//! - [`mul_wide_4`], [`redc_wide_4`], [`sub_wide_4`], [`sub_wide_mod_4`], [`add_plain_4`] and those
//!   of six limbs: the double-width values of a lazy reduction, which sums and differences of
//!   products before one reduction of the result (`Mont::mul_complex`).

use core::arch::asm;
'''

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
    for n in (4, 6):
        a, b = {4: (A4, B4), 6: (A6, B6)}[n]
        out.append(registers(f"mont_mul_nc_{n}",
            [f"`a b / 2^{64*n}` modulo `m`, below it, for `a` and `b` below it and `m` below `2^{64*n-1}`;",
             "`minv` is `-1 / m` modulo `2^64`."],
            f"{a}, {b}, m: &[u64; {n}], minv: u64", mont_mul_nc(n), readonly=(n == 6)))
    for n in (4, 6):
        a, b = {4: (A4, B4), 6: (A6, B6)}[n]
        w = f"&[u64; {2 * n}]"
        out.append(registers(f"mul_wide_{n}", [f"The {2 * n} limbs of `a b`."],
            f"{a}, {b}", mul_wide(n), readonly=(n == 6)))
        out.append(registers(f"redc_wide_{n}",
            [f"`t / 2^{64*n}` modulo `m`, below it, for `t` below `m 2^{64*n}` and `m` below `2^{64*n-1}`;",
             "`minv` is `-1 / m` modulo `2^64`."],
            f"t: {w}, m: &[u64; {n}], minv: u64", redc_wide(n)))
        out.append(registers(f"sub_wide_{n}", [f"`a - b` in {2 * n} limbs, for `a` at least `b`."],
            f"a: {w}, b: {w}", sub_wide(n, False), readonly=True))
        out.append(registers(f"sub_wide_mod_{n}",
            [f"`a - b` modulo `m 2^{64*n}`, for `a` and `b` below it."],
            f"a: {w}, b: {w}, m: &[u64; {n}]", sub_wide(n, True), readonly=True))
        out.append(registers(f"add_plain_{n}", [f"`a + b`, not reduced, for a sum below `2^{64*n}`."],
            f"{a}, {b}", add_plain(n)))
    sys.stdout.write("\n".join(out))
main()

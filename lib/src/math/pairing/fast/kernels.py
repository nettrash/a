#!/usr/bin/env python3
"""Generates aarch64.rs next to this file: the aarch64 kernels of the pairing engine's
prime fields for 4 and 6 limbs (modular sum, difference and negation, Montgomery's product, double-width
product and square, Montgomery's reduction, double-width sums and differences).

    python3 kernels.py > aarch64.rs

Every kernel is one `asm!` block on registers the compiler allocates; the carry chains are
`adds`/`adcs` and `subs`/`sbcs` runs that no flag-setting instruction interrupts (the products,
loads and `and`s between them leave the flags alone), and every selection is a `csel`, so no
kernel branches on its data.
"""

import sys

out = []


def emit(s=""):
    out.append(s)


def asm_block(lines, operands, options):
    emit("    // SAFETY: the block reads the operands' limbs through the pointers given, which point")
    emit("    // to arrays of the lengths read, writes only the registers declared, and leaves the")
    emit("    // stack alone.")
    emit("    unsafe {")
    emit("        asm!(")
    for l in lines:
        emit(f'            "{l}",')
    for o in operands:
        emit(f"            {o},")
    emit(f"            options({options}),")
    emit("        );")
    emit("    }")


def loads(dst, ptr, n, off=0):
    """ldp pairs of n limbs into the registers dst from [ptr + off]."""
    l = []
    for i in range(0, n, 2):
        o = off + 8 * i
        addr = f"[{{{ptr}}}]" if o == 0 else f"[{{{ptr}}}, #{o}]"
        l.append(f"ldp {{{dst[i]}}}, {{{dst[i + 1]}}}, {addr}")
    return l


def gen_add(n):
    r = [f"r{i}" for i in range(n)]
    b = [f"b{i}" for i in range(n)]
    lines = []
    lines.append(f"adds {{{r[0]}}}, {{{r[0]}}}, {{{b[0]}}}")
    for i in range(1, n):
        lines.append(f"adcs {{{r[i]}}}, {{{r[i]}}}, {{{b[i]}}}")
    # d = r - p into b, two limbs of p at a time.
    for i in range(0, n, 2):
        o = 8 * i
        addr = "[{pp}]" if o == 0 else f"[{{pp}}, #{o}]"
        lines.append(f"ldp {{q0}}, {{q1}}, {addr}")
        op0 = "subs" if i == 0 else "sbcs"
        lines.append(f"{op0} {{{b[i]}}}, {{{r[i]}}}, {{q0}}")
        lines.append(f"sbcs {{{b[i + 1]}}}, {{{r[i + 1]}}}, {{q1}}")
    for i in range(n):
        lines.append(f"csel {{{r[i]}}}, {{{r[i]}}}, {{{b[i]}}}, lo")
    ops = [f"{r[i]} = inout(reg) r[{i}]" for i in range(n)]
    ops += [f"{b[i]} = inout(reg) b[{i}] => _" for i in range(n)]
    ops += ["q0 = out(reg) _", "q1 = out(reg) _", "pp = in(reg) p.as_ptr()"]
    emit(f"/// `a + b mod p` for `a, b < p < 2^{64 * n - 1}`: the sum, the sum less `p`, and the one")
    emit("/// of them in `[0, p)` by the borrow of the difference.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn add{n}(a: &[u64; {n}], b: &[u64; {n}], p: &[u64; {n}]) -> [u64; {n}] {{")
    emit("    let mut r = *a;")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    r")
    emit("}")
    emit()


def gen_sub(n):
    r = [f"r{i}" for i in range(n)]
    b = [f"b{i}" for i in range(n)]
    lines = []
    lines.append(f"subs {{{r[0]}}}, {{{r[0]}}}, {{{b[0]}}}")
    for i in range(1, n):
        lines.append(f"sbcs {{{r[i]}}}, {{{r[i]}}}, {{{b[i]}}}")
    lines.append("sbc {m}, xzr, xzr")
    for i in range(0, n, 2):
        o = 8 * i
        addr = "[{pp}]" if o == 0 else f"[{{pp}}, #{o}]"
        lines.append(f"ldp {{q0}}, {{q1}}, {addr}")
        lines.append("and {q0}, {q0}, {m}")
        lines.append("and {q1}, {q1}, {m}")
        op0 = "adds" if i == 0 else "adcs"
        lines.append(f"{op0} {{{r[i]}}}, {{{r[i]}}}, {{q0}}")
        lines.append(f"adcs {{{r[i + 1]}}}, {{{r[i + 1]}}}, {{q1}}")
    ops = [f"{r[i]} = inout(reg) r[{i}]" for i in range(n)]
    ops += [f"{b[i]} = in(reg) b[{i}]" for i in range(n)]
    ops += ["m = out(reg) _", "q0 = out(reg) _", "q1 = out(reg) _", "pp = in(reg) p.as_ptr()"]
    emit(f"/// `a − b mod p` for `a, b < p`: the difference, plus `p` masked by its borrow.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn sub{n}(a: &[u64; {n}], b: &[u64; {n}], p: &[u64; {n}]) -> [u64; {n}] {{")
    emit("    let mut r = *a;")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    r")
    emit("}")
    emit()


def gen_neg(n):
    r = [f"r{i}" for i in range(n)]
    a = [f"a{i}" for i in range(n)]
    lines = []
    for i in range(0, n, 2):
        o = 8 * i
        addr = "[{pp}]" if o == 0 else f"[{{pp}}, #{o}]"
        lines.append(f"ldp {{{r[i]}}}, {{{r[i + 1]}}}, {addr}")
    lines.append(f"subs {{{r[0]}}}, {{{r[0]}}}, {{{a[0]}}}")
    for i in range(1, n):
        lines.append(f"sbcs {{{r[i]}}}, {{{r[i]}}}, {{{a[i]}}}")
    # 0 stays 0: p − 0 = p is replaced by 0 when every limb of a is 0.
    lines.append(f"orr {{z}}, {{{a[0]}}}, {{{a[1]}}}")
    for i in range(2, n):
        lines.append(f"orr {{z}}, {{z}}, {{{a[i]}}}")
    lines.append("cmp {z}, #0")
    for i in range(n):
        lines.append(f"csel {{{r[i]}}}, xzr, {{{r[i]}}}, eq")
    ops = [f"{r[i]} = out(reg) r[{i}]" for i in range(n)]
    ops += [f"{a[i]} = in(reg) a[{i}]" for i in range(n)]
    ops += ["z = out(reg) _", "pp = in(reg) p.as_ptr()"]
    emit(f"/// `−a mod p` for `a < p`: `p − a`, or 0 for `a = 0`.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn neg{n}(a: &[u64; {n}], p: &[u64; {n}]) -> [u64; {n}] {{")
    emit(f"    let mut r = [0u64; {n}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    r")
    emit("}")
    emit()


def gen_mont(n):
    # Registers: a_i, p_i (loaded), T rotating over n+1 names whose first n are the outputs.
    a = [f"a{i}" for i in range(n)]
    p = [f"p{i}" for i in range(n)]
    regs = [f"o{i}" for i in range(n)] + ["t"]
    lines = loads(a, "ap", n) + loads(p, "pp", n)
    tmp = ["x", "y"]
    ti = 0

    def nxt():
        nonlocal ti
        v = tmp[ti % 2]
        ti += 1
        return v

    T = list(regs)  # T[k] is the register of limb k
    for i in range(n):
        lines.append(f"ldr {{bi}}, [{{bp}}, #{8 * i}]" if i else "ldr {bi}, [{bp}]")
        if i == 0:
            for j in range(n):
                lines.append(f"mul {{{T[j]}}}, {{{a[j]}}}, {{bi}}")
            for j in range(n):
                v = nxt()
                lines.append(f"umulh {{{v}}}, {{{a[j]}}}, {{bi}}")
                if j == 0:
                    lines.append(f"adds {{{T[1]}}}, {{{T[1]}}}, {{{v}}}")
                elif j < n - 1:
                    lines.append(f"adcs {{{T[j + 1]}}}, {{{T[j + 1]}}}, {{{v}}}")
                else:
                    lines.append(f"adc {{{T[n]}}}, {{{v}}}, xzr")
        else:
            for j in range(n):
                v = nxt()
                lines.append(f"mul {{{v}}}, {{{a[j]}}}, {{bi}}")
                op = "adds" if j == 0 else "adcs"
                lines.append(f"{op} {{{T[j]}}}, {{{T[j]}}}, {{{v}}}")
            lines.append(f"adc {{{T[n]}}}, xzr, xzr")
            for j in range(n):
                v = nxt()
                lines.append(f"umulh {{{v}}}, {{{a[j]}}}, {{bi}}")
                op = "adds" if j == 0 else "adcs"
                if j < n - 1:
                    lines.append(f"{op} {{{T[j + 1]}}}, {{{T[j + 1]}}}, {{{v}}}")
                else:
                    lines.append(f"adc {{{T[n]}}}, {{{T[n]}}}, {{{v}}}")
        # The reduction: T += m·p with m = T₀·(−p⁻¹), then T /= 2⁶⁴.
        lines.append(f"mul {{m}}, {{{T[0]}}}, {{inv}}")
        for j in range(n):
            v = nxt()
            lines.append(f"mul {{{v}}}, {{m}}, {{{p[j]}}}")
            op = "adds" if j == 0 else "adcs"
            lines.append(f"{op} {{{T[j]}}}, {{{T[j]}}}, {{{v}}}")
        lines.append(f"adc {{{T[n]}}}, {{{T[n]}}}, xzr")
        for j in range(n):
            v = nxt()
            lines.append(f"umulh {{{v}}}, {{m}}, {{{p[j]}}}")
            op = "adds" if j == 0 else "adcs"
            # T < 2p + 2p·2⁶⁴ < 2^(64(n+1)): no carry out of the top limb.
            lines.append(f"{op} {{{T[j + 1]}}}, {{{T[j + 1]}}}, {{{v}}}")
        T = T[1:] + T[:1]
    # T[0..n] < 2p: the difference with p into the a registers, then the selection into the
    # outputs, from the top down, as limb k sits in the register of output k − 1 (or t).
    for j in range(n):
        op = "subs" if j == 0 else "sbcs"
        lines.append(f"{op} {{{a[j]}}}, {{{T[j]}}}, {{{p[j]}}}")
    for k in reversed(range(n)):
        lines.append(f"csel {{{regs[k]}}}, {{{T[k]}}}, {{{a[k]}}}, lo")
    lines = [l.replace("{bi}", "{ap}").replace("{m}", "{pp}") for l in lines]
    ops = [f"o{i} = out(reg) o[{i}]" for i in range(n)]
    ops += ["t = out(reg) _"]
    ops += [f"{a[i]} = out(reg) _" for i in range(n)]
    ops += [f"{p[i]} = out(reg) _" for i in range(n)]
    ops += ["x = out(reg) _", "y = out(reg) _"]
    # a's and p's pointers are dead after their loads: their registers hold bᵢ and m.
    ops += ["ap = inout(reg) a.as_ptr() => _", "bp = in(reg) b.as_ptr()"]
    ops += ["pp = inout(reg) p.as_ptr() => _", "inv = in(reg) inv"]
    emit(f"/// Montgomery's product `a b / 2^{64 * n} mod p` for `a, b < p < 2^{64 * n - 2}`, by the CIOS")
    emit("/// loop on registers: each row adds `a·bᵢ` and `m·p` and drops the low limb, with the")
    emit("/// limbs' registers renamed rather than moved.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn mont_mul{n}(a: &[u64; {n}], b: &[u64; {n}], p: &[u64; {n}], inv: u64) -> [u64; {n}] {{")
    emit(f"    let mut o = [0u64; {n}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


def gen_mul_wide(n):
    a = [f"a{i}" for i in range(n)]
    t = [f"o{i}" for i in range(2 * n)]
    lines = loads(a, "ap", n)
    tmp = ["x", "y"]
    ti = 0

    def nxt():
        nonlocal ti
        v = tmp[ti % 2]
        ti += 1
        return v

    for i in range(n):
        lines.append(f"ldr {{bi}}, [{{bp}}, #{8 * i}]" if i else "ldr {bi}, [{bp}]")
        if i == 0:
            for j in range(n):
                lines.append(f"mul {{{t[j]}}}, {{{a[j]}}}, {{bi}}")
            for j in range(n):
                v = nxt()
                lines.append(f"umulh {{{v}}}, {{{a[j]}}}, {{bi}}")
                if j == 0:
                    lines.append(f"adds {{{t[1]}}}, {{{t[1]}}}, {{{v}}}")
                elif j < n - 1:
                    lines.append(f"adcs {{{t[j + 1]}}}, {{{t[j + 1]}}}, {{{v}}}")
                else:
                    lines.append(f"adc {{{t[n]}}}, {{{v}}}, xzr")
        else:
            for j in range(n):
                v = nxt()
                lines.append(f"mul {{{v}}}, {{{a[j]}}}, {{bi}}")
                op = "adds" if j == 0 else "adcs"
                lines.append(f"{op} {{{t[i + j]}}}, {{{t[i + j]}}}, {{{v}}}")
            lines.append(f"adc {{{t[i + n]}}}, xzr, xzr")
            for j in range(n):
                v = nxt()
                lines.append(f"umulh {{{v}}}, {{{a[j]}}}, {{bi}}")
                op = "adds" if j == 0 else "adcs"
                if j < n - 1:
                    lines.append(f"{op} {{{t[i + j + 1]}}}, {{{t[i + j + 1]}}}, {{{v}}}")
                else:
                    lines.append(f"adc {{{t[i + n]}}}, {{{t[i + n]}}}, {{{v}}}")
    ops = [f"o{i} = out(reg) o[{i}]" for i in range(2 * n)]
    ops += [f"{a[i]} = out(reg) _" for i in range(n)]
    ops += ["x = out(reg) _", "y = out(reg) _", "bi = out(reg) _"]
    ops += ["ap = in(reg) a.as_ptr()", "bp = in(reg) b.as_ptr()"]
    emit(f"/// The double-width product `a b` of {n} limbs each, row by row.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn mul_wide{n}(a: &[u64; {n}], b: &[u64; {n}]) -> [u64; {2 * n}] {{")
    emit(f"    let mut o = [0u64; {2 * n}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


def gen_sqr_wide(n):
    a = [f"a{i}" for i in range(n)]
    t = [f"o{i}" for i in range(2 * n)]
    lines = loads(a, "ap", n)
    tmp = ["x", "y"]
    ti = 0

    def nxt():
        nonlocal ti
        v = tmp[ti % 2]
        ti += 1
        return v

    # Cross products a_i a_j (i < j) into positions i + j, row by row; t0 and t_{2n-1} untouched.
    for i in range(n - 1):
        js = list(range(i + 1, n))
        if i == 0:
            for j in js:
                lines.append(f"mul {{{t[j]}}}, {{{a[0]}}}, {{{a[j]}}}")
            for k, j in enumerate(js):
                v = nxt()
                lines.append(f"umulh {{{v}}}, {{{a[0]}}}, {{{a[j]}}}")
                if k == 0:
                    lines.append(f"adds {{{t[j + 1]}}}, {{{t[j + 1]}}}, {{{v}}}")
                elif k < len(js) - 1:
                    lines.append(f"adcs {{{t[j + 1]}}}, {{{t[j + 1]}}}, {{{v}}}")
                else:
                    lines.append(f"adc {{{t[n]}}}, {{{v}}}, xzr")
        else:
            for k, j in enumerate(js):
                v = nxt()
                lines.append(f"mul {{{v}}}, {{{a[i]}}}, {{{a[j]}}}")
                op = "adds" if k == 0 else "adcs"
                lines.append(f"{op} {{{t[i + j]}}}, {{{t[i + j]}}}, {{{v}}}")
            lines.append(f"adc {{{t[i + n]}}}, xzr, xzr")
            for k, j in enumerate(js):
                v = nxt()
                lines.append(f"umulh {{{v}}}, {{{a[i]}}}, {{{a[j]}}}")
                op = "adds" if k == 0 else "adcs"
                if k < len(js) - 1:
                    lines.append(f"{op} {{{t[i + j + 1]}}}, {{{t[i + j + 1]}}}, {{{v}}}")
                else:
                    # A row of one product has no chain to continue: a plain sum.
                    last = "adc" if k > 0 else "add"
                    lines.append(f"{last} {{{t[i + n]}}}, {{{t[i + n]}}}, {{{v}}}")
    # Double positions 1 .. 2n-2 into 1 .. 2n-1.
    lines.append(f"adds {{{t[1]}}}, {{{t[1]}}}, {{{t[1]}}}")
    for k in range(2, 2 * n - 1):
        lines.append(f"adcs {{{t[k]}}}, {{{t[k]}}}, {{{t[k]}}}")
    lines.append(f"adc {{{t[2 * n - 1]}}}, xzr, xzr")
    # Add the squares a_i² at positions 2i, 2i + 1, in one chain.
    lines.append(f"mul {{{t[0]}}}, {{{a[0]}}}, {{{a[0]}}}")
    for i in range(n):
        if i > 0:
            v = nxt()
            lines.append(f"mul {{{v}}}, {{{a[i]}}}, {{{a[i]}}}")
            lines.append(f"adcs {{{t[2 * i]}}}, {{{t[2 * i]}}}, {{{v}}}")
        v = nxt()
        lines.append(f"umulh {{{v}}}, {{{a[i]}}}, {{{a[i]}}}")
        op = "adds" if i == 0 else ("adc" if i == n - 1 else "adcs")
        lines.append(f"{op} {{{t[2 * i + 1]}}}, {{{t[2 * i + 1]}}}, {{{v}}}")
    ops = [f"o{i} = out(reg) o[{i}]" for i in range(2 * n)]
    ops += [f"{a[i]} = out(reg) _" for i in range(n)]
    ops += ["x = out(reg) _", "y = out(reg) _"]
    ops += ["ap = in(reg) a.as_ptr()"]
    emit(f"/// The double-width square `a²` of {n} limbs: the cross products once, doubled, then the")
    emit("/// squares of the limbs.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn sqr_wide{n}(a: &[u64; {n}]) -> [u64; {2 * n}] {{")
    emit(f"    let mut o = [0u64; {2 * n}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


def gen_redc(n):
    # The wide value in t0..t_{2n-1}; the result ends in t_n..t_{2n-1}, which are the outputs.
    t = [f"w{i}" for i in range(n)] + [f"o{i}" for i in range(n)]
    p = [f"p{i}" for i in range(n)]
    lines = loads(t[:n], "tp", n) + loads(t[n:], "tp", n, 8 * n) + loads(p, "pp", n)
    tmp = ["x", "y"]
    ti = 0

    def nxt():
        nonlocal ti
        v = tmp[ti % 2]
        ti += 1
        return v

    for i in range(n):
        lines.append(f"mul {{m}}, {{{t[i]}}}, {{inv}}")
        for j in range(n):
            v = nxt()
            lines.append(f"mul {{{v}}}, {{m}}, {{{p[j]}}}")
            op = "adds" if j == 0 else "adcs"
            lines.append(f"{op} {{{t[i + j]}}}, {{{t[i + j]}}}, {{{v}}}")
        lines.append("adc {c}, xzr, xzr")
        for j in range(n):
            v = nxt()
            lines.append(f"umulh {{{v}}}, {{m}}, {{{p[j]}}}")
            if j < n - 1:
                op = "adds" if j == 0 else "adcs"
                lines.append(f"{op} {{{t[i + j + 1]}}}, {{{t[i + j + 1]}}}, {{{v}}}")
            else:
                lines.append(f"adc {{c}}, {{c}}, {{{v}}}")
        # t_{i+n} += c + top, the carry out into top.
        if i == 0:
            lines.append(f"adds {{{t[n]}}}, {{{t[n]}}}, {{c}}")
        else:
            lines.append("cmp {top}, #1")
            lines.append(f"adcs {{{t[i + n]}}}, {{{t[i + n]}}}, {{c}}")
        lines.append("cset {top}, hs")
    # (t + m p)/R < 2p < R: top is 0; the difference with p into the low registers.
    for j in range(n):
        op = "subs" if j == 0 else "sbcs"
        lines.append(f"{op} {{{t[j]}}}, {{{t[n + j]}}}, {{{p[j]}}}")
    for j in range(n):
        lines.append(f"csel {{{t[n + j]}}}, {{{t[n + j]}}}, {{{t[j]}}}, lo")
    nl = len(loads(t[:n], "tp", n) + loads(t[n:], "tp", n, 8 * n) + loads(p, "pp", n))
    lines = lines[:nl] + [l.replace("{top}", "{tp}").replace("{c}", "{pp}") for l in lines[nl:]]
    ops = [f"o{i} = out(reg) o[{i}]" for i in range(n)]
    ops += [f"w{i} = out(reg) _" for i in range(n)]
    ops += [f"{p[i]} = out(reg) _" for i in range(n)]
    ops += ["x = out(reg) _", "y = out(reg) _", "m = out(reg) _"]
    # t's and p's pointers are dead after their loads: their registers hold the carries.
    ops += ["tp = inout(reg) t.as_ptr() => _", "pp = inout(reg) p.as_ptr() => _", "inv = in(reg) inv"]
    emit(f"/// Montgomery's reduction `t / 2^{64 * n} mod p` of a double-width `t < p·2^{64 * n}`.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn redc{n}(t: &[u64; {2 * n}], p: &[u64; {n}], inv: u64) -> [u64; {n}] {{")
    emit(f"    let mut o = [0u64; {n}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


def gen_wide(n, kind):
    m = 2 * n
    o = [f"o{i}" for i in range(m)]
    lines = loads(o, "ap", m)
    # b's limbs two at a time.
    first = "adds" if kind == "add" else "subs"
    nextop = "adcs" if kind == "add" else "sbcs"
    for i in range(0, m, 2):
        off = 8 * i
        addr = "[{bp}]" if off == 0 else f"[{{bp}}, #{off}]"
        lines.append(f"ldp {{q0}}, {{q1}}, {addr}")
        lines.append(f"{first if i == 0 else nextop} {{{o[i]}}}, {{{o[i]}}}, {{q0}}")
        lines.append(f"{nextop} {{{o[i + 1]}}}, {{{o[i + 1]}}}, {{q1}}")
    if kind == "submod":
        pass
    ops = [f"o{i} = out(reg) o[{i}]" for i in range(m)]
    ops += ["q0 = out(reg) _", "q1 = out(reg) _", "ap = in(reg) a.as_ptr()", "bp = in(reg) b.as_ptr()"]
    return lines, ops


def gen_wide_add(n):
    m = 2 * n
    lines, ops = gen_wide(n, "add")
    emit(f"/// The double-width sum `a + b` of {m} limbs, for a sum that fits.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn wide_add{n}(a: &[u64; {m}], b: &[u64; {m}]) -> [u64; {m}] {{")
    emit(f"    let mut o = [0u64; {m}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


def gen_wide_sub_plain(n):
    m = 2 * n
    lines, ops = gen_wide(n, "sub")
    emit(f"/// The double-width difference `a − b` of {m} limbs, for `a ≥ b`.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn wide_sub_plain{n}(a: &[u64; {m}], b: &[u64; {m}]) -> [u64; {m}] {{")
    emit(f"    let mut o = [0u64; {m}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


def gen_wide_sub(n):
    m = 2 * n
    lines, ops = gen_wide(n, "sub")
    o = [f"o{i}" for i in range(m)]
    lines.append("sbc {mk}, xzr, xzr")
    for i in range(0, n, 2):
        off = 8 * i
        addr = "[{pp}]" if off == 0 else f"[{{pp}}, #{off}]"
        lines.append(f"ldp {{q0}}, {{q1}}, {addr}")
        lines.append("and {q0}, {q0}, {mk}")
        lines.append("and {q1}, {q1}, {mk}")
        op0 = "adds" if i == 0 else "adcs"
        lines.append(f"{op0} {{{o[n + i]}}}, {{{o[n + i]}}}, {{q0}}")
        lines.append(f"adcs {{{o[n + i + 1]}}}, {{{o[n + i + 1]}}}, {{q1}}")
    ops += ["mk = out(reg) _", "pp = in(reg) p.as_ptr()"]
    emit(f"/// The double-width `a − b` of {m} limbs for `a, b < p²`, plus `p·2^{64 * n}` when negative.")
    emit("#[inline(always)]")
    emit(f"pub(super) fn wide_sub{n}(a: &[u64; {m}], b: &[u64; {m}], p: &[u64; {n}]) -> [u64; {m}] {{")
    emit(f"    let mut o = [0u64; {m}];")
    asm_block(lines, ops, "pure, readonly, nostack")
    emit("    o")
    emit("}")
    emit()


emit("//! The aarch64 kernels of the engine's prime fields, for 4 and 6 limbs: generated by")
emit("//! `kernels.py` next to this file (do not edit by hand).")
emit("//!")
emit("//! The portable code in `fp.rs` computes the same values; it is what runs elsewhere and under")
emit("//! Miri, and the tests run it on x86_64. Each kernel is one `asm!` block on registers the")
emit("//! compiler allocates, with unbroken `adds`/`adcs` and `subs`/`sbcs` carry chains and `csel`")
emit("//! selections: no branch depends on the data, where the compiled portable code selects the")
emit("//! reduced sum by a branch and builds its borrow chains from comparisons.")
emit()
emit("use core::arch::asm;")
emit()
for n in (4, 6):
    gen_add(n)
    gen_sub(n)
    gen_neg(n)
    gen_mont(n)
    gen_mul_wide(n)
    gen_sqr_wide(n)
    gen_redc(n)
    gen_wide_add(n)
    gen_wide_sub_plain(n)
    gen_wide_sub(n)
sys.stdout.write("\n".join(out).rstrip() + "\n")

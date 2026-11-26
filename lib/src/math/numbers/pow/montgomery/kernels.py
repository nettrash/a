#!/usr/bin/env python3
"""Generates aarch64.rs, the inline assembly of Montgomery's products:
python3 kernels.py > aarch64.rs."""
import re
import sys
import textwrap

# The operands' places by the number of limbs, in registers ('reg') or read from memory by a
# pointer ('mem'): up to four limbs all in registers; from five the modulus and the second factor
# from memory, and from seven the square's operand too (its doubled operand stays in registers), so
# that a kernel takes at most 26 registers. From nine limbs everything is read from memory and only
# the window is in registers.
MUL = {2: ('reg', 'reg', 'reg'), 3: ('reg', 'reg', 'reg'), 4: ('reg', 'reg', 'reg'),
       5: ('reg', 'mem', 'mem'), 6: ('reg', 'mem', 'mem'), 7: ('reg', 'mem', 'mem'),
       8: ('reg', 'mem', 'mem')}
SQR = {2: ('reg', 'reg'), 3: ('reg', 'reg'), 4: ('reg', 'reg'), 5: ('reg', 'mem'),
       6: ('reg', 'mem'), 7: ('mem', 'mem'), 8: ('mem', 'mem')}
LONG = range(9, 17)
KEYWORDS = {"xzr", "cs", "cc", "lo", "hs", "ne", "eq"}


class Kernel:
    """A kernel's lines, on symbolic register names, each of which becomes an operand of its own."""

    def __init__(self, n, a, b, m, pointers=False):
        self.n, self.modes, self.lines, self.inputs = n, {'a': a, 'b': b, 'm': m}, [], {}
        # Whether the operands are raw pointers (`a`) rather than references (`a.as_ptr()`).
        self.pointers = pointers

    def op(self, line):
        self.lines.append(line)

    def pointer(self, x):
        self.inputs[f"{x}p"] = x if self.pointers else f"{x}.as_ptr()"
        return f"{x}p"

    def limb(self, x, j, tmp):
        """The register that holds limb `j` of the operand `x`: its own, or `tmp`, loaded."""
        if self.modes[x] == 'reg':
            name = f"{x}{j}"
            self.inputs[name] = f"{x}[{j}]"
            return name
        self.op(f"ldr {tmp}, [{self.pointer(x)}, #{8 * j}]")
        return tmp

    def each(self, x, js, pair):
        """The limbs `js` of the operand `x`, ascending, as `(j, register, loaded)`: their own
        registers, or loaded two at a time by `ldp` into the registers of `pair` (by `ldr` when one
        is left), which the caller may overwrite once it is done with them."""
        js = list(js)
        if self.modes[x] == 'reg':
            for j in js:
                yield j, self.limb(x, j, None), False
            return
        p = self.pointer(x)
        i = 0
        while i < len(js):
            if i + 1 < len(js) and js[i + 1] == js[i] + 1:
                self.op(f"ldp {pair[0]}, {pair[1]}, [{p}, #{8 * js[i]}]")
                yield js[i], pair[0], True
                yield js[i + 1], pair[1], True
                i += 2
            else:
                self.op(f"ldr {pair[0]}, [{p}, #{8 * js[i]}]")
                yield js[i], pair[0], True
                i += 1

    def minv(self):
        self.inputs['minv'] = 'minv'
        return 'minv'


def first_row(k, x, y, acc, pair, p):
    """`acc[0..N] = x y`, one chain: the low limbs, and the high ones a position up."""
    n = k.n
    for j, r, loaded in k.each(x, range(n), pair):
        k.op(f"umulh {acc[j + 1]}, {y}, {r}")
        if j == 0:
            k.op(f"mul {acc[0]}, {y}, {r}")
            continue
        d = r if loaded else p
        k.op(f"mul {d}, {y}, {r}")
        k.op(f"{'adds' if j == 1 else 'adcs'} {acc[j]}, {acc[j]}, {d}")
    k.op(f"adc {acc[n]}, {acc[n]}, xzr")


def row(k, x, y, acc, low, pair, p, fold):
    """`acc += x y` over positions `low..N`: chain A the low limbs of `x_j y` for `j` from `low` (1
    after the reduction's comparison, which gives its carry in), its carry folded into the high
    limb of `x_(N-1) y`, in `fold`, which has room for it; chain B the high limbs a position up, to
    position `N - 1`. The caller adds `fold` at position `N`."""
    n = k.n
    for x_, (j, r, loaded) in enumerate(k.each(x, range(low, n), pair)):
        if j == n - 1:
            k.op(f"umulh {fold}, {y}, {r}")
        d = r if loaded else p
        k.op(f"mul {d}, {y}, {r}")
        k.op(f"{('adcs' if low else 'adds') if x_ == 0 else 'adcs'} {acc[j]}, {acc[j]}, {d}")
    k.op(f"adc {fold}, {fold}, xzr")
    for x_, (j, r, loaded) in enumerate(k.each(x, range(n - 1), pair)):
        d = r if loaded else p
        k.op(f"umulh {d}, {y}, {r}")
        k.op(f"{'adds' if x_ == 0 else 'adcs'} {acc[j + 1]}, {acc[j + 1]}, {d}")


def reduce_row(k, acc, pair, p, q, fold, lazy, minv):
    """A row of the reduction: `q = acc_0 (-1/m) mod 2^64`, `acc += q m`, which clears `acc_0`, its
    carry being 1 unless `acc_0` was 0, as the comparison gives it; the caller then moves the window
    up a limb. `minv` is its register, or None to load it from after the modulus."""
    n = k.n
    if minv is None:
        k.op(f"ldr {q}, [{k.pointer('m')}, #{8 * n}]")
        k.op(f"mul {q}, {acc[0]}, {q}")
    else:
        k.op(f"mul {q}, {acc[0]}, {minv}")
    k.op(f"cmp {acc[0]}, #1")
    row(k, 'm', q, acc, 1, pair, p, fold)
    if lazy:
        k.op(f"adc {acc[n]}, {acc[n]}, {fold}")
    else:
        k.op(f"adcs {acc[n]}, {acc[n]}, {fold}")
        k.op(f"adc {acc[n + 1]}, {acc[n + 1]}, xzr")


def subtract(k, r, top, free, pair):
    """`r - m` when `top B^N + r` is at least `m`: with `N` free registers, a chain of subtractions
    and a selection by its borrow; with two, a comparison and then the subtraction of `m` masked by
    its outcome."""
    n = k.n
    if len(free) >= n:
        for x, mx, _ in k.each('m', range(n), pair):
            k.op(f"{'subs' if x == 0 else 'sbcs'} {free[x]}, {r[x]}, {mx}")
        k.op(f"sbcs xzr, {top}, xzr")
        for x in range(n):
            k.op(f"csel {r[x]}, {free[x]}, {r[x]}, cs")
        return
    mask = free[0]
    for x, mx, _ in k.each('m', range(n), pair):
        k.op(f"cmp {r[x]}, {mx}" if x == 0 else f"sbcs xzr, {r[x]}, {mx}")
    k.op(f"sbcs xzr, {top}, xzr")
    k.op(f"csetm {mask}, cs")
    for x, mx, _ in k.each('m', range(n), pair):
        k.op(f"and {mx}, {mx}, {mask}")
        k.op(f"{'subs' if x == 0 else 'sbcs'} {r[x]}, {r[x]}, {mx}")


def mul(n, lazy, long=False):
    """`a b / R mod m`: for each limb `b_i`, a row of the product and a row of the reduction on a
    window of `N + 2` limbs (`N + 1` when lazy), the coarsely integrated operand scanning."""
    k = Kernel(n, *(('mem', 'mem', 'mem') if long else MUL[n]), pointers=long)
    p, q = 'p', 'q'
    pair = ('aj', 'ak')
    width = n + 1 if lazy else n + 2
    acc = [f"t{i}" for i in range(width)]
    minv = None if long else k.minv()
    for i in range(n):
        bi = k.limb('b', i, 'bi')
        if i == 0:
            first_row(k, 'a', bi, acc, pair, p)
            if not lazy:
                k.op(f"mov {acc[n + 1]}, #0")
        else:
            # The fold in `q`, free until the reduction.
            row(k, 'a', bi, acc, 0, pair, p, q)
            if lazy:
                # Position N is new: the window holds less than `B^(N+1)`.
                k.op(f"adc {acc[n]}, {q}, xzr")
            else:
                k.op(f"adcs {acc[n]}, {acc[n]}, {q}")
                k.op(f"cset {acc[n + 1]}, cs")
        # The reduction's fold in `bi`'s register, free after the product's row.
        reduce_row(k, acc, pair, p, q, bi, lazy, minv)
        acc = acc[1:] + acc[:1]
    r = acc[:n]
    if not lazy:
        free = [acc[n + 1], q, p] if long else (
            [acc[n + 1], q, p] + ([f"a{x}" for x in range(n)] if k.modes['a'] == 'reg' else []))
        subtract(k, r, acc[n], free, pair)
    return k, r


def sqr(n, lazy, long=False):
    """`a^2 / R mod m`: row `i` adds `a_i [a_i, e_(i+1), d_(i+2), ..., d_(N-1), c]` at positions
    `i..N` of the window, for the doubled operand `d = 2a` (`d_j = a_j << 1 | a_(j-1) >> 63`),
    `e_(i+1)` its limb without the bit from `a_i`, and `c` the top bit of `a`, a digit of rows
    `i < N - 1` only: the products of different limbs once, doubled, and the squares, in rows that
    end where the product's do, so that each is followed by a row of the reduction as in [`mul`].
    Up to eight limbs `d` is computed into registers; from nine it is read from memory, as `dp`."""
    if long:
        k = Kernel(n, 'mem', 'reg', 'mem', pointers=True)
    else:
        am, mm = SQR[n]
        k = Kernel(n, am, 'reg', mm)
    p, q, mk = 'p', 'q', 'mk'
    pair = ('aj', 'ak')
    acc = [f"t{i}" for i in range(n + 2)]
    minv = None if long else k.minv()
    if long:
        k.inputs['dp'] = 'd'
        d = None
    else:
        d = [None] + [f"d{j}" for j in range(1, n)]
        prev = None
        for j, r, _ in k.each('a', range(n), pair):
            if prev is not None:
                k.op(f"extr {d[j]}, {r}, {prev}, #63")
            if k.modes['a'] == 'mem' and j % 2 == 1:
                # The pair is reloaded next: keep the odd limb, needed for the next limb's `extr`.
                k.op(f"mov {p}, {r}")
                prev = p
            else:
                prev = r
        # All ones when the top bit of `a` is set: the top digit, times `a_i`, is `a_i` or 0.
        k.op(f"asr {mk}, {prev}, #63")
    for i in range(n):
        ai = k.limb('a', i, 'ai')
        # The digits from i + 1: `e_(i+1)` in `q`, free until the reduction, then `d_(i+2)..`.
        if i + 1 < n:
            if long:
                k.op(f"ldr {q}, [dp, #{8 * (i + 1)}]")
                k.op(f"and {q}, {q}, #0xfffffffffffffffe")
            else:
                k.op(f"and {q}, {d[i + 1]}, #0xfffffffffffffffe")

        def digits():
            """`(j, register, loaded)` for the digits `j = i + 1..N-1`."""
            if i + 1 < n:
                yield i + 1, q, False
            if long:
                yield from k_each_d(k, range(i + 2, n), pair)
            else:
                for j in range(i + 2, n):
                    yield j, d[j], False

        def masked():
            if long:
                k.op(f"ldr {p}, [dp]")
                k.op(f"and {p}, {ai}, {p}")
            else:
                k.op(f"and {p}, {ai}, {mk}")
        if i == 0:
            k.op(f"mul {acc[0]}, {ai}, {ai}")
            k.op(f"umulh {acc[1]}, {ai}, {ai}")
            for j, r, loaded in digits():
                k.op(f"umulh {acc[j + 1]}, {ai}, {r}")
                dd = r if loaded else p
                k.op(f"mul {dd}, {ai}, {r}")
                k.op(f"{'adds' if j == 1 else 'adcs'} {acc[j]}, {acc[j]}, {dd}")
            masked()
            k.op(f"adcs {acc[n]}, {acc[n]}, {p}")
            k.op(f"cset {acc[n + 1]}, cs")
        else:
            k.op(f"mul {p}, {ai}, {ai}")
            k.op(f"adds {acc[i]}, {acc[i]}, {p}")
            for j, r, loaded in digits():
                dd = r if loaded else p
                k.op(f"mul {dd}, {ai}, {r}")
                k.op(f"adcs {acc[j]}, {acc[j]}, {dd}")
            if i < n - 1:
                masked()
                k.op(f"adcs {acc[n]}, {acc[n]}, {p}")
            else:
                k.op(f"adcs {acc[n]}, {acc[n]}, xzr")
            k.op(f"cset {acc[n + 1]}, cs")
            k.op(f"umulh {p}, {ai}, {ai}")
            k.op(f"adds {acc[i + 1]}, {acc[i + 1]}, {p}")
            for j, r, loaded in digits():
                dd = r if loaded else p
                k.op(f"umulh {dd}, {ai}, {r}")
                k.op(f"adcs {acc[j + 1]}, {acc[j + 1]}, {dd}")
            k.op(f"adc {acc[n + 1]}, {acc[n + 1]}, xzr")
        # The square's window can pass `R` even when lazy: the reduction keeps its top limb. Its
        # fold in `ai`'s register, free after the row.
        reduce_row(k, acc, pair, p, q, ai, False, minv)
        acc = acc[1:] + acc[:1]
    r = acc[:n]
    if not lazy:
        free = [acc[n + 1], q, p] if long else [acc[n + 1], q, p, mk] + [d[j] for j in range(1, n)]
        subtract(k, r, acc[n], free, pair)
    return k, r


def low_mul(n):
    """`a b mod B^N`, the low half of a product, for the low halves of the powers modulo a power of
    two: row `i` adds `a_j b_i` at positions `i + j < N`, chain A the low limbs and chain B the high
    ones a position up, the carries out of limb `N - 1` dropped; the result in registers."""
    k = Kernel(n, 'mem', 'mem', 'mem', pointers=True)
    pair = ('aj', 'ak')
    acc = [f"t{i}" for i in range(n)]
    for i in range(n):
        k.op(f"ldr bi, [{k.pointer('b')}, #{8 * i}]")
        last = n - 1 - i
        if i == 0:
            for j, r, _ in k.each('a', range(n), pair):
                if j + 1 < n:
                    k.op(f"umulh {acc[j + 1]}, bi, {r}")
                if j == 0:
                    k.op(f"mul {acc[0]}, bi, {r}")
                    continue
                k.op(f"mul {r}, bi, {r}")
                k.op(f"{'adds' if j == 1 else 'adcs'} {acc[j]}, {acc[j]}, {r}")
            continue
        for x_, (j, r, _) in enumerate(k.each('a', range(last + 1), pair)):
            k.op(f"mul {r}, bi, {r}")
            k.op(f"{'adds' if x_ == 0 else 'adcs'} {acc[i + j]}, {acc[i + j]}, {r}")
        for x_, (j, r, _) in enumerate(k.each('a', range(last), pair)):
            k.op(f"umulh {r}, bi, {r}")
            k.op(f"{'adds' if x_ == 0 else 'adcs'} {acc[i + j + 1]}, {acc[i + j + 1]}, {r}")
    return k, acc


def low_sqr(n):
    """`a^2 mod B^N`: row `i` adds `a_i [a_i, e_(i+1), d_(i+2), ...]` at positions `2i..N-1`, as the
    square of [`sqr`] does, the doubled operand `d` read from memory; the top bit's digit falls
    beyond `N`, and so do the carries out of limb `N - 1`."""
    k = Kernel(n, 'mem', 'reg', 'mem', pointers=True)
    k.inputs['dp'] = 'd'
    pair = ('aj', 'ak')
    acc = [f"t{i}" for i in range(n)]
    i = 0
    while 2 * i < n:
        k.op(f"ldr ai, [{k.pointer('a')}, #{8 * i}]")
        top = n - 1 - i
        if i + 1 <= top:
            k.op(f"ldr q, [dp, #{8 * (i + 1)}]")
            k.op("and q, q, #0xfffffffffffffffe")

        def digits(stop):
            if i + 1 <= stop:
                yield i + 1, 'q', False
            yield from k_each_d(k, range(i + 2, stop + 1), pair)
        if i == 0:
            k.op(f"mul {acc[0]}, ai, ai")
            if n > 1:
                k.op(f"umulh {acc[1]}, ai, ai")
            for j, r, loaded in digits(top):
                if j + 1 < n:
                    k.op(f"umulh {acc[j + 1]}, ai, {r}")
                d = r if loaded else 'p'
                k.op(f"mul {d}, ai, {r}")
                k.op(f"{'adds' if j == 1 else 'adcs'} {acc[j]}, {acc[j]}, {d}")
        else:
            k.op("mul p, ai, ai")
            k.op(f"adds {acc[2 * i]}, {acc[2 * i]}, p")
            for j, r, loaded in digits(top):
                d = r if loaded else 'p'
                k.op(f"mul {d}, ai, {r}")
                k.op(f"adcs {acc[i + j]}, {acc[i + j]}, {d}")
            if 2 * i + 1 < n:
                k.op("umulh p, ai, ai")
                k.op(f"adds {acc[2 * i + 1]}, {acc[2 * i + 1]}, p")
                for j, r, loaded in digits(top - 1):
                    d = r if loaded else 'p'
                    k.op(f"umulh {d}, ai, {r}")
                    k.op(f"adcs {acc[i + j + 1]}, {acc[i + j + 1]}, {d}")
        i += 1
    return k, acc


def k_each_d(k, js, pair):
    """The limbs `js` of the doubled operand at `dp`, two at a time, as [`Kernel::each`] gives an
    operand's."""
    js = list(js)
    i = 0
    while i < len(js):
        if i + 1 < len(js):
            k.op(f"ldp {pair[0]}, {pair[1]}, [dp, #{8 * js[i]}]")
            yield js[i], pair[0], True
            yield js[i + 1], pair[1], True
            i += 2
        else:
            k.op(f"ldr {pair[0]}, [dp, #{8 * js[i]}]")
            yield js[i], pair[0], True
            i += 1


def rust(name, doc, args, kernel, unsafe_fn=False):
    """The kernel as a Rust function: every symbolic register an operand for the compiler to place,
    an input of the operands' limbs, pointers or `minv`, an output of the result's limbs, or a
    scratch one."""
    k, outputs = kernel
    lines = k.lines
    names, written = set(), set()
    for l in lines:
        parts = l.replace(",", " ").replace("[", " ").replace("]", " ").split()
        for t in parts[1:]:
            if not t.startswith("#") and t not in KEYWORDS:
                names.add(t)
        if parts[0] not in ("cmp", "cmn", "tst") and parts[1] != "xzr":
            written.add(parts[1])
    ops = []
    for r in sorted(names, key=lambda x: (re.sub(r"\d+", "", x), int(re.sub(r"\D", "", x) or 0))):
        if r in k.inputs and r in outputs:
            ops.append(f"{r} = inout(reg) {k.inputs[r]} => r[{outputs.index(r)}]")
        elif r in k.inputs and r in written:
            ops.append(f"{r} = inout(reg) {k.inputs[r]} => _")
        elif r in k.inputs:
            ops.append(f"{r} = in(reg) {k.inputs[r]}")
        elif r in outputs:
            ops.append(f"{r} = out(reg) r[{outputs.index(r)}]")
        else:
            ops.append(f"{r} = out(reg) _")

    def placed(line):
        op, _, rest = line.partition(" ")
        rest = re.sub(r"(?<![#\w])([A-Za-z_]\w*)",
                      lambda m: m.group(1) if m.group(1) in KEYWORDS else "{" + m.group(1) + "}", rest)
        return f"{op} {rest}"
    body = ",\n".join(f'            "{placed(l)}"' for l in lines)
    ops = "\n".join(f"            {o}," for o in ops)
    memory = any(v == 'mem' for v in k.modes.values())
    options = "pure, readonly, nostack" if memory else "pure, nomem, nostack"
    safety = ("the block reads and writes registers only, and reads the limbs of the operands,\n"
              "    // which the references hold." if memory else "the block reads and writes registers only.")
    n = len(outputs)
    docs = "\n".join("/// " + line for d in doc for line in textwrap.wrap(d, 96))
    if unsafe_fn:
        safety = ("the block reads and writes registers only, and reads the limbs of the operands,\n"
                  "    // which the caller guarantees.")
        docs += "\n///\n/// # Safety\n///\n/// The limbs of the operands must be valid for reads."
    return f'''{docs}
#[inline(always)]
{"unsafe " if unsafe_fn else ""}fn {name}({args}) -> [u64; {n}] {{
    let mut r = [0; {n}];
    // SAFETY: {safety}
    unsafe {{
        asm!(
{body},
{ops}
            options({options}),
        );
    }}
    r
}}
'''


HEAD = '''//! The aarch64 kernels of Montgomery's products and of the low halves of products, in inline
//! assembly. **This file is generated by `kernels.py`, next to it: change that, not this.**
//!
//! - Two to eight limbs, [`mul`], [`sqr`], [`lazy_mul`], [`lazy_sqr`]: each kernel takes its
//!   operands' limbs and gives its result in registers and is inlined, so that the residues of a
//!   computation stay in registers from one product to the next; the compiler chooses the
//!   registers, at most 26 a kernel. From five limbs the modulus and the second factor, and from
//!   seven the square's operand, are read from memory where each limb is needed, two at a time
//!   (`ldp`), into registers that the products then take.
//! - Eight to sixteen limbs, [`long`]: every operand read from memory and only the window in
//!   registers, one kernel for each length, out of line, so that it is in the code once.
//! - Nine to sixteen limbs, [`low`]: the low halves of products and squares, for the powers modulo
//!   a power of two, their limbs in registers.
//!
//! The products are the coarsely integrated operand scanning of Koç, Acar and Kaliski: for each
//! limb `b_i` a row of the product `a b_i` and a row of the reduction `q m`, `q = t_0 (-1/m) mod
//! 2^64`, on a window of `N + 2` limbs that moves up a limb after each row of the reduction, which
//! clears its low limb. A row is two carry chains, the low limbs of its products and then their
//! high limbs a position up, each `mul` or `umulh` next to the addition that takes it, as `mul` and
//! `umulh` leave the flags alone; the first chain's carry out goes into the high limb of the last
//! product, which has room for it (`(2^64 - 1)^2` has a high limb of `2^64 - 2`), so that a row
//! takes one carry flag from end to end. The low limb of `q m_0` is known to clear `t_0`: its
//! carry is 1 unless `t_0` is 0, which one comparison gives. The rows of the product and of the
//! reduction interleave so that the reduction's chain of quotients starts with the product: the
//! same work as a whole product and then REDC took 1.1 to 1.2 times as long.
//!
//! A square takes the products of different limbs once: with `d = 2a`, row `i` adds
//! `a_i [a_i, e_(i+1), d_(i+2), ..., d_(N-1), c]` from position `i` of the window, where
//! `e_(i+1)` is `d_(i+1)` without the bit from `a_i` and `c`, the top bit of `a`, is a digit of
//! every row but the last; as the rows of a product, each is followed by a row of the reduction.
//! The rows end where a product's do, so the window, which stays below `3 B^N` (`B = 2^64`),
//! holds them.
//!
//! The fully reduced kernels take residues below `m` and give one below `m`, by one subtraction of
//! `m` chosen by its borrow at the end (from nine limbs, a comparison and then the subtraction of
//! `m` masked by its outcome, which takes two registers rather than `N`); the lazy ones, for
//! `4m < R`, take residues below `2m` and give one below `2m` with no subtraction, and their
//! products need a window of `N + 1` limbs only, as `3m (B + 1) < B^(N+1)`. The kernels from nine
//! limbs also take residues below `R` and give one below `R`, as the powers' ring keeps them.
//!
//! On an Apple M4 Max the kernels are bound by the carry chains' additions, which the core runs
//! about 2.5 a cycle in these chains (four independent ones a cycle), two a limb product: a product
//! of eight limbs, 128 limb products, took 99 cycles beside others. Measured with other work on
//! the machine, in cycles a product after the one before / beside three others, against the Rust
//! rows they replaced (in registers to eight limbs, which square by a product) or the
//! multiplication's kernels and REDC (from nine), and against GMP 6.3's `mpn_mul_basecase` and
//! `mpn_redc_1`:
//!
//! | Limbs | product | square | before | GMP |
//! | --- | --- | --- | --- | --- |
//! | 2 | 24 / 7.5 | 23 / 7.8 | 25 / 10.8 | 52 / 64 |
//! | 4 | 46 / 29 | 44 / 27 | 57 / 51 | 98 / 81 |
//! | 8 | 107 / 99 | 111 / 87 | 206 / 207 | 211 / 201 |
//! | 9 | 142 / 128 | 133 / 113 | 254 / 273 | 249 / 230 |
//! | 12 | 237 / 222 | 208 / 189 | 344 / 378 | 403 / 386 |
//! | 16 | 423 / 410 | 347 / 329 | 548 / 583 | 757 / 757 |

use core::arch::asm;
'''

TAIL = '''
/// `x` as an array of `M` limbs, which it is: `N == M`.
#[inline(always)]
fn same<const N: usize, const M: usize>(x: &[u64; N]) -> &[u64; M] {
    x.as_slice().try_into().unwrap_or_else(|_| unreachable!())
}

/// `x` as an array of `M` limbs, which it is: `N == M`.
#[inline(always)]
fn into<const M: usize, const N: usize>(x: [u64; M]) -> [u64; N] {
    x.as_slice().try_into().unwrap_or_else(|_| unreachable!())
}
'''

LONG_DISPATCH = '''
/// Whether [`long`] takes residues of `n` limbs: from eight to sixteen.
#[inline(always)]
pub(in crate::math::numbers) const fn has_long(n: usize) -> bool {
    8 <= n && n <= 16
}

/// Sets the `n` limbs at `r` to `a b / R mod m` (or `a^2` with `b` null), `R = 2^(64 n)`, for the
/// `n` limbs at `a` and `b` and the `n + 1` limbs at `m`, the modulus and then `-1/m mod 2^64`:
/// below `m` for `a` and `b` below `m`, below `R` for `a` and `b` below `R`. Out of line, so that
/// each length's kernel is in the code once; eight limbs by the kernels in registers.
///
/// # Safety
///
/// The limbs must be valid, `n` a length that [`has_long`] takes; `r` may be `a` or `b`, which are
/// read before it is written.
#[inline(never)]
pub(in crate::math::numbers) unsafe fn long(
    r: *mut u64,
    a: *const u64,
    b: *const u64,
    m: *const u64,
    n: usize,
) {
    /// The doubled operand of a square of `N` limbs, as the kernels take it.
    #[inline(always)]
    unsafe fn doubled<const N: usize>(a: *const u64) -> [u64; N] {
        let mut d = [0; N];
        // SAFETY: as the caller guarantees, the `N` limbs at `a` are valid.
        unsafe {
            d[0] = ((*a.add(N - 1) as i64) >> 63) as u64;
            for (j, dj) in d.iter_mut().enumerate().skip(1) {
                *dj = (*a.add(j) << 1) | (*a.add(j - 1) >> 63);
            }
        }
        d
    }
    /// Stores the limbs `x` at `r`.
    #[inline(always)]
    unsafe fn store<const N: usize>(r: *mut u64, x: [u64; N]) {
        // SAFETY: as the caller guarantees, the `N` limbs at `r` are valid for writes.
        unsafe { core::ptr::copy_nonoverlapping(x.as_ptr(), r, N) }
    }
    /// The `N` limbs at `x`.
    #[inline(always)]
    unsafe fn load<const N: usize>(x: *const u64) -> [u64; N] {
        let mut a = [0; N];
        // SAFETY: as the caller guarantees, the `N` limbs at `x` are valid.
        unsafe { core::ptr::copy_nonoverlapping(x, a.as_mut_ptr(), N) };
        a
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        if n == 8 {
            let (x, mm) = (load::<8>(a), load::<8>(m));
            let minv = *m.add(8);
            store::<8>(
                r,
                if b.is_null() {
                    sqr_8(&x, &mm, minv)
                } else {
                    mul_8(&x, &load::<8>(b), &mm, minv)
                },
            );
            return;
        }
        macro_rules! lengths {
            ($($n:literal => $mul:ident, $sqr:ident;)*) => {
                match n {
                    $($n => store::<$n>(r, if b.is_null() {
                        $sqr(a, doubled::<$n>(a).as_ptr(), m)
                    } else {
                        $mul(a, b, m)
                    }),)*
                    _ => unreachable!(),
                }
            };
        }
        lengths! {
            9 => long_mul_9, long_sqr_9;
            10 => long_mul_10, long_sqr_10;
            11 => long_mul_11, long_sqr_11;
            12 => long_mul_12, long_sqr_12;
            13 => long_mul_13, long_sqr_13;
            14 => long_mul_14, long_sqr_14;
            15 => long_mul_15, long_sqr_15;
            16 => long_mul_16, long_sqr_16;
        }
    }
}
'''

LOW_DISPATCH = '''
/// Whether [`low`] takes residues of `n` limbs: from nine to sixteen.
#[inline(always)]
pub(in crate::math::numbers) const fn has_low(n: usize) -> bool {
    9 <= n && n <= 16
}

/// Sets the `n` limbs at `r` to the low `n` limbs of the product of the `n` limbs at `a` and `b`,
/// or of the square of `a` with `b` null, for the powers modulo a power of two. Out of line, so
/// that each length's kernel is in the code once.
///
/// # Safety
///
/// The limbs must be valid, `n` a length that [`has_low`] takes; `r` may be `a` or `b`, which are
/// read before it is written.
#[inline(never)]
pub(in crate::math::numbers) unsafe fn low(r: *mut u64, a: *const u64, b: *const u64, n: usize) {
    /// The doubled operand of a square of `N` limbs, from limb 1, as the kernels take it.
    #[inline(always)]
    unsafe fn doubled<const N: usize>(a: *const u64) -> [u64; N] {
        let mut d = [0; N];
        // SAFETY: as the caller guarantees, the `N` limbs at `a` are valid.
        unsafe {
            for (j, dj) in d.iter_mut().enumerate().skip(1) {
                *dj = (*a.add(j) << 1) | (*a.add(j - 1) >> 63);
            }
        }
        d
    }
    /// Stores the limbs `x` at `r`.
    #[inline(always)]
    unsafe fn store<const N: usize>(r: *mut u64, x: [u64; N]) {
        // SAFETY: as the caller guarantees, the `N` limbs at `r` are valid for writes.
        unsafe { core::ptr::copy_nonoverlapping(x.as_ptr(), r, N) }
    }
    // SAFETY: as the caller guarantees.
    unsafe {
        macro_rules! lengths {
            ($($n:literal => $mul:ident, $sqr:ident;)*) => {
                match n {
                    $($n => store::<$n>(r, if b.is_null() {
                        $sqr(a, doubled::<$n>(a).as_ptr())
                    } else {
                        $mul(a, b)
                    }),)*
                    _ => unreachable!(),
                }
            };
        }
        lengths! {
            9 => low_mul_9, low_sqr_9;
            10 => low_mul_10, low_sqr_10;
            11 => low_mul_11, low_sqr_11;
            12 => low_mul_12, low_sqr_12;
            13 => low_mul_13, low_sqr_13;
            14 => low_mul_14, low_sqr_14;
            15 => low_mul_15, low_sqr_15;
            16 => low_mul_16, low_sqr_16;
        }
    }
}
'''


def dispatcher(fname, kind, lazy):
    sq = kind == 'sqr'
    params = ["a: &[u64; N]"] + ([] if sq else ["b: &[u64; N]"]) + ["m: &[u64; N]", "minv: u64"]
    call = "same(a), same(m), minv" if sq else "same(a), same(b), same(m), minv"
    what = "a^2" if sq else "a b"
    bound = ("below `2m`, for residues below `2m` and `4m < R`" if lazy
             else "below `m`, for residues below `m`")
    arms = "\n".join(f"        {n} => into({fname}_{n}({call}))," for n in range(2, 9))
    params = "\n".join(f"    {p}," for p in params)
    return f'''
/// `{what} / R mod m`, {bound}, `R = 2^(64 N)`,
/// `minv = -1/m mod 2^64`: by the kernel of `N` limbs, `None` for the lengths without one.
#[inline(always)]
pub(in crate::math::numbers) fn {fname}<const N: usize>(
{params}
) -> Option<[u64; N]> {{
    Some(match N {{
{arms}
        _ => return None,
    }})
}}'''


def main():
    out = [HEAD]
    for lazy in (False, True):
        pre = "lazy_" if lazy else ""
        for n in range(2, 9):
            a, b, m = f"a: &[u64; {n}]", f"b: &[u64; {n}]", f"m: &[u64; {n}]"
            if lazy:
                doc_m = [f"`a b / R mod m` below `2m` for `a` and `b` below `2m`, `R = 2^{64 * n}`, `4m < R`."]
                doc_s = [f"`a^2 / R mod m` below `2m` for `a` below `2m`, `R = 2^{64 * n}`, `4m < R`."]
            else:
                doc_m = [f"`a b / R mod m` below `m` for `a` and `b` below `m`, `R = 2^{64 * n}`."]
                doc_s = [f"`a^2 / R mod m` below `m` for `a` below `m`, `R = 2^{64 * n}`."]
            out.append(rust(f"{pre}mul_{n}", doc_m, f"{a}, {b}, {m}, minv: u64", mul(n, lazy)))
            out.append(rust(f"{pre}sqr_{n}", doc_s, f"{a}, {m}, minv: u64", sqr(n, lazy)))
    for lazy in (False, True):
        pre = "lazy_" if lazy else ""
        out.append(dispatcher(f"{pre}mul", 'mul', lazy))
        out.append(dispatcher(f"{pre}sqr", 'sqr', lazy))
    for n in LONG:
        p = "*const u64"
        out.append(rust(f"long_mul_{n}", [f"`a b / R mod m` for the {n} limbs at `a` and `b`, `R = 2^{64 * n}`, below `m` for `a` and `b` below `m`, and below `R` for `a` and `b` below `R`; `m` is followed by `-1/m mod 2^64`."],
                        f"a: {p}, b: {p}, m: {p}", mul(n, False, long=True), unsafe_fn=True))
        out.append(rust(f"long_sqr_{n}", [f"`a^2 / R mod m` for the {n} limbs at `a`, `R = 2^{64 * n}`, as [`long_mul_{n}`], with `d` the doubled operand: the mask of `a`'s top bit, then `a_j << 1 | a_(j-1) >> 63`."],
                        f"a: {p}, d: {p}, m: {p}", sqr(n, False, long=True), unsafe_fn=True))
    for n in LONG:
        p = "*const u64"
        out.append(rust(f"low_mul_{n}", [f"`a b mod 2^{64 * n}` for the {n} limbs at `a` and `b`."],
                        f"a: {p}, b: {p}", low_mul(n), unsafe_fn=True))
        out.append(rust(f"low_sqr_{n}", [f"`a^2 mod 2^{64 * n}` for the {n} limbs at `a`, with `d` the doubled operand: `a_j << 1 | a_(j-1) >> 63` from limb 1."],
                        f"a: {p}, d: {p}", low_sqr(n), unsafe_fn=True))
    out.append(LONG_DISPATCH)
    out.append(LOW_DISPATCH)
    out.append(TAIL)
    # At most one blank line between items, as rustfmt leaves them.
    sys.stdout.write(re.sub(r"\n{3,}", "\n\n", "\n".join(out)))


main()

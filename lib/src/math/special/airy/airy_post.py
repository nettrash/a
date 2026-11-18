"""Pads the piece and zero tables of the Airy tables to common widths (P_N, C_N): `post(src)`,
called by gen_airy.py. As a script: python3 airy_post.py in.rs [patch.rs] > out.rs (`patch`
replaces the arrays of the same names first)."""

import re
import sys

PAT = re.compile(r'pub\(super\) const (\w+): \[\[f64; (\d+)\]; (\d+)\] = \[\n(.*?)\n\];\n', re.S)
ONE = r'/// [^\n]*\npub\(super\) const %s: \[f64; \d+\] = \[\n.*?\];\n'


def post(src, patch=None):
    if patch is not None:
        for m in re.finditer(ONE % r'(\w+)', patch, re.S):
            src = re.sub(ONE % m.group(1), lambda _: m.group(0), src, flags=re.S)

    def width(suffix):
        return max(int(m.group(2)) for m in PAT.finditer(src) if m.group(1).endswith(suffix))

    pn, cn = width('_P'), width('_C')

    def pad(m):
        name, n = m.group(1), int(m.group(2))
        target = pn if name.endswith('_P') else cn if name.endswith('_C') else n
        if target == n:
            return m.group(0)
        rows = re.findall(r'    \[\n(.*?)\n    \],', m.group(4), re.S)
        body = ""
        for r in rows:
            vals = [v.strip().rstrip(',') for v in r.split('\n') if v.strip()]
            vals += ['0.0'] * (target - len(vals))
            body += "    [\n" + "".join(f"        {v},\n" for v in vals) + "    ],\n"
        return f"pub(super) const {name}: [[f64; {target}]; {m.group(3)}] = [\n{body}];\n"

    src = PAT.sub(pad, src)
    return src + ("/// The widths of the piece and zero tables.\n"
                  f"pub(super) const P_N: usize = {pn};\npub(super) const C_N: usize = {cn};\n")


if __name__ == "__main__":
    patch = open(sys.argv[2]).read() if len(sys.argv) > 2 else None
    sys.stdout.write(post(open(sys.argv[1]).read(), patch))

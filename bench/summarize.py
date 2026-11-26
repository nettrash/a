#!/usr/bin/env python3
"""Summarises scoreboards: bench/summarize.py [TSV ...] (default bench/results/*.tsv) prints, as
Markdown, per area the rows, references and operations, per group the worst ratio of ours to the
fastest reference of each (operation, params), and the number of losses (that ratio > 1.05).
The ratio is ours_ns / ref_ns of the row of the fastest reference (ours and that reference were
timed in alternation in the same process). Rows without a reference time are not counted."""
import glob
import os
import sys

HEADER = "area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes"


def num(s):
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v > 0 else None


def summarise(path):
    rows, refs, fastest = 0, set(), {}
    with open(path) as f:
        head = f.readline().rstrip("\n")
        if head != HEADER:
            return f"## {os.path.basename(path)}\n\nnot a scoreboard (header: {head[:60]!r})\n"
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) != 9:
                continue
            rows += 1
            ref, ours = num(c[6]), num(c[4])
            if ref is None or ours is None:
                continue
            refs.add(c[5])
            key = (c[1], c[2], c[3])
            if key not in fastest or ref < fastest[key][0]:
                fastest[key] = (ref, ours / ref, c[5])
    area = os.path.basename(path).rsplit(".", 1)[0]
    out = [f"## {area}", "", f"{rows} rows, {len(refs)} references, {len(fastest)} cases with a reference", ""]
    if not fastest:
        return "\n".join(out) + "\n"
    groups = {}
    for (g, op, p), (ref, ratio, name) in fastest.items():
        w = groups.setdefault(g, [0.0, "", 0, 0])
        w[3] += 1
        if ratio > 1.05:
            w[2] += 1
        if ratio > w[0]:
            w[0], w[1] = ratio, f"{op} [{p}] vs {name}"
    out += ["| group | cases | losses (> 1.05) | worst ratio vs fastest | worst case |", "|---|---:|---:|---:|---|"]
    for g, (worst, what, losses, n) in groups.items():
        out.append(f"| {g} | {n} | {losses} | {worst:.3g} | {what.replace('|', '/')} |")
    total = sum(w[2] for w in groups.values())
    out += ["", f"losses: {total} of {len(fastest)} cases", ""]
    return "\n".join(out) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "results", "*.tsv")))
    print("# Scoreboard summary\n")
    for p in paths:
        print(summarise(p))


if __name__ == "__main__":
    main()

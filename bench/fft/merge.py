#!/usr/bin/env python3
"""Merges several runs of the scoreboard: per row (group, operation, params, reference) the
median of the runs' ours_ns and ref_ns, the ratio recomputed from them. The notes keep the
first run's text without its per-run spreads, and say how far the runs' ratios spread when
that is more than 10 %, and the widest within-run spread when one was over 10 %.

Usage: python3 -I bench/fft/merge.py RUN1.tsv RUN2.tsv ... > fft.tsv
"""
import csv
import re
import statistics
import sys

HEADER = ["area", "group", "operation", "params", "ours_ns", "reference", "ref_ns", "ratio", "notes"]
IQR = re.compile(r"(?:; )?(ours|ref) iqr (\d+)%")


def fmt_ns(v):
    if v >= 1000:
        return f"{v:.0f}"
    if v >= 100:
        return f"{v:.1f}"
    if v >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def fmt_ratio(r):
    import math
    e = math.floor(math.log10(r))
    d = max(2 - e, 0)
    return f"{r:.{d}f}"


def main():
    runs = []
    for p in sys.argv[1:]:
        with open(p) as f:
            runs.append(list(csv.DictReader(f, delimiter="\t")))
    order, rows = [], {}
    for k, run in enumerate(runs):
        for r in run:
            key = (r["group"], r["operation"], r["params"], r["reference"])
            if key not in rows:
                rows[key] = []
                order.append(key)
            rows[key].append(r)
    w = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    w.writerow(HEADER)
    for key in order:
        rs = rows[key]
        ours = statistics.median(float(r["ours_ns"]) for r in rs)
        ref = statistics.median(float(r["ref_ns"]) for r in rs)
        ratios = [float(r["ours_ns"]) / float(r["ref_ns"]) for r in rs]
        base = IQR.sub("", rs[0]["notes"]).strip("; ").strip()
        extra = []
        if len(rs) > 1 and max(ratios) / min(ratios) > 1.10:
            extra.append(f"ratio across {len(rs)} runs {min(ratios):.3g} to {max(ratios):.3g}")
        iqrs = [int(m.group(2)) for r in rs for m in IQR.finditer(r["notes"])]
        if iqrs:
            extra.append(f"widest within-run iqr {max(iqrs)}%")
        if len(rs) < len(runs):
            extra.append(f"in {len(rs)} of {len(runs)} runs")
        notes = "; ".join([x for x in [base] + extra if x])
        w.writerow([rs[0]["area"], key[0], key[1], key[2], fmt_ns(ours), key[3], fmt_ns(ref),
                    fmt_ratio(ours / ref), notes])


main()

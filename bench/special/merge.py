"""Merges the harness's row files into the scoreboard, in the case order of `special-bench
--list`, each case's SciPy row after its other references. Files are read in the order given
(oldest first): a later row of the same case and reference replaces an earlier one.

usage: python3 merge.py LISTFILE ROWFILE... > special.tsv
"""
import sys

HEADER = "area\tgroup\toperation\tparams\tours_ns\treference\tref_ns\tratio\tnotes"


def main():
    order = []
    for line in open(sys.argv[1]):
        key = line.rstrip("\n").split("\t")[0]
        order.append(tuple(key.split("|", 2)))
    rank = {k: i for i, k in enumerate(order)}
    rows = []
    seen = {}
    for f in sys.argv[2:]:
        for line in open(f):
            line = line.rstrip("\n")
            if not line or line.startswith("area\t"):
                continue
            c = line.split("\t")
            if len(c) != 9:
                print(f"bad row in {f}: {line[:120]}", file=sys.stderr)
                continue
            k = (c[1], c[2], c[3])
            r = rank.get(k, len(order))
            # a later file's row of the same case and reference replaces the earlier one
            key = (k, c[5])
            if key in seen:
                rows[seen[key]] = None
            seen[key] = len(rows)
            rows.append((r, c[5] == "scipy", len(rows), line))
    rows = sorted(x for x in rows if x is not None)
    print(HEADER)
    for r in rows:
        print(r[3])


if __name__ == "__main__":
    main()

"""mpmath's pslq as a co-process of the harness, so that its samples alternate with the crate's.

Protocol, one command a line on stdin, one answer a line on stdout:
  case <prec> <maxcoeff> <maxsteps>   then the numbers, one a line: "<mantissa> <exponent>"
                                      (x = mantissa·2^exponent), after "end"   -> "ok"
  check                               -> the relation, space-separated, or "None"
  run <k>                             -> runs pslq k times, prints the elapsed ns
  quit
"""
import sys
import time

import mpmath
from mpmath import mp, mpf


def main():
    x = []
    opts = (53, 1000, 100)
    out = sys.stdout
    lines = iter(sys.stdin.readline, "")
    for line in lines:
        w = line.split()
        if not w:
            continue
        if w[0] == "case":
            opts = (int(w[1]), int(w[2]), int(w[3]))
            mp.prec = opts[0]
            x = []
            for l2 in lines:
                v = l2.split()
                if v[0] == "end":
                    break
                x.append(mpf((int(v[0]), int(v[1]))))
            out.write("ok\n")
        elif w[0] == "check":
            mp.prec = opts[0]
            r = mpmath.pslq(x, maxcoeff=opts[1], maxsteps=opts[2])
            out.write(("None" if r is None else " ".join(str(c) for c in r)) + "\n")
        elif w[0] == "run":
            mp.prec = opts[0]
            k = int(w[1])
            t = time.perf_counter_ns()
            for _ in range(k):
                mpmath.pslq(x, maxcoeff=opts[1], maxsteps=opts[2])
            out.write("%d\n" % (time.perf_counter_ns() - t))
        elif w[0] == "nop":
            out.write("0\n")
        elif w[0] == "quit":
            break
        out.flush()


if __name__ == "__main__":
    out = sys.stdout
    out.write("mpmath %s backend %s\n" % (mpmath.__version__, mpmath.libmp.BACKEND))
    out.flush()
    main()

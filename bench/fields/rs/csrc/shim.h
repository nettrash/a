/* The interface of the C shims of the fields scoreboard: each reference library gets the same
 * small set of entry points per family, so that the Rust harness can drive it like its own code
 * and alternate it with ours sample by sample.
 *
 *  prime fields:  X_pf_new(p, n, xs, count, e, en) -> handle; X_pf_out(h, op, i, out) -> int;
 *                 X_pf_run(h, op, iters); X_pf_free(h)
 *  extensions:    X_ext_new(p, n, g, k, xs, count, e, en) ...
 *  binary fields: X_bin_new(exps, nexp, xs, count, e) ...
 *
 * Elements cross the interface as canonical little-endian 64-bit limbs (n per prime-field element,
 * k*n per extension element, ceil(m/64) per binary element). `count` is a power of two. `out`
 * computes one operation on input i (and i+1 for binary operations) and writes the canonical
 * result; it returns the Legendre symbol or a flag where the operation gives one, or UNSUPPORTED.
 * `run` is the timed loop, `iters` operations on inputs taken in turn (see harness.rs for the exact
 * loop of each operation, which every side follows). */
#ifndef FIELDBENCH_SHIM_H
#define FIELDBENCH_SHIM_H
#include <stddef.h>
#include <stdint.h>

enum {
    OP_MUL = 0, OP_SQR = 1, OP_ADD = 2, OP_SUB = 3, OP_INV = 4, OP_SQRT = 5, OP_LEG = 6,
    OP_POW = 7, OP_FROB = 8, OP_NORM = 9, OP_TRACE = 10, OP_ISSQ = 11, OP_DIV = 12,
    OP_HTRACE = 13, OP_QROOT = 14, OP_POWU = 15
};
#define UNSUPPORTED (-1000)

#endif

/* Stands in for a reference library that is not installed: every operation is unsupported. */
#include <stdint.h>
#include <stddef.h>
#define CAT2(a, b) a##b
#define CAT(a, b) CAT2(a, b)
void* CAT(P, _prep)(const char* op, const uint64_t* b, size_t n) { (void)op; (void)b; (void)n; return 0; }
void CAT(P, _run)(void* h, uint64_t r) { (void)h; (void)r; }
size_t CAT(P, _check)(void* h, uint64_t* o, size_t c) { (void)h; (void)o; (void)c; return 0; }
void CAT(P, _free)(void* h) { (void)h; }
const char* CAT(P, _label)(void* h) { (void)h; return ""; }

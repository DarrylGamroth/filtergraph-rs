/* Compile the pinned, unmodified OpenBLAS CBLAS interface with bounded stack
 * scratch. The numerical kernels and interface source remain upstream code,
 * including their copyright and license, in openblas-src's dependency tree.
 * Unique symbols avoid overriding/interposing other CBLAS implementations. */
#ifdef MAX_STACK_ALLOC
#undef MAX_STACK_ALLOC
#endif
#define MAX_STACK_ALLOC (64 * 1024)
#include "common.h"
#if defined(__x86_64__) && !defined(DOUBLE)
#include <xmmintrin.h>
#endif
#if defined(SMP) || defined(USE64BITINT) || defined(FUNCTION_PROFILE) || defined(DEBUG) || defined(ENABLE_SSE_EXCEPTION)
#error "FilterGraph requires single-thread LP64 GEMV without runtime diagnostics"
#endif
#if !defined(CBLAS) || !defined(DYNAMIC_ARCH) || defined(COMPLEX) || defined(XDOUBLE) || defined(BFLOAT16) || defined(HFLOAT16)
#error "FilterGraph requires real Float32/Float64 CPU-dispatched CBLAS"
#endif
#if BUILD_SINGLE != 1 || BUILD_DOUBLE != 1 || BUILD_COMPLEX != 1 || BUILD_COMPLEX16 != 1 || !defined(EXPRECISION)
#error "FilterGraph must match the pinned OpenBLAS dispatch-table layout"
#endif
_Static_assert(sizeof(FLOAT) == FILTERGRAPH_FLOAT_BYTES, "GEMV scalar ABI");
_Static_assert(sizeof(blasint) == 4, "LP64 CBLAS integer ABI");
#include "interface/gemv.c"

#ifndef DOUBLE
int filtergraph_sgemv_default_float_mode(void)
{
#if defined(__x86_64__)
    /* MXCSR round-to-nearest, gradual underflow and masked exceptions. */
    return (_mm_getcsr() & 0xffc0u) == 0x1f80u;
#else
    return 0;
#endif
}

/* The matching row-major, non-transposed branch of interface/gemv.c calls
 * this transposed kernel with exchanged dimensions. The Rust boundary
 * initializes y for beta=0 and supplies aligned, caller-owned scratch.
 * The pinned interface reserves m+n plus platform padding for this kernel. */
void filtergraph_sgemv_with_workspace(blasint rows, blasint columns,
                                     const float *a, const float *x,
                                     float *y, float *scratch)
{
    GEMV_T(columns, rows, 0, 1.0f, (float *)a, columns, (float *)x, 1,
           y, 1, scratch);
}
#endif

#ifndef SSE3_H
#define SSE3_H

#include <stdbool.h>
#include <stdint.h>

extern const bool SSE3;

#ifndef __m128
typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
#endif

#ifndef __m128d
typedef double __m128d __attribute((__vector_size__(16), __aligned__(16)));
#endif

#ifndef __m128i
typedef long long __m128i __attribute((__vector_size__(16), __aligned__(16)));
#endif

#ifdef __cplusplus
extern "C" {
#endif

__m128d sse3_addsub_pd(__m128d a, __m128d b);
__m128 sse3_addsub_ps(__m128 a, __m128 b);
__m128d sse3_hadd_pd(__m128d a, __m128d b);
__m128 sse3_hadd_ps(__m128 a, __m128 b);
__m128d sse3_hsub_pd(__m128d a, __m128d b);
__m128 sse3_hsub_ps(__m128 a, __m128 b);
__m128i sse3_lddqu_si128(const __m128i* mem_addr);
__m128d sse3_loaddup_pd(const double* mem_addr);
__m128d sse3_movedup_pd(__m128d a);
__m128 sse3_movehdup_ps(__m128 a);
__m128 sse3_moveldup_ps(__m128 a);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SSE3_H

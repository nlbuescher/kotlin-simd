#include <pmmintrin.h>

#include "sse3.h"

__m128d sse3_addsub_pd(__m128d a, __m128d b) { return _mm_addsub_pd(a, b); }
__m128 sse3_addsub_ps(__m128 a, __m128 b) { return _mm_addsub_ps(a, b); }
__m128d sse3_hadd_pd(__m128d a, __m128d b) { return _mm_hadd_pd(a, b); }
__m128 sse3_hadd_ps(__m128 a, __m128 b) { return _mm_hadd_ps(a, b); }
__m128d sse3_hsub_pd(__m128d a, __m128d b) { return _mm_hsub_pd(a, b); }
__m128 sse3_hsub_ps(__m128 a, __m128 b) { return _mm_hsub_ps(a, b); }
__m128i sse3_lddqu_si128(const __m128i* mem_addr) { return _mm_lddqu_si128(mem_addr); }
__m128d sse3_loaddup_pd(const double* mem_addr) { return _mm_loaddup_pd(mem_addr); }
__m128d sse3_movedup_pd(__m128d a) { return _mm_movedup_pd(a); }
__m128 sse3_movehdup_ps(__m128 a) { return _mm_movehdup_ps(a); }
__m128 sse3_moveldup_ps(__m128 a) { return _mm_moveldup_ps(a); }

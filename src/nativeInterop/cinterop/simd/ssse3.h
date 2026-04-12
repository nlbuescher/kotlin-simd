#ifndef SSSE3_H
#define SSSE3_H

#include <stdbool.h>
#include <stdint.h>

extern const bool SSSE3;

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

__m128i ssse3_abs_epi16(__m128i a);
__m128i ssse3_abs_epi32(__m128i a);
__m128i ssse3_abs_epi8(__m128i a);
__m128i ssse3_alignr_epi8(__m128i a, __m128i b, uint8_t imm8);
__m128i ssse3_hadd_epi16(__m128i a, __m128i b);
__m128i ssse3_hadd_epi32(__m128i a, __m128i b);
__m128i ssse3_hadds_epi16(__m128i a, __m128i b);
__m128i ssse3_hsub_epi16(__m128i a, __m128i b);
__m128i ssse3_hsub_epi32(__m128i a, __m128i b);
__m128i ssse3_hsubs_epi16(__m128i a, __m128i b);
__m128i ssse3_maddubs_epi16(__m128i a, __m128i b);
__m128i ssse3_mulhrs_epi16(__m128i a, __m128i b);
__m128i ssse3_shuffle_epi8(__m128i a, __m128i b);
__m128i ssse3_sign_epi16(__m128i a, __m128i b);
__m128i ssse3_sign_epi32(__m128i a, __m128i b);
__m128i ssse3_sign_epi8(__m128i a, __m128i b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SSSE3_H

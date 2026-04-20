#ifndef SSE4_1_H
#define SSE4_1_H

#include <stdbool.h>
#include <stdint.h>

extern const bool SSE4_1;

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

const int32_t SSE4_1_FROUND_TO_NEAREST_INT = 0x00;
const int32_t SSE4_1_FROUND_TO_NEG_INF     = 0x01;
const int32_t SSE4_1_FROUND_TO_POS_INF     = 0x02;
const int32_t SSE4_1_FROUND_TO_ZERO        = 0x03;
const int32_t SSE4_1_FROUND_CUR_DIRECTION  = 0x04;

const int32_t SSE4_1_FROUND_RAISE_EXC      = 0x00;
const int32_t SSE4_1_FROUND_NO_EXC         = 0x08;

const int32_t SSE4_1_FROUND_NINT           = (SSE4_1_FROUND_TO_NEAREST_INT | SSE4_1_FROUND_RAISE_EXC);
const int32_t SSE4_1_FROUND_FLOOR          = (SSE4_1_FROUND_TO_NEG_INF     | SSE4_1_FROUND_RAISE_EXC);
const int32_t SSE4_1_FROUND_CEIL           = (SSE4_1_FROUND_TO_POS_INF     | SSE4_1_FROUND_RAISE_EXC);
const int32_t SSE4_1_FROUND_TRUNC          = (SSE4_1_FROUND_TO_ZERO        | SSE4_1_FROUND_RAISE_EXC);
const int32_t SSE4_1_FROUND_RINT           = (SSE4_1_FROUND_CUR_DIRECTION  | SSE4_1_FROUND_RAISE_EXC);
const int32_t SSE4_1_FROUND_NEARBYINT      = (SSE4_1_FROUND_CUR_DIRECTION  | SSE4_1_FROUND_NO_EXC);

__m128i sse4_1_blend_epi16(__m128i a, __m128i b, uint8_t imm8);
__m128d sse4_1_blend_pd(__m128d a, __m128d b, uint8_t imm8);
__m128 sse4_1_blend_ps(__m128 a, __m128 b, uint8_t imm8);
__m128i sse4_1_blendv_epi8(__m128i a, __m128i b, __m128i mask);
__m128d sse4_1_blendv_pd(__m128d a, __m128d b, __m128d mask);
__m128 sse4_1_blendv_ps(__m128 a, __m128 b, __m128 mask);
__m128d sse4_1_ceil_pd(__m128d a);
__m128 sse4_1_ceil_ps(__m128 a);
__m128d sse4_1_ceil_sd(__m128d a, __m128d b);
__m128 sse4_1_ceil_ss(__m128 a, __m128 b);
__m128i sse4_1_cmpeq_epi64(__m128i a, __m128i b);
__m128i sse4_1_cvtepi16_epi32(__m128i a);
__m128i sse4_1_cvtepi16_epi64(__m128i a);
__m128i sse4_1_cvtepi32_epi64(__m128i a);
__m128i sse4_1_cvtepi8_epi16(__m128i a);
__m128i sse4_1_cvtepi8_epi32(__m128i a);
__m128i sse4_1_cvtepi8_epi64(__m128i a);
__m128i sse4_1_cvtepu16_epi32(__m128i a);
__m128i sse4_1_cvtepu16_epi64(__m128i a);
__m128i sse4_1_cvtepu32_epi64(__m128i a);
__m128i sse4_1_cvtepu8_epi16(__m128i a);
__m128i sse4_1_cvtepu8_epi32(__m128i a);
__m128i sse4_1_cvtepu8_epi64(__m128i a);
__m128d sse4_1_dp_pd(__m128d a, __m128d b, uint8_t imm8);
__m128 sse4_1_dp_ps(__m128 a, __m128 b, uint8_t imm8);
int32_t sse4_1_extract_epi32(__m128i a, uint8_t imm8);
int64_t sse4_1_extract_epi64(__m128i a, uint8_t imm8);
int8_t sse4_1_extract_epi8(__m128i a, uint8_t imm8);
float sse4_1_extract_ps(__m128 a, uint8_t imm8);
__m128d sse4_1_floor_pd(__m128d a);
__m128 sse4_1_floor_ps(__m128 a);
__m128d sse4_1_floor_sd(__m128d a, __m128d b);
__m128 sse4_1_floor_ss(__m128 a, __m128 b);
__m128i sse4_1_insert_epi32(__m128i a, int32_t i, uint8_t imm8);
__m128i sse4_1_insert_epi64(__m128i a, int64_t i, uint8_t imm8);
__m128i sse4_1_insert_epi8(__m128i a, int8_t i, uint8_t imm8);
__m128 sse4_1_insert_ps(__m128 a, __m128 b, uint8_t imm8);
__m128i sse4_1_max_epi32(__m128i a, __m128i b);
__m128i sse4_1_max_epi8(__m128i a, __m128i b);
__m128i sse4_1_max_epu16(__m128i a, __m128i b);
__m128i sse4_1_max_epu32(__m128i a, __m128i b);
__m128i sse4_1_min_epi32(__m128i a, __m128i b);
__m128i sse4_1_min_epi8(__m128i a, __m128i b);
__m128i sse4_1_min_epu16(__m128i a, __m128i b);
__m128i sse4_1_min_epu32(__m128i a, __m128i b);
__m128i sse4_1_minpos_epu16(__m128i a);
__m128i sse4_1_mpsadbw_epu8(__m128i a, __m128i b, uint8_t imm8);
__m128i sse4_1_mul_epi32(__m128i a, __m128i b);
__m128i sse4_1_mullo_epi32(__m128i a, __m128i b);
__m128i sse4_1_packus_epi32(__m128i a, __m128i b);
__m128d sse4_1_round_pd(__m128d a, int32_t rounding);
__m128 sse4_1_round_ps(__m128 a, int32_t rounding);
__m128d sse4_1_round_sd(__m128d a, __m128d b, int32_t rounding);
__m128 sse4_1_round_ss(__m128 a, __m128 b, int32_t rounding);
__m128i sse4_1_stream_load_si128(__m128i* mem_addr);
bool sse4_1_test_all_ones(__m128i a);
bool sse4_1_test_all_zeros(__m128i a, __m128i mask);
bool sse4_1_test_mix_ones_zeros(__m128i a, __m128i mask);
bool sse4_1_testc_si128(__m128i a, __m128i b);
bool sse4_1_testnzc_si128(__m128i a, __m128i b);
bool sse4_1_testz_si128(__m128i a, __m128i b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SSE4_1_H

#ifndef SIMD_H
#define SIMD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifndef __m128
typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));
#endif

#ifndef __m128d
typedef double __m128d __attribute__((__vector_size__(16), __aligned__(16)));
#endif

#ifndef __m128i
typedef long long __m128i __attribute__((__vector_size__(16), __aligned__(16)));
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern const bool
    SSE,
    SSE2,
    SSE3,
    SSSE3,
    SSE4_1,
    SSE4_2;


// SSE

extern const int32_t
    SSE_HINT_T0,
    SSE_HINT_T1,
    SSE_HINT_T2,
    SSE_HINT_NTA;

extern const uint32_t
    SSE_EXCEPT_MASK,
    SSE_EXCEPT_INVALID,
    SSE_EXCEPT_DENORM,
    SSE_EXCEPT_DIV_ZERO,
    SSE_EXCEPT_OVERFLOW,
    SSE_EXCEPT_UNDERFLOW,
    SSE_EXCEPT_INEXACT;

extern const uint32_t
    SSE_MASK_MASK,
    SSE_MASK_INVALID,
    SSE_MASK_DENORM,
    SSE_MASK_DIV_ZERO,
    SSE_MASK_OVERFLOW,
    SSE_MASK_UNDERFLOW,
    SSE_MASK_INEXACT;

extern const uint32_t
    SSE_ROUND_MASK,
    SSE_ROUND_NEAREST,
    SSE_ROUND_DOWN,
    SSE_ROUND_UP,
    SSE_ROUND_TOWARD_ZERO;

extern const uint32_t
    SSE_FLUSH_ZERO_MASK,
    SSE_FLUSH_ZERO_ON,
    SSE_FLUSH_ZERO_OFF;

__m128 sse_add_ps(__m128 a, __m128 b);
__m128 sse_add_ss(__m128 a, __m128 b);
__m128 sse_and_ps(__m128 a, __m128 b);
__m128 sse_andnot_ps(__m128 a, __m128 b);
__m128 sse_cmpeq_ps(__m128 a, __m128 b);
__m128 sse_cmpeq_ss(__m128 a, __m128 b);
__m128 sse_cmpge_ps(__m128 a, __m128 b);
__m128 sse_cmpge_ss(__m128 a, __m128 b);
__m128 sse_cmpgt_ps(__m128 a, __m128 b);
__m128 sse_cmpgt_ss(__m128 a, __m128 b);
__m128 sse_cmple_ps(__m128 a, __m128 b);
__m128 sse_cmple_ss(__m128 a, __m128 b);
__m128 sse_cmplt_ps(__m128 a, __m128 b);
__m128 sse_cmplt_ss(__m128 a, __m128 b);
__m128 sse_cmpneq_ps(__m128 a, __m128 b);
__m128 sse_cmpneq_ss(__m128 a, __m128 b);
__m128 sse_cmpnge_ps(__m128 a, __m128 b);
__m128 sse_cmpnge_ss(__m128 a, __m128 b);
__m128 sse_cmpngt_ps(__m128 a, __m128 b);
__m128 sse_cmpngt_ss(__m128 a, __m128 b);
__m128 sse_cmpnle_ps(__m128 a, __m128 b);
__m128 sse_cmpnle_ss(__m128 a, __m128 b);
__m128 sse_cmpnlt_ps(__m128 a, __m128 b);
__m128 sse_cmpnlt_ss(__m128 a, __m128 b);
__m128 sse_cmpord_ps(__m128 a, __m128 b);
__m128 sse_cmpord_ss(__m128 a, __m128 b);
__m128 sse_cmpunord_ps(__m128 a, __m128 b);
__m128 sse_cmpunord_ss(__m128 a, __m128 b);
bool sse_comieq_ss(__m128 a, __m128 b);
bool sse_comige_ss(__m128 a, __m128 b);
bool sse_comigt_ss(__m128 a, __m128 b);
bool sse_comile_ss(__m128 a, __m128 b);
bool sse_comilt_ss(__m128 a, __m128 b);
bool sse_comineq_ss(__m128 a, __m128 b);
__m128 sse_cvt_si2ss(__m128 a, int32_t b);
int32_t sse_cvt_ss2si(__m128 a);
__m128 sse_cvtsi32_ss(__m128 a, int32_t b);
__m128 sse_cvtsi64_ss(__m128 a, int64_t b);
float sse_cvtss_f32(__m128 a);
int32_t sse_cvtss_si32(__m128 a);
int64_t sse_cvtss_si64(__m128 a);
int32_t sse_cvtt_ss2si(__m128 a);
int32_t sse_cvttss_si32(__m128 a);
int64_t sse_cvttss_si64(__m128 a);
__m128 sse_div_ps(__m128 a, __m128 b);
__m128 sse_div_ss(__m128 a, __m128 b);
void sse_free(void* mem_addr);
uint32_t sse_get_exception_mask();
uint32_t sse_get_exception_state();
uint32_t sse_get_flush_zero_mode();
uint32_t sse_get_rounding_mode();
uint32_t sse_getcsr(void);
__m128 sse_load_ps(const float* mem_addr);
__m128 sse_load_ps1(const float* mem_addr);
__m128 sse_load_ss(const float* mem_addr);
__m128 sse_load1_ps(const float* mem_addr);
__m128 sse_loadr_ps(const float* mem_addr);
__m128 sse_loadu_ps(const float* mem_addr);
void* sse_malloc(size_t size, size_t align);
__m128 sse_max_ps(__m128 a, __m128 b);
__m128 sse_max_ss(__m128 a, __m128 b);
__m128 sse_min_ps(__m128 a, __m128 b);
__m128 sse_min_ss(__m128 a, __m128 b);
__m128 sse_move_ss(__m128 a, __m128 b);
__m128 sse_movehl_ps(__m128 a, __m128 b);
__m128 sse_movelh_ps(__m128 a, __m128 b);
int32_t sse_movemask_ps(__m128 a);
__m128 sse_mul_ps(__m128 a, __m128 b);
__m128 sse_mul_ss(__m128 a, __m128 b);
__m128 sse_or_ps(__m128 a, __m128 b);
void sse_prefetch(const int8_t* p, uint8_t i);
__m128 sse_rcp_ps(__m128 a);
__m128 sse_rcp_ss(__m128 a);
__m128 sse_rsqrt_ps(__m128 a);
__m128 sse_rsqrt_ss(__m128 a);
void sse_set_exception_mask(uint32_t a);
void sse_set_exception_state(uint32_t a);
void sse_set_flush_zero_mode(uint32_t a);
__m128 sse_set_ps(float e3, float e2, float e1, float e0);
__m128 sse_set_ps1(float a);
void sse_set_rounding_mode(uint32_t a);
__m128 sse_set_ss(float a);
__m128 sse_set1_ps(float a);
void sse_setcsr(uint32_t a);
__m128 sse_setr_ps(float e3, float e2, float e1, float e0);
__m128 sse_setzero_ps(void);
void sse_sfence(void);
__m128 sse_shuffle_ps(__m128 a, __m128 b, uint8_t imm8);
__m128 sse_sqrt_ps(__m128 a);
__m128 sse_sqrt_ss(__m128 a);
void sse_store_ps(float* mem_addr, __m128 a);
void sse_store_ps1(float* mem_addr, __m128 a);
void sse_store_ss(float* mem_addr, __m128 a);
void sse_store1_ps(float* mem_addr, __m128 a);
void sse_storer_ps(float* mem_addr, __m128 a);
void sse_storeu_ps(float* mem_addr, __m128 a);
void sse_stream_ps(float* mem_addr, __m128 a);
__m128 sse_sub_ps(__m128 a, __m128 b);
__m128 sse_sub_ss(__m128 a, __m128 b);
void sse_transpose4_ps(__m128* row0, __m128* row1, __m128* row2, __m128* row3);
bool sse_ucomieq_ss(__m128 a, __m128 b);
bool sse_ucomige_ss(__m128 a, __m128 b);
bool sse_ucomigt_ss(__m128 a, __m128 b);
bool sse_ucomile_ss(__m128 a, __m128 b);
bool sse_ucomilt_ss(__m128 a, __m128 b);
bool sse_ucomineq_ss(__m128 a, __m128 b);
__m128 sse_undefined_ps(void);
__m128 sse_unpackhi_ps(__m128 a, __m128 b);
__m128 sse_unpacklo_ps(__m128 a, __m128 b);
__m128 sse_xor_ps(__m128 a, __m128 b);


// SSE2

__m128i sse2_add_epi16(__m128i a, __m128i b);
__m128i sse2_add_epi32(__m128i a, __m128i b);
__m128i sse2_add_epi64(__m128i a, __m128i b);
__m128i sse2_add_epi8(__m128i a, __m128i b);
__m128d sse2_add_pd(__m128d a, __m128d b);
__m128d sse2_add_sd(__m128d a, __m128d b);
__m128i sse2_adds_epi16(__m128i a, __m128i b);
__m128i sse2_adds_epi8(__m128i a, __m128i b);
__m128i sse2_adds_epu16(__m128i a, __m128i b);
__m128i sse2_adds_epu8(__m128i a, __m128i b);
__m128d sse2_and_pd(__m128d a, __m128d b);
__m128i sse2_and_si128(__m128i a, __m128i b);
__m128d sse2_andnot_pd(__m128d a, __m128d b);
__m128i sse2_andnot_si128(__m128i a, __m128i b);
__m128i sse2_avg_epu16(__m128i a, __m128i b);
__m128i sse2_avg_epu8(__m128i a, __m128i b);
__m128i sse2_bslli_si128(__m128i a, uint8_t imm8);
__m128i sse2_bsrli_si128(__m128i a, uint8_t imm8);
__m128 sse2_castpd_ps(__m128d a);
__m128i sse2_castpd_si128(__m128d a);
__m128d sse2_castps_pd(__m128 a);
__m128i sse2_castps_si128(__m128 a);
__m128d sse2_castsi128_pd(__m128i a);
__m128 sse2_castsi128_ps(__m128i a);
void sse2_clflush(const void* p);
__m128i sse2_cmpeq_epi16(__m128i a, __m128i b);
__m128i sse2_cmpeq_epi32(__m128i a, __m128i b);
__m128i sse2_cmpeq_epi8(__m128i a, __m128i b);
__m128d sse2_cmpeq_pd(__m128d a, __m128d b);
__m128d sse2_cmpeq_sd(__m128d a, __m128d b);
__m128d sse2_cmpge_pd(__m128d a, __m128d b);
__m128d sse2_cmpge_sd(__m128d a, __m128d b);
__m128i sse2_cmpgt_epi16(__m128i a, __m128i b);
__m128i sse2_cmpgt_epi32(__m128i a, __m128i b);
__m128i sse2_cmpgt_epi8(__m128i a, __m128i b);
__m128d sse2_cmpgt_pd(__m128d a, __m128d b);
__m128d sse2_cmpgt_sd(__m128d a, __m128d b);
__m128d sse2_cmple_pd(__m128d a, __m128d b);
__m128d sse2_cmple_sd(__m128d a, __m128d b);
__m128i sse2_cmplt_epi16(__m128i a, __m128i b);
__m128i sse2_cmplt_epi32(__m128i a, __m128i b);
__m128i sse2_cmplt_epi8(__m128i a, __m128i b);
__m128d sse2_cmplt_pd(__m128d a, __m128d b);
__m128d sse2_cmplt_sd(__m128d a, __m128d b);
__m128d sse2_cmpneq_pd(__m128d a, __m128d b);
__m128d sse2_cmpneq_sd(__m128d a, __m128d b);
__m128d sse2_cmpnge_pd(__m128d a, __m128d b);
__m128d sse2_cmpnge_sd(__m128d a, __m128d b);
__m128d sse2_cmpngt_pd(__m128d a, __m128d b);
__m128d sse2_cmpngt_sd(__m128d a, __m128d b);
__m128d sse2_cmpnle_pd(__m128d a, __m128d b);
__m128d sse2_cmpnle_sd(__m128d a, __m128d b);
__m128d sse2_cmpnlt_pd(__m128d a, __m128d b);
__m128d sse2_cmpnlt_sd(__m128d a, __m128d b);
__m128d sse2_cmpord_pd(__m128d a, __m128d b);
__m128d sse2_cmpord_sd(__m128d a, __m128d b);
__m128d sse2_cmpunord_pd(__m128d a, __m128d b);
__m128d sse2_cmpunord_sd(__m128d a, __m128d b);
bool sse2_comieq_sd(__m128d a, __m128d b);
bool sse2_comige_sd(__m128d a, __m128d b);
bool sse2_comigt_sd(__m128d a, __m128d b);
bool sse2_comile_sd(__m128d a, __m128d b);
bool sse2_comilt_sd(__m128d a, __m128d b);
bool sse2_comineq_sd(__m128d a, __m128d b);
__m128d sse2_cvtepi32_pd(__m128i a);
__m128 sse2_cvtepi32_ps(__m128i a);
__m128i sse2_cvtpd_epi32(__m128d a);
__m128 sse2_cvtpd_ps(__m128d a);
__m128i sse2_cvtps_epi32(__m128 a);
__m128d sse2_cvtps_pd(__m128 a);
double sse2_cvtsd_f64(__m128d a);
int32_t sse2_cvtsd_si32(__m128d a);
int64_t sse2_cvtsd_si64(__m128d a);
__m128 sse2_cvtsd_ss(__m128 a, __m128d b);
int32_t sse2_cvtsi128_si32(__m128i a);
int64_t sse2_cvtsi128_si64(__m128i a);
__m128d sse2_cvtsi32_sd(__m128d a, int32_t b);
__m128i sse2_cvtsi32_si128(int32_t a);
__m128d sse2_cvtsi64_sd(__m128d a, int64_t b);
__m128i sse2_cvtsi64_si128(int64_t a);
__m128d sse2_cvtss_sd(__m128d a, __m128 b);
__m128i sse2_cvttpd_epi32(__m128d a);
__m128i sse2_cvttps_epi32(__m128 a);
int32_t sse2_cvttsd_si32(__m128d a);
int64_t sse2_cvttsd_si64(__m128d a);
__m128d sse2_div_pd(__m128d a, __m128d b);
__m128d sse2_div_sd(__m128d a, __m128d b);
int16_t sse2_extract_epi16(__m128i a, uint8_t imm8);
__m128i sse2_insert_epi16(__m128i a, int16_t i, uint8_t imm8);
void sse2_lfence(void);
__m128d sse2_load_pd(const double* mem_addr);
__m128d sse2_load_pd1(const double* mem_addr);
__m128d sse2_load_sd(const double* mem_addr);
__m128i sse2_load_si128(const __m128i* mem_addr);
__m128d sse2_load1_pd(const double* mem_addr);
__m128d sse2_loadh_pd(__m128d a, const double* mem_addr);
__m128i sse2_loadl_epi64(const __m128i* mem_addr);
__m128d sse2_loadl_pd(__m128d a, const double* mem_addr);
__m128d sse2_loadr_pd(const double* mem_addr);
__m128d sse2_loadu_pd(const double* mem_addr);
__m128i sse2_loadu_si128(const __m128i* mem_addr);
__m128i sse2_madd_epi16(__m128i a, __m128i b);
void sse2_maskmoveu_si128(__m128i a, __m128i mask, int8_t* mem_addr);
__m128i sse2_max_epi16(__m128i a, __m128i b);
__m128i sse2_max_epu8(__m128i a, __m128i b);
__m128d sse2_max_pd(__m128d a, __m128d b);
__m128d sse2_max_sd(__m128d a, __m128d b);
void sse2_mfence(void);
__m128i sse2_min_epi16(__m128i a, __m128i b);
__m128i sse2_min_epu8(__m128i a, __m128i b);
__m128d sse2_min_pd(__m128d a, __m128d b);
__m128d sse2_min_sd(__m128d a, __m128d b);
__m128i sse2_move_epi64(__m128i a);
__m128d sse2_move_sd(__m128d a, __m128d b);
int32_t sse2_movemask_epi8(__m128i a);
int32_t sse2_movemask_pd(__m128d a);
__m128i sse2_mul_epu32(__m128i a, __m128i b);
__m128d sse2_mul_pd(__m128d a, __m128d b);
__m128d sse2_mul_sd(__m128d a, __m128d b);
__m128i sse2_mulhi_epi16(__m128i a, __m128i b);
__m128i sse2_mulhi_epu16(__m128i a, __m128i b);
__m128i sse2_mullo_epi16(__m128i a, __m128i b);
__m128d sse2_or_pd(__m128d a, __m128d b);
__m128i sse2_or_si128(__m128i a, __m128i b);
__m128i sse2_packs_epi16(__m128i a, __m128i b);
__m128i sse2_packs_epi32(__m128i a, __m128i b);
__m128i sse2_packus_epi16(__m128i a, __m128i b);
void sse2_pause(void);
__m128i sse2_sad_epu8(__m128i a, __m128i b);
__m128i sse2_set_epi16(int16_t e7, int16_t e6, int16_t e5, int16_t e4, int16_t e3, int16_t e2, int16_t e1, int16_t e0);
__m128i sse2_set_epi32(int32_t e3, int32_t e2, int32_t e1, int32_t e0);
__m128i sse2_set_epi64(int64_t e1, int64_t e0);
__m128i sse2_set_epi8(
    int8_t e15, int8_t e14, int8_t e13, int8_t e12, int8_t e11, int8_t e10, int8_t e9, int8_t e8,
    int8_t e7, int8_t e6, int8_t e5, int8_t e4, int8_t e3, int8_t e2, int8_t e1, int8_t e0
);
__m128d sse2_set_pd(double e1, double e0);
__m128d sse2_set_pd1(double a);
__m128d sse2_set_sd(double a);
__m128i sse2_set1_epi16(int16_t a);
__m128i sse2_set1_epi32(int32_t a);
__m128i sse2_set1_epi64(int64_t a);
__m128i sse2_set1_epi8(int8_t a);
__m128d sse2_set1_pd(double a);
__m128i sse2_setr_epi16(int16_t e7, int16_t e6, int16_t e5, int16_t e4, int16_t e3, int16_t e2, int16_t e1, int16_t e0);
__m128i sse2_setr_epi32(int32_t e3, int32_t e2, int32_t e1, int32_t e0);
__m128i sse2_setr_epi8(
    int8_t e15, int8_t e14, int8_t e13, int8_t e12, int8_t e11, int8_t e10, int8_t e9, int8_t e8,
    int8_t e7, int8_t e6, int8_t e5, int8_t e4, int8_t e3, int8_t e2, int8_t e1, int8_t e0
);
__m128d sse2_setr_pd(double e1, double e0);
__m128d sse2_setzero_pd(void);
__m128i sse2_setzero_si128();
__m128i sse2_shuffle_epi32(__m128i a, uint8_t imm8);
__m128d sse2_shuffle_pd(__m128d a, __m128d b, uint8_t imm8);
__m128i sse2_shufflehi_epi16(__m128i a, uint8_t imm8);
__m128i sse2_shufflelo_epi16(__m128i a, uint8_t imm8);
__m128i sse2_sll_epi16(__m128i a, __m128i count);
__m128i sse2_sll_epi32(__m128i a, __m128i count);
__m128i sse2_sll_epi64(__m128i a, __m128i count);
__m128i sse2_slli_epi16(__m128i a, uint8_t imm8);
__m128i sse2_slli_epi32(__m128i a, uint8_t imm8);
__m128i sse2_slli_epi64(__m128i a, uint8_t imm8);
__m128i sse2_slli_si128(__m128i a, uint8_t imm8);
__m128d sse2_sqrt_pd(__m128d a);
__m128d sse2_sqrt_sd(__m128d a, __m128d b);
__m128i sse2_sra_epi16(__m128i a, __m128i count);
__m128i sse2_sra_epi32(__m128i a, __m128i count);
__m128i sse2_srai_epi16(__m128i a, uint8_t imm8);
__m128i sse2_srai_epi32(__m128i a, uint8_t imm8);
__m128i sse2_srl_epi16(__m128i a, __m128i count);
__m128i sse2_srl_epi32(__m128i a, __m128i count);
__m128i sse2_srl_epi64(__m128i a, __m128i count);
__m128i sse2_srli_epi16(__m128i a, uint8_t imm8);
__m128i sse2_srli_epi32(__m128i a, uint8_t imm8);
__m128i sse2_srli_epi64(__m128i a, uint8_t imm8);
__m128i sse2_srli_si128(__m128i a, uint8_t imm8);
void sse2_store_pd(double* mem_addr, __m128d a);
void sse2_store_pd1(double* mem_addr, __m128d a);
void sse2_store_sd(double* mem_addr, __m128d a);
void sse2_store_si128(__m128i* mem_addr, __m128i a);
void sse2_store1_pd(double* mem_addr, __m128d a);
void sse2_storeh_pd(double* mem_addr, __m128d a);
void sse2_storel_epi64(__m128i* mem_addr, __m128i a);
void sse2_storel_pd(double* mem_addr, __m128d a);
void sse2_storer_pd(double* mem_addr, __m128d a);
void sse2_storeu_pd(double* mem_addr, __m128d a);
void sse2_storeu_si128(__m128i* mem_addr, __m128i a);
void sse2_stream_pd(double* mem_addr, __m128d a);
void sse2_stream_si128(__m128i* mem_addr, __m128i a);
void sse2_stream_si32(int32_t* mem_addr, int32_t a);
void sse2_stream_si64(int64_t* mem_addr, int64_t a);
__m128i sse2_sub_epi16(__m128i a, __m128i b);
__m128i sse2_sub_epi32(__m128i a, __m128i b);
__m128i sse2_sub_epi64(__m128i a, __m128i b);
__m128i sse2_sub_epi8(__m128i a, __m128i b);
__m128d sse2_sub_pd(__m128d a, __m128d b);
__m128d sse2_sub_sd(__m128d a, __m128d b);
__m128i sse2_subs_epi16(__m128i a, __m128i b);
__m128i sse2_subs_epi8(__m128i a, __m128i b);
__m128i sse2_subs_epu16(__m128i a, __m128i b);
__m128i sse2_subs_epu8(__m128i a, __m128i b);
bool sse2_ucomieq_sd(__m128d a, __m128d b);
bool sse2_ucomige_sd(__m128d a, __m128d b);
bool sse2_ucomigt_sd(__m128d a, __m128d b);
bool sse2_ucomile_sd(__m128d a, __m128d b);
bool sse2_ucomilt_sd(__m128d a, __m128d b);
bool sse2_ucomineq_sd(__m128d a, __m128d b);
__m128d sse2_undefined_pd(void);
__m128i sse2_undefined_si128(void);
__m128i sse2_unpackhi_epi16(__m128i a, __m128i b);
__m128i sse2_unpackhi_epi32(__m128i a, __m128i b);
__m128i sse2_unpackhi_epi64(__m128i a, __m128i b);
__m128i sse2_unpackhi_epi8(__m128i a, __m128i b);
__m128d sse2_unpackhi_pd(__m128d a, __m128d b);
__m128i sse2_unpacklo_epi16(__m128i a, __m128i b);
__m128i sse2_unpacklo_epi32(__m128i a, __m128i b);
__m128i sse2_unpacklo_epi64(__m128i a, __m128i b);
__m128i sse2_unpacklo_epi8(__m128i a, __m128i b);
__m128d sse2_unpacklo_pd(__m128d a, __m128d b);
__m128d sse2_xor_pd(__m128d a, __m128d b);
__m128i sse2_xor_si128(__m128i a, __m128i b);


// SSE3

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


// SSSE3

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


// SSE4.1

extern const int32_t
    SSE4_1_FROUND_TO_NEAREST_INT,
    SSE4_1_FROUND_TO_NEG_INF,
    SSE4_1_FROUND_TO_POS_INF,
    SSE4_1_FROUND_TO_ZERO,
    SSE4_1_FROUND_CUR_DIRECTION;

extern const int32_t
    SSE4_1_FROUND_RAISE_EXC,
    SSE4_1_FROUND_NO_EXC;

extern const int32_t
    SSE4_1_FROUND_NINT,
    SSE4_1_FROUND_FLOOR,
    SSE4_1_FROUND_CEIL,
    SSE4_1_FROUND_TRUNC,
    SSE4_1_FROUND_RINT,
    SSE4_1_FROUND_NEARBYINT;

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


// SSE4.2

// These macros specify the source data format
extern const int32_t
    SIDD_UBYTE_OPS,
    SIDD_UWORD_OPS,
    SIDD_SBYTE_OPS,
    SIDD_SWORD_OPS;

// These macros specify the comparison operation
extern const int32_t
    SIDD_CMP_EQUAL_ANY,
    SIDD_CMP_RANGES,
    SIDD_CMP_EQUAL_EACH,
    SIDD_CMP_EQUAL_ORDERED;

// These macros specify the polarity
extern const int32_t
    SIDD_POSITIVE_POLARITY,
    SIDD_NEGATIVE_POLARITY,
    SIDD_MASKED_POSITIVE_POLARITY,
    SIDD_MASKED_NEGATIVE_POLARITY;

// These macros specify the output selection in _mm_cmpXstri()
extern const int32_t
    SIDD_LEAST_SIGNIFICANT,
    SIDD_MOST_SIGNIFICANT;

// These macros specify the output selection in _mm_cmpXstrm()
extern const int32_t
    SIDD_BIT_MASK,
    SIDD_UNIT_MASK;

bool sse4_2_cmpestra(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
bool sse4_2_cmpestrc(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
int32_t sse4_2_cmpestri(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
__m128i sse4_2_cmpestrm(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
bool sse4_2_cmpestro(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
bool sse4_2_cmpestrs(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
bool sse4_2_cmpestrz(__m128i a, int32_t la, __m128i b, int32_t lb, uint8_t imm8);
__m128i sse4_2_cmpgt_epi64(__m128i a, __m128i b);
bool sse4_2_cmpistra(__m128i a, __m128i b, uint8_t imm8);
bool sse4_2_cmpistrc(__m128i a, __m128i b, uint8_t imm8);
int32_t sse4_2_cmpistri(__m128i a, __m128i b, uint8_t imm8);
__m128i sse4_2_cmpistrm(__m128i a, __m128i b, uint8_t imm8);
bool sse4_2_cmpistro(__m128i a, __m128i b, uint8_t imm8);
bool sse4_2_cmpistrs(__m128i a, __m128i b, uint8_t imm8);
bool sse4_2_cmpistrz(__m128i a, __m128i b, uint8_t imm8);
uint32_t sse4_2_crc32_u16(uint32_t crc, uint16_t v);
uint32_t sse4_2_crc32_u32(uint32_t crc, uint32_t v);
uint64_t sse4_2_crc32_u64(uint64_t crc, uint64_t v);
uint32_t sse4_2_crc32_u8(uint32_t crc, uint8_t v);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SIMD_H

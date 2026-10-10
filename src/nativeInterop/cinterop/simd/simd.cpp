#include "simd.h"

#include <cpuid.h>
#include <immintrin.h>

namespace {
    struct cpu_features_t {
        bool sse;
        bool sse2;
        bool sse3;
        bool ssse3;
        bool sse4_1;
        bool sse4_2;
    };

    cpu_features_t detect_cpu_features() {
        int info[4];

        __cpuid_count(1, 0, info[0], info[1], info[2], info[3]);

        return {
            .sse = (info[3] & bit_SSE) != 0,
            .sse2 = (info[3] & bit_SSE2) != 0,
            .sse3 = (info[2] & bit_SSE3) != 0,
            .ssse3 = (info[2] & bit_SSSE3) != 0,
            .sse4_1 = (info[2] & bit_SSE4_1) != 0,
            .sse4_2 = (info[2] & bit_SSE4_2) != 0,
        };
    }

    const auto cpu_features = detect_cpu_features();
} // namespace


extern "C" const bool
    SSE = cpu_features.sse,
    SSE2 = cpu_features.sse2,
    SSE3 = cpu_features.sse3,
    SSSE3 = cpu_features.ssse3,
    SSE4_1 = cpu_features.sse4_1,
    SSE4_2 = cpu_features.sse4_2;


//region === UTILS ===

#define TARGET(x) [[gnu::target(x)]]

#define unreachable() __builtin_unreachable()

#define bit_cast(T, x) __builtin_bit_cast(T, x)

#define CASE(N, FN, ...) case N: return FN(__VA_ARGS__, N);

#define CASES_2(FN, ...) \
    CASE(0, FN, __VA_ARGS__); \
    CASE(1, FN, __VA_ARGS__)

#define CASES_2_CONT(S, FN, ...) \
    CASE(S, FN, __VA_ARGS__); \
    CASE(S + 1, FN, __VA_ARGS__)

#define CASES_4(FN, ...) \
    CASES_2(FN, __VA_ARGS__); \
    CASES_2_CONT(2, FN, __VA_ARGS__)

#define CASES_4_CONT(S, FN, ...) \
    CASES_2_CONT(S, FN, __VA_ARGS__); \
    CASES_2_CONT(S + 2, FN, __VA_ARGS__)

#define CASES_8(FN, ...) \
    CASES_4(FN, __VA_ARGS__); \
    CASES_4_CONT(4, FN, __VA_ARGS__)

#define CASES_8_CONT(S, FN, ...) \
    CASES_4_CONT(S, FN, __VA_ARGS__); \
    CASES_4_CONT(S + 4, FN, __VA_ARGS__)

#define CASES_16(FN, ...) \
    CASES_8(FN, __VA_ARGS__); \
    CASES_8_CONT(8, FN, __VA_ARGS__)

#define CASES_16_CONT(S, FN, ...) \
    CASES_8_CONT(S, FN, __VA_ARGS__); \
    CASES_8_CONT(S + 8, FN, __VA_ARGS__)

#define CASES_17(FN, ...) \
    CASES_16(FN, __VA_ARGS__); \
    CASE(16, FN, __VA_ARGS__)

#define CASES_32(FN, ...) \
    CASES_16(FN, __VA_ARGS__); \
    CASES_16_CONT(16, FN, __VA_ARGS__)

#define CASES_32_CONT(S, FN, ...) \
    CASES_16_CONT(S, FN, __VA_ARGS__); \
    CASES_16_CONT(S + 16, FN, __VA_ARGS__)

#define CASES_33(FN, ...) \
    CASES_32(FN, __VA_ARGS__); \
    CASE(32, FN, __VA_ARGS__)

#define CASES_64(FN, ...) \
    CASES_32(FN, __VA_ARGS__); \
    CASES_32_CONT(32, FN, __VA_ARGS__)

#define CASES_64_CONT(S, FN, ...) \
    CASES_32_CONT(S, FN, __VA_ARGS__); \
    CASES_32_CONT(S + 32, FN, __VA_ARGS__)

#define CASES_128(FN, ...) \
    CASES_64(FN, __VA_ARGS__); \
    CASES_64_CONT(64, FN, __VA_ARGS__)

#define CASES_128_CONT(S, FN, ...) \
    CASES_64_CONT(S, FN, __VA_ARGS__); \
    CASES_64_CONT(S + 64, FN, __VA_ARGS__)

#define CASES_256(FN, ...) \
    CASES_128(FN, __VA_ARGS__); \
    CASES_128_CONT(128, FN, __VA_ARGS__)

#define DISPATCH(N, FN, I, ...) \
    switch(I) { \
        CASES_##N(FN, __VA_ARGS__); \
        default: unreachable(); \
    }

//endregion === UTILS ===

//region === SSE ===

const int32_t
    SSE_HINT_T0 = _MM_HINT_T0,
    SSE_HINT_T1 = _MM_HINT_T1,
    SSE_HINT_T2 = _MM_HINT_T2,
    SSE_HINT_NTA = _MM_HINT_NTA;

const uint32_t
    SSE_EXCEPT_MASK = _MM_EXCEPT_MASK,
    SSE_EXCEPT_INVALID = _MM_EXCEPT_INVALID,
    SSE_EXCEPT_DENORM = _MM_EXCEPT_DENORM,
    SSE_EXCEPT_DIV_ZERO = _MM_EXCEPT_DIV_ZERO,
    SSE_EXCEPT_OVERFLOW = _MM_EXCEPT_OVERFLOW,
    SSE_EXCEPT_UNDERFLOW = _MM_EXCEPT_UNDERFLOW,
    SSE_EXCEPT_INEXACT = _MM_EXCEPT_INEXACT;

const uint32_t
    SSE_MASK_MASK = _MM_MASK_MASK,
    SSE_MASK_INVALID = _MM_MASK_INVALID,
    SSE_MASK_DENORM = _MM_MASK_DENORM,
    SSE_MASK_DIV_ZERO = _MM_MASK_DIV_ZERO,
    SSE_MASK_OVERFLOW = _MM_MASK_OVERFLOW,
    SSE_MASK_UNDERFLOW = _MM_MASK_UNDERFLOW,
    SSE_MASK_INEXACT = _MM_MASK_INEXACT;

const uint32_t
    SSE_ROUND_MASK = _MM_ROUND_MASK,
    SSE_ROUND_NEAREST = _MM_ROUND_NEAREST,
    SSE_ROUND_DOWN = _MM_ROUND_DOWN,
    SSE_ROUND_UP = _MM_ROUND_UP,
    SSE_ROUND_TOWARD_ZERO = _MM_ROUND_TOWARD_ZERO;

const uint32_t
    SSE_FLUSH_ZERO_MASK = _MM_FLUSH_ZERO_MASK,
    SSE_FLUSH_ZERO_ON = _MM_FLUSH_ZERO_ON,
    SSE_FLUSH_ZERO_OFF = _MM_FLUSH_ZERO_OFF;


TARGET("sse")
__m128 sse_add_ps(const __m128 a, const __m128 b) {
    return _mm_add_ps(a, b);
}

TARGET("sse")
__m128 sse_add_ss(const __m128 a, const __m128 b) {
    return _mm_add_ss(a, b);
}

TARGET("sse")
__m128 sse_and_ps(const __m128 a, const __m128 b) {
    return _mm_and_ps(a, b);
}

TARGET("sse")
__m128 sse_andnot_ps(const __m128 a, const __m128 b) {
    return _mm_andnot_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpeq_ps(const __m128 a, const __m128 b) {
    return _mm_cmpeq_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpeq_ss(const __m128 a, const __m128 b) {
    return _mm_cmpeq_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpge_ps(const __m128 a, const __m128 b) {
    return _mm_cmpge_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpge_ss(const __m128 a, const __m128 b) {
    return _mm_cmpge_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpgt_ps(const __m128 a, const __m128 b) {
    return _mm_cmpgt_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpgt_ss(const __m128 a, const __m128 b) {
    return _mm_cmpgt_ss(a, b);
}

TARGET("sse")
__m128 sse_cmple_ps(const __m128 a, const __m128 b) {
    return _mm_cmple_ps(a, b);
}

TARGET("sse")
__m128 sse_cmple_ss(const __m128 a, const __m128 b) {
    return _mm_cmple_ss(a, b);
}

TARGET("sse")
__m128 sse_cmplt_ps(const __m128 a, const __m128 b) {
    return _mm_cmplt_ps(a, b);
}

TARGET("sse")
__m128 sse_cmplt_ss(const __m128 a, const __m128 b) {
    return _mm_cmplt_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpneq_ps(const __m128 a, const __m128 b) {
    return _mm_cmpneq_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpneq_ss(const __m128 a, const __m128 b) {
    return _mm_cmpneq_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpnge_ps(const __m128 a, const __m128 b) {
    return _mm_cmpnge_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpnge_ss(const __m128 a, const __m128 b) {
    return _mm_cmpnge_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpngt_ps(const __m128 a, const __m128 b) {
    return _mm_cmpngt_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpngt_ss(const __m128 a, const __m128 b) {
    return _mm_cmpngt_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpnle_ps(const __m128 a, const __m128 b) {
    return _mm_cmpnle_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpnle_ss(const __m128 a, const __m128 b) {
    return _mm_cmpnle_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpnlt_ps(const __m128 a, const __m128 b) {
    return _mm_cmpnlt_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpnlt_ss(const __m128 a, const __m128 b) {
    return _mm_cmpnlt_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpord_ps(const __m128 a, const __m128 b) {
    return _mm_cmpord_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpord_ss(const __m128 a, const __m128 b) {
    return _mm_cmpord_ss(a, b);
}

TARGET("sse")
__m128 sse_cmpunord_ps(const __m128 a, const __m128 b) {
    return _mm_cmpunord_ps(a, b);
}

TARGET("sse")
__m128 sse_cmpunord_ss(const __m128 a, const __m128 b) {
    return _mm_cmpunord_ss(a, b);
}

TARGET("sse")
bool sse_comieq_ss(const __m128 a, const __m128 b) {
    return _mm_comieq_ss(a, b);
}

TARGET("sse")
bool sse_comige_ss(const __m128 a, const __m128 b) {
    return _mm_comige_ss(a, b);
}

TARGET("sse")
bool sse_comigt_ss(const __m128 a, const __m128 b) {
    return _mm_comigt_ss(a, b);
}

TARGET("sse")
bool sse_comile_ss(const __m128 a, const __m128 b) {
    return _mm_comile_ss(a, b);
}

TARGET("sse")
bool sse_comilt_ss(const __m128 a, const __m128 b) {
    return _mm_comilt_ss(a, b);
}

TARGET("sse")
bool sse_comineq_ss(const __m128 a, const __m128 b) {
    return _mm_comineq_ss(a, b);
}

TARGET("sse")
__m128 sse_cvt_si2ss(const __m128 a, const int32_t b) {
    return _mm_cvt_si2ss(a, b);
}

TARGET("sse")
int32_t sse_cvt_ss2si(const __m128 a) {
    return _mm_cvt_ss2si(a);
}

TARGET("sse")
__m128 sse_cvtsi32_ss(const __m128 a, const int32_t b) {
    return _mm_cvtsi32_ss(a, b);
}

TARGET("sse")
__m128 sse_cvtsi64_ss(const __m128 a, const int64_t b) {
    return _mm_cvtsi64_ss(a, b);
}

TARGET("sse")
float sse_cvtss_f32(const __m128 a) {
    return _mm_cvtss_f32(a);
}

TARGET("sse")
int32_t sse_cvtss_si32(const __m128 a) {
    return _mm_cvtss_si32(a);
}

TARGET("sse")
int64_t sse_cvtss_si64(const __m128 a) {
    return _mm_cvtss_si64(a);
}

TARGET("sse")
int32_t sse_cvtt_ss2si(const __m128 a) {
    return _mm_cvtt_ss2si(a);
}

TARGET("sse")
int32_t sse_cvttss_si32(const __m128 a) {
    return _mm_cvttss_si32(a);
}

TARGET("sse")
int64_t sse_cvttss_si64(const __m128 a) {
    return _mm_cvttss_si64(a);
}

TARGET("sse")
__m128 sse_div_ps(const __m128 a, const __m128 b) {
    return _mm_div_ps(a, b);
}

TARGET("sse")
__m128 sse_div_ss(const __m128 a, const __m128 b) {
    return _mm_div_ss(a, b);
}

TARGET("sse")
void sse_free(void* mem_addr) {
    _mm_free(mem_addr);
}

TARGET("sse")
uint32_t sse_get_exception_mask() {
    return _MM_GET_EXCEPTION_MASK();
}

TARGET("sse")
uint32_t sse_get_exception_state() {
    return _MM_GET_EXCEPTION_STATE();
}

TARGET("sse")
uint32_t sse_get_flush_zero_mode() {
    return _MM_GET_FLUSH_ZERO_MODE();
}

TARGET("sse")
uint32_t sse_get_rounding_mode() {
    return _MM_GET_ROUNDING_MODE();
}

TARGET("sse")
uint32_t sse_getcsr() {
    return _mm_getcsr();
}

TARGET("sse")
__m128 sse_load_ps(const float* mem_addr) {
    return _mm_load_ps(mem_addr);
}

TARGET("sse")
__m128 sse_load_ps1(const float* mem_addr) {
    return _mm_load_ps1(mem_addr);
}

TARGET("sse")
__m128 sse_load_ss(const float* mem_addr) {
    return _mm_load_ss(mem_addr);
}

TARGET("sse")
__m128 sse_load1_ps(const float* mem_addr) {
    return _mm_load1_ps(mem_addr);
}

TARGET("sse")
__m128 sse_loadr_ps(const float* mem_addr) {
    return _mm_loadr_ps(mem_addr);
}

TARGET("sse")
__m128 sse_loadu_ps(const float* mem_addr) {
    return _mm_loadu_ps(mem_addr);
}

TARGET("sse")
void* sse_malloc(const size_t size, const size_t align) {
    return _mm_malloc(size, align);
}

TARGET("sse")
__m128 sse_max_ps(const __m128 a, const __m128 b) {
    return _mm_max_ps(a, b);
}

TARGET("sse")
__m128 sse_max_ss(const __m128 a, const __m128 b) {
    return _mm_max_ss(a, b);
}

TARGET("sse")
__m128 sse_min_ps(const __m128 a, const __m128 b) {
    return _mm_min_ps(a, b);
}

TARGET("sse")
__m128 sse_min_ss(const __m128 a, const __m128 b) {
    return _mm_min_ss(a, b);
}

TARGET("sse")
__m128 sse_move_ss(const __m128 a, const __m128 b) {
    return _mm_move_ss(a, b);
}

TARGET("sse")
__m128 sse_movehl_ps(const __m128 a, const __m128 b) {
    return _mm_movehl_ps(a, b);
}

TARGET("sse")
__m128 sse_movelh_ps(const __m128 a, const __m128 b) {
    return _mm_movelh_ps(a, b);
}

TARGET("sse")
int32_t sse_movemask_ps(const __m128 a) {
    return _mm_movemask_ps(a);
}

TARGET("sse")
__m128 sse_mul_ps(const __m128 a, const __m128 b) {
    return _mm_mul_ps(a, b);
}

TARGET("sse")
__m128 sse_mul_ss(const __m128 a, const __m128 b) {
    return _mm_mul_ss(a, b);
}

TARGET("sse")
__m128 sse_or_ps(const __m128 a, const __m128 b) {
    return _mm_or_ps(a, b);
}

TARGET("sse")
void sse_prefetch(const int8_t* p, const uint8_t i) {
    DISPATCH(4, _mm_prefetch, i & 0x03, p);
}

TARGET("sse")
__m128 sse_rcp_ps(const __m128 a) {
    return _mm_rcp_ps(a);
}

TARGET("sse")
__m128 sse_rcp_ss(const __m128 a) {
    return _mm_rcp_ss(a);
}

TARGET("sse")
__m128 sse_rsqrt_ps(const __m128 a) {
    return _mm_rsqrt_ps(a);
}

TARGET("sse")
__m128 sse_rsqrt_ss(const __m128 a) {
    return _mm_rsqrt_ss(a);
}

TARGET("sse")
void sse_set_exception_mask(const uint32_t a) {
    _MM_SET_EXCEPTION_MASK(a);
}

TARGET("sse")
void sse_set_exception_state(const uint32_t a) {
    _MM_SET_EXCEPTION_STATE(a);
}

TARGET("sse")
void sse_set_flush_zero_mode(const uint32_t a) {
    _MM_SET_FLUSH_ZERO_MODE(a);
}

TARGET("sse")
__m128 sse_set_ps(const float e3, const float e2, const float e1, const float e0) {
    return _mm_set_ps(e3, e2, e1, e0);
}

TARGET("sse")
__m128 sse_set_ps1(const float a) {
    return _mm_set_ps1(a);
}

TARGET("sse")
void sse_set_rounding_mode(const uint32_t a) {
    _MM_SET_ROUNDING_MODE(a);
}

TARGET("sse")
__m128 sse_set_ss(const float a) {
    return _mm_set_ss(a);
}

TARGET("sse")
__m128 sse_set1_ps(const float a) {
    return _mm_set1_ps(a);
}

TARGET("sse")
void sse_setcsr(const uint32_t a) {
    _mm_setcsr(a);
}

TARGET("sse")
__m128 sse_setr_ps(const float e3, const float e2, const float e1, const float e0) {
    return _mm_setr_ps(e3, e2, e1, e0);
}

TARGET("sse")
__m128 sse_setzero_ps() {
    return _mm_setzero_ps();
}

TARGET("sse")
void sse_sfence() {
    _mm_sfence();
}

TARGET("sse")
__m128 sse_shuffle_ps(const __m128 a, const __m128 b, const uint8_t imm8) {
    DISPATCH(256, _mm_shuffle_ps, imm8, a, b);
}

TARGET("sse")
__m128 sse_sqrt_ps(const __m128 a) {
    return _mm_sqrt_ps(a);
}

TARGET("sse")
__m128 sse_sqrt_ss(const __m128 a) {
    return _mm_sqrt_ss(a);
}

TARGET("sse")
void sse_store_ps(float* mem_addr, const __m128 a) {
    _mm_store_ps(mem_addr, a);
}

TARGET("sse")
void sse_store_ps1(float* mem_addr, const __m128 a) {
    _mm_store_ps1(mem_addr, a);
}

TARGET("sse")
void sse_store_ss(float* mem_addr, const __m128 a) {
    _mm_store_ss(mem_addr, a);
}

TARGET("sse")
void sse_store1_ps(float* mem_addr, const __m128 a) {
    _mm_store1_ps(mem_addr, a);
}

TARGET("sse")
void sse_storer_ps(float* mem_addr, const __m128 a) {
    _mm_storer_ps(mem_addr, a);
}

TARGET("sse")
void sse_storeu_ps(float* mem_addr, const __m128 a) {
    _mm_storeu_ps(mem_addr, a);
}

TARGET("sse")
void sse_stream_ps(float* mem_addr, const __m128 a) {
    _mm_stream_ps(mem_addr, a);
}

TARGET("sse")
__m128 sse_sub_ps(const __m128 a, const __m128 b) {
    return _mm_sub_ps(a, b);
}

TARGET("sse")
__m128 sse_sub_ss(const __m128 a, const __m128 b) {
    return _mm_sub_ss(a, b);
}

TARGET("sse")
void sse_transpose4_ps(__m128* row0, __m128* row1, __m128* row2, __m128* row3) {
    _MM_TRANSPOSE4_PS(*row0, *row1, *row2, *row3);
}

TARGET("sse")
bool sse_ucomieq_ss(const __m128 a, const __m128 b) {
    return _mm_ucomieq_ss(a, b);
}

TARGET("sse")
bool sse_ucomige_ss(const __m128 a, const __m128 b) {
    return _mm_ucomige_ss(a, b);
}

TARGET("sse")
bool sse_ucomigt_ss(const __m128 a, const __m128 b) {
    return _mm_ucomigt_ss(a, b);
}

TARGET("sse")
bool sse_ucomile_ss(const __m128 a, const __m128 b) {
    return _mm_ucomile_ss(a, b);
}

TARGET("sse")
bool sse_ucomilt_ss(const __m128 a, const __m128 b) {
    return _mm_ucomilt_ss(a, b);
}

TARGET("sse")
bool sse_ucomineq_ss(const __m128 a, const __m128 b) {
    return _mm_ucomineq_ss(a, b);
}

TARGET("sse")
__m128 sse_undefined_ps() {
    return _mm_undefined_ps();
}

TARGET("sse")
__m128 sse_unpackhi_ps(const __m128 a, const __m128 b) {
    return _mm_unpackhi_ps(a, b);
}

TARGET("sse")
__m128 sse_unpacklo_ps(const __m128 a, const __m128 b) {
    return _mm_unpacklo_ps(a, b);
}

TARGET("sse")
__m128 sse_xor_ps(const __m128 a, const __m128 b) {
    return _mm_xor_ps(a, b);
}

//endregion === SSE ===

//region === SSE2 ===

TARGET("sse2")
__m128i sse2_add_epi16(const __m128i a, const __m128i b) {
    return _mm_add_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_add_epi32(const __m128i a, const __m128i b) {
    return _mm_add_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_add_epi64(const __m128i a, const __m128i b) {
    return _mm_add_epi64(a, b);
}

TARGET("sse2")
__m128i sse2_add_epi8(const __m128i a, const __m128i b) {
    return _mm_add_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_add_pd(const __m128d a, const __m128d b) {
    return _mm_add_pd(a, b);
}

TARGET("sse2")
__m128d sse2_add_sd(const __m128d a, const __m128d b) {
    return _mm_add_sd(a, b);
}

TARGET("sse2")
__m128i sse2_adds_epi16(const __m128i a, const __m128i b) {
    return _mm_adds_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_adds_epi8(const __m128i a, const __m128i b) {
    return _mm_adds_epi8(a, b);
}

TARGET("sse2")
__m128i sse2_adds_epu16(const __m128i a, const __m128i b) {
    return _mm_adds_epu16(a, b);
}

TARGET("sse2")
__m128i sse2_adds_epu8(const __m128i a, const __m128i b) {
    return _mm_adds_epu8(a, b);
}

TARGET("sse2")
__m128d sse2_and_pd(const __m128d a, const __m128d b) {
    return _mm_and_pd(a, b);
}

TARGET("sse2")
__m128i sse2_and_si128(const __m128i a, const __m128i b) {
    return _mm_and_si128(a, b);
}

TARGET("sse2")
__m128d sse2_andnot_pd(const __m128d a, const __m128d b) {
    return _mm_andnot_pd(a, b);
}

TARGET("sse2")
__m128i sse2_andnot_si128(const __m128i a, const __m128i b) {
    return _mm_andnot_si128(a, b);
}

TARGET("sse2")
__m128i sse2_avg_epu16(const __m128i a, const __m128i b) {
    return _mm_avg_epu16(a, b);
}

TARGET("sse2")
__m128i sse2_avg_epu8(const __m128i a, const __m128i b) {
    return _mm_avg_epu8(a, b);
}

TARGET("sse2")
__m128i sse2_bslli_si128(const __m128i a, const uint8_t imm8) {
    const auto _imm8 = imm8 < 16 ? imm8 : 16;
    DISPATCH(17, _mm_bslli_si128, _imm8, a);
}

TARGET("sse2")
__m128i sse2_bsrli_si128(const __m128i a, const uint8_t imm8) {
    const auto _imm8 = imm8 < 16 ? imm8 : 16;
    DISPATCH(17, _mm_bsrli_si128, _imm8, a);
}

TARGET("sse2")
__m128 sse2_castpd_ps(const __m128d a) {
    return _mm_castpd_ps(a);
}

TARGET("sse2")
__m128i sse2_castpd_si128(const __m128d a) {
    return _mm_castpd_si128(a);
}

TARGET("sse2")
__m128d sse2_castps_pd(const __m128 a) {
    return _mm_castps_pd(a);
}

TARGET("sse2")
__m128i sse2_castps_si128(const __m128 a) {
    return _mm_castps_si128(a);
}

TARGET("sse2")
__m128d sse2_castsi128_pd(const __m128i a) {
    return _mm_castsi128_pd(a);
}

TARGET("sse2")
__m128 sse2_castsi128_ps(const __m128i a) {
    return _mm_castsi128_ps(a);
}

TARGET("sse2")
void sse2_clflush(const void* p) {
    _mm_clflush(p);
}

TARGET("sse2")
__m128i sse2_cmpeq_epi16(const __m128i a, const __m128i b) {
    return _mm_cmpeq_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_cmpeq_epi32(const __m128i a, const __m128i b) {
    return _mm_cmpeq_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_cmpeq_epi8(const __m128i a, const __m128i b) {
    return _mm_cmpeq_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_cmpeq_pd(const __m128d a, const __m128d b) {
    return _mm_cmpeq_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpeq_sd(const __m128d a, const __m128d b) {
    return _mm_cmpeq_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpge_pd(const __m128d a, const __m128d b) {
    return _mm_cmpge_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpge_sd(const __m128d a, const __m128d b) {
    return _mm_cmpge_sd(a, b);
}

TARGET("sse2")
__m128i sse2_cmpgt_epi16(const __m128i a, const __m128i b) {
    return _mm_cmpgt_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_cmpgt_epi32(const __m128i a, const __m128i b) {
    return _mm_cmpgt_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_cmpgt_epi8(const __m128i a, const __m128i b) {
    return _mm_cmpgt_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_cmpgt_pd(const __m128d a, const __m128d b) {
    return _mm_cmpgt_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpgt_sd(const __m128d a, const __m128d b) {
    return _mm_cmpgt_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmple_pd(const __m128d a, const __m128d b) {
    return _mm_cmple_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmple_sd(const __m128d a, const __m128d b) {
    return _mm_cmple_sd(a, b);
}

TARGET("sse2")
__m128i sse2_cmplt_epi16(const __m128i a, const __m128i b) {
    return _mm_cmplt_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_cmplt_epi32(const __m128i a, const __m128i b) {
    return _mm_cmplt_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_cmplt_epi8(const __m128i a, const __m128i b) {
    return _mm_cmplt_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_cmplt_pd(const __m128d a, const __m128d b) {
    return _mm_cmplt_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmplt_sd(const __m128d a, const __m128d b) {
    return _mm_cmplt_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpneq_pd(const __m128d a, const __m128d b) {
    return _mm_cmpneq_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpneq_sd(const __m128d a, const __m128d b) {
    return _mm_cmpneq_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpnge_pd(const __m128d a, const __m128d b) {
    return _mm_cmpnge_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpnge_sd(const __m128d a, const __m128d b) {
    return _mm_cmpnge_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpngt_pd(const __m128d a, const __m128d b) {
    return _mm_cmpngt_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpngt_sd(const __m128d a, const __m128d b) {
    return _mm_cmpngt_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpnle_pd(const __m128d a, const __m128d b) {
    return _mm_cmpnle_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpnle_sd(const __m128d a, const __m128d b) {
    return _mm_cmpnle_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpnlt_pd(const __m128d a, const __m128d b) {
    return _mm_cmpnlt_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpnlt_sd(const __m128d a, const __m128d b) {
    return _mm_cmpnlt_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpord_pd(const __m128d a, const __m128d b) {
    return _mm_cmpord_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpord_sd(const __m128d a, const __m128d b) {
    return _mm_cmpord_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpunord_pd(const __m128d a, const __m128d b) {
    return _mm_cmpunord_pd(a, b);
}

TARGET("sse2")
__m128d sse2_cmpunord_sd(const __m128d a, const __m128d b) {
    return _mm_cmpunord_sd(a, b);
}

TARGET("sse2")
bool sse2_comieq_sd(const __m128d a, const __m128d b) {
    return _mm_comieq_sd(a, b);
}

TARGET("sse2")
bool sse2_comige_sd(const __m128d a, const __m128d b) {
    return _mm_comige_sd(a, b);
}

TARGET("sse2")
bool sse2_comigt_sd(const __m128d a, const __m128d b) {
    return _mm_comigt_sd(a, b);
}

TARGET("sse2")
bool sse2_comile_sd(const __m128d a, const __m128d b) {
    return _mm_comile_sd(a, b);
}

TARGET("sse2")
bool sse2_comilt_sd(const __m128d a, const __m128d b) {
    return _mm_comilt_sd(a, b);
}

TARGET("sse2")
bool sse2_comineq_sd(const __m128d a, const __m128d b) {
    return _mm_comineq_sd(a, b);
}

TARGET("sse2")
__m128d sse2_cvtepi32_pd(const __m128i a) {
    return _mm_cvtepi32_pd(a);
}

TARGET("sse2")
__m128 sse2_cvtepi32_ps(const __m128i a) {
    return _mm_cvtepi32_ps(a);
}

TARGET("sse2")
__m128i sse2_cvtpd_epi32(const __m128d a) {
    return _mm_cvtpd_epi32(a);
}

TARGET("sse2")
__m128 sse2_cvtpd_ps(const __m128d a) {
    return _mm_cvtpd_ps(a);
}

TARGET("sse2")
__m128i sse2_cvtps_epi32(const __m128 a) {
    return _mm_cvtps_epi32(a);
}

TARGET("sse2")
__m128d sse2_cvtps_pd(const __m128 a) {
    return _mm_cvtps_pd(a);
}

TARGET("sse2")
double sse2_cvtsd_f64(const __m128d a) {
    return _mm_cvtsd_f64(a);
}

TARGET("sse2")
int32_t sse2_cvtsd_si32(const __m128d a) {
    return _mm_cvtsd_si32(a);
}

TARGET("sse2")
int64_t sse2_cvtsd_si64(const __m128d a) {
    return _mm_cvtsd_si64(a);
}

TARGET("sse2")
__m128 sse2_cvtsd_ss(const __m128 a, const __m128d b) {
    return _mm_cvtsd_ss(a, b);
}

TARGET("sse2")
int32_t sse2_cvtsi128_si32(const __m128i a) {
    return _mm_cvtsi128_si32(a);
}

TARGET("sse2")
int64_t sse2_cvtsi128_si64(const __m128i a) {
    return _mm_cvtsi128_si64(a);
}

TARGET("sse2")
__m128d sse2_cvtsi32_sd(const __m128d a, const int32_t b) {
    return _mm_cvtsi32_sd(a, b);
}

TARGET("sse2")
__m128i sse2_cvtsi32_si128(const int32_t a) {
    return _mm_cvtsi32_si128(a);
}

TARGET("sse2")
__m128d sse2_cvtsi64_sd(const __m128d a, const int64_t b) {
    return _mm_cvtsi64_sd(a, b);
}

TARGET("sse2")
__m128i sse2_cvtsi64_si128(const int64_t a) {
    return _mm_cvtsi64_si128(a);
}

TARGET("sse2")
__m128d sse2_cvtss_sd(const __m128d a, const __m128 b) {
    return _mm_cvtss_sd(a, b);
}

TARGET("sse2")
__m128i sse2_cvttpd_epi32(const __m128d a) {
    return _mm_cvttpd_epi32(a);
}

TARGET("sse2")
__m128i sse2_cvttps_epi32(const __m128 a) {
    return _mm_cvttps_epi32(a);
}

TARGET("sse2")
int32_t sse2_cvttsd_si32(const __m128d a) {
    return _mm_cvttsd_si32(a);
}

TARGET("sse2")
int64_t sse2_cvttsd_si64(const __m128d a) {
    return _mm_cvttsd_si64(a);
}

TARGET("sse2")
__m128d sse2_div_pd(const __m128d a, const __m128d b) {
    return _mm_div_pd(a, b);
}

TARGET("sse2")
__m128d sse2_div_sd(const __m128d a, const __m128d b) {
    return _mm_div_sd(a, b);
}

TARGET("sse2")
int16_t sse2_extract_epi16(const __m128i a, const uint8_t imm8) {
    DISPATCH(8, _mm_extract_epi16, imm8 & 0x07, a);
}

TARGET("sse2")
__m128i sse2_insert_epi16(const __m128i a, const int16_t i, const uint8_t imm8) {
    DISPATCH(8, _mm_insert_epi16, imm8 & 0x07, a, i);
}

TARGET("sse2")
void sse2_lfence() {
    _mm_lfence();
}

TARGET("sse2")
__m128d sse2_load_pd(const double* mem_addr) {
    return _mm_load_pd(mem_addr);
}

TARGET("sse2")
__m128d sse2_load_pd1(const double* mem_addr) {
    return _mm_load_pd1(mem_addr);
}

TARGET("sse2")
__m128d sse2_load_sd(const double* mem_addr) {
    return _mm_load_sd(mem_addr);
}

TARGET("sse2")
__m128i sse2_load_si128(const __m128i* mem_addr) {
    return _mm_load_si128(mem_addr);
}

TARGET("sse2")
__m128d sse2_load1_pd(const double* mem_addr) {
    return _mm_load1_pd(mem_addr);
}

TARGET("sse2")
__m128d sse2_loadh_pd(const __m128d a, const double* mem_addr) {
    return _mm_loadh_pd(a, mem_addr);
}

TARGET("sse2")
__m128i sse2_loadl_epi64(const __m128i* mem_addr) {
    return _mm_loadl_epi64(mem_addr);
}

TARGET("sse2")
__m128d sse2_loadl_pd(const __m128d a, const double* mem_addr) {
    return _mm_loadl_pd(a, mem_addr);
}

TARGET("sse2")
__m128d sse2_loadr_pd(const double* mem_addr) {
    return _mm_loadr_pd(mem_addr);
}

TARGET("sse2")
__m128d sse2_loadu_pd(const double* mem_addr) {
    return _mm_loadu_pd(mem_addr);
}

TARGET("sse2")
__m128i sse2_loadu_si128(const __m128i* mem_addr) {
    return _mm_loadu_si128(mem_addr);
}

TARGET("sse2")
__m128i sse2_madd_epi16(const __m128i a, const __m128i b) {
    return _mm_madd_epi16(a, b);
}

TARGET("sse2")
void sse2_maskmoveu_si128(const __m128i a, const __m128i mask, int8_t* mem_addr) {
    _mm_maskmoveu_si128(a, mask, reinterpret_cast<char*>(mem_addr));
}

TARGET("sse2")
__m128i sse2_max_epi16(const __m128i a, const __m128i b) {
    return _mm_max_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_max_epu8(const __m128i a, const __m128i b) {
    return _mm_max_epu8(a, b);
}

TARGET("sse2")
__m128d sse2_max_pd(const __m128d a, const __m128d b) {
    return _mm_max_pd(a, b);
}

TARGET("sse2")
__m128d sse2_max_sd(const __m128d a, const __m128d b) {
    return _mm_max_sd(a, b);
}

TARGET("sse2")
void sse2_mfence() {
    _mm_mfence();
}

TARGET("sse2")
__m128i sse2_min_epi16(const __m128i a, const __m128i b) {
    return _mm_min_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_min_epu8(const __m128i a, const __m128i b) {
    return _mm_min_epu8(a, b);
}

TARGET("sse2")
__m128d sse2_min_pd(const __m128d a, const __m128d b) {
    return _mm_min_pd(a, b);
}

TARGET("sse2")
__m128d sse2_min_sd(const __m128d a, const __m128d b) {
    return _mm_min_sd(a, b);
}

TARGET("sse2")
__m128i sse2_move_epi64(const __m128i a) {
    return _mm_move_epi64(a);
}

TARGET("sse2")
__m128d sse2_move_sd(const __m128d a, const __m128d b) {
    return _mm_move_sd(a, b);
}

TARGET("sse2")
int32_t sse2_movemask_epi8(const __m128i a) {
    return _mm_movemask_epi8(a);
}

TARGET("sse2")
int32_t sse2_movemask_pd(const __m128d a) {
    return _mm_movemask_pd(a);
}

TARGET("sse2")
__m128i sse2_mul_epu32(const __m128i a, const __m128i b) {
    return _mm_mul_epu32(a, b);
}

TARGET("sse2")
__m128d sse2_mul_pd(const __m128d a, const __m128d b) {
    return _mm_mul_pd(a, b);
}

TARGET("sse2")
__m128d sse2_mul_sd(const __m128d a, const __m128d b) {
    return _mm_mul_sd(a, b);
}

TARGET("sse2")
__m128i sse2_mulhi_epi16(const __m128i a, const __m128i b) {
    return _mm_mulhi_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_mulhi_epu16(const __m128i a, const __m128i b) {
    return _mm_mulhi_epu16(a, b);
}

TARGET("sse2")
__m128i sse2_mullo_epi16(const __m128i a, const __m128i b) {
    return _mm_mullo_epi16(a, b);
}

TARGET("sse2")
__m128d sse2_or_pd(const __m128d a, const __m128d b) {
    return _mm_or_pd(a, b);
}

TARGET("sse2")
__m128i sse2_or_si128(const __m128i a, const __m128i b) {
    return _mm_or_si128(a, b);
}

TARGET("sse2")
__m128i sse2_packs_epi16(const __m128i a, const __m128i b) {
    return _mm_packs_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_packs_epi32(const __m128i a, const __m128i b) {
    return _mm_packs_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_packus_epi16(const __m128i a, const __m128i b) {
    return _mm_packus_epi16(a, b);
}

TARGET("sse2")
void sse2_pause() {
    _mm_pause();
}

TARGET("sse2")
__m128i sse2_sad_epu8(const __m128i a, const __m128i b) {
    return _mm_sad_epu8(a, b);
}

TARGET("sse2")
__m128i sse2_set_epi16(
    const int16_t e7, const int16_t e6, const int16_t e5, const int16_t e4,
    const int16_t e3, const int16_t e2, const int16_t e1, const int16_t e0
) {
    return _mm_set_epi16(e7, e6, e5, e4, e3, e2, e1, e0);
}

TARGET("sse2")
__m128i sse2_set_epi32(const int32_t e3, const int32_t e2, const int32_t e1, const int32_t e0) {
    return _mm_set_epi32(e3, e2, e1, e0);
}

TARGET("sse2")
__m128i sse2_set_epi64(const int64_t e1, const int64_t e0) {
    return _mm_set_epi64x(e1, e0);
}

TARGET("sse2")
__m128i sse2_set_epi8(
    const int8_t e15, const int8_t e14, const int8_t e13, const int8_t e12,
    const int8_t e11, const int8_t e10, const int8_t e9, const int8_t e8,
    const int8_t e7, const int8_t e6, const int8_t e5, const int8_t e4,
    const int8_t e3, const int8_t e2, const int8_t e1, const int8_t e0
) {
    return _mm_set_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0);
}

TARGET("sse2")
__m128d sse2_set_pd(const double e1, const double e0) {
    return _mm_set_pd(e1, e0);
}

TARGET("sse2")
__m128d sse2_set_pd1(const double a) {
    return _mm_set_pd1(a);
}

TARGET("sse2")
__m128d sse2_set_sd(const double a) {
    return _mm_set_sd(a);
}

TARGET("sse2")
__m128i sse2_set1_epi16(const int16_t a) {
    return _mm_set1_epi16(a);
}

TARGET("sse2")
__m128i sse2_set1_epi32(const int32_t a) {
    return _mm_set1_epi32(a);
}

TARGET("sse2")
__m128i sse2_set1_epi64(const int64_t a) {
    return _mm_set1_epi64x(a);
}

TARGET("sse2")
__m128i sse2_set1_epi8(const int8_t a) {
    return _mm_set1_epi8(a);
}

TARGET("sse2")
__m128d sse2_set1_pd(const double a) {
    return _mm_set1_pd(a);
}

TARGET("sse2")
__m128i sse2_setr_epi16(
    const int16_t e7, const int16_t e6, const int16_t e5, const int16_t e4,
    const int16_t e3, const int16_t e2, const int16_t e1, const int16_t e0
) {
    return _mm_setr_epi16(e7, e6, e5, e4, e3, e2, e1, e0);
}

TARGET("sse2")
__m128i sse2_setr_epi32(const int32_t e3, const int32_t e2, const int32_t e1, const int32_t e0) {
    return _mm_setr_epi32(e3, e2, e1, e0);
}

TARGET("sse2")
__m128i sse2_setr_epi8(
    const int8_t e15, const int8_t e14, const int8_t e13, const int8_t e12,
    const int8_t e11, const int8_t e10, const int8_t e9, const int8_t e8,
    const int8_t e7, const int8_t e6, const int8_t e5, const int8_t e4,
    const int8_t e3, const int8_t e2, const int8_t e1, const int8_t e0
) {
    return _mm_setr_epi8(e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0);
}

TARGET("sse2")
__m128d sse2_setr_pd(const double e1, const double e0) {
    return _mm_setr_pd(e1, e0);
}

TARGET("sse2")
__m128d sse2_setzero_pd() {
    return _mm_setzero_pd();
}

TARGET("sse2")
__m128i sse2_setzero_si128() {
    return _mm_setzero_si128();
}

TARGET("sse2")
__m128i sse2_shuffle_epi32(const __m128i a, const uint8_t imm8) {
    DISPATCH(256, _mm_shuffle_epi32, imm8, a);
}

TARGET("sse2")
__m128d sse2_shuffle_pd(const __m128d a, const __m128d b, const uint8_t imm8) {
    DISPATCH(4, _mm_shuffle_pd, imm8 & 0x03, a, b);
}

TARGET("sse2")
__m128i sse2_shufflehi_epi16(const __m128i a, const uint8_t imm8) {
    DISPATCH(256, _mm_shufflehi_epi16, imm8, a);
}

TARGET("sse2")
__m128i sse2_shufflelo_epi16(const __m128i a, const uint8_t imm8) {
    DISPATCH(256, _mm_shufflelo_epi16, imm8, a);
}

TARGET("sse2")
__m128i sse2_sll_epi16(const __m128i a, const __m128i count) {
    return _mm_sll_epi16(a, count);
}

TARGET("sse2")
__m128i sse2_sll_epi32(const __m128i a, const __m128i count) {
    return _mm_sll_epi32(a, count);
}

TARGET("sse2")
__m128i sse2_sll_epi64(const __m128i a, const __m128i count) {
    return _mm_sll_epi64(a, count);
}

TARGET("sse2")
__m128i sse2_slli_epi16(const __m128i a, const uint8_t imm8) {
    return _mm_slli_epi16(a, imm8);
}

TARGET("sse2")
__m128i sse2_slli_epi32(const __m128i a, const uint8_t imm8) {
    return _mm_slli_epi32(a, imm8);
}

TARGET("sse2")
__m128i sse2_slli_epi64(const __m128i a, const uint8_t imm8) {
    return _mm_slli_epi64(a, imm8);
}

TARGET("sse2")
__m128i sse2_slli_si128(const __m128i a, const uint8_t imm8) {
    const auto _imm8 = imm8 < 16 ? imm8 : 16;
    DISPATCH(17, _mm_slli_si128, _imm8, a);
}

TARGET("sse2")
__m128d sse2_sqrt_pd(const __m128d a) {
    return _mm_sqrt_pd(a);
}

TARGET("sse2")
__m128d sse2_sqrt_sd(const __m128d a, const __m128d b) {
    return _mm_sqrt_sd(a, b);
}

TARGET("sse2")
__m128i sse2_sra_epi16(const __m128i a, const __m128i count) {
    return _mm_sra_epi16(a, count);
}

TARGET("sse2")
__m128i sse2_sra_epi32(const __m128i a, const __m128i count) {
    return _mm_sra_epi32(a, count);
}

TARGET("sse2")
__m128i sse2_srai_epi16(const __m128i a, const uint8_t imm8) {
    return _mm_srai_epi16(a, imm8);
}

TARGET("sse2")
__m128i sse2_srai_epi32(const __m128i a, const uint8_t imm8) {
    return _mm_srai_epi32(a, imm8);
}

TARGET("sse2")
__m128i sse2_srl_epi16(const __m128i a, const __m128i count) {
    return _mm_srl_epi16(a, count);
}

TARGET("sse2")
__m128i sse2_srl_epi32(const __m128i a, const __m128i count) {
    return _mm_srl_epi32(a, count);
}

TARGET("sse2")
__m128i sse2_srl_epi64(const __m128i a, const __m128i count) {
    return _mm_srl_epi64(a, count);
}

TARGET("sse2")
__m128i sse2_srli_epi16(const __m128i a, const uint8_t imm8) {
    return _mm_srli_epi16(a, imm8);
}

TARGET("sse2")
__m128i sse2_srli_epi32(const __m128i a, const uint8_t imm8) {
    return _mm_srli_epi32(a, imm8);
}

TARGET("sse2")
__m128i sse2_srli_epi64(const __m128i a, const uint8_t imm8) {
    return _mm_srli_epi64(a, imm8);
}

TARGET("sse2")
__m128i sse2_srli_si128(const __m128i a, const uint8_t imm8) {
    const auto _imm8 = imm8 < 16 ? imm8 : 16;
    DISPATCH(17, _mm_srli_si128, _imm8, a);
}

TARGET("sse2")
void sse2_store_pd(double* mem_addr, const __m128d a) {
    _mm_store_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_store_pd1(double* mem_addr, const __m128d a) {
    _mm_store_pd1(mem_addr, a);
}

TARGET("sse2")
void sse2_store_sd(double* mem_addr, const __m128d a) {
    _mm_store_sd(mem_addr, a);
}

TARGET("sse2")
void sse2_store_si128(__m128i* mem_addr, const __m128i a) {
    _mm_store_si128(mem_addr, a);
}

TARGET("sse2")
void sse2_store1_pd(double* mem_addr, const __m128d a) {
    _mm_store1_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_storeh_pd(double* mem_addr, const __m128d a) {
    _mm_storeh_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_storel_epi64(__m128i* mem_addr, const __m128i a) {
    _mm_storel_epi64(mem_addr, a);
}

TARGET("sse2")
void sse2_storel_pd(double* mem_addr, const __m128d a) {
    _mm_storel_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_storer_pd(double* mem_addr, const __m128d a) {
    _mm_storer_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_storeu_pd(double* mem_addr, const __m128d a) {
    _mm_storeu_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_storeu_si128(__m128i* mem_addr, const __m128i a) {
    _mm_storeu_si128(mem_addr, a);
}

TARGET("sse2")
void sse2_stream_pd(double* mem_addr, const __m128d a) {
    _mm_stream_pd(mem_addr, a);
}

TARGET("sse2")
void sse2_stream_si128(__m128i* mem_addr, const __m128i a) {
    _mm_stream_si128(mem_addr, a);
}

TARGET("sse2")
void sse2_stream_si32(int32_t* mem_addr, const int32_t a) {
    _mm_stream_si32(mem_addr, a);
}

TARGET("sse2")
void sse2_stream_si64(int64_t* mem_addr, const int64_t a) {
    _mm_stream_si64(reinterpret_cast<long long*>(mem_addr), a);
}

TARGET("sse2")
__m128i sse2_sub_epi16(const __m128i a, const __m128i b) {
    return _mm_sub_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_sub_epi32(const __m128i a, const __m128i b) {
    return _mm_sub_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_sub_epi64(const __m128i a, const __m128i b) {
    return _mm_sub_epi64(a, b);
}

TARGET("sse2")
__m128i sse2_sub_epi8(const __m128i a, const __m128i b) {
    return _mm_sub_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_sub_pd(const __m128d a, const __m128d b) {
    return _mm_sub_pd(a, b);
}

TARGET("sse2")
__m128d sse2_sub_sd(const __m128d a, const __m128d b) {
    return _mm_sub_sd(a, b);
}

TARGET("sse2")
__m128i sse2_subs_epi16(const __m128i a, const __m128i b) {
    return _mm_subs_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_subs_epi8(const __m128i a, const __m128i b) {
    return _mm_subs_epi8(a, b);
}

TARGET("sse2")
__m128i sse2_subs_epu16(const __m128i a, const __m128i b) {
    return _mm_subs_epu16(a, b);
}

TARGET("sse2")
__m128i sse2_subs_epu8(const __m128i a, const __m128i b) {
    return _mm_subs_epu8(a, b);
}

TARGET("sse2")
bool sse2_ucomieq_sd(const __m128d a, const __m128d b) {
    return _mm_ucomieq_sd(a, b);
}

TARGET("sse2")
bool sse2_ucomige_sd(const __m128d a, const __m128d b) {
    return _mm_ucomige_sd(a, b);
}

TARGET("sse2")
bool sse2_ucomigt_sd(const __m128d a, const __m128d b) {
    return _mm_ucomigt_sd(a, b);
}

TARGET("sse2")
bool sse2_ucomile_sd(const __m128d a, const __m128d b) {
    return _mm_ucomile_sd(a, b);
}

TARGET("sse2")
bool sse2_ucomilt_sd(const __m128d a, const __m128d b) {
    return _mm_ucomilt_sd(a, b);
}

TARGET("sse2")
bool sse2_ucomineq_sd(const __m128d a, const __m128d b) {
    return _mm_ucomineq_sd(a, b);
}

TARGET("sse2")
__m128d sse2_undefined_pd() {
    return _mm_undefined_pd();
}

TARGET("sse2")
__m128i sse2_undefined_si128() {
    return _mm_undefined_si128();
}

TARGET("sse2")
__m128i sse2_unpackhi_epi16(const __m128i a, const __m128i b) {
    return _mm_unpackhi_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_unpackhi_epi32(const __m128i a, const __m128i b) {
    return _mm_unpackhi_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_unpackhi_epi64(const __m128i a, const __m128i b) {
    return _mm_unpackhi_epi64(a, b);
}

TARGET("sse2")
__m128i sse2_unpackhi_epi8(const __m128i a, const __m128i b) {
    return _mm_unpackhi_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_unpackhi_pd(const __m128d a, const __m128d b) {
    return _mm_unpackhi_pd(a, b);
}

TARGET("sse2")
__m128i sse2_unpacklo_epi16(const __m128i a, const __m128i b) {
    return _mm_unpacklo_epi16(a, b);
}

TARGET("sse2")
__m128i sse2_unpacklo_epi32(const __m128i a, const __m128i b) {
    return _mm_unpacklo_epi32(a, b);
}

TARGET("sse2")
__m128i sse2_unpacklo_epi64(const __m128i a, const __m128i b) {
    return _mm_unpacklo_epi64(a, b);
}

TARGET("sse2")
__m128i sse2_unpacklo_epi8(const __m128i a, const __m128i b) {
    return _mm_unpacklo_epi8(a, b);
}

TARGET("sse2")
__m128d sse2_unpacklo_pd(const __m128d a, const __m128d b) {
    return _mm_unpacklo_pd(a, b);
}

TARGET("sse2")
__m128d sse2_xor_pd(const __m128d a, const __m128d b) {
    return _mm_xor_pd(a, b);
}

TARGET("sse2")
__m128i sse2_xor_si128(const __m128i a, const __m128i b) {
    return _mm_xor_si128(a, b);
}

//endregion === SSE2 ===

//region === SSE3 ===

TARGET("sse3")
__m128d sse3_addsub_pd(const __m128d a, const __m128d b) {
    return _mm_addsub_pd(a, b);
}

TARGET("sse3")
__m128 sse3_addsub_ps(const __m128 a, const __m128 b) {
    return _mm_addsub_ps(a, b);
}

TARGET("sse3")
__m128d sse3_hadd_pd(const __m128d a, const __m128d b) {
    return _mm_hadd_pd(a, b);
}

TARGET("sse3")
__m128 sse3_hadd_ps(const __m128 a, const __m128 b) {
    return _mm_hadd_ps(a, b);
}

TARGET("sse3")
__m128d sse3_hsub_pd(const __m128d a, const __m128d b) {
    return _mm_hsub_pd(a, b);
}

TARGET("sse3")
__m128 sse3_hsub_ps(const __m128 a, const __m128 b) {
    return _mm_hsub_ps(a, b);
}

TARGET("sse3")
__m128i sse3_lddqu_si128(const __m128i* mem_addr) {
    return _mm_lddqu_si128(mem_addr);
}

TARGET("sse3")
__m128d sse3_loaddup_pd(const double* mem_addr) {
    return _mm_loaddup_pd(mem_addr);
}

TARGET("sse3")
__m128d sse3_movedup_pd(const __m128d a) {
    return _mm_movedup_pd(a);
}

TARGET("sse3")
__m128 sse3_movehdup_ps(const __m128 a) {
    return _mm_movehdup_ps(a);
}

TARGET("sse3")
__m128 sse3_moveldup_ps(const __m128 a) {
    return _mm_moveldup_ps(a);
}

//endregion === SSE3 ===

//region === SSSE3 ===

TARGET("ssse3")
__m128i ssse3_abs_epi16(const __m128i a) {
    return _mm_abs_epi16(a);
}

TARGET("ssse3")
__m128i ssse3_abs_epi32(const __m128i a) {
    return _mm_abs_epi32(a);
}

TARGET("ssse3")
__m128i ssse3_abs_epi8(const __m128i a) {
    return _mm_abs_epi8(a);
}

TARGET("ssse3")
__m128i ssse3_alignr_epi8(const __m128i a, const __m128i b, const uint8_t imm8) {
    const auto _imm8 = imm8 < 32 ? imm8 : 32;
    DISPATCH(33, _mm_alignr_epi8, _imm8, a, b);
}


TARGET("ssse3")
__m128i ssse3_hadd_epi16(const __m128i a, const __m128i b) {
    return _mm_hadd_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_hadd_epi32(const __m128i a, const __m128i b) {
    return _mm_hadd_epi32(a, b);
}

TARGET("ssse3")
__m128i ssse3_hadds_epi16(const __m128i a, const __m128i b) {
    return _mm_hadds_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_hsub_epi16(const __m128i a, const __m128i b) {
    return _mm_hsub_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_hsub_epi32(const __m128i a, const __m128i b) {
    return _mm_hsub_epi32(a, b);
}

TARGET("ssse3")
__m128i ssse3_hsubs_epi16(const __m128i a, const __m128i b) {
    return _mm_hsubs_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_maddubs_epi16(const __m128i a, const __m128i b) {
    return _mm_maddubs_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_mulhrs_epi16(const __m128i a, const __m128i b) {
    return _mm_mulhrs_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_shuffle_epi8(const __m128i a, const __m128i b) {
    return _mm_shuffle_epi8(a, b);
}

TARGET("ssse3")
__m128i ssse3_sign_epi16(const __m128i a, const __m128i b) {
    return _mm_sign_epi16(a, b);
}

TARGET("ssse3")
__m128i ssse3_sign_epi32(const __m128i a, const __m128i b) {
    return _mm_sign_epi32(a, b);
}

TARGET("ssse3")
__m128i ssse3_sign_epi8(const __m128i a, const __m128i b) {
    return _mm_sign_epi8(a, b);
}

//endregion === SSSE3 ===

//region === SSE4.1 ===

const int32_t
    SSE4_1_FROUND_TO_NEAREST_INT = _MM_FROUND_TO_NEAREST_INT,
    SSE4_1_FROUND_TO_NEG_INF = _MM_FROUND_TO_NEG_INF,
    SSE4_1_FROUND_TO_POS_INF = _MM_FROUND_TO_POS_INF,
    SSE4_1_FROUND_TO_ZERO = _MM_FROUND_TO_ZERO,
    SSE4_1_FROUND_CUR_DIRECTION = _MM_FROUND_CUR_DIRECTION;

const int32_t
    SSE4_1_FROUND_RAISE_EXC = _MM_FROUND_RAISE_EXC,
    SSE4_1_FROUND_NO_EXC = _MM_FROUND_NO_EXC;

const int32_t
    SSE4_1_FROUND_NINT = _MM_FROUND_NINT,
    SSE4_1_FROUND_FLOOR = _MM_FROUND_FLOOR,
    SSE4_1_FROUND_CEIL = _MM_FROUND_CEIL,
    SSE4_1_FROUND_TRUNC = _MM_FROUND_TRUNC,
    SSE4_1_FROUND_RINT = _MM_FROUND_RINT,
    SSE4_1_FROUND_NEARBYINT = _MM_FROUND_NEARBYINT;


TARGET("sse4.1")
__m128i sse4_1_blend_epi16(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(256, _mm_blend_epi16, imm8, a, b);
}

TARGET("sse4.1")
__m128d sse4_1_blend_pd(const __m128d a, const __m128d b, const uint8_t imm8) {
    DISPATCH(4, _mm_blend_pd, imm8 & 0x03, a, b);
}

TARGET("sse4.1")
__m128 sse4_1_blend_ps(const __m128 a, const __m128 b, const uint8_t imm8) {
    DISPATCH(16, _mm_blend_ps, imm8 & 0x0F, a, b);
}

TARGET("sse4.1")
__m128i sse4_1_blendv_epi8(const __m128i a, const __m128i b, const __m128i mask) {
    return _mm_blendv_epi8(a, b, mask);
}

TARGET("sse4.1")
__m128d sse4_1_blendv_pd(const __m128d a, const __m128d b, const __m128d mask) {
    return _mm_blendv_pd(a, b, mask);
}

TARGET("sse4.1")
__m128 sse4_1_blendv_ps(const __m128 a, const __m128 b, const __m128 mask) {
    return _mm_blendv_ps(a, b, mask);
}

TARGET("sse4.1")
__m128d sse4_1_ceil_pd(const __m128d a) {
    return _mm_ceil_pd(a);
}

TARGET("sse4.1")
__m128 sse4_1_ceil_ps(const __m128 a) {
    return _mm_ceil_ps(a);
}

TARGET("sse4.1")
__m128d sse4_1_ceil_sd(const __m128d a, const __m128d b) {
    return _mm_ceil_sd(a, b);
}

TARGET("sse4.1")
__m128 sse4_1_ceil_ss(const __m128 a, const __m128 b) {
    return _mm_ceil_ss(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_cmpeq_epi64(const __m128i a, const __m128i b) {
    return _mm_cmpeq_epi64(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepi16_epi32(const __m128i a) {
    return _mm_cvtepi16_epi32(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepi16_epi64(const __m128i a) {
    return _mm_cvtepi16_epi64(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepi32_epi64(const __m128i a) {
    return _mm_cvtepi32_epi64(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepi8_epi16(const __m128i a) {
    return _mm_cvtepi8_epi16(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepi8_epi32(const __m128i a) {
    return _mm_cvtepi8_epi32(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepi8_epi64(const __m128i a) {
    return _mm_cvtepi8_epi64(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepu16_epi32(const __m128i a) {
    return _mm_cvtepu16_epi32(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepu16_epi64(const __m128i a) {
    return _mm_cvtepu16_epi64(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepu32_epi64(const __m128i a) {
    return _mm_cvtepu32_epi64(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepu8_epi16(const __m128i a) {
    return _mm_cvtepu8_epi16(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepu8_epi32(const __m128i a) {
    return _mm_cvtepu8_epi32(a);
}

TARGET("sse4.1")
__m128i sse4_1_cvtepu8_epi64(const __m128i a) {
    return _mm_cvtepu8_epi64(a);
}


TARGET("sse4.1")
__m128d sse4_1_dp_pd(const __m128d a, const __m128d b, const uint8_t imm8) {
    DISPATCH(256, _mm_dp_pd, imm8, a, b);
}

TARGET("sse4.1")
__m128 sse4_1_dp_ps(const __m128 a, const __m128 b, const uint8_t imm8) {
    DISPATCH(256, _mm_dp_ps, imm8, a, b);
}

TARGET("sse4.1")
int32_t sse4_1_extract_epi32(const __m128i a, const uint8_t imm8) {
    DISPATCH(4, _mm_extract_epi32, imm8 & 0x03, a);
}

TARGET("sse4.1")
int64_t sse4_1_extract_epi64(const __m128i a, const uint8_t imm8) {
    DISPATCH(2, _mm_extract_epi64, imm8 & 0x01, a);
}

TARGET("sse4.1")
int8_t sse4_1_extract_epi8(const __m128i a, const uint8_t imm8) {
    DISPATCH(16, _mm_extract_epi8, imm8 & 0x0F, a);
}

#define extract_ps(A, IMM8) bit_cast(float, _mm_extract_ps(A, IMM8))

TARGET("sse4.1")
float sse4_1_extract_ps(const __m128 a, const uint8_t imm8) {
    DISPATCH(4, extract_ps, imm8 & 0x03, a);
}

TARGET("sse4.1")
__m128d sse4_1_floor_pd(const __m128d a) {
    return _mm_floor_pd(a);
}

TARGET("sse4.1")
__m128 sse4_1_floor_ps(const __m128 a) {
    return _mm_floor_ps(a);
}

TARGET("sse4.1")
__m128d sse4_1_floor_sd(const __m128d a, const __m128d b) {
    return _mm_floor_sd(a, b);
}

TARGET("sse4.1")
__m128 sse4_1_floor_ss(const __m128 a, const __m128 b) {
    return _mm_floor_ss(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_insert_epi32(const __m128i a, const int32_t i, const uint8_t imm8) {
    DISPATCH(4, _mm_insert_epi32, imm8 & 0x03, a, i);
}

TARGET("sse4.1")
__m128i sse4_1_insert_epi64(const __m128i a, const int64_t i, const uint8_t imm8) {
    DISPATCH(2, _mm_insert_epi64, imm8 & 0x01, a, i);
}

TARGET("sse4.1")
__m128i sse4_1_insert_epi8(const __m128i a, const int8_t i, const uint8_t imm8) {
    DISPATCH(16, _mm_insert_epi8, imm8 & 0x0F, a, i);
}

TARGET("sse4.1")
__m128 sse4_1_insert_ps(const __m128 a, const __m128 b, const uint8_t imm8) {
    DISPATCH(256, _mm_insert_ps, imm8, a, b);
}

TARGET("sse4.1")
__m128i sse4_1_max_epi32(const __m128i a, const __m128i b) {
    return _mm_max_epi32(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_max_epi8(const __m128i a, const __m128i b) {
    return _mm_max_epi8(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_max_epu16(const __m128i a, const __m128i b) {
    return _mm_max_epu16(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_max_epu32(const __m128i a, const __m128i b) {
    return _mm_max_epu32(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_min_epi32(const __m128i a, const __m128i b) {
    return _mm_min_epi32(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_min_epi8(const __m128i a, const __m128i b) {
    return _mm_min_epi8(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_min_epu16(const __m128i a, const __m128i b) {
    return _mm_min_epu16(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_min_epu32(const __m128i a, const __m128i b) {
    return _mm_min_epu32(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_minpos_epu16(const __m128i a) {
    return _mm_minpos_epu16(a);
}

TARGET("sse4.1")
__m128i sse4_1_mpsadbw_epu8(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(8, _mm_mpsadbw_epu8, imm8 & 0x07, a, b);
}

TARGET("sse4.1")
__m128i sse4_1_mul_epi32(const __m128i a, const __m128i b) {
    return _mm_mul_epi32(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_mullo_epi32(const __m128i a, const __m128i b) {
    return _mm_mullo_epi32(a, b);
}

TARGET("sse4.1")
__m128i sse4_1_packus_epi32(const __m128i a, const __m128i b) {
    return _mm_packus_epi32(a, b);
}

TARGET("sse4.1")
__m128d sse4_1_round_pd(const __m128d a, int32_t rounding) {
    DISPATCH(16, _mm_round_pd, rounding & 0x0F, a);
}

TARGET("sse4.1")
__m128 sse4_1_round_ps(const __m128 a, int32_t rounding) {
    DISPATCH(16, _mm_round_ps, rounding & 0x0F, a);
}

TARGET("sse4.1")
__m128d sse4_1_round_sd(const __m128d a, const __m128d b, int32_t rounding) {
    DISPATCH(16, _mm_round_sd, rounding & 0x0F, a, b);
}

TARGET("sse4.1")
__m128 sse4_1_round_ss(const __m128 a, const __m128 b, int32_t rounding) {
    DISPATCH(16, _mm_round_ss, rounding & 0x0F, a, b);
}

TARGET("sse4.1")
__m128i sse4_1_stream_load_si128(__m128i* mem_addr) {
    return _mm_stream_load_si128(mem_addr);
}

TARGET("sse4.1")
bool sse4_1_test_all_ones(const __m128i a) {
    return _mm_test_all_ones(a);
}

TARGET("sse4.1")
bool sse4_1_test_all_zeros(const __m128i a, const __m128i mask) {
    return _mm_test_all_zeros(a, mask);
}

TARGET("sse4.1")
bool sse4_1_test_mix_ones_zeros(const __m128i a, const __m128i mask) {
    return _mm_test_mix_ones_zeros(a, mask);
}

TARGET("sse4.1")
bool sse4_1_testc_si128(const __m128i a, const __m128i b) {
    return _mm_testc_si128(a, b);
}

TARGET("sse4.1")
bool sse4_1_testnzc_si128(const __m128i a, const __m128i b) {
    return _mm_testnzc_si128(a, b);
}

TARGET("sse4.1")
bool sse4_1_testz_si128(const __m128i a, const __m128i b) {
    return _mm_testz_si128(a, b);
}

//endregion === SSE4.1 ===

//region === SSE4.2 ===

const int32_t
    SIDD_UBYTE_OPS = _SIDD_UBYTE_OPS,
    SIDD_UWORD_OPS = _SIDD_UWORD_OPS,
    SIDD_SBYTE_OPS = _SIDD_SBYTE_OPS,
    SIDD_SWORD_OPS = _SIDD_SWORD_OPS;

const int32_t
    SIDD_CMP_EQUAL_ANY = _SIDD_CMP_EQUAL_ANY,
    SIDD_CMP_RANGES = _SIDD_CMP_RANGES,
    SIDD_CMP_EQUAL_EACH = _SIDD_CMP_EQUAL_EACH,
    SIDD_CMP_EQUAL_ORDERED = _SIDD_CMP_EQUAL_ORDERED;

const int32_t
    SIDD_POSITIVE_POLARITY = _SIDD_POSITIVE_POLARITY,
    SIDD_NEGATIVE_POLARITY = _SIDD_NEGATIVE_POLARITY,
    SIDD_MASKED_POSITIVE_POLARITY = _SIDD_MASKED_POSITIVE_POLARITY,
    SIDD_MASKED_NEGATIVE_POLARITY = _SIDD_MASKED_NEGATIVE_POLARITY;

const int32_t
    SIDD_LEAST_SIGNIFICANT = _SIDD_LEAST_SIGNIFICANT,
    SIDD_MOST_SIGNIFICANT = _SIDD_MOST_SIGNIFICANT;

const int32_t
    SIDD_BIT_MASK = _SIDD_BIT_MASK,
    SIDD_UNIT_MASK = _SIDD_UNIT_MASK;


TARGET("sse4.2")
bool sse4_2_cmpestra(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(64, _mm_cmpestra, imm8 & 0x3F, a, la, b, lb);
}

TARGET("sse4.2")
bool sse4_2_cmpestrc(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(64, _mm_cmpestrc, imm8 & 0x3F, a, la, b, lb);
}

TARGET("sse4.2")
int32_t sse4_2_cmpestri(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(128, _mm_cmpestri, imm8 & 0x7F, a, la, b, lb);
}

TARGET("sse4.2")
__m128i sse4_2_cmpestrm(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(128, _mm_cmpestrm, imm8 & 0x7F, a, la, b, lb);
}

TARGET("sse4.2")
bool sse4_2_cmpestro(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(64, _mm_cmpestro, imm8 & 0x3F, a, la, b, lb);
}

TARGET("sse4.2")
bool sse4_2_cmpestrs(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(2, _mm_cmpestrs, imm8, a, la, b, lb);
}

TARGET("sse4.2")
bool sse4_2_cmpestrz(const __m128i a, const int32_t la, const __m128i b, const int32_t lb, const uint8_t imm8) {
    DISPATCH(2, _mm_cmpestrz, imm8, a, la, b, lb);
}

TARGET("sse4.2")
__m128i sse4_2_cmpgt_epi64(const __m128i a, const __m128i b) {
    return _mm_cmpgt_epi64(a, b);
}

TARGET("sse4.2")
bool sse4_2_cmpistra(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(64, _mm_cmpistra, imm8 & 0x3F, a, b);
}

TARGET("sse4.2")
bool sse4_2_cmpistrc(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(64, _mm_cmpistrc, imm8 & 0x3F, a, b);
}

TARGET("sse4.2")
int32_t sse4_2_cmpistri(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(128, _mm_cmpistri, imm8 & 0x7F, a, b);
}

TARGET("sse4.2")
__m128i sse4_2_cmpistrm(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(128, _mm_cmpistrm, imm8 & 0x7F, a, b);
}

TARGET("sse4.2")
bool sse4_2_cmpistro(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(64, _mm_cmpistro, imm8 & 0x3F, a, b);
}

TARGET("sse4.2")
bool sse4_2_cmpistrs(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(2, _mm_cmpistrs, imm8 & 0x01, a, b);
}

TARGET("sse4.2")
bool sse4_2_cmpistrz(const __m128i a, const __m128i b, const uint8_t imm8) {
    DISPATCH(2, _mm_cmpistrz, imm8 & 0x01, a, b);
}

TARGET("sse4.2")
uint32_t sse4_2_crc32_u16(const uint32_t crc, const uint16_t v) {
    return _mm_crc32_u16(crc, v);
}

TARGET("sse4.2")
uint32_t sse4_2_crc32_u32(const uint32_t crc, const uint32_t v) {
    return _mm_crc32_u32(crc, v);
}

TARGET("sse4.2")
uint64_t sse4_2_crc32_u64(const uint64_t crc, const uint64_t v) {
    return _mm_crc32_u64(crc, v);
}

TARGET("sse4.2")
uint32_t sse4_2_crc32_u8(const uint32_t crc, const uint8_t v) {
    return _mm_crc32_u8(crc, v);
}

//endregion === SSE4.2 ===

#include <xmmintrin.h>

#include "sse.h"

__m128 sse_add_ps(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
__m128 sse_add_ss(__m128 a, __m128 b) { return _mm_add_ss(a, b); }
__m128 sse_and_ps(__m128 a, __m128 b) { return _mm_and_ps(a, b); }
__m128 sse_andnot_ps(__m128 a, __m128 b) { return _mm_andnot_ps(a, b); }
__m128 sse_cmpeq_ps(__m128 a, __m128 b) { return _mm_cmpeq_ps(a, b); }
__m128 sse_cmpeq_ss(__m128 a, __m128 b) { return _mm_cmpeq_ss(a, b); }
__m128 sse_cmpge_ps(__m128 a, __m128 b) { return _mm_cmpge_ps(a, b); }
__m128 sse_cmpge_ss(__m128 a, __m128 b) { return _mm_cmpge_ss(a, b); }
__m128 sse_cmpgt_ps(__m128 a, __m128 b) { return _mm_cmpgt_ps(a, b); }
__m128 sse_cmpgt_ss(__m128 a, __m128 b) { return _mm_cmpgt_ss(a, b); }
__m128 sse_cmple_ps(__m128 a, __m128 b) { return _mm_cmple_ps(a, b); }
__m128 sse_cmple_ss(__m128 a, __m128 b) { return _mm_cmple_ss(a, b); }
__m128 sse_cmplt_ps(__m128 a, __m128 b) { return _mm_cmplt_ps(a, b); }
__m128 sse_cmplt_ss(__m128 a, __m128 b) { return _mm_cmplt_ss(a, b); }
__m128 sse_cmpneq_ps(__m128 a, __m128 b) { return _mm_cmpneq_ps(a, b); }
__m128 sse_cmpneq_ss(__m128 a, __m128 b) { return _mm_cmpneq_ss(a, b); }
__m128 sse_cmpnge_ps(__m128 a, __m128 b) { return _mm_cmpnge_ps(a, b); }
__m128 sse_cmpnge_ss(__m128 a, __m128 b) { return _mm_cmpnge_ss(a, b); }
__m128 sse_cmpngt_ps(__m128 a, __m128 b) { return _mm_cmpngt_ps(a, b); }
__m128 sse_cmpngt_ss(__m128 a, __m128 b) { return _mm_cmpngt_ss(a, b); }
__m128 sse_cmpnle_ps(__m128 a, __m128 b) { return _mm_cmpnle_ps(a, b); }
__m128 sse_cmpnle_ss(__m128 a, __m128 b) { return _mm_cmpnle_ss(a, b); }
__m128 sse_cmpnlt_ps(__m128 a, __m128 b) { return _mm_cmpnlt_ps(a, b); }
__m128 sse_cmpnlt_ss(__m128 a, __m128 b) { return _mm_cmpnlt_ss(a, b); }
__m128 sse_cmpord_ps(__m128 a, __m128 b) { return _mm_cmpord_ps(a, b); }
__m128 sse_cmpord_ss(__m128 a, __m128 b) { return _mm_cmpord_ss(a, b); }
__m128 sse_cmpunord_ps(__m128 a, __m128 b) { return _mm_cmpunord_ps(a, b); }
__m128 sse_cmpunord_ss(__m128 a, __m128 b) { return _mm_cmpunord_ss(a, b); }
bool sse_comieq_ss(__m128 a, __m128 b) { return _mm_comieq_ss(a, b); }
bool sse_comige_ss(__m128 a, __m128 b) { return _mm_comige_ss(a, b); }
bool sse_comigt_ss(__m128 a, __m128 b) { return _mm_comigt_ss(a, b); }
bool sse_comile_ss(__m128 a, __m128 b) { return _mm_comile_ss(a, b); }
bool sse_comilt_ss(__m128 a, __m128 b) { return _mm_comilt_ss(a, b); }
bool sse_comineq_ss(__m128 a, __m128 b) { return _mm_comineq_ss(a, b); }
__m128 sse_cvt_si2ss(__m128 a, int32_t b) { return _mm_cvt_si2ss(a, b); }
int32_t sse_cvt_ss2si(__m128 a) { return _mm_cvt_ss2si(a); }
__m128 sse_cvtsi32_ss(__m128 a, int32_t b) { return _mm_cvtsi32_ss(a, b); }
__m128 sse_cvtsi64_ss(__m128 a, int64_t b) { return _mm_cvtsi64_ss(a, b); }
float sse_cvtss_f32(__m128 a) { return _mm_cvtss_f32(a); }
int32_t sse_cvtss_si32(__m128 a) { return _mm_cvtss_si32(a); }
int64_t sse_cvtss_si64(__m128 a) { return _mm_cvtss_si64(a); }
int32_t sse_cvtt_ss2si(__m128 a) { return _mm_cvtt_ss2si(a); }
int32_t sse_cvttss_si32(__m128 a) { return _mm_cvttss_si32(a); }
int64_t sse_cvttss_si64(__m128 a) { return _mm_cvttss_si64(a); }
__m128 sse_div_ps(__m128 a, __m128 b) { return _mm_div_ps(a, b); }
__m128 sse_div_ss(__m128 a, __m128 b) { return _mm_div_ss(a, b); }
void sse_free(void* mem_addr) { _mm_free(mem_addr); }
uint32_t sse_get_exception_mask() { return _MM_GET_EXCEPTION_MASK(); }
uint32_t sse_get_exception_state() { return _MM_GET_EXCEPTION_STATE(); }
uint32_t sse_get_flush_zero_mode() { return _MM_GET_FLUSH_ZERO_MODE(); }
uint32_t sse_get_rounding_mode() { return _MM_GET_ROUNDING_MODE(); }
uint32_t sse_getcsr(void) { return _mm_getcsr(); }
__m128 sse_load_ps(const float* mem_addr) { return _mm_load_ps(mem_addr); }
__m128 sse_load_ps1(const float* mem_addr) { return _mm_load_ps1(mem_addr); }
__m128 sse_load_ss(const float* mem_addr) { return _mm_load_ss(mem_addr); }
__m128 sse_load1_ps(const float* mem_addr) { return _mm_load1_ps(mem_addr); }
__m128 sse_loadr_ps(const float* mem_addr) { return _mm_loadr_ps(mem_addr); }
__m128 sse_loadu_ps(const float* mem_addr) { return _mm_loadu_ps(mem_addr); }
void* sse_malloc(size_t size, size_t align) { return _mm_malloc(size, align); }
__m128 sse_max_ps(__m128 a, __m128 b) { return _mm_max_ps(a, b); }
__m128 sse_max_ss(__m128 a, __m128 b) { return _mm_max_ss(a, b); }
__m128 sse_min_ps(__m128 a, __m128 b) { return _mm_min_ps(a, b); }
__m128 sse_min_ss(__m128 a, __m128 b) { return _mm_min_ss(a, b); }
__m128 sse_move_ss(__m128 a, __m128 b) { return _mm_move_ss(a, b); }
__m128 sse_movehl_ps(__m128 a, __m128 b) { return _mm_movehl_ps(a, b); }
__m128 sse_movelh_ps(__m128 a, __m128 b) { return _mm_movelh_ps(a, b); }
int32_t sse_movemask_ps(__m128 a) { return _mm_movemask_ps(a); }
__m128 sse_mul_ps(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }
__m128 sse_mul_ss(__m128 a, __m128 b) { return _mm_mul_ss(a, b); }
__m128 sse_or_ps(__m128 a, __m128 b) { return _mm_or_ps(a, b); }
namespace prefetch {
	typedef void(*function)(const int8_t*);
	const function map[4] {
		[](const int8_t* p) { return _mm_prefetch((const char *)p, 0); },
		[](const int8_t* p) { return _mm_prefetch((const char *)p, 1); },
		[](const int8_t* p) { return _mm_prefetch((const char *)p, 2); },
		[](const int8_t* p) { return _mm_prefetch((const char *)p, 3); },
	};
} // namespace prefetch
void sse_prefetch(const int8_t* p, int32_t i) { prefetch::map[i & 0x3](p); }
__m128 sse_rcp_ps(__m128 a) { return _mm_rcp_ps(a); }
__m128 sse_rcp_ss(__m128 a) { return _mm_rcp_ss(a); }
__m128 sse_rsqrt_ps(__m128 a) { return _mm_rsqrt_ps(a); }
__m128 sse_rsqrt_ss(__m128 a) { return _mm_rsqrt_ss(a); }
void sse_set_exception_mask(uint32_t a) { _MM_SET_EXCEPTION_MASK(a); }
void sse_set_exception_state(uint32_t a) { _MM_SET_EXCEPTION_STATE(a); }
void sse_set_flush_zero_mode(uint32_t a) { _MM_SET_FLUSH_ZERO_MODE(a); }
__m128 sse_set_ps(float e3, float e2, float e1, float e0) { return _mm_set_ps(e3, e2, e1, e0); }
__m128 sse_set_ps1(float a) { return _mm_set_ps1(a); }
void sse_set_rounding_mode(uint32_t a) { _MM_SET_ROUNDING_MODE(a); }
__m128 sse_set_ss(float a) { return _mm_set_ss(a); }
__m128 sse_set1_ps(float a) { return _mm_set1_ps(a); }
void sse_setcsr(uint32_t a) { _mm_setcsr(a); }
__m128 sse_setr_ps(float e3, float e2, float e1, float e0) { return _mm_setr_ps(e3, e2, e1, e0); }
__m128 sse_setzero_ps(void) { return _mm_setzero_ps(); }
void sse_sfence(void) { _mm_sfence(); }
namespace shuffle_ps {
	typedef __m128(*function)(__m128, __m128);
	const function map[256] {
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x00); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x01); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x02); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x03); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x04); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x05); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x06); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x07); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x08); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x09); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x0A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x0B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x0C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x0D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x0E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x0F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x10); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x11); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x12); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x13); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x14); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x15); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x16); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x17); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x18); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x19); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x1A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x1B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x1C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x1D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x1E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x1F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x20); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x21); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x22); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x23); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x24); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x25); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x26); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x27); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x28); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x29); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x2A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x2B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x2C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x2D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x2E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x2F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x30); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x31); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x32); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x33); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x34); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x35); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x36); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x37); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x38); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x39); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x3A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x3B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x3C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x3D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x3E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x3F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x40); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x41); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x42); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x43); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x44); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x45); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x46); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x47); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x48); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x49); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x4A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x4B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x4C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x4D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x4E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x4F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x50); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x51); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x52); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x53); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x54); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x55); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x56); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x57); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x58); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x59); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x5A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x5B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x5C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x5D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x5E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x5F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x60); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x61); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x62); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x63); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x64); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x65); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x66); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x67); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x68); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x69); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x6A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x6B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x6C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x6D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x6E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x6F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x70); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x71); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x72); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x73); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x74); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x75); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x76); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x77); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x78); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x79); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x7A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x7B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x7C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x7D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x7E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x7F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x80); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x81); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x82); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x83); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x84); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x85); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x86); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x87); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x88); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x89); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x8A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x8B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x8C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x8D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x8E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x8F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x90); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x91); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x92); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x93); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x94); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x95); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x96); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x97); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x98); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x99); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x9A); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x9B); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x9C); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x9D); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x9E); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0x9F); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA0); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA1); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA2); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA3); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA4); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA5); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA6); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA7); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA8); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xA9); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xAA); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xAB); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xAC); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xAD); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xAE); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xAF); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB0); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB1); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB2); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB3); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB4); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB5); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB6); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB7); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB8); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xB9); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xBA); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xBB); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xBC); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xBD); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xBE); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xBF); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC0); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC1); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC2); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC3); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC4); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC5); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC6); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC7); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC8); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xC9); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xCA); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xCB); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xCC); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xCD); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xCE); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xCF); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD0); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD1); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD2); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD3); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD4); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD5); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD6); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD7); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD8); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xD9); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xDA); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xDB); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xDC); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xDD); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xDE); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xDF); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE0); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE1); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE2); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE3); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE4); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE5); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE6); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE7); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE8); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xE9); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xEA); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xEB); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xEC); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xED); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xEE); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xEF); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF0); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF1); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF2); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF3); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF4); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF5); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF6); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF7); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF8); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xF9); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xFA); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xFB); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xFC); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xFD); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xFE); },
		[](__m128 a, __m128 b) { return _mm_shuffle_ps(a, b, 0xFF); },
	};
} // namespace shuffle_ps
__m128 sse_shuffle_ps(__m128 a, __m128 b, uint8_t imm8) { return shuffle_ps::map[imm8](a, b); }
__m128 sse_sqrt_ps(__m128 a) { return _mm_sqrt_ps(a); }
__m128 sse_sqrt_ss(__m128 a) { return _mm_sqrt_ss(a); }
void sse_store_ps(float* mem_addr, __m128 a) { _mm_store_ps(mem_addr, a); }
void sse_store_ps1(float* mem_addr, __m128 a) { _mm_store_ps1(mem_addr, a); }
void sse_store_ss(float* mem_addr, __m128 a) { _mm_store_ss(mem_addr, a); }
void sse_store1_ps(float* mem_addr, __m128 a) { _mm_store1_ps(mem_addr, a); }
void sse_storer_ps(float* mem_addr, __m128 a) { _mm_storer_ps(mem_addr, a); }
void sse_storeu_ps(float* mem_addr, __m128 a) { _mm_storeu_ps(mem_addr, a); }
void sse_stream_ps(float* mem_addr, __m128 a) { _mm_stream_ps(mem_addr, a); }
__m128 sse_sub_ps(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }
__m128 sse_sub_ss(__m128 a, __m128 b) { return _mm_sub_ss(a, b); }
void sse_transpose4_ps(__m128 row0, __m128 row1, __m128 row2, __m128 row3) { _MM_TRANSPOSE4_PS(row0, row1, row2, row3); }
bool sse_ucomieq_ss(__m128 a, __m128 b) { return _mm_ucomieq_ss(a, b); }
bool sse_ucomige_ss(__m128 a, __m128 b) { return _mm_ucomige_ss(a, b); }
bool sse_ucomigt_ss(__m128 a, __m128 b) { return _mm_ucomigt_ss(a, b); }
bool sse_ucomile_ss(__m128 a, __m128 b) { return _mm_ucomile_ss(a, b); }
bool sse_ucomilt_ss(__m128 a, __m128 b) { return _mm_ucomilt_ss(a, b); }
bool sse_ucomineq_ss(__m128 a, __m128 b) { return _mm_ucomineq_ss(a, b); }
__m128 sse_undefined_ps(void) { return _mm_undefined_ps(); }
__m128 sse_unpackhi_ps(__m128 a, __m128 b) { return _mm_unpackhi_ps(a, b); }
__m128 sse_unpacklo_ps(__m128 a, __m128 b) { return _mm_unpacklo_ps(a, b); }
__m128 sse_xor_ps(__m128 a, __m128 b) { return _mm_xor_ps(a, b); }

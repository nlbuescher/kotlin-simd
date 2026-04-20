#ifndef SSE4_2_H
#define SSE4_2_H

#include <stdbool.h>
#include <stdint.h>

extern const bool SSE4_2;

#ifndef __m128i
typedef long long __m128i __attribute__((__vector_size__(16), __may_alias__));
#endif

#ifdef __cplusplus
extern "C" {
#endif

// These macros specify the source data format
const int32_t SIDD_UBYTE_OPS = 0x00;
const int32_t SIDD_UWORD_OPS = 0x01;
const int32_t SIDD_SBYTE_OPS = 0x02;
const int32_t SIDD_SWORD_OPS = 0x03;

// These macros specify the comparison operation
const int32_t SIDD_CMP_EQUAL_ANY     = 0x00;
const int32_t SIDD_CMP_RANGES        = 0x04;
const int32_t SIDD_CMP_EQUAL_EACH    = 0x08;
const int32_t SIDD_CMP_EQUAL_ORDERED = 0x0C;

// These macros specify the polarity
const int32_t SIDD_POSITIVE_POLARITY        = 0x00;
const int32_t SIDD_NEGATIVE_POLARITY        = 0x10;
const int32_t SIDD_MASKED_POSITIVE_POLARITY = 0x20;
const int32_t SIDD_MASKED_NEGATIVE_POLARITY = 0x30;

// These macros specify the output selection in _mm_cmpXstri()
const int32_t SIDD_LEAST_SIGNIFICANT = 0x00;
const int32_t SIDD_MOST_SIGNIFICANT  = 0x40;

// These macros specify the output selection in _mm_cmpXstrm()
const int32_t SIDD_BIT_MASK  = 0x00;
const int32_t SIDD_UNIT_MASK = 0x40;

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

#endif // SSE4_2_H

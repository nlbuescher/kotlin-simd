package dev.buescher.kotlin.simd

import kotlinx.cinterop.*
import platform.simd.*
import kotlin.test.*

@Test
fun `that abstraction works`() {
	println(
		"""
		SSE:    $SSE
		SSE2:   $SSE2
		SSE3:   $SSE3
		SSSE3:  $SSSE3
		SSE4.1: $SSE4_1
		SSE4.2: $SSE4_2
		""".trimIndent(),
	)

	assertEquals(Float4(3f), Float4(1f) + Float4(2f), "plus")
	assertEquals(Float4(-1f), Float4(1f) - Float4(2f), "minus")
	assertEquals(Float4(2f), Float4(1f) * Float4(2f), "times")
	assertEquals(Float4(.5f), Float4(1f) / Float4(2f), "div")
	assertEquals(Float4(1f), Float4(1f) % Float4(2f), "rem")
	assertEquals(Float4(2f), round(Float4(1.5f)), "round")
	assertEquals(Float4(1f), floor(Float4(1.5f)), "floor")
	assertEquals(Float4(-1f), truncate(Float4(-1.5f)), "truncate")
}

@Test
fun `that intrinsics are enabled`() {
	assertTrue(SSE)
	assertTrue(SSE2)
	assertTrue(SSE3)
	assertTrue(SSSE3)
	assertTrue(SSE4_1)
	assertTrue(SSE4_2)
}

@Test
fun `that calling sse_add_ps directly works`() {
	val input1 = vectorOf(1f, 2f, 3f, 4f)
	val input2 = vectorOf(2f, 4f, 6f, 8f)
	val expected = vectorOf(3f, 6f, 9f, 12f)

	val actual = sse_add_ps(input1, input2)

	assertEquals(expected, actual)
}

@Test
fun `that calling sse4_1_round_ps directly works`() {
	val input = vectorOf(2.2f, 4.4f, 6.6f, 8.8f)
	val expected = vectorOf(2f, 4f, 7f, 9f)

	val actual = sse4_1_round_ps(input, SSE4_1_FROUND_TO_NEAREST_INT)

	assertEquals(expected, actual)
}

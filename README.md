# Kotlin SIMD

[![Maven Central Version](https://img.shields.io/maven-central/v/dev.buescher/kotlin-simd?logo=apachemaven)](https://central.sonatype.com/search?namespace=dev.buescher&name=kotlin-simd)
[![Build](https://github.com/nlbuescher/kotlin-simd/actions/workflows/build.yaml/badge.svg)](https://github.com/nlbuescher/kotlin-simd/actions/workflows/build.yaml)

Exposes SSE, SSE2, SSE3, SSSE3, SSE4.1, and SSE4.2 compiler intrincs to Kotlin/Native as functions.

SSE Support can be checked by using the boolean constants defined by the library (simply called SSE, SSE2, etc). This allows changing implementation at runtime based on hardware SSE support:

```kotlin
fun round(x: Vector128): Vector128 = when {
	SSE4_1 -> sse4_1_round_ps(x, SSE4_1_FROUND_NEARBYINT)
	else -> {
		vectorOf(
			round(x.getFloatAt(0)),
			round(x.getFloatAt(1)),
			round(x.getFloatAt(2)),
			round(x.getFloatAt(3)),
		)
	}
}
```

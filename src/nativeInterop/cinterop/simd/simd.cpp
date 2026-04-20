// CPUID

#include <cpuid.h>

inline void cpuid(int info[4], int x) {
	__cpuid_count(x, 0, info[0], info[1], info[2], info[3]);
}

// INTRINSICS

bool SSE, SSE2, SSE3, SSSE3, SSE4_1, SSE4_2;

bool loadIntrinsics() {
	int info[4];
	cpuid(info, 1);

	SSE = (info[3] & bit_SSE) != 0;
	SSE2 = (info[3] & bit_SSE2) != 0;
	SSE3 = (info[2] & bit_SSE3) != 0;
	SSSE3 = (info[2] & bit_SSSE3) != 0;
	SSE4_1 = (info[2] & bit_SSE4_1) != 0;
	SSE4_2 = (info[2] & bit_SSE4_2) != 0;

	return true;
}

bool unused = loadIntrinsics();

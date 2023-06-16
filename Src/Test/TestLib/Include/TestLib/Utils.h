#ifndef TEST_TESTLIB_UTILS__H
#define TEST_TESTLIB_UTILS__H
#include <algorithm>
#include <optix.h>
namespace TestLib
{
	inline auto max(const OptixStackSizes& v1, const OptixStackSizes& v2) noexcept -> OptixStackSizes
	{
		OptixStackSizes res = {};
		res.cssRG = std::max(v1.cssRG, v2.cssRG);
		res.cssMS = std::max(v1.cssMS, v2.cssMS);
		res.cssCH = std::max(v1.cssCH, v2.cssCH);
		res.cssAH = std::max(v1.cssAH, v2.cssAH);
		res.cssIS = std::max(v1.cssIS, v2.cssIS);
		res.cssCC = std::max(v1.cssCC, v2.cssCC);
		return res;
	}
	template <typename T>
	inline constexpr auto compute_aligned_size(T size, T alignment) -> T
	{
		return ((size + alignment - static_cast<T>(1)) / alignment) * alignment;
	}
}
#endif

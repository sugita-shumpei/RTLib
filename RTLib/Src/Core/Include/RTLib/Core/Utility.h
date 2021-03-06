#ifndef RTLIB_CORE_UTILITY_H
#define RTLIB_CORE_UTILITY_H
#include <cstdint>
namespace RTLib {
	namespace Core {
		namespace Utility{
			inline constexpr auto Log2(uint64_t v)->uint32_t
			{
				uint32_t val = 0;
				v /= 2;
				while (v > 0) {
					v /= 2;
					val++;
				};
				return val;
			}
			inline constexpr auto GetAlignmentSize(size_t size, size_t alignment)-> size_t {
				return ((size + alignment - 1) / alignment) * alignment;
			}
		}
	}
}
#endif
#ifndef RTLIB_CORE_DATA_TYPES__H
#define RTLIB_CORE_DATA_TYPES__H

#ifndef __CUDACC__

#include <string>
#include <cstdint>
#include <Imath/half.h>

#endif

namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		using Int8    = int8_t ;
		using Int16   = int16_t;
		using Int32   = int32_t;
		using Int64   = int64_t;

		using UInt8   = uint8_t ;
		using UInt16  = uint16_t;
		using UInt32  = uint32_t;
		using UInt64  = uint64_t;
		
		using Float16 = half;
#else
		using Int8    = signed char;
		using Int16   = short;
		using Int32   = int;
		using Int64   = long long;

		using UInt8   = unsigned char;
		using UInt16  = unsigned short;
		using UInt32  = unsigned int;
		using UInt64  = unsigned long long;

		using Float16 = unsigned short;
#endif

		using Float32 = float;
		using Float64 = double;

		using Char = char;
		using Byte = unsigned char;
		using Bool = bool;

		using CStr = const char*;

#ifndef __CUDACC__
		using String = std::string;
#endif
	}
}
#endif


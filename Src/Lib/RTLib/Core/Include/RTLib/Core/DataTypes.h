#ifndef RTLIB_CORE_DATA_TYPES__H
#define RTLIB_CORE_DATA_TYPES__H

#ifndef __CUDACC__
#include <Imath/half.h>
#include <string>
#else

#endif
namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		using Float16 = Imath::half;
#else
		using Float16 = unsigned short;
#endif
		using Float32 = float;
		using Float64 = double;

		using UInt8  = unsigned char;
		using UInt16 = unsigned short;
		using UInt32 = unsigned int;
		using UInt64 = unsigned long long;

		using Int8   = signed char;
		using Int16  = short;
		using Int32  = int;
		using Int64  = long long;

		using Bool   = bool;
		using Byte   = unsigned short;
		using Char   = char;

		using WChar  = wchar_t;

#ifndef __CUDACC__
		using WString = std::wstring;
		using String  = std::string;
#else
#endif


	}
}
#endif

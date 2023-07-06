#ifndef RTLIB_CORE_UUID__H
#define RTLIB_CORE_UUID__H
#ifndef __CUDACC__
#include <uuid.h>
namespace RTLib
{
	inline namespace Core
	{
		using Uuid = uuids::uuid;
	}
}
#endif
#endif

#ifndef RTLIB_CORE_OBJECT_TYPE_ID__H
#define RTLIB_CORE_OBJECT_TYPE_ID__H
#ifndef __CUDACC__

#include <RTLib/Core/Uuid.h>
#define RTLIB_CORE_DEFINE_OBJECT_TYPE_ID(TYPE, TYPE_ID_STR)	\
	static inline constexpr ObjectTypeID ObjectTypeID_##TYPE = RTLib::Core::Uuid::from_string(TYPE_ID_STR).value_or(RTLib::Core::Uuid{}); \
	struct TYPE; \
	template<> struct ObjectTraits<TYPE> { static inline constexpr ObjectTypeID typeID = ObjectTypeID_##TYPE; }

#define RTLIB_CORE_DEFINE_OBJECT_TYPE_ID_2(TYPE, TYPE_ID_BASE, TYPE_ID_STR) \
	static inline constexpr ObjectTypeID ObjectTypeID_##TYPE_ID_BASE = RTLib::Core::Uuid::from_string(TYPE_ID_STR).value_or(RTLib::Core::Uuid{}); \
	struct TYPE; \
	template<> struct ObjectTraits<TYPE> { static inline constexpr ObjectTypeID typeID = ObjectTypeID_##TYPE_ID_BASE; }

namespace RTLib
{
	inline namespace Core
	{

		template<typename T>
		struct ObjectTraits;

		using  ObjectTypeID = Uuid;
	}
}

#endif
#endif

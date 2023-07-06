#ifndef RTLIB_CORE_OBJECT__H
#define RTLIB_CORE_OBJECT__H

#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/ObjectTypeID.h>

#ifndef __CUDACC__
#include <atomic>
#include <memory>
namespace RTLib
{
	inline namespace Core
	{
		static inline constexpr auto ObjectTypeID_Unknown = ObjectTypeID{};

	}
	namespace Core
	{
		struct Object;
		template<> struct ObjectTraits<Object> { static inline constexpr ObjectTypeID typeID = ObjectTypeID_Unknown; };

		struct Object : public std::enable_shared_from_this<Object>
		{
			using TypeID = ObjectTypeID;

			virtual ~Object() noexcept {}

			virtual auto query_object(const TypeID& typeID) -> std::shared_ptr<Object> = 0;
			// Type ID
			virtual auto get_type_id() const noexcept -> TypeID = 0;
			// Name
			virtual auto get_name() const noexcept -> String = 0;
		};
	}
}
#endif
#endif

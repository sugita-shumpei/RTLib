#ifndef RTLIB_CORE_RT_SERIALIZED_H
#define RTLIB_CORE_RT_SERIALIZED_H
#include <RTLib/Core/RTDataType.h>
#include <RTLib/Core/RTUuid.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <unordered_map>
#include <utility>
namespace RTLib
{
	namespace Core
	{
		enum class RTSerializedType:unsigned int
		{
			eBool       = static_cast<unsigned int>(RTValueType::eBool   ),
			eInt        = static_cast<unsigned int>(RTValueType::eSInt32 ),
			eFloat      = static_cast<unsigned int>(RTValueType::eFloat32),
			eString     = static_cast<unsigned int>(RTValueType::eString ),
			eArray     ,
			eSerialized,
		};

		class RTSerialized
		{
		public:
			RTSerialized()          noexcept {}
			virtual ~RTSerialized() noexcept {}

			virtual auto TypeString()const noexcept -> RTString = 0;
			virtual auto TypeUUID  ()const noexcept -> RTUuid   = 0;

			virtual void   Serialize(      nlohmann::json& json)const = 0;
			virtual void Deserialize(const nlohmann::json& json)      = 0;
		}; 
		
		using RTSerializedPtr = std::shared_ptr<RTSerialized>;

		class RTSerializedParser
		{
		public:
			void SetVerifier() noexcept;

		};

	}
}
#undef RTLIB_CORE_RT_SERIALIZED_RT_SERIALIZED_MAP_IMPL_GETTER_FOR
#undef RTLIB_CORE_RT_SERIALIZED_RT_SERIALIZED_MAP_IMPL_APPEND_FOR
#undef RTLIB_CORE_RT_SERIALIZED_RT_SERIALIZED_MAP_IMPL_INSERT_FOR
#endif

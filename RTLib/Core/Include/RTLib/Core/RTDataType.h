#ifndef RTLIB_CORE_RT_DATA_TYPE_H
#define RTLIB_CORE_RT_DATA_TYPE_H
#include <cstdint>
#include <string>
#include <Imath/half.h>
namespace RTLib
{
	namespace Core {
		enum class RTValueType:unsigned int
		{
			eBool,

			eUInt8,
			eUInt16,
			eUInt32,
			eUInt64,

			eSInt8,
			eSInt16,
			eSInt32,
			eSInt64,

			eFloat16,
			eFloat32,
			eFloat64,

			eString,
		};

		using RTBool    = bool;
		using RTUInt8   = std::uint8_t;
		using RTUInt16  = std::uint16_t;
		using RTUInt32  = std::uint32_t;
		using RTUInt64  = std::uint64_t;
		using RTSInt8   = std::int8_t;
		using RTSInt16  = std::int16_t;
		using RTSInt32  = std::int32_t;
		using RTSInt64  = std::int64_t;
		using RTFloat16 = Imath::half;
		using RTFloat32 = float;
		using RTFloat64 = double;
		using RTString  = std::string;

		template<typename T>
		struct RTValueTypeTraits;

		template<> struct RTValueTypeTraits<RTBool>   { static inline constexpr auto value = RTValueType::eBool;   };
		template<> struct RTValueTypeTraits<RTUInt8>  { static inline constexpr auto value = RTValueType::eUInt8;  };
		template<> struct RTValueTypeTraits<RTUInt16> { static inline constexpr auto value = RTValueType::eUInt16; };
		template<> struct RTValueTypeTraits<RTUInt32> { static inline constexpr auto value = RTValueType::eUInt32; };
		template<> struct RTValueTypeTraits<RTUInt64> { static inline constexpr auto value = RTValueType::eUInt64; };
		template<> struct RTValueTypeTraits<RTSInt8>  { static inline constexpr auto value = RTValueType::eSInt8;  };
		template<> struct RTValueTypeTraits<RTSInt16> { static inline constexpr auto value = RTValueType::eSInt16; };
		template<> struct RTValueTypeTraits<RTSInt32> { static inline constexpr auto value = RTValueType::eSInt32; };
		template<> struct RTValueTypeTraits<RTSInt64> { static inline constexpr auto value = RTValueType::eSInt64; };
		template<> struct RTValueTypeTraits<RTFloat16> { static inline constexpr auto value = RTValueType::eFloat16; };
		template<> struct RTValueTypeTraits<RTFloat32> { static inline constexpr auto value = RTValueType::eFloat32; };
		template<> struct RTValueTypeTraits<RTFloat64> { static inline constexpr auto value = RTValueType::eFloat64; };
		template<> struct RTValueTypeTraits<RTString>  { static inline constexpr auto value = RTValueType::eString; };
	}
}
#endif

#ifndef RTLIB_CORE_MATH__H
#define RTLIB_CORE_MATH__H
#include <RTLib/Core/DataTypes.h>
#ifndef __CUDACC__
#include <Imath/ImathPlatform.h>
#include <type_traits>
#include <cmath>
#endif
namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		template <typename T> auto sqrt(T v) -> T {
			if constexpr (std::is_same_v<T, Float16>)
			{
				return static_cast<Float16>(std::sqrtf(static_cast<Float32>(v)));
			}
			else {
				return std::sqrt(v);
			}
		}
		template <typename T> auto asin(T v) -> T {
			if constexpr (std::is_same_v<T, Float16>)
			{
				return static_cast<Float16>(std::acos(static_cast<Float32>(v)));
			}
			else {
				return std::asin(v);
			}
		}
		template <typename T> auto acos(T v) -> T {
			if constexpr (std::is_same_v<T, Float16>)
			{
				return static_cast<Float16>(std::acos(static_cast<Float32>(v)));
			}
			else {
				return std::acos(v);
			}
		}
		template <typename T> auto atan(T v) -> T {
			if constexpr (std::is_same_v<T, Float16>)
			{
				return static_cast<Float16>(std::atan(static_cast<Float32>(v)));
			}
			else {
				return std::atan(v);
			}
		}
#endif
		constexpr auto radians(Float32 v) -> Float32
		{
			return v * static_cast<Float32>(M_PI) / 180.0f;
		}
		constexpr auto radians(Float64 v) -> Float64
		{
			return v * static_cast<Float64>(M_PI) / 180.0;
		}

		constexpr auto degrees(Float32 v) -> Float64
		{
			return v * 180.0f/ static_cast<Float32>(M_PI);
		}
		constexpr auto degrees(Float64 v) -> Float64
		{
			return v * 180.0 / static_cast<Float64>(M_PI);
		}
	}
}
#endif

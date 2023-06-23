#ifndef RTLIB_CORE_VECTOR__H
#define RTLIB_CORE_VECTOR__H
namespace RTLib
{
	namespace Core
	{
		template <typename T> struct Vector2;
		template <typename T> struct Vector3;
		template <typename T> struct Vector4;
		
		template <typename T> struct Vector2
		{
			constexpr Vector2() noexcept;
			constexpr Vector2(const Vector2&) noexcept = default;
			constexpr Vector2& operator=(const Vector2&) noexcept = default;

			constexpr Vector2();

			T x;
			T y;
		};
		template <typename T> struct Vector3
		{
			constexpr Vector3() noexcept;
			constexpr Vector3(const Vector3&) noexcept = default;
			constexpr Vector3& operator=(const Vector3&) noexcept = default;

			constexpr Vector3();

			T x;
			T y;
			T z;
		};
		template <typename T> struct Vector4
		{
			constexpr Vector4() noexcept;
			constexpr Vector4(const Vector4&) noexcept = default;
			constexpr Vector4& operator=(const Vector4&) noexcept = default;

			constexpr Vector4();

			T x;
			T y;
			T z;
			T w;
		};
	}
}

#endif
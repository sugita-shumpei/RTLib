#ifndef RTLIB_CORE_VECTOR__H
#define RTLIB_CORE_VECTOR__H
#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/Math.h>
#ifndef __CUDACC__
#include <type_traits>
#include <cmath>
#include <numeric>
#endif
namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		template <typename T> struct Vector2;
		template <typename T> struct Vector3;
		template <typename T> struct Vector4;

		template <typename T> struct Vector2 {
			using value_type = T;
			constexpr Vector2() noexcept : x{ static_cast<T>(0) }, y{ static_cast<T>(0) } {}
			constexpr Vector2(const Vector2&) noexcept = default;
			constexpr Vector2& operator=(const Vector2&) noexcept = default;

			constexpr Vector2(T x_, T y_) noexcept :x{ x_ }, y{ y_ } {}
			constexpr explicit Vector2(T v_) noexcept :x{ v_ }, y{ v_ } {}

			template<typename U    > constexpr explicit Vector2(const Vector2<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) } {}
			template<typename U = T> constexpr explicit Vector2(const Vector3<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) } {}
			template<typename U = T> constexpr explicit Vector2(const Vector4<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) } {}

			constexpr Vector2 operator+()const noexcept { return *this; }
			constexpr Vector2 operator-()const noexcept { return Vector2(-x,-y); }

			constexpr Vector2& operator+=(const Vector2& v) noexcept { x += v.x; y += v.y; return *this; }
			constexpr Vector2& operator-=(const Vector2& v) noexcept { x -= v.x; y -= v.y; return *this; }
			constexpr Vector2& operator*=(const Vector2& v) noexcept { x *= v.x; y *= v.y; return *this; }
			constexpr Vector2& operator/=(const Vector2& v) noexcept { x /= v.x; y /= v.y; return *this; }
			constexpr Vector2& operator*=(T v) noexcept { x *= v; y *= v; return *this; }
			constexpr Vector2& operator/=(T v) noexcept { x /= v; y /= v; return *this; }

			constexpr Bool operator==(const Vector2& v)const noexcept {
				if constexpr (std::is_same_v<T, Float16> || std::is_same_v<T, Float32> || std::is_same_v<T, Float64>)
				{
					constexpr Float32 eps = std::is_same_v<T, Float16> ? 1e-6f : 1e-10f;
					auto len0 = this->length_sqr();
					auto len1 = v.length_sqr();
					auto len2 = Vector2(*this - v).length_sqr();
					auto len_min = len0 > len1 ? len1 : len0;
					return len2 <= len_min * eps;
				}
				else
				{
					return (x == v.x) && (y == v.y) && (z == v.z) && (w == v.w);
				}
			}
			constexpr Bool operator!=(const Vector2& v)const noexcept { return !(*this == v); }

			constexpr value_type dot(const Vector2& v) const noexcept { return x * v.x + y * v.y; }
			constexpr value_type length_sqr() const noexcept { return dot(*this); }
			value_type length() const noexcept { return RTLib::Core::sqrt(length_sqr()); }

			value_type angle_rad(const Vector2& v)const noexcept { return RTLib::Core::acos(dot(v) / (length() * v.length())); }
			value_type angle_deg(const Vector2& v)const noexcept { return degrees(angle_rad(v)); }

			Vector2& normalized() noexcept { *this /= length(); return *this; }
			Vector2  normalize() const noexcept {
				return Vector2(*this).normalized();
			}

			value_type x;
			value_type y;
		};
		template <typename T> struct Vector3 {
			using value_type = T;
			constexpr Vector3() noexcept : x{ static_cast<T>(0) }, y{ static_cast<T>(0) }, z{ static_cast<T>(0) } {}
			constexpr Vector3(const Vector3&) noexcept = default;
			constexpr Vector3& operator=(const Vector3&) noexcept = default;

			constexpr Vector3(T x_, T y_, T z_ = static_cast<T>(0)) noexcept :x{ x_ }, y{ y_ }, z{ z_ } {}
			constexpr explicit Vector3(T v_) noexcept :x{ v_ }, y{ v_ }, z{ v_ } {}

			template<typename U = T> constexpr explicit Vector3(const Vector2<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) }, z{ static_cast<T>(0)    } {}
			template<typename U    > constexpr explicit Vector3(const Vector3<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) }, z{ static_cast<T>(v.z)  } {}
			template<typename U = T> constexpr explicit Vector3(const Vector4<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) }, z{ static_cast<T>(v.z)  } {}

			constexpr Vector3(T v0_, const Vector2<T>& v1_) noexcept : Vector3(v0_, v1_.x, v1_.y) {}
			constexpr Vector3(const Vector2<T>& v0_, T v1_) noexcept : Vector3(v0_.x, v0_.y, v1_) {}

			constexpr Vector3 operator+()const noexcept { return *this; }
			constexpr Vector3 operator-()const noexcept { return Vector3(-x, -y,-z); }

			constexpr Vector3& operator+=(const Vector3& v) noexcept { x += v.x; y += v.y; z += v.z; return *this; }
			constexpr Vector3& operator-=(const Vector3& v) noexcept { x -= v.x; y -= v.y; z -= v.z; return *this; }
			constexpr Vector3& operator*=(const Vector3& v) noexcept { x *= v.x; y *= v.y; z *= v.z; return *this; }
			constexpr Vector3& operator/=(const Vector3& v) noexcept { x /= v.x; y /= v.y; z /= v.z; return *this; }
			constexpr Vector3& operator*=(T v) noexcept { x *= v; y *= v; z *= v; return *this; }
			constexpr Vector3& operator/=(T v) noexcept { x /= v; y /= v; z /= v; return *this; }

			constexpr Bool operator==(const Vector3& v)const noexcept {
				if constexpr (std::is_same_v<T, Float16> || std::is_same_v<T, Float32> || std::is_same_v<T, Float64>)
				{
					constexpr Float32 eps = std::is_same_v<T, Float16> ? 1e-6f : 1e-10f;
					auto len0 = this->length_sqr();
					auto len1 = v.length_sqr();
					auto len2 = Vector3(*this - v).length_sqr();
					auto len_min = len0 > len1 ? len1 : len0;
					return len2 <= len_min * eps;
				}
				else
				{
					return (x == v.x) && (y == v.y) && (z == v.z) && (w == v.w);
				}
			}
			constexpr Bool operator!=(const Vector3& v)const noexcept { return !(*this == v); }

			constexpr value_type dot(const Vector3& v) const noexcept { return x * v.x + y * v.y + z * v.z; }
			constexpr value_type length_sqr() const noexcept { return dot(*this); }
			value_type length() const noexcept { return RTLib::Core::sqrt(length_sqr()); }
			constexpr Vector3 cross(const Vector3& v) const noexcept {
				return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
			}

			value_type angle_rad(const Vector3& v)const noexcept { return RTLib::Core::acos(dot(v) / (length() * v.length())); }
			value_type angle_deg(const Vector3& v)const noexcept { return degrees(angle_rad(v)); }

			Vector3& normalized() noexcept { *this /= length(); return *this; }
			Vector3  normalize() const noexcept {
				return Vector3(*this).normalized();
			}

			value_type x;
			value_type y;
			value_type z;
		};
		template <typename T> struct Vector4 {
			using value_type = T;
			constexpr Vector4() noexcept : x{ static_cast<T>(0) }, y{ static_cast<T>(0) }, z{ static_cast<T>(0) }, w{ static_cast<T>(0) } {}
			constexpr Vector4(const Vector4&) noexcept = default;
			constexpr Vector4& operator=(const Vector4&) noexcept = default;

			constexpr Vector4(T x_, T y_, T z_ = static_cast<T>(0), T w_ = static_cast<T>(0)) noexcept :x{ x_ }, y{ y_ }, z{ z_ }, w{ w_ } {}
			constexpr explicit Vector4(T v_) noexcept :x{ v_ }, y{ v_ }, z{ v_ }, w{ v_ } {}

			template<typename U = T> constexpr explicit Vector4(const Vector2<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) }, z{ static_cast<T>(v.z) }, w{ static_cast<T>(0) } {}
			template<typename U = T> constexpr explicit Vector4(const Vector3<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) }, z{ static_cast<T>(v.z) }, w{ static_cast<T>(v.w) } {}
			template<typename U    > constexpr explicit Vector4(const Vector4<U>& v)noexcept :x{ static_cast<T>(v.x) }, y{ static_cast<T>(v.y) }, z{ static_cast<T>(v.z) }, w{ static_cast<T>(v.w) } {}

			constexpr Vector4(T v0_, const Vector3<T>& v1_) noexcept : Vector4(v0_, v1_.x, v1_.y, v1_.z) {}
			constexpr Vector4(const Vector3<T>& v0_, T v1_) noexcept : Vector4(v0_.x, v0_.y, v0_.z, v1_) {}
			constexpr Vector4(T v0_, T v1_, const Vector2<T>& v2_) noexcept : Vector4(v0_, v1_, v2_.x, v2_.y) {}
			constexpr Vector4(T v0_, const Vector2<T>& v1_, T v2_) noexcept : Vector4(v0_, v1_.x, v1_.y, v2_) {}
			constexpr Vector4(const Vector2<T>& v0_, T v1_, T v2_) noexcept : Vector4(v0_.x, v0_.y, v1_, v2_) {}
			constexpr Vector4(const Vector2<T>& v0_, const Vector2<T>& v1_) noexcept : Vector4(v0_.x, v0_.y, v1_.x, v1_.y) {}

			constexpr Vector4 operator+()const noexcept { return *this; }
			constexpr Vector4 operator-()const noexcept { return Vector4(-x, -y, -z, -w); }

			constexpr Vector4& operator+=(const Vector4& v) noexcept { x += v.x; y += v.y; z += z.y; w += v.w; return *this; }
			constexpr Vector4& operator-=(const Vector4& v) noexcept { x -= v.x; y -= v.y; z -= z.y; w -= v.w; return *this; }
			constexpr Vector4& operator*=(const Vector4& v) noexcept { x *= v.x; y *= v.y; z *= z.y; w *= v.w; return *this; }
			constexpr Vector4& operator/=(const Vector4& v) noexcept { x /= v.x; y /= v.y; z /= z.y; w /= v.w; return *this; }
			constexpr Vector4& operator*=(T v) noexcept { x *= v; y *= v; z *= v; w *= v; return *this; }
			constexpr Vector4& operator/=(T v) noexcept { x /= v; y /= v; z /= v; w /= v; return *this; }

			constexpr Bool operator==(const Vector4& v)const noexcept { 
				if constexpr (std::is_same_v<T, Float16>||std::is_same_v<T,Float32>||std::is_same_v<T,Float64>)
				{
					constexpr Float32 eps = std::is_same_v<T, Float16> ? 1e-6f : 1e-10f;
					auto len0    = this->length_sqr();
					auto len1    = v.length_sqr();
					auto len2    = Vector4(*this - v).length_sqr();
					auto len_min = len0 > len1 ? len1 : len0;
					return len2 <= len_min * eps;
				}
				else
				{
					return (x == v.x) && (y == v.y) && (z == v.z) && (w == v.w);
				}
			}
			constexpr Bool operator!=(const Vector4& v)const noexcept { return !(*this == v); }

			constexpr value_type dot(const Vector4& v) const noexcept { return x * v.x + y * v.y + z * v.z + w * v.w; }
			constexpr value_type length_sqr() const noexcept { return dot(*this); }
			value_type length() const noexcept { return RTLib::Core::sqrt(length_sqr()); }

			value_type angle_rad(const Vector4& v)const noexcept { return RTLib::Core::acos(dot(v) / (length() * v.length())); }
			value_type angle_deg(const Vector4& v)const noexcept { return degrees(angle_rad(v)); }

			Vector4& normalized() noexcept { *this /= length(); return *this; }
			Vector4  normalize() const noexcept {
				return Vector4(*this).normalized();
			}

			value_type x;
			value_type y;
			value_type z;
			value_type w;
		};

		template <typename T> Vector2<T> operator+(const Vector2<T>& v0, const Vector2<T>& v1) noexcept { return Vector2<T>(v0) += v1; }
		template <typename T> Vector3<T> operator+(const Vector3<T>& v0, const Vector3<T>& v1) noexcept { return Vector3<T>(v0) += v1; }
		template <typename T> Vector4<T> operator+(const Vector4<T>& v0, const Vector4<T>& v1) noexcept { return Vector4<T>(v0) += v1; }

		template <typename T> Vector2<T> operator-(const Vector2<T>& v0, const Vector2<T>& v1) noexcept { return Vector2<T>(v0) -= v1; }
		template <typename T> Vector3<T> operator-(const Vector3<T>& v0, const Vector3<T>& v1) noexcept { return Vector3<T>(v0) -= v1; }
		template <typename T> Vector4<T> operator-(const Vector4<T>& v0, const Vector4<T>& v1) noexcept { return Vector4<T>(v0) -= v1; }

		template <typename T> Vector2<T> operator*(const Vector2<T>& v0, const Vector2<T>& v1) noexcept { return Vector2<T>(v0) *= v1; }
		template <typename T> Vector3<T> operator*(const Vector3<T>& v0, const Vector3<T>& v1) noexcept { return Vector3<T>(v0) *= v1; }
		template <typename T> Vector4<T> operator*(const Vector4<T>& v0, const Vector4<T>& v1) noexcept { return Vector4<T>(v0) *= v1; }

		template <typename T> Vector2<T> operator/(const Vector2<T>& v0, const Vector2<T>& v1) noexcept { return Vector2<T>(v0) /= v1; }
		template <typename T> Vector3<T> operator/(const Vector3<T>& v0, const Vector3<T>& v1) noexcept { return Vector3<T>(v0) /= v1; }
		template <typename T> Vector4<T> operator/(const Vector4<T>& v0, const Vector4<T>& v1) noexcept { return Vector4<T>(v0) /= v1; }

		template <typename T> Vector2<T> operator*(const Vector2<T>& v0, T v1) noexcept { return Vector2<T>(v0) *= v1; }
		template <typename T> Vector3<T> operator*(const Vector3<T>& v0, T v1) noexcept { return Vector3<T>(v0) *= v1; }
		template <typename T> Vector4<T> operator*(const Vector4<T>& v0, T v1) noexcept { return Vector4<T>(v0) *= v1; }

		template <typename T> Vector2<T> operator*(T v0, const Vector2<T>& v1) noexcept { return Vector2<T>(v1) *= v0; }
		template <typename T> Vector3<T> operator*(T v0, const Vector3<T>& v1) noexcept { return Vector3<T>(v1) *= v0; }
		template <typename T> Vector4<T> operator*(T v0, const Vector4<T>& v1) noexcept { return Vector4<T>(v1) *= v0; }

		template <typename T> Vector2<T> operator/(const Vector2<T>& v0, T v1) noexcept { return Vector2<T>(v0) /= v1; }
		template <typename T> Vector3<T> operator/(const Vector3<T>& v0, T v1) noexcept { return Vector3<T>(v0) /= v1; }
		template <typename T> Vector4<T> operator/(const Vector4<T>& v0, T v1) noexcept { return Vector4<T>(v0) /= v1; }

		template <typename T> constexpr T dot(const Vector2<T>& v0, const Vector2<T>& v1)  noexcept { return v0.dot(v1); }
		template <typename T> constexpr T dot(const Vector3<T>& v0, const Vector3<T>& v1)  noexcept { return v0.dot(v1); }
		template <typename T> constexpr T dot(const Vector4<T>& v0, const Vector4<T>& v1)  noexcept { return v0.dot(v1); }

		template <typename T> constexpr T length_sqr(const Vector2<T>& v)  noexcept { return v.length_sqr(); }
		template <typename T> constexpr T length_sqr(const Vector3<T>& v)  noexcept { return v.length_sqr(); }
		template <typename T> constexpr T length_sqr(const Vector4<T>& v)  noexcept { return v.length_sqr(); }

		template <typename T> T length(const Vector2<T>& v)  noexcept { return v.length(); }
		template <typename T> T length(const Vector3<T>& v)  noexcept { return v.length(); }
		template <typename T> T length(const Vector4<T>& v)  noexcept { return v.length(); }

		template <typename T> T angle_rad(const Vector2<T>& v0, const Vector2<T>& v1) { return v0.angle_rad(v1); }
		template <typename T> T angle_rad(const Vector3<T>& v0, const Vector3<T>& v1) { return v0.angle_rad(v1); }
		template <typename T> T angle_rad(const Vector4<T>& v0, const Vector4<T>& v1) { return v0.angle_rad(v1); }

		template <typename T> T angle_deg(const Vector2<T>& v0, const Vector2<T>& v1) { return v0.angle_deg(v1); }
		template <typename T> T angle_deg(const Vector3<T>& v0, const Vector3<T>& v1) { return v0.angle_deg(v1); }
		template <typename T> T angle_deg(const Vector4<T>& v0, const Vector4<T>& v1) { return v0.angle_deg(v1); }

		using Vector2_I16 = Vector2<Int16>;
		using Vector2_I32 = Vector2<Int32>;
		using Vector2_I64 = Vector2<Int64>;

		using Vector3_I16 = Vector3<Int16>;
		using Vector3_I32 = Vector3<Int32>;
		using Vector3_I64 = Vector3<Int64>;

		using Vector4_I16 = Vector4<Int16>;
		using Vector4_I32 = Vector4<Int32>;
		using Vector4_I64 = Vector4<Int64>;

		using Vector2_U16 = Vector2<UInt16>;
		using Vector2_U32 = Vector2<UInt32>;
		using Vector2_U64 = Vector2<UInt64>;

		using Vector3_U16 = Vector3<UInt16>;
		using Vector3_U32 = Vector3<UInt32>;
		using Vector3_U64 = Vector3<UInt64>;

		using Vector4_U16 = Vector4<UInt16>;
		using Vector4_U32 = Vector4<UInt32>;
		using Vector4_U64 = Vector4<UInt64>;

		using Vector2_F16 = Vector2<Float16>;
		using Vector2_F32 = Vector2<Float32>;
		using Vector2_F64 = Vector2<Float64>;

		using Vector3_F16 = Vector3<Float16>;
		using Vector3_F32 = Vector3<Float32>;
		using Vector3_F64 = Vector3<Float64>;

		using Vector4_F16 = Vector4<Float16>;
		using Vector4_F32 = Vector4<Float32>;
		using Vector4_F64 = Vector4<Float64>;
#else
		using Vector2_I16 = short2;
		using Vector2_I32 = int2;
		using Vector2_I64 = longlong2;

		using Vector3_I16 = short3;
		using Vector3_I32 = int3;
		using Vector3_I64 = longlong3;

		using Vector4_I16 = short4;
		using Vector4_I32 = int4;
		using Vector4_I64 = longlong4;

		using Vector2_U16 = ushort2;
		using Vector2_U32 = uint2;
		using Vector2_U64 = ulonglong2;

		using Vector3_U16 = ushort3;
		using Vector3_U32 = uint3;
		using Vector3_U64 = ulonglong3;

		using Vector4_U16 = ushort4;
		using Vector4_U32 = uint4;
		using Vector4_U64 = ulonglong4;

		using Vector2_F16 = ushort2;
		using Vector2_F32 = float2;
		using Vector2_F64 = double2;

		using Vector3_F16 = ushort3;
		using Vector3_F32 = float3;
		using Vector3_F64 = double3;

		using Vector4_F16 = ushort4>;
		using Vector4_F32 = float4;
		using Vector4_F64 = double4;
#endif

	}
}
#endif

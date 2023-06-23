#ifndef RTLIB_CORE_MATRIX__H
#define RTLIB_CORE_MATRIX__H
#include <RTLib/Core/Vector.h>
namespace RTLib
{
	namespace Core
	{
		template<typename T> struct Matrix2x2;
		template<typename T> struct Matrix3x3;
		template<typename T> struct Matrix4x4;
		
		template<typename T> struct Matrix2x2
		{
			constexpr Matrix2x2() : m_Col_0{}, m_Col_1{} {}
			constexpr Matrix2x2(const Matrix2x2&) = default;


			Vector2<T> m_Col_0;
			Vector2<T> m_Col_1;
		};
		template<typename T> struct Matrix3x3
		{
			constexpr Matrix3x3() : m_Col_0{}, m_Col_1{}, m_Col_2{} {}
			constexpr Matrix3x3(const Matrix3x3&) = default;

			constexpr Matrix3x3(
				T m00, T m01, T m02,
				T m10, T m11, T m12,
				T m20, T m21, T m22
			) noexcept :
				m_Col_0{ m00,m10, m20 },
				m_Col_1{ m01,m11, m21 },
				m_Col_2{ m02,m12, m22 }
			{}

			constexpr Matrix3x3(
				const Vector3<T>& col_0,
				const Vector3<T>& col_1,
				const Vector3<T>& col_2
			) noexcept :
				m_Col_0{ col_0 },
				m_Col_1{ col_1 },
				m_Col_2{ col_2 }
			{}

			constexpr Matrix3x3(
				const Vector2<T>& col_0,
				const Vector2<T>& col_1
			) noexcept :



			Vector3<T> m_Col_0;
			Vector3<T> m_Col_1;
			Vector3<T> m_Col_2;
		};
		template<typename T> struct Matrix4x4
		{
			constexpr Matrix4x4() : m_Col_0{}, m_Col_1{}, m_Col_2{}, m_Col_3{} {}
			constexpr Matrix4x4(const Matrix4x4&)noexcept = default;
			constexpr Matrix4x4& operator=(const Matrix4x4&)noexcept = default;

			constexpr Matrix4x4(
				T m00, T m01, T m02, T m03,
				T m10, T m11, T m12, T m13,
				T m20, T m21, T m22, T m23,
				T m30, T m31, T m32, T m33
			) noexcept:
				m_Col_0{ m00,m10, m20, m30 },
				m_Col_1{ m01,m11, m21, m31 },
				m_Col_2{ m02,m12, m22, m32 },
				m_Col_3{ m03,m13, m23, m33 }
			{}

			constexpr Matrix4x4(
				const Vector4<T>& col_0,
				const Vector4<T>& col_1,
				const Vector4<T>& col_2,
				const Vector4<T>& col_3
			) noexcept :
				m_Col_0{ col_0 },
				m_Col_1{ col_1 },
				m_Col_2{ col_2 },
				m_Col_3{ col_3 } 
			{}



			Vector4<T> m_Col_0;
			Vector4<T> m_Col_1;
			Vector4<T> m_Col_2;
			Vector4<T> m_Col_3;
		};

		using Matrix2x2_I8  = Matrix2x2<Int8 >;
		using Matrix2x2_I16 = Matrix2x2<Int16>;
		using Matrix2x2_I32 = Matrix2x2<Int32>;
		using Matrix2x2_I64 = Matrix2x2<Int64>;

		using Matrix2x2_U8  = Matrix2x2<UInt8>;
		using Matrix2x2_U16 = Matrix2x2<UInt16>;
		using Matrix2x2_U32 = Matrix2x2<UInt32>;
		using Matrix2x2_U64 = Matrix2x2<UInt64>;

		using Matrix2x2_F16 = Matrix2x2<Float16>;
		using Matrix2x2_F32 = Matrix2x2<Float32>;
		using Matrix2x2_F64 = Matrix2x2<Float64>;

		using Matrix3x3_I8  = Matrix3x3<Int8 >;
		using Matrix3x3_I16 = Matrix3x3<Int16>;
		using Matrix3x3_I32 = Matrix3x3<Int32>;
		using Matrix3x3_I64 = Matrix3x3<Int64>;

		using Matrix3x3_U8 = Matrix3x3<UInt8>;
		using Matrix3x3_U16 = Matrix3x3<UInt16>;
		using Matrix3x3_U32 = Matrix3x3<UInt32>;
		using Matrix3x3_U64 = Matrix3x3<UInt64>;

		using Matrix3x3_F16 = Matrix3x3<Float16>;
		using Matrix3x3_F32 = Matrix3x3<Float32>;
		using Matrix3x3_F64 = Matrix3x3<Float64>;

		using Matrix4x4_I8  = Matrix4x4<Int8 >;
		using Matrix4x4_I16 = Matrix4x4<Int16>;
		using Matrix4x4_I32 = Matrix4x4<Int32>;
		using Matrix4x4_I64 = Matrix4x4<Int64>;

		using Matrix4x4_U8  = Matrix4x4<UInt8>;
		using Matrix4x4_U16 = Matrix4x4<UInt16>;
		using Matrix4x4_U32 = Matrix4x4<UInt32>;
		using Matrix4x4_U64 = Matrix4x4<UInt64>;

		using Matrix4x4_F16 = Matrix4x4<Float16>;
		using Matrix4x4_F32 = Matrix4x4<Float32>;
		using Matrix4x4_F64 = Matrix4x4<Float64>;
	}
}

#endif


#ifndef RTLIB_CORE_MATRIX__H
#define RTLIB_CORE_MATRIX__H
#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/Preprocessor.h>
#include <RTLib/Core/Vector.h>
#ifndef __CUDACC__
#include <glm/glm.hpp>
#else
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#endif
namespace RTLib
{
	inline namespace Core
	{

#ifndef __CUDACC__
		// 列優先
		using Matrix2x2_F32 = glm::f32mat2x2;
		using Matrix3x3_F32 = glm::f32mat3x3;
		using Matrix4x4_F32 = glm::f32mat4x4;

		using Matrix2x2_F64 = glm::f64mat2x2;
		using Matrix3x3_F64 = glm::f64mat3x3;
		using Matrix4x4_F64 = glm::f64mat4x4;

		using Matrix2x2_U8 = glm::u8mat2x2;
		using Matrix3x3_U8 = glm::u8mat3x3;
		using Matrix4x4_U8 = glm::u8mat4x4;

		using Matrix2x2_U16 = glm::u16mat2x2;
		using Matrix3x3_U16 = glm::u16mat3x3;
		using Matrix4x4_U16 = glm::u16mat4x4;

		using Matrix2x2_U32 = glm::u32mat2x2;
		using Matrix3x3_U32 = glm::u32mat3x3;
		using Matrix4x4_U32 = glm::u32mat4x4;

		using Matrix2x2_U64 = glm::u64mat2x2;
		using Matrix3x3_U64 = glm::u64mat3x3;
		using Matrix4x4_U64 = glm::u64mat4x4;

		using Matrix2x2_I8 = glm::i8mat2x2;
		using Matrix3x3_I8 = glm::i8mat3x3;
		using Matrix4x4_I8 = glm::i8mat4x4;

		using Matrix2x2_I16 = glm::i16mat2x2;
		using Matrix3x3_I16 = glm::i16mat3x3;
		using Matrix4x4_I16 = glm::i16mat4x4;

		using Matrix2x2_I32 = glm::i32mat2x2;
		using Matrix3x3_I32 = glm::i32mat3x3;
		using Matrix4x4_I32 = glm::i32mat4x4;

		using Matrix2x2_I64 = glm::i64mat2x2;
		using Matrix3x3_I64 = glm::i64mat3x3;
		using Matrix4x4_I64 = glm::i64mat4x4;
#else
		// 列優先, 計算はm*vだと非効率的, 必ずv*mで計算すること
		// 場合によっては cpu -> gpuで転置が必要
		namespace internals
		{
			template<typename T>
			struct Matrix2x2
			{
				RTLIB_DEVICE Matrix2x2() noexcept : col0{}, col1{} {}
				RTLIB_DEVICE Matrix2x2(const Matrix2x2&) noexcept = default;
				RTLIB_DEVICE Matrix2x2& operator=(const Matrix2x2&) noexcept = default;

				RTLIB_DEVICE Matrix2x2(
					T m0x, T m0y,
					T m1x, T m1y
				)noexcept :
					col0{ make_vector2(m0x,m0y) },
					col1{ make_vector2(m1x,m1y) }{}

				RTLIB_DEVICE Matrix2x2(
					const MathVector2<T>& col0_,
					const MathVector2<T>& col1_
				)noexcept :
					col0{ col0_ },
					col1{ col1_ } {}

				RTLIB_DEVICE auto operator+() const noexcept -> Matrix2x2 { return *this; }
				RTLIB_DEVICE auto operator-() const noexcept -> Matrix2x2
				{
					return Matrix2x2(
						-col0.x, -col0.y,
						-col1.x, -col1.y
					);
				}

				RTLIB_DEVICE auto operator+(const Matrix2x2& m) const noexcept -> Matrix2x2
				{
					using namespace otk;
					return Matrix2x2(col0 + m.col0, col1 + m.col1);
				}
				RTLIB_DEVICE auto operator-(const Matrix2x2& m) const noexcept -> Matrix2x2
				{
					using namespace otk;
					return Matrix2x2(col0 - m.col0, col1 - m.col1);
				}
				RTLIB_DEVICE auto operator*(const Matrix2x2& m) const noexcept -> Matrix2x2;
				RTLIB_DEVICE auto operator*(T s) const noexcept -> Matrix2x2
				{
					using namespace otk;
					return Matrix2x2(col0 * s, col1 * s);
				}
				RTLIB_DEVICE auto operator/(T s) const noexcept -> Matrix2x2
				{
					using namespace otk;
					return Matrix2x2(col0 /s, col1 / s);
				}


				RTLIB_DEVICE auto operator*(const MathVector2<T>& v) const noexcept -> MathVector2<T>;

				MathVector2<T> col0;
				MathVector2<T> col1;
			};

			template<typename T>
			struct Matrix3x3
			{
				RTLIB_DEVICE Matrix3x3() noexcept : col0{}, col1{}, col2{} {}
				RTLIB_DEVICE Matrix3x3(const Matrix3x3&) noexcept = default;
				RTLIB_DEVICE Matrix3x3& operator=(const Matrix3x3&) noexcept = default;

				RTLIB_DEVICE Matrix3x3(
					T m0x, T m0y, T m0z,
					T m1x, T m1y, T m1z,
					T m2x, T m2y, T m2z
				)noexcept :
					col0{ make_vector3(m0x,m0y,m0z) },
					col1{ make_vector3(m1x,m1y,m1z) },
					col2{ make_vector3(m2x,m2y,m2z) } {}

				RTLIB_DEVICE Matrix3x3(
					const MathVector3<T>& col0_,
					const MathVector3<T>& col1_,
					const MathVector3<T>& col2_
				)noexcept :
					col0{ col0_ },
					col1{ col1_ },
					col2{ col2_ } {}

				RTLIB_DEVICE auto operator+() const noexcept -> Matrix3x3 { return *this; }
				RTLIB_DEVICE auto operator-() const noexcept -> Matrix3x3 
				{ 
					return Matrix3x3(
						-col0.x, -col0.y, -col0.z,
						-col1.x, -col1.y, -col1.z,
						-col2.x, -col2.y, -col2.z
					);
				}

				RTLIB_DEVICE auto operator+(const Matrix3x3& m) const noexcept -> Matrix3x3
				{
					using namespace otk;
					return Matrix3x3(col0 + m.col0, col1 + m.col1, col2+m.col2);
				}
				RTLIB_DEVICE auto operator-(const Matrix3x3& m) const noexcept -> Matrix3x3
				{
					using namespace otk;
					return Matrix3x3(col0 - m.col0, col1 - m.col1, col2 - m.col2);
				}
				RTLIB_DEVICE auto operator*(const Matrix3x3& m) const noexcept -> Matrix3x3;
				RTLIB_DEVICE auto operator*(T s) const noexcept -> Matrix3x3
				{
					using namespace otk;
					return Matrix3x3(col0 * s, col1 * s, col2 * s);
				}
				RTLIB_DEVICE auto operator/(T s) const noexcept -> Matrix3x3
				{
					using namespace otk;
					return Matrix3x3(col0 / s, col1 / s, col2 / s);
				}


				RTLIB_DEVICE auto operator*(const MathVector3<T>& v) const noexcept -> MathVector3<T>;

				MathVector3<T> col0;
				MathVector3<T> col1;
				MathVector3<T> col2;
			};

			template<typename T>
			struct Matrix4x4
			{
				RTLIB_DEVICE Matrix4x4() noexcept : col0{}, col1{}, col2{}, col3{} {}
				RTLIB_DEVICE Matrix4x4(const Matrix4x4&) noexcept = default;
				RTLIB_DEVICE Matrix4x4& operator=(const Matrix4x4&) noexcept = default;

				RTLIB_DEVICE Matrix4x4(
					T m0x, T m0y, T m0z, T m0w,
					T m1x, T m1y, T m1z, T m1w,
					T m2x, T m2y, T m2z, T m2w,
					T m3x, T m3y, T m3z, T m3w
				)noexcept :
					col0{ make_vector4(m0x,m0y,m0z,m0w) },
					col1{ make_vector4(m1x,m1y,m1z,m1w) },
					col2{ make_vector4(m2x,m2y,m2z,m2w) },
					col3{ make_vector4(m3x,m3y,m3z,m3w) } {}

				RTLIB_DEVICE Matrix4x4(
					const MathVector4<T>& col0_,
					const MathVector4<T>& col1_,
					const MathVector4<T>& col2_,
					const MathVector4<T>& col3_
				)noexcept :
					col0{ col0_ },
					col1{ col1_ },
					col2{ col2_ },
					col3{ col3_ } {}

				RTLIB_DEVICE auto operator+() const noexcept -> Matrix4x4 { return *this; }
				RTLIB_DEVICE auto operator-() const noexcept -> Matrix4x4
				{
					return Matrix4x4(
						-col0.x, -col0.y, -col0.z, -col0.w,
						-col1.x, -col1.y, -col1.z, -col1.w,
						-col2.x, -col2.y, -col2.z, -col2.w,
						-col3.x, -col3.y, -col3.z, -col3.w
					);
				}

				RTLIB_DEVICE auto operator+(const Matrix4x4& m) const noexcept -> Matrix4x4
				{
					using namespace otk;
					return Matrix4x4(col0 + m.col0, col1 + m.col1, col2 + m.col2, col3 + m.col3);
				}
				RTLIB_DEVICE auto operator-(const Matrix4x4& m) const noexcept -> Matrix4x4
				{
					using namespace otk;
					return Matrix4x4(col0 - m.col0, col1 - m.col1, col2 - m.col2, col3 - m.col3);
				}
				RTLIB_DEVICE auto operator*(const Matrix4x4& m) const noexcept -> Matrix4x4;
				RTLIB_DEVICE auto operator*(T s) const noexcept -> Matrix4x4
				{
					using namespace otk;
					return Matrix4x4(col0 * s, col1 * s, col2 * s, col3 * s);
				}
				RTLIB_DEVICE auto operator/(T s) const noexcept -> Matrix4x4
				{
					using namespace otk;
					return Matrix4x4(col0 / s, col1 / s, col2 / s, col3 / s);
				}

				RTLIB_DEVICE auto operator*(const MathVector4<T>& v) const noexcept -> MathVector4<T>;

				MathVector4<T> col0;
				MathVector4<T> col1;
				MathVector4<T> col2;
				MathVector4<T> col3;
			};

			template<typename T>
			RTLIB_DEVICE auto operator*(const MathVector2<T>& v, const Matrix2x2<T>& m) noexcept -> MathVector2<T> {
				using namespace otk;
				return make_vector2(dot(v, m.col0),dot(v, m.col1));
			}

			template<typename T>
			RTLIB_DEVICE auto operator*(const MathVector3<T>& v, const Matrix3x3<T>& m) noexcept -> MathVector3<T> {
				using namespace otk;
				return make_vector3(dot(v, m.col0), dot(v, m.col1), dot(v, m.col2));
			}

			template<typename T>
			RTLIB_DEVICE auto operator*(const MathVector4<T>& v, const Matrix4x4<T>& m) noexcept -> MathVector4<T> {
				using namespace otk;
				return make_vector4(dot(v, m.col0), dot(v, m.col1), dot(v, m.col2), dot(v, m.col3));
			}

			template<typename T>
			RTLIB_DEVICE auto Matrix2x2<T>::operator*(const MathVector2<T>& v) const noexcept -> MathVector2<T> {
				return make_vector2(col0.x * v.x + col1.x * v.y, col0.y * v.x + col1.y * v.y);
			}

			template<typename T>
			RTLIB_DEVICE auto Matrix3x3<T>::operator*(const MathVector3<T>& v) const noexcept -> MathVector3<T> {
				return make_vector3(
					col0.x * v.x + col1.x * v.y + col2.x * v.z,
					col0.y * v.x + col1.y * v.y + col2.y * v.z,
					col0.z * v.x + col1.z * v.y + col2.z * v.z
				);
			}

			template<typename T>
			RTLIB_DEVICE auto Matrix4x4<T>::operator*(const MathVector4<T>& v) const noexcept -> MathVector4<T> {
				return make_vector4(
					col0.x * v.x + col1.x * v.y + col2.x * v.z + col3.x * v.w,
					col0.y * v.x + col1.y * v.y + col2.y * v.z + col3.y * v.w,
					col0.z * v.x + col1.z * v.y + col2.z * v.z + col3.z * v.w,
					col0.w * v.x + col1.w * v.y + col2.w * v.z + col3.w * v.w
				);
			}


			template<typename T>
			RTLIB_DEVICE auto Matrix2x2<T>::operator*(const Matrix2x2& m) const noexcept -> Matrix2x2 {
				// ------  |
				return Matrix2x2(
					col0.x * m.col0.x + col1.x * m.col0.y ,
					col0.y * m.col0.x + col1.y * m.col0.y ,

					col0.x * m.col1.x + col1.x * m.col1.y ,
					col0.y * m.col1.x + col1.y * m.col1.y 
				);
			}

			template<typename T>
			RTLIB_DEVICE auto Matrix3x3<T>::operator*(const Matrix3x3& m) const noexcept -> Matrix3x3
			{
				return Matrix3x3(
					col0.x * m.col0.x + col1.x * m.col0.y + col2.x * m.col0.z,
					col0.y * m.col0.x + col1.y * m.col0.y + col2.y * m.col0.z,
					col0.z * m.col0.x + col1.z * m.col0.y + col2.z * m.col0.z,

					col0.x * m.col1.x + col1.x * m.col1.y + col2.x * m.col1.z,
					col0.y * m.col1.x + col1.y * m.col1.y + col2.y * m.col1.z,
					col0.z * m.col1.x + col1.z * m.col1.y + col2.z * m.col1.z,

					col0.x * m.col2.x + col1.x * m.col2.y + col2.x * m.col2.z,
					col0.y * m.col2.x + col1.y * m.col2.y + col2.y * m.col2.z,
					col0.z * m.col2.x + col1.z * m.col2.y + col2.z * m.col2.z
				);
			}

			template<typename T>
			RTLIB_DEVICE auto Matrix4x4<T>::operator*(const Matrix4x4& m) const noexcept -> Matrix4x4 {
				// ------  |
				return Matrix4x4(
					col0.x * m.col0.x + col1.x * m.col0.y + col2.x * m.col0.z + col3.x * m.col0.w,
					col0.y * m.col0.x + col1.y * m.col0.y + col2.y * m.col0.z + col3.y * m.col0.w,
					col0.z * m.col0.x + col1.z * m.col0.y + col2.z * m.col0.z + col3.z * m.col0.w,
					col0.w * m.col0.x + col1.w * m.col0.y + col2.w * m.col0.z + col3.w * m.col0.w,

					col0.x * m.col1.x + col1.x * m.col1.y + col2.x * m.col1.z + col3.x * m.col1.w,
					col0.y * m.col1.x + col1.y * m.col1.y + col2.y * m.col1.z + col3.y * m.col1.w,
					col0.z * m.col1.x + col1.z * m.col1.y + col2.z * m.col1.z + col3.z * m.col1.w,
					col0.w * m.col1.x + col1.w * m.col1.y + col2.w * m.col1.z + col3.w * m.col1.w,

					col0.x * m.col2.x + col1.x * m.col2.y + col2.x * m.col2.z + col3.x * m.col2.w,
					col0.y * m.col2.x + col1.y * m.col2.y + col2.y * m.col2.z + col3.y * m.col2.w,
					col0.z * m.col2.x + col1.z * m.col2.y + col2.z * m.col2.z + col3.z * m.col2.w,
					col0.w * m.col2.x + col1.w * m.col2.y + col2.w * m.col2.z + col3.w * m.col2.w,

					col0.x * m.col3.x + col1.x * m.col3.y + col2.x * m.col3.z + col3.x * m.col3.w,
					col0.y * m.col3.x + col1.y * m.col3.y + col2.y * m.col3.z + col3.y * m.col3.w,
					col0.z * m.col3.x + col1.z * m.col3.y + col2.z * m.col3.z + col3.z * m.col3.w,
					col0.w * m.col3.x + col1.w * m.col3.y + col2.w * m.col3.z + col3.w * m.col3.w
				);
			}



		}

		using Matrix2x2_F32 = internals::Matrix2x2<Float32>;
		using Matrix3x3_F32 = internals::Matrix3x3<Float32>;
		using Matrix4x4_F32 = internals::Matrix4x4<Float32>;

		using Matrix2x2_F64 = internals::Matrix2x2<Float64>;
		using Matrix3x3_F64 = internals::Matrix3x3<Float64>;
		using Matrix4x4_F64 = internals::Matrix4x4<Float64>;

		using Matrix2x2_I8 = internals::Matrix2x2<Int8>;
		using Matrix3x3_I8 = internals::Matrix3x3<Int8>;
		using Matrix4x4_I8 = internals::Matrix4x4<Int8>;

		using Matrix2x2_I16 = internals::Matrix2x2<Int16>;
		using Matrix3x3_I16 = internals::Matrix3x3<Int16>;
		using Matrix4x4_I16 = internals::Matrix4x4<Int16>;

		using Matrix2x2_I32 = internals::Matrix2x2<Int32>;
		using Matrix3x3_I32 = internals::Matrix3x3<Int32>;
		using Matrix4x4_I32 = internals::Matrix4x4<Int32>;

		using Matrix2x2_I64 = internals::Matrix2x2<Int64>;
		using Matrix3x3_I64 = internals::Matrix3x3<Int64>;
		using Matrix4x4_I64 = internals::Matrix4x4<Int64>;

		using Matrix2x2_U8 = internals::Matrix2x2<UInt8>;
		using Matrix3x3_U8 = internals::Matrix3x3<UInt8>;
		using Matrix4x4_U8 = internals::Matrix4x4<UInt8>;

		using Matrix2x2_U16 = internals::Matrix2x2<UInt16>;
		using Matrix3x3_U16 = internals::Matrix3x3<UInt16>;
		using Matrix4x4_U16 = internals::Matrix4x4<UInt16>;

		using Matrix2x2_U32 = internals::Matrix2x2<UInt32>;
		using Matrix3x3_U32 = internals::Matrix3x3<UInt32>;
		using Matrix4x4_U32 = internals::Matrix4x4<UInt32>;

		using Matrix2x2_U64 = internals::Matrix2x2<UInt64>;
		using Matrix3x3_U64 = internals::Matrix3x3<UInt64>;
		using Matrix4x4_U64 = internals::Matrix4x4<UInt64>;
#endif
		template<typename T, Int32 row, Int32 col> struct MathMatrixTraits;

		template<> struct MathMatrixTraits<Float32, 2,2> { using type = Matrix2x2_F32; };
		template<> struct MathMatrixTraits<Float32, 3,3> { using type = Matrix3x3_F32; };
		template<> struct MathMatrixTraits<Float32, 4,4> { using type = Matrix4x4_F32; };

		template<> struct MathMatrixTraits<Float64, 2,2> { using type = Matrix2x2_F64; };
		template<> struct MathMatrixTraits<Float64, 3,3> { using type = Matrix3x3_F64; };
		template<> struct MathMatrixTraits<Float64, 4,4> { using type = Matrix4x4_F64; };

		template<> struct MathMatrixTraits<Int8, 2,2> { using type = Matrix2x2_I8; };
		template<> struct MathMatrixTraits<Int8, 3,3> { using type = Matrix3x3_I8; };
		template<> struct MathMatrixTraits<Int8, 4,4> { using type = Matrix4x4_I8; };

		template<> struct MathMatrixTraits<Int16, 2,2> { using type = Matrix2x2_I16; };
		template<> struct MathMatrixTraits<Int16, 3,3> { using type = Matrix3x3_I16; };
		template<> struct MathMatrixTraits<Int16, 4,4> { using type = Matrix4x4_I16; };

		template<> struct MathMatrixTraits<Int32, 2,2> { using type = Matrix2x2_I32; };
		template<> struct MathMatrixTraits<Int32, 3,3> { using type = Matrix3x3_I32; };
		template<> struct MathMatrixTraits<Int32, 4,4> { using type = Matrix4x4_I32; };

		template<> struct MathMatrixTraits<Int64, 2,2> { using type = Matrix2x2_I64; };
		template<> struct MathMatrixTraits<Int64, 3,3> { using type = Matrix3x3_I64; };
		template<> struct MathMatrixTraits<Int64, 4,4> { using type = Matrix4x4_I64; };

		template<> struct MathMatrixTraits<UInt8, 2,2> { using type = Matrix2x2_U8; };
		template<> struct MathMatrixTraits<UInt8, 3,3> { using type = Matrix3x3_U8; };
		template<> struct MathMatrixTraits<UInt8, 4,4> { using type = Matrix4x4_U8; };

		template<> struct MathMatrixTraits<UInt16, 2,2> { using type = Matrix2x2_U16; };
		template<> struct MathMatrixTraits<UInt16, 3,3> { using type = Matrix3x3_U16; };
		template<> struct MathMatrixTraits<UInt16, 4,4> { using type = Matrix4x4_U16; };

		template<> struct MathMatrixTraits<UInt32, 2,2> { using type = Matrix2x2_U32; };
		template<> struct MathMatrixTraits<UInt32, 3,3> { using type = Matrix3x3_U32; };
		template<> struct MathMatrixTraits<UInt32, 4,4> { using type = Matrix4x4_U32; };

		template<> struct MathMatrixTraits<UInt64, 2,2> { using type = Matrix2x2_U64; };
		template<> struct MathMatrixTraits<UInt64, 3,3> { using type = Matrix3x3_U64; };
		template<> struct MathMatrixTraits<UInt64, 4,4> { using type = Matrix4x4_U64; };

		using Matrix2x2 = Matrix2x2_F32;
		using Matrix3x3 = Matrix3x3_F32;
		using Matrix4x4 = Matrix4x4_F32;

		template<typename T, Int32 row, Int32 col>
		using MathMatrix = typename MathMatrixTraits<T, row, col>::type;

		template<typename T>
		using MathMatrix2x2 = MathMatrix<T, 2, 2>;

		template<typename T>
		using MathMatrix3x3 = MathMatrix<T, 3, 3>;

		template<typename T>
		using MathMatrix4x4 = MathMatrix<T, 4, 4>;


		template<typename T>
		RTLIB_INLINE RTLIB_DEVICE auto make_matrix2x2(
			T m0x, T m0y, 
			T m1x, T m1y
		) noexcept -> MathMatrix<T, 2,2> {
			return { m0x,m0y,m1x,m1y };
		}
		template<typename T>
		RTLIB_INLINE RTLIB_DEVICE auto make_matrix3x3(
			T m0x, T m0y, T m0z, 
			T m1x, T m1y, T m1z, 
			T m2x, T m2y, T m2z
		) noexcept -> MathMatrix<T, 3,3> {
			return { m0x,m0y,m0z,m1x,m1y,m1z,m2x,m2y,m2z };
		}
		template<typename T>
		RTLIB_INLINE RTLIB_DEVICE auto make_matrix4x4(
			T m0x, T m0y, T m0z, T m0w, 
			T m1x, T m1y, T m1z, T m1w, 
			T m2x, T m2y, T m2z, T m2w,
			T m3x, T m3y, T m3z, T m3w
		) noexcept -> MathMatrix<T, 4,4> {
			return { m0x,m0y,m0z,m0w,m1x,m1y,m1z,m1w,m2x,m2y,m2z,m2w,m3x,m3y,m3z,m3w };
		}
	}
}
#endif

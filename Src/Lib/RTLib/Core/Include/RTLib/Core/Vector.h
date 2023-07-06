#ifndef RTLIB_CORE_VECTOR__H
#define RTLIB_CORE_VECTOR__H
#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/Preprocessor.h>
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
		
		using Vector2_F32 = glm::f32vec2;
		using Vector3_F32 = glm::f32vec3;
		using Vector4_F32 = glm::f32vec4;

		using Vector2_F64 = glm::f64vec2;
		using Vector3_F64 = glm::f64vec3;
		using Vector4_F64 = glm::f64vec4;

		using Vector2_U8 = glm::u8vec2;
		using Vector3_U8 = glm::u8vec3;
		using Vector4_U8 = glm::u8vec4;

		using Vector2_U16 = glm::u16vec2;
		using Vector3_U16 = glm::u16vec3;
		using Vector4_U16 = glm::u16vec4;

		using Vector2_U32 = glm::u32vec2;
		using Vector3_U32 = glm::u32vec3;
		using Vector4_U32 = glm::u32vec4;

		using Vector2_U64 = glm::u64vec2;
		using Vector3_U64 = glm::u64vec3;
		using Vector4_U64 = glm::u64vec4;

		using Vector2_I8 = glm::i8vec2;
		using Vector3_I8 = glm::i8vec3;
		using Vector4_I8 = glm::i8vec4;

		using Vector2_I16 = glm::i16vec2;
		using Vector3_I16 = glm::i16vec3;
		using Vector4_I16 = glm::i16vec4;

		using Vector2_I32 = glm::i32vec2;
		using Vector3_I32 = glm::i32vec3;
		using Vector4_I32 = glm::i32vec4;

		using Vector2_I64 = glm::i64vec2;
		using Vector3_I64 = glm::i64vec3;
		using Vector4_I64 = glm::i64vec4;
#else
		using Vector2_F32 = float2;
		using Vector3_F32 = float3;
		using Vector4_F32 = float4;

		using Vector2_F64 = double2;
		using Vector3_F64 = double3;
		using Vector4_F64 = double4;

		using Vector2_U8 = uchar2;
		using Vector3_U8 = uchar3;
		using Vector4_U8 = uchar4;

		using Vector2_U16 = ushort2;
		using Vector3_U16 = ushort3;
		using Vector4_U16 = ushort4;

		using Vector2_U32 = uint2;
		using Vector3_U32 = uint3;
		using Vector4_U32 = uint4;

		using Vector2_U64 = ulonglong2;
		using Vector3_U64 = ulonglong3;
		using Vector4_U64 = ulonglong4;

		using Vector2_I8 = char2;
		using Vector3_I8 = char3;
		using Vector4_I8 = char4;

		using Vector2_I16 = short2;
		using Vector3_I16 = short3;
		using Vector4_I16 = short4;

		using Vector2_I32 = int2;
		using Vector3_I32 = int3;
		using Vector4_I32 = int4;

		using Vector2_I64 = longlong2;
		using Vector3_I64 = longlong3;
		using Vector4_I64 = longlong4;
#endif
		template<typename T, Int32 dim> struct MathVectorTraits;
		template<> struct MathVectorTraits<Float32, 2> { using type = Vector2_F32; };
		template<> struct MathVectorTraits<Float32, 3> { using type = Vector3_F32; };
		template<> struct MathVectorTraits<Float32, 4> { using type = Vector4_F32; };

		template<> struct MathVectorTraits<Float64, 2> { using type = Vector2_F64; };
		template<> struct MathVectorTraits<Float64, 3> { using type = Vector3_F64; };
		template<> struct MathVectorTraits<Float64, 4> { using type = Vector4_F64; };

		template<> struct MathVectorTraits<Int8, 2> { using type = Vector2_I8; };
		template<> struct MathVectorTraits<Int8, 3> { using type = Vector3_I8; };
		template<> struct MathVectorTraits<Int8, 4> { using type = Vector4_I8; };

		template<> struct MathVectorTraits<Int16, 2> { using type = Vector2_I16; };
		template<> struct MathVectorTraits<Int16, 3> { using type = Vector3_I16; };
		template<> struct MathVectorTraits<Int16, 4> { using type = Vector4_I16; };

		template<> struct MathVectorTraits<Int32, 2> { using type = Vector2_I32; };
		template<> struct MathVectorTraits<Int32, 3> { using type = Vector3_I32; };
		template<> struct MathVectorTraits<Int32, 4> { using type = Vector4_I32; };

		template<> struct MathVectorTraits<Int64, 2> { using type = Vector2_I64; };
		template<> struct MathVectorTraits<Int64, 3> { using type = Vector3_I64; };
		template<> struct MathVectorTraits<Int64, 4> { using type = Vector4_I64; };

		template<> struct MathVectorTraits<UInt8, 2> { using type = Vector2_U8; };
		template<> struct MathVectorTraits<UInt8, 3> { using type = Vector3_U8; };
		template<> struct MathVectorTraits<UInt8, 4> { using type = Vector4_U8; };

		template<> struct MathVectorTraits<UInt16, 2> { using type = Vector2_U16; };
		template<> struct MathVectorTraits<UInt16, 3> { using type = Vector3_U16; };
		template<> struct MathVectorTraits<UInt16, 4> { using type = Vector4_U16; };

		template<> struct MathVectorTraits<UInt32, 2> { using type = Vector2_U32; };
		template<> struct MathVectorTraits<UInt32, 3> { using type = Vector3_U32; };
		template<> struct MathVectorTraits<UInt32, 4> { using type = Vector4_U32; };

		template<> struct MathVectorTraits<UInt64, 2> { using type = Vector2_U64; };
		template<> struct MathVectorTraits<UInt64, 3> { using type = Vector3_U64; };
		template<> struct MathVectorTraits<UInt64, 4> { using type = Vector4_U64; };

		using Vector2 = Vector2_F32;
		using Vector3 = Vector3_F32;
		using Vector4 = Vector4_F32;

		template<typename T, Int32 dim>
		using MathVector = typename MathVectorTraits<T, dim>::type;

		template<typename T>
		using MathVector2 = MathVector<T, 2>;

		template<typename T>
		using MathVector3 = MathVector<T, 3>;

		template<typename T>
		using MathVector4 = MathVector<T, 4>;


		template<typename T>
		RTLIB_INLINE RTLIB_DEVICE auto make_vector2(T x, T y) noexcept -> MathVector<T, 2> {
			return { x,y };
		}
		template<typename T>
		RTLIB_INLINE RTLIB_DEVICE auto make_vector3(T x, T y, T z) noexcept -> MathVector<T, 3> {
			return { x,y,z };
		}
		template<typename T>
		RTLIB_INLINE RTLIB_DEVICE auto make_vector4(T x, T y, T z, T w) noexcept -> MathVector<T, 4> {
			return { x,y,z,w };
		}
	}
}
#endif

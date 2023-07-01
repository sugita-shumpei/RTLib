#ifndef RTLIB_CORE_VECTOR__H
#define RTLIB_CORE_VECTOR__H
#include <RTLib/Core/DataTypes.h>
#include <RTLib/Core/Math.h>
#ifndef __CUDACC__
#include <glm/glm.hpp>
#include <RTLib/Core/Json.h>
#else
#include <OptiXToolkit/ShaderUtil/vec_math.h>
#endif
namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		using namespace glm;

		using Vector2 = glm::vec2;
		using Vector3 = glm::vec3;
		using Vector4 = glm::vec4;

		using Vector2_I8  = glm::i8vec2;
		using Vector2_I16 = glm::i16vec2;
		using Vector2_I32 = glm::i32vec2;
		using Vector2_I64 = glm::i64vec2;

		using Vector3_I8  = glm::i8vec3;
		using Vector3_I16 = glm::i16vec3;
		using Vector3_I32 = glm::i32vec3;
		using Vector3_I64 = glm::i64vec3;

		using Vector4_I8  = glm::i8vec4;
		using Vector4_I16 = glm::i16vec4;
		using Vector4_I32 = glm::i32vec4;
		using Vector4_I64 = glm::i64vec4;

		using Vector2_U8  = glm::u8vec2;
		using Vector2_U16 = glm::u16vec2;
		using Vector2_U32 = glm::u32vec2;
		using Vector2_U64 = glm::u64vec2;

		using Vector3_U8  = glm::u8vec3;
		using Vector3_U16 = glm::u16vec3;
		using Vector3_U32 = glm::u32vec3;
		using Vector3_U64 = glm::u64vec3;

		using Vector4_U8  = glm::u8vec4;
		using Vector4_U16 = glm::u16vec4;
		using Vector4_U32 = glm::u32vec4;
		using Vector4_U64 = glm::u64vec4;

		using Vector2_F32 = glm::f32vec2;
		using Vector2_F64 = glm::f64vec2;

		using Vector3_F32 = glm::f32vec3;
		using Vector3_F64 = glm::f64vec3;

		using Vector4_F32 = glm::f32vec4;
		using Vector4_F64 = glm::f64vec4;

#else
		using Vector2 = float2;
		using Vector3 = float3;
		using Vector4 = float4;

		using Vector2_I8  = char2;
		using Vector2_I16 = short2;
		using Vector2_I32 = int2;
		using Vector2_I64 = longlong2;

		using Vector3_I8  = char3;
		using Vector3_I16 = short3;
		using Vector3_I32 = int3;
		using Vector3_I64 = longlong3;
		
		using Vector4_I8  = char4;
		using Vector4_I16 = short4;
		using Vector4_I32 = int4;
		using Vector4_I64 = longlong4;

		using Vector2_U8  = uchar2;
		using Vector2_U16 = ushort2;
		using Vector2_U32 = uint2;
		using Vector2_U64 = ulonglong2;

		using Vector3_U8  = uchar3;
		using Vector3_U16 = ushort3;
		using Vector3_U32 = uint3;
		using Vector3_U64 = ulonglong3;

		using Vector4_U8  = uchar4;
		using Vector4_U16 = ushort4;
		using Vector4_U32 = uint4;
		using Vector4_U64 = ulonglong4;

		using Vector2_F32 = float2;
		using Vector2_F64 = double2;

		using Vector3_F32 = float3;
		using Vector3_F64 = double3;

		using Vector4_F32 = float4;
		using Vector4_F64 = double4;
#endif
	}
}

#ifndef __CUDACC__
namespace glm
{
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Vector2& v);
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Vector3& v);
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Vector4& v);

	void from_json(const RTLib::Core::Json& json, RTLib::Core::Vector2& v);
	void from_json(const RTLib::Core::Json& json, RTLib::Core::Vector3& v);
	void from_json(const RTLib::Core::Json& json, RTLib::Core::Vector4& v);
}
#endif

#endif

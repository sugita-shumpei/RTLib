#ifndef RTLIB_CORE_MATRIX__H
#define RTLIB_CORE_MATRIX__H
#ifndef __CUDACC__
#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Json.h>
#include <glm/glm.hpp>
#endif
namespace RTLib
{
	inline namespace Core
	{
#ifndef __CUDACC__
		using Matrix2x2 = glm::mat2;
		using Matrix3x3 = glm::mat3;
		using Matrix4x4 = glm::mat4;

		using Matrix2x2_I8  = glm::i8mat2x2;
		using Matrix2x2_I16 = glm::i16mat2x2;
		using Matrix2x2_I32 = glm::i32mat2x2;
		using Matrix2x2_I64 = glm::i64mat2x2;

		using Matrix3x3_I8  = glm::i8mat3x3;
		using Matrix3x3_I16 = glm::i16mat3x3;
		using Matrix3x3_I32 = glm::i32mat3x3;
		using Matrix3x3_I64 = glm::i64mat3x3;

		using Matrix4x4_I8  = glm::i8mat4x4;
		using Matrix4x4_I16 = glm::i16mat4x4;
		using Matrix4x4_I32 = glm::i32mat4x4;
		using Matrix4x4_I64 = glm::i64mat4x4;

		using Matrix2x2_U8  = glm::u8mat2x2;
		using Matrix2x2_U16 = glm::u16mat2x2;
		using Matrix2x2_U32 = glm::u32mat2x2;
		using Matrix2x2_U64 = glm::u64mat2x2;

		using Matrix3x3_U8  = glm::u8mat3x3;
		using Matrix3x3_U16 = glm::u16mat3x3;
		using Matrix3x3_U32 = glm::u32mat3x3;
		using Matrix3x3_U64 = glm::u64mat3x3;

		using Matrix4x4_U8  = glm::u8mat4x4;
		using Matrix4x4_U16 = glm::u16mat4x4;
		using Matrix4x4_U32 = glm::u32mat4x4;
		using Matrix4x4_U64 = glm::u64mat4x4;

		using Matrix2x2_F32 = glm::f32mat2x2;
		using Matrix2x2_F64 = glm::f64mat2x2;

		using Matrix3x3_F32 = glm::f32mat3x3;
		using Matrix3x3_F64 = glm::f64mat3x3;

		using Matrix4x4_F32 = glm::f32mat4x4;
		using Matrix4x4_F64 = glm::f64mat4x4;
		
#else
		
#endif
	}
}

#ifndef __CUDACC__
namespace glm {
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Matrix2x2& m);
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Matrix3x3& m);
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Matrix4x4& m);

	void from_json(const RTLib::Core::Json& json, RTLib::Core::Matrix2x2& m);
	void from_json(const RTLib::Core::Json& json, RTLib::Core::Matrix3x3& m);
	void from_json(const RTLib::Core::Json& json, RTLib::Core::Matrix4x4& m);
}
#endif

#endif
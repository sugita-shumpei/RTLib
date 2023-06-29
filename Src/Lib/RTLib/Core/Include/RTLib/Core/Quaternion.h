#ifndef RTLIB_CORE_QUATERNION__H
#define RTLIB_CORE_QUATERNION__H

#ifndef __CUDACC__
#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Json.h>
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
namespace RTLib
{
	namespace Core
	{
		using Quat     = glm::quat;
		using Quat_F32 = glm::f32quat;
		using Quat_F64 = glm::f64quat;

	}
}

namespace glm
{
	void to_json(RTLib::Core::Json& json, const RTLib::Core::Quat& q);
	void from_json(const RTLib::Core::Json& json, RTLib::Core::Quat& q);
}
#endif

#endif

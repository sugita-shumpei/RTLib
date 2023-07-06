#ifndef RTLIB_CORE_TRANSFORM__H
#define RTLIB_CORE_TRANSFORM__H
#include <RTLib/Core/Vector.h>
#include <RTLib/Core/Matrix.h>
#include <RTLib/Core/Quaternion.h>
#include <RTLib/Core/Transform.h>
namespace RTLib
{
	namespace Core
	{
		struct Transform
		{
			Transform() noexcept {}
			Transform(const Transform& transform) noexcept = default;
			Transform& operator=(const Transform& transform) noexcept = default;

			auto get_local_to_parent_matrix() const noexcept -> Matrix4x4;
			auto get_parent_to_local_matrix() const noexcept -> Matrix4x4;

			Vector3 position = Vector3(1.0f);
			Quat    rotation = Quat(1.0f,0.0f,0.0f,0.0f);
			Vector3 scaling  = Vector3(1.0f);
		};
	}
}
#endif

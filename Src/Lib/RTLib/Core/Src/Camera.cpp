#include <RTLib/Core/Camera.h>
RTLib::Core::Camera::Camera(
	const Vector3& scaling_, 
	const Quat& rotation_,
	const Vector3& position_,
	Float32 fovy_,
	Float32 aspect_,
	Float32 zNear_,
	Float32 zFar_) noexcept
	:scaling{scaling_}
	,rotation{rotation_}
	,position{position_}
	,fovY{fovy_}
	,aspect{aspect_}
	,zNear{zNear_}
	,zFar{zFar_}{}
//Proj * View * Model
auto RTLib::Core::Camera::transform_camera_to_world_point(const Vector3& p) const noexcept -> Vector3
{
	// T * R * S
	return (rotation *(scaling * p)) + position;
}

auto RTLib::Core::Camera::transform_camera_to_world_vector(const Vector3& v) const noexcept -> Vector3
{
	// T * R * S
	return (rotation * (scaling * v));
}

auto RTLib::Core::Camera::transform_camera_to_world_direction(const Vector3& d) const noexcept -> Vector3
{
	return rotation * d;
}

auto RTLib::Core::Camera::transform_world_to_camera_point(const Vector3& p) const noexcept -> Vector3
{
	return (glm::inverse(rotation) * (p - position))/scaling;
}

auto RTLib::Core::Camera::transform_world_to_camera_vector(const Vector3& v) const noexcept -> Vector3
{
	return  (glm::inverse(rotation) * v) / scaling;
}

auto RTLib::Core::Camera::transform_world_to_camera_direction(const Vector3& d) const noexcept -> Vector3
{
	return glm::inverse(rotation) * d;
}

auto RTLib::Core::Camera::set_front(Vector3 front, const Vector3& vup) noexcept -> Camera&
{
	using namespace glm;
	auto len   = glm::length(front) / scaling.z;
	front      = glm::normalize(front);
	auto right = glm::normalize(cross(vup,   front));
	auto up    = glm::normalize(cross(front, right));
	rotation   = toQuat(Matrix3x3(right, up, front));
	scaling    = scaling * len;
	return *this;
}

RTLib::Core::Matrix4x4 RTLib::Core::Camera::get_view_matrix() const noexcept
{
	//     V * p= T * R * S * p = (R * (s * p)) + t
	// INV_V * p= S^* R^* T^* p = (S^* R^)*(p - t)
	// INV_V * p=               = ((R^)*(p - t))/s
	// world to local matrix
	return glm::scale(glm::identity<Matrix4x4>(), Vector3(1.0f) / scaling) *
		   glm::inverse(glm::toMat4(rotation)) *
		   glm::translate(glm::identity<Matrix4x4>(), -position);
}

RTLib::Core::Matrix4x4 RTLib::Core::Camera::get_proj_matrix() const noexcept
{
	return glm::perspective(glm::radians(fovY), aspect, zNear, zFar);
}

void RTLib::Core::to_json(Json& json, const Camera& camera)
{
	json = {
		{"position",camera.position },
		{"scaling" ,camera.scaling  },
		{"rotation",camera.rotation },
		{"aspect"  ,camera.aspect   },
		{"fovY"    ,camera.fovY     },
		{"zNear"   ,camera.zNear    },
		{"zFar"    ,camera.zFar     }
		//{"rotation",camera.rotation }
	};
}

void RTLib::Core::from_json(const Json& json, Camera& camera)
{
	camera.position = json.at("position").get<Vector3>();
	camera.scaling  = json.at("scaling").get<Vector3>();
	camera.rotation = json.at("rotation").get<Quat>();
	camera.aspect   = json.at("aspect").get<Float32>();
	camera.fovY     = json.at("fovY").get<Float32>();
	camera.zNear    = json.at("zNear").get<Float32>();
	camera.zFar     = json.at("zFar").get<Float32>();

}

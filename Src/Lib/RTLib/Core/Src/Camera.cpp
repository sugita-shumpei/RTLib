#include <RTLib/Core/Camera.h>
#include <glm/gtx/transform.hpp>

auto RTLib::Core::Camera::get_proj_matrix() const noexcept -> Matrix4x4
{
	if (m_Type == CameraType::ePerspective) {
		return glm::perspective(m_FieldOfViewOrOrthographicsSize, m_Aspect, m_NearClipPlane, m_FarClipPlane);
	}
	else {
		const float height = m_FieldOfViewOrOrthographicsSize;
		const float width  = m_Aspect * height;
		return glm::orthoZO(-width, width, -height, height, m_NearClipPlane, m_FarClipPlane);
	}
}
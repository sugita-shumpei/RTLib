#include <RTLib/Core/Transform.h>

auto RTLib::Core::Transform::get_local_to_parent_matrix() const noexcept -> Matrix4x4
{
	return glm::translate(glm::mat4(1.0f), position) * glm::toMat4(rotation) * glm::scale(glm::mat4(1.0f), scaling);
}

auto RTLib::Core::Transform::get_parent_to_local_matrix() const noexcept -> Matrix4x4
{
	// LP = q * (s * LP) + t
	// WP = T+1 * R+1 * S+1
	// WP = ((LP - t) / q) *s
	// WP = S-1 * R-1 * T-1 * LP
	return glm::scale(glm::mat4(1.0f), 1.0f / scaling) * glm::toMat4(glm::inverse(rotation)) * glm::translate(glm::mat4(), -position);
}

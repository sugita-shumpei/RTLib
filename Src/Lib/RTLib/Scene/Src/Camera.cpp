#include <RTLib/Scene/Camera.h>
#include <RTLib/Scene/Object.h>
auto RTLib::Scene::Camera::New(std::shared_ptr<Scene::Object> object) -> std::shared_ptr<Scene::Camera>
{
	return std::shared_ptr<Scene::Camera>(new Scene::Camera(object));
}
RTLib::Scene::Camera::Camera(std::shared_ptr<Scene::Object> object)
	:m_Object{ object }
{
}

RTLib::Scene::Camera::~Camera() noexcept
{
}

auto RTLib::Scene::Camera::query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object>
{
	
	if (typeID == ObjectTypeID_Unknown || typeID == ObjectTypeID_SceneComponent || typeID == ObjectTypeID_SceneCamera)
	{
		return shared_from_this();
	}
	else {
		return nullptr;
	}
}

auto RTLib::Scene::Camera::get_type_id() const noexcept -> TypeID
{
	return ObjectTypeID_SceneCamera;
}

auto RTLib::Scene::Camera::get_name() const noexcept -> String
{
	auto object = m_Object.lock();
	if (object) {
		return object->get_name();
	}
	else {
		return "";
	}
}

auto RTLib::Scene::Camera::get_transform() -> std::shared_ptr<Scene::Transform>
{
	return internal_get_transform();
}

auto RTLib::Scene::Camera::get_object() -> std::shared_ptr<Scene::Object>
{
	return internal_get_object();
}

auto RTLib::Scene::Camera::get_position() const noexcept -> Vector3
{
	auto transform = internal_get_transform();
	if (transform) {
		return transform->get_position();
	}
	else {
		return Vector3(0.0f);
	}
}

auto RTLib::Scene::Camera::get_rotation() const noexcept -> Quat
{
	auto transform = internal_get_transform();
	if (transform) {
		return transform->get_rotation();
	}
	else {
		return Quat(1.0f, 0.0f, 0.0f, 0.0f);
	}
}

auto RTLib::Scene::Camera::get_scaling() const noexcept -> Vector3
{
	auto transform = internal_get_transform();
	if (transform) {
		return transform->get_scaling();
	}
	else {
		return Vector3(1.0f, 1.0f, 1.0f);
	}
}

auto RTLib::Scene::Camera::get_proj_matrix() const noexcept -> Matrix4x4
{
	return m_Camera.get_proj_matrix();
}

auto RTLib::Scene::Camera::get_view_matrix() const noexcept -> Matrix4x4
{
	auto transform = internal_get_transform();
	if (transform) {
		return transform->get_wolrd_to_local_matrix();
	}
	else {
		return Matrix4x4(1.0f);
	}
}

auto RTLib::Scene::Camera::internal_get_transform() const noexcept -> std::shared_ptr<Scene::Transform>
{
	auto object = internal_get_object();
	if (object) {
		return object->get_transform();
	}
	else {
		return nullptr;
	}
}

auto RTLib::Scene::Camera::internal_get_object() const noexcept -> std::shared_ptr<Scene::Object>
{
	return m_Object.lock();
}

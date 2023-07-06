#include <RTLib/Scene/Mesh.h>
#include <RTLib/Scene/Object.h>
RTLib::Scene::Mesh::Mesh(std::shared_ptr<Scene::Object> object)
	:m_SceneObject{ object }
{
}
auto RTLib::Scene::Mesh::New(std::shared_ptr<Scene::Object> object)->std::shared_ptr<Scene::Mesh>
{
	return std::shared_ptr<Scene::Mesh>(new Scene::Mesh(object));
}
RTLib::Scene::Mesh::~Mesh() noexcept
{

}

auto RTLib::Scene::Mesh::query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object>
{
	if (typeID == ObjectTypeID_Unknown || typeID == ObjectTypeID_SceneComponent || typeID == ObjectTypeID_SceneMesh)
	{
		return shared_from_this();
	}
	else {
		return nullptr;
	}
}

auto RTLib::Scene::Mesh::get_transform() -> std::shared_ptr<Scene::Transform>
{
	auto object = get_object();
	if (object) {
		return object->get_transform();
	}
	else {
		return nullptr;
	}
}

auto RTLib::Scene::Mesh::get_object() -> std::shared_ptr<Scene::Object>
{
	return m_SceneObject.lock();
}

auto RTLib::Scene::Mesh::get_type_id() const noexcept -> TypeID
{
	return ObjectTypeID_SceneMesh;
}

auto RTLib::Scene::Mesh::get_name() const noexcept -> String
{
	if (mesh) {
		return mesh->get_name();
	}
	else {
		return "";
	}
}

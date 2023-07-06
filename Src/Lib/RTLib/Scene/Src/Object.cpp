#include <RTLib/Scene/Object.h>
#include <RTLib/Scene/Transform.h>
#include <vector>
#include <stack>
RTLib::Scene::Object::Object(String name) noexcept
	:m_Name{ name }
{}

auto RTLib::Scene::Object::New(String name, const Core::Transform& transform) -> std::shared_ptr<Scene::Object>
{
	auto object = std::shared_ptr<Scene::Object>(new Scene::Object(name));
	auto transformPtr = std::shared_ptr<Scene::Transform>(new Scene::Transform(object, transform));
	object->internal_set_transform(transformPtr);
	return object;
}

RTLib::Scene::Object::~Object() noexcept
{
}

auto RTLib::Scene::Object::get_transform() const noexcept -> std::shared_ptr<Scene::Transform>
{
	return m_Transform;
}

auto RTLib::Scene::Object::query_object(const TypeID& typeID) -> std::shared_ptr<Core::Object>
{
	if (typeID == ObjectTypeID_Unknown || typeID == ObjectTypeID_SceneObject)
	{
		return shared_from_this();
	}
	return nullptr;
}
// Type ID
auto RTLib::Scene::Object::get_type_id() const noexcept -> TypeID {
	return ObjectTypeID_SceneObject;
}
// Name
auto RTLib::Scene::Object::get_name() const noexcept -> String
{
	return m_Name;
}

void RTLib::Scene::Object::internal_set_transform(std::shared_ptr<Scene::Transform> transform)
{
	m_Transform = transform;
}

auto RTLib::Scene::Object::internal_get_component(ObjectTypeID typeID) const noexcept ->  std::shared_ptr<Scene::Component>
{
	if (typeID == ObjectTypeID_Unknown) { return std::shared_ptr<Scene::Component>(); }
	if (typeID == ObjectTypeID_SceneTransform) { return get_transform(); }
	for (auto& comp : m_Components) {
		auto component = comp->query_object(typeID);
		if (component) {
			return std::static_pointer_cast<Scene::Component>(component);
		}
	}
	return nullptr;
}

auto RTLib::Scene::Object::internal_get_components(ObjectTypeID typeID) const noexcept ->  std::vector<std::shared_ptr<Scene::Component>>
{
	if (typeID == ObjectTypeID_Unknown) { return  std::vector<std::shared_ptr<Scene::Component>>(); }
	if (typeID == ObjectTypeID_SceneTransform) {
		auto transform = get_transform();
		if (transform) {
			return  std::vector<std::shared_ptr<Scene::Component>>{transform};
		}
		else {
			return  std::vector<std::shared_ptr<Scene::Component>>();
		}
	}
	std::vector<std::shared_ptr<Scene::Component>> res = {};
	for (auto& comp : m_Components) {
		auto component = comp->query_object(typeID);
		if (component) {
			res.push_back(std::static_pointer_cast<Scene::Component>(component));
		}
	}
	return res;
}

auto RTLib::Scene::Object::internal_get_transforms_in_children() const noexcept -> std::vector<std::shared_ptr<Scene::Transform>>
{
	auto transform = get_transform();
	if (transform) {
		std::stack<std::shared_ptr<Scene::Transform>> stack;
		stack.push(transform);
		std::vector<std::shared_ptr<Scene::Transform>> res{ transform };
		while (!stack.empty())
		{
			auto tmpTransform = stack.top();
			stack.pop();
			if (tmpTransform) {
				auto numChildren = tmpTransform->get_child_count();
				res.reserve(res.size() + numChildren);
				for (auto i = 0; i < numChildren; ++i) {
					auto tmpChildTransform = tmpTransform->get_child(i);
					if (tmpChildTransform) {
						stack.push(tmpChildTransform);
						res.push_back(tmpChildTransform);
					}
				}

			}
		}
		return res;
	}
	else {
		return {};
	}
}

auto RTLib::Scene::Object::internal_get_components_in_children(ObjectTypeID typeID) const noexcept ->  std::vector < std::shared_ptr<Scene::Component>> {
	if (typeID == ObjectTypeID_Unknown) { return  std::vector<std::shared_ptr<Scene::Component>>(); }
	if (typeID == ObjectTypeID_SceneTransform) {
		auto transforms = internal_get_transforms_in_children();
		auto res = std::vector<std::shared_ptr<Scene::Component>>();
		res.reserve(transforms.size());
		std::copy(std::begin(transforms), std::end(transforms), std::back_inserter(res));
		return res;
	}
	std::vector<std::shared_ptr<Scene::Component>> res = internal_get_components(typeID);
	auto transforms = internal_get_transforms_in_children();
	for (auto& transform : transforms)
	{
		if (!transform) {
			continue;
		}
		auto object = transform->get_object();
		if (!object) {
			continue;
		}
		auto tmp = object->internal_get_components(typeID);
		if (!tmp.empty()) {
			res.reserve(res.size() + tmp.size());
			std::copy(std::begin(tmp), std::end(tmp), std::back_inserter(res));
		}
	}
	return res;
}

auto RTLib::Scene::Object::internal_get_components_in_parent(ObjectTypeID typeID) const noexcept ->  std::vector < std::shared_ptr<Scene::Component>> {
	auto transform = get_transform();
	if (!transform) { return {}; }
	auto parent = transform->get_parent();
	if (!parent) {
		return internal_get_components_in_children(typeID);
	}
	else {

		auto parentObject = parent->get_object();
		if (parentObject) {
			return parentObject->internal_get_components_in_children(typeID);
		}
	}

	return {};
}

void RTLib::Scene::Object::internal_remove_component(ObjectTypeID typeID)noexcept
{
	if (typeID == ObjectTypeID_SceneTransform)
	{
		return;
	}
	else {
		size_t removeIdx = SIZE_MAX;
		for (auto i = 0; i < m_Components.size(); ++i) {
			if (m_Components[i]->get_type_id() == typeID)
			{
				removeIdx = i;
				break;
			}
		}
		if (removeIdx != SIZE_MAX)
		{
			m_Components.erase(std::begin(m_Components) + removeIdx);
		}
	}
}
auto RTLib::Scene::Object::internal_get_object() noexcept -> std::shared_ptr<Scene::Object>
{
	return std::static_pointer_cast<Scene::Object>(shared_from_this());
}
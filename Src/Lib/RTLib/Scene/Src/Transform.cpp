#include <RTLib/Scene/Transform.h>
#include <RTLib/Scene/TransformGraph.h>
#include <RTLib/Scene/Component.h>
#include <RTLib/Scene/Object.h>
RTLib::Scene::Transform::Transform(std::shared_ptr<Scene::Object> object, const Core::Transform& localTransform)
	:m_Object{ object }, m_Local{ localTransform }, m_Node{}
{
}

RTLib::Scene::Transform::~Transform() noexcept
{
}

auto RTLib::Scene::Transform::get_local_to_parent_matrix() const noexcept -> Matrix4x4
{
	return m_Local.get_local_to_parent_matrix();
}

auto RTLib::Scene::Transform::get_parent_to_local_matrix() const noexcept -> Matrix4x4
{
	return m_Local.get_parent_to_local_matrix();
}

auto RTLib::Scene::Transform::get_local_to_world_matrix() const noexcept -> Matrix4x4
{
	auto node = internal_get_node();
	if (node) {
		return node->get_cache_transform_point_matrix();
	}
	else {
		return get_local_to_parent_matrix();
	}
}

auto RTLib::Scene::Transform::get_wolrd_to_local_matrix() const noexcept -> Matrix4x4
{
	auto node = internal_get_node();
	if (node) {
		return glm::inverse(node->get_cache_transform_point_matrix());
	}
	else {
		return get_parent_to_local_matrix();
	}
}

void RTLib::Scene::Transform::set_local_position(const Vector3& localPosition) noexcept
{
	m_Local.position = localPosition;
	internal_update_cache();
}

void RTLib::Scene::Transform::set_local_rotation(const Quat& localRotation) noexcept
{
	m_Local.rotation = localRotation;
	internal_update_cache();
}

void RTLib::Scene::Transform::set_local_scaling(const Vector3& localScaling) noexcept
{
	m_Local.scaling = localScaling;
	internal_update_cache();
}

auto RTLib::Scene::Transform::get_position() const noexcept -> Vector3
{
	// T*R*S
	auto cache_transform = internal_get_parent_cache_point_matrix();
	auto pos = cache_transform * Vector4(get_local_position(), 1.0f);
	return Vector3(pos);
}

auto RTLib::Scene::Transform::get_rotation() const noexcept -> Quat
{
	auto cache_transform = internal_get_parent_cache_direction_matrix();
	auto rot = glm::toQuat(cache_transform) * get_local_rotation();
	return rot;
}

auto RTLib::Scene::Transform::get_scaling() const noexcept -> Vector3
{
	auto cache_scaling = internal_get_parent_cache_scaling();
	auto scl = cache_scaling * get_local_scaling();
	return scl;
}

void RTLib::Scene::Transform::set_position(const Vector3& position) noexcept
{
	auto localPosition = glm::inverse(internal_get_parent_cache_point_matrix()) * Vector4(position, 1.0f);
	set_local_position(localPosition);
}

void RTLib::Scene::Transform::set_rotation(const Quat& rotation) noexcept
{
	// T * R * S * (T1 * R1 * S1)
	auto localRotation = glm::inverse(glm::toQuat(internal_get_parent_cache_direction_matrix())) * rotation;
	set_local_rotation(localRotation);
}

void RTLib::Scene::Transform::set_scaling(const Vector3& scaling) noexcept
{
	auto localScaling = scaling / internal_get_parent_cache_scaling();
	set_local_scaling(localScaling);
}

auto RTLib::Scene::Transform::get_transform() -> std::shared_ptr<Scene::Transform>
{
	return std::static_pointer_cast<Scene::Transform>(shared_from_this());
}

auto RTLib::Scene::Transform::get_object() -> std::shared_ptr<Scene::Object>
{
	return m_Object.lock();
}

auto RTLib::Scene::Transform::get_child_count() const noexcept -> UInt32
{
	auto node = m_Node.lock();
	if (node) {
		return node->get_num_children();
	}
	else {
		return 0;
	}
}

auto RTLib::Scene::Transform::get_child(UInt32 idx) const noexcept -> std::shared_ptr<Scene::Transform>
{
	auto node = m_Node.lock();
	if (node) {
		auto childNode = node->get_child(idx);
		return childNode->get_transform();
	}
	else {
		return  std::shared_ptr<Scene::Transform>();
	}

}

auto RTLib::Scene::Transform::get_parent() const noexcept -> std::shared_ptr<Scene::Transform>
{
	auto node = m_Node.lock();
	if (!node) {
		return std::shared_ptr<Scene::Transform>();
	}
	auto parentNode = node->get_parent();
	if (parentNode) {
		return parentNode->get_transform();
	}
	return std::shared_ptr<Scene::Transform>();
}

void RTLib::Scene::Transform::set_parent(std::shared_ptr<Scene::Transform> parent)
{
	if (!m_Node.expired())
	{
		// 古いノードを削除する必要有
		auto oldNode = m_Node.lock();
		auto parentNode = oldNode->get_parent();
		if (parentNode) {
			parentNode->remove_child(oldNode);
		}
		m_Node = {};
	}
	// 新しいSceneTransformGraphNodeを作成して追加
	auto parentNode = parent->m_Node.lock();
	if (!parentNode) { return; }

	auto newNode = parentNode->attach_child(get_transform());
	m_Node = newNode;
}

auto RTLib::Scene::Transform::internal_get_node() const noexcept-> std::shared_ptr<Scene::TransformGraphNode>
{
	return m_Node.lock();
}
auto RTLib::Scene::Transform::internal_get_parent_node() const noexcept-> std::shared_ptr<Scene::TransformGraphNode>
{
	auto node = internal_get_node();
	if (node) {
		return node->get_parent();
	}
	else {
		return nullptr;
	}
}
void RTLib::Scene::Transform::internal_set_node(std::shared_ptr<Scene::TransformGraphNode> node)
{
	m_Node = node;
}

auto RTLib::Scene::Transform::internal_get_parent_cache_point_matrix() const noexcept -> Matrix4x4
{
	auto node = internal_get_parent_node();
	if (node) {
		return node->get_cache_transform_point_matrix();
	}
	else {
		return Matrix4x4(1.0f);
	}
}
auto RTLib::Scene::Transform::internal_get_parent_cache_vector_matrix() const noexcept -> Matrix4x4 {
	auto node = internal_get_parent_node();
	if (node) {
		return node->get_cache_transform_vector_matrix();
	}
	else {
		return Matrix4x4(1.0f);
	}
}
auto RTLib::Scene::Transform::internal_get_parent_cache_direction_matrix() const noexcept -> Matrix4x4
{
	auto node = internal_get_parent_node();
	if (node) {
		return node->get_cache_transform_direction_matrix();
	}
	else {
		return Matrix4x4(1.0f);
	}
}
auto RTLib::Scene::Transform::internal_get_parent_cache_scaling() const noexcept -> Vector3
{
	auto node = internal_get_parent_node();
	if (node) {
		return node->get_cache_transform_scaling();
	}
	else {
		return Vector3(1.0f);
	}
}
void RTLib::Scene::Transform::internal_update_cache()
{
	auto node = internal_get_node();
	if (node) {
		node->update_cache_transform();
	}
}

auto RTLib::Scene::Transform::query_object(const TypeID& typeID) -> std::shared_ptr<Object>
{
	if (typeID == ObjectTypeID_SceneObject || typeID == ObjectTypeID_SceneComponent || typeID == ObjectTypeID_SceneTransform)
	{
		return shared_from_this();
	}
	return std::shared_ptr<Object>();
}
// Type ID
auto RTLib::Scene::Transform::get_type_id() const noexcept -> TypeID {
	return ObjectTypeID_SceneTransform;
}
// Name
auto RTLib::Scene::Transform::get_name() const noexcept -> String
{
	auto object = m_Object.lock();
	if (object) {
		return object->get_name();
	}
	else {
		return "";
	}
}
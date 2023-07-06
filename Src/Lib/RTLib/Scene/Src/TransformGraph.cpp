#include <RTLib/Scene/TransformGraph.h>
#include <RTLib/Scene/Transform.h>
#include <RTLib/Scene/Object.h>

RTLib::Scene::TransformGraphNode::TransformGraphNode(Scene::TransformPtr transform, std::shared_ptr<Scene::TransformGraphNode> parent)
    :m_Parent{ parent }, m_Children{}, m_Object{},
    m_CacheTransformPointMatrix{ 1.0f },
    m_CacheTransformVectorMatrix{ 1.0f },
    m_CacheTransformDirectionMatrix{ 1.0f },
    m_CacheTransformScaling{ 1.0f } {
    if (!transform) {
        return;
    }
    // 古いノードが割り当てられている場合
    // 割り当てを削除
    {
        auto oldNode = transform->internal_get_node();
        if (oldNode) {
            auto oldParent = oldNode->get_parent();
            if (oldParent) {
                oldParent->remove_child(oldNode);
            }
        }
    }
    {
        auto object = transform->get_object();
        if (object) {
            m_Object = object;
        }
    }
}

auto RTLib::Scene::TransformGraphNode::New(std::shared_ptr<Scene::Transform> transform, std::shared_ptr<Scene::TransformGraphNode> parent) -> std::shared_ptr<TransformGraphNode>
{
    auto res = std::shared_ptr<Scene::TransformGraphNode>(new TransformGraphNode(transform, parent));
    if (parent) {
        parent->m_Children.push_back(res);
    }
    if (transform) {
        transform->internal_set_node(res);
    }
    return res;
}

RTLib::Scene::TransformGraphNode::~TransformGraphNode() noexcept
{
}


auto RTLib::Scene::TransformGraphNode::get_transform() const noexcept -> std::shared_ptr<Scene::Transform>
{
    if (m_Object) { return m_Object->get_transform(); }
    return std::shared_ptr<Scene::Transform>();
}

auto RTLib::Scene::TransformGraphNode::attach_child(std::shared_ptr<Scene::Transform> transform) -> std::shared_ptr<Scene::TransformGraphNode>
{
    if (!transform) { return  std::shared_ptr<TransformGraphNode>(); }
    return TransformGraphNode::New(transform, shared_from_this());
}

auto RTLib::Scene::TransformGraphNode::remove_child(std::shared_ptr<TransformGraphNode> node) -> Scene::TransformPtr
{
    auto idx = 0;
    for (auto i = 0; i < m_Children.size(); ++i) {
        if (m_Children.at(i) == node) {
            idx = i;
            break;
        }
    }
    if (idx != m_Children.size()) {
        auto transform = m_Children[idx]->get_transform();
        m_Children.erase(std::begin(m_Children) + idx);
        transform->internal_set_node(nullptr);
        return transform;
    }
    return nullptr;
}

auto RTLib::Scene::TransformGraphNode::get_child(UInt32 idx) const noexcept -> std::shared_ptr<Scene::TransformGraphNode>
{
    if (idx < m_Children.size()) {
        return m_Children.at(idx);
    }
    return nullptr;
}

auto RTLib::Scene::TransformGraphNode::get_parent() const noexcept -> std::shared_ptr<Scene::TransformGraphNode>
{
    auto parent = m_Parent.lock();
    return parent;
}
auto RTLib::Scene::TransformGraphNode::get_local_to_parent_matrix() const noexcept -> Matrix4x4
{
    auto transform = get_transform();
    if (transform) { return transform->get_local_to_parent_matrix(); }
    else { return Matrix4x4(1.0f); }
}

auto RTLib::Scene::TransformGraphNode::get_local_position() const noexcept -> Vector3
{
    auto transform = get_transform();

    if (transform) { return transform->get_local_position(); }
    else { return Vector3(1.0f); }
}
auto RTLib::Scene::TransformGraphNode::get_local_rotation() const noexcept -> Quat
{

    auto transform = get_transform();
    if (transform) { return transform->get_local_rotation(); }
    else { return Quat(1.0f, 0.0f, 0.0f, 0.0f); }
}
auto RTLib::Scene::TransformGraphNode::get_local_scaling()  const noexcept -> Vector3
{
    auto transform = get_transform();
    if (transform) { return transform->get_local_scaling(); }
    else { return Vector3(1.0f); }
}

auto RTLib::Scene::TransformGraphNode::get_cache_transform_point_matrix() const noexcept -> Matrix4x4
{
    return m_CacheTransformPointMatrix;
}

auto RTLib::Scene::TransformGraphNode::get_cache_transform_vector_matrix() const noexcept -> Matrix4x4
{
    return m_CacheTransformVectorMatrix;
}
auto RTLib::Scene::TransformGraphNode::get_cache_transform_direction_matrix() const noexcept -> Matrix4x4
{
    return m_CacheTransformDirectionMatrix;
}
auto RTLib::Scene::TransformGraphNode::get_cache_transform_scaling() const noexcept -> Vector3
{
    return m_CacheTransformScaling;
}
void RTLib::Scene::TransformGraphNode::update_cache_transform() noexcept
{
    auto parent = get_parent();
    auto parentCacheTransformPointMatrix = Matrix4x4(1.0f);
    auto parentCacheTransformVectorMatrix = Matrix4x4(1.0f);
    auto parentCacheTransformDirectionMatrix = Matrix4x4(1.0f);
    auto parentCacheTransformScaling = Vector3(1.0f);
    if (parent) {
        parentCacheTransformPointMatrix = parent->get_cache_transform_point_matrix();
        parentCacheTransformVectorMatrix = parent->get_cache_transform_vector_matrix();
        parentCacheTransformDirectionMatrix = parent->get_cache_transform_direction_matrix();
        parentCacheTransformScaling = parent->get_cache_transform_scaling();
    }
    auto transform = get_transform();
    if (!transform) { return; }
    m_CacheTransformPointMatrix = parentCacheTransformPointMatrix * transform->get_local_to_parent_matrix();
    m_CacheTransformVectorMatrix = parentCacheTransformVectorMatrix * glm::toMat4(transform->get_local_rotation()) * glm::scale(Matrix4x4(1.0f), transform->get_local_scaling());
    m_CacheTransformDirectionMatrix = parentCacheTransformDirectionMatrix * glm::toMat4(transform->get_local_rotation());
    m_CacheTransformScaling = parentCacheTransformScaling * transform->get_local_scaling();
    for (auto& child : m_Children) {
        child->update_cache_transform();
    }
}
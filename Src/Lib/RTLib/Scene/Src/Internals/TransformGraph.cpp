#include "TransformGraph.h"
#include "TransformGraphNode.h"
RTLib::Scene::Internals::TransformGraph::TransformGraph()  noexcept :m_Root{ RTLib::Scene::Internals::TransformGraphNode::New() } {}
RTLib::Scene::Internals::TransformGraph::~TransformGraph() noexcept {}

void RTLib::Scene::Internals::TransformGraph::attach_child(std::shared_ptr<RTLib::Scene::Transform> transform)
{
    if (!transform) { return; }
    m_Root->attach_child(transform);
}

auto RTLib::Scene::Internals::TransformGraph::get_num_children() const noexcept    -> UInt32 { return m_Root->get_num_children(); }
auto RTLib::Scene::Internals::TransformGraph::get_child(UInt32 idx) const noexcept -> std::shared_ptr<Scene::Transform>
{
    auto child = m_Root->get_child(idx);
    if (child) {
        return child->get_transform();
    }
    else {
        return nullptr;
    }
}
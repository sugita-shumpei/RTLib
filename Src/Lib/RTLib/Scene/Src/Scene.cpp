#include <RTLib/Scene/Scene.h>
#include "Internals/TransformGraph.h"


struct RTLib::Scene::Scene::Impl {
	Internals::TransformGraph graph;
};

RTLib::Scene::Scene:: Scene() noexcept : m_Impl{ new Impl() } {}
RTLib::Scene::Scene::~Scene() noexcept { m_Impl.reset(); }

void RTLib::Scene::Scene::attach_child(std::shared_ptr<RTLib::Scene::Transform>     transform)
{
	m_Impl->graph.attach_child(transform);
}

auto RTLib::Scene::Scene::get_num_children() const noexcept    -> UInt32
{
	return m_Impl->graph.get_num_children();
}

auto RTLib::Scene::Scene::get_child(UInt32 idx) const noexcept -> std::shared_ptr<RTLib::Scene::Transform>
{
	return m_Impl->graph.get_child(idx);
}

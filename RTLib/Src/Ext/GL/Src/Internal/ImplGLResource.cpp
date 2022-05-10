#include "ImplGLResource.h"

RTLib::Ext::GL::Internal::ImplGLResourceTable::~ImplGLResourceTable() noexcept {
	/*m_Resources is Read/Write on Destructor*/
	auto tempResources = m_Resources;
	for (auto& resource : tempResources) {
		if (resource) {
			resource->Destroy();
		}
	}
	m_Resources.clear();
}

bool RTLib::Ext::GL::Internal::ImplGLResourceTable::Register(ImplGLResource* resource)
{
	if (!resource) { return false; }
	if (m_Resources.count(resource) > 0) {
		return false;
	}
	m_Resources.insert(resource);
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLResourceTable::Unregister(ImplGLResource* resource)
{
	if (!resource) { return false; }
	if (m_Resources.count(resource) == 0) {
		return false;
	}
	m_Resources.erase(resource);

	return true;
}

RTLib::Ext::GL::Internal::ImplGLResource::~ImplGLResource() noexcept {
	Destroy();
}

bool RTLib::Ext::GL::Internal::ImplGLResource::Create() {
	if (!m_Table || !m_Base) {
		return false;
	}
	m_Base->Create();
	m_Table->Register(this);
	return true;
}

void RTLib::Ext::GL::Internal::ImplGLResource::Destroy() noexcept {
	if (m_Table) {
		m_Table->Unregister(this);
		m_Table = nullptr;
	}
	if (m_Base) {
		m_Base->Destroy();
		m_Base.reset();
	}
}

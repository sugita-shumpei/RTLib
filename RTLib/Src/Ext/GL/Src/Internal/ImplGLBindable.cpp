#include "ImplGLBindable.h"

bool RTLib::Ext::GL::Internal::ImplGLBindingPoint::Register(GLenum target, ImplGLBindable* bindable)
{
	if (!HasTarget(target)||!bindable) {
		return false;
	}
	if (GetBindable(target)) {
		return false;
	}
	m_Bindables[target] = bindable;
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBindingPoint::Unregister(GLenum target)
{
	if (!HasTarget(target)) {
		return false;
	}
	if (!GetBindable(target)) {
		return false;
	}
	m_Bindables[target] = nullptr;
	return true;
}

auto RTLib::Ext::GL::Internal::ImplGLBindingPoint::GetBindable(GLenum target) -> ImplGLBindable*
{
	return m_Bindables.count(target) > 0?m_Bindables.at(target):nullptr;
}
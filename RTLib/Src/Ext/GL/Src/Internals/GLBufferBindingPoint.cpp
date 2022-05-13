#include "GLBufferBindingPoint.h"
bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::TakeTargetBinding(GLenum target, GLBufferBindable *bindable)
{
	return false;
}

void RTLib::Ext::GL::Internals::GLBufferBindingPoint::ReleaseTargetBinding(GLenum target)
{
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::ResetTargetBinding(GLenum target)
{
	return false;
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::TakeRangeBinding(GLenum target, GLuint index, GLBufferBindable *bindable, const GLBufferBindingRange &range)
{
	return false;
}

void RTLib::Ext::GL::Internals::GLBufferBindingPoint::ReleaseRangeBinding(GLenum target, GLuint index)
{
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::ResetRangeBinding(GLenum target, GLuint index)
{
	return false;
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::IsBindedTarget(GLenum target) const noexcept
{
	return false;
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::IsBindedIndex(GLenum target, GLuint index)
{
	return false;
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::IsBindedRange(GLenum target, GLuint index, const GLBufferBindingRange &range)
{
	return false;
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::IsValidTarget(GLenum target)const noexcept
{
	return m_TargetHandles.count(target) > 0;
}

bool RTLib::Ext::GL::Internals::GLBufferBindingPoint::IsValidRangeTarget(GLenum target) const noexcept
{
	return m_RangesHandles.count(target) > 0;
}

auto RTLib::Ext::GL::Internals::GLBufferBindingPoint::EnumerateValidTargets() const noexcept -> std::vector<GLenum>
{
	std::vector<GLenum> validTargets = {};
	for (auto& [target, handle] : m_TargetHandles) {
		validTargets.push_back(target);
	}
	return validTargets;
}

auto RTLib::Ext::GL::Internals::GLBufferBindingPoint::EnumerateValidTargetRanges() const noexcept -> std::vector<std::pair<GLenum, size_t>>
{
	std::vector<std::pair<GLenum, size_t>> validTargetRanges = {};
	for (auto& [target, handles] : m_RangesHandles) {
		validTargetRanges.push_back({ target,handles.size() });
	}
	return validTargetRanges;
}

auto RTLib::Ext::GL::Internals::GLBufferBindingPoint::EnumerateRangeCapacity(GLenum target) const noexcept -> size_t
{
	if (!IsValidRangeTarget(target)) { return 0; }
	else {
		return m_RangesHandles.at(target).size();
	}
}

void RTLib::Ext::GL::Internals::GLBufferBindingPoint::AddValidTarget(GLenum target) noexcept
{
	m_TargetHandles[target] = {};
}

void RTLib::Ext::GL::Internals::GLBufferBindingPoint::AddValidTargetRange(GLenum target, size_t maxRangeCount) noexcept
{
	m_RangesHandles[target] = {};
}

RTLib::Ext::GL::Internals::GLBufferBindable::~GLBufferBindable() noexcept {}

RTLib::Ext::GL::Internals::GLBufferBindable::GLBufferBindable(GLuint resId, RTLib::Ext::GL::Internals::GLBufferBindingPoint* bufferBP):m_ResId{resId}
{
	if (bufferBP) {
		auto validTargetRanges = bufferBP->EnumerateValidTargetRanges();
		for (auto& [target,capacity] : validTargetRanges)
		{
			m_Ranges[target]          = {};
			m_Ranges[target].capacity = capacity;
		}
	}
}

auto RTLib::Ext::GL::Internals::GLBufferBindable::GetBufferBP() const noexcept -> GLBufferBindingPoint*
{
	return m_BufferBP;
}

auto RTLib::Ext::GL::Internals::GLBufferBindable::GetResId() const noexcept -> GLuint
{
	return m_ResId;
}

auto RTLib::Ext::GL::Internals::GLBufferBindable::GetBindingTarget() const noexcept -> OpBindingTarget
{
	return m_Target;
}

auto RTLib::Ext::GL::Internals::GLBufferBindable::GetBindingRanges() const noexcept -> const OpBindingRanges&
{
	// TODO: return ステートメントをここに挿入します
	return m_Ranges;
}

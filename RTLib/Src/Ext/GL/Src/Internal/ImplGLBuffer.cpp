#include "ImplGLBuffer.h"
#include "ImplGLUtils.h"
#include <iostream>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				struct ImplGLBufferBindExtData {
					GLuint   index;
					GLsizei  size;
					GLintptr offset;
				};
				struct ImplGLBufferUnbindExtData {
					GLuint   index;
				};
				class ImplGLBufferBase : public ImplGLBindableBase {
				private:
					friend class ImplGLBuffer;
					struct BindRangeInfo {
						GLenum			  target;
						GLuint            index;
						ImplGLBufferRange range;
					};
				public:
					ImplGLBufferBase(ImplGLBufferBindingPointRange* bPointRange)noexcept:ImplGLBindableBase(),m_BPBufferRange{bPointRange}{}
					virtual ~ImplGLBufferBase()noexcept {}
					
					bool IsMapped()const noexcept { return m_IsMapped; }
					bool MapMemory(GLenum target, void** pMappedData, GLenum access) {
						if (IsMapped()) {
							return false;
						}
						*pMappedData = glMapBuffer(target, access);
						m_IsMapped = true;
						return true;
					}
					bool MapMemory(GLenum target, void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size) {
						if (IsMapped()) {
							return false;
						}
						*pMappedData = glMapBufferRange(target, offset, size, access);
						m_IsMapped = true;
						return true;
					}
					bool UnmapMemory(GLenum target) {
						if (!IsMapped()) {
							return false;
						}
						glUnmapBuffer(target);
						m_IsMapped = false;
						return true;
					}

					auto GetBindingPointRange()const noexcept -> const ImplGLBufferBindingPointRange* { return m_BPBufferRange; }
					auto GetBindingPointRange()      noexcept ->       ImplGLBufferBindingPointRange* { return m_BPBufferRange; }
					bool IsBindedRange(GLenum target, GLuint index) const noexcept
					{
						return std::find_if(std::begin(m_BindRanges), std::end(m_BindRanges), [target, index](const auto& rangeInfo) {
							return rangeInfo.target == target && rangeInfo.index == index;
						}) != std::end(m_BindRanges);
					}
					void EraseRange(GLenum target, GLuint index)noexcept {
						auto iter = std::find_if(std::begin(m_BindRanges), std::end(m_BindRanges), [target, index](const auto& rangeInfo) {
							return rangeInfo.target == target && rangeInfo.index == index;
						});
						if (iter != std::end(m_BindRanges)) {
							m_BindRanges.erase(iter);
						}
					}
					auto GetBindedRange(GLenum target, GLuint index) const noexcept -> std::optional<ImplGLBufferRange>
					{
						auto iter = std::find_if(std::begin(m_BindRanges), std::end(m_BindRanges), [target, index](const auto& rangeInfo) {
							return rangeInfo.target == target && rangeInfo.index == index;
							});
						if (iter != std::end(m_BindRanges)) {
							return iter->range;
						}
						else {
							return std::nullopt;
						}
					}
				protected:
					virtual bool      Create()noexcept override {
						GLuint resId;
						glGenBuffers(1, &resId);
						if (resId == 0) {
							return false;
						}
						SetResId(resId);
						m_IsMapped = false;
						return true;
					}
					virtual void     Destroy()noexcept {
						GLuint resId = GetResId();
						glDeleteBuffers(1, &resId);
						SetResId(0);
					}
					virtual void   Bind(GLenum target) {
						GLuint resId = GetResId();
						if (resId > 0) {
#ifndef NDEBUG
							std::cout << "BIND " << ToString(target) << ": " << GetName() << std::endl;
#endif
							glBindBuffer(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
#ifndef NDEBUG
						std::cout << "UNBIND " << ToString(target) << ": " << GetName() << std::endl;
#endif
						UnmapMemory(target);
#ifndef NDEBUG
						glBindBuffer(target, 0);
#endif
					}

					void   BindRange(GLenum target, GLuint index, GLsizei size, GLintptr offset) {
						GLuint resId = GetResId();
						if (resId > 0) {
#ifndef NDEBUG
							std::cout << "BIND RANGE " << ToString(target) << ": " << GetName() << std::endl;
#endif
							glBindBufferRange(target, index, resId, offset, size);

							m_BindRanges.push_back({ target,index, {size, offset} });
						}
					}
					void UnbindRange(GLenum target, GLuint index) {
#ifndef NDEBUG
						std::cout << "UNBIND BASE" << ToString(target) << ": " << GetName() << std::endl;
#endif
						glBindBufferBase(target, index, 0);
						EraseRange(target, index);
					}
				private:
					bool                           m_IsMapped      = false;
					ImplGLBufferBindingPointRange* m_BPBufferRange = nullptr;
					std::vector<BindRangeInfo>     m_BindRanges    = {};
				};
			}
		}
	}
}
RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::~ImplGLBufferBindingPointRange() noexcept {}
void RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::AddTarget(GLenum target, GLint numBindings) noexcept {
	if (!HasTarget(target)) {
		m_Handles[target] = {};
		m_Handles[target].resize(numBindings);
		for (auto i = 0; i < numBindings; ++i) {
			m_Handles[target][i] = {};
		}
	}
}
bool RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::HasTarget(GLenum target) const noexcept {
	return m_Handles.count(target) > 0;
}
bool RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::IsBindable(GLenum target, GLuint index) const noexcept {
	if (!HasTarget(target)) {
		return false;
	}
	if (m_Handles.at(target).size() >= index) {
		return false;
	}
	return m_Handles.at(target).at(index).buffer == nullptr;
}
auto RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::GetBindable(GLenum target, GLuint index) -> ImplGLBuffer* {
	if (!HasTarget(target)) {
		return nullptr;
	}
	if (m_Handles.at(target).size() >= index) {
		return nullptr;
	}
	return m_Handles.at(target).at(index).buffer;
}
bool RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::Register(GLenum target, GLuint index, ImplGLBuffer* buffer, GLsizei size, GLintptr offset)
{
if (!HasTarget(target) || !buffer) {
	return false;
}
if (!buffer->IsAllocated()) {
	return false;
}
if (GetBindable(target, index)) {
	return false;
}
m_Handles.at(target)[index].buffer = buffer;
m_Handles.at(target)[index].range.offset = offset;
m_Handles.at(target)[index].range.size = size;
return true;
}
bool RTLib::Ext::GL::Internal::ImplGLBufferBindingPointRange::Unregister(GLenum target, GLuint index)
{
	if (!HasTarget(target)) {
		return false;
	}
	if (!GetBindable(target, index)) {
		return false;
	}
	m_Handles[target][index] = {};
	return true;
}
auto RTLib::Ext::GL::Internal::ImplGLBuffer::New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, ImplGLBufferBindingPointRange* bPointRange, GLenum defaultTarget) -> ImplGLBuffer* {
	if (!table || !bPoint || !bPointRange) {
		return nullptr;
	}
	if (!bPoint->HasTarget(defaultTarget)) { return false; }
	auto buffer = new ImplGLBuffer(table, bPoint, defaultTarget);
	if (buffer) {
		buffer->InitBase<ImplGLBufferBase>(bPointRange);
		auto res = buffer->Create();
		if (!res) {
			delete buffer;
			return nullptr;
		}
	}
	return buffer;
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::Bind()
{
	return ImplGLBindable::Bind(m_DefaultTarget);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::Bind(GLenum target) {
	return ImplGLBindable::Bind(target);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::BindBase(GLuint index)
{
	return BindBase(m_DefaultTarget, index);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::BindBase(GLenum target, GLuint index)
{
	if (!IsAllocated()) { return false; }
	return BindRange(target, index, m_AllocationInfo->size, 0);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::BindRange(GLuint index, GLsizei size, GLintptr offset) {
	return BindRange(m_DefaultTarget, index, size, offset);
}
bool RTLib::Ext::GL::Internal::ImplGLBuffer::BindRange(GLenum target, GLuint index, GLsizei size, GLintptr offset)
{
	auto base = static_cast<ImplGLBufferBase*>(GetBase());
	if (!IsAllocated()) {
		return false;
	}
	auto bpBufferRange = base->GetBindingPointRange();
	if (!bpBufferRange) {
		return false;
	}
	auto bindedTarget = GetBindedTarget();
	bool shouldUnbind = false;
	if ( bindedTarget) {
		if (bindedTarget != target) {
			return false;
		}
	}
	else {
		shouldUnbind = true;
	}
	auto bindedRange = GetBindedRange(target, index);
	if (bindedRange) {
		return bindedRange->size == size && bindedRange->offset == offset;
	}
	if (m_AllocationInfo->size < size + offset) { return false; }

	ImplGLBufferBindExtData bindExtData = {};
	bindExtData.index = index;
	bindExtData.size = size;
	bindExtData.offset = offset;

	if (!bpBufferRange->Register(target, index, this, size, offset)) {
		return false;
	}
	base->BindRange(target, index, size, offset);
	if (shouldUnbind) {
		glBindBuffer(target,0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::UnbindBase(GLenum target, GLuint index)
{
	if (!IsAllocated()) {
		return false;
	}
	auto bindedTarget = GetBindedTarget();
	if ( bindedTarget) {
		return false;
	}
	else {
		if (!GetBindingPoint()->IsBindable(target)) {
			return false;
		}
	}
	auto bindedRange = GetBindedRange(target, index);
	if (!bindedRange) {
		return true;
	}

	ImplGLBufferUnbindExtData unbindExtData = {};
	unbindExtData.index = index;
	reinterpret_cast<ImplGLBufferBase*>(GetBase())->UnbindRange(target, index);
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::UnbindBase(GLuint index) {
	return UnbindBase(m_DefaultTarget, index);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::Allocate(GLenum usage, size_t size, const void* pInitialData)
{
	if (IsAllocated()) {
		return false;
	}
	bool bindForAllocated = false;
	if (IsBinded()) {
		if (GetBindedTarget() != GetDefTarget()) {
			return false;
		}
	}
	else {
		bindForAllocated = true;
	}
	if (bindForAllocated) {
		if (!Bind()) { return false; }
	}
	glBufferData(*GetBindedTarget(), size, pInitialData, usage);
	m_AllocationInfo = AllocationInfo{ size, usage };
	if (bindForAllocated) {
		Unbind();
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyFromMemory(const void* pSrcData, size_t size, size_t offset)
{
	if (!pSrcData || !IsAllocated() || IsMapped()) {
		return false;
	}
	bool isBindedCopyDst  = false;
	auto targetForCopyDst = GetBindedTarget();
	if (!targetForCopyDst) {
		 targetForCopyDst = GetBindableTargetForCopyDst();
		 if (!targetForCopyDst) { return false; }
	}
	else {
		isBindedCopyDst = true;
	}
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, GetResId());
	}
	glBufferSubData( *targetForCopyDst, offset, size, pSrcData);
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyToMemory(void* pDstData, size_t size, size_t offset)
{
	if (!pDstData || !IsAllocated() || IsMapped()) {
		return false;
	}
	bool isBindedCopySrc = false;
	auto targetForCopySrc= GetBindedTarget();
	if (!targetForCopySrc) {
		targetForCopySrc = GetBindableTargetForCopySrc();
		if (!targetForCopySrc) { return false; }
	}
	else {
		isBindedCopySrc  = true;
	}
	if ( !isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, GetResId());
	}
	glGetBufferSubData(*targetForCopySrc, offset, size, pDstData);
	if ( !isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset, size_t dstOffset)
{
	return CopyBuffer2Buffer(srcBuffer, this, size, srcOffset, dstOffset);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyToBuffer(  ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset, size_t srcOffset)
{
	return CopyBuffer2Buffer(this, dstBuffer, size, srcOffset, dstOffset);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::MapMemory(void** pMappedData, GLenum access)
{
	auto base = GetBase();
	if (!IsAllocated()||!pMappedData || !base) {
		return false;
	}
	GLenum target;
	if (!IsBinded()) {
		if (Bind()) {
			target = GetDefTarget();
		}
		else {
			if (access == GL_READ_ONLY) {
				if (!Bind(GL_COPY_READ_BUFFER)) {
					return false;
				}
				target = GL_COPY_READ_BUFFER;
			}
			else if (access == GL_WRITE_ONLY) {
				if (!Bind(GL_COPY_WRITE_BUFFER)) {
					return false;
				}
				target = GL_COPY_WRITE_BUFFER;
			}
			else {
				return false;
			}
		}
	}
	else {
		target = *GetBindedTarget();
	}
	return static_cast<ImplGLBufferBase*>(base)->MapMemory(target,pMappedData, access);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::MapMemory(void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size)
{
	auto base = GetBase();
	if (!IsAllocated() || !pMappedData || !base) {
		return false;
	}
	GLenum target;
	if (!IsBinded()) {
		if (Bind()) {
			target = GetDefTarget();
		}
		else {
			if ((access & GL_MAP_READ_BIT) == GL_MAP_READ_BIT) {
				if (!Bind(GL_COPY_READ_BUFFER)) {
					return false;
				}
				target = GL_COPY_READ_BUFFER;
			}
			else if ((access & GL_MAP_WRITE_BIT) == GL_MAP_WRITE_BIT) {
				if (!Bind(GL_COPY_WRITE_BUFFER)) {
					return false;
				}
				target = GL_COPY_WRITE_BUFFER;
			}
			else {
				return false;
			}
		}
	}
	else {
		target = *GetBindedTarget();
	}
	return static_cast<ImplGLBufferBase*>(base)->MapMemory(target, pMappedData, access, offset, size);
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::UnmapMemory()
{
	auto base = GetBase();
	if (!IsAllocated() || !IsBinded() || !base) {
		return false;
	}
	return static_cast<ImplGLBufferBase*>(base)->UnmapMemory(*GetBindedTarget());
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::IsMapped() const noexcept { auto base = GetBase(); return base ? static_cast<const ImplGLBufferBase*>(base)->IsMapped() : false; }

bool RTLib::Ext::GL::Internal::ImplGLBuffer::CopyBuffer2Buffer(ImplGLBuffer* srcBuffer, ImplGLBuffer* dstBuffer, size_t size, size_t srcOffset, size_t dstOffset)
{
	if (!srcBuffer || !dstBuffer) {
		return false;
	}
	if (!srcBuffer->IsAllocated() || srcBuffer->IsMapped()) {
		return false;
	}
	if (!dstBuffer->IsAllocated() || dstBuffer->IsMapped()) {
		return false;
	}

	bool isBindedCopySrc = false;
	auto targetForCopySrc= srcBuffer->GetBindedTarget();

	bool isBindedCopyDst = false;
	auto targetForCopyDst= dstBuffer->GetBindedTarget();
	bool isSameDefTarget = srcBuffer->GetDefTarget() == dstBuffer->GetDefTarget();
	if (!targetForCopySrc) {
		targetForCopySrc = srcBuffer->GetBindableTargetForCopySrc();
		if (!targetForCopySrc) {
			return false;
		}
	}
	else {
		isBindedCopySrc = true;
	}
	if (!targetForCopyDst) {
		if (isSameDefTarget) {
			if (dstBuffer->IsBindable(GL_COPY_WRITE_BUFFER)) {
				targetForCopyDst = GL_COPY_WRITE_BUFFER;
			}
			else {
				return false;
			}
		}
		else {
			targetForCopyDst = srcBuffer->GetBindableTargetForCopyDst();
			if (!targetForCopyDst) {
				return false;
			}
		}
	}
	else {
		isBindedCopyDst = true;
	}
	if (!isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, srcBuffer->GetResId());
	}
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, dstBuffer->GetResId());
	}
	glCopyBufferSubData(*targetForCopySrc, *targetForCopyDst, srcOffset, dstOffset, size);
	if (!isBindedCopySrc) {
		glBindBuffer(*targetForCopySrc, 0);
	}
	if (!isBindedCopyDst) {
		glBindBuffer(*targetForCopyDst, 0);
	}
	return true;
}

bool RTLib::Ext::GL::Internal::ImplGLBuffer::IsBindedRange(GLenum target, GLuint index) const noexcept
{
	auto base = GetBase();
	return base ? static_cast<const ImplGLBufferBase*>(base)->IsBindedRange(target, index) : false;
}

void RTLib::Ext::GL::Internal::ImplGLBuffer::EraseBindedRange(GLenum target, GLuint index) noexcept
{
	auto base = GetBase();
	if (base){
		static_cast<ImplGLBufferBase*>(base)->EraseRange(target, index);
	}
}

auto RTLib::Ext::GL::Internal::ImplGLBuffer::GetBindedRange(GLenum target, GLuint index) const noexcept -> std::optional<ImplGLBufferRange>
{
	auto base = GetBase();
	if (base) {
		return static_cast<const ImplGLBufferBase*>(base)->GetBindedRange(target, index);
	}
	else {
		return std::nullopt;
	}
}

auto RTLib::Ext::GL::Internal::ImplGLBuffer::GetBindableTargetForCopySrc() const noexcept -> std::optional<GLenum>
{
	if (IsBindable(m_DefaultTarget)) {
		return m_DefaultTarget;
	}
	if (IsBindable(GL_COPY_READ_BUFFER)) {
		return GL_COPY_READ_BUFFER;
	}
	return std::optional<GLenum>();
}

auto RTLib::Ext::GL::Internal::ImplGLBuffer::GetBindableTargetForCopyDst() const noexcept -> std::optional<GLenum>
{
	if (IsBindable(m_DefaultTarget)) {
		return m_DefaultTarget;
	}
	if (IsBindable(GL_COPY_WRITE_BUFFER)) {
		return GL_COPY_WRITE_BUFFER;
	}
	return std::optional<GLenum>();
}

#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_BUFFER_H
#include "ImplGLBindable.h"
#include <iostream>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLBuffer : public ImplGLBindable {
				public:
					friend class ImplGLTexture;
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, GLenum defaultTarget)->ImplGLBuffer*;
					virtual ~ImplGLBuffer()noexcept {}

					bool Bind();
					bool Bind(GLenum target);
					bool Allocate(GLenum usage, size_t size, const void* pInitialData = nullptr);
					bool IsAllocated()const noexcept { return m_AllocationInfo != std::nullopt; }

					bool CopyFromMemory(const void* pSrcData, size_t size, size_t offset = 0);
					bool CopyToMemory(void* pDstData, size_t size, size_t offset = 0);

					bool CopyFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset = 0, size_t dstOffset = 0);
					bool CopyToBuffer(  ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset = 0, size_t srcOffset = 0);

					bool MapMemory(void** pMappedData, GLenum access);
					bool MapMemory(void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size);
					bool UnmapMemory();

					bool IsMapped()const noexcept;
					auto GetDefTarget()const noexcept -> GLenum { return m_DefaultTarget; }
				protected:
					ImplGLBuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, GLenum defaultTarget)noexcept :ImplGLBindable(table, bPoint), m_DefaultTarget{ defaultTarget }{}
				private:
					static bool CopyBuffer2Buffer(ImplGLBuffer* srcBuffer, ImplGLBuffer* dstBuffer, size_t size, size_t srcOffset = 0, size_t dstOffset = 0);
					auto GetBindableTargetForCopySrc()const noexcept -> std::optional<GLenum>;
					auto GetBindableTargetForCopyDst()const noexcept -> std::optional<GLenum>;
				private:
					struct AllocationInfo {
						size_t size;
						GLenum usage;
					};
					GLenum m_DefaultTarget;
					std::optional<AllocationInfo> m_AllocationInfo = std::nullopt;
				};
			}
		}
	}
}
#endif
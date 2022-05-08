#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_BUFFER_H
#include "ImplGLBindable.h"
#include <iostream>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLBufferBase : public ImplGLBindableBase {
				public:
					friend class ImplGLBindable;
				public:
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
							glBindBuffer(target, resId);
						}
					}
					virtual void Unbind(GLenum target) {
						UnmapMemory(target);
						glBindBuffer(target, 0);
					}
				private:
					bool m_IsMapped = false;
				};
				class ImplGLBuffer : public ImplGLBindable {
				public:
					friend class ImplGLTexture;
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)->ImplGLBuffer* {
						if (!table || !bPoint) {
							return nullptr;
						}
						auto buffer = new ImplGLBuffer(table,bPoint);
						if (buffer) {
							buffer->InitBase<ImplGLBufferBase>();
							auto res = buffer->Create();
							if (!res) {
								delete buffer;
								return nullptr;
							}
						}
						return buffer;
					}
					virtual ~ImplGLBuffer()noexcept {}

					bool Bind(GLenum target) {
						return ImplGLBindable::Bind(target);
					}
					bool Allocate(GLenum target, GLenum usage, size_t size, const void* pInitialData = nullptr);
					bool IsAllocated()const noexcept { return m_AllocationInfo != std::nullopt; }

					bool CopyFromMemory(const void* pSrcData, size_t size, size_t offset = 0);
					bool CopyToMemory(void* pDstData, size_t size, size_t offset = 0);

					bool CopyFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset = 0, size_t dstOffset = 0);
					bool CopyToBuffer(  ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset = 0, size_t srcOffset = 0);

					bool MapMemory(void** pMappedData, GLenum access);
					bool MapMemory(void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size);
					bool UnmapMemory();

					bool IsMapped()const noexcept { auto base = GetBase(); return base ? static_cast<const ImplGLBufferBase*>(base)->IsMapped() : false; }
				protected:
					ImplGLBuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint)noexcept :ImplGLBindable(table, bPoint) {}
				private:
					struct AllocationInfo {
						size_t size;
						GLenum usage;
					};
					std::optional<AllocationInfo> m_AllocationInfo = std::nullopt;
				};
			}
		}
	}
}
#endif
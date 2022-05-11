#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_BUFFER_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_BUFFER_H
#include "ImplGLBindable.h"
#include <unordered_map>
#include <optional>
#include <iostream>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				struct ImplGLBufferRange {
					GLsizei  size   = 0;
					GLintptr offset = 0;
				};
				class ImplGLBuffer;
				class ImplGLBufferBindingPointRange
				{
				public:
					friend class ImplGLBuffer;
				public:
					ImplGLBufferBindingPointRange()  {}
					virtual ~ImplGLBufferBindingPointRange()noexcept;
					void   AddTarget(GLenum target, GLint numBindings) noexcept;
					bool   HasTarget(GLenum target)const noexcept;
					bool  IsBindable(GLenum target, GLuint index)const noexcept;
					auto GetBindable(GLenum target, GLuint index)->ImplGLBuffer*;
				private:
					bool   Register(GLenum target, GLuint index, ImplGLBuffer* buffer, GLsizei size, GLintptr offset);
					bool Unregister(GLenum target, GLuint index);
				private:
					struct InternalHandle {
						ImplGLBuffer*     buffer = nullptr; 
						ImplGLBufferRange range  = {};
					};
					std::unordered_map<GLenum, std::vector<InternalHandle>> m_Handles = {};
				};
				class ImplGLBuffer : public ImplGLBindable {
				public:
					friend class ImplGLTexture;
					friend class ImplGLVertexArray;
				private:
					struct AllocationInfo {
						size_t size;
						GLenum usage;
					};
				public:
					static auto New(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, ImplGLBufferBindingPointRange* bPointRange, GLenum defaultTarget)->ImplGLBuffer*;
					virtual ~ImplGLBuffer()noexcept {}
					//Bind Function(For Optimization)//
					bool       Bind();
					bool       Bind( GLenum target);
					bool   BindBase( GLuint index);
					bool   BindBase( GLenum target, GLuint index);
					bool   BindRange(GLuint  index, GLsizei size, GLintptr offset);
					bool   BindRange(GLenum target, GLuint index, GLsizei  size, GLintptr offset);
					bool UnbindBase(GLuint  index);
					bool UnbindBase(GLenum target, GLuint  index);

					bool Allocate(GLenum usage, size_t size, const void* pInitialData = nullptr);
					bool IsAllocated()const noexcept { 
						return (GetResId() > 0) && m_AllocationInfo != std::nullopt; 
					}

					bool CopyFromMemory(const void* pSrcData, size_t size, size_t offset = 0);
					bool CopyToMemory(void* pDstData, size_t size, size_t offset = 0);

					bool CopyFromBuffer(ImplGLBuffer* srcBuffer, size_t size, size_t srcOffset = 0, size_t dstOffset = 0);
					bool CopyToBuffer(  ImplGLBuffer* dstBuffer, size_t size, size_t dstOffset = 0, size_t srcOffset = 0);

					bool MapMemory(void** pMappedData, GLenum access);
					bool MapMemory(void** pMappedData, GLenum access, GLsizeiptr offset, GLsizeiptr size);
					bool UnmapMemory();

					bool IsMapped()const noexcept;
					auto GetDefTarget()const noexcept -> GLenum { return m_DefaultTarget; }
					auto GetBindedRange(GLenum target, GLuint index)const noexcept->std::optional<ImplGLBufferRange>;
				protected:
					ImplGLBuffer(ImplGLResourceTable* table, ImplGLBindingPoint* bPoint, GLenum defaultTarget)noexcept :ImplGLBindable(table, bPoint), m_DefaultTarget{ defaultTarget }{}
				private:
					static bool CopyBuffer2Buffer(ImplGLBuffer* srcBuffer, ImplGLBuffer* dstBuffer, size_t size, size_t srcOffset = 0, size_t dstOffset = 0);
					auto GetBindableTargetForCopySrc()const noexcept -> std::optional<GLenum>;
					auto GetBindableTargetForCopyDst()const noexcept -> std::optional<GLenum>;
					bool    IsBindedRange(GLenum target, GLuint index)const noexcept;
					void EraseBindedRange(GLenum target, GLuint index)noexcept;
				private:
					GLenum						   m_DefaultTarget;
					std::optional<AllocationInfo>  m_AllocationInfo = std::nullopt;
				};
			}
		}
	}
}
#endif
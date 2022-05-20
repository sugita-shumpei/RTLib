#ifndef RTLIB_EXT_GL_GL_BUFFER_H
#define RTLIB_EXT_GL_GL_BUFFER_H
#include <RTLib/Core/Buffer.h>
#include <RTLib/Ext/GL/GLCommon.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
#include <optional>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			class GLContext;
			class GLVertexArray;
			class GLBuffer: public Core::Buffer{
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLBuffer, Core::Buffer, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_BUFFER);

				friend class GLContext;
				friend class GLVertexArray;
			public:
				static auto Allocate(GLContext* context, const GLBufferCreateDesc& desc)->GLBuffer*;
				virtual void Destroy()override;
				auto GetBufferUsage()const noexcept -> GLBufferUsageFlags;
			private:
				GLBuffer(GLContext* context, GLuint resId, const GLBufferCreateDesc& desc)noexcept;
				auto GetResId()const noexcept -> GLuint;
				auto GetCurrentTarget() const noexcept -> std::optional<GLenum>;
				auto GetMemoryProperty()const noexcept-> GLMemoryPropertyFlags;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

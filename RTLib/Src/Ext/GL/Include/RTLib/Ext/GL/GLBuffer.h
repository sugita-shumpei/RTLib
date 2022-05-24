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
			class GLNatives;
			class GLBuffer: public Core::Buffer{
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLBuffer, Core::Buffer, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_BUFFER);
				friend class GLContext;
				friend class GLNatives;
			public:
				static auto Allocate(GLContext* context, const GLBufferCreateDesc& desc)->GLBuffer*;
                virtual ~GLBuffer()noexcept;
                
				virtual void Destroy()noexcept override;
				auto GetUsages()const noexcept  -> GLBufferUsageFlags;
				auto GetMainUsage()const noexcept -> GLBufferUsageFlagBits;
				auto GetSizeInBytes() const noexcept -> size_t;
			private:
				GLBuffer(GLContext* context, const GLBufferCreateDesc& desc)noexcept;
				auto GetMemoryProperty()const noexcept -> GLMemoryPropertyFlags;
				auto GetResId()const noexcept -> GLuint;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			struct GLBufferView
			{
			public:
				friend class GLNatives;
			public:
				GLBufferView()noexcept;
				GLBufferView(GLBuffer* base)noexcept;
				GLBufferView(GLBuffer* base, size_t offsetInBytes, size_t sizeInBytes)noexcept;
				GLBufferView(const GLBufferView& bufferView, size_t offsetInBytes, size_t sizeInBytes)noexcept;

				GLBufferView(const GLBufferView& bufferView)noexcept;
				GLBufferView& operator=(const GLBufferView& bufferView)noexcept;

				auto GetBaseBuffer()const noexcept -> const GLBuffer* { return m_Base; }
				auto GetBaseBuffer()      noexcept ->       GLBuffer* { return m_Base; }
				auto GetSizeInBytes() const noexcept -> size_t { return m_SizeInBytes; }
				auto GetOffsetInBytes()const noexcept-> size_t { return m_OffsetInBytes; }
			private:
				GLBuffer* m_Base;
				size_t      m_OffsetInBytes;
				size_t      m_SizeInBytes;
			};
		}
	}
}
#endif

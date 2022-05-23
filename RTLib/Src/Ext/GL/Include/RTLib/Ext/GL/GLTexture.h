#ifndef RTLIB_EXT_GL_GL_TEXTURE_H
#define RTLIB_EXT_GL_GL_TEXTURE_H
#include <RTLib/Core/Texture.h>
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
			class GLTexture : public Core::Texture {
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLTexture, Core::Texture, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_TEXTURE);
				friend class GLContext;
			public:
				static auto Allocate(GLContext* context, const GLTextureCreateDesc& desc)->GLTexture*;
				virtual ~GLTexture()noexcept;

				virtual void Destroy()noexcept override;
				auto GetBufferUsage()const noexcept -> GLBufferUsageFlags;
			private:
				GLTexture(GLContext* context, const GLTextureCreateDesc& desc)noexcept;
				auto GetResId()const noexcept -> GLuint;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

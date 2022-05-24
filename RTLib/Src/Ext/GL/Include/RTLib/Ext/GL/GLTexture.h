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
			class GLNatives;
			class GLTexture : public Core::Texture {
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLTexture, Core::Texture, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_TEXTURE);
				friend class GLContext;
				friend class GLNatives;
			public:
				static auto Allocate(GLContext* context, const GLTextureCreateDesc& desc)->GLTexture*;
				virtual ~GLTexture()noexcept;

				virtual void Destroy()noexcept override;

				auto GetImage()      noexcept ->       GLImage*;
				auto GetImage()const noexcept -> const GLImage*;
				auto GetType() const noexcept -> GLImageViewType;
				auto GetFormat()const noexcept -> GLFormat;
				auto GetExtent()const noexcept -> GLExtent3D;
				auto GetMipExtent(uint32_t level)const noexcept -> GLExtent3D;
				auto GetMipLevels()const noexcept -> uint32_t;
				auto GetArrayLayers()const noexcept -> uint32_t;
				auto GetFlags()const noexcept  -> GLImageCreateFlags;
			private:
				GLTexture()noexcept;
				auto GetResId()const noexcept -> GLuint;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

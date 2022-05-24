#ifndef RTLIB_EXT_GL_GL_IMAGE_H
#define RTLIB_EXT_GL_GL_IMAGE_H
#include <RTLib/Core/Image.h>
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
			class GLImage : public Core::Image {
				RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(GLImage, Core::Image, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_IMAGE);
				friend class GLContext;
				friend class GLNatives;
			public:
				static auto Allocate(GLContext* context, const GLImageCreateDesc& desc)->GLImage*;
				virtual ~GLImage()noexcept;
				virtual void Destroy()noexcept override;

				auto GetViewType()const noexcept      -> GLImageViewType;
				auto GetImageType()    const noexcept -> GLImageType;
				auto GetFormat()const noexcept        -> GLFormat;
				auto GetExtent()const noexcept        -> GLExtent3D;
				auto GetMipExtent(uint32_t level)const noexcept -> GLExtent3D;
				auto GetMipLevels()const noexcept     -> uint32_t;
				auto GetArrayLayers()const noexcept   -> uint32_t;
				auto GetFlags()const noexcept         -> GLImageCreateFlags;
			private:
				GLImage(GLContext* context, const GLImageCreateDesc& desc)noexcept;
				auto GetResId()const noexcept -> GLuint;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

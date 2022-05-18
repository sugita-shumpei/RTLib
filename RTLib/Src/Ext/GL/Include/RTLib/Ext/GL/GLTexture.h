#ifndef RTLIB_EXT_GL_GL_TEXTURE_H
#define RTLIB_EXT_GL_GL_TEXTURE_H
#include <RTLib/Core/Texture.h>
#include <RTLib/Ext/GL/UuidDefinitions.h>
namespace RTLib
{
	namespace Ext
	{
		namespace GL
		{
			RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(GLTexture, Core::BaseTexture, RTLIB_TYPE_UUID_RTLIB_EXT_GL_GL_TEXTURE);
			RTLIB_CORE_TYPE_OBJECT_DECLARE_END();
		}
	}
}
#endif

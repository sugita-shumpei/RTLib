#ifndef RTLIB_CORE_TEXTURE_H
#define RTLIB_CORE_TEXTURE_H
#include <RTLib/Core/Common.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/UuidDefinitions.h>
namespace RTLib
{
	namespace Core {
		RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(Texture, BaseObject, RTLIB_TYPE_UUID_RTLIB_CORE_TEXTURE);
		public:
			Texture()noexcept;

		RTLIB_CORE_TYPE_OBJECT_DECLARE_END();
	}
}
#endif

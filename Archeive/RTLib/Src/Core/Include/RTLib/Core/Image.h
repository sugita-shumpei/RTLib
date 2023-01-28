#ifndef RTLIB_CORE_IMAGE_H
#define RTLIB_CORE_IMAGE_H
#include <RTLib/Core/Common.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/UuidDefinitions.h>
namespace RTLib
{
	namespace Core
	{
		RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(Image, BaseObject, RTLIB_TYPE_UUID_RTLIB_CORE_IMAGE);
	public:
		virtual void Destroy()noexcept = 0;
		RTLIB_CORE_TYPE_OBJECT_DECLARE_END();
	}
}
#endif

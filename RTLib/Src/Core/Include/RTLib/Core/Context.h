#ifndef RTLIB_CORE_CONTEXT_H
#define RTLIB_CORE_CONTEXT_H
#include <RTLib/Core/Common.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/UuidDefinitions.h>
namespace RTLib
{
	namespace Core
	{
		RTLIB_CORE_TYPE_OBJECT_DECLARE_BEGIN(Context, BaseObject, RTLIB_TYPE_UUID_RTLIB_CORE_CONTEXT);
		public:
			virtual bool Initialize() = 0;
			virtual void Terminate()  = 0;
		RTLIB_CORE_TYPE_OBJECT_DECLARE_END();
	}
}
#endif

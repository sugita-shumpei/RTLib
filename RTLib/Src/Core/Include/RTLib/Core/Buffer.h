#ifndef RTLIB_CORE_BUFFER_H
#define RTLIB_CORE_BUFFER_H
#include <RTLib/Core/Common.h>
#include <RTLib/Core/BaseObject.h>
#include <RTLib/Core/UuidDefinitions.h>
namespace RTLib
{
	namespace Core {
        class Buffer:public BaseObject{
		RTLIB_CORE_TYPE_OBJECT_DECLARE_DERIVED_METHOD(Buffer, BaseObject, RTLIB_TYPE_UUID_RTLIB_CORE_BUFFER);
		public:
            virtual ~Buffer()noexcept;
			virtual void Destroy()noexcept = 0;
        };
	}
}
#endif

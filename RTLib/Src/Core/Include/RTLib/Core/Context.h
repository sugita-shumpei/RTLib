#ifndef RTLIB_CORE_CONTEXT_H
#define RTLIB_CORE_CONTEXT_H
#include <RTLib/Core/Context.h>
namespace RTLib
{
	namespace Core
	{
		class Context
		{
		public:
			virtual ~Context()noexcept{}

			virtual bool Initialize() = 0;

			virtual void Terminate()  = 0;
		};
	}
}
#endif

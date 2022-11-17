#ifndef RTLIB_BACKENDS_OGL4_OGL4_CONTEXT_H
#define RTLIB_BACKENDS_OGL4_OGL4_CONTEXT_H
namespace RTLib
{
	namespace Backends
	{
		namespace Ogl4
		{
			class Context
			{
				 Context() noexcept {}
			private:
				 Context(const Context&) = delete;
				 Context& operator=(const Context&) = delete;
				 Context(Context&&) = delete;
				 Context& operator=(Context&&) = delete;
				~Context() noexcept {}
			};
		}
	}
}
#endif

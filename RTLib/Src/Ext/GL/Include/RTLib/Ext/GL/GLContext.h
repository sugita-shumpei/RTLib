#ifndef RTLIB_EXT_GL_GL_CONTEXT_H
#define RTLIB_EXT_GL_GL_CONTEXT_H
#include <optional>
#include <memory>
namespace RTLib {
	namespace Ext {
		namespace GL {
			class GLContext {
			public:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

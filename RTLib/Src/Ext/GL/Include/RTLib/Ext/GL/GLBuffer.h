#ifndef RTLIB_EXT_GL_GL_BUFFER_H
#define RTLIB_EXT_GL_GL_BUFFER_H
#include <RTLib/Core/Buffer.h>
#include <memory>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLBuffer;
			}
			class              Buffer {
			public:
				virtual ~Buffer()noexcept {}
			private:
				std::unique_ptr<Internal::ImplGLBuffer> m_Impl;
			};
			class        VertexBuffer : public Buffer {
			public:
				virtual ~VertexBuffer()noexcept {}
			};
			class         IndexBuffer : public Buffer {
				virtual  ~IndexBuffer()noexcept {}
			};
			class       UniformBuffer : public Buffer {
			public:
				virtual ~UniformBuffer()noexcept {}
			};
			class ShaderStorageBuffer {
				virtual ~ShaderStorageBuffer()noexcept {}
			};
		}
	}
}
#endif
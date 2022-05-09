#ifndef RTLIB_EXT_GL_INTERNAL_IMPL_GL_CONTEXT_H
#define RTLIB_EXT_GL_INTERNAL_IMPL_GL_CONTEXT_H
#include "ImplGLBindable.h"
#include "ImplGLResource.h"
#include "ImplGLBuffer.h"
#include "ImplGLTexture.h"
#include "ImplGLSampler.h"
#include "ImplGLVertexArray.h"
#include "ImplGLFramebuffer.h"
#include "ImplGLRenderbuffer.h"
#include "ImplGLProgram.h"
#include "ImplGLShader.h"
#include <glad/glad.h>
#include <vector>
#include <string>
#include <unordered_map>
namespace RTLib {
	namespace Ext {
		namespace GL {
			namespace Internal {
				class ImplGLContext {
				public:
					static auto New() ->ImplGLContext* {
						auto ptr = new ImplGLContext();
						ptr->Init();
						return ptr;
					}
					virtual ~ImplGLContext()noexcept {}

					auto CreateBuffer (GLenum target)->ImplGLBuffer*;
					auto CreateTexture(GLenum target)->ImplGLTexture*;
					auto CreateSampler()             ->ImplGLSampler*;
					auto CreateFramebuffer()         ->ImplGLFramebuffer*;
					auto CreateRenderbuffer()        ->ImplGLRenderbuffer*;
					auto CreateVertexArray()         ->ImplGLVertexArray*;
					auto CreateShader(GLenum shaderT)->ImplGLShader*;
					auto CreateGraphicsProgram()     ->ImplGLGraphicsProgram*;
					auto CreateComputeProgram()      ->ImplGLComputeProgram*;

					auto GetMajorVersion()const noexcept -> GLint { return m_MajorVersion; }
					auto GetMinorVersion()const noexcept -> GLint { return m_MinorVersion; }
					auto GetProfile()const noexcept      -> GLint { return m_Profile; }
					auto GetFlags()const noexcept -> GLint { return m_Flags; }
					bool IsSpirvSupported()const noexcept  { return m_SpirvSupported;}
				private:
					ImplGLContext()noexcept {}
					void Init() {
						if (m_IsInit) {
							return;
						}
						
						glGetIntegerv(GL_MAJOR_VERSION             , &m_MajorVersion);
						glGetIntegerv(GL_MINOR_VERSION             , &m_MinorVersion);
						glGetIntegerv(GL_CONTEXT_PROFILE_MASK      , &m_Profile);
						glGetIntegerv(GL_CONTEXT_FLAGS             , &m_Flags);
						m_SpirvSupported = IsSupportedVersion(4, 6);
						GLint numProgramBinaryFormats;
						glGetIntegerv(GL_NUM_PROGRAM_BINARY_FORMATS, &numProgramBinaryFormats );
						std::vector<GLint> tProgramBinaryFormats(numProgramBinaryFormats);
						glGetIntegerv(GL_PROGRAM_BINARY_FORMATS    , tProgramBinaryFormats.data());
						glGetIntegerv(GL_NUM_SHADER_BINARY_FORMATS , &m_NumShaderBinaryFormats);
						GLint numExtensions = 0;
						glGetIntegerv(GL_NUM_EXTENSIONS            , &numExtensions);
						
						m_ExtensionNames.reserve(numExtensions);
						for (GLint i = 0; i < numExtensions; ++i) {
							const char* extensionName = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
							m_ExtensionNames.push_back(std::string(extensionName));
						}
						for (GLint i = 0; i < numExtensions; ++i) {
							std::cout << "Extension Enabled: " << m_ExtensionNames[i] << std::endl;
						}

						m_BPBuffer.AddTarget(GL_ARRAY_BUFFER);
						m_BPBuffer.AddTarget(GL_ATOMIC_COUNTER_BUFFER);
						m_BPBuffer.AddTarget(GL_COPY_READ_BUFFER);
						m_BPBuffer.AddTarget(GL_COPY_WRITE_BUFFER);
						m_BPBuffer.AddTarget(GL_DISPATCH_INDIRECT_BUFFER);
						m_BPBuffer.AddTarget(GL_DRAW_INDIRECT_BUFFER);
						m_BPBuffer.AddTarget(GL_ELEMENT_ARRAY_BUFFER);
						m_BPBuffer.AddTarget(GL_PIXEL_PACK_BUFFER);
						m_BPBuffer.AddTarget(GL_PIXEL_UNPACK_BUFFER);
						m_BPBuffer.AddTarget(GL_QUERY_BUFFER);

						if (IsSupportedVersion(4, 3)) {
							m_BPBuffer.AddTarget(GL_SHADER_STORAGE_BUFFER);
						}

						m_BPBuffer.AddTarget(GL_TEXTURE_BUFFER);
						m_BPBuffer.AddTarget(GL_TRANSFORM_FEEDBACK_BUFFER);
						m_BPBuffer.AddTarget(GL_UNIFORM_BUFFER);

						m_BPTexture.AddTarget(GL_TEXTURE_1D);
						m_BPTexture.AddTarget(GL_TEXTURE_2D);
						m_BPTexture.AddTarget(GL_TEXTURE_3D);
						m_BPTexture.AddTarget(GL_TEXTURE_1D_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_2D_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_2D_MULTISAMPLE);
						m_BPTexture.AddTarget(GL_TEXTURE_2D_MULTISAMPLE_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_RECTANGLE);
						m_BPTexture.AddTarget(GL_TEXTURE_CUBE_MAP);
						m_BPTexture.AddTarget(GL_TEXTURE_CUBE_MAP_ARRAY);
						m_BPTexture.AddTarget(GL_TEXTURE_BUFFER);

						m_BPVertexArray.AddTarget(GL_VERTEX_ARRAY);

						m_BPFramebuffer.AddTarget(GL_FRAMEBUFFER);
						m_BPFramebuffer.AddTarget(GL_DRAW_FRAMEBUFFER);
						m_BPFramebuffer.AddTarget(GL_READ_FRAMEBUFFER);

						m_BPRenderbuffer.AddTarget(GL_RENDERBUFFER);

						GLint maxTexUnits;
						glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS, &maxTexUnits);
						for (uint32_t i = 0; i < maxTexUnits; ++i) {
							m_BPSampler.AddTarget(i);
						}
						m_IsInit = true;
					}
					bool IsSupportedVersion(GLint majorVersion, GLint minorVersion)const noexcept {
						if (majorVersion < m_MajorVersion) {
							return false;
						}
						if (majorVersion > m_MajorVersion) {
							return true;
						}
						return minorVersion >= m_MinorVersion;
					}
				private:
					bool                m_IsInit            = false;
					ImplGLResourceTable m_ResourceTable     = {};
					ImplGLBindingPoint  m_BPBuffer          = {};
					ImplGLBindingPoint  m_BPTexture         = {};
					ImplGLBindingPoint  m_BPVertexArray     = {};
					ImplGLBindingPoint  m_BPSampler         = {};
					ImplGLBindingPoint  m_BPFramebuffer     = {};
					ImplGLBindingPoint  m_BPRenderbuffer    = {};

					GLint                    m_MajorVersion            = 0;
					GLint                    m_MinorVersion            = 0;
					GLint                    m_Profile                 = 0;
					GLint                    m_Flags                   = 0;
					bool                     m_SpirvSupported          = false;
					std::vector<GLenum>      m_ProgramBinaryFormats    = {};
					GLint                    m_NumShaderBinaryFormats  = 0;
					std::vector<std::string> m_ExtensionNames		   = {};
				};
			}
		}
	}
}
#endif
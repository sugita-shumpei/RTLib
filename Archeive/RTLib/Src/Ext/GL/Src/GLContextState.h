#ifndef RTLIB_EXT_GL_GL_CONTEXT_STATE_H
#define RTLIB_EXT_GL_GL_CONTEXT_STATE_H
#include <RTLib/Ext/GL/GLCommon.h>
#include "GLUniqueIdentifier.h"
#include <unordered_map>
namespace RTLib
{
    namespace Ext
    {
        namespace GL
        {
            class GLContextState
            {
            private:
                GLUniqueId m_UidVertexArray     = -1;
                GLUniqueId m_UidProgram         = -1;
                GLUniqueId m_UidProgramPipeline = -1;
            };
        }
    }
}
#endif

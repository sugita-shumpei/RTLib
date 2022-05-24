#ifndef RTLIB_EXT_CUGL_CUGL_COMMON_H
#define RTLIB_EXT_CUGL_CUGL_COMMON_H
#include <RTLib/Ext/CUGL/UuidDefinitions.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/GL/GLContext.h>
#include <cudaGL.h>
namespace RTLib
{
	namespace Ext
	{
		namespace CUGL {
			enum CUGLGraphicsRegisterFlags
			{
				CUGLGraphicsRegisterFlagsNone           = CU_GRAPHICS_REGISTER_FLAGS_NONE,
				CUGLGraphicsRegisterFlagsReadOnly       = CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY,
				CUGLGraphicsRegisterFlagsWriteDiscard   = CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
				CUGLGraphicsRegisterFlagsSurfaceLDST    = CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST,
				CUGLGraphicsRegisterFlagsTextureGather  = CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER,
			};
			
		}
	}
}
#endif

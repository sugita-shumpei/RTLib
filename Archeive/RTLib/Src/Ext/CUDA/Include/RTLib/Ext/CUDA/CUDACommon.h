#ifndef RTLIB_EXT_CUDA_CUDA_COMMON_H
#define RTLIB_EXT_CUDA_CUDA_COMMON_H
#include <RTLib/Core/Common.h>
#include <RTLib/Ext/CUDA/CUDATypeFormat.h>
#include <cuda.h>
#include <vector>
namespace RTLib {
	namespace Ext {
		namespace CUDA {
			enum class CUDAResourceType
			{
				eArray,
				eMipmappedArray,
				eLinear,
				ePitch2D
			};
			using      CUDAOffset2D = Core::Offset2D;
			using      CUDAOffset3D = Core::Offset3D;
			using      CUDAExtent2D = Core::Extent2D;
			using      CUDAExtent3D = Core::Extent3D;
			using	   CUDAImageSubresourceLayers = Core::ImageSubresourceLayers;
			using      CUDABufferCopy       = Core::BufferCopy;
			using      CUDABufferImageCopy  = Core::BufferImageCopy;
			using      CUDAMemoryBufferCopy = Core::MemoryBufferCopy;
			using      CUDABufferMemoryCopy = Core::BufferMemoryCopy;
			using      CUDAImageMemoryCopy  = Core::ImageMemoryCopy;
			using      CUDAMemoryImageCopy  = Core::MemoryImageCopy;
			using      CUDAImageCopy        = Core::ImageCopy;
			enum class CUDAMemoryFlags {
				eDefault
			};
			struct     CUDABufferCreateDesc {
				CUDAMemoryFlags flags = CUDAMemoryFlags::eDefault;
				size_t sizeInBytes = 0;
				void* pData = nullptr;
			};
			using CUDAImageType = Core::ImageType;
			enum       CUDAImageCreateFlagBits
			{
				CUDAImageCreateFlagBitsDefault         = 0,
				CUDAImageCreateFlagBitsSurfaceLDST     = CUDA_ARRAY3D_SURFACE_LDST,
				CUDAImageCreateFlagBitsCubemap         = CUDA_ARRAY3D_CUBEMAP,
				CUDAImageCreateFlagBitsDepthTexture    = CUDA_ARRAY3D_DEPTH_TEXTURE,
				CUDAImageCreateFlagBitsTextureGather   = CUDA_ARRAY3D_TEXTURE_GATHER,
				CUDAImageCreateFlagBitsColorAttachment = CUDA_ARRAY3D_COLOR_ATTACHMENT,
			};
			using      CUDAImageCreateFlags = unsigned int;
			struct     CUDAImageCreateDesc {
				CUDAImageCreateFlags    flags       = CUDAImageCreateFlagBitsDefault;
				CUDAImageType			imageType   = CUDAImageType::e1D;
				CUDAExtent3D            extent      = CUDAExtent3D();
				unsigned int			mipLevels   = 0;
				unsigned int			arrayLayers = 0;
				CUDAImageFormat	  	    format      = CUDAImageFormat::eUndefined;
			};
			enum class CUDATextureAddressMode
			{
				eWarp   = static_cast<uint32_t>(Core::SamplerAddressMode::eRepeat),
				eClamp  = static_cast<uint32_t>(Core::SamplerAddressMode::eClampToEdge),
				eMirror = static_cast<uint32_t>(Core::SamplerAddressMode::eMirroredRepeat),
				eBorder = static_cast<uint32_t>(Core::SamplerAddressMode::eClampToBorder),
			};
			enum class CUDATextureFilterMode
			{
				ePoint  = static_cast<uint32_t>(Core::FilterMode::eNearest),
				eLinear = static_cast<uint32_t>(Core::FilterMode::eLinear),
			};
			enum       CUDATextureSamplerFlagBits
			{
				CUDATextureFlagBitsReadAsInteger                = CU_TRSF_READ_AS_INTEGER,
				CUDATextureFlagBitsNormalizedCordinates         = CU_TRSF_NORMALIZED_COORDINATES,
				CUDATextureFlagBitsDisableTrilinearOptimization = CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION,
			};
			using      CUDATextureSamplerFlags = unsigned int;
			struct     CUDATextureSamplerDesc
			{
				CUDATextureAddressMode  addressMode[3]      = {};
				float				    borderColor[4]      = {};
				CUDATextureFilterMode   filterMode          = CUDATextureFilterMode::ePoint;
				unsigned int	        maxAnisotropy       = 0;
				float				    maxMipmapLevelClamp = 1.0f;
				float				    minMipmapLevelClamp = 0.0f;
				CUDATextureFilterMode   mipmapFilterMode    = CUDATextureFilterMode::ePoint;
				float                   mipmapLevelBias     = 0.0f;
				CUDATextureSamplerFlags flags               = 0;
			};
			struct     CUDATextureResourceViewDesc
			{
				CUDAResourceViewFormat format;
				size_t     		       width;
				size_t     		       height;
				size_t     		       depth;
				unsigned int           baseLayer;
				unsigned int           baseLevel;
				unsigned int           numLayers;
				unsigned int           numLevels;
			};
			class      CUDAImage;
			struct     CUDATextureImageCreateDesc
			{
				CUDAImage*                  image;
				CUDATextureResourceViewDesc view;
				CUDATextureSamplerDesc      sampler;
			};
			enum class CUDAJitOptionFlags
			{
				eMaxRegister				   = CU_JIT_MAX_REGISTERS,
				eThreadPerBlock			       = CU_JIT_THREADS_PER_BLOCK,
				eWallTime					   = CU_JIT_WALL_TIME,
				eLogBuffer				       = CU_JIT_INFO_LOG_BUFFER,
				eLogBufferSizeBytes            = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				eOptimizationLevel             = CU_JIT_OPTIMIZATION_LEVEL,
				eTargetFromCUContext           = CU_JIT_TARGET_FROM_CUCONTEXT,
				eTarget                        = CU_JIT_TARGET,
				eFallbackStrategy              = CU_JIT_FALLBACK_STRATEGY,
				eGenerateDebugInfo             = CU_JIT_GENERATE_DEBUG_INFO,
				eLogVerbose                    = CU_JIT_LOG_VERBOSE,
				eGenerateLineInfo              = CU_JIT_GENERATE_LINE_INFO,
				eCacheMode                     = CU_JIT_CACHE_MODE,
				eNewSM3XOpt                    = CU_JIT_NEW_SM3X_OPT,
				eFastCompile                   = CU_JIT_FAST_COMPILE,
				eGlobalSymbolNames             = CU_JIT_GLOBAL_SYMBOL_NAMES,
				eGlobalSymbolAddresses         = CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
				eGlobalSymbolCount             = CU_JIT_GLOBAL_SYMBOL_COUNT,
				eNumOptions				       = CU_JIT_NUM_OPTIONS,
			};
			struct     CUDAJitOptionValue
			{
				CUDAJitOptionFlags option;
				void*               value;
			};
			class      CUDAStream;
			struct     CUDAKernelLaunchDesc
			{
				unsigned int           gridDimX;
				unsigned int           gridDimY;
				unsigned int           gridDimZ;
				unsigned int          blockDimX;
				unsigned int          blockDimY;
				unsigned int          blockDimZ;
				unsigned int     sharedMemBytes;
				std::vector<void*> kernelParams;
				CUDAStream*              stream;
			};
		}
	}
}
#endif

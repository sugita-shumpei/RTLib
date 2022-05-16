#ifndef RTLIB_EXT_CUDA_CUDA_COMMON_H
#define RTLIB_EXT_CUDA_CUDA_COMMON_H
#include <RTLib/Core/Common.h>
#include <cuda.h>
#include <vector>
namespace RTLib {
	namespace Ext {
		namespace CUDA {
			enum class CUDAImageDataType: uint64_t {
				eUndefined = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUndefined),
				eInt8      = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8),
				eInt16     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16),
				eInt32     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32),
				eUInt8     = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8),
				eUInt16    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16),
				eUInt32    = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32),
				eFloat16   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16),
				eFloat32   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32),
				eNV12      = 1<<9,
			};
			inline constexpr auto GetCUDAImageDataTypeSize(CUDAImageDataType dataType) -> uint32_t
			{
				return static_cast<uint32_t>(static_cast<uint64_t>(dataType) & 63);
			}
			static_assert(GetCUDAImageDataTypeSize(CUDAImageDataType::eFloat16) == 16);
			static_assert(GetCUDAImageDataTypeSize(CUDAImageDataType::eInt32) == 32);
			enum class CUDAResourceViewFormat : uint64_t
			{
				eUndefined = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUndefined),
				eSInt8X1   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X1),
				eSInt8X2   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X2),
				eSInt8X4   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt8X4),
				eUInt8X1   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X1),
				eUInt8X2   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X2),
				eUInt8X4   = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt8X4),
				eSInt16X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X1),
				eSInt16X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X2),
				eSInt16X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt16X4),
				eUInt16X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X1),
				eUInt16X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X2),
				eUInt16X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt16X4),
				eSInt32X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X1),
				eSInt32X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X2),
				eSInt32X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eInt32X4),
				eUInt32X1  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X1),
				eUInt32X2  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X2),
				eUInt32X4  = static_cast<uint64_t>(Core::SizedTypeFlagBits::eUInt32X4),
				eFloat16X1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X1),
				eFloat16X2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X2),
				eFloat16X4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat16X4),
				eFloat32X1 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X1),
				eFloat32X2 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X2),
				eFloat32X4 = static_cast<uint64_t>(Core::SizedTypeFlagBits::eFloat32X4),
				eUnsignedBC1  = (((uint64_t)1) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eUnsignedBC2  = (((uint64_t)2) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eUnsignedBC3  = (((uint64_t)3) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eUnsignedBC4  = (((uint64_t)4) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eSignedBC4    = (((uint64_t)4) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eInteger),
				eUnsignedBC5  = (((uint64_t)5) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eSignedBC5    = (((uint64_t)5) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eInteger),
				eUnsignedBC6H = (((uint64_t)6) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
				eSignedBC6H   = (((uint64_t)6) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eInteger),
				eUnsignedBC7  = (((uint64_t)7) << 38) | static_cast<uint64_t>(Core::BaseTypeFlagBits::eUnsigned),
			};
			enum class CUDAResourceType
			{
				eArray,
				eMipmappedArray,
				eLinear,
				ePitch2D
			};
			using CUDAImageSubresourceLayers = Core::ImageSubresourceLayers;
			using CUDABufferCopy       = Core::BufferCopy;
			using CUDABufferImageCopy  = Core::BufferImageCopy;
			using CUDAMemoryBufferCopy = Core::MemoryBufferCopy;
			using CUDABufferMemoryCopy = Core::BufferMemoryCopy;
			using CUDAImageMemoryCopy  = Core::ImageMemoryCopy;
			using CUDAMemoryImageCopy  = Core::MemoryImageCopy;
			using CUDAImageCopy        = Core::ImageCopy;
			enum class CUDAMemoryFlags {
				eDefault,
				ePageLocked   ,
			};
			struct     CUDABufferDesc {
				CUDAMemoryFlags flags = CUDAMemoryFlags::eDefault;
				size_t sizeInBytes = 0;
			};
			enum class CUDAImageType {
				e1D,
				e2D,
				e3D
			};
			enum       CUDAImageFlagBits
			{
				CUDAImageFlagBitsDefault         = 0,
				CUDAImageFlagBitsSurfaceLDST     = CUDA_ARRAY3D_SURFACE_LDST,
				CUDAImageFlagBitsCubemap         = CUDA_ARRAY3D_CUBEMAP,
				CUDAImageFlagBitsDepthTexture    = CUDA_ARRAY3D_DEPTH_TEXTURE,
				CUDAImageFlagBitsTextureGather   = CUDA_ARRAY3D_TEXTURE_GATHER,
				CUDAImageFlagBitsColorAttachment = CUDA_ARRAY3D_COLOR_ATTACHMENT,
			};
			using      CUDAImageFlags = unsigned int;
			struct     CUDAImageDesc {
				CUDAImageType     imageType   = CUDAImageType::e1D;
				unsigned int      width       = 1;
				unsigned int      height      = 0;
				unsigned int      depth       = 0;
				unsigned int      levels      = 0;
				unsigned int      layers      = 0;
				CUDAImageDataType format      = CUDAImageDataType::eUndefined;
				unsigned int      channels = 1;
				CUDAImageFlags    flags       = CUDAImageFlagBitsDefault;
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
			struct     CUDATextureImageDesc
			{
				CUDAImage*                  image;
				CUDATextureResourceViewDesc view;
				CUDATextureSamplerDesc      sampler;
			};
			enum class CUDAJitOptionFlagBits
			{
				CUDAJitOptionFlagBitsMaxRegister				   = CU_JIT_MAX_REGISTERS,
				CUDAJitOptionFlagBitsThreadPerBlock			       = CU_JIT_THREADS_PER_BLOCK,
				CUDAJitOptionFlagBitsWallTime					   = CU_JIT_WALL_TIME,
				CUDAJitOptionFlagBitsLogBuffer				       = CU_JIT_INFO_LOG_BUFFER,
				CUDAJitOptionFlagBitsLogBufferSizeBytes            = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				CUDAJitOptionFlagBitsOptimizationLevel             = CU_JIT_OPTIMIZATION_LEVEL,
				CUDAJitOptionFlagBitsTargetFromCUContext           = CU_JIT_TARGET_FROM_CUCONTEXT,
				CUDAJitOptionFlagBitsTarget                        = CU_JIT_TARGET,
				CUDAJitOptionFlagBitsFallbackStrategy              = CU_JIT_FALLBACK_STRATEGY,
				CUDAJitOptionFlagBitsGenerateDebugInfo             = CU_JIT_GENERATE_DEBUG_INFO,
				CUDAJitOptionFlagBitsLogVerbose                    = CU_JIT_LOG_VERBOSE,
				CUDAJitOptionFlagBitsGenerateLineInfo              = CU_JIT_GENERATE_LINE_INFO,
				CUDAJitOptionFlagBitsCacheMode                     = CU_JIT_CACHE_MODE,
				CUDAJitOptionFlagBitsNewSM3XOpt                    = CU_JIT_NEW_SM3X_OPT,
				CUDAJitOptionFlagBitsFastCompile                   = CU_JIT_FAST_COMPILE,
				CUDAJitOptionFlagBitsGlobalSymbolNames             = CU_JIT_GLOBAL_SYMBOL_NAMES,
				CUDAJitOptionFlagBitsGlobalSymbolAddresses         = CU_JIT_GLOBAL_SYMBOL_ADDRESSES,
				CUDAJitOptionFlagBitsGlobalSymbolCount             = CU_JIT_GLOBAL_SYMBOL_COUNT,
				CUDAJitOptionFlagBitsNumOptions				       = CU_JIT_NUM_OPTIONS,
			};
			struct     CUDAJitOptionValue
			{
				CUDAJitOptionFlagBits option;
				void*                 value;
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

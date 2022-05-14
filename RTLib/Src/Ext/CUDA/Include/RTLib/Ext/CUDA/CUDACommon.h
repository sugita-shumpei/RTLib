#ifndef RTLIB_EXT_CUDA_CUDA_COMMON_H
#define RTLIB_EXT_CUDA_CUDA_COMMON_H
#include <RTLib/Core/Common.h>
#include <cuda.h>
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
			};
			enum class CUDAMemoryFlags {
				eDefault,
				ePageLocked   ,
			};
			struct     CUDABufferDesc {
				CUDAMemoryFlags flags = CUDAMemoryFlags::eDefault;
				size_t sizeInBytes = 0;
			};
			struct     CUDAImageDesc {
				unsigned int    width       = 1;
				unsigned int    height      = 1;
				unsigned int    depth       = 1;
				unsigned int    layers      = 1;
				CUDAImageDataType format      = CUDAImageDataType::eUndefined;
				unsigned int    numChannels = 1;
				CUDAImageFlags  flags       = CUDAImageFlags::eDefault;
			};
			enum class CUDAImageFlags
			{
				eDefault,
				eSurfaceLDST,
				eCubemap,
				eTextureGather
			};
		}
	}
}
#endif

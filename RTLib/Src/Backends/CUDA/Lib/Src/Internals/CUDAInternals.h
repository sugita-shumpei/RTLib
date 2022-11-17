#ifndef RTLIB_BACKENDS_CUDA_INTERNALS_CUDA_INTERNALS_H
#define RTLIB_BACKENDS_CUDA_INTERNALS_CUDA_INTERNALS_H
#include <cuda.h>
#include <fmt/printf.h>
#include <cassert>
#ifndef NDEBUG
#define RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(EXPR) \
	do { CUresult v_result_RTLIB_BACKENDS_CUDA_DEBUG_ASSERT = EXPR; \
		if (v_result_RTLIB_BACKENDS_CUDA_DEBUG_ASSERT != CUDA_SUCCESS){ \
			const char* errorString; \
			(void)cuGetErrorString(v_result_RTLIB_BACKENDS_CUDA_DEBUG_ASSERT,&errorString); \
			fmt::print("RTLib Error '{}' Happen In Call: {}, File: {}, Line: {}",errorString, #EXPR,__FILE__,__LINE__); \
			assert(false); \
		} \
	}while(0)
#else
#define RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(EXPR) (void)EXPR
#endif
namespace RTLib
{
	namespace Backends {
		namespace Cuda {
			enum class ArrayFormat;
			enum class FilterMode;
			enum class AddressMode;
			enum class JitOption;
			class Device;
			class Context;
			class Stream;
			class LinearMemory;
			class LinearMemory2D;
			class Array;
			class MipmappedArray;
			class Module;
			class Function;
			namespace Internals {
				auto GetCUarray_format(const ArrayFormat& arrayFormat)->CUarray_format;
				auto GetCUdevice(const Device& device) noexcept -> CUdevice;
				auto GetCUcontext(const Context* context)noexcept -> CUcontext;
				auto GetCUstream(const Stream* stream)noexcept -> CUstream;
				auto GetCUdeviceptr(const LinearMemory* memory)noexcept -> CUdeviceptr;
				auto GetCUarray(const Array* array)noexcept -> CUarray;
				auto GetCUmipmappedArray(const MipmappedArray* array)noexcept -> CUmipmappedArray;
				auto GetCUfunction(const Function* function)->CUfunction;
				auto GetCUfilter_mode(FilterMode filterMode)->CUfilter_mode;
				auto GetCUaddress_mode(AddressMode addressMode)->CUaddress_mode;
				auto GetCUjit_option(JitOption jitOption)->CUjit_option;
				
				void SetCudaMemcpy2DMemoryCopy2D     (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::Memory2DCopy& memCpy);
				void SetCudaMemcpy2DSrcLinearMemory  (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory  * pSrc);
				void SetCudaMemcpy2DSrcLinearMemory2D(CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory2D* pSrc);
				void SetCudaMemcpy2DSrcArray         (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::Array*          pSrc);
				void SetCudaMemcpy2DSrcHost          (CUDA_MEMCPY2D& copy2D, const void* pSrc);
				void SetCudaMemcpy2DDstLinearMemory  (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory  * pDst);
				void SetCudaMemcpy2DDstLinearMemory2D(CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory2D* pDst);
				void SetCudaMemcpy2DDstArray         (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::Array*          pDst);
				void SetCudaMemcpy2DDstHost          (CUDA_MEMCPY2D& copy2D,       void* pDst);

				void SetCudaTextureDesc(CUDA_TEXTURE_DESC& desc, const TextureDesc& tex_desc);
				
			}
		}
	}
}
#endif

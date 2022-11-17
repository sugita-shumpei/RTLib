#include <RTLib/Backends/CUDA/CUDADevice.h>
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include <RTLib/Backends/CUDA/CUDAStream.h>
#include <RTLib/Backends/CUDA/CUDALinearMemory.h>
#include <RTLib/Backends/CUDA/CUDAArray.h>
#include <RTLib/Backends/CUDA/CUDAMipmappedArray.h>
#include <RTLib/Backends/CUDA/CUDAModule.h>
#include <RTLib/Backends/CUDA/CUDAFunction.h>
#include "CUDAInternals.h"

auto RTLib::Backends::Cuda::Internals::GetCUarray_format(const ArrayFormat& arrayFormat)->CUarray_format
{
	switch (arrayFormat)
	{
	case RTLib::Backends::Cuda::ArrayFormat::eUnknown: return CU_AD_FORMAT_UNSIGNED_INT8;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUInt8: return CU_AD_FORMAT_UNSIGNED_INT8;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUInt16: return CU_AD_FORMAT_UNSIGNED_INT16;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUInt32: return CU_AD_FORMAT_UNSIGNED_INT32;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSInt8: return CU_AD_FORMAT_SIGNED_INT8;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSInt16: return CU_AD_FORMAT_SIGNED_INT16;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSInt32: return CU_AD_FORMAT_SIGNED_INT32;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eFloat16: return CU_AD_FORMAT_HALF;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eFloat32:return CU_AD_FORMAT_FLOAT;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eNV12:return CU_AD_FORMAT_NV12;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUnormInt8X1:return CU_AD_FORMAT_UNORM_INT8X1;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUnormInt8X2:return CU_AD_FORMAT_UNORM_INT8X2;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUnormInt8X4:return CU_AD_FORMAT_UNORM_INT8X4;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUnormInt16X1:return CU_AD_FORMAT_UNORM_INT16X1;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUnormInt16X2:return CU_AD_FORMAT_UNORM_INT16X2;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eUnormInt16X4:return CU_AD_FORMAT_UNORM_INT16X4;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSnormInt8X1:return CU_AD_FORMAT_SNORM_INT8X1;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSnormInt8X2:return CU_AD_FORMAT_SNORM_INT8X2;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSnormInt8X4:return CU_AD_FORMAT_SNORM_INT8X4;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSnormInt16X1:return CU_AD_FORMAT_SNORM_INT16X1;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSnormInt16X2:return CU_AD_FORMAT_SNORM_INT16X2;
		break;
	case RTLib::Backends::Cuda::ArrayFormat::eSnormInt16X4:return CU_AD_FORMAT_SNORM_INT16X4;
		break;
	default: return CU_AD_FORMAT_UNSIGNED_INT8;
		break;
	}
}
auto RTLib::Backends::Cuda::Internals::GetCUdevice(const Device& device) noexcept -> CUdevice
{
	return *static_cast<const CUdevice*>(device.GetHandle());
}

auto RTLib::Backends::Cuda::Internals::GetCUcontext(const Context* context)noexcept -> CUcontext {
	if (!context) { return nullptr; }
	return static_cast<CUcontext>(context->GetHandle());
}

auto RTLib::Backends::Cuda::Internals::GetCUstream(const Stream* stream)noexcept -> CUstream {
	if (!stream) { return nullptr; }
	return static_cast<CUstream>(stream->GetHandle());
}

auto RTLib::Backends::Cuda::Internals::GetCUdeviceptr(const LinearMemory* memory)noexcept -> CUdeviceptr {
	if (!memory) { return 0; }
	return reinterpret_cast<CUdeviceptr>(memory->GetHandle());
}

auto RTLib::Backends::Cuda::Internals::GetCUarray(const Array* array)noexcept -> CUarray {
	if (!array) { return nullptr; }
	return reinterpret_cast<CUarray>(array->GetHandle());
}

auto RTLib::Backends::Cuda::Internals::GetCUmipmappedArray(const MipmappedArray* array)noexcept -> CUmipmappedArray
{
	if (!array) { return nullptr; }
	return reinterpret_cast<CUmipmappedArray>(array->GetHandle());
}
auto RTLib::Backends::Cuda::Internals::GetCUfunction(const Function* function) -> CUfunction
{
	return static_cast<CUfunction>(function->GetHandle());
}
auto RTLib::Backends::Cuda::Internals::GetCUfilter_mode(FilterMode filterMode) -> CUfilter_mode
{
	switch (filterMode)
	{
	case RTLib::Backends::Cuda::FilterMode::ePoint: return CU_TR_FILTER_MODE_POINT;
		break;
	case RTLib::Backends::Cuda::FilterMode::eLinear:return CU_TR_FILTER_MODE_LINEAR;
		break;
	default:
		break;
	}
}
auto RTLib::Backends::Cuda::Internals::GetCUaddress_mode(AddressMode addressMode) -> CUaddress_mode
{
	switch (addressMode)
	{
	case RTLib::Backends::Cuda::AddressMode::eWarp: return CU_TR_ADDRESS_MODE_WRAP;
		break;
	case RTLib::Backends::Cuda::AddressMode::eClamp:return CU_TR_ADDRESS_MODE_CLAMP;
		break;
	case RTLib::Backends::Cuda::AddressMode::eMirror:return CU_TR_ADDRESS_MODE_MIRROR;
		break;
	case RTLib::Backends::Cuda::AddressMode::eBorder:return CU_TR_ADDRESS_MODE_BORDER;
		break;
	default:
		break;
	}
}
auto RTLib::Backends::Cuda::Internals::GetCUjit_option(JitOption jitOption) -> CUjit_option
{
	switch (jitOption)
	{
	case RTLib::Backends::Cuda::JitOption::eMaxRegisters: return CU_JIT_MAX_REGISTERS;
		break;
	case RTLib::Backends::Cuda::JitOption::eThreadsPerBlock:return CU_JIT_THREADS_PER_BLOCK;
		break;
	case RTLib::Backends::Cuda::JitOption::eWallTime:return CU_JIT_WALL_TIME;
		break;
	case RTLib::Backends::Cuda::JitOption::eInfoLogBuffer:return CU_JIT_INFO_LOG_BUFFER;
		break;
	case RTLib::Backends::Cuda::JitOption::eInfoLogBufferSizeBytes:return CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
		break;
	case RTLib::Backends::Cuda::JitOption::eErrorLogBuffer:return CU_JIT_ERROR_LOG_BUFFER;
		break;
	case RTLib::Backends::Cuda::JitOption::eErrorLogBufferSizeBytes:return CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
		break;
	case RTLib::Backends::Cuda::JitOption::eOptimizationLevel:return CU_JIT_OPTIMIZATION_LEVEL;
		break;
	case RTLib::Backends::Cuda::JitOption::eTargetFromContext:return CU_JIT_TARGET_FROM_CUCONTEXT;
		break;
	case RTLib::Backends::Cuda::JitOption::eTarget:return CU_JIT_TARGET;
		break;
	case RTLib::Backends::Cuda::JitOption::eFallbackStrategy:return CU_JIT_FALLBACK_STRATEGY;
		break;
	case RTLib::Backends::Cuda::JitOption::eGenerateDebugInfo:return CU_JIT_GENERATE_DEBUG_INFO;
		break;
	case RTLib::Backends::Cuda::JitOption::eLogVerbose:return CU_JIT_LOG_VERBOSE;
		break;
	case RTLib::Backends::Cuda::JitOption::eGenerateLineInfo:return CU_JIT_GENERATE_LINE_INFO;
		break;
	case RTLib::Backends::Cuda::JitOption::eCacheMode:return CU_JIT_CACHE_MODE;
		break;
	case RTLib::Backends::Cuda::JitOption::eFastCompile:return CU_JIT_FAST_COMPILE;
		break;
	case RTLib::Backends::Cuda::JitOption::eGlobalSymbolNames:return CU_JIT_GLOBAL_SYMBOL_NAMES;
		break;
	case RTLib::Backends::Cuda::JitOption::eGlobalSymbolAddresses:return CU_JIT_GLOBAL_SYMBOL_ADDRESSES;
		break;
	case RTLib::Backends::Cuda::JitOption::eGlobalSymbolCount:return CU_JIT_GLOBAL_SYMBOL_COUNT;
		break;
	case RTLib::Backends::Cuda::JitOption::eLTO:return CU_JIT_LTO;
		break;
	case RTLib::Backends::Cuda::JitOption::eFTZ:return CU_JIT_FTZ;
		break;
	case RTLib::Backends::Cuda::JitOption::ePrecDiv:return CU_JIT_PREC_DIV;
		break;
	case RTLib::Backends::Cuda::JitOption::ePrecSqrt:return CU_JIT_PREC_SQRT;
		break;
	case RTLib::Backends::Cuda::JitOption::eFma:return CU_JIT_FMA;
		break;
	case RTLib::Backends::Cuda::JitOption::eReferencedKernelNames:return CU_JIT_REFERENCED_KERNEL_NAMES;
		break;
	case RTLib::Backends::Cuda::JitOption::eReferencedKernelCount:return CU_JIT_REFERENCED_KERNEL_COUNT;
		break;
	case RTLib::Backends::Cuda::JitOption::eReferencedVariableNames:return CU_JIT_REFERENCED_VARIABLE_NAMES;
		break;
	case RTLib::Backends::Cuda::JitOption::eReferencedVariableCount:return CU_JIT_REFERENCED_VARIABLE_COUNT;
		break;
	case RTLib::Backends::Cuda::JitOption::eOptimizeUnusedDeviceVariables:return CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES;
		break;
	default:
		break;
	}
}
void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DMemoryCopy2D     (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::Memory2DCopy& memCpy) {
	copy2D.dstXInBytes = memCpy.dstXOffsetInBytes;
	copy2D.dstY = memCpy.dstYOffset;
	copy2D.dstPitch = memCpy.dstPitchInBytes;
	copy2D.srcXInBytes = memCpy.srcXOffsetInBytes;
	copy2D.srcY= memCpy.srcYOffset;
	copy2D.srcPitch = memCpy.srcPitchInBytes;
	copy2D.WidthInBytes = memCpy.widthInBytes;
	copy2D.Height = memCpy.height;
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DSrcLinearMemory  (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory  * pSrc) {
	copy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copy2D.srcDevice = GetCUdeviceptr(pSrc);
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DSrcLinearMemory2D(CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory2D* pSrc) {
	assert(pSrc);
	copy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copy2D.srcDevice = GetCUdeviceptr(pSrc);
	copy2D.srcPitch = pSrc->GetPitchSizeInBytes();
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DSrcArray         (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::Array*          pSrc) {
	copy2D.srcMemoryType = CU_MEMORYTYPE_ARRAY;
	copy2D.srcArray = GetCUarray(pSrc);
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DSrcHost          (CUDA_MEMCPY2D& copy2D, const void* pSrc) {
	copy2D.srcMemoryType = CU_MEMORYTYPE_HOST;
	copy2D.srcHost = pSrc;
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DDstLinearMemory  (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory  * pDst) {
	copy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copy2D.dstDevice = GetCUdeviceptr(pDst);
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DDstLinearMemory2D(CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::LinearMemory2D* pDst) {
	assert(pDst);
	copy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
	copy2D.dstDevice = GetCUdeviceptr(pDst);
	copy2D.dstPitch = pDst->GetPitchSizeInBytes();
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DDstArray         (CUDA_MEMCPY2D& copy2D, const RTLib::Backends::Cuda::Array*          pDst) {
	copy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy2D.dstArray = GetCUarray(pDst);
}

void RTLib::Backends::Cuda::Internals::SetCudaMemcpy2DDstHost          (CUDA_MEMCPY2D& copy2D,       void* pDst) {
	copy2D.dstMemoryType = CU_MEMORYTYPE_HOST;
	copy2D.dstHost = pDst;
}

void RTLib::Backends::Cuda::Internals::SetCudaTextureDesc(CUDA_TEXTURE_DESC& desc, const TextureDesc& tex_desc)
{
	desc.addressMode[0] = GetCUaddress_mode(tex_desc.addressMode[0]);
	desc.addressMode[1] = GetCUaddress_mode(tex_desc.addressMode[1]);
	desc.addressMode[2] = GetCUaddress_mode(tex_desc.addressMode[2]);
	desc.filterMode     = GetCUfilter_mode(tex_desc.filterMode);
	desc.flags = 0;
	if (tex_desc.disableTrilinearOptimization) {
		desc.flags |= CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION;
	}
	if (tex_desc.readMode == TextureReadMode::eElementType) {
		desc.flags |= CU_TRSF_READ_AS_INTEGER;
	}
	if (tex_desc.normalizedCoords) {
		desc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
	}
	if (tex_desc.sRGB) {
		desc.flags |= CU_TRSF_SRGB;
	}
	desc.maxAnisotropy = tex_desc.maxAnisotropy;
	desc.mipmapFilterMode  = GetCUfilter_mode(tex_desc.mipmapFilterMode);
	desc.mipmapLevelBias = tex_desc.mipmapFilterBias;
	desc.maxMipmapLevelClamp = tex_desc.maxMipmapLevelClamp;
	desc.minMipmapLevelClamp = tex_desc.minMipmapLevelClamp;
	desc.borderColor[0] = tex_desc.borderColor[0];
	desc.borderColor[1] = tex_desc.borderColor[1];
	desc.borderColor[2] = tex_desc.borderColor[2];
	desc.borderColor[3] = tex_desc.borderColor[3];
}

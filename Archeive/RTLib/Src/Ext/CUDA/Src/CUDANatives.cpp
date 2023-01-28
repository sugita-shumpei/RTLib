#include <RTLib/Ext/CUDA/CUDANatives.h>
#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Ext/CUDA/CUDATexture.h>
#include <RTLib/Ext/CUDA/CUDAModule.h>
#include <RTLib/Ext/CUDA/CUDAFunction.h>
auto RTLib::Ext::CUDA::CUDANatives::GetCUDABuffer(CUDAContext* ctx, CUdeviceptr deviceptr, size_t sizeInBytes, CUDAMemoryFlags flags) -> CUDABuffer*
{
    CUDABufferCreateDesc desc = {};
    desc.flags = flags;
    desc.sizeInBytes = sizeInBytes;
    return new CUDABuffer(ctx, desc, deviceptr, false);
}
auto RTLib::Ext::CUDA::CUDANatives::GetCUDAImage(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUarray cuArray) -> CUDAImage*
{
    return new CUDA::CUDAImage(ctx,desc,cuArray,false);
}
auto RTLib::Ext::CUDA::CUDANatives::GetCUDAImage(CUDAContext* ctx, const CUDAImageCreateDesc& desc, CUmipmappedArray cuArray, const std::vector<CUarray>& cuArrayRefs) -> CUDAImage*
{
    return new CUDA::CUDAImage(ctx,desc,cuArray,cuArrayRefs,false);
}
auto RTLib::Ext::CUDA::CUDANatives::GetCUcontext(CUDAContext* context) -> CUcontext
{
    return context ? context->GetCUcontext() : nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUdevice(CUDAContext* context) -> CUdevice
{
    return context ? context->GetCUdevice() : 0;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(CUDABuffer* buffer) -> CUdeviceptr
{
    return buffer ? buffer->GetCUdeviceptr() : 0;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUdeviceptr(const CUDABufferView& bufferView) -> CUdeviceptr
{
    return bufferView.GetCUdeviceptr();
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUmodule(CUDAModule* module) -> CUmodule
{
    return module ? module->GetCUmodule() : nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUfunction(CUDAFunction* function) -> CUfunction
{
    return function ? function->GetCUfunction() : nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUarray(CUDAImage* image) -> CUarray
{
    return image ? image->GetCUarray():nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUarrayWithLevel(CUDAImage* image, unsigned int level) -> CUarray
{
    return image ? image->GetCUarrayWithLevel(level) : nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUmipmappedArray(CUDAImage* image) -> CUmipmappedArray
{
    return image ? image->GetCUmipmappedArray() : nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUstream(CUDAStream* stream) -> CUstream
{
    return stream ? stream->GetCUstream() : nullptr;
}

auto RTLib::Ext::CUDA::CUDANatives::GetCUtexObject(CUDATexture* texture) -> CUtexObject
{
    return texture ? texture->GetCUtexObject() : 0;
}

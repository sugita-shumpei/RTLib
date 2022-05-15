#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Core/Context.h>
#include <iostream>

RTLib::Ext::CUDA::CUDAContext::~CUDAContext() noexcept
{

}

bool RTLib::Ext::CUDA::CUDAContext::Initialize()
{
    {
        auto res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            const char* errString = nullptr;
            (void)cuGetErrorString(res,&errString);
            std::cout << __FILE__ << ":" << __LINE__<< ":" << std::string(errString) << "\n";
            return false;
        }
    }
    {
        auto res = cuDeviceGet(&m_DevCU, 0);
        if (res != CUDA_SUCCESS) {
            const char* errString = nullptr;
            (void)cuGetErrorString(res, &errString);
            std::cout << __FILE__ << ":" << __LINE__<< ":" << std::string(errString) << "\n";
            return false;
        }
    }
    {
        auto res = cuCtxCreate(&m_CtxCU, 0,m_DevCU);
        if (res != CUDA_SUCCESS) {
            const char* errString = nullptr;
            (void)cuGetErrorString(res, &errString);
            std::cout << __FILE__ << ":" << __LINE__<< ":" << std::string(errString) << "\n";
            return false;
        }
    }
    return true;
}

void RTLib::Ext::CUDA::CUDAContext::Terminate()
{
    if (m_CtxCU) {
        auto res = cuCtxDestroy(m_CtxCU);
        if (res != CUDA_SUCCESS) {
            const char* errString = nullptr;
            (void)cuGetErrorString(res, &errString);
            std::cout << __FILE__ << ":" << __LINE__<< ":" << std::string(errString) << "\n";
        }
        m_CtxCU = nullptr;
    }
}

auto RTLib::Ext::CUDA::CUDAContext::CreateBuffer(const CUDABufferDesc& desc) -> CUDABuffer*
{
    return CUDABuffer::Allocate(this,desc);
}

auto RTLib::Ext::CUDA::CUDAContext::CreateImage(const CUDAImageDesc& desc) -> CUDAImage*
{
    return CUDAImage::Allocate(this,desc);
}

bool RTLib::Ext::CUDA::CUDAContext::CopyBuffer(CUDABuffer* srcBuffer, CUDABuffer* dstBuffer, const std::vector<CUDABufferCopy>& regions)
{
    if (!srcBuffer || !dstBuffer) { return false; }
    auto srcAddress = srcBuffer->GetDeviceAddress();
    auto dstAddress = dstBuffer->GetDeviceAddress();
    if (!srcAddress || !dstAddress) { return false; }
    auto result = CUDA_SUCCESS;
    for (auto& region : regions) {
        result =cuMemcpyDtoD(dstAddress + region.dstOffset, srcAddress + region.srcOffset, region.size);
        if (result != CUDA_SUCCESS) {
            break;
        }
    }
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return false;
    }
    return true;
}

bool RTLib::Ext::CUDA::CUDAContext::CopyMemoryToBuffer(CUDABuffer* buffer, const std::vector<CUDAMemoryBufferCopy>& regions)
{
    if(!buffer) { return false; }
    auto dstAddress = buffer->GetDeviceAddress();
    auto result = CUDA_SUCCESS;
    for (auto& region : regions) {
        result = cuMemcpyHtoD(dstAddress + region.dstOffset, region.srcData, region.size);
        if (result != CUDA_SUCCESS) {
            break;
        }
    }
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return false;
    }
    return true;
}

bool RTLib::Ext::CUDA::CUDAContext::CopyBufferToMemory(CUDABuffer* buffer, const std::vector<CUDABufferMemoryCopy>& regions)
{
    if (!buffer) { return false; }
    auto srcAddress = buffer->GetDeviceAddress();
    auto result = CUDA_SUCCESS;
    for (auto& region : regions) {
        result = cuMemcpyDtoH(region.dstData, srcAddress + region.srcOffset, region.size);
        if (result != CUDA_SUCCESS) {
            break;
        }
    }
    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return false;
    }
    return true;
}

bool RTLib::Ext::CUDA::CUDAContext::CopyImageToBuffer(CUDAImage* srcImage, CUDABuffer* dstBuffer, const std::vector<CUDABufferImageCopy>& regions)
{
    if (!srcImage || !dstBuffer) { return false; }
    auto dstAddress = dstBuffer->GetDeviceAddress();
    if (!dstAddress) { return false; }
    auto srcImageType = srcImage->GetImageType();
    auto srcFormat   = srcImage->GetFormat();
    auto srcChannel  = srcImage->GetChannels();
    auto srcTypeSize = srcChannel * GetCUDAImageDataTypeSize(srcFormat) / 8;
    auto srcLayers   = srcImage->GetLayers();
    auto srcLevels   = srcImage->GetLevels();
    auto result      = CUDA_SUCCESS;
    if (srcLayers == 0) {
        if (srcImageType == CUDAImageType::e1D) {
            for (auto& region : regions) {
                if (region.imageSubresources.mipLevels >= srcLevels) {
                    return false;
                }
            }
            for (auto& region : regions) {
                auto srcArray = srcImage->GetArrays(region.imageSubresources.mipLevels);
                result = cuMemcpyAtoD(dstAddress + region.bufferOffset, srcArray, region.imageOffset.x * srcTypeSize, region.imageExtent.width * srcTypeSize);
                if (result != CUDA_SUCCESS) {
                    break;
                }
            }
        }
        if (srcImageType == CUDAImageType::e2D) {
            auto memCpy2Ds = std::vector<CUDA_MEMCPY2D>();
            for (auto& region   : regions) {
                auto srcArray = srcImage->GetArrays(region.imageSubresources.mipLevels);
                if (!srcArray) { return false; }
                CUDA_MEMCPY2D memCpy2D = {};
                memCpy2D.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memCpy2D.srcArray      = srcArray;
                memCpy2D.srcXInBytes   = region.imageOffset.x * srcTypeSize;//OK
                memCpy2D.srcY          = region.imageOffset.y;//OK
                memCpy2D.srcPitch      = 0;//Ignore
                memCpy2D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memCpy2D.dstDevice     = dstAddress;
                memCpy2D.dstXInBytes   = region.bufferOffset;
                memCpy2D.dstY          = 0;
                memCpy2D.dstPitch      = 0;
                memCpy2D.WidthInBytes  = region.imageExtent.width * srcTypeSize;
                memCpy2D.Height        = region.imageExtent.height;
                memCpy2Ds.push_back(memCpy2D);
            }
            for (auto& memCpy2D : memCpy2Ds) {
                result = cuMemcpy2D(&memCpy2D);
                if (result != CUDA_SUCCESS) {
                    break;
                }
            }
        }
        if (srcImageType == CUDAImageType::e3D) {
            auto memCpy3Ds = std::vector<CUDA_MEMCPY3D>();
            for (auto& region : regions) {
                auto srcArray = srcImage->GetArrays(region.imageSubresources.mipLevels);
                if (!srcArray) { return false; }
                CUDA_MEMCPY3D memCpy3D = {};

                memCpy3D.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memCpy3D.srcArray      = srcArray;
                memCpy3D.srcXInBytes   = region.imageOffset.x * srcTypeSize;//OK
                memCpy3D.srcY          = region.imageOffset.y;//OK
                memCpy3D.srcZ          = region.imageOffset.z;
                memCpy3D.srcPitch      = 0;//Ignore
                memCpy3D.srcHeight     = 0;

                memCpy3D.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memCpy3D.dstDevice     = dstAddress;
                memCpy3D.dstXInBytes   = region.bufferOffset;
                memCpy3D.dstY          = 0;
                memCpy3D.dstZ          = 0;
                memCpy3D.dstPitch      = 0;
                memCpy3D.dstHeight     = 0;

                memCpy3D.WidthInBytes  = region.imageExtent.width * srcTypeSize;
                memCpy3D.Height        = region.imageExtent.height;
                memCpy3D.Depth         = region.imageExtent.depth;
                memCpy3Ds.push_back(memCpy3D);
            }
            for (auto& memCpy3D : memCpy3Ds) {
                result = cuMemcpy3D(&memCpy3D);
                if (result != CUDA_SUCCESS) {
                    break;
                }
            }
        }
    }else{

    }

    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return false;
    }
    return true;
}

bool RTLib::Ext::CUDA::CUDAContext::CopyBufferToImage(CUDABuffer* srcBuffer, CUDAImage* dstImage, const std::vector<CUDABufferImageCopy>& regions)
{
    if (!srcBuffer || !dstImage) { return false; }
    auto srcAddress = srcBuffer->GetDeviceAddress();
    if (!srcAddress) { return false; }
    auto dstImageType = dstImage->GetImageType();
    auto dstFormat    = dstImage->GetFormat();
    auto dstChannel   = dstImage->GetChannels();
    auto dstTypeSize  = dstChannel * GetCUDAImageDataTypeSize(dstFormat) / 8;
    auto dstLayers    = dstImage->GetLayers();
    auto result = CUDA_SUCCESS;
    if ((dstImageType == CUDAImageType::e2D) && (dstLayers == 0)) {
        auto memCpy2Ds = std::vector<CUDA_MEMCPY2D>();
        for (auto& region : regions) {
            auto dstArray = dstImage->GetArrays(region.imageSubresources.mipLevels);
            if (!dstArray) { return false; }
            CUDA_MEMCPY2D memCpy2D = {};
            memCpy2D.srcMemoryType = CU_MEMORYTYPE_DEVICE;
            memCpy2D.srcDevice     = srcAddress;
            memCpy2D.srcXInBytes   = region.bufferOffset;
            memCpy2D.srcY          = 0;//OK
            memCpy2D.srcPitch      = 0;//Ignore
            memCpy2D.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            memCpy2D.dstArray      = dstArray;
            memCpy2D.dstXInBytes   = region.imageOffset.x * dstTypeSize;
            memCpy2D.dstY          = region.imageOffset.y;
            memCpy2D.dstPitch      = 0;
            memCpy2D.WidthInBytes  = region.imageExtent.width * dstTypeSize;
            memCpy2D.Height        = region.imageExtent.height;
            memCpy2Ds.push_back(memCpy2D);
        }
        for (auto& memCpy2D : memCpy2Ds) {
            result = cuMemcpy2D(&memCpy2D);
            if (result != CUDA_SUCCESS) {
                break;
            }
        }
    }

    if (result != CUDA_SUCCESS) {
        const char* errString = nullptr;
        (void)cuGetErrorString(result, &errString);
        std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
        return false;
    }
    return true;
}

bool RTLib::Ext::CUDA::CUDAContext::CopyImageToMemory(CUDAImage* image, const std::vector<CUDAImageMemoryCopy>& regions)
{
    return false;
}

bool RTLib::Ext::CUDA::CUDAContext::CopyMemoryToImage(CUDAImage* image, const std::vector<CUDAImageMemoryCopy>& regions)
{
    return false;
}

#include <RTLib/Ext/CUDA/CUDAContext.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <RTLib/Ext/CUDA/CUDATexture.h>
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

auto RTLib::Ext::CUDA::CUDAContext::CreateTexture(const CUDATextureImageDesc& desc) -> CUDATexture*
{
    return CUDATexture::Allocate(this, desc);
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

bool RTLib::Ext::CUDA::CUDAContext::CopyImageToBuffer(CUDAImage* image, CUDABuffer* buffer, const std::vector<CUDABufferImageCopy>& regions)
{
    if (!image || !buffer) { return false; }
    auto bffAddress  = buffer->GetDeviceAddress();
    auto imgType     = image->GetImageType();
    auto imgFormat   = image->GetFormat();
    auto imgDataSize = image->GetChannels()* GetCUDAImageDataTypeSize(imgFormat)/8;
    auto imgLevels   = image->GetLevels();
    auto imgLayers   = image->GetLayers();
    auto result      = CUDA_SUCCESS;
    if (imgLayers == 0) {
        if (imgType == CUDAImageType::e1D) {
            for (auto& region : regions) {
                if (imgLevels > 0) {
                    if (region.imageSubresources.mipLevels >= imgLevels) { return false; }
                }
                else {
                    if (region.imageSubresources.mipLevels != 0) { return false; }
                }
            }
            for (auto& region : regions) {
                result = cuMemcpyAtoD(bffAddress + region.bufferOffset, image->GetArrays(region.imageSubresources.mipLevels), region.imageOffset.x * imgDataSize, region.imageExtent.width * imgDataSize);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memcpy2d.dstDevice = bffAddress+ region.bufferOffset;
                memcpy2d.dstXInBytes = 0;
                memcpy2d.dstY = 0;
                memcpy2d.dstPitch = 0;
                memcpy2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy2d.srcArray = image->GetArrays(region.imageSubresources.mipLevels);
                memcpy2d.srcXInBytes = region.imageOffset.x * imgDataSize;
                memcpy2d.srcY = region.imageOffset.y;
                memcpy2d.srcPitch = 0;
                memcpy2d.WidthInBytes = region.imageExtent.width * imgDataSize;
                memcpy2d.Height = region.imageExtent.height;
                if (!memcpy2d.dstDevice || !memcpy2d.srcArray) { return false; }
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2D(&memcpy2d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e3D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memcpy3d.dstDevice = bffAddress + region.bufferOffset;
                memcpy3d.dstXInBytes = 0;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = 0;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.srcArray = image->GetArrays(region.imageSubresources.mipLevels);
                memcpy3d.srcXInBytes = region.imageOffset.x * imgDataSize;
                memcpy3d.srcY = region.imageOffset.y;
                memcpy3d.srcZ = region.imageOffset.z;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.WidthInBytes = region.imageExtent.width * imgDataSize;
                memcpy3d.Height = region.imageExtent.height;
                memcpy3d.Depth = region.imageExtent.depth;
                if (!memcpy3d.dstDevice || !memcpy3d.srcArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
    }
    else {
        if (imgType == CUDAImageType::e1D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memcpy3d.dstDevice = bffAddress + region.bufferOffset;
                memcpy3d.dstXInBytes = 0;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = 0;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.srcArray = image->GetArrays(region.imageSubresources.mipLevels);
                memcpy3d.srcXInBytes = region.imageOffset.x * imgDataSize;
                memcpy3d.srcY = 0;
                memcpy3d.srcZ = region.imageSubresources.baseArrayLayer;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.WidthInBytes = region.imageExtent.width * imgDataSize;
                memcpy3d.Height = 0;
                memcpy3d.Depth = region.imageSubresources.layerCount;
                if (!memcpy3d.dstDevice || !memcpy3d.srcArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memcpy3d.dstDevice = bffAddress + region.bufferOffset;
                memcpy3d.dstXInBytes = 0;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = 0;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.srcArray = image->GetArrays(region.imageSubresources.mipLevels);
                memcpy3d.srcXInBytes = region.imageOffset.x * imgDataSize;
                memcpy3d.srcY = region.imageOffset.y;
                memcpy3d.srcZ = region.imageSubresources.baseArrayLayer;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.WidthInBytes = region.imageExtent.width * imgDataSize;
                memcpy3d.Height = region.imageExtent.height;
                memcpy3d.Depth = region.imageSubresources.layerCount;
                if (!memcpy3d.dstDevice || !memcpy3d.srcArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
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

bool RTLib::Ext::CUDA::CUDAContext::CopyBufferToImage(CUDABuffer* buffer, CUDAImage* image, const std::vector<CUDABufferImageCopy>& regions)
{
    if (!image || !buffer) { return false; }
    auto bffAddress = buffer->GetDeviceAddress();
    auto imgType = image->GetImageType();
    auto imgFormat = image->GetFormat();
    auto imgDataSize = image->GetChannels() * GetCUDAImageDataTypeSize(imgFormat) / 8;
    auto imgLevels = image->GetLevels();
    auto imgLayers = image->GetLayers();
    auto result = CUDA_SUCCESS;
    if (imgLayers == 0) {
        if (imgType == CUDAImageType::e1D) {
            for (auto& region : regions) {
                if (imgLevels > 0) {
                    if (region.imageSubresources.mipLevels >= imgLevels) { return false; }
                }
                else {
                    if (region.imageSubresources.mipLevels != 0) { return false; }
                }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2D(&memcpy2d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e3D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
    }
    else {
        if (imgType == CUDAImageType::e1D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2D(&memcpy2d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
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
    if (!image) { return false; }
    auto imgType = image->GetImageType();
    auto imgFormat = image->GetFormat();
    auto imgDataSize = image->GetChannels() * GetCUDAImageDataTypeSize(imgFormat) / 8;
    auto imgLevels = image->GetLevels();
    auto imgLayers = image->GetLayers();
    auto result = CUDA_SUCCESS;
    if (imgLayers == 0) {
        if (imgType == CUDAImageType::e1D) {
            for (auto& region : regions) {
                if (imgLevels > 0) {
                    if (region.srcImageSubresources.mipLevels >= imgLevels) { return false; }
                }
                else {
                    if (region.srcImageSubresources.mipLevels != 0) { return false; }
                }
            }
            for (auto& region : regions) {
                result = cuMemcpyHtoA(image->GetArrays(region.srcImageSubresources.mipLevels), region.srcImageOffset.x * imgDataSize, region.dstData, region.srcImageExtent.width * imgDataSize);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
                memcpy2d.dstHost       = region.dstData;
                memcpy2d.dstXInBytes   = 0;
                memcpy2d.dstY          = 0;
                memcpy2d.dstPitch      = 0;
                memcpy2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy2d.srcArray      = image->GetArrays(region.srcImageSubresources.mipLevels);
                memcpy2d.srcXInBytes   = region.srcImageOffset.x * imgDataSize;
                memcpy2d.srcY          = region.srcImageOffset.y;
                memcpy2d.srcPitch      = 0;
                memcpy2d.WidthInBytes  = region.srcImageExtent.width* imgDataSize;
                memcpy2d.Height        = region.srcImageExtent.height;
                if (!memcpy2d.dstHost || !memcpy2d.srcArray) { return false; }
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2D(&memcpy2d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e3D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
                memcpy3d.dstHost = region.dstData;
                memcpy3d.dstXInBytes = 0;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = 0;
                memcpy3d.dstPitch  = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.srcArray = image->GetArrays(region.srcImageSubresources.mipLevels);
                memcpy3d.srcXInBytes = region.srcImageOffset.x * imgDataSize;
                memcpy3d.srcY = region.srcImageOffset.y;
                memcpy3d.srcZ = region.srcImageOffset.z;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.WidthInBytes = region.srcImageExtent.width * imgDataSize;
                memcpy3d.Height = region.srcImageExtent.height;
                memcpy3d.Depth = region.srcImageExtent.depth;
                if (!memcpy3d.dstHost || !memcpy3d.srcArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
    }
    else {
        if (imgType == CUDAImageType::e1D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
                memcpy3d.dstHost = region.dstData;
                memcpy3d.dstXInBytes = 0;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = 0;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.srcArray = image->GetArrays(region.srcImageSubresources.mipLevels);
                memcpy3d.srcXInBytes = region.srcImageOffset.x * imgDataSize;
                memcpy3d.srcY = 0;
                memcpy3d.srcZ = region.srcImageSubresources.baseArrayLayer;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.WidthInBytes = region.srcImageExtent.width * imgDataSize;
                memcpy3d.Height = 0;
                memcpy3d.Depth = region.srcImageSubresources.layerCount;
                if (!memcpy3d.dstHost || !memcpy3d.srcArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_HOST;
                memcpy3d.dstHost = region.dstData;
                memcpy3d.dstXInBytes = 0;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = 0;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.srcArray = image->GetArrays(region.srcImageSubresources.mipLevels);
                memcpy3d.srcXInBytes = region.srcImageOffset.x * imgDataSize;
                memcpy3d.srcY = region.srcImageOffset.y;
                memcpy3d.srcZ = region.srcImageSubresources.baseArrayLayer;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.WidthInBytes = region.srcImageExtent.width * imgDataSize;
                memcpy3d.Height = region.srcImageExtent.height;
                memcpy3d.Depth = region.srcImageSubresources.layerCount;
                if (!memcpy3d.dstHost || !memcpy3d.srcArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
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

bool RTLib::Ext::CUDA::CUDAContext::CopyMemoryToImage(CUDAImage* image, const std::vector<CUDAMemoryImageCopy>& regions)
{
    if (!image) { return false; }
    auto imgType = image->GetImageType();
    auto imgFormat = image->GetFormat();
    auto imgDataSize = image->GetChannels() * GetCUDAImageDataTypeSize(imgFormat) / 8;
    auto imgLevels = image->GetLevels();
    auto imgLayers = image->GetLayers();
    auto result = CUDA_SUCCESS;
    if (imgLayers == 0) {
        if (imgType == CUDAImageType::e1D) {
            for (auto& region : regions) {
                if (imgLevels > 0) {
                    if (region.dstImageSubresources.mipLevels >= imgLevels) { return false; }
                }
                else {
                    if (region.dstImageSubresources.mipLevels != 0) { return false; }
                }
            }
            for (auto& region : regions) {
                result = cuMemcpyHtoA(image->GetArrays(region.dstImageSubresources.mipLevels), region.dstImageOffset.x * imgDataSize, region.srcData, region.dstImageExtent.width * imgDataSize);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2d.srcMemoryType = CU_MEMORYTYPE_HOST;
                memcpy2d.srcHost       = region.srcData;
                memcpy2d.srcXInBytes   = 0;
                memcpy2d.srcY          = 0;
                memcpy2d.srcPitch      = 0;
                memcpy2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy2d.dstArray      = image->GetArrays(region.dstImageSubresources.mipLevels);
                memcpy2d.dstXInBytes   = region.dstImageOffset.x * imgDataSize;
                memcpy2d.dstY          = region.dstImageOffset.y;
                memcpy2d.dstPitch      = 0;
                memcpy2d.WidthInBytes  = region.dstImageExtent.width* imgDataSize;
                memcpy2d.Height        = region.dstImageExtent.height;
                if (!memcpy2d.srcHost || !memcpy2d.dstArray) { return false; }
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2D(&memcpy2d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e3D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
                memcpy3d.srcHost = region.srcData;
                memcpy3d.srcXInBytes = 0;
                memcpy3d.srcY = 0;
                memcpy3d.srcZ = 0;
                memcpy3d.srcPitch  = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.dstArray = image->GetArrays(region.dstImageSubresources.mipLevels);
                memcpy3d.dstXInBytes = region.dstImageOffset.x * imgDataSize;
                memcpy3d.dstY = region.dstImageOffset.y;
                memcpy3d.dstZ = region.dstImageOffset.z;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.WidthInBytes = region.dstImageExtent.width * imgDataSize;
                memcpy3d.Height = region.dstImageExtent.height;
                memcpy3d.Depth = region.dstImageExtent.depth;
                if (!memcpy3d.srcHost || !memcpy3d.dstArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
    }
    else {
        if (imgType == CUDAImageType::e1D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
                memcpy3d.srcHost = region.srcData;
                memcpy3d.srcXInBytes = 0;
                memcpy3d.srcY = 0;
                memcpy3d.srcZ = 0;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.dstArray = image->GetArrays(region.dstImageSubresources.mipLevels);
                memcpy3d.dstXInBytes = region.dstImageOffset.x * imgDataSize;
                memcpy3d.dstY = 0;
                memcpy3d.dstZ = region.dstImageSubresources.baseArrayLayer;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.WidthInBytes = region.dstImageExtent.width * imgDataSize;
                memcpy3d.Height = 0;
                memcpy3d.Depth = region.dstImageSubresources.layerCount;
                if (!memcpy3d.srcHost || !memcpy3d.dstArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY3D> memcpy3ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY3D memcpy3d = {};
                memcpy3d.srcMemoryType = CU_MEMORYTYPE_HOST;
                memcpy3d.srcHost = region.srcData;
                memcpy3d.srcXInBytes = 0;
                memcpy3d.srcY = 0;
                memcpy3d.srcZ = 0;
                memcpy3d.srcPitch = 0;
                memcpy3d.srcHeight = 0;
                memcpy3d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy3d.dstArray = image->GetArrays(region.dstImageSubresources.mipLevels);
                memcpy3d.dstXInBytes = region.dstImageOffset.x * imgDataSize;
                memcpy3d.dstY = region.dstImageOffset.y;
                memcpy3d.dstZ = region.dstImageSubresources.baseArrayLayer;
                memcpy3d.dstPitch = 0;
                memcpy3d.dstHeight = 0;
                memcpy3d.WidthInBytes = region.dstImageExtent.width * imgDataSize;
                memcpy3d.Height = region.dstImageExtent.height;
                memcpy3d.Depth = region.dstImageSubresources.layerCount;
                if (!memcpy3d.srcHost || !memcpy3d.dstArray) { return false; }
                memcpy3ds.push_back(memcpy3d);
            }
            for (auto& memcpy3d : memcpy3ds) {
                result = cuMemcpy3D(&memcpy3d);
                if (result != CUDA_SUCCESS) { break; }
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


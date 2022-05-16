#include <RTLib/Ext/CUDA/CUDAStream.h>
#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <RTLib/Ext/CUDA/CUDAImage.h>
#include <iostream>
#include <string>
auto RTLib::Ext::CUDA::CUDAStream::New(CUDAContext* context) -> CUDAStream*
{
	if (!context) { return nullptr; }
	CUstream stream;
	auto result = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
	if (result != CUDA_SUCCESS) {
		const char* errString = nullptr;
		(void)cuGetErrorString(result, &errString);
		std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
		return nullptr;
	}
	return new CUDAStream(context, stream);
}

RTLib::Ext::CUDA::CUDAStream::~CUDAStream() noexcept
{
	if (!m_Stream) {
		return;
	}
	auto result = cuStreamDestroy(m_Stream);
	if (result != CUDA_SUCCESS) {
		const char* errString = nullptr;
		(void)cuGetErrorString(result, &errString);
		std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
		
	}
	m_Stream = nullptr;
}

void RTLib::Ext::CUDA::CUDAStream::Destroy()
{
}

bool RTLib::Ext::CUDA::CUDAStream::Synchronize()
{
	if (!m_Stream) {
		return false;
	}
	auto result = cuStreamSynchronize(m_Stream);
	if (result != CUDA_SUCCESS) {
		const char* errString = nullptr;
		(void)cuGetErrorString(result, &errString);
		std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
		return false;
	}
	return true;
}

bool RTLib::Ext::CUDA::CUDAStream::CopyBuffer(CUDABuffer* srcBuffer, CUDABuffer* dstBuffer, const std::vector<CUDABufferCopy>& regions)
{
    if (!srcBuffer || !dstBuffer|| !m_Stream) { return false; }
    auto srcAddress = srcBuffer->GetDeviceAddress();
    auto dstAddress = dstBuffer->GetDeviceAddress();
    if (!srcAddress || !dstAddress) { return false; }
    auto result = CUDA_SUCCESS;
    for (auto& region : regions) {
        result = cuMemcpyDtoDAsync(dstAddress + region.dstOffset, srcAddress + region.srcOffset, region.size, m_Stream);
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

bool RTLib::Ext::CUDA::CUDAStream::CopyMemoryToBuffer(CUDABuffer* buffer, const std::vector<CUDAMemoryBufferCopy>& regions)
{
    if (!buffer|| !m_Stream) { return false; }
    auto dstAddress = buffer->GetDeviceAddress();
    auto result = CUDA_SUCCESS;
    for (auto& region : regions) {
        result = cuMemcpyHtoDAsync(dstAddress + region.dstOffset, region.srcData, region.size,m_Stream);
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

bool RTLib::Ext::CUDA::CUDAStream::CopyBufferToMemory(CUDABuffer* buffer, const std::vector<CUDABufferMemoryCopy>& regions)
{
    if (!buffer||!m_Stream) { return false; }
    auto srcAddress = buffer->GetDeviceAddress();
    auto result = CUDA_SUCCESS;
    for (auto& region : regions) {
        result = cuMemcpyDtoHAsync(region.dstData, srcAddress + region.srcOffset, region.size,m_Stream);
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

bool RTLib::Ext::CUDA::CUDAStream::CopyImageToBuffer(CUDAImage* image, CUDABuffer* buffer, const std::vector<CUDABufferImageCopy>& regions)
{
    if (!image || !buffer || !m_Stream) { return false; }
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
            if (Synchronize()) {
                for (auto& region : regions) {
                    result = cuMemcpyAtoD(bffAddress + region.bufferOffset, image->GetArrays(region.imageSubresources.mipLevels), region.imageOffset.x * imgDataSize, region.imageExtent.width * imgDataSize);
                    if (result != CUDA_SUCCESS) { break; }
                }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2d.dstMemoryType = CU_MEMORYTYPE_DEVICE;
                memcpy2d.dstDevice = bffAddress + region.bufferOffset;
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
                result = cuMemcpy2DAsync(&memcpy2d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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

bool RTLib::Ext::CUDA::CUDAStream::CopyBufferToImage(CUDABuffer* buffer, CUDAImage* image, const std::vector<CUDABufferImageCopy>& regions)
{
    if (!image || !buffer || !m_Stream) { return false; }
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
            if (Synchronize()) {
                for (auto& region : regions) {
                    result = cuMemcpyDtoA(image->GetArrays(region.imageSubresources.mipLevels), region.imageOffset.x * imgDataSize, bffAddress + region.bufferOffset, region.imageExtent.width * imgDataSize);
                    if (result != CUDA_SUCCESS) { break; }
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
                result = cuMemcpy2DAsync(&memcpy2d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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
                result = cuMemcpy2DAsync(&memcpy2d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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

bool RTLib::Ext::CUDA::CUDAStream::CopyImageToMemory(CUDAImage* image, const std::vector<CUDAImageMemoryCopy>& regions)
{
    if (!image || !m_Stream) { return false; }
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
                result = cuMemcpyHtoAAsync(image->GetArrays(region.srcImageSubresources.mipLevels), region.srcImageOffset.x * imgDataSize, region.dstData, region.srcImageExtent.width * imgDataSize,m_Stream);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2d.dstMemoryType = CU_MEMORYTYPE_HOST;
                memcpy2d.dstHost = region.dstData;
                memcpy2d.dstXInBytes = 0;
                memcpy2d.dstY = 0;
                memcpy2d.dstPitch = 0;
                memcpy2d.srcMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy2d.srcArray = image->GetArrays(region.srcImageSubresources.mipLevels);
                memcpy2d.srcXInBytes = region.srcImageOffset.x * imgDataSize;
                memcpy2d.srcY = region.srcImageOffset.y;
                memcpy2d.srcPitch = 0;
                memcpy2d.WidthInBytes = region.srcImageExtent.width * imgDataSize;
                memcpy2d.Height = region.srcImageExtent.height;
                if (!memcpy2d.dstHost || !memcpy2d.srcArray) { return false; }
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2DAsync(&memcpy2d,m_Stream);
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
                memcpy3d.dstPitch = 0;
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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

bool RTLib::Ext::CUDA::CUDAStream::CopyMemoryToImage(CUDAImage* image, const std::vector<CUDAMemoryImageCopy>& regions)
{
    if (!image || !m_Stream) { return false; }
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
                result = cuMemcpyHtoAAsync(image->GetArrays(region.dstImageSubresources.mipLevels), region.dstImageOffset.x * imgDataSize, region.srcData, region.dstImageExtent.width * imgDataSize,m_Stream);
                if (result != CUDA_SUCCESS) { break; }
            }
        }
        if (imgType == CUDAImageType::e2D) {
            std::vector<CUDA_MEMCPY2D> memcpy2ds = {};
            for (auto& region : regions) {
                CUDA_MEMCPY2D memcpy2d = {};
                memcpy2d.srcMemoryType = CU_MEMORYTYPE_HOST;
                memcpy2d.srcHost = region.srcData;
                memcpy2d.srcXInBytes = 0;
                memcpy2d.srcY = 0;
                memcpy2d.srcPitch = 0;
                memcpy2d.dstMemoryType = CU_MEMORYTYPE_ARRAY;
                memcpy2d.dstArray = image->GetArrays(region.dstImageSubresources.mipLevels);
                memcpy2d.dstXInBytes = region.dstImageOffset.x * imgDataSize;
                memcpy2d.dstY = region.dstImageOffset.y;
                memcpy2d.dstPitch = 0;
                memcpy2d.WidthInBytes = region.dstImageExtent.width * imgDataSize;
                memcpy2d.Height = region.dstImageExtent.height;
                if (!memcpy2d.srcHost || !memcpy2d.dstArray) { return false; }
                memcpy2ds.push_back(memcpy2d);
            }
            for (auto& memcpy2d : memcpy2ds) {
                result = cuMemcpy2DAsync(&memcpy2d, m_Stream);
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
                memcpy3d.srcPitch = 0;
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
                result = cuMemcpy3DAsync(&memcpy3d, m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d, m_Stream);
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
                result = cuMemcpy3DAsync(&memcpy3d,m_Stream);
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


RTLib::Ext::CUDA::CUDAStream::CUDAStream(CUDAContext* context, CUstream stream) noexcept:m_Context{context},m_Stream{stream}
{
}

auto RTLib::Ext::CUDA::CUDAStream::GetCUStream() noexcept -> CUstream
{
	return m_Stream;
}

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

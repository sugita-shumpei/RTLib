#include <RTLib/Backends/CUDA/CUDAFunction.h>
#include <RTLib/Backends/CUDA/CUDAModule.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include "CUDAInternals.h"
struct RTLib::Backends::Cuda::Function::Impl {
    Impl(void* pHandle) noexcept :function{ static_cast<CUfunction>(pHandle) } {

    }
    ~Impl() noexcept {}
    CUfunction function;
};
RTLib::Backends::Cuda::Function::Function(void* pHandle) noexcept
    :m_Impl{new Impl(pHandle)}
{}

RTLib::Backends::Cuda::Function::~Function() noexcept
{
    m_Impl.reset();
}

auto RTLib::Backends::Cuda::Function::GetHandle() const noexcept -> void*
{
    return m_Impl->function;
}

#include <RTLib/Backends/CUDA/CUDAStream.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include <RTLib/Backends/CUDA/CUDALinearMemory.h>
#include "CUDAInternals.h"
struct RTLib::Backends::Cuda::Stream::Impl {
    Impl(std::shared_ptr<void> stream) noexcept :m_Stream{ stream } {}
    ~Impl()noexcept {}

    std::weak_ptr<void> m_Stream;
};
RTLib::Backends::Cuda::Stream::Stream(std::shared_ptr<void> stream) noexcept:m_Impl{new Impl(stream)}{}

RTLib::Backends::Cuda::Stream::~Stream() noexcept
{
    m_Impl.reset();
}

bool RTLib::Backends::Cuda::Stream::IsValid() const noexcept
{
    return !m_Impl->m_Stream.expired();
}

auto RTLib::Backends::Cuda::Stream::GetHandle() const noexcept -> void*
{
    auto ptr = m_Impl->m_Stream.lock();
    if (ptr)
    {
        return ptr.get();
    }
    else {
        return nullptr;
    }
}


void RTLib::Backends::Cuda::Stream::Copy1DLinearMemory(const LinearMemory* dstMemory, const LinearMemory* srcMemory, const Memory1DCopy& copy)const noexcept {
	assert(IsValid()&&(dstMemory!=nullptr)&&(srcMemory!=nullptr));
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpyAsync(Internals::GetCUdeviceptr(dstMemory)+copy.dstOffsetInBytes, Internals::GetCUdeviceptr(srcMemory)+copy.srcOffsetInBytes,copy.sizeInBytes, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy1DFromLinearMemoryToHost(void* dstMemory, const LinearMemory* srcMemory, const Memory1DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpyDtoHAsync(static_cast<char*>(dstMemory) + copy.dstOffsetInBytes, Internals::GetCUdeviceptr(srcMemory) + copy.srcOffsetInBytes, copy.sizeInBytes, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy1DFromHostToLinearMemory(const LinearMemory* dstMemory,const void* srcMemory, const Memory1DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpyHtoDAsync(Internals::GetCUdeviceptr(dstMemory) + copy.dstOffsetInBytes, static_cast<const char*>(srcMemory) + copy.srcOffsetInBytes, copy.sizeInBytes, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DLinearMemory(const LinearMemory* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromLinearMemoryToHost(void* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstHost(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromHostToLinearMemory(const LinearMemory* dstMemory, const void* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcHost(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DLinearMemory2D(const LinearMemory2D* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromLinearMemory2DToLinearMemory(const LinearMemory* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromLinearMemoryToLinearMemory2D(const LinearMemory2D* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromLinearMemory2DToHost(void* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstHost(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromHostToLinearMemory2D(const LinearMemory2D* dstMemory,const void* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcHost(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DArray(const Array* dstArray, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstArray != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromArrayToHost(void* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstHost(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromHostToArray(const Array* dstArray, const void* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstArray != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcHost(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromArrayToLinearMemory(const LinearMemory* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstLinearMemory(memCpy2D,dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromLinearMemoryToArray(const Array* dstArray, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstArray != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromArrayToLinearMemory2D(const LinearMemory2D* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstMemory != nullptr) && (srcArray != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcArray(memCpy2D, srcArray);
	Internals::SetCudaMemcpy2DDstLinearMemory2D(memCpy2D, dstMemory);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}

void RTLib::Backends::Cuda::Stream::Copy2DFromLinearMemory2DToArray(const Array* dstArray, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept {
	assert(IsValid() && (dstArray != nullptr) && (srcMemory != nullptr));
	CUDA_MEMCPY2D memCpy2D;
	Internals::SetCudaMemcpy2DMemoryCopy2D(memCpy2D, copy);
	Internals::SetCudaMemcpy2DSrcLinearMemory2D(memCpy2D, srcMemory);
	Internals::SetCudaMemcpy2DDstArray(memCpy2D, dstArray);
	RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMemcpy2DAsync(&memCpy2D, Internals::GetCUstream(this)));
}



void RTLib::Backends::Cuda::Stream::Synchronize() noexcept
{
    assert(IsValid());
    auto stream = static_cast<CUstream>(m_Impl->m_Stream.lock().get());
    RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(
        cuStreamSynchronize(stream)
    );
}

#ifndef RTLIB_BACKENDS_CUDA_CUDA_STREAM_H
#define RTLIB_BACKENDS_CUDA_CUDA_STREAM_H
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <memory>
namespace RTLib {
    namespace Backends {
        namespace Cuda {
            class LinearMemory;
            class LinearMemory2D;
            class Array;
            class Stream 
            {
                friend class CurrentContext;
                friend class Context;

                Stream(std::shared_ptr<void> stream) noexcept;
            public:
                Stream(Stream&&)noexcept = delete;
                Stream(const Stream&) = delete;
                Stream& operator=(Stream&&)noexcept = delete;
                Stream& operator=(const Stream&) = delete;
                ~Stream() noexcept;

                bool IsValid()const noexcept;
                auto GetHandle()const noexcept -> void*;


                void Copy1DLinearMemory(const LinearMemory* dstMemory, const LinearMemory* srcMemory, const Memory1DCopy& copy)const noexcept;
                void Copy1DFromLinearMemoryToHost(void* dstMemory, const LinearMemory* srcMemory, const Memory1DCopy& copy)const noexcept;
                void Copy1DFromHostToLinearMemory(const LinearMemory* dstMemory, const void* srcMemory, const Memory1DCopy& copy)const noexcept;

                void Copy2DLinearMemory(const LinearMemory* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemoryToHost(void* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromHostToLinearMemory(const LinearMemory* dstMemory, const void* srcMemory, const Memory2DCopy& copy)const noexcept;

                void Copy2DLinearMemory2D(const LinearMemory2D* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemory2DToLinearMemory(const LinearMemory* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemoryToLinearMemory2D(const LinearMemory2D* dstMemory, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemory2DToHost(void* dstMemory, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromHostToLinearMemory2D(const LinearMemory2D* dstMemory, const void* srcMemory, const Memory2DCopy& copy)const noexcept;

                void Copy2DArray(const Array* dstArray, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromArrayToHost(void* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromHostToArray(const Array* dstArray, const void* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromArrayToLinearMemory(const LinearMemory* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemoryToArray(const Array* dstArray, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromArrayToLinearMemory2D(const LinearMemory2D* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemory2DToArray(const Array* dstArray, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept;

                void Synchronize() noexcept;
            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
        }
    }
}
#endif
#ifndef RTLIB_BACKENDS_CUDA_CUDA_CONTEXT_H
#define RTLIB_BACKENDS_CUDA_CUDA_CONTEXT_H
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <memory>
#include <unordered_map>
namespace RTLib
{
    namespace Backends
    {
        namespace Cuda
        {

            class Entry;
            class Device;
            class Context
            {
                friend class CurrentContext;
                friend class Stream;

            private:
                friend class RTLib::Backends::Cuda::Entry;

            public:
                Context(const Device &device, ContextCreateFlags flags = ContextCreateDefault) noexcept;
                ~Context() noexcept;

                Context(Context &&) noexcept = delete;
                Context(const Context &) = delete;
                Context &operator=(Context &&) noexcept = delete;
                Context &operator=(const Context &) = delete;

                bool operator==(const Context &ctx) const noexcept;
                bool operator!=(const Context &ctx) const noexcept;

                static auto GetCurrent() noexcept -> Context *;
                void SetCurrent() noexcept;
                void PopFromStack();

                auto GetHandle() const noexcept -> void *;
                bool IsBindedStack() const noexcept;

            private:
                auto GetDefaultStream() const noexcept -> Stream *;

            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };

            class Stream;
            class LinearMemory;
            class LinearMemory1D;
            class LinearMemory2D;
            class PinnedHostMemory;
            class Array;
            class Array1D;
            class Array2D;
            class Array3D;
            class MipmappedArray;
            class MipmappedArray1D;
            class MipmappedArray2D;
            class MipmappedArray3D;
            class Texture;
            class Module;
            class Function;
            class CurrentContext
            {
                friend class Context;
                CurrentContext() noexcept;

            public:
                CurrentContext(CurrentContext&&) noexcept = delete;
                CurrentContext(const CurrentContext&) = delete;
                CurrentContext& operator=(CurrentContext&&) noexcept = delete;
                CurrentContext& operator=(const CurrentContext&) = delete;

                static auto Handle() noexcept -> CurrentContext&;
                ~CurrentContext() noexcept;

                void Set(Context* ctx) noexcept
                {
                    return Set(ctx, false);
                }
                auto Get() const noexcept -> Context*;
                auto Pop() noexcept -> Context*
                {
                    return Pop(false);
                }
                void Push(Context* ctx) noexcept;
                void PopFromStack(Context* ctx) noexcept
                {
                    PopFromStack(ctx, false);
                }

                auto CreateStream(StreamCreateFlags flags = StreamCreateDefault) const noexcept -> Stream*;
                auto CreateLinearMemory1D(size_t sizeInBytes) const noexcept -> LinearMemory1D*;
                auto CreateLinearMemory2D(size_t width, size_t height, unsigned int elementSizeInBytes) const noexcept -> LinearMemory2D*;
                auto CreatePinnedHostMemory(size_t sizeInBytes) const noexcept -> PinnedHostMemory*;
                auto CreateArray1D(unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> Array1D*;
                auto CreateArray2D(unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> Array2D*;
                auto CreateArray3D(unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> Array3D*;
                auto CreateMipmappedArray1D(unsigned int levels,unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> MipmappedArray1D*;
                auto CreateMipmappedArray2D(unsigned int levels, unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> MipmappedArray2D*;
                auto CreateMipmappedArray3D(unsigned int levels, unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> MipmappedArray3D*;
                auto CreateTextureFromArray(const Array* arr, const TextureDesc& desc)const noexcept -> Texture*; 
                auto CreateTextureFromMipmappedArray(const MipmappedArray* mipmapped, const TextureDesc& desc)const noexcept -> Texture*;
                auto CreateModuleFromFile(const char* filename)const noexcept -> Module*;
                auto CreateModuleFromData(const std::vector<char>& image)const noexcept -> Module*;
                auto CreateModuleFromData2(const std::vector<char>& image, const std::unordered_map<JitOption,void*>& options)const noexcept -> Module*;

                auto CreateStreamUnique(StreamCreateFlags flags = StreamCreateDefault) const noexcept -> std::unique_ptr<Stream>;
                auto CreateLinearMemory1DUnique(size_t sizeInBytes) const noexcept -> std::unique_ptr<LinearMemory1D>;
                auto CreateLinearMemory2DUnique(size_t width, size_t height, unsigned int elementSizeInBytes) const noexcept -> std::unique_ptr<LinearMemory2D>;
                auto CreatePinnedHostMemoryUnique(size_t sizeInBytes) const noexcept -> std::unique_ptr<PinnedHostMemory>;
                auto CreateArray1DUnique(unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> std::unique_ptr<Array1D>;
                auto CreateArray2DUnique(unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> std::unique_ptr<Array2D>;
                auto CreateArray3DUnique(unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> std::unique_ptr<Array3D>;
                auto CreateMipmappedArray1DUnique(unsigned int levels, unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> std::unique_ptr<MipmappedArray1D>;
                auto CreateMipmappedArray2DUnique(unsigned int levels, unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> std::unique_ptr<MipmappedArray2D>;
                auto CreateMipmappedArray3DUnique(unsigned int levels, unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface = false) const noexcept -> std::unique_ptr<MipmappedArray3D>;
                auto CreateTextureUniqueFromArray(const Array* arr, const TextureDesc& desc)const noexcept -> std::unique_ptr<Texture>;
                auto CreateTextureUniqueFromMipmappedArray(const MipmappedArray* mipmapped, const TextureDesc& desc)const noexcept -> std::unique_ptr<Texture>;
                auto CreateModuleUniqueFromFile(const char* filename)const noexcept -> std::unique_ptr<Module>;
                auto CreateModuleUniqueFromData(const std::vector<char>& image)const noexcept -> std::unique_ptr<Module>;
                auto CreateModuleUniqueFromData2(const std::vector<char>& image, const std::unordered_map<JitOption, void*>& options)const noexcept -> std::unique_ptr<Module>;

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
                void Copy2DFromHostToLinearMemory2D(const LinearMemory2D* dstMemory,const void* srcMemory, const Memory2DCopy& copy)const noexcept;

                void Copy2DArray(const Array* dstArray, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromArrayToHost(void* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromHostToArray(const Array* dstArray, const void* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromArrayToLinearMemory(const LinearMemory* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemoryToArray(const Array* dstArray, const LinearMemory* srcMemory, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromArrayToLinearMemory2D(const LinearMemory2D* dstMemory, const Array* srcArray, const Memory2DCopy& copy)const noexcept;
                void Copy2DFromLinearMemory2DToArray(const Array* dstArray, const LinearMemory2D* srcMemory, const Memory2DCopy& copy)const noexcept;

                void LaunchKernel(const Function* function, const KernelLaunchDesc& desc);

                auto GetDefaultStream() const noexcept -> Stream *;

                void Synchronize()noexcept;
                void SynchronizeDefaultStream() noexcept;
            private:
                void Set(Context *ctx, bool sysOp) noexcept;
                auto Pop(bool sysOp) noexcept -> Context *;
                void PopFromStack(Context *ctx, bool sysOp);

            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
        }
    }
}
#endif

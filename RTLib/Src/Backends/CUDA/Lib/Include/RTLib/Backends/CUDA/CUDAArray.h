#ifndef RTLIB_BACKENDS_CUDA_ARRAY_H
#define RTLIB_BACKENDS_CUDA_ARRAY_H
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <vector>
#include <memory>
#include <any>
namespace RTLib
{
    namespace Backends
    {
        namespace Cuda
        {
            class Array
            {
            public:
                Array() noexcept {}
                virtual ~Array() noexcept {}

                virtual auto GetHandle() const noexcept -> void * = 0;
                virtual auto GetWidth() const noexcept -> unsigned int = 0;
                virtual auto GetHeight() const noexcept -> unsigned int = 0;
                virtual auto GetDepth() const noexcept -> unsigned int = 0;
                virtual auto GetLayers() const noexcept -> unsigned int = 0;
                virtual auto GetFormat() const noexcept -> ArrayFormat = 0;
                virtual auto GetChannels() const noexcept -> unsigned int = 0;
                virtual auto GetDimensionType() const noexcept -> DimensionType = 0;
            };
            class Array1D : public Array
            {
                friend class MipmappedArray1D;
                friend class CurrentContext;
                Array1D(unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
                Array1D(void* pHandle, unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
            public:
                virtual ~Array1D() noexcept;

                Array1D(Array1D &&) noexcept = delete;
                Array1D(const Array1D &) = delete;
                Array1D &operator=(Array1D &&) noexcept = delete;
                Array1D &operator=(const Array1D &) = delete;

                virtual auto GetHandle() const noexcept -> void * override;
                virtual auto GetWidth() const noexcept -> unsigned int override;
                virtual auto GetHeight() const noexcept -> unsigned int override;
                virtual auto GetDepth() const noexcept -> unsigned int override;
                virtual auto GetLayers() const noexcept -> unsigned int override;
                virtual auto GetFormat() const noexcept -> ArrayFormat override;
                virtual auto GetChannels() const noexcept -> unsigned int override;
                virtual auto GetDimensionType() const noexcept -> DimensionType override;

            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
            class Array2D : public Array
            {
                friend class MipmappedArray2D;
                friend class CurrentContext;
                Array2D(unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
                Array2D(void* pHandle, unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
            public:
                virtual ~Array2D() noexcept;

                Array2D(Array2D &&) noexcept = delete;
                Array2D(const Array2D &) = delete;
                Array2D &operator=(Array2D &&) noexcept = delete;
                Array2D &operator=(const Array2D &) = delete;

                virtual auto GetHandle() const noexcept -> void * override;
                virtual auto GetWidth() const noexcept -> unsigned int override;
                virtual auto GetHeight() const noexcept -> unsigned int override;
                virtual auto GetDepth() const noexcept -> unsigned int override;
                virtual auto GetLayers() const noexcept -> unsigned int override;
                virtual auto GetFormat() const noexcept -> ArrayFormat override;
                virtual auto GetChannels() const noexcept -> unsigned int override;
                virtual auto GetDimensionType() const noexcept -> DimensionType override;

            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
            class Array3D : public Array
            {
                friend class MipmappedArray3D;
                friend class CurrentContext;
                Array3D(unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
                Array3D(void* pHandle, unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
            public:
                virtual ~Array3D() noexcept;

                Array3D(Array3D &&) noexcept = delete;
                Array3D(const Array3D &) = delete;
                Array3D &operator=(Array3D &&) noexcept = delete;
                Array3D &operator=(const Array3D &) = delete;

                virtual auto GetHandle() const noexcept -> void * override;
                virtual auto GetWidth() const noexcept -> unsigned int override;
                virtual auto GetHeight() const noexcept -> unsigned int override;
                virtual auto GetDepth() const noexcept -> unsigned int override;
                virtual auto GetLayers() const noexcept -> unsigned int override;
                virtual auto GetFormat() const noexcept -> ArrayFormat override;
                virtual auto GetChannels() const noexcept -> unsigned int override;
                virtual auto GetDimensionType() const noexcept -> DimensionType override;

            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
            class ArrayCubemap : public Array
            {
            };
            class LayeredArray1D : public Array
            {
            };
            class LayeredArray2D : public Array
            {
            };
            class LayeredArrayCubemap : public Array
            {
            };
        }
    }
}
#endif

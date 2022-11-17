#ifndef RTLIB_BACKENDS_CUDA_MIPMAPPED_ARRAY_H
#define RTLIB_BACKENDS_CUDA_MIPMAPPED_ARRAY_H
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <vector>
#include <memory>
#include <any>
namespace RTLib
{
	namespace Backends {
		namespace Cuda {
			class Array;
			class MipmappedArray {
			public:
				MipmappedArray() noexcept {}
				virtual ~MipmappedArray() noexcept {}

				virtual auto GetHandle() const noexcept -> void* = 0;
				virtual auto GetWidth() const noexcept -> unsigned int = 0;
				virtual auto GetHeight() const noexcept -> unsigned int = 0;
				virtual auto GetDepth() const noexcept -> unsigned int = 0;
				virtual auto GetLayers() const noexcept -> unsigned int = 0;
				virtual auto GetLevels() const noexcept -> unsigned int = 0;
				virtual auto GetFormat() const noexcept -> ArrayFormat = 0;
				virtual auto GetChannels() const noexcept -> unsigned int = 0;
				virtual auto GetDimensionType() const noexcept -> DimensionType = 0;
				virtual auto GetMipArray(unsigned int level)const noexcept -> Array* = 0;
			};
            class MipmappedArray1D : public MipmappedArray
            {

                friend class CurrentContext;
                MipmappedArray1D(unsigned int levels, unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
            public:
                virtual ~MipmappedArray1D() noexcept;

                MipmappedArray1D(MipmappedArray1D&&) noexcept = delete;
                MipmappedArray1D(const MipmappedArray1D&) = delete;
                MipmappedArray1D& operator=(MipmappedArray1D&&) noexcept = delete;
                MipmappedArray1D& operator=(const MipmappedArray1D&) = delete;

                virtual auto GetHandle() const noexcept -> void* override;
                virtual auto GetWidth() const noexcept -> unsigned int override;
                virtual auto GetHeight() const noexcept -> unsigned int override;
                virtual auto GetDepth() const noexcept -> unsigned int override;
                virtual auto GetLayers() const noexcept -> unsigned int override;
                virtual auto GetLevels() const noexcept -> unsigned int override;
                virtual auto GetFormat() const noexcept -> ArrayFormat override;
                virtual auto GetChannels() const noexcept -> unsigned int override;
                virtual auto GetDimensionType() const noexcept -> DimensionType override;
                virtual auto GetMipArray(unsigned int level)const noexcept -> Array* override;

            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
            class MipmappedArray2D : public MipmappedArray
            {

                friend class CurrentContext;
                MipmappedArray2D(unsigned int levels, unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
            public:
                virtual ~MipmappedArray2D() noexcept;

                MipmappedArray2D(MipmappedArray2D&&) noexcept = delete;
                MipmappedArray2D(const MipmappedArray2D&) = delete;
                MipmappedArray2D& operator=(MipmappedArray2D&&) noexcept = delete;
                MipmappedArray2D& operator=(const MipmappedArray2D&) = delete;

                virtual auto GetHandle() const noexcept -> void* override;
                virtual auto GetWidth() const noexcept -> unsigned int override;
                virtual auto GetHeight() const noexcept -> unsigned int override;
                virtual auto GetDepth() const noexcept -> unsigned int override;
                virtual auto GetLayers() const noexcept -> unsigned int override;
                virtual auto GetLevels() const noexcept -> unsigned int override;
                virtual auto GetFormat() const noexcept -> ArrayFormat override;
                virtual auto GetChannels() const noexcept -> unsigned int override;
                virtual auto GetDimensionType() const noexcept -> DimensionType override;
                virtual auto GetMipArray(unsigned int level)const noexcept -> Array* override;


            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };
            class MipmappedArray3D : public MipmappedArray
            {
                friend class CurrentContext;
                MipmappedArray3D(unsigned int levels, unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept;
            public:
                virtual ~MipmappedArray3D() noexcept;

                MipmappedArray3D(MipmappedArray3D&&) noexcept = delete;
                MipmappedArray3D(const MipmappedArray3D&) = delete;
                MipmappedArray3D& operator=(MipmappedArray3D&&) noexcept = delete;
                MipmappedArray3D& operator=(const MipmappedArray3D&) = delete;

                virtual auto GetHandle() const noexcept -> void* override;
                virtual auto GetWidth() const noexcept -> unsigned int override;
                virtual auto GetHeight() const noexcept -> unsigned int override;
                virtual auto GetDepth() const noexcept -> unsigned int override;
                virtual auto GetLayers() const noexcept -> unsigned int override;
                virtual auto GetLevels() const noexcept -> unsigned int override;
                virtual auto GetFormat() const noexcept -> ArrayFormat override;
                virtual auto GetChannels() const noexcept -> unsigned int override;
                virtual auto GetDimensionType() const noexcept -> DimensionType override;
                virtual auto GetMipArray(unsigned int level)const noexcept -> Array* override;
            private:
                struct Impl;
                std::unique_ptr<Impl> m_Impl;
            };

            class MipmappedArrayCubemap : public MipmappedArray
            {
            };
            class MipmappedLayeredMipmappedArray1D : public MipmappedArray
            {
            };
            class MipmappedLayeredMipmappedArray2D : public MipmappedArray
            {
            };
            class MipmappedLayeredArrayCubemap : public MipmappedArray
            {
            };
		}
	}
}
#endif

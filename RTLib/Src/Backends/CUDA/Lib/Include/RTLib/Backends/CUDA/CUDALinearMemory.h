#ifndef RTLIB_BACKENDS_CUDA_LINEAR_MEMORY_H
#define RTLIB_BACKENDS_CUDA_LINEAR_MEMORY_H
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <vector>
#include <memory>
#include <any>
namespace RTLib
{
	namespace Backends {
		namespace Cuda {
			class LinearMemory
			{
			public:
				LinearMemory() noexcept {}
				virtual ~LinearMemory() noexcept {}

				virtual auto GetHandle()const noexcept -> void* = 0;
				virtual auto GetSizeInBytes()const noexcept -> size_t = 0;
				virtual auto GetDimensionType()const noexcept -> DimensionType = 0;
			};
			class LinearMemory1D: public LinearMemory
			{
				friend class CurrentContext;
				LinearMemory1D(size_t sizeInBytes) noexcept;
			public:
				virtual ~LinearMemory1D() noexcept;

				LinearMemory1D(LinearMemory1D&&)noexcept = delete;
				LinearMemory1D(const LinearMemory1D&) = delete;
				LinearMemory1D& operator=(LinearMemory1D&&)noexcept = delete;
				LinearMemory1D& operator=(const LinearMemory1D&) = delete;

				virtual auto GetHandle()const noexcept -> void* override;
				virtual auto GetSizeInBytes()const noexcept -> size_t override;
				virtual auto GetDimensionType()const noexcept -> DimensionType override { return DimensionType::e1D; }
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
			class LinearMemory2D: public LinearMemory
			{
				friend class CurrentContext;
				LinearMemory2D(size_t width, size_t height, unsigned int elementSizeInBytes) noexcept;
			public:
				virtual ~LinearMemory2D() noexcept;

				LinearMemory2D(LinearMemory2D&&)noexcept = delete;
				LinearMemory2D(const LinearMemory2D&) = delete;
				LinearMemory2D& operator=(LinearMemory2D&&)noexcept = delete;
				LinearMemory2D& operator=(const LinearMemory2D&) = delete;

				virtual auto GetHandle()const noexcept -> void* override;
				virtual auto GetSizeInBytes()const noexcept -> size_t override;
				virtual auto GetDimensionType()const noexcept -> DimensionType override { return DimensionType::e2D; }

				auto GetWidth ()const noexcept -> size_t;
				auto GetHeight()const noexcept -> size_t;
				auto GetElementSizeInBytes()const noexcept -> unsigned int;
				auto GetPitchSizeInBytes()const noexcept -> size_t;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

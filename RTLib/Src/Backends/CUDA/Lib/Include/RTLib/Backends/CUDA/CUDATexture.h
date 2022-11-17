#ifndef RTLIB_BACKENDS_CUDA_CUDA_TEXTURE_H
#define RTLIB_BACKENDS_CUDA_CUDA_TEXTURE_H
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
			class Array;
			class MipmappedArray;
			class Texture
			{
				friend class CurrentContext;
				Texture(const MipmappedArray* mipmappedArray, const TextureDesc& desc) noexcept;
				Texture(const Array* array, const TextureDesc& desc) noexcept;

			public:
				~Texture() noexcept;

				Texture(Texture&&) noexcept = delete;
				Texture(const Texture&) = delete;
				Texture& operator=(Texture&&) noexcept = delete;
				Texture& operator=(const Texture&) = delete;

				auto GetHandle() const noexcept -> void*;
				auto GetResourceType() const noexcept -> TextureResourceType;
				auto GetMipmappedArray() const noexcept -> const MipmappedArray*;
				auto GetArray() const noexcept ->const Array*;
				auto GetDesc()const noexcept -> const TextureDesc&;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

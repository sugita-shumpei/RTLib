#include <RTLib/Backends/CUDA/CUDAMipmappedArray.h>
#include <RTLib/Backends/CUDA/CUDAArray.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include "CUDAInternals.h"
#include <vector>
#include <memory>

struct RTLib::Backends::Cuda::MipmappedArray1D::Impl {
	Impl(unsigned int levels_, unsigned int count_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
		:mipmappedArray{}, mipArrays{},
		count{ std::max<unsigned int>(count_,1) },
		channels{ std::max<unsigned int>(numChannels_,1) },
		format{ format_ },
		levels{ levels_ },
		useSurface{ useSurface_ } {
		assert(CurrentContext::Handle().Get());
		CUDA_ARRAY3D_DESCRIPTOR desc = {};
		assert(CurrentContext::Handle().Get());
		desc.Width = count;
		desc.Height = 0;
		desc.Depth = 0;
		desc.NumChannels = channels;
		desc.Format = Internals::GetCUarray_format(format_);
		desc.Flags = 0;
		if (useSurface_) {
			desc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
		}
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayCreate(&mipmappedArray, &desc, levels));
		for (unsigned int i = 0; i < levels; ++i) {
			CUarray arr;
			CUDA_ARRAY3D_DESCRIPTOR desc;
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayGetLevel(&arr, mipmappedArray, i));
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArray3DGetDescriptor(&desc, arr));
			mipArrays.push_back(std::unique_ptr<Array1D>(new Array1D((void*)arr, desc.Width, channels, format, useSurface)));
		}
	}
	~Impl() noexcept {
		assert(CurrentContext::Handle().Get());
		mipArrays.clear();
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayDestroy(mipmappedArray));
	}
	CUmipmappedArray mipmappedArray;
	std::vector<std::unique_ptr<Array1D>> mipArrays;
	unsigned int count;
	ArrayFormat  format;
	unsigned int channels;
	unsigned int levels;
	bool useSurface;
};

RTLib::Backends::Cuda::MipmappedArray1D::MipmappedArray1D(unsigned int levels,unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept
	: MipmappedArray(),
	  m_Impl(new Impl(levels,count, numChannels,format,useSurface))
{
}

RTLib::Backends::Cuda::MipmappedArray1D::~MipmappedArray1D() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetHandle() const noexcept -> void* {
    return m_Impl->mipmappedArray;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetWidth () const noexcept -> unsigned int {
    return m_Impl->count;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetHeight() const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetDepth () const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetLayers() const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetLevels() const noexcept -> unsigned int {
	return m_Impl->levels;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetDimensionType()const noexcept -> RTLib::Backends::Cuda::DimensionType {
    return DimensionType::e1D;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetFormat() const noexcept -> ArrayFormat {
    return m_Impl->format;
}

auto RTLib::Backends::Cuda::MipmappedArray1D::GetChannels()const noexcept-> unsigned int {
    return m_Impl->channels;
}
auto RTLib::Backends::Cuda::MipmappedArray1D::GetMipArray(unsigned int level)const noexcept -> Array* {
	if (level >= GetLevels()) {
		return nullptr;
	}
	else {
		return m_Impl->mipArrays[level].get();
	}
}

struct RTLib::Backends::Cuda::MipmappedArray2D::Impl {
	Impl(unsigned int levels_, unsigned int width_, unsigned int height_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
		:mipmappedArray{},
		mipArrays{},
		width{ std::max<unsigned int>(width_,1) },
		height{ std::max<unsigned int>(height_,1) },
		channels{ std::max<unsigned int>(numChannels_,1) },
		format{ format_ },
		levels{levels_},
		useSurface{ useSurface_ } {
		assert(CurrentContext::Handle().Get());
		CUDA_ARRAY3D_DESCRIPTOR desc = {};
		assert(CurrentContext::Handle().Get());
		desc.Width = width;
		desc.Height = height;
		desc.Depth = 0;
		desc.NumChannels = channels;
		desc.Format = Internals::GetCUarray_format(format_);
		desc.Flags = 0;
		if (useSurface_) {
			desc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
		}
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayCreate(&mipmappedArray, &desc, levels));
		for (unsigned int i = 0; i < levels; ++i) {
			CUarray arr;
			CUDA_ARRAY3D_DESCRIPTOR desc;
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayGetLevel(&arr, mipmappedArray, i));
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArray3DGetDescriptor(&desc, arr));
			mipArrays.push_back(std::unique_ptr<Array2D>(new Array2D((void*)arr, desc.Width, desc.Height, channels, format, useSurface)));
		}
	}
	~Impl() noexcept {
		assert(CurrentContext::Handle().Get());
		mipArrays.clear();
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayDestroy(mipmappedArray));
	}
	CUmipmappedArray mipmappedArray;
	std::vector<std::unique_ptr<Array2D>> mipArrays;
	unsigned int width;
	unsigned int height;
	ArrayFormat format;
	unsigned int channels;
	unsigned int levels;
	bool useSurface;
};
RTLib::Backends::Cuda::MipmappedArray2D::MipmappedArray2D(unsigned int levels,unsigned int width, unsigned int height, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept
	: MipmappedArray(),
	  m_Impl(new Impl(levels,width,height,numChannels,format,useSurface))
{
	
}

RTLib::Backends::Cuda::MipmappedArray2D::~MipmappedArray2D() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetHandle() const noexcept -> void* {
	return m_Impl->mipmappedArray;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetWidth() const noexcept -> unsigned int {
	return m_Impl->width;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetHeight() const noexcept -> unsigned int {
	return m_Impl->height;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetDepth() const noexcept -> unsigned int {
	return 1;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetLayers() const noexcept -> unsigned int {
	return 1;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetLevels() const noexcept -> unsigned int {
	return m_Impl->levels;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetDimensionType()const noexcept -> RTLib::Backends::Cuda::DimensionType {
	return DimensionType::e2D;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetFormat() const noexcept -> ArrayFormat {
	return m_Impl->format;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetChannels()const noexcept-> unsigned int {
	return m_Impl->channels;
}

auto RTLib::Backends::Cuda::MipmappedArray2D::GetMipArray(unsigned int level)const noexcept -> Array* {
	if (level >= GetLevels()) {
		return nullptr;
	}
	else {
		return m_Impl->mipArrays[level].get();
	}
}

struct RTLib::Backends::Cuda::MipmappedArray3D::Impl {
	Impl(unsigned int levels_,unsigned int width_, unsigned int height_, unsigned int depth_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
		:mipmappedArray{}, mipArrays{},
		width{ std::max<unsigned int>(width_,1) },
		height{ std::max<unsigned int>(height_,1) },
		depth{ std::max<unsigned int>(depth_,1) },
		channels{ std::max<unsigned int>(numChannels_,1) },
		format{ format_ },
		levels{levels_},
		useSurface{ useSurface_ } {
		assert(CurrentContext::Handle().Get());
		CUDA_ARRAY3D_DESCRIPTOR desc = {};
		assert(CurrentContext::Handle().Get());
		desc.Width = width;
		desc.Height = height;
		desc.Depth = depth;
		desc.NumChannels = channels;
		desc.Format = Internals::GetCUarray_format(format_);
		desc.Flags = 0;
		if (useSurface_) {
			desc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
		}
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayCreate(&mipmappedArray, &desc,levels));
		for (unsigned int i = 0; i < levels; ++i) {
			CUarray arr;
			CUDA_ARRAY3D_DESCRIPTOR desc;
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayGetLevel(&arr, mipmappedArray, i));
			RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArray3DGetDescriptor(&desc, arr));
			mipArrays.push_back(std::unique_ptr<Array3D>(new Array3D((void*)arr, desc.Width, desc.Height, desc.Depth, channels, format, useSurface)));
		}
	}
	~Impl() noexcept {
		assert(CurrentContext::Handle().Get());
		mipArrays.clear();
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuMipmappedArrayDestroy(mipmappedArray));
	}
	CUmipmappedArray mipmappedArray;
	std::vector<std::unique_ptr<Array3D>> mipArrays;
	unsigned int width;
	unsigned int height;
	unsigned int depth;
	ArrayFormat format;
	unsigned int channels;
	unsigned int levels;
	bool useSurface;
};
RTLib::Backends::Cuda::MipmappedArray3D::MipmappedArray3D(unsigned int levels,unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept
	: MipmappedArray(),
	  m_Impl(new Impl(levels,width,height,depth,numChannels,format,useSurface))
{
	
}

RTLib::Backends::Cuda::MipmappedArray3D::~MipmappedArray3D() noexcept
{
	m_Impl.reset();
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetHandle() const noexcept -> void* {
	return m_Impl->mipmappedArray;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetWidth() const noexcept -> unsigned int {
	return m_Impl->width;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetHeight() const noexcept -> unsigned int {
	return m_Impl->height;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetDepth() const noexcept -> unsigned int {
	return m_Impl->depth;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetLayers() const noexcept -> unsigned int {
	return 1;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetLevels() const noexcept -> unsigned int {
	return m_Impl->levels;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetDimensionType()const noexcept -> RTLib::Backends::Cuda::DimensionType {
	return DimensionType::e3D;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetFormat() const noexcept -> ArrayFormat {
	return m_Impl->format;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetChannels()const noexcept-> unsigned int {
	return m_Impl->channels;
}

auto RTLib::Backends::Cuda::MipmappedArray3D::GetMipArray(unsigned int level)const noexcept -> Array* {
	if (level >= GetLevels()) {
		return nullptr;
	}
	else {
		return m_Impl->mipArrays[level].get();
	}
}
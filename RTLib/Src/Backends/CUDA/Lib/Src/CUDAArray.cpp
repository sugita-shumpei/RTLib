#include <RTLib/Backends/CUDA/CUDAArray.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include "CUDAInternals.h"
struct RTLib::Backends::Cuda::Array1D::Impl {
    Impl(void* pHandle, unsigned int count_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
        :arr{ static_cast<CUarray>(pHandle) },
        count{ std::max<unsigned int>(count_,1) },
        channels{ std::max<unsigned int>(numChannels_,1) },
        format{ format_ },
        useSurface{ useSurface_ },
        hasOwnership{ false } {}
    Impl(unsigned int count_, unsigned int numChannels_, ArrayFormat format_,bool useSurface_) noexcept
        :arr{}, 
        count{ std::max<unsigned int>(count_,1) },
        channels{ std::max<unsigned int>(numChannels_,1) },
        format{ format_ }, 
        useSurface{ useSurface_ } ,
        hasOwnership{true} {
        assert(CurrentContext::Handle().Get());
        CUDA_ARRAY3D_DESCRIPTOR desc = {};
		assert(CurrentContext::Handle().Get());
        desc.Width  = count;
        desc.Height = 0;
        desc.Depth = 0;
        desc.NumChannels = channels;
        desc.Format = Internals::GetCUarray_format(format_);
        desc.Flags = 0;
        if (useSurface_) {
            desc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
        }
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArray3DCreate(&arr,&desc));
    }
    ~Impl() noexcept {
        if (hasOwnership) {
            assert(CurrentContext::Handle().Get());
            RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArrayDestroy(arr));
        }
    }
    CUarray      arr;
    unsigned int count;
    ArrayFormat  format;
    unsigned int channels;
    bool useSurface;
    bool hasOwnership;
};

RTLib::Backends::Cuda::Array1D::Array1D(unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface ) noexcept
    : Array(), m_Impl{new Impl(count,numChannels,format,useSurface)}{}
RTLib::Backends::Cuda::Array1D::Array1D(void* pHandle, unsigned int count, unsigned int numChannels, ArrayFormat format, bool useSurface) noexcept
    : Array(), m_Impl{ new Impl(pHandle,count,numChannels,format,useSurface) } {}

RTLib::Backends::Cuda::Array1D::~Array1D() noexcept
{
    m_Impl.reset();
}

auto RTLib::Backends::Cuda::Array1D::GetHandle() const noexcept -> void* {
    return m_Impl->arr;
}

auto RTLib::Backends::Cuda::Array1D::GetWidth () const noexcept -> unsigned int {
    return m_Impl->count;
}

auto RTLib::Backends::Cuda::Array1D::GetHeight() const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::Array1D::GetDepth () const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::Array1D::GetLayers() const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::Array1D::GetDimensionType()const noexcept -> RTLib::Backends::Cuda::DimensionType {
    return DimensionType::e1D;
}

auto RTLib::Backends::Cuda::Array1D::GetFormat() const noexcept -> ArrayFormat {
    return m_Impl->format;
}

auto RTLib::Backends::Cuda::Array1D::GetChannels()const noexcept-> unsigned int {
    return m_Impl->channels;
}

struct RTLib::Backends::Cuda::Array2D::Impl {
    Impl(unsigned int width_, unsigned int height_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
        :arr{}, 
        width{ std::max<unsigned int>(width_,1)   }, 
        height{ std::max<unsigned int>(height_,1) },
        channels{ std::max<unsigned int>(numChannels_,1) },
        format{ format_ }, 
        useSurface{ useSurface_ },
        hasOwnership{ true } {
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
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArray3DCreate(&arr, &desc));
    }
    Impl(void* pHandle, unsigned int width_, unsigned int height_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
        :arr{static_cast<CUarray>(pHandle)},
        width{ std::max<unsigned int>(width_,1) },
        height{ std::max<unsigned int>(height_,1) },
        channels{ std::max<unsigned int>(numChannels_,1) },
        format{ format_ },
        useSurface{ useSurface_ },
        hasOwnership{ false } {
    }

    ~Impl() noexcept {
        if (hasOwnership) {
            assert(CurrentContext::Handle().Get());
            RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArrayDestroy(arr));
        }
    }
    CUarray     arr;
    unsigned int width;
    unsigned int height;
    ArrayFormat format;
    unsigned int channels;
    bool useSurface;
    bool hasOwnership;
}; 

RTLib::Backends::Cuda::Array2D::Array2D(unsigned int width, unsigned int height, unsigned int numChannels, RTLib::Backends::Cuda::ArrayFormat format, bool useSurface) noexcept
    : Array(), m_Impl{ new Impl(width,height,numChannels,format,useSurface) }
{}

RTLib::Backends::Cuda::Array2D::Array2D(void* pHandle, unsigned int width, unsigned int height, unsigned int numChannels, RTLib::Backends::Cuda::ArrayFormat format, bool useSurface) noexcept
    : Array(),m_Impl{new Impl(pHandle, width,height,numChannels,format,useSurface)}
{}

RTLib::Backends::Cuda::Array2D::~Array2D() noexcept
{
    m_Impl.reset();
}

auto RTLib::Backends::Cuda::Array2D::GetHandle() const noexcept -> void* {
    return m_Impl->arr;
}

auto RTLib::Backends::Cuda::Array2D::GetWidth () const noexcept -> unsigned int {
    return m_Impl->width;
}

auto RTLib::Backends::Cuda::Array2D::GetHeight() const noexcept -> unsigned int {
    return m_Impl->height;
}

auto RTLib::Backends::Cuda::Array2D::GetDepth () const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::Array2D::GetLayers() const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::Array2D::GetDimensionType()const noexcept -> RTLib::Backends::Cuda::DimensionType {
    return DimensionType::e2D;
}

auto RTLib::Backends::Cuda::Array2D::GetFormat() const noexcept -> ArrayFormat {
    return m_Impl->format;
}

auto RTLib::Backends::Cuda::Array2D::GetChannels()const noexcept-> unsigned int {
    return m_Impl->channels;
}

struct RTLib::Backends::Cuda::Array3D::Impl {
    Impl(unsigned int width_, unsigned int height_, unsigned int depth_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
        :arr{},
        width{ std::max<unsigned int>(width_,1) },
        height{ std::max<unsigned int>(height_,1) },
        depth{ std::max<unsigned int>(depth_,1) },
        channels{ std::max<unsigned int>(numChannels_,1) },
        format{ format_ },
        useSurface{ useSurface_ },
        hasOwnership{ true } {
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
        RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArray3DCreate(&arr, &desc));
    }
    Impl(void* pHandle, unsigned int width_, unsigned int height_, unsigned int depth_, unsigned int numChannels_, ArrayFormat format_, bool useSurface_) noexcept
        :arr{static_cast<CUarray>(pHandle)},
        width{ std::max<unsigned int>(width_,1) },
        height{ std::max<unsigned int>(height_,1) },
        depth{ std::max<unsigned int>(depth_,1) },
        channels{ std::max<unsigned int>(numChannels_,1) },
        format{ format_ },
        useSurface{ useSurface_ } ,
        hasOwnership{false} {
    }
    ~Impl() noexcept {
        if (hasOwnership) {
            assert(CurrentContext::Handle().Get());
            RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuArrayDestroy(arr));
        }
    }
    CUarray     arr;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    ArrayFormat format;
    unsigned int channels;
    bool useSurface;
    bool hasOwnership;
};

RTLib::Backends::Cuda::Array3D::Array3D(unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, RTLib::Backends::Cuda::ArrayFormat format, bool useSurface) noexcept
    : Array(), m_Impl{ new Impl(width,height, depth, numChannels,format,useSurface) }
{
    
}

RTLib::Backends::Cuda::Array3D::Array3D(void* pHandle,unsigned int width, unsigned int height, unsigned int depth, unsigned int numChannels, RTLib::Backends::Cuda::ArrayFormat format, bool useSurface) noexcept
    : Array(), m_Impl{ new Impl(pHandle,width,height, depth, numChannels,format,useSurface) }
{

}

RTLib::Backends::Cuda::Array3D::~Array3D() noexcept
{
    m_Impl.reset();
}

auto RTLib::Backends::Cuda::Array3D::GetHandle() const noexcept -> void* {
    return m_Impl->arr;
    
}

auto RTLib::Backends::Cuda::Array3D::GetWidth () const noexcept -> unsigned int {
    return m_Impl->width;
}

auto RTLib::Backends::Cuda::Array3D::GetHeight() const noexcept -> unsigned int {
    return m_Impl->height;
}

auto RTLib::Backends::Cuda::Array3D::GetDepth () const noexcept -> unsigned int {
    return m_Impl->depth;
}

auto RTLib::Backends::Cuda::Array3D::GetLayers() const noexcept -> unsigned int {
    return 1;
}

auto RTLib::Backends::Cuda::Array3D::GetDimensionType()const noexcept -> RTLib::Backends::Cuda::DimensionType {
    return DimensionType::e3D;
}

auto RTLib::Backends::Cuda::Array3D::GetFormat() const noexcept -> ArrayFormat {
    return m_Impl->format;
}

auto RTLib::Backends::Cuda::Array3D::GetChannels()const noexcept-> unsigned int {
    return m_Impl->channels;
}

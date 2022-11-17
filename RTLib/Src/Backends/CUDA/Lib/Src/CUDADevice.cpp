#include <RTLib/Backends/CUDA/CUDADevice.h>
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include "CUDAInternals.h"

struct RTLib::Backends::Cuda::Device::Impl {
	Impl(int deviceIdx) :index{ deviceIdx } {
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuDeviceGet(&device, deviceIdx));
	}
	~Impl(){}
	CUdevice device;
	int      index;
};

RTLib::Backends::Cuda::Device::Device(int deviceIdx) noexcept : m_Impl{ new Impl(deviceIdx) } {}

RTLib::Backends::Cuda::Device::Device(Device&&) noexcept = default;

auto RTLib::Backends::Cuda::Device::operator=(Device&&) noexcept->Device & = default;

RTLib::Backends::Cuda::Device::~Device() noexcept = default;

bool RTLib::Backends::Cuda::Device::operator==(const Device& device) const noexcept
{
	return this->m_Impl->device==device.m_Impl->device;
}

bool RTLib::Backends::Cuda::Device::operator!=(const Device& device) const noexcept
{
	return this->m_Impl->device!=device.m_Impl->device;
}

auto RTLib::Backends::Cuda::Device::GetHandle() const noexcept -> const void*
{
	return &this->m_Impl->device;
}

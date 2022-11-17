#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <RTLib/Backends/CUDA/CUDADevice.h>
#include "CUDAInternals.h"
#include <vector>
struct RTLib::Backends::Cuda::Entry::Impl {
	RTLib::Backends::Cuda::Entry::Impl::Impl() noexcept :devices{} {
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuInit(0));
		int count;
		RTLIB_BACKENDS_CUDA_DEBUG_ASSERT(cuDeviceGetCount(&count));
		devices.reserve(count);
		for (int i = 0; i < count; ++i) {
			devices.emplace_back(Device(i));
		}
	}
	~Impl()noexcept {}
	std::vector<Device> devices;
};

RTLib::Backends::Cuda::Entry::Entry() noexcept :m_Impl{ new Impl() } {}

auto RTLib::Backends::Cuda::Entry::Handle() noexcept -> Entry&
{
	static Cuda::Entry entry;
	return entry;
}

RTLib::Backends::Cuda::Entry::~Entry() noexcept
{
}

auto RTLib::Backends::Cuda::Entry::EnumerateDevices() const noexcept ->const std::vector<Cuda::Device>&
{
	return m_Impl->devices;
}

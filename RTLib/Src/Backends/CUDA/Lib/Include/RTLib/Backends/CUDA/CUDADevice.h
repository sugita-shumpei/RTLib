#ifndef RTLIB_BACKENDS_CUDA_DEVICE_H
#define RTLIB_BACKENDS_CUDA_DEVICE_H
#include <vector>
#include <memory>
#include <any>
namespace RTLib
{
	namespace Backends {
		namespace Cuda {
			class Entry;
			class Device {
			private:
				friend class RTLib::Backends::Cuda::Entry;
				Device(int deviceIdx) noexcept;
				Device& operator=(Device&&)noexcept;
			public:
				Device(Device&&)noexcept;
				Device(const Device&) = delete;
				Device& operator=(const Device&) = delete;
				~Device()noexcept;

				bool operator==(const Device& device)const noexcept;
				bool operator!=(const Device& device)const noexcept;

				auto GetHandle()const noexcept -> const void*;
			private:
				struct Impl;
				std::unique_ptr<Impl> m_Impl;
			};
		}
	}
}
#endif

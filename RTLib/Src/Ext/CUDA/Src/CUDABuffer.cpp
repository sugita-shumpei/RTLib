#include <RTLib/Ext/CUDA/CUDABuffer.h>
#include <iostream>
#include <string>
auto RTLib::Ext::CUDA::CUDABuffer::Allocate(CUDAContext* ctx, const CUDABufferDesc& desc) -> CUDABuffer*
{
	if (desc.sizeInBytes == 0) { return nullptr; }
	CUdeviceptr deviceptr = 0;
	void* hostptr = nullptr;
	if (desc.flags == CUDAMemoryFlags::eDefault) {
		bool isSuccess = true;
		do {
			auto res = cuMemAlloc(&deviceptr, desc.sizeInBytes);
			if (res != CUDA_SUCCESS) {
				const char* errString = nullptr;
				(void)cuGetErrorString(res, &errString);
				std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
				isSuccess = false;
				break;
			}
		} while (0);
		if (!isSuccess) {
			if (deviceptr) {
				cuMemFree(deviceptr);
				deviceptr = 0;
			}
			return nullptr;
		}
	}
	if (desc.flags == CUDAMemoryFlags::ePageLocked) {
		bool isSuccess = true;
		do{
			{
				auto res = cuMemAlloc(&deviceptr, desc.sizeInBytes);
				if (res != CUDA_SUCCESS) {
					const char* errString = nullptr;
					(void)cuGetErrorString(res, &errString);
					std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
					isSuccess = false;
					break;
				}
			}
			{
				auto res = cuMemAllocHost((void**)&hostptr, desc.sizeInBytes);
				if (res != CUDA_SUCCESS) {
					const char* errString = nullptr;
					(void)cuGetErrorString(res, &errString);
					std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
					isSuccess = false;
					break;
				}
			}
		} while (0);
		if (!isSuccess) {
			if (deviceptr) {
				cuMemFree(deviceptr);
				deviceptr = 0;
			}
			if (hostptr) {
				cuMemFreeHost(hostptr);
				hostptr = nullptr;
			}
			return nullptr;
		}
	}
	auto buffer = new CUDABuffer(ctx, desc,deviceptr,hostptr);
	return buffer;
}

void RTLib::Ext::CUDA::CUDABuffer::Destroy() noexcept
{
	m_Context = nullptr;
	m_flags = CUDAMemoryFlags::eDefault;
	m_SizeInBytes = 0;
	if (m_Deviceptr) {
		auto res = cuMemFree(m_Deviceptr);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __FILE__ << ":" << std::string(errString) << "\n";
		}
		m_Deviceptr = 0;
	}
	if (m_Hostptr) {
		auto res = cuMemFreeHost(m_Hostptr);
		if (res != CUDA_SUCCESS) {
			const char* errString = nullptr;
			(void)cuGetErrorString(res, &errString);
			std::cout << __FILE__ << ":" << __LINE__ << ":" << std::string(errString) << "\n";
		}
		m_Hostptr = nullptr;
	}
}

RTLib::Ext::CUDA::CUDABuffer::~CUDABuffer() noexcept
{
	
}

RTLib::Ext::CUDA::CUDABuffer::CUDABuffer(CUDAContext* ctx, const CUDABufferDesc& desc, CUdeviceptr deviceptr, void* hostptr) noexcept
{
	m_Context = ctx;
	m_flags = desc.flags;
	m_SizeInBytes = desc.sizeInBytes;
	m_Deviceptr = deviceptr;
	m_Hostptr = hostptr;
}

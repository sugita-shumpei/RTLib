#ifndef TEST_TESTLIB_BUFFER_VIEW__H
#define TEST_TESTLIB_BUFFER_VIEW__H
#include <OptiXToolkit/Error/ErrorCheck.h>
#include <OptiXToolkit/Memory/SyncVector.h>
#include <assert.h>
#include <cuda.h>
namespace TestLib
{
	struct BufferView
	{
		BufferView(CUdeviceptr devicePtr_, size_t sizeInBytes_, size_t strideInBytes_) noexcept:
			devicePtr{ devicePtr_ }, 
			sizeInBytes{ ((sizeInBytes_+ strideInBytes_-1)/ strideInBytes_)* strideInBytes_ },
			strideInBytes{ strideInBytes_ }
		{}

		BufferView(const otk::DeviceBuffer& deviceBuffer, size_t strideInBytes_) noexcept :
			devicePtr{ static_cast<CUdeviceptr>(deviceBuffer) },
			sizeInBytes{ deviceBuffer.size() },
			strideInBytes{ strideInBytes_ } 
		{}

		~BufferView()noexcept
		{}

		BufferView(const BufferView&) = default;

		BufferView& operator=(const BufferView&) = default;

		template<typename T>
		auto at(size_t idx, size_t offset = 0) const noexcept -> T*
		{
			assert(sizeof(T) + offset <= strideInBytes);
			return reinterpret_cast<T*>(devicePtr + idx * strideInBytes + offset);
		}

		auto get_device_ptr() const noexcept -> CUdeviceptr { return devicePtr; }

		template<typename T = void>
		auto get_device_type_ptr() const noexcept ->T* { return reinterpret_cast<T*>(devicePtr); }

		auto get_count() const noexcept -> size_t { return sizeInBytes / strideInBytes; }
		
		auto get_stride_in_bytes() const noexcept -> size_t { return strideInBytes; }

		auto get_size_in_bytes() const noexcept -> size_t { return sizeInBytes; }

		auto get_sub_view(size_t idxCount, size_t idxBegin = 0) const noexcept -> BufferView
		{
			assert(idxBegin < get_count() && idxBegin + idxCount <= get_count());
			return BufferView(devicePtr + idxBegin * strideInBytes, idxCount * strideInBytes, strideInBytes);
		}
		
		void copy_to_device(const void* pData) {
			OTK_ERROR_CHECK(cuMemcpyHtoD(devicePtr, pData, sizeInBytes));
		}

		void copy_to_device_async(CUstream stream, const void* pData){
			OTK_ERROR_CHECK(cuMemcpyHtoDAsync(devicePtr, pData, sizeInBytes, stream));
		}

		void copy_from_device(void* pData) {
			OTK_ERROR_CHECK(cuMemcpyDtoH(pData, devicePtr, sizeInBytes));
		}

		void copy_from_device_async(CUstream stream, void* pData) {
			OTK_ERROR_CHECK(cuMemcpyDtoHAsync(pData, devicePtr, sizeInBytes, stream));
		}

		void clear_8u(unsigned char   v) {
			OTK_ERROR_CHECK(cuMemsetD8(devicePtr, v, sizeInBytes / sizeof(unsigned char)));
		}
		void clear_8u_async(CUstream stream, unsigned char   v) {
			OTK_ERROR_CHECK(cuMemsetD8Async(devicePtr, v, sizeInBytes / sizeof(unsigned char), stream));
		}

		void clear_16u(unsigned short v) {
			OTK_ERROR_CHECK(cuMemsetD16(devicePtr, v, sizeInBytes / sizeof(unsigned short)));
		}
		void clear_16u_async(CUstream stream, unsigned short v) {
			OTK_ERROR_CHECK(cuMemsetD16Async(devicePtr, v, sizeInBytes / sizeof(unsigned short), stream));
		}

		void clear_32u(unsigned int   v) {
			OTK_ERROR_CHECK(cuMemsetD32(devicePtr, v, sizeInBytes / sizeof(unsigned int)));
		}
		void clear_32u_async(CUstream stream, unsigned int  v) {
			OTK_ERROR_CHECK(cuMemsetD32Async(devicePtr, v, sizeInBytes / sizeof(unsigned int), stream));
		}

		void clear_32f(float          v) {
			unsigned int uv;
			std::memcpy(&uv, &v, sizeof(float));
			clear_32u(uv);
		}
		void clear_32f_async(CUstream stream, float         v) {
			unsigned int uv;
			std::memcpy(&uv, &v, sizeof(float));
			clear_32u_async(stream, uv);
		}

		CUdeviceptr devicePtr;
		size_t sizeInBytes;
		size_t strideInBytes;
	};
}
#endif

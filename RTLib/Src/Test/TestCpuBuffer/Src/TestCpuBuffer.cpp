#include <RTLib/Core/CpuBuffer.h>
#include <iostream>
int main(int argc, const char** argv) {
	auto factory = std::unique_ptr<RTLib::Core::IBuffer::Factory>(RTLib::Core::TCpuBuffer<int>::Factory::New());
	auto buffer  = std::unique_ptr<RTLib::Core::IBuffer>(factory->NewBuffer(sizeof(int) * 100));
	{
		std::cout << "  size: " << buffer->Size()   << std::endl;
		std::cout << "stride: " << buffer->Stride() << std::endl;
		{
			auto mappable    = std::unique_ptr<RTLib::Core::IBufferMappableView>(buffer->Mappable());
			int* mappedData  = nullptr;
			if (mappable->Map(RTLib::Core::BufferMapAccess::eWriteOnly,(void**)&mappedData, buffer->Stride() * 100, 0)) {
				for (int i = 0; i < buffer->Size()/buffer->Stride(); ++i) {
					mappedData[i] = i;
				}
				mappable->Unmap();
			}
		}
		{
			auto resizable   = std::unique_ptr<RTLib::Core::IBufferResizableView>(buffer->Resizable());
			if (resizable->Resize(sizeof(int) * 200)) {
				std::cout << "  size: " << buffer->Size() << std::endl;
				std::cout << "stride: " << buffer->Stride() << std::endl;
			}
		}
		{
			auto addressable = std::unique_ptr<RTLib::Core::IBufferAddressableView>(buffer->Addressable());
			auto address     = reinterpret_cast<int*>(addressable->Address());
			for (int i = 0; i < buffer->Size() / buffer->Stride(); ++i) {
				address[i] = 2* i;
			}
		}
	}

	return 0;
}
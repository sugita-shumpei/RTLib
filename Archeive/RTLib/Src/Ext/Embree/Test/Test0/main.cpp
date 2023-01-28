#include <embree3/rtcore.h>
#include <memory>
template<typename T>
using InternalEmbreeUniquePointer = std::unique_ptr<std::remove_pointer_t<T>, void(*)(T)>;
template<typename T, void(*Deleter)(T)>
struct InternalEmbreeMakeHandleImpl {
	static auto Eval(T ptr) ->InternalEmbreeUniquePointer<T> {
		return InternalEmbreeUniquePointer<T>(ptr, Deleter);
	}
};
auto  Internal_MakeEmbreeDevice(const char* args)->InternalEmbreeUniquePointer<RTCDevice> {
	return InternalEmbreeMakeHandleImpl<RTCDevice,rtcReleaseDevice>::Eval(rtcNewDevice(args));
}
auto  Internal_MakeEmbreeScene(RTCDevice device)->InternalEmbreeUniquePointer<RTCScene> {
	return InternalEmbreeMakeHandleImpl<RTCScene, rtcReleaseScene>::Eval(rtcNewScene(device));
}
auto  Internal_MakeEmbreeGeometry(RTCDevice device,RTCGeometryType geoType)->InternalEmbreeUniquePointer<RTCGeometry> {
	return InternalEmbreeMakeHandleImpl<RTCGeometry, rtcReleaseGeometry>::Eval(rtcNewGeometry(device, geoType));
}
auto  Internal_MakeEmbreeBuffer(RTCDevice device, size_t sizeInBytes) {
	return InternalEmbreeMakeHandleImpl<RTCBuffer, rtcReleaseBuffer>::Eval(rtcNewBuffer(device, sizeInBytes));
}
int main()
{
	auto device   = Internal_MakeEmbreeDevice(nullptr);
	auto scene    = Internal_MakeEmbreeScene (nullptr);
	auto buffer   = Internal_MakeEmbreeBuffer(device.get(), );
	auto geometry = Internal_MakeEmbreeGeometry(device.get(), RTC_GEOMETRY_TYPE_TRIANGLE);

}
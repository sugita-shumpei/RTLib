#include <RTLib/Ext/OPX7/OPX7Exceptions.h>
#include <optix_stubs.h>
#include <string>
auto RTLib::Ext::OPX7::OPX7Exception::ResultToString(OptixResult result) noexcept -> std::string {
	return "Error: " + std::string(optixGetErrorString(result));
}

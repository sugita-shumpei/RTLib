#include <iostream>
#include <RTLib/Backends/CUDA/CUDAEntry.h>
#include <RTLib/Backends/CUDA/CUDADevice.h>
#include <RTLib/Backends/CUDA/CUDAContext.h>
#include <RTLib/Backends/CUDA/CUDAStream.h>
#include <RTLib/Backends/CUDA/CUDALinearMemory.h>
#include <RTLib/Backends/CUDA/CUDAPinnedHostMemory.h>
#include <RTLib/Backends/CUDA/CUDAArray.h>
#include <RTLib/Backends/CUDA/CUDAMipmappedArray.h>
#include <RTLib/Backends/CUDA/CUDATexture.h>
int main(int argc, const char** argv)
{
	auto& entry    = RTLib::Backends::Cuda::Entry::Handle();
	auto& devices  = entry.EnumerateDevices();
	auto  context  = std::make_unique<RTLib::Backends::Cuda::Context>(devices[0]);
	auto& current  = RTLib::Backends::Cuda::CurrentContext::Handle();
	auto  stream   = current.CreateStreamUnique();
	auto  memory1  = current.CreateLinearMemory1DUnique(1024);
	auto  memory2  = current.CreateLinearMemory1DUnique(1024);
	auto  array1d1 = current.CreateArray1DUnique(1024, 4, RTLib::Backends::Cuda::ArrayFormat::eUInt8);
	auto  array2d1 = current.CreateArray2DUnique(1024, 1024, 4, RTLib::Backends::Cuda::ArrayFormat::eUInt8);
	auto  array3d1 = current.CreateArray3DUnique(128, 128, 128, 4, RTLib::Backends::Cuda::ArrayFormat::eUInt8);
	auto  mipmapped_array1d1 = current.CreateMipmappedArray1DUnique(3,1024, 4, RTLib::Backends::Cuda::ArrayFormat::eUInt8);
	auto  mipmapped_array2d1 = current.CreateMipmappedArray2DUnique(3,1024, 1024, 4, RTLib::Backends::Cuda::ArrayFormat::eUInt8);
	auto  mipmapped_array3d1 = current.CreateMipmappedArray3DUnique(3,128, 128, 128, 4, RTLib::Backends::Cuda::ArrayFormat::eUInt8);
	{
		auto desc             = RTLib::Backends::Cuda::TextureDesc();
		desc.addressMode[0]   = RTLib::Backends::Cuda::AddressMode::eWarp;
		desc.addressMode[1]   = RTLib::Backends::Cuda::AddressMode::eWarp;
		desc.filterMode       = RTLib::Backends::Cuda::FilterMode::eLinear;
		desc.readMode         = RTLib::Backends::Cuda::TextureReadMode::eNormalizedFloat;
		desc.mipmapFilterMode = RTLib::Backends::Cuda::FilterMode::eLinear;
		auto  texture1 = current.CreateTextureUniqueFromMipmappedArray(mipmapped_array2d1.get(),desc);
	}
	{
		auto desc = RTLib::Backends::Cuda::TextureDesc();
		desc.addressMode[0] = RTLib::Backends::Cuda::AddressMode::eWarp;
		desc.addressMode[1] = RTLib::Backends::Cuda::AddressMode::eWarp;
		desc.filterMode = RTLib::Backends::Cuda::FilterMode::eLinear;
		desc.readMode = RTLib::Backends::Cuda::TextureReadMode::eNormalizedFloat;
		desc.mipmapFilterMode = RTLib::Backends::Cuda::FilterMode::eLinear;
		auto  texture1 = current.CreateTextureUniqueFromArray(array2d1.get(), desc);
	}
	return 0;
}
#define STB_IMAGE_WRITE_IMPLEMENTATION
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
#include <RTLib/Backends/CUDA/CUDAModule.h>
#include "RTLibBackendsCudaTest0Config.h"
#include <stb_image_write.h>
#include <random>
#include <algorithm>
int main(int argc, const char** argv)
{
	auto& entry    = RTLib::Backends::Cuda::Entry::Handle();
	auto& devices  = entry.EnumerateDevices();
	auto  context  = std::make_unique<RTLib::Backends::Cuda::Context>(devices[0]);
	auto& current  = RTLib::Backends::Cuda::CurrentContext::Handle();
	auto  stream   = current.CreateStreamUnique();
	auto  memory1  = current.CreateLinearMemory1DUnique(1024 * 1024 * sizeof(unsigned int));
	auto  memory2  = current.CreateLinearMemory1DUnique(1024 * 1024 * sizeof(unsigned char)*4);

	{
		auto random_seeds = std::vector<unsigned int>(1024 * 1024);
		std::random_device rd;
		std::mt19937 mt(rd());
		std::generate(std::begin(random_seeds), std::end(random_seeds), std::ref(mt));
		current.Copy1DFromHostToLinearMemory(memory1.get(), random_seeds.data(), { 0,0,sizeof(random_seeds[0])*random_seeds.size()});
	}
	auto  module1 = current.CreateModuleFromFile(RTLIB_BACKENDS_CUDA_TEST_TEST0_CUDA_PATH"/sampleKernel.ptx");
	auto  function1 = module1->GetFunction("randomKernel");
	{
		auto launchDesc = RTLib::Backends::Cuda::KernelLaunchDesc();
		launchDesc.gridDimX = 32;
		launchDesc.gridDimY = 32;
		launchDesc.gridDimZ = 1;

		launchDesc.blockDimX = 1024/32;
		launchDesc.blockDimY = 1024/32;
		launchDesc.blockDimZ = 1;

		launchDesc.params.reserve(4);
		unsigned long long memory1Address = reinterpret_cast<unsigned long long>(memory1->GetHandle());
		unsigned long long memory2Address = reinterpret_cast<unsigned long long>(memory2->GetHandle());
		launchDesc.params.push_back(&memory1Address);
		launchDesc.params.push_back(&memory2Address);
		unsigned int width  = 1024;
		unsigned int height = 1024;
		launchDesc.params.push_back(&width);
		launchDesc.params.push_back(&height);

		stream->LaunchKernel(function1, launchDesc);
		stream->Synchronize();
	}
	auto resultImages = std::vector<unsigned char>(1024 * 1024 * 4);
	{
		current.Copy1DFromLinearMemoryToHost(resultImages.data(), memory2.get(), { 0,0,sizeof(resultImages[0]) * resultImages.size() });
	}
	stbi_write_png(RTLIB_BACKENDS_CUDA_TEST_TEST0_CUDA_PATH"/../result.png", 1024, 1024, 4, (const void*)resultImages.data(), 1024 * 4);
	return 0;
}